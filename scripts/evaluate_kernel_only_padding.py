"""Extract a single Kernel output, verify CUDA predictions and compare padding."""

import argparse
import gc
import json
import os
from pathlib import Path
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.kernel_only import KernelOnlyAGUNet
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts import run_adaspot_gap as source_runner
from scripts.evaluate_adaspot_gap_ocr import aggregate_rows
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _geometry_match, _train_gt
from scripts.evaluate_unet_word_research_ocr import cache_record, cached_text, score_predictions
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.training import load_checkpoint, seed_everything

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / 'configs/approximate/kernel_only_padding_v1.json'


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != 'approximate' or REPO/'configs/approximate' not in path.parents:
        raise ValueError('Explicit approximate configuration required; no namespace fallback')
    config = json.loads(path.read_text())
    if (config.get('reproduction_mode') != mode or config.get('claim_label') != 'approximate reproduction'
            or config.get('experiment_kind') != 'extracted_kernel_only_padding_ablation'
            or config.get('source_branch') != 'control' or config.get('connectivity') != 8
            or config.get('input_size') != [512,512] or config.get('native_mask_size') != [256,256]
            or config.get('min_component_area') != 0 or config.get('max_instances') is not None
            or config.get('precision') != 'float32' or config.get('allow_tf32') is not False
            or config.get('deterministic_algorithms') is not True):
        raise ValueError('Invalid kernel-only scientific configuration')
    pads = config['padding_native_pixels']
    if (not pads or any(type(v) is not int or v < 0 for v in pads)
            or pads != sorted(set(pads)) or pads[0] != 0
            or not 0 < config['kernel_threshold'] < 1
            or type(config['ocr_batch_size']) is not int or config['ocr_batch_size'] < 1):
        raise ValueError('Invalid padding, threshold or batch size')
    source_path = (REPO/config['source_config']).resolve()
    if REPO/'configs/approximate' not in source_path.parents or sha256_file(source_path) != config['source_config_sha256']:
        raise ValueError('Qualified source configuration changed')
    source, preprocessing = source_runner.read_config(source_path, mode)
    return config, source, preprocessing


def kernel_instances(probability, canvas, threshold):
    p, canvas = np.asarray(probability), np.asarray(canvas)
    if (p.ndim != 2 or p.shape != canvas.shape or canvas.dtype != np.bool_
            or not np.isfinite(p).all() or np.any((p < 0)|(p > 1))
            or not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('Invalid Kernel probability or RGB canvas')
    return cv2.connectedComponents(((p >= threshold)&canvas).astype(np.uint8), connectivity=8)[1]


def ignored_instances(labels, validity):
    return [int(v) for v in np.unique(labels[labels > 0]) if not np.any((labels == v)&validity)]


def add_geometry_metrics(rows, expected_words):
    result = aggregate_rows(rows, expected_words)
    for key in ('geometry_iou_ge_50', 'unpadded_geometry_iou_ge_50'):
        tp = sum(row[key] for row in rows)
        result[key] = tp
        result[key+'_precision'] = tp/result['proposals'] if result['proposals'] else 0.
        result[key+'_recall'] = tp/expected_words
        result[key+'_hmean'] = 2*tp/(result['proposals']+expected_words)
    return result


def run(args):
    cfg, source, preprocessing = read_config(args.config, args.mode)
    output, trained, word_root = args.output_root.resolve(), args.source_run.resolve(), args.word_root.resolve()
    if (output == REPO or REPO in output.parents or word_root == output or word_root in output.parents
            or trained == output or trained in output.parents):
        raise ValueError('Use a separate output outside repository, source run and dataset')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise ValueError('Results already exist; preserve the completed experiment')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA; no CPU fallback')
    if device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_everything(source['seed'],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cv2.setNumThreads(1)
    receipt=json.loads((trained/'run_manifest.json').read_text())
    contract=receipt['run_contract']; checkpoint=trained/'best.pt'
    if (receipt.get('status') != 'SUCCESS' or not receipt.get('training_complete')
            or receipt.get('completed_epochs') != source['epochs']
            or contract['resolved_config'] != source
            or contract['branch'] != cfg['source_branch']
            or sha256_file(checkpoint) != cfg['source_checkpoint_sha256']
            or receipt['checkpoint']['sha256'] != cfg['source_checkpoint_sha256']):
        raise ValueError('Source must be the qualified completed Text+Kernel control')
    for name,digest in contract['source_sha256'].items():
        if sha256_file(REPO/name) != digest: raise ValueError('Source code drift: '+name)
    report=json.loads((trained/'validation.json').read_text())
    if report['run_contract'] != contract or report['checkpoint_sha256'] != cfg['source_checkpoint_sha256']:
        raise ValueError('Source validation/checkpoint provenance differs')
    manifest_path=word_root/source['train_manifest']
    if sha256_file(manifest_path) != source['word_manifest_sha256']: raise ValueError('Dataset manifest changed')
    manifest=load_segmentation_manifest(manifest_path,require_checksums=True)
    _,indices,split=_split_indices(manifest,source['seed'],source['validation_count'])
    if (manifest.dataset != 'total_text' or manifest.split_role != 'train'
            or split != source['validation_ids_sha256']): raise ValueError('Wrong dataset or heldout split')
    rows_by_id={r['sample_id']:r for r in report['per_image']}
    if len(rows_by_id) != len(indices) or set(rows_by_id) != {manifest.samples[i].sample_id for i in indices}:
        raise ValueError('Incomplete source prediction population')
    artifacts=args.parseq_artifacts.resolve();revision=PARSEQ_RUNTIME_REVISION
    weights=artifacts/'parseq-bb5792a6.pt';archive=artifacts/('parseq-'+revision+'.tar.gz')
    if sha256_file(weights) != PARSEQ_CHECKPOINT_SHA256 or sha256_file(archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError('PARSeq checkpoint/archive checksum mismatch')
    protocol={'resolved_config':cfg,'resolved_source_config':source,'source_run_contract':contract,
        'source_checkpoint_sha256':sha256_file(checkpoint),'source_validation_sha256':sha256_file(trained/'validation.json'),
        'result_class':'approximate reproduction','mode':'approximate','source_training':'joint Text+Kernel, no retraining',
        'runtime_inputs':{'source_run':str(trained),'word_root':str(word_root),'parseq_artifacts':str(artifacts)},
        'source_sha256':{name:sha256_file(REPO/name) for name in ('models/kernel_only.py','scripts/evaluate_kernel_only_padding.py',
            'localize/word_multitask.py','recognition/parseq.py','scripts/evaluate_unet_word_research_ocr.py',
            'scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py','segmentation/training.py')},
        'parseq_checkpoint_sha256':PARSEQ_CHECKPOINT_SHA256,'parseq_archive_sha256':PARSEQ_RUNTIME_ARCHIVE_SHA256,
        'device':str(device),'gpu_name':torch.cuda.get_device_name(device),'backend':source_runner.backend_state()}
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != protocol:
        raise ValueError('Output protocol differs; select another output directory')
    _json(output/'protocol.json',protocol)
    baseline=AdaSpotGapAGUNet('control',skip_floor=source['skip_floor'],initial_gap_bias=source['initial_gap_bias'])
    load_checkpoint(checkpoint,model=baseline,expected_architecture=baseline.architecture_identity,
                    expected_run_contract=contract,expected_training_config=source_runner.training_config(source))
    model=KernelOnlyAGUNet(skip_floor=source['skip_floor'],encoder_pretrained=False)
    extraction=model.initialize_text_kernel_state(baseline.state_dict());del baseline;gc.collect()
    dataset=SegmentationDataset(manifest,word_root,transform=_transform(preprocessing,'letterbox',train=False))
    loader=DataLoader(Subset(dataset,indices),batch_size=source['batch_size'],shuffle=False,num_workers=2,
                      pin_memory=True,timeout=source['loader_timeout_seconds'])
    _prepare_cuda_memory_tracking(device);model.to(device).eval()
    maps={};max_error=0.;seen=[]
    with torch.inference_mode():
        for batch in loader:
            prediction=model(batch['image'].to(device)).sigmoid().cpu().numpy()
            if prediction.shape[1:] != (1,256,256): raise RuntimeError('Kernel-only output shape changed')
            for j,sample_id in enumerate(batch['sample_id']):
                path=trained/'validation_exports'/sample_id.replace(':','__');row=rows_by_id[sample_id]
                if sha256_file(path/'maps.npz') != row['maps_sha256']: raise ValueError('Frozen source map changed')
                with np.load(path/'maps.npz',allow_pickle=False) as z:
                    a={k:z[k].copy() for k in ('probabilities','labels','canvas','validity')}
                p=prediction[j,0];ref=a['probabilities'][1]
                max_error=max(max_error,float(np.abs(p-ref).max()))
                np.testing.assert_allclose(p,ref,atol=1e-6,rtol=1e-5)
                if not np.array_equal((p>=cfg['kernel_threshold'])&a['canvas'],(ref>=cfg['kernel_threshold'])&a['canvas']):
                    raise RuntimeError('Extraction changed native thresholded Kernel mask')
                k=kernel_instances(p,a['canvas'],cfg['kernel_threshold'])
                maps[sample_id]=(k,a['labels'],a['validity']);seen.append(sample_id)
                dest=output/'masks'/sample_id.replace(':','__');dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'kernel.npz',kernel_probability=p,labels=k,canvas=a['canvas'])
                Image.fromarray((k>0).astype(np.uint8)*255).save(dest/'kernel.png')
            print('[Kernel CUDA] {}/{} verified'.format(len(seen),len(indices)),flush=True)
    if len(seen)!=len(indices) or len(set(seen))!=len(indices): raise RuntimeError('Missing or duplicate inference samples')
    segmentation_memory=source_runner.memory(device,source);segmentation_memory['model_set']='single extracted Kernel-only AG-UNet'
    model.cpu()
    torch.save({'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'architecture':model.architecture_identity,'model_state_dict':model.state_dict(),'run_contract':protocol,
        'extraction':extraction},output/'kernel_only.pt')
    del model;gc.collect();torch.cuda.empty_cache()
    arms={};pending={};texts={};cache=output/'ocr_cache';shared=trained.parent/'parseq_ocr_cache'
    samples={s.sample_id:s for s in manifest.samples}
    for sample_id in seen:
        sample=samples[sample_id];image_path=word_root/sample.image_path
        if sha256_file(image_path)!=sample.image_sha256: raise ValueError('RGB source changed')
        with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB'))
        frozen=[]
        for name,labels in zip(('kernel_only','text_kernel'),maps[sample_id][:2]):
            ignored=ignored_instances(labels,maps[sample_id][2])
            for padding in cfg['padding_native_pixels']:
                arm=name+'/pad'+str(padding)
                proposals=parseq_word_proposals(labels,(rgb.shape[1],rgb.shape[0]),padding=padding,min_area=0,max_instances=None)
                for p in proposals:
                    x1,y1,x2,y2=p['bbox'];digest=crop_array_hash(np.ascontiguousarray(rgb[y1:y2,x1:x2]))
                    p['crop_rgb_sha256']=digest
                    found=next((q for q in (cache/(digest+'.json'),shared/(digest+'.json')) if q.exists()),None)
                    if found is not None:texts[digest]=cached_text(json.loads(found.read_text()),digest)
                    elif digest not in texts:pending[digest]=(image_path,p['bbox'])
                frozen.append((arm,proposals,ignored))
        # Ground truth is read only after every arm's geometry is frozen.
        gt,annotation_sha=_train_gt(sample,word_root,(rgb.shape[1],rgb.shape[0]))
        for arm,proposals,ignored in frozen:
            arms.setdefault(arm,[]).append(dict(sample_id=sample_id,proposals=proposals,ignored_ids=ignored,
                ground_truth=gt,annotation_sha256=annotation_sha))
    _json(output/'frozen_crops.json',arms)
    _prepare_cuda_memory_tracking(device);recognizer=None
    try:
        if pending:
            recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=artifacts/('parseq-'+revision),
                checkpoint=weights,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=archive,device=str(device)))
            jobs=sorted(pending.items())
            for start in range(0,len(jobs),cfg['ocr_batch_size']):
                items=jobs[start:start+cfg['ocr_batch_size']];crops=[]
                for digest,(image_path,box) in items:
                    with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB'))
                    x1,y1,x2,y2=box;crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                    if crop_array_hash(crop)!=digest: raise ValueError('Frozen crop changed')
                    crops.append(crop)
                predictions=recognizer.recognize_candidate_batch(crops)
                if len(predictions)!=len(items): raise RuntimeError('Incomplete PARSeq outputs')
                for (digest,_),prediction in zip(items,predictions):
                    record=cache_record(prediction,digest,recognizer.config);texts[digest]=cached_text(record,digest)
                    _json(cache/(digest+'.json'),record)
                if start//cfg['ocr_batch_size']%25==0:
                    print('[PARSeq CUDA] {}/{} unique crops'.format(min(start+len(items),len(jobs)),len(jobs)),flush=True)
    finally:
        if recognizer is not None:recognizer.release()
    ocr_memory=source_runner.memory(device,{**source,'batch_size':cfg['ocr_batch_size']})
    ocr_memory.update(model_set='PARSeq only',input_shape=[cfg['ocr_batch_size'],3,32,128])
    results={}
    for arm,records in arms.items():
        rows=[]
        for record in records:
            predictions=[{**p,'text':texts[p['crop_rgb_sha256']]} for p in record['proposals'] if p['instance_id'] not in record['ignored_ids']]
            metrics,matches=score_predictions(predictions,record['ground_truth'])
            unpadded=[{**p,'bbox':p['bbox_original']} for p in predictions]
            metrics['unpadded_geometry_iou_ge_50']=len(_geometry_match(unpadded,record['ground_truth'],.5))
            rows.append({**record,**metrics,'predictions':predictions,'matched_gt_indices':{k:sorted(v) for k,v in matches.items()}})
        results[arm]={'aggregate':add_geometry_metrics(rows,source['expected_validation_words']),'per_image':rows}
    # The unchanged reference at padding0 must reproduce its original full OCR result.
    reference=json.loads((trained.parent/'ocr_comparison.json').read_text())['arms']['control']['aggregate']
    if any(results['text_kernel/pad0']['aggregate'][k]!=reference[k] for k in reference):
        raise RuntimeError('Unchanged Text+Kernel padding0 reference did not reproduce')
    result=dict(status='SUCCESS',result_class='approximate reproduction',protocol=protocol,extraction=extraction,
        max_kernel_probability_difference=max_error,all_native_kernel_masks_identical=True,arms=results,
        memory={'segmentation':segmentation_memory,'ocr':ocr_memory},new_unique_crops=len(pending),cached_unique_crops=len(texts)-len(pending))
    _json(output/'results.json',result)
    lines=['# Kernel-only padding — approximate reproduction','','Joint-trained Kernel extracted; no new training. Reused TRAIN125 /983 words. Padding in native256 pixels.',
        '','| Model | Padding | Localized padded box | Localized raw box | Exact OCR | OCR Hmean | Proposals |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,v in results.items():
        a=v['aggregate'];name,pad=arm.split('/pad')
        lines.append('| {} | {} | {} | {} | {} | {:.2%} | {} |'.format(name,pad,a['geometry_iou_ge_50'],a['unpadded_geometry_iou_ge_50'],a['exact_casefold_iou_ge_50'],a['exact_casefold_hmean'],a['proposals']))
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    for name in ('source-run','word-root','output-root','parseq-artifacts'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    try:run(args)
    except Exception as error:
        args.output_root.mkdir(parents=True,exist_ok=True)
        _json(args.output_root/'failure.json',{'status':'FAILED','error':str(error),'type':type(error).__name__,
            'oom':isinstance(error,torch.cuda.OutOfMemoryError),'traceback':traceback.format_exc(),
            'device':args.device,'dtype':'float32','model_set':'sequential extracted Kernel-only AG-UNet then PARSeq',
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None})
        raise
