"""Resumable CUDA fine-tuning supervised only by K, then fixed-padding OCR."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from models.kernel_only import KernelOnlyAGUNet
from segmentation.kernel_only_training import KernelTrainingConfig,train_kernel_epoch
from segmentation.totaltext_word_kernel import WordKernelDataset
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.data import SegmentationDataset,load_segmentation_manifest
from segmentation.training import load_checkpoint,save_checkpoint,seed_everything,runtime_environment
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.train import EpochShuffleSampler
from scripts.train_totaltext_v4_highres_ablation import _json,_split_indices,_append_epoch,_epoch_log
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts import evaluate_kernel_only_padding as P
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt,_geometry_match
from scripts.evaluate_unet_word_research_ocr import score_predictions,cached_text,cache_record
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from recognition.parseq import PARSEQ_RUNTIME_REVISION,PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256,PARSEQ_ARCHIVE_SOURCE,ParseqApproxConfig,ParseqApproxRecognizer
from scripts.prune_segmentation_checkpoints import sha256_file

REPO=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=REPO/'configs/approximate/kernel_only_training_v1.json'
FILES=('scripts/train_kernel_only.py','segmentation/kernel_only_training.py','models/kernel_only.py',
       'scripts/evaluate_kernel_only_padding.py','segmentation/totaltext_word_kernel.py',
       'segmentation/word_multitask.py','segmentation/training.py','segmentation/data.py',
       'segmentation/transforms.py','segmentation/totaltext_image_frequency_augment.py',
       'models/unet_adv.py','recognition/parseq.py','localize/word_multitask.py',
       'scripts/evaluate_unet_word_research_ocr.py','scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py')


def read_config(path,mode):
    path=Path(path).resolve()
    if mode!='approximate' or REPO/'configs/approximate' not in path.parents:raise ValueError('Explicit approximate configuration required')
    cfg=json.loads(path.read_text())
    if (cfg.get('reproduction_mode')!=mode or cfg.get('claim_label')!='approximate reproduction'
        or cfg.get('experiment_kind')!='kernel_only_supervised_finetune' or cfg.get('supervision')!='kernel_only'
        or cfg.get('encoder')!='frozen_including_BN' or cfg.get('precision')!='float32'
        or cfg.get('allow_tf32') is not False or cfg.get('deterministic_algorithms') is not True
        or cfg.get('selection')!='macro_kernel_IoU_then_macro_kernel_F1_then_earliest_include_epoch0'):
        raise ValueError('Invalid Kernel-only scientific configuration')
    source_path=(REPO/cfg['source_config']).resolve()
    if REPO/'configs/approximate' not in source_path.parents or sha256_file(source_path)!=cfg['source_config_sha256']:raise ValueError('Source configuration changed')
    padding,source,preprocessing=P.read_config(source_path,mode)
    if cfg['seed']!=source['seed'] or cfg['batch_size']!=source['batch_size']:raise ValueError('Matched seed/batch changed')
    if any(type(cfg[k]) is not int or cfg[k]<minimum for k,minimum in [('epochs',1),('num_workers',0)]):raise ValueError('Invalid training budget/workers')
    if set(cfg['loss'])!={'bce_weight','dice_weight','smooth'} or not np.isfinite(list(cfg['loss'].values())).all() or min(cfg['loss'].values())<=0:raise ValueError('Invalid loss weights')
    training_config(cfg)
    return cfg,padding,source,preprocessing


def training_config(cfg):
    return KernelTrainingConfig(epochs=cfg['epochs'],seed=cfg['seed'],adam_betas=tuple(cfg['adam_betas']),adam_epsilon=cfg['adam_epsilon'],
        learning_rate_schedule='none',checkpoint_selection='validation_foreground_iou',deterministic_algorithms=True,
        learning_rate=cfg['learning_rate'],weight_decay=cfg['weight_decay'],batch_size=cfg['batch_size'])


def memory(device,source):
    value=P.source_runner.memory(device,source);value['model_set']='one single-output Kernel AG-UNet; no OCR resident'
    return value


def selection(metrics,epoch):return metrics['kernel_iou_macro'],metrics['kernel_f1_macro'],-epoch


def evaluate_kernel(model,loader,device,source,export_dir=None):
    model.eval();rows=[]
    with torch.inference_mode():
        for batch in loader:
            p=model(batch['image'].to(device)).sigmoid().cpu().numpy()[:,0]
            for j,sample_id in enumerate(batch['sample_id']):
                target=batch['kernel'][j,0].numpy().astype(bool);valid=batch['validity'][j,0].numpy().astype(bool)
                if p[j].shape!=target.shape:raise ValueError('Kernel prediction/target resolution differs')
                pred=p[j]>=source['postprocessing']['kernel_threshold']
                tp=int((pred&target&valid).sum());fp=int((pred&~target&valid).sum());fn=int((~pred&target&valid).sum())
                rows.append({'sample_id':sample_id,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.})
                if export_dir is not None:
                    with Image.open(batch['image_path'][j]) as im:canvas=letterbox_canvas_mask(im.size).numpy()
                    labels=P.kernel_instances(p[j],canvas,source['postprocessing']['kernel_threshold'])
                    folder=Path(export_dir)/sample_id.replace(':','__');folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/'maps.npz',kernel_probability=p[j],labels=labels,canvas=canvas,validity=valid,kernel_target=target)
                    Image.fromarray((labels>0).astype(np.uint8)*255).save(folder/'kernel.png')
    if len(rows)!=len(loader.dataset) or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Incomplete/duplicate validation samples')
    return {'kernel_iou_macro':float(np.mean([r['iou'] for r in rows])),
            'kernel_f1_macro':float(np.mean([r['f1'] for r in rows])),'samples':len(rows),'per_image':rows}


def train_loop(directory,model,optimizer,train_loader,val_loader,transform,sampler,cfg,source,contract,device,stop_after=None):
    tc=training_config(cfg);last=directory/'last.pt';best=directory/'best.pt';history=_epoch_log(directory/'epochs.jsonl')
    epoch=step=0;best_epoch=None;best_metrics=None
    def save(path,e,trained,metrics):
        return save_checkpoint(path,model=model,optimizer=optimizer,epoch=e,global_step=step,config=tc,
            resolved_config=json.dumps(contract,sort_keys=True),architecture=model.architecture_identity,result_class='approximate reproduction',
            run_contract=contract,train_metrics=trained,validation_metrics=metrics)
    if best.exists():
        saved=load_checkpoint(best,model=model,expected_architecture=model.architecture_identity,expected_run_contract=contract,expected_training_config=tc)
        best_epoch,best_metrics=saved['epoch'],saved['validation_metrics'];del saved
    if last.exists():
        saved=load_checkpoint(last,model=model,optimizer=optimizer,expected_architecture=model.architecture_identity,
            expected_run_contract=contract,expected_training_config=tc,restore_rng=True)
        epoch,step=saved['epoch'],saved['global_step']
        if epoch and len(history)==epoch-1:
            row={'epoch':epoch,'global_step':step,'train':saved['train_metrics'],'validation':saved['validation_metrics'],
                 'last_checkpoint_sha256':sha256_file(last),'recovered_from_checkpoint':True}
            _append_epoch(directory/'epochs.jsonl',row);history.append(row)
        if len(history)!=epoch or (history and history[-1]['last_checkpoint_sha256']!=sha256_file(last)):raise ValueError('Checkpoint/history mismatch')
        if best_metrics is None or selection(saved['validation_metrics'],epoch)>selection(best_metrics,best_epoch):
            best_epoch,best_metrics=epoch,saved['validation_metrics'];save(best,epoch,saved['train_metrics'],best_metrics)
        del saved
    elif history or best.exists():raise ValueError('Training artifacts lack resumable latest checkpoint')
    else:
        best_metrics=evaluate_kernel(model,val_loader,device,source);best_epoch=0
        save(last,0,{},best_metrics);save(best,0,{},best_metrics);_json(directory/'epoch0.json',best_metrics)
    if [r['epoch'] for r in history]!=list(range(1,epoch+1)):raise ValueError('Noncontiguous epoch history')
    budget=cfg['epochs'] if stop_after is None else stop_after
    if type(budget) is not int or not 1<=budget<=cfg['epochs'] or epoch>cfg['epochs']:raise ValueError('Invalid stop epoch')
    for e in range(epoch+1,budget+1):
        started=time.monotonic();sampler.set_epoch(e-1);transform.set_epoch(e-1)
        trained=train_kernel_epoch(model,train_loader,optimizer,device,cfg)
        metrics=evaluate_kernel(model,val_loader,device,source);step+=trained['steps']
        digest=save(last,e,trained,metrics)
        if selection(metrics,e)>selection(best_metrics,best_epoch):best_epoch,best_metrics=e,metrics;save(best,e,trained,metrics)
        row={'epoch':e,'global_step':step,'train':trained,'validation':metrics,'last_checkpoint_sha256':digest,
             'epoch_seconds':time.monotonic()-started,'memory':memory(device,source)}
        _append_epoch(directory/'epochs.jsonl',row);history.append(row)
        _json(directory/'progress.json',{'status':'RUNNING','completed_epochs':e,'global_step':step,'best_epoch':best_epoch,
            'last_epoch':row,'result_class':'approximate reproduction'})
        print('[Kernel-only] epoch={}/{} K_IoU={:.5f} best={} seconds={:.1f}'.format(e,cfg['epochs'],metrics['kernel_iou_macro'],best_epoch,row['epoch_seconds']),flush=True)
    complete=len(history)==cfg['epochs']
    result={'status':'SUCCESS' if complete else 'IN_PROGRESS','training_complete':complete,'completed_epochs':len(history),
        'global_step':step,'checkpoint':{'path':'best.pt','epoch':best_epoch,'sha256':sha256_file(best)},
        'validation_kernel':best_metrics,'run_contract':contract,'memory':memory(device,source),'result_class':'approximate reproduction'}
    _json(directory/'run_manifest.json',result)
    _json(directory/'progress.json',{k:v for k,v in result.items() if k not in ('validation_kernel','run_contract')})
    return result


def padding_ocr(output,reference,manifest,indices,word_root,padding,source,artifacts,device):
    inputs={};texts={};pending={};cache=output/'ocr_cache'
    for index in indices:
        sample=manifest.samples[index];folder=output/'validation_exports'/sample.sample_id.replace(':','__')
        with np.load(folder/'maps.npz',allow_pickle=False) as z:labels=z['labels'];valid=z['validity']
        ignored=P.ignored_instances(labels,valid);image_path=word_root/sample.image_path
        if sha256_file(image_path)!=sample.image_sha256:raise ValueError('RGB checksum changed')
        with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB'))
        frozen=[]
        for pad in padding['padding_native_pixels']:
            proposals=P.parseq_word_proposals(labels,(rgb.shape[1],rgb.shape[0]),padding=pad,min_area=0,max_instances=None)
            for p in proposals:
                x1,y1,x2,y2=p['bbox'];digest=crop_array_hash(np.ascontiguousarray(rgb[y1:y2,x1:x2]));p['crop_rgb_sha256']=digest
                found=next((q for q in (cache/(digest+'.json'),reference/'ocr_cache'/(digest+'.json')) if q.exists()),None)
                if found is not None:texts[digest]=cached_text(json.loads(found.read_text()),digest)
                else:pending[digest]=(image_path,p['bbox'])
            frozen.append((pad,proposals))
        gt,annotation_sha=_train_gt(sample,word_root,(rgb.shape[1],rgb.shape[0]))
        for pad,proposals in frozen:inputs.setdefault('pad'+str(pad),[]).append(dict(sample_id=sample.sample_id,proposals=proposals,
            ignored_ids=ignored,ground_truth=gt,annotation_sha256=annotation_sha))
    _json(output/'frozen_crops.json',inputs);_prepare_cuda_memory_tracking(device);recognizer=None
    try:
        if pending:
            recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=artifacts/('parseq-'+PARSEQ_RUNTIME_REVISION),
                checkpoint=artifacts/'parseq-bb5792a6.pt',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=artifacts/('parseq-'+PARSEQ_RUNTIME_REVISION+'.tar.gz'),device=str(device)))
            jobs=sorted(pending.items())
            for start in range(0,len(jobs),padding['ocr_batch_size']):
                items=jobs[start:start+padding['ocr_batch_size']];crops=[]
                for digest,(path,box) in items:
                    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
                    x1,y1,x2,y2=box;crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                    if crop_array_hash(crop)!=digest:raise ValueError('Crop changed before OCR')
                    crops.append(crop)
                predictions=recognizer.recognize_candidate_batch(crops)
                if len(predictions)!=len(items):raise RuntimeError('Incomplete OCR output')
                for (digest,_),prediction in zip(items,predictions):
                    record=cache_record(prediction,digest,recognizer.config);texts[digest]=cached_text(record,digest);_json(cache/(digest+'.json'),record)
                if start//padding['ocr_batch_size']%25==0:print('[PARSeq CUDA] {}/{}'.format(min(start+len(items),len(jobs)),len(jobs)),flush=True)
    finally:
        if recognizer is not None:recognizer.release()
    ocr_memory=memory(device,source);ocr_memory.update(model_set='PARSeq only',input_shape=[padding['ocr_batch_size'],3,32,128])
    results={}
    for arm,records in inputs.items():
        rows=[]
        for r in records:
            preds=[{**p,'text':texts[p['crop_rgb_sha256']]} for p in r['proposals'] if p['instance_id'] not in r['ignored_ids']]
            counts,matches=score_predictions(preds,r['ground_truth'])
            counts['unpadded_geometry_iou_ge_50']=len(_geometry_match([{**p,'bbox':p['bbox_original']} for p in preds],r['ground_truth'],.5))
            rows.append({**r,**counts,'predictions':preds,'matched_gt_indices':{k:sorted(v) for k,v in matches.items()}})
        results[arm]={'aggregate':P.add_geometry_metrics(rows,source['expected_validation_words']),'per_image':rows}
    prior=json.loads((reference/'results.json').read_text())
    result={'status':'SUCCESS','result_class':'approximate reproduction','arms':results,'memory':ocr_memory,
            'reference_results_sha256':sha256_file(reference/'results.json'),'reference_aggregates':{k:v['aggregate'] for k,v in prior['arms'].items()}}
    _json(output/'results.json',result)
    lines=['# Kernel-only supervised fine-tuning — approximate reproduction','','Reused TRAIN125 /983 words; fixed native256 padding.',
        '','| Padding | Trained K localized | Trained K exact OCR | Extracted K localized | Extracted K exact OCR | M+K localized | M+K exact OCR |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for pad in padding['padding_native_pixels']:
        a=results['pad'+str(pad)]['aggregate'];b=prior['arms']['kernel_only/pad'+str(pad)]['aggregate'];c=prior['arms']['text_kernel/pad'+str(pad)]['aggregate']
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(pad,a['geometry_iou_ge_50'],a['exact_casefold_iou_ge_50'],b['geometry_iou_ge_50'],b['exact_casefold_iou_ge_50'],c['geometry_iou_ge_50'],c['exact_casefold_iou_ge_50']))
    (output/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return result


def run(args):
    cfg,padding,source,preprocessing=read_config(args.config,args.mode)
    output=args.output_root.resolve();reference=args.reference_run.resolve();word_root=args.word_root.resolve();artifacts=args.parseq_artifacts.resolve()
    if output==REPO or REPO in output.parents or any(output==r or r in output.parents for r in (reference,word_root)):raise ValueError('Separate external output required')
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Kernel training requires CUDA; no CPU fallback')
    device=torch.device(args.device);device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    if sha256_file(reference/'kernel_only.pt')!=cfg['source_checkpoint_sha256'] or sha256_file(reference/'results.json')!=cfg['reference_results_sha256']:raise ValueError('Qualified extracted source/reference changed')
    if sha256_file(artifacts/'parseq-bb5792a6.pt')!=PARSEQ_CHECKPOINT_SHA256 or sha256_file(artifacts/('parseq-'+PARSEQ_RUNTIME_REVISION+'.tar.gz'))!=PARSEQ_RUNTIME_ARCHIVE_SHA256:raise ValueError('Qualified PARSeq artifacts changed')
    manifest_path=word_root/source['train_manifest']
    if sha256_file(manifest_path)!=source['word_manifest_sha256']:raise ValueError('Dataset manifest changed')
    manifest=load_segmentation_manifest(manifest_path,require_checksums=True)
    train_ids,val_ids,split=_split_indices(manifest,cfg['seed'],source['validation_count'])
    if manifest.split_role!='train' or len(train_ids)!=1130 or len(val_ids)!=125 or split!=source['validation_ids_sha256']:raise ValueError('Training population/split changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');seed_everything(cfg['seed'],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cv2.setNumThreads(1)
    contract={'resolved_config':cfg,'padding_config':padding,'source_config':source,'resolved_preprocessing':preprocessing,
        'source_checkpoint_sha256':cfg['source_checkpoint_sha256'],'reference_results_sha256':cfg['reference_results_sha256'],
        'validation_ids_sha256':split,'active_loss_components':['kernel'],'training_config':asdict(training_config(cfg)),
        'runtime_inputs':{'reference_run':str(reference),'word_root':str(word_root),'parseq_artifacts':str(artifacts)},
        'device':str(device),'gpu_name':torch.cuda.get_device_name(device),'backend':P.source_runner.backend_state(),
        'source_sha256':{f:sha256_file(REPO/f) for f in FILES},'environment':runtime_environment()}
    contract=json.loads(json.dumps(contract))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'run_contract.json'
        if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(contract)):raise ValueError('Run contract changed; use a new output')
        _json(path,contract)
        model=KernelOnlyAGUNet(skip_floor=source['skip_floor'],encoder_pretrained=False)
        original_contract=json.loads((reference/'protocol.json').read_text())
        def initialize():load_checkpoint(reference/'kernel_only.pt',model=model,expected_architecture=model.architecture_identity,expected_run_contract=original_contract)
        initialize()
        transforms=[_transform(preprocessing,'letterbox',train=v) for v in (True,False)]
        datasets=[WordKernelDataset(SegmentationDataset(manifest,word_root,transform=t),distance_fraction=source['targets']['distance_fraction']) for t in transforms]
        sampler=EpochShuffleSampler(len(train_ids),seed=cfg['seed']);options={'num_workers':cfg['num_workers'],'pin_memory':True,'timeout':source['loader_timeout_seconds'] if cfg['num_workers'] else 0}
        train_loader=DataLoader(Subset(datasets[0],train_ids),batch_size=cfg['batch_size'],sampler=sampler,**options)
        val_loader=DataLoader(Subset(datasets[1],val_ids),batch_size=cfg['batch_size'],shuffle=False,**options)
        _prepare_cuda_memory_tracking(device);model.to(device)
        def optimizer():return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_epsilon'],weight_decay=cfg['weight_decay'])
        if not (output/'smoke.json').exists():
            opt=optimizer();before=model.head.weight.detach().clone();smoke=train_kernel_epoch(model,[next(iter(train_loader))],opt,device,cfg)
            update=float((model.head.weight.detach()-before).abs().max());del opt,before
            if smoke['steps']!=1 or update<=0:raise RuntimeError('Real Kernel-only GPU smoke did not update head')
            _json(output/'smoke.json',{'status':'PASS','result_class':'smoke-test result','train':smoke,'head_update_linf':update,'memory':memory(device,source),'run_contract':contract})
            initialize();seed_everything(cfg['seed'],deterministic_algorithms=True)
        opt=optimizer()
        trained=train_loop(output,model,opt,train_loader,val_loader,transforms[0],sampler,cfg,source,contract,device,args.stop_after_epoch)
        del opt
        if not trained['training_complete']:return trained
        load_checkpoint(output/'best.pt',model=model,expected_architecture=model.architecture_identity,expected_run_contract=contract,expected_training_config=training_config(cfg))
        metrics=evaluate_kernel(model,val_loader,device,source,export_dir=output/'validation_exports')
        _json(output/'validation.json',{'aggregate':metrics,'checkpoint':trained['checkpoint'],'run_contract':contract,'result_class':'approximate reproduction'})
        del model;gc.collect();torch.cuda.empty_cache()
        result=padding_ocr(output,reference,manifest,val_ids,word_root,padding,source,artifacts,device)
        result.update(reproduction_mode='approximate',run_contract=contract,trained_checkpoint=trained['checkpoint'],
                      completed_epochs=cfg['epochs'],training_memory=trained['memory'])
        _json(output/'results.json',result)
        _json(output/'suite_success.json',{'status':'SUCCESS','completed_epochs':cfg['epochs'],'results_sha256':sha256_file(output/'results.json'),'result_class':'approximate reproduction'})
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--device',required=True)
    for name in ('reference-run','word-root','parseq-artifacts','output-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stop-after-epoch',type=int);return p


if __name__=='__main__':
    args=parser().parse_args()
    try:run(args)
    except Exception as error:
        args.output_root.mkdir(parents=True,exist_ok=True)
        _json(args.output_root/'failure.json',{'status':'FAILED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'oom':isinstance(error,torch.cuda.OutOfMemoryError),'device':args.device,'precision':'float32',
            'model_set':'single Kernel AG-UNet then PARSeq','segmentation_input_shape':[8,3,512,512],'native_shape':[8,1,256,256],
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None})
        raise
