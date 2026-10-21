"""Frozen V3/447: TRAIN-calibrated graph-cut vs matched geodesic and TEST OCR."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import scipy
import torch

from localize.word_graph_cut import proposal_pool, select
from recognition.parseq import (ParseqApproxConfig,ParseqApproxRecognizer,PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.audit_totaltext_compact_v2_selected_ocr import independent_metrics
from scripts.evaluate_cached_totaltext_padding import crop_hash,correct_word_ids,proposals_for_padding
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
from scripts.review_native128_ocr import inference_config
from segmentation.data import file_sha256,load_segmentation_manifest
from segmentation.evaluate import _model_from_config,_preprocess,validate_checkpoint
from segmentation.training import seed_everything

REPO=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def policies(c):
    result=[{'id':'baseline','method':'baseline'}]
    for method in ['geodesic','graphcut']:
        for e in (range(len(c['graph']['energies'])) if method=='graphcut' else [0]):
            for p in c['calibration']['max_cut_probability']:
                for a in c['calibration']['max_cut_affinity']:
                    result.append({'id':f'{method}-e{e}-p{p}-a{a}','method':method,
                        'energy_index':e,'max_cut_probability':p,'max_cut_affinity':a})
    return result


def encode_labels(labels):
    a=np.asarray(labels,dtype=np.uint8).ravel();starts=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1]
    lengths=np.diff(np.r_[starts,len(a)])
    return {'shape':list(labels.shape),'values':a[starts].tolist(),'lengths':lengths.tolist()}


def as_rows(shards,gt,ocr,policy):
    return [{'sample_id':s['sample_id'],'ground_truth':gt[s['sample_id']],
        'predictions':[{'bbox':p['bbox'],'text':ocr[p['crop_array_sha256']]['raw_text'],'rank':i+1}
            for i,p in enumerate(s['selected'][policy['id']]['proposals'])]} for s in shards]


def metrics(shards,gt,ocr,policy):
    rows=as_rows(shards,gt,ocr,policy);value=evaluate_images(rows,text_available=True)
    n=value['geometry']['retained_proposals'];g=value['geometry']['gt_instances'];tp=value['recognition']['exact_casefold_spatial_one_to_one']
    ids=sorted(correct_word_ids(rows))
    assert len(ids)==tp
    return {'correct_words':tp,'proposals':n,'gt_words':g,'images':len(rows),
        'localized_words':value['geometry']['iou_ge_0_50_one_to_one'],
        'precision':tp/n if n else 0.,'recall':tp/g if g else 0.,'f1':2*tp/(n+g) if n+g else 0.,
        'split_operations':sum(len(s['selected'][policy['id']]['chosen_groups']) for s in shards),
        'changed_images':sum(s['selected'][policy['id']]['changed'] for s in shards),
        'correct_gt_ids':ids,'full_metrics':value}


def choose(validation,all_policies,c):
    baseline=validation['baseline'];output={}
    by_id={p['id']:p for p in all_policies}
    def ranking(item):
        k,v=item
        return (v['correct_words'],v['f1'],-v['changed_images'],-v['split_operations'],k=='baseline',k)
    for method in ['graphcut','geodesic']:
        candidates=[(k,v) for k,v in validation.items() if by_id[k]['method']==method and v['changed_images']>0]
        diagnostic=max(candidates,key=ranking)[0] if candidates else 'baseline'
        eligible=[('baseline',baseline)]+[(k,v) for k,v in candidates
            if v['correct_words']>=baseline['correct_words'] and v['f1']>=baseline['f1']]
        winner=max(eligible,key=ranking)[0]
        output[method]={'promoted_policy':by_id[winner],'diagnostic_policy':by_id[diagnostic],
            'promoted_validation':validation[winner],'diagnostic_validation':validation[diagnostic]}
    return output


def run_phase(name,samples,c,output,active_policies,ocr,device):
    started=time.monotonic();folder=output/name;folder.mkdir(parents=True,exist_ok=False)
    model=_model_from_config(inference_config(Path(c['model_config']))).eval()
    info=validate_checkpoint(Path(c['checkpoint']),model,c['checkpoint_sha256'],c['architecture_identity'])
    assert info['epoch']==447
    model.to(device);shards=[];unique={};paths={};inference_seconds=0.;post_seconds=0.
    data=Path(c['data_root']);counters=Counter()
    historical=Path(c['source_run_root'])/'ocr/area0/epoch-0447/proposals'
    for i,sample in enumerate(samples,1):
        path=data/sample.image_path;paths[sample.sample_id]=path
        if file_sha256(path)!=sample.image_sha256:raise ValueError('RGB source checksum differs')
        with Image.open(path) as im:rgb=im.convert('RGB');image=np.ascontiguousarray(rgb)
        begin=time.monotonic()
        with torch.inference_mode():logits=model(_preprocess(rgb).unsqueeze(0).to(device))
        if tuple(logits.shape)!=(1,1,128,128) or not torch.isfinite(logits).all():raise ValueError('Invalid segmentation output')
        prob=logits.sigmoid()[0,0].cpu().numpy().astype(np.float32)
        inference_seconds+=time.monotonic()-begin
        binary=(prob>=c['graph']['probability_threshold']).astype(np.uint8)
        plan={'padding_native_pixels':[c['graph']['padding_native']],
            'min_unpadded_bbox_area_native':0,'max_proposals_per_image':c['cap']}
        baseline=proposals_for_padding(binary,image.shape[:2],plan)[str(c['graph']['padding_native'])]['proposals']
        for index,p in enumerate(baseline):p.update(parent_index=index,method='baseline')
        if name=='test':
            frozen=json.loads((historical/(sample.sample_id.replace(':','__')+'.json')).read_text())
            if sha256(binary.tobytes()).hexdigest()!=frozen['native_mask_array_sha256']:
                raise ValueError('Native mask replay differs from original V3/447')
            keys=['bbox','bbox_native','padded_bbox_native','padded_bbox_original','contour_index']
            assert [{k:p[k] for k in keys} for p in baseline]==[{k:p[k] for k in keys} for p in frozen['variants']['1']['proposals']]
            counters['exact_native_replays']+=1
        key=sample.sample_id.replace(':','__')
        (folder/'probabilities').mkdir(exist_ok=True)
        np.savez_compressed(folder/'probabilities'/f'{key}.npz',probability=prob)
        begin=time.monotonic();groups,labels,stats=proposal_pool(prob,image,baseline,c['graph'])
        post_seconds+=time.monotonic()-begin;counters.update(stats)
        selected={}
        for policy in active_policies:
            proposals,chosen=select(baseline,groups,policy,c['cap'])
            for p in proposals:
                a,b,d,e=p['padded_bbox_original'];p['crop_array_sha256']=crop_hash(image[b:e,a:d])
                unique.setdefault(p['crop_array_sha256'],(sample.sample_id,p))
            signature=lambda ps:[(p['bbox'],p['padded_bbox_original']) for p in ps]
            selected[policy['id']]={'proposals':proposals,'chosen_groups':chosen,
                'changed':signature(proposals)!=signature(baseline)}
        shard={'sample_id':sample.sample_id,'image_sha256':sample.image_sha256,'image_hw':list(image.shape[:2]),
            'native_mask_sha256':sha256(binary.tobytes()).hexdigest(),'selected':selected,
            'candidate_groups':groups,'generation_stats':stats,
            'partitions':{k:encode_labels(v) for k,v in labels.items()}}
        (folder/'proposals').mkdir(exist_ok=True)
        with gzip.open(folder/'proposals'/f'{key}.json.gz','wt') as f:json.dump(shard,f,ensure_ascii=False,allow_nan=False)
        # Keep numeric proposals in memory, not large partition rasters.
        del shard['partitions'];shards.append(shard)
        if i%25==0 or i==len(samples):
            print(f'[{name}] maps+cuts {i}/{len(samples)} elapsed={time.monotonic()-started:.1f}s solves={counters["solves"]}',flush=True)
    del model,logits,rgb,image,prob,binary,labels;gc.collect();torch.cuda.empty_cache()
    requests=[(k,v) for k,v in unique.items() if k not in ocr]
    print(f'[{name}] crops={len(unique)} cached={len(unique)-len(requests)} new={len(requests)}',flush=True)
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['parseq_source']),
        checkpoint=Path(c['parseq_checkpoint']),device=str(device),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=Path(c['parseq_archive'])))
    previous=None;image=None
    try:
        for start in range(0,len(requests),c['runtime']['parseq_batch_size']):
            batch=requests[start:start+c['runtime']['parseq_batch_size']];crops=[]
            for digest,(sid,p) in batch:
                if sid!=previous:
                    with Image.open(paths[sid]) as im:image=np.ascontiguousarray(im.convert('RGB'))
                    previous=sid
                a,b,d,e=p['padded_bbox_original'];crop=np.ascontiguousarray(image[b:e,a:d])
                if crop_hash(crop)!=digest:raise ValueError('RAM crop checksum differs')
                crops.append(crop)
            decoded=recognizer.recognize_candidate_batch(crops)
            if len(decoded)!=len(batch):raise ValueError('OCR cardinality differs')
            with (folder/'new_ocr.jsonl').open('a') as f:
                for (digest,_),value in zip(batch,decoded):
                    row={'crop_array_sha256':digest,**value.to_dict()};ocr[digest]=row
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if start%512==0 or start+len(batch)==len(requests):print(f'[{name}] PARSeq {start+len(batch)}/{len(requests)}',flush=True)
    finally:recognizer.release()
    del recognizer,image;gc.collect();torch.cuda.empty_cache()
    # Freeze all predictions and readings before any annotation is parsed.
    write(folder/'inference_complete.json',{'status':'SUCCESS','images':len(samples),
        'candidate_generation':dict(counters),'unique_crops':len(unique),'new_crops':len(requests),
        'saved_individual_crops':0,'segmentation_seconds':inference_seconds,'postprocessing_seconds':post_seconds})
    gt={};annotation_records=[]
    for sample in samples:
        path=paths[sample.sample_id];annotation=path.parent.parent/'annotations'/f'{path.stem}.txt'
        with Image.open(path) as im:gt[sample.sample_id]=_gt(annotation,im.size)
        annotation_records.append({'sample_id':sample.sample_id,'path':str(annotation),'sha256':file_sha256(annotation)})
    write(folder/'ground_truth.json',{'ground_truth':gt,'annotations':annotation_records})
    values={p['id']:metrics(shards,gt,ocr,p) for p in active_policies}
    base=values['baseline'];baseids=set(base['correct_gt_ids'])
    for policy in active_policies:
        ids=set(values[policy['id']]['correct_gt_ids'])
        values[policy['id']].update(gained_correct_gt_ids=sorted(ids-baseids),lost_correct_gt_ids=sorted(baseids-ids))
    # Independent per-image NumPy/SciPy audit for every measured policy.
    audited=[]
    for policy in active_policies:
        per=[]
        for shard in shards:
            pred=shard['selected'][policy['id']]['proposals']
            v=independent_metrics(pred,gt[shard['sample_id']],'bbox',ocr)
            per.append(v);audited.append({'policy':policy['id'],'sample_id':shard['sample_id'],**v})
        assert sum(r['exact_casefold_spatial_one_to_one'] for r in per)==values[policy['id']]['correct_words']
        assert sum(r['iou_ge_0_50_one_to_one'] for r in per)==values[policy['id']]['localized_words']
        assert sum(r['retained_proposals'] for r in per)==values[policy['id']]['proposals']
    write(folder/'audit.json',{'status':'PASS','per_image_evaluations':len(audited),'rows':audited})
    if name=='test':
        assert base['images']==300 and base['gt_words']==2204 and base['correct_words']==795
        historical_ids=json.loads((Path(c['source_run_root'])/'ocr/area0/results.json').read_text())['checkpoints']['447']['correct_gt_ids']['1']['unpadded_detection_box']
        assert base['correct_gt_ids']==historical_ids
    elif len(samples)==125:assert base['gt_words']==983
    write(folder/'results.json',{'status':'SUCCESS','claim_label':'approximate reproduction',
        'policies':values,'elapsed_seconds':time.monotonic()-started,'generation':dict(counters),
        'postprocessing_seconds':post_seconds,'new_ocr_crops':len(requests)})
    return values


def run(args):
    c=json.loads(args.config.read_text())
    if args.config.resolve().parent!=REPO/'configs/approximate' or c['reproduction_mode']!='approximate':
        raise ValueError('Explicit isolated approximate config required')
    if not torch.cuda.is_available() or c['runtime']['precision']!='float32':raise ValueError('Qualified CUDA FP32 required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=c['runtime']['cublas_workspace_config']:raise ValueError('Deterministic CUDA workspace must be explicit')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    status={'status':'RUNNING','stage':'preflight','claim_label':'approximate reproduction','pid':os.getpid()}
    write(output/'status.json',status);started=time.monotonic()
    try:
        root=Path(c['source_run_root']);run=root/'training/total-text-guarded-v3-word-selection-500ep-v1'
        manifest=json.loads((run/'run_manifest.json').read_text());contract=json.loads((run/'run_contract.json').read_text())
        selected=json.loads((run/'checkpoint_selection.json').read_text())['best_bbox_word_f1']
        if not manifest['training_complete'] or selected['epoch']!=447 or selected['checkpoint_sha256']!=c['checkpoint_sha256']:
            raise ValueError('Frozen TRAIN-only V3 checkpoint selection differs')
        train_manifest=load_segmentation_manifest(Path(c['train_manifest']),require_checksums=True)
        test_manifest=load_segmentation_manifest(Path(c['test_manifest']),require_checksums=True)
        if file_sha256(Path(c['train_manifest']))!=c['train_manifest_sha256'] or file_sha256(Path(c['test_manifest']))!=c['test_manifest_sha256']:
            raise ValueError('Manifest changed')
        validation_ids=contract['validation_ids']
        assert len(validation_ids)==125 and not set(validation_ids)&set(contract['train_ids'])
        assert sha256(''.join(s+'\n' for s in validation_ids).encode()).hexdigest()==c['validation_ids_sha256']
        by_id={s.sample_id:s for s in train_manifest.samples};samples=[by_id[s] for s in validation_ids]
        if args.limit_validation:samples=samples[:args.limit_validation]
        torch.set_num_threads(c['runtime']['cpu_threads']);torch.set_num_interop_threads(1)
        seed_everything(c['runtime']['seed'],deterministic_algorithms=True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.device(c['runtime']['device']);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        assert file_sha256(Path(c['parseq_checkpoint']))==PARSEQ_CHECKPOINT_SHA256
        assert file_sha256(Path(c['parseq_archive']))==PARSEQ_RUNTIME_ARCHIVE_SHA256
        code=['scripts/run_guarded_v3_graph_cut.py','localize/word_graph_cut.py','localize/crop_recovery.py',
              'models/unet_adv.py','segmentation/evaluate.py','recognition/parseq.py','recognition/scene_text_v1_evaluation.py']
        write(output/'protocol.json',{'config':c,'config_sha256':file_sha256(args.config),
            'claim_label':'approximate reproduction','validation_ids':validation_ids,
            'runtime':{'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,'scipy':scipy.__version__,
                'python':platform.python_version(),'precision':'float32','segmentation_batch_size':1,
                'parseq_batch_size':c['runtime']['parseq_batch_size'],'input_shape':[1,3,256,256],
                'native_shape':[128,128],'model_set':['AuthorProtocolAGUNet V3/447','pinned PARSeq AR/refine1']},
            'code_sha256':{f:file_sha256(REPO/f) for f in code},'source_contract_sha256':file_sha256(run/'run_contract.json'),
            'result_class':'smoke-test result' if args.limit_validation else 'approximate reproduction; validation-calibrated TEST300 diagnostic',
            'metric':'micro unpadded original bboxIoU>=.5 + casefold exact, punctuation retained; care GT; all predictions; cap10',
            'ground_truth_use':'each phase freezes all proposals and OCR before loading annotations; calibration uses TRAIN holdout only'})
        # Import only immutable crop readings; no TEST annotations or metrics.
        source_ocr=root/'ocr/area0/ocr.jsonl'
        ocr={r['crop_array_sha256']:r for r in map(json.loads,source_ocr.read_text().splitlines())}
        all_policies=policies(c);status.update(stage='validation');write(output/'status.json',status)
        validation=run_phase('validation',samples,c,output,all_policies,ocr,device)
        choice=choose(validation,all_policies,c)
        write(output/'selection.json',{'selection_source':'TRAIN validation only','choice':choice,
            'rule':'max exact correct count subject to nondecreasing OCR F1; tie fewer changes; off permitted; best active diagnostic reported separately',
            'test_annotations_loaded':False,'smoke':bool(args.limit_validation)})
        if not args.limit_validation:
            selection_hash=file_sha256(output/'selection.json')
            active={'baseline':{'id':'baseline','method':'baseline'}}
            for m in ['graphcut','geodesic']:
                for role in ['promoted_policy','diagnostic_policy']:
                    p=choice[m][role];active[p['id']]=p
            graph=choice['graphcut']['diagnostic_policy']
            if graph['method']=='graphcut':
                control={**graph,'method':'geodesic','id':'matched-geodesic-control'};active[control['id']]=control
            status.update(stage='test',selection_sha256=selection_hash);write(output/'status.json',status)
            test=run_phase('test',list(test_manifest.samples),c,output,list(active.values()),ocr,device)
            assert file_sha256(output/'selection.json')==selection_hash
            write(output/'comparison.json',{'status':'SUCCESS','claim_label':'approximate reproduction',
                'validation':validation,'selection':choice,'test':test,
                'selection_sha256':selection_hash,'saved_individual_crops':0})
        status.update(status='SUCCESS',stage='complete',elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            out_of_memory=False,saved_individual_crops=0)
        write(output/'status.json',status)
        print(json.dumps(status),flush=True)
    except Exception as e:
        status.update(status='FAILED',error=str(e),traceback=traceback.format_exc(),out_of_memory=isinstance(e,torch.cuda.OutOfMemoryError))
        write(output/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-validation',type=int)
    run(p.parse_args())
