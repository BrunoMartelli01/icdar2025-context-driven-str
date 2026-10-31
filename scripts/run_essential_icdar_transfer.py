"""Four-way ICDAR transfer of the frozen essential gate, with explicit stage caches."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image,ImageDraw
import torch

from recognition.essential_portable_router import predict_portable_essential
from recognition.group_word_selection import reading_confidence
from recognition.unet_corrected_token_interfaces import lexical_counts,route_summary
from recognition.yolo_word_refinement import arm_predictions

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
key=lambda s:s.replace(':','__')

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def pixels(sample):
    if sha(sample['image_path'])!=sample['image_sha256']:raise ValueError('Changed inference RGB')
    rgb=np.asarray(Image.open(sample['image_path']).convert('RGB'))
    if list(rgb.shape[:2][::-1])!=sample['image_wh']:raise ValueError('Changed image dimensions')
    return np.ascontiguousarray(rgb)

def normalize_yolo(rows,cfg):
    return [dict(r,confidence=reading_confidence(r)['confidence_balanced']) for r in arm_predictions(rows,cfg)[cfg['primary_method']]]

def unet_stage(root,plan,limit):
    from recognition.corrected_unet_transfer import CorrectedUNetTransfer
    from scripts.qualify_corrected_unet_transfer import setup
    c=plan['config'];qualification=read(root/'corrected_runtime_qualification_v1.json')
    code=Path(__file__).parents[1]/'recognition/corrected_unet_transfer.py'
    if qualification['status']!='PASS' or qualification['code_sha256']!=sha(code):raise ValueError('Matching complete corrected-expert qualification required')
    setup(c)
    if torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory')
    deployment=read(c['paths']['router_deployment']);runtime=CorrectedUNetTransfer(c);torch.cuda.reset_peak_memory_stats()
    binding=dict(plan_sha256=sha(root/'resolved_configuration.json'),runtime_code_sha256=sha(code));files={};start=time.monotonic()
    for phase in ['calibration','test']:
        for ds,samples in plan['population'][phase].items():
            samples=samples[:limit] if limit else samples
            for number,s in enumerate(samples,1):
                path=root/'unet'/(key(s['sample_id'])+'.json');contract=dict(binding,sample_id=s['sample_id'],image_sha256=s['image_sha256'])
                if path.exists():
                    old=read(path)
                    if old['contract']!=contract or sha(root/old['maps_path'])!=old['maps_sha256']:raise ValueError('Changed U-Net export')
                else:
                    rows,probabilities,trace=runtime.predict(pixels(s));mp=root/'unet_maps'/(key(s['sample_id'])+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(mp,probabilities=probabilities)
                    predicted=predict_portable_essential(deployment,s['sample_id'],rows,s['image_wh'],probabilities)
                    save(path,dict(contract=contract,rows=rows,score=predicted['score'],ground_truth_used=False,
                        maps_path=str(mp.relative_to(root)),maps_sha256=sha(mp),trace=trace));runtime.clear_image_cache()
                files[str(path.relative_to(root))]=sha(path)
                if number%25==0 or number==len(samples):print('UNET',phase,ds,number,len(samples),flush=True)
    result=dict(status='SUCCESS',binding=binding,files_sha256=files,counters=runtime.counters,
        gpu=torch.cuda.get_device_name(),precision='float32',batch_size=dict(unet=1,parseq=c['parseq_batch_size']),
        model_set='U-Net/PARSeq/pre-selector/pointwise selector/GNN/RGB verifier/text editor',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        oom=False,seconds=time.monotonic()-start,result_class='smoke-test result' if limit else 'approximate reproduction')
    runtime.close();save(root/('unet_receipt_limit_'+str(limit)+'.json'),result)

def yolo_stage(root,plan,limit):
    from ultralytics import YOLO
    from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots
    from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer,PARSEQ_ARCHIVE_SOURCE
    c=plan['config'];paths=c['paths'];torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg=read(Path(paths['yolo_cache'])/'icdar13/resolved_configuration.json')['config'];cfg['execution']['parseq_batch_size']=c['parseq_batch_size']
    samples=[s for values in plan['population']['calibration'].values() for s in (values[:limit] if limit else values)]
    binding=dict(plan_sha256=sha(root/'resolved_configuration.json'),code_sha256=sha(__file__))
    torch.cuda.reset_peak_memory_stats();model=YOLO(paths['yolo_checkpoint']);start=time.monotonic();files={}
    for number,s in enumerate(samples,1):
        p=root/'yolo_calibration_detector'/(key(s['sample_id'])+'.json');contract=dict(binding,sample_id=s['sample_id'],image_sha256=s['image_sha256'])
        if not p.exists():
            rgb=pixels(s);shapes=[];handle=model.model.register_forward_pre_hook(lambda m,x:shapes.append(list(x[0].shape)))
            inference=cfg['inference']
            try:output=model.predict(np.ascontiguousarray(rgb[:,:,::-1]),imgsz=inference['image_size'],conf=inference['confidence_floor'],iou=inference['nms_iou'],max_det=inference['max_detections'],device=c['device'],half=False,verbose=False)[0]
            finally:handle.remove()
            slots,rejected=boxes_to_slots(output.boxes.xyxy.cpu().numpy(),output.boxes.conf.cpu().numpy(),output.boxes.cls.cpu().numpy(),tuple(s['image_wh']),inference['confidence_floor'])
            save(p,dict(contract=contract,slots=slots,rejected=rejected,input_shape=shapes[0]))
        elif read(p)['contract']!=contract:raise ValueError('Changed YOLO detector cache')
        files[str(p.relative_to(root))]=sha(p)
    detector_memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del model;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths['parseq_source']),checkpoint=Path(paths['parseq_checkpoint']),
        runtime_source_archive=Path(paths['parseq_archive']),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,device=c['device']))
    from scripts.crop_clip_runtime import array_sha
    crops_count=0
    for number,s in enumerate(samples,1):
        p=root/'yolo_calibration_ocr'/(key(s['sample_id'])+'.json');detector=root/'yolo_calibration_detector'/(key(s['sample_id'])+'.json')
        contract=dict(binding,sample_id=s['sample_id'],detector_sha256=sha(detector),image_sha256=s['image_sha256'])
        if not p.exists():
            rgb=pixels(s);slots=read(detector)['slots'];rows=[]
            for start_index in range(0,len(slots),c['parseq_batch_size']):
                group=slots[start_index:start_index+c['parseq_batch_size']];crops=[np.ascontiguousarray(rgb[r['bbox'][1]:r['bbox'][3],r['bbox'][0]:r['bbox'][2]]) for r in group]
                readings=recognizer.recognize_candidate_batch(crops)
                if len(readings)!=len(group):raise ValueError('Incomplete YOLO/PARSeq batch')
                for r,crop,reading in zip(group,crops,readings):rows.append(dict(candidate_id=s['sample_id']+':yolo:'+str(r['source_index']),bbox=r['bbox'],detector_score=r['score'],crop_rgb_sha256=array_sha(crop),crop_shape=list(crop.shape),**reading.to_dict()))
                crops_count+=len(crops)
            save(p,dict(contract=contract,rows=rows,ground_truth_used=False))
        elif read(p)['contract']!=contract:raise ValueError('Changed YOLO OCR cache')
        files[str(p.relative_to(root))]=sha(p)
        if number%25==0 or number==len(samples):print('YOLO CAL',number,len(samples),flush=True)
    ocr_memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());recognizer.release()
    save(root/('yolo_calibration_receipt_limit_'+str(limit)+'.json'),dict(status='SUCCESS',binding=binding,files_sha256=files,detector_memory=detector_memory,ocr_memory=ocr_memory,
        precision='float32',gpu=torch.cuda.get_device_name(),batch_size=dict(yolo=1,parseq=c['parseq_batch_size']),
        images=len(samples),new_crops=crops_count,seconds=time.monotonic()-start,oom=False))

def cut(scores,fraction):
    eligible=sorted((s for s in scores if scores[s] is not None),key=lambda s:(-scores[s],s));n=round(len(scores)*fraction)
    if not 0<n<=len(eligible):return dict(status='UNATTAINABLE_CALIBRATION_BUDGET',threshold=None,tie_sample_id=None,eligible_images=len(eligible),total_images=len(scores))
    return dict(status='SUCCESS',threshold=scores[eligible[n-1]],tie_sample_id=eligible[n-1],eligible_images=len(eligible),total_images=len(scores))

def accept(scores,policy):
    if policy['threshold'] is None:return []
    return sorted(s for s,v in scores.items() if v is not None and (v>policy['threshold'] or v==policy['threshold'] and s<=policy['tie_sample_id']))

def yolo_rows(root,plan,phase,ds,s):
    path=(root/'yolo_calibration_ocr'/(key(s['sample_id'])+'.json') if phase=='calibration' else Path(plan['config']['paths']['yolo_cache'])/ds/'ocr'/(key(s['sample_id'])+'.json'))
    raw=read(path);cfg=read(Path(plan['config']['paths']['yolo_cache'])/ds/'resolved_configuration.json')['config']
    if raw['contract']['sample_id']!=s['sample_id']:raise ValueError('YOLO cache sample identity mismatch')
    return normalize_yolo(raw['rows'],cfg),path

def score_population(root,plan,phase):
    deployment=read(plan['config']['paths']['router_deployment']);outputs={};scores={};inputs={}
    for ds,samples in plan['population'][phase].items():
        outputs[ds]={name:{} for name in ['unet','yolo']};scores[ds]={name:{} for name in ['unet','yolo']}
        for s in samples:
            sid=s['sample_id'];up=root/'unet'/(key(sid)+'.json');u=read(up);mp=root/u['maps_path']
            if sha(mp)!=u['maps_sha256']:raise ValueError('Changed predicted maps')
            with np.load(mp,allow_pickle=False) as z:prob=z['probabilities']
            yr,yp=yolo_rows(root,plan,phase,ds,s);inputs[str(yp)]=sha(yp);inputs[str(up)]=sha(up);inputs[str(mp)]=sha(mp)
            for name,rows in [('unet',u['rows']),('yolo',yr)]:
                out=predict_portable_essential(deployment,sid,rows,s['image_wh'],prob)
                if name=='unet' and out['score']!=u['score']:raise ValueError('Unchanged U-Net gate score failed replay')
                outputs[ds][name][sid]=rows;scores[ds][name][sid]=out['score']
    return outputs,scores,inputs

def truth_and_validity(s,ds,data_root):
    if sha(s['annotation_path'])!=s['annotation_sha256']:raise ValueError('Changed evaluation annotation')
    if ds=='icdar13':
        from recognition.icdar13_annotations import parse_task1_annotation
        words=parse_task1_annotation(Path(s['annotation_path']),image_size_wh=tuple(s['image_wh']))
        return [dict(text=w.transcription,bbox=[w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1]) for w in words],np.ones((256,256),bool),None
    from recognition.scene_text_datasets import parse_icdar2015_annotation
    from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
    from recognition.clip_router_evaluation import native_mask
    instances,_=clip_instances_to_pixel_extent(parse_icdar2015_annotation(Path(s['annotation_path'])),tuple(s['image_wh']))
    words=[dict(text=w.transcription,bbox=[w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1]) for w in instances if not w.ignore]
    vp=Path(data_root)/'datasets/icdar2015/test/validity'/(s['sample_id'].split(':')[-1]+'.png')
    with Image.open(vp) as image:valid=native_mask(image.convert('L'),256)
    return words,valid,vp

def ignored_rows(rows,wh,valid):
    from recognition.clip_router_evaluation import native_mask
    flags=[]
    for r in rows:
        a,b,c,d=r['bbox'];mask=Image.new('L',tuple(wh),0);ImageDraw.Draw(mask).rectangle([a,b,c-1,d-1],fill=255)
        footprint=native_mask(mask,256);flags.append(bool(footprint.any() and not np.any(footprint&valid)))
    return flags

def evaluate(root,plan):
    c=plan['config'];cal_rows,cal_scores,cal_inputs=score_population(root,plan,'calibration');deployment=read(c['paths']['router_deployment'])
    choices={}
    for ds,experts in cal_scores.items():
        choices[ds]={}
        for name,scores in experts.items():
            choices[ds][name]=[dict(target_calibration_acceptance=f,**cut(scores,f)) for f in c['calibration_acceptances']]
    frozen=dict(mode='approximate',claim_label='approximate reproduction',plan_sha256=sha(root/'resolved_configuration.json'),
        calibration_scores=cal_scores,calibration_inputs_sha256=cal_inputs,policies=choices,original_totaltext_policies=deployment['policies'],
        no_training=True,test_used=False,calibration_ground_truth_used=False)
    fp=root/'target_validation_frozen.json'
    if fp.exists():
        if read(fp)!=frozen:raise ValueError('Frozen validation cuts changed')
    else:save(fp,frozen)
    outputs,scores,inputs=score_population(root,plan,'test');results={};all_branches={};records={}
    from recognition.clip_router_evaluation import deepsolo_predictions
    from recognition.deepsolo_word_f1 import recognition_confidence,apply_policy
    from recognition.unet_corrected_token_interfaces import ranked_ids
    profile=read(c['paths']['deepsolo_profile'])['selected_policy']
    # Dataset root is taken from the already authenticated source population/configuration.
    source_config=read(Path(c['paths']['yolo_cache'])/'icdar2015/resolved_configuration.json')
    data_root=source_config['paths']['datasets']['icdar2015']['data_root']
    for ds,samples in plan['population']['test'].items():
        branches={name:{} for name in ['unet','yolo']};ids=[s['sample_id'] for s in samples];records[ds]={}
        for s in samples:
            sid=s['sample_id'];gt,valid,vp=truth_and_validity(s,ds,data_root);hp=root/'fallback'/(key(sid)+'.json');raw=read(hp)
            if not raw['response']['success'] or raw['contract']['image_sha256']!=s['image_sha256']:raise ValueError('Failed or mismatched native fallback output')
            heavy=deepsolo_predictions(raw['response'],s['image_wh'],s['orientation'],valid,profile['score_threshold']);by={d['source_index']:d for d in raw['response']['detections']}
            for p in heavy:p['recognition_confidence']=recognition_confidence(by[p['instance_id']]['raw']['recognition_class_probabilities'])
            heavy=apply_policy(heavy,profile)
            hc=lexical_counts([dict(raw_text=p['text'] if p.get('invalid_envelope') is None else '') for p in heavy],gt,[p['ignored'] for p in heavy])
            record=dict(ground_truth=gt,deepsolo=heavy,light={},annotation_path=s['annotation_path'],annotation_sha256=s['annotation_sha256'])
            inputs[str(hp)]=sha(hp);inputs[s['annotation_path']]=s['annotation_sha256']
            if vp is not None:inputs[str(vp)]=sha(vp)
            for name in branches:
                rows=outputs[ds][name][sid];ignored=ignored_rows(rows,s['image_wh'],valid);lc=lexical_counts(rows,gt,ignored)
                branches[name][sid]=dict(light=lc,deepsolo=hc);record['light'][name]=dict(rows=rows,ignored=ignored)
            records[ds][sid]=record
        if len(ids)!=c['expected_test_images'][ds] or sum(branches['unet'][s]['light']['gt_words'] for s in ids)!=c['expected_care_words'][ds]:raise ValueError('Full dataset denominator mismatch')
        all_branches[ds]=branches;results[ds]={}
        for name,b in branches.items():
            values=scores[ds][name];policies=[]
            for p in deployment['policies']:
                policies.append(dict(kind='transferred_totaltext_cut',target_calibration_acceptance=p['target_calibration_acceptance'],
                    threshold=p['threshold'],tie_sample_id=p['tie_sample_id'],**route_summary(b,ids,accept(values,p))))
            for p in choices[ds][name]:policies.append(dict(kind='target_validation_quantile',target_calibration_acceptance=p['target_calibration_acceptance'],
                threshold=p['threshold'],tie_sample_id=p['tie_sample_id'],calibration_status=p['status'],**route_summary(b,ids,accept(values,p))))
            quotas=[dict(target_unet_acceptance=f,**route_summary(b,ids,ranked_ids(values,ids,f))) for f in c['report_unet_acceptances']]
            results[ds][name]=dict(policies=policies,quotas=quotas,standalone_light=route_summary(b,ids,ids),standalone_deepsolo=route_summary(b,ids,[]),
                input_contract='Same frozen30-feature head; YOLO native detector confidence replaces the corrected U-Net verifier score only for the YOLO arm, an explicit expert-feature domain shift.')
    save(root/'test_scores.json',scores);save(root/'branches.json',all_branches);save(root/'evaluation_records.json',records);save(root/'evaluation_inputs.json',inputs)
    result=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',datasets=results,
        primary='transferred_totaltext_cut_0.5',validation_secondary='target_validation_quantile_0.5',
        frozen_validation_sha256=sha(fp),new_router_training=False,logical_routing=True,online_latency_measured=False,
        unique_new_deepsolo_test_inferences=sum(c['expected_test_images'].values()),fallback_cache_shared_between_experts=True,
        metric='Symbol-free Unicode casefold+isalnum per-image lexical multiset microF1; native256 ignore; no IoU; all care, empty and failed-image denominators retained.',
        paper_equivalence='Unresolved: paper recognition accuracy undefined; IC13 official233 differs from paper229; approximate reproduction only.')
    save(root/'results.json',result);save(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json')))
    for ds,experts in results.items():
        for name,v in experts.items():
            for p in v['policies']:
                if p['target_calibration_acceptance']==.5:print(ds,name,p['kind'],'F1',p['f1'],'D',p['deepsolo_images'],'N',p['total_images'],'percent',p['deepsolo_percent'],flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',choices=['unet','yolo','evaluate'],required=True)
    parser.add_argument('--limit-per-dataset',type=int,default=0);args=parser.parse_args();plan=read(args.root/'resolved_configuration.json');c=plan['config']
    if c['mode']!='approximate':raise ValueError('Explicit approximate experiment required')
    for path,digest in c['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed scientific source: '+path)
    try:
        if args.stage=='unet':unet_stage(args.root,plan,args.limit_per_dataset)
        elif args.stage=='yolo':yolo_stage(args.root,plan,args.limit_per_dataset)
        else:
            if args.limit_per_dataset:raise ValueError('Full evaluation requires all explicitly expected images')
            evaluate(args.root,plan)
    except Exception as error:
        save(args.root/('failure_'+args.stage+'_'+str(time.time_ns())+'.json'),dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),oom='out of memory' in str(error).lower()));raise

if __name__=='__main__':main()
