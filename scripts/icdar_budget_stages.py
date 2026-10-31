"""Frozen experts, native map features and validation-only lightweight heads."""
from __future__ import annotations
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import hashlib

import numpy as np
from PIL import Image
import torch
import joblib

from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.run_yolo_f1_quality_study import seal
from recognition.icdar_budget_router import observation,lexical_counts,rank,calibration_cut,accept,budget_policy
from recognition.lightweight_unet_router import fit_head,predict_head,paired_utility
from recognition.group_word_selection import reading_confidence
from recognition.f1_quality_study import aggregate
from recognition.trainable_image_router import oracle
from scripts.replay_router_improved_deepsolo import independent_counts,independent_route

REPO=Path(__file__).resolve().parents[1]


def log(stage,**v):print(json.dumps(dict(stage=stage,**v)),flush=True)


def pixels(r):
    if sha(r['image_path'])!=r['image_sha256']:raise ValueError('Image bytes changed')
    with Image.open(r['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
    if list(rgb.shape[:2][::-1])!=r['image_wh']:raise ValueError('Raster frame changed')
    return np.ascontiguousarray(rgb)


def sample_lookup(plan):return {r['sample_id']:r for d in plan['records'].values() for rows in d.values() for r in rows}


def location(root,model,sid):return root/'experts'/model/(key(sid)+'.json')


def checked(root,model,r):
    p=location(root,model,r['sample_id'])
    if not p.exists():return None
    d=read(p)
    if d['image_sha256']!=r['image_sha256'] or d['model']!=model or d['ground_truth_used']:raise ValueError('Expert cache identity changed')
    if d.get('maps_path') and sha(d['maps_path'])!=d['maps_sha256']:raise ValueError('Native predicted maps changed')
    return d


def recognizer(plan):
    from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer,PARSEQ_ARCHIVE_SOURCE
    p=plan['paths'];return ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(p['parseq_source']),checkpoint=Path(p['parseq_checkpoint']),
        runtime_source_archive=Path(p['parseq_archive']),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,device=plan['config']['device']))


def yolo_detect(plan,root):
    from ultralytics import YOLO
    from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots
    cfg=read(plan['paths']['yolo_config']);model=None;done=0
    for r in sample_lookup(plan).values():
        sid=r['sample_id'];target=root/'yolo_detector'/(key(sid)+'.json')
        if target.exists():
            if read(target)['image_sha256']!=r['image_sha256']:raise ValueError('YOLO cache raster changed')
            continue
        prior=Path(plan['paths']['yolo_test_cache'])/sid.split(':')[0]/'detector'/(key(sid)+'.json')
        if r['split']=='test' and prior.exists():
            d=read(prior);seal(target,dict(image_sha256=r['image_sha256'],slots=d['slots'],reused_path=str(prior),reused_sha256=sha(prior)));continue
        if model is None:model=YOLO(plan['paths']['yolo_checkpoint'])
        rgb=pixels(r);settings=cfg['inference'];torch.cuda.reset_peak_memory_stats()
        out=model.predict(np.ascontiguousarray(rgb[:,:,::-1]),imgsz=settings['image_size'],conf=settings['confidence_floor'],iou=settings['nms_iou'],
            max_det=settings['max_detections'],device=plan['config']['device'],half=False,verbose=False)[0]
        slots,rejected=boxes_to_slots(out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy(),out.boxes.cls.cpu().numpy(),tuple(r['image_wh']),settings['confidence_floor'])
        seal(target,dict(image_sha256=r['image_sha256'],slots=slots,rejected=rejected,precision='float32',batch=1,gpu=peak()))
        done+=1
        if done%100==0:log('yolo_detection',new_images=done)
    del model;gc.collect();torch.cuda.empty_cache();log('yolo_detection_complete',new_images=done)


def peak():return dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())


def yolo_read(plan,root):
    from recognition.yolo_word_refinement import arm_predictions
    from scripts.crop_clip_runtime import array_sha
    cfg=read(plan['paths']['yolo_config']);ocr=None;done=0
    for r in sample_lookup(plan).values():
        sid=r['sample_id']
        if checked(root,'yolo',r):continue
        prior=Path(plan['paths']['yolo_test_cache'])/sid.split(':')[0]/'ocr'/(key(sid)+'.json');rgb=pixels(r)
        if r['split']=='test' and prior.exists():
            raw=read(prior)['rows']
            for row in raw:
                a,b,c,d=row['bbox']
                if array_sha(np.ascontiguousarray(rgb[b:d,a:c]))!=row['crop_rgb_sha256']:raise ValueError('Cached YOLO/PARSeq crop changed')
            metadata=dict(reused_path=str(prior),reused_sha256=sha(prior))
        else:
            slots=read(root/'yolo_detector'/(key(sid)+'.json'))['slots'];raw=[]
            if ocr is None:ocr=recognizer(plan)
            torch.cuda.reset_peak_memory_stats()
            for offset in range(0,len(slots),plan['config']['parseq_batch_size']):
                group=slots[offset:offset+plan['config']['parseq_batch_size']];crops=[np.ascontiguousarray(rgb[s['bbox'][1]:s['bbox'][3],s['bbox'][0]:s['bbox'][2]]) for s in group]
                values=ocr.recognize_candidate_batch(crops)
                if len(values)!=len(group):raise RuntimeError('Incomplete PARSeq response')
                for s,v in zip(group,values):raw.append(dict(candidate_id=sid+':'+str(s['source_index']),bbox=s['bbox'],detector_score=s['score'],**v.to_dict()))
            metadata=dict(gpu=peak(),precision='float32',parseq_batch_size=plan['config']['parseq_batch_size'])
        rows=[dict(v,confidence=reading_confidence(v)['confidence_balanced']) for v in arm_predictions(raw,cfg)[cfg['primary_method']]]
        seal(location(root,'yolo',sid),dict(model='yolo',image_sha256=r['image_sha256'],rows=rows,ground_truth_used=False,metadata=metadata))
        done+=1
        if done%100==0:log('yolo_read',images=done)
    if ocr is not None:ocr.release()
    log('yolo_read_complete',images=done)


def unet_records(plan,root,model,records):
    pending=[r for r in records if checked(root,model,r) is None]
    if not pending:return
    c=plan['config'];paths=plan['paths'];device=c['device'];torch.cuda.reset_peak_memory_stats();runtime=None;net=None;ocr=None
    if model=='tt_corrected':
        from recognition.corrected_unet_transfer import CorrectedUNetTransfer
        transfer=read(paths['corrected_transfer_config']);qualified=read(paths['corrected_qualification'])
        if qualified['status']!='PASS' or qualified['code_sha256']!=sha(REPO/'recognition/corrected_unet_transfer.py'):raise ValueError('Corrected-expert runtime qualification changed')
        transfer.update(device=device,parseq_batch_size=c['parseq_batch_size'],cpu_threads=c['cpu_threads']);runtime=CorrectedUNetTransfer(transfer)
    elif model=='tt_raw':
        from models.adaspot_gap import AdaSpotGapAGUNet
        saved=torch.load(paths['tt_checkpoint'],map_location='cpu',weights_only=False);contract=read(paths['tt_contract'])
        if saved['run_contract']!=contract:raise ValueError('Text+Kernel checkpoint contract changed')
        net=AdaSpotGapAGUNet('control',skip_floor=contract['resolved_config']['skip_floor'],initial_gap_bias=-2.)
        net.load_state_dict(saved['model_state_dict'],strict=True);del saved;net.requires_grad_(False).eval().to(device);ocr=recognizer(plan)
    else:
        from segmentation.icdar2015_target70_selection import load_selected_model
        selected=dict(plan['icdar2015_native_selection']);chosen=next(v for v in selected['candidates'] if v['name']==model);selected['selected']=chosen
        net=load_selected_model(selected).requires_grad_(False).eval().to(device);ocr=recognizer(plan)
        from segmentation.icdar2015_target70 import load_profile
        from segmentation.icdar2015_improvements import make_transform
        profile=load_profile(REPO/selected['config']['training_profile']);transform=make_transform(profile,chosen['image_hw'],augment=False)
    try:
        for index,r in enumerate(pending,1):
            sid=r['sample_id'];rgb=pixels(r);tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            if model=='tt_corrected':rows,prob,trace=runtime.predict(rgb);runtime.clear_image_cache();frame='square_letterbox'
            elif model=='tt_raw':
                from segmentation.word_slot_data import image_tensor
                from recognition.unet_token_routing import unet_proposals,selected_rows
                settings=c['tt_raw'];frame='square_letterbox'
                with torch.inference_mode():prob=net(image_tensor(Image.fromarray(rgb)).to(device))[0].sigmoid().cpu().numpy()
                proposals,_,_=unet_proposals(prob,r['image_wh'],settings);raw=[]
                for offset in range(0,len(proposals),c['parseq_batch_size']):
                    group=proposals[offset:offset+c['parseq_batch_size']];crops=[np.ascontiguousarray(rgb[s['bbox'][1]:s['bbox'][3],s['bbox'][0]:s['bbox'][2]]) for s in group]
                    values=ocr.recognize_candidate_batch(crops)
                    if len(values)!=len(group):raise RuntimeError('Incomplete PARSeq response')
                    raw.extend(dict(s,**v.to_dict(),candidate_id=sid+':'+str(s['instance_id'])) for s,v in zip(group,values))
                rows=selected_rows(raw,settings)
            else:
                from segmentation.icdar2015_improvements import binary_proposals,valid_canvas
                from segmentation.icdar2015_operating_policy import component_crop
                from segmentation.icdar2015_target70 import emitted_predictions
                image=Image.fromarray(rgb);tensor,_,_=transform.transform_with_validity(image,Image.new('L',image.size),Image.new('L',image.size,255),0)
                with torch.inference_mode():prob=net(tensor[None].to(device))[0].sigmoid().cpu().numpy()
                policy=chosen['policy'];proposals,labels=binary_proposals((prob[0]>=policy['probability_threshold'])&valid_canvas(image.size,chosen['image_hw']),image.size,chosen['image_hw']);raw=[]
                for offset in range(0,len(proposals),c['parseq_batch_size']):
                    group=proposals[offset:offset+c['parseq_batch_size']];values=ocr.recognize_candidate_batch([component_crop(rgb,labels,s,chosen['image_hw'],policy['crop_method']) for s in group])
                    if len(values)!=len(group):raise RuntimeError('Incomplete PARSeq response')
                    for s,v in zip(group,values):
                        reading=v.to_dict();raw.append(dict(s,**reading,text=reading['raw_text'],sequence_confidence=reading['sequence_confidence_native'],
                            candidate_id=sid+':'+str(s['component_index']),confidence=reading_confidence(reading)['confidence_balanced'],
                            detector_score=float(prob[0][labels==s['component_index']].mean())))
                rows=emitted_predictions(raw,policy['emission_policy']);frame='aspect_letterbox'
            if any('confidence' not in row for row in rows):raise ValueError('Incomplete native OCR observation')
            mp=root/'maps'/model/(key(sid)+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
            if mp.exists():
                with np.load(mp,allow_pickle=False) as z:
                    if not np.array_equal(z['probabilities'],prob):raise ValueError('Orphaned native map differs')
            else:np.savez_compressed(mp,probabilities=prob)
            seal(location(root,model,sid),dict(model=model,image_sha256=r['image_sha256'],rows=rows,maps_path=str(mp),maps_sha256=sha(mp),frame=frame,
                ground_truth_used=False,gpu=peak(),seconds=time.monotonic()-tick,precision='float32',unet_batch=1,parseq_batch=c['parseq_batch_size'],input_shape=[1,3,512,512] if model.startswith('tt_') else [1,3,*chosen['image_hw']]))
            if index%50==0 or index==len(pending):log('unet',model=model,images=index,total=len(pending))
    finally:
        if runtime is not None:runtime.close()
        if ocr is not None:ocr.release()
        del runtime,net,ocr;gc.collect();torch.cuda.empty_cache()


def light_count(root,model,r):
    data=checked(root,model,r);gt,validity=annotation(r);pp=predictions(data['rows'],r,validity);value=lexical_counts(pp,gt)
    if value!=independent_counts(pp,gt,'casefold_alnum'):raise ValueError('Independent lexical count mismatch')
    return value


def select_unet(plan,root):
    lookup=sample_lookup(plan);choices={};trials={}
    for ds,parts in plan['partitions'].items():
        validation=parts['selection']+parts['calibration'];trials[ds]={}
        for candidate in plan['candidates'][ds]:
            unet_records(plan,root,candidate,[lookup[s] for s in validation])
            counts=aggregate(light_count(root,candidate,lookup[s]) for s in parts['selection']);trials[ds][candidate]=counts
        chosen=max(trials[ds],key=lambda n:(trials[ds][n]['f1'],-plan['candidates'][ds].index(n)))
        choices[ds]=chosen;log('unet_selected',dataset=ds,model=chosen,selection_f1=trials[ds][chosen]['f1'])
    seal(root/'unet_selection.json',dict(choices=choices,trials=trials,selection_ids={ds:p['selection'] for ds,p in plan['partitions'].items()},test_used=False))


def fallback_count(plan,root,r):
    if 'replacement_fallback_counts' in plan['config']:
        from recognition.replacement_fallback_counts import replacement_count
        return replacement_count(plan['config'],r)
    from recognition.clip_router_evaluation import deepsolo_predictions
    from recognition.deepsolo_word_f1 import recognition_confidence,apply_policy
    if r['split']=='train':p=root/'fallback_train'/(key(r['sample_id'])+'.json')
    else:p=Path(plan['paths']['fallback_test_cache'])/(key(r['sample_id'])+'.json')
    if not p.exists():raise ValueError('Fallback inference pending: '+str(p))
    row=read(p)
    payload={k:v for k,v in row.items() if k!='payload_sha256'}
    if row['contract']['image_sha256']!=r['image_sha256'] or not row['response']['success'] or row['payload_sha256']!=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():raise ValueError('Fallback raw shard failed')
    expected_profile=sha(plan['paths']['deepsolo_profile'])
    if row['contract']['profile_sha256']!=expected_profile:raise ValueError('Fallback policy differs')
    gt,validity=annotation(r);policy=read(plan['paths']['deepsolo_profile'])['selected_policy']
    pp=deepsolo_predictions(row['response'],r['image_wh'],r['orientation'],validity,policy['score_threshold']);by={d['source_index']:d for d in row['response']['detections']}
    for p0 in pp:p0['recognition_confidence']=recognition_confidence(by[p0['instance_id']]['raw']['recognition_class_probabilities'])
    pp=apply_policy(pp,policy);value=lexical_counts(pp,gt)
    if value!=independent_counts(pp,gt,'casefold_alnum'):raise ValueError('Independent fallback count mismatch')
    return value


def collect(plan,root,ds,model,ids):
    lookup=sample_lookup(plan);features={};branches={};keys={}
    for sid in ids:
        r=lookup[sid];d=checked(root,model,r)
        if d is None:raise ValueError('Expert output missing: '+sid)
        if 'maps_path' in d:
            with np.load(d['maps_path'],allow_pickle=False) as z:maps=z['probabilities']
        else:maps=None
        f=observation(d['rows'],r['image_wh'],maps,d.get('frame'),settings=plan['config']['features'])
        features[sid]=dict(groups=f,scale=len(d['rows'])+1,eligible=bool(d['rows']) and all(r['raw_text'].strip() for r in d['rows']))
        branches[sid]=dict(light=light_count(root,model,r),deepsolo=fallback_count(plan,root,r));keys[sid]=r['rgb_sha256']
    return features,branches,keys


def route(branches,ids,accepted):
    metrics=independent_route(branches,ids,accepted)
    return dict(combined=metrics,accepted_ids=sorted(accepted),accepted_images=len(accepted),deepsolo_images=len(ids)-len(accepted),
        total_images=len(ids),deepsolo_percentage=100*(len(ids)-len(accepted))/len(ids),scope='logical routing on cached expert outputs')


def score(bundle,features,ids):
    matrix=np.stack([np.concatenate([features[s]['groups'][g] for g in bundle['groups']]) for s in ids]);values=predict_head(bundle,matrix)
    return {s:float(v)*features[s]['scale'] if features[s]['eligible'] else None for s,v in zip(ids,values)}


def train(plan,root):
    selected=read(root/'unet_selection.json')['choices'];frozen={};trials={}
    for ds,p in plan['partitions'].items():
        primary,targets=budget_policy(plan['config'],ds)
        frozen[ds]={};trials[ds]={}
        for arm,model in [('yolo','yolo'),('unet',selected[ds])]:
            ids=p['fit']+p['selection']+p['calibration'];features,branches,keys=collect(plan,root,ds,model,ids)
            seal(root/'train_branches'/(ds+'__'+arm+'.json'),dict(branches=branches,tie_keys=keys))
            archive=root/'training_features'/(ds+'__'+arm+'.npz');archive.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(archive,ids=np.array(ids),word_scale=np.array([features[s]['scale'] for s in ids]),
                eligible=np.array([features[s]['eligible'] for s in ids]),
                **{g:np.stack([features[s]['groups'][g] for s in ids]) for g in features[ids[0]]['groups']})
            seal(archive.with_suffix('.json'),dict(npz_sha256=sha(archive),fit_ids=p['fit'],selection_ids=p['selection'],calibration_ids=p['calibration'],
                groups=list(features[ids[0]]['groups']),expert=model,actual_reference_f1=aggregate(branches[s]['deepsolo'] for s in p['fit'])['f1']))
            reference=aggregate(branches[s]['deepsolo'] for s in p['fit'])['f1'];weights=np.array([features[s]['scale'] for s in p['fit']],float)
            target=np.array([paired_utility(branches[s],reference)/features[s]['scale'] for s in p['fit']]);choices=[]
            group_sets=[['ocr'],['ocr','tokens']]
            if arm=='unet':group_sets += [['ocr','maps'],['ocr','tokens','maps']]
            for groups in group_sets:
                x=np.stack([np.concatenate([features[s]['groups'][g] for g in groups]) for s in p['fit']])
                for index,recipe in enumerate(plan['config']['heads']):
                    bundle=fit_head(groups,x,target,weights,recipe,plan['config']['seed'])
                    bundle.update(training_ids=p['fit'],reference_f1=reference,expert=model)
                    name=ds+'__'+arm+'__'+'_'.join(groups)+'__'+str(index);path=root/'models'/(name+'.joblib');path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():raise ValueError('Refuse to refit an existing trained head')
                    joblib.dump(bundle,path);values=score(bundle,features,p['selection']);point=route(branches,p['selection'],rank(values,p['selection'],primary,keys))
                    choices.append(dict(name=name,model_path=str(path.relative_to(root)),model_sha256=sha(path),groups=groups,recipe=recipe,
                        scalar_count=bundle['stored_scalar_count'],dimension=bundle['feature_dimension'],selection=point,serialized_bytes=path.stat().st_size))
            best=max(v['selection']['combined']['f1'] for v in choices);eligible=[v for v in choices if v['selection']['combined']['f1']>=best-plan['config']['selection_tolerance']]
            choice=min(eligible,key=lambda v:(v['scalar_count'],v['serialized_bytes'],-v['selection']['combined']['f1'],v['name']))
            bundle=joblib.load(root/choice['model_path']);values=score(bundle,features,p['calibration'])
            policies=[dict(calibration_cut(values,p['calibration'],q,keys),model_sha256=choice['model_sha256'],selected_on='VALIDATION_CALIBRATION',
                calibration=route(branches,p['calibration'],accept(values,calibration_cut(values,p['calibration'],q,keys),keys))) for q in targets]
            frozen[ds][arm]=dict(expert=model,choice=choice,primary_acceptance=primary,policies=policies,calibration_scores=values,training_ids=p['fit'],selection_ids=p['selection'],calibration_ids=p['calibration'])
            trials[ds][arm]=choices;log('router_frozen',dataset=ds,arm=arm,features=choice['groups'],head=choice['recipe'])
    seal(root/'trials.json',trials);seal(root/'frozen_choices.json',dict(choices=frozen,test_used_for_selection=False,no_refit=True,plan_sha256=sha(root/'resolved_configuration.json')))


def evaluate(plan,root):
    frozen=read(root/'frozen_choices.json');out={};checks=0
    for ds,p in plan['partitions'].items():
        primary,targets=budget_policy(plan['config'],ds)
        out[ds]={}
        for arm,choice in frozen['choices'][ds].items():
            path=root/choice['choice']['model_path']
            if sha(path)!=choice['choice']['model_sha256']:raise ValueError('Model changed after calibration')
            bundle=joblib.load(path)
            if bundle['training_ids']!=p['fit']:raise ValueError('Actual fitting scope changed')
            features,branches,keys=collect(plan,root,ds,choice['expert'],p['test']);scores=score(bundle,features,p['test']);thresholds=[];quotas={};oracles={}
            if plan['result_class']!='smoke-test result' and (len(p['test'])!=plan['config']['expected_test_images'][ds] or sum(v['light']['gt_words'] for v in branches.values())!=plan['config']['expected_test_gt'][ds]):
                raise ValueError('Complete official TEST image/care denominator changed')
            for policy in choice['policies']:
                if policy['model_sha256']!=sha(path):raise ValueError('Cut refers to different weights')
                thresholds.append(dict(policy=policy,**route(branches,p['test'],accept(scores,policy,keys))));checks+=1
            for q in targets:
                accepted=rank(scores,p['test'],q,keys);quotas[str(q)]=route(branches,p['test'],accepted)
                upper=oracle(branches,p['test'],[s for s in p['test'] if scores[s] is not None],len(accepted));oracles[str(q)]=dict(**route(branches,p['test'],upper['accepted_ids']),certificate=upper['certificate_max_residual']);checks+=1
            seal(root/'test_scores'/(ds+'__'+arm+'.json'),dict(scores=scores,tie_keys=keys,branches=branches))
            out[ds][arm]=dict(expert=choice['expert'],head=choice['choice'],primary_acceptance=primary,thresholds=thresholds,predeclared_batch_quotas=quotas,conditional_gt_oracles=oracles,
                all_light=route(branches,p['test'],p['test']),all_deepsolo=route(branches,p['test'],[]))
    seal(root/'results.json',dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',result_class=plan['result_class'],
        datasets=out,partitions={ds:{k:len(v) for k,v in p.items()} for ds,p in plan['partitions'].items()},
        frozen_choices_sha256=sha(root/'frozen_choices.json'),actual_new_deepsolo_calls=plan.get('actual_new_deepsolo_calls',len([r for d in plan['records'].values() for r in d['train']])),
        test_used_for_selection=False,no_refit=True,routing_scope='Cached expert outputs; actual offline inference distinct from logical fallback counts'))
    seal(root/'independent_audit.json',dict(status='PASS',checks=checks,results_sha256=sha(root/'results.json'),scope='Independent multiset counts, fitting scope, bound cuts and exact oracle certificates; supplemental audit required'))
    seal(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),audit_sha256=sha(root/'independent_audit.json')))
    log('COMPLETE')


def execute(root,stage):
    plan=read(root/'resolved_configuration.json');c=plan['config'];torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    import cv2
    cv2.setNumThreads(1)
    try:
        for file,digest in c['pinned_inputs'].items():
            if sha(file)!=digest:raise ValueError('Pinned scientific source changed after preparation: '+file)
        if stage not in ('train','evaluate') and torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory; no device substitution')
        if stage=='yolo_detect':yolo_detect(plan,root)
        elif stage=='yolo_read':yolo_read(plan,root)
        elif stage=='unet_selection':select_unet(plan,root)
        elif stage in ('unet_fit','unet_test'):
            if stage=='unet_test' and not (root/'frozen_choices.json').exists():raise ValueError('Freeze routing before TEST U-Net inference')
            lookup=sample_lookup(plan);choices=read(root/'unet_selection.json')['choices']
            for ds,model in choices.items():unet_records(plan,root,model,[lookup[s] for s in plan['partitions'][ds]['fit' if stage=='unet_fit' else 'test']])
        elif stage=='train':train(plan,root)
        elif stage=='evaluate':evaluate(plan,root)
    except BaseException as e:
        seal(root/('failure_'+stage+'_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=traceback.format_exc(),out_of_memory=isinstance(e,torch.cuda.OutOfMemoryError),gpu=peak() if torch.cuda.is_initialized() else None));raise
