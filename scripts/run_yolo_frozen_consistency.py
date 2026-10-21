"""APPROXIMATE frozen-model routing: scale/crop stability, coverage and CLIP controls."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from recognition.clip_router_prompt_ablation import budget_accept, coverage_threshold
from recognition.parseq import (ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_CHECKPOINT_SHA256, verify_parseq_runtime_source)
from recognition.training_free_consistency_router import FAMILIES, make_features, calibrate, evaluate
from recognition.yolo_router_comparison import aggregate
from recognition.yolo_word_refinement import crop_box
from scripts import run_yolo_router_comparison as base
from scripts import run_yolo_coverage_resolution as previous
from scripts.crop_clip_runtime import sha, array_sha
from scripts.run_clip_high_recall_admission import seal

REPO=Path(__file__).resolve().parents[1]
CODE=('scripts/run_yolo_frozen_consistency.py','recognition/training_free_consistency_router.py',*previous.CODE)


def prepare(args):
    for path in [args.config,args.paths]:
        if path.resolve().parent!=REPO/'configs/approximate':
            raise ValueError('Use only isolated configs/approximate profiles')
    cfg=base.read(args.config);paths=base.read(args.paths)
    if (args.mode!='approximate' or cfg['reproduction_mode']!='approximate'
            or paths['reproduction_mode']!='approximate' or cfg['claim_label']!='approximate reproduction'
            or cfg['table_5_claim'] is not False or cfg['families']!=list(FAMILIES)
            or cfg['crop_fractions']!=[-.05,.05] or cfg['base_resolution']!=640 or cfg['check_resolution']!=960
            or cfg['execution']['device']!='cpu' or cfg['execution']['precision']!='float32'
            or cfg['quality_f1_threshold']!=.8 or cfg['minimum_calibration_accepted']<1):
        raise ValueError('Unsupported frozen consistency protocol')
    source=Path(paths['source_run']);legacy=Path(paths['legacy_crop_run'])
    output=(args.output_root or Path(paths['output_root'])).resolve()
    if output.is_relative_to(REPO) or any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
            for p in [source,legacy]):
        raise ValueError('Output must be external and isolated from all frozen inputs')
    p=base.checked(source/'resolved_configuration.json',cfg['source_plan_sha256'])
    r=base.checked(source/'results.json',cfg['source_results_sha256'])
    v=base.checked(source/'analysis/verification.json',cfg['source_verification_sha256'])
    fit=base.checked(source/'640/frozen_train_selector.json',cfg['source_selector_sha256'])
    lp=base.checked(legacy/'resolved_configuration.json',cfg['legacy_plan_sha256'])
    if (r['status']!='SUCCESS' or v['status']!='PASS' or v['results_sha256']!=cfg['source_results_sha256']
            or p['result_class']!='APPROXIMATE_EXPLORATORY_COMPARISON'
            or lp['device']!='cpu' or lp['precision']!='float32'
            or lp['parseq_checkpoint_sha256']!=PARSEQ_CHECKPOINT_SHA256):
        raise ValueError('Complete paired source and CPU view provenance are required')
    for name,digest in p['code_sha256'].items():
        if sha(REPO/name)!=digest or sha(source/'code_snapshot'/name)!=digest:
            raise ValueError('Pinned scientific source changed: '+name)
    software={k:importlib.metadata.version(k) for k in p['software']}
    if software!=p['software']:raise ValueError('Runtime differs from the frozen source')
    runtime=p['baseline_runtime_paths']
    if sha(Path(runtime['parseq_checkpoint']))!=PARSEQ_CHECKPOINT_SHA256:
        raise ValueError('PARSeq checkpoint changed')
    identity=verify_parseq_runtime_source(Path(runtime['parseq_source']),source_type=PARSEQ_ARCHIVE_SOURCE,
        source_archive=Path(runtime['parseq_archive']))
    if identity!=p['baseline_plan']['parseq_identity']:raise ValueError('PARSeq source identity changed')
    samples={s:list(rows) for s,rows in p['samples'].items()}
    selection=list(p['calibration_ids']);calibration=list(p['validation_ids'])
    if args.smoke:
        selection=selection[:2];calibration=calibration[:2]
        samples['train']=[s for s in samples['train'] if s['sample_id'] in selection+calibration]
        samples['test']=samples['test'][:2]
    if set(selection)&set(calibration):raise ValueError('Family selection and threshold calibration overlap')
    inputs={};features={};branches={}
    for split,rows in samples.items():
        features[split]={};branches[split]={}
        for s in rows:
            sid=s['sample_id'];key=base.key(sid)+'.json';pair={}
            if sha(Path(s['image_path']))!=s['image_sha256']:raise ValueError('Original image changed')
            for resolution in (640,960):
                fp=source/str(resolution)/split/'features'/key
                f=base.checked_shard(fp,cfg['source_plan_sha256'])
                if f['image_sha256']!=s['image_sha256']:raise ValueError('Paired feature image differs')
                inputs[str(fp)]=sha(fp);pair[resolution]=f
            features[split][sid]=pair
            branches[split][sid]=r['resolutions']['640']['branch_metrics'][split][sid]
            if split=='test':
                cp=legacy/'crop_views'/key;cv=base.read(cp);dp=legacy/'detector'/key
                if cv['contract']!={'plan_sha256':cfg['legacy_plan_sha256'],'detector_sha256':sha(dp),'family':'views'}:
                    raise ValueError('Cached crop views have incompatible provenance')
                if cv['sample_id']!=sid or cv['ground_truth_use']!='none':raise ValueError('Cached views differ')
                inputs[str(cp)]=sha(cp);inputs[str(dp)]=sha(dp)
                expected={(r['candidate_id'],f) for r in pair[640]['rows'] for f in cfg['crop_fractions']}
                found=[(r['original_candidate_id'],r['fraction']) for r in cv['rows']
                    if (r['original_candidate_id'],r['fraction']) in expected]
                if len(found)!=len(set(found)) or set(found)!=expected:raise ValueError('Missing or duplicate TEST views')
    plan={'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'result_class':'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_COMPARISON',
        'table_5_claim':False,'config':cfg,'paths':paths,'config_sha256':sha(args.config),'paths_sha256':sha(args.paths),
        'samples':samples,'selection_ids':selection,'threshold_calibration_ids':calibration,
        'source_runtime_paths':runtime,'source_selector':fit,'parseq_identity':identity,'software':software,
        'python':sys.version,'code_sha256':{name:sha(REPO/name) for name in CODE},'input_sha256':inputs,
        'metric':r['metric'],'test_labels_used_for_selection':False,'new_network_training':False,
        'test_history':'TEST and TRAIN holdout previously explored; empirical comparison, not untouched validation'}
    root=output/('smoke' if args.smoke else 'full')/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg,plan,root,features,branches


def state_sha(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rgb_image(sample):
    if sha(Path(sample['image_path']))!=sample['image_sha256']:raise ValueError('Image changed during inference')
    with Image.open(sample['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
    if list(rgb.shape[:2][::-1])!=sample['image_wh']:raise ValueError('Stored RGB frame changed')
    return rgb


def views_for_split(cfg,plan,root,split,features):
    ph=sha(root/'resolved_configuration.json');records={};recognizer=None;before=None;shapes=[];start=time.monotonic()
    try:
        for index,s in enumerate(plan['samples'][split],1):
            sid=s['sample_id'];key=base.key(sid)+'.json';target=root/split/'views'/key
            if target.exists():records[sid]=base.checked_shard(target,ph);continue
            rgb=rgb_image(s);rows=features[sid][640]['rows'];proposals=[];crops=[]
            for row in rows:
                x1,y1,x2,y2=row['bbox']
                if array_sha(rgb[y1:y2,x1:x2])!=row['crop_rgb_sha256']:raise ValueError('Original640 crop changed')
                for fraction in cfg['crop_fractions']:
                    box=crop_box(row['bbox'],tuple(s['image_wh']),fraction);x1,y1,x2,y2=box
                    crop=np.ascontiguousarray(rgb[y1:y2,x1:x2]);crops.append(crop)
                    proposals.append({'original_candidate_id':row['candidate_id'],'fraction':fraction,
                        'bbox':row['bbox'],'crop_bbox':box,'crop_rgb_sha256':array_sha(crop),'crop_shape':list(crop.shape)})
            if split=='test':
                cp=Path(plan['paths']['legacy_crop_run'])/'crop_views'/key
                if sha(cp)!=plan['input_sha256'][str(cp)]:raise ValueError('TEST crop cache changed')
                source=base.read(cp);bykey={(r['original_candidate_id'],r['fraction']):r for r in source['rows']}
                values=[]
                for proposal in proposals:
                    old=bykey[(proposal['original_candidate_id'],proposal['fraction'])]
                    if any(old[k]!=value for k,value in proposal.items()):raise ValueError('Cached crop geometry/pixels changed')
                    if old['ocr_execution']['device']!='cpu' or old['ocr_execution']['precision']!='float32':
                        raise ValueError('Cached PARSeq execution differs')
                    values.append({**old,**proposal})
                receipt={'new_inference':False,'source_cache':str(cp),'source_sha256':sha(cp),'device':'cpu','precision':'float32'}
            else:
                if crops and recognizer is None:
                    runtime=plan['source_runtime_paths']
                    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(runtime['parseq_source']),
                        checkpoint=Path(runtime['parseq_checkpoint']),device='cpu',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=Path(runtime['parseq_archive'])))
                    recognizer.model.requires_grad_(False)
                    if recognizer.model.training or any(p.requires_grad for p in recognizer.model.parameters()):
                        raise RuntimeError('PARSeq must remain eval/frozen')
                    before=state_sha(recognizer.model)
                values=[];batch=cfg['execution']['parseq_batch_size'];image_start=time.monotonic();batch_shapes=[]
                for offset in range(0,len(crops),batch):
                    group=crops[offset:offset+batch];decoded=recognizer.recognize_candidate_batch(group)
                    if len(decoded)!=len(group):raise RuntimeError('Incomplete perturbed-crop OCR batch')
                    shape=[len(group),3,32,128];shapes.append(shape);batch_shapes.append(shape)
                    values.extend({**proposal,**v.to_dict()} for proposal,v in zip(proposals[offset:offset+batch],decoded))
                receipt={'new_inference':bool(crops),'device':'cpu','precision':'float32','seconds':time.monotonic()-image_start,
                    'batch_shapes':batch_shapes,'peak_allocated_bytes':None,'peak_reserved_bytes':None}
            records[sid]=base.save_shard(target,ph,{'sample_id':sid,'image_sha256':s['image_sha256'],
                'rows':values,'execution':receipt,'ground_truth_used':False,'original_outputs_changed':False})
            if index%25==0 or index==len(plan['samples'][split]):
                print(f'[crop probes {split}] {index}/{len(plan["samples"][split])}',flush=True)
        if recognizer is not None:
            after=state_sha(recognizer.model)
            if before!=after:raise RuntimeError('Frozen PARSeq model state changed')
            seal(root/split/'model_receipt.json',{'model':'qualified PARSeq','device':'cpu','precision':'float32',
                'state_sha256_before':before,'state_sha256_after':after,'optimizer_steps':0,'input_shapes':shapes,
                'seconds_including_model_load_and_hashing':time.monotonic()-start,
                'peak_allocated_bytes':None,'peak_reserved_bytes':None})
    finally:
        if recognizer is not None:recognizer.release()
        del recognizer;gc.collect()
    return records


def features_for_split(cfg,plan,root,split,source_features,views):
    ph=sha(root/'resolved_configuration.json');result={}
    for s in plan['samples'][split]:
        sid=s['sample_id'];path=root/split/'features'/(base.key(sid)+'.json')
        if path.exists():result[sid]=base.checked_shard(path,ph);continue
        f=make_features(source_features[sid][640],source_features[sid][960],views[sid]['rows'],
            reference=plan['source_selector']['confidence_reference'],previous_selector=plan['source_selector']['winner'],
            fractions=cfg['crop_fractions'],min_iou=cfg['scale_pair_iou'])
        result[sid]=base.save_shard(path,ph,{'sample_id':sid,'image_sha256':s['image_sha256'],**f})
    return result


def freeze_thresholds(cfg,plan,root,features,branches):
    selection=plan['selection_ids'];calibration=plan['threshold_calibration_ids'];frozen={'families':{},'selected_families':{},'selected_budget_families':{}}
    trials={str(t):[] for t in cfg['target_good_image_fractions']}
    for family in FAMILIES:
        values={s:features[s]['scores'][family] for s in features}
        frozen['families'][family]={'targets':{},'coverage_cuts':{
            str(t):coverage_threshold(values,calibration,t) for t in cfg['bypass_fractions']}}
        for target in cfg['target_good_image_fractions']:
            kw=dict(target=target,quality_threshold=cfg['quality_f1_threshold'],minimum_accepted=cfg['minimum_calibration_accepted'])
            candidate=calibrate({s:values[s] for s in selection},{s:branches[s] for s in selection},selection,**kw)
            trials[str(target)].append({'family':family,**candidate})
            frozen['families'][family]['targets'][str(target)]=calibrate(
                {s:values[s] for s in calibration},{s:branches[s] for s in calibration},calibration,**kw)
    for target,rows in trials.items():
        best=min(rows,key=lambda r:(-(r['calibration']['accepted_images'] if r['calibration'] else 0),
            -(r['calibration']['good_fraction'] if r['calibration'] else 0),FAMILIES.index(r['family'])))
        frozen['selected_families'][target]=best['family']
    budget_trials={}
    for fraction in cfg['bypass_fractions']:
        rows=[]
        for family in FAMILIES:
            values={s:features[s]['scores'][family] for s in selection}
            accepted=budget_accept(values,selection,round(len(selection)*fraction))
            if len(accepted)!=round(len(selection)*fraction):raise ValueError('Selection budget cannot be filled')
            point=evaluate(branches,selection,accepted,cfg['quality_f1_threshold'])
            rows.append({'family':family,'accepted_good_images':point['accepted_good_images']})
        best=min(rows,key=lambda r:(-r['accepted_good_images'],FAMILIES.index(r['family'])))
        budget_trials[str(fraction)]=rows;frozen['selected_budget_families'][str(fraction)]=best['family']
    frozen.update(selection_trials=trials,selection_ids=selection,threshold_calibration_ids=calibration,
        budget_selection_trials=budget_trials,test_labels_used=False,models_trained=False,formal_risk_guarantee=False)
    path=root/'frozen_thresholds.json';count=len(list((root/'test/features').glob('*.json')))
    if path.exists():frozen['test_features_present_at_freeze']=base.read(path)['test_features_present_at_freeze']
    elif count:raise ValueError('Refusing new threshold calibration after TEST consistency features exist')
    else:frozen['test_features_present_at_freeze']=0
    seal(path,frozen)
    return frozen


def split_metrics(cfg,features,branches,ids,frozen):
    result={'images':len(ids),'families':{},'light_only':aggregate(branches[s]['light'] for s in ids),
        'deepsolo_only':aggregate(branches[s]['deepsolo'] for s in ids)}
    for family in FAMILIES:
        values={s:features[s]['scores'][family] for s in ids};points={}
        for target in cfg['target_good_image_fractions']:
            f=frozen['families'][family]['targets'][str(target)];cut=f['threshold']
            accepted={s for s in ids if values[s] is not None and cut is not None and values[s]>=cut}
            points['quality_'+str(target)]={'threshold':cut,'calibration_status':f['status'],
                **evaluate(branches,ids,accepted,cfg['quality_f1_threshold'])}
        for fraction in cfg['bypass_fractions']:
            count=round(len(ids)*fraction);accepted=budget_accept(values,ids,count)
            if len(accepted)!=count:raise ValueError('Not enough nonempty OCR images for the requested rank budget')
            points['budget_'+str(fraction)]={'threshold':None,**evaluate(branches,ids,accepted,cfg['quality_f1_threshold'])}
            cut=frozen['families'][family]['coverage_cuts'][str(fraction)]
            accepted={s for s in ids if values[s] is not None and cut is not None and values[s]>=cut}
            points['coverage_cut_'+str(fraction)]={'threshold':cut,**evaluate(branches,ids,accepted,cfg['quality_f1_threshold'])}
        result['families'][family]=points
    return result


def write_report(root,cfg,result):
    rows=[]
    for split,item in result['splits'].items():
        for family,points in item['families'].items():
            for rule,p in points.items():
                rows.append({'split':split,'family':family,'rule':rule,'threshold':p['threshold'],
                    **{k:p[k] for k in ('images','accepted_images','fallback_images','accepted_good_images','accepted_bad_images','good_fraction_among_accepted','good_image_recall')},
                    **{k:p['combined'][k] for k in ('tp','fp','fn','detected_words','f1','accuracy_tn0')}})
    path=root/'comparison.csv'
    import io
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if path.exists():
        with path.open(newline='') as current:
            if list(csv.DictReader(current))!=list(csv.DictReader(io.StringIO(stream.getvalue()))):raise ValueError('CSV report changed')
    else:path.write_text(stream.getvalue())
    target=str(cfg['primary_good_image_fraction']);winner=result['selected_families'][target]
    lines=['# Frozen-model consistency routing','', '**Approximate reproduction; exploratory comparison.** No model training or output-word changes.',
        'Quality means actual per-image local F1>=0.8. Good-image precision differs from word precision and final-system F1.',
        'Family selection uses TRAIN80; threshold calibration uses separate TRAIN45. TEST300 was explored historically. No formal risk guarantee.','',
        f'Primary target: {target} good-image fraction; TRAIN-selected family: `{winner}`.','',
        '| Family | TRAIN45 accepted/good | TEST accepted | TEST good | Good among accepted | Final TEST F1 | Final TEST TN0 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        p=result['splits']['test']['families'][family]['quality_'+target]
        c=result['splits']['calibration']['families'][family]['quality_'+target]
        frac=f'{100*p["good_fraction_among_accepted"]:.2f}%' if p['accepted_images'] else 'undefined (abstain all)'
        lines.append(f'| {family} | {c["accepted_images"]}/{c["accepted_good_images"]} | {p["accepted_images"]} | '
            f'{p["accepted_good_images"]} | {frac} | {p["combined"]["f1"]*100:.4f}% | {p["combined"]["accuracy_tn0"]*100:.4f}% |')
    lines.extend(['','## Fixed GT-free batch budgets, selected family','',
        '| Bypass | Family selected on TRAIN80 | TEST accepted/good | Good among accepted | Final F1 | Final TN0 | Predicted words |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for fraction in cfg['bypass_fractions']:
        family=result['selected_budget_families'][str(fraction)]
        p=result['splits']['test']['families'][family]['budget_'+str(fraction)]
        quality=f'{p["good_fraction_among_accepted"]*100:.2f}%' if p['accepted_images'] else 'undefined'
        lines.append(f'| {fraction} | {family} | {p["accepted_images"]}/{p["accepted_good_images"]} | {quality} | '
            f'{p["combined"]["f1"]*100:.4f}% | {p["combined"]["accuracy_tn0"]*100:.4f}% | {p["combined"]["detected_words"]} |')
    lines.extend(['','The output branch always retains the exact original640 word set or full frozen ViTAE DeepSolo output.',
        'Perturbed crops,960 readings, coverage and CLIP scores are probes only. Thresholds are empirical; failure to find one means fallback on every image.',
        'Larger resolutions/crop views cost inference even when reused here. No full-pipeline speed or new GPU qualification is claimed.',
        'Counts preserve all images/GT/empty predictions with the existing local casefold/punctuation/ignore protocol, not official polygon WordSpotting.',
        'All targets/families, per-image scores, state hashes and scalar/budget curves remain in results.json, features, model_receipt.json and comparison.csv.'])
    text='\n'.join(lines)+'\n';path=root/'report.md'
    if path.exists():
        if path.read_text()!=text:raise ValueError('Markdown report changed')
    else:path.write_text(text)


def run(args):
    cfg,plan,root,source_features,branches=prepare(args)
    if args.check:
        print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root),'images':{s:len(r) for s,r in plan['samples'].items()}}));return
    seal(root/'resolved_configuration.json',plan)
    for name,digest in plan['code_sha256'].items():
        data=(REPO/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Source changed during snapshot')
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=digest:raise ValueError('Source snapshot changed')
        if not dest.exists():dest.write_bytes(data)
    random.seed(cfg['execution']['seed']);np.random.seed(cfg['execution']['seed']);torch.manual_seed(cfg['execution']['seed'])
    torch.set_num_threads(cfg['execution']['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device':'cpu','precision':'float32',
        'optimizer_steps':0,'gpu_inventory':subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free',
            '--format=csv,noheader'],check=True,text=True,capture_output=True).stdout,
        'co_resident_gpu_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'],check=True,text=True,capture_output=True).stdout}
    seal(root/'execution_sessions'/(str(time.time_ns())+'.json'),receipt)
    try:
        views=views_for_split(cfg,plan,root,'train',source_features['train'])
        train=features_for_split(cfg,plan,root,'train',source_features['train'],views)
        frozen=freeze_thresholds(cfg,plan,root,train,branches['train'])
        print('[FROZEN] '+json.dumps(frozen['selected_families']),flush=True)
        views=views_for_split(cfg,plan,root,'test',source_features['test'])
        test=features_for_split(cfg,plan,root,'test',source_features['test'],views)
        result={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
            'result_class':plan['result_class'],'metric':plan['metric'],'new_network_training':False,
            'selected_families':frozen['selected_families'],'formal_risk_guarantee':False,
            'selected_budget_families':frozen['selected_budget_families'],
            'thresholds_sha256':sha(root/'frozen_thresholds.json'),'splits':{}}
        for name,features,bb,ids in [('train',train,branches['train'],list(train)),
                ('selection',train,branches['train'],plan['selection_ids']),
                ('calibration',train,branches['train'],plan['threshold_calibration_ids']),
                ('test',test,branches['test'],list(test))]:
            result['splits'][name]=split_metrics(cfg,features,bb,ids,frozen)
        seal(root/'results.json',result);write_report(root,cfg,result)
        point=result['splits']['test']['families'][frozen['selected_families'][str(cfg['primary_good_image_fraction'])]]['quality_'+str(cfg['primary_good_image_fraction'])]
        print(json.dumps({'status':'SUCCESS','root':str(root),'primary_test_point':point}),flush=True)
    except Exception as error:
        failure={'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
            'out_of_memory':isinstance(error,torch.cuda.OutOfMemoryError),'partial_results_are_not_final':True}
        name=hashlib.sha256(json.dumps(failure,sort_keys=True).encode()).hexdigest()[:16]+'.json'
        seal(root/'failures'/name,failure);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,default=REPO/'configs/approximate/yolo_frozen_consistency_v1.json')
    parser.add_argument('--paths',type=Path,default=REPO/'configs/approximate/yolo_frozen_consistency_local_paths_v1.json')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--check',action='store_true');parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
