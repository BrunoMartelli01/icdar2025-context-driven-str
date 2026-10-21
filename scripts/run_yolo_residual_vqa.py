"""APPROXIMATE frozen BLIP2 missing-text probes with TRAIN-only image routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch

from recognition.residual_text_vqa import FrozenResidualVQA,residual_views
from recognition.clip_crop_admission import percentile
from recognition.clip_router_prompt_ablation import budget_accept,coverage_threshold
from recognition.training_free_consistency_router import calibrate,evaluate
from recognition.yolo_router_comparison import aggregate
from scripts import run_yolo_context_quality as base
from scripts.crop_clip_runtime import sha,_cuda_peak,write_json
from scripts.run_clip_high_recall_admission import seal

REPO=Path(__file__).resolve().parents[1]
FIELDS=('vqa_residual_plain','vqa_residual_context','vqa_visual_drop_plain','vqa_visual_drop_context','vqa_context_specific')


def prepare(args):
    for f in (args.config,args.paths):
        if f.resolve().parent!=REPO/'configs/approximate':raise ValueError('Isolated approximate configs required')
    cfg=base.source.read(args.config);paths=base.source.read(args.paths)
    if (args.mode!='approximate' or cfg['reproduction_mode']!='approximate' or paths['reproduction_mode']!='approximate'
            or cfg['table_5_claim']is not False or cfg['execution']['precision']!='float32'
            or cfg['families']!=['coverage_baseline','all_checks_softcrop',*FIELDS,*('base_'+f for f in FIELDS)]):
        raise ValueError('Unsupported frozen residual VQA protocol')
    context=Path(paths['context_run']);old=base.source.checked(context/'resolved_configuration.json',cfg['context_plan_sha256'])
    result=base.source.checked(context/'results.json',cfg['context_results_sha256'])
    audit=base.source.read(context/'independent_audit.json')
    if result['status']!='SUCCESS' or audit['status']!='PASS' or audit['results_sha256']!=cfg['context_results_sha256']:
        raise ValueError('Prior425 context comparison must be complete and audited')
    old_args=argparse.Namespace(config=Path(paths['base_config']),paths=Path(paths['base_paths']),
        output_root=None,mode='approximate',smoke=args.smoke)
    _,input_plan,_,observed,branches=base.prepare(old_args)
    first=next(iter(observed['train'].values()))
    cap=base.source.read(Path(first['caption']['path']));asset=cap['contract']['asset'];snapshot=Path(asset['snapshot'])
    for name,d in {**asset['configuration_file_sha256'],**asset['weight_sha256']}.items():
        if sha(snapshot/name)!=d:raise ValueError('BLIP2 artifact differs from caption provenance: '+name)
    output=(args.output_root or Path(paths['output_root'])).resolve()
    protected=[REPO,context,*(Path(old['paths'][k]) for k in ('paired_run','coverage_run','consistency_run'))]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):raise ValueError('Output must be external and separate')
    code=tuple(dict.fromkeys(('scripts/run_yolo_residual_vqa.py','recognition/residual_text_vqa.py',*base.CODE)))
    plan={'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction','table_5_claim':False,
        'result_class':'SMOKE_TEST'if args.smoke else'APPROXIMATE_EXPLORATORY_COMPARISON',
        'config':cfg,'paths':paths,'config_sha256':sha(args.config),'paths_sha256':sha(args.paths),
        'input_plan':input_plan,'samples':input_plan['samples'],'selection_ids':input_plan['selection_ids'],
        'threshold_calibration_ids':input_plan['threshold_calibration_ids'],'metric':input_plan['metric'],
        'blip2_identity':asset,'code_sha256':{p:sha(REPO/p)for p in code},'gpu':torch.cuda.get_device_name(0),
        'source_context_audit_sha256':sha(context/'independent_audit.json'),
        'new_network_training':False,'test_labels_used_for_selection':False,'formal_risk_guarantee':False,
        'test_history':'Historically exploredTEST; this follow-up chosen after inspecting earlier context methods; exploratory only'}
    tag=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg,plan,output/('smoke'if args.smoke else'full')/tag,observed,branches


def raw_signals(vqa):
    risks={k:max(vqa[m][k]['yes_probability']for m in ('hard_mask','blur_mask'))for k in ('plain','context')}
    drop={k:vqa['original'][k]['yes_probability']-risks[k]for k in risks}
    return {'vqa_residual_plain':-risks['plain'],'vqa_residual_context':-risks['context'],
        'vqa_visual_drop_plain':drop['plain'],'vqa_visual_drop_context':drop['context'],
        'vqa_context_specific':drop['context']-drop['plain']}


def probe_split(cfg,plan,root,split,observed):
    ph=sha(root/'resolved_configuration.json');target=root/split/'vqa_receipt.json'
    if target.exists():
        receipt=base.receipt(target,ph)
        return {s:base.receipt(root/split/'probes'/(base.source.key(s)+'.json'),ph)for s in observed}
    judge=FrozenResidualVQA(plan['blip2_identity']['snapshot'],device=cfg['execution']['device'],
        plain_question=cfg['plain_question'],context_template=cfg['context_template'])
    records={};started=time.monotonic()
    try:
        for number,(sid,v)in enumerate(observed.items(),1):
            path=root/split/'probes'/(base.source.key(sid)+'.json')
            cached=base.receipt(path,ph)
            if cached is None:
                s=v['sample']
                if sha(Path(s['image_path']))!=s['image_sha256']:raise ValueError('Original image changed')
                with Image.open(s['image_path'])as im:rgb=np.asarray(im.convert('RGB'))
                if list(rgb.shape[:2][::-1])!=s['image_wh']:raise ValueError('Original RGB shape/orientation changed')
                views,geometry=residual_views(rgb,[r['bbox']for r in v['rows']],mask_value=cfg['mask_value'],blur_fraction=cfg['blur_fraction'])
                probabilities=judge.score(views,v['caption']['text'])
                cached=base.stamp({'sample_id':sid,'image_sha256':s['image_sha256'],'caption':v['caption'],
                    'transcript':v['transcript'],'source_scores':v['source_scores'],'vqa':probabilities,'geometry':geometry,
                    'signals':raw_signals(probabilities),'eligible':bool(v['rows']),
                    'ground_truth_used':False,'models_trained':False,'original_word_outputs_changed':False},ph)
                seal(path,cached)
            records[sid]=cached
            if number%10==0 or number==len(observed):print(f'[BLIP2 {split}] {number}/{len(observed)}',flush=True)
        provenance=judge.finish()
        seal(target,base.stamp({'model':provenance,'seconds':time.monotonic()-started,'images':len(observed),
            'image_processed_shape':[3,224,224],'optimizer_steps':0,'ground_truth_used':False},ph))
    finally:judge.close()
    return records


def score_features(cfg,plan,root,split,raw,reference=None):
    ph=sha(root/'resolved_configuration.json')
    if reference is None:
        reference={'reference_ids':plan['selection_ids'],'labels_used':False,
            'fields':{k:sorted(raw[s]['signals'][k]for s in plan['selection_ids'])for k in FIELDS}}
        seal(root/'frozen_references.json',base.stamp(reference,ph))
    features={}
    for sid,r in raw.items():
        if not r['eligible']:
            scores=dict.fromkeys(cfg['families'])
        else:
            scaled={k:percentile(reference['fields'][k],r['signals'][k])for k in FIELDS}
            baseline=r['source_scores']['all_checks_softcrop']
            scores={'coverage_baseline':r['source_scores']['coverage_baseline'],'all_checks_softcrop':baseline,
                **scaled,**{'base_'+k:(baseline*v)**.5 for k,v in scaled.items()}}
        features[sid]=base.stamp({**r,'scores':scores},ph)
        seal(root/split/'features'/(base.source.key(sid)+'.json'),features[sid])
    return features,reference


def freeze(cfg,plan,root,features,branches):
    selection=plan['selection_ids'];calibration=plan['threshold_calibration_ids'];out={'families':{},'selected_families':{},'selected_budget_families':{}}
    trials={str(t):[]for t in cfg['target_good_image_fractions']}
    for family in cfg['families']:
        values={s:f['scores'][family]for s,f in features.items()}
        out['families'][family]={'targets':{},'coverage_cuts':{str(b):coverage_threshold(values,calibration,b)for b in cfg['bypass_fractions']}}
        for target in cfg['target_good_image_fractions']:
            kw={'target':target,'quality_threshold':.8,'minimum_accepted':cfg['minimum_calibration_accepted']}
            trials[str(target)].append({'family':family,**calibrate({s:values[s]for s in selection},{s:branches[s]for s in selection},selection,**kw)})
            out['families'][family]['targets'][str(target)]=calibrate({s:values[s]for s in calibration},{s:branches[s]for s in calibration},calibration,**kw)
    for target,rows in trials.items():
        out['selected_families'][target]=min(rows,key=lambda t:(-(t['calibration']['accepted_images']if t['calibration']else 0),
            -(t['calibration']['good_fraction']if t['calibration']else 0),cfg['families'].index(t['family'])))['family']
    budget_trials={}
    for b in cfg['bypass_fractions']:
        rows=[]
        for family in cfg['families']:
            values={s:features[s]['scores'][family]for s in selection};accepted=budget_accept(values,selection,round(len(selection)*b))
            rows.append({'family':family,'accepted_good_images':evaluate(branches,selection,accepted,.8)['accepted_good_images']})
        out['selected_budget_families'][str(b)]=min(rows,key=lambda t:(-t['accepted_good_images'],cfg['families'].index(t['family'])))['family']
        budget_trials[str(b)]=rows
    path=root/'frozen_thresholds.json'
    if not path.exists()and list((root/'test').rglob('*.json')):raise ValueError('Cannot calibrate after newTEST VQA')
    out.update(selection_ids=selection,threshold_calibration_ids=calibration,selection_trials=trials,budget_selection_trials=budget_trials,
        test_features_present_at_freeze=0,test_labels_used=False,models_trained=False,formal_risk_guarantee=False)
    seal(path,base.stamp(out,sha(root/'resolved_configuration.json')))
    print('[frozen] TRAIN-only VQA references,thresholds,family choices before TEST probes',flush=True)
    return out


def metrics(cfg,features,branches,ids,frozen):
    out={'images':len(ids),'families':{},'light_only':aggregate(branches[s]['light']for s in ids),'deepsolo_only':aggregate(branches[s]['deepsolo']for s in ids)}
    for family in cfg['families']:
        values={s:features[s]['scores'][family]for s in ids};points={}
        for t in cfg['target_good_image_fractions']:
            cut=frozen['families'][family]['targets'][str(t)]['threshold'];accepted={s for s in ids if cut is not None and values[s]is not None and values[s]>=cut}
            points['quality_'+str(t)]={'threshold':cut,**evaluate(branches,ids,accepted,.8)}
        for b in cfg['bypass_fractions']:
            n=round(len(ids)*b);accepted=budget_accept(values,ids,n)
            points['budget_'+str(b)]={'requested_images':n,'unfilled_budget':n-len(accepted),'threshold':None,**evaluate(branches,ids,accepted,.8)}
            cut=frozen['families'][family]['coverage_cuts'][str(b)];accepted={s for s in ids if cut is not None and values[s]is not None and values[s]>=cut}
            points['coverage_cut_'+str(b)]={'threshold':cut,**evaluate(branches,ids,accepted,.8)}
        out['families'][family]=points
    return out


def run(args):
    cfg,plan,root,observed,branches=prepare(args)
    if args.preflight:print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root)}));return
    seal(root/'resolved_configuration.json',plan)
    for name,d in plan['code_sha256'].items():
        target=root/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists()and sha(target)!=d:raise ValueError('Code snapshot differs')
        if not target.exists():target.write_bytes((REPO/name).read_bytes())
    torch.set_num_threads(cfg['execution']['cpu_threads']);seed=cfg['execution']['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    started=time.monotonic()
    try:
        raw=probe_split(cfg,plan,root,'train',observed['train']);train,reference=score_features(cfg,plan,root,'train',raw)
        frozen=freeze(cfg,plan,root,train,branches['train'])
        raw=probe_split(cfg,plan,root,'test',observed['test']);test,_=score_features(cfg,plan,root,'test',raw,reference)
        result={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction','table_5_claim':False,
            'result_class':plan['result_class'],'plan_sha256':sha(root/'resolved_configuration.json'),'metric':plan['metric'],
            'new_network_training':False,'optimizer_steps':0,'formal_risk_guarantee':False,
            'thresholds_sha256':sha(root/'frozen_thresholds.json'),'references_sha256':sha(root/'frozen_references.json'),
            'primary_family':cfg['primary_family'],'selected_families':frozen['selected_families'],'selected_budget_families':frozen['selected_budget_families'],
            'splits':{name:metrics(cfg,features,bb,ids,frozen)for name,features,bb,ids in[
                ('train',train,branches['train'],list(train)),('selection',train,branches['train'],plan['selection_ids']),
                ('calibration',train,branches['train'],plan['threshold_calibration_ids']),('test',test,branches['test'],list(test))]}}
        seal(root/'results.json',result)
        if not(root/'runtime.json').exists():seal(root/'runtime.json',{'status':'SUCCESS','seconds':time.monotonic()-started,'optimizer_steps':0})
    except Exception as exc:
        write_json(root/'failure.json',{'status':'FAILED','type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'out_of_memory':isinstance(exc,torch.cuda.OutOfMemoryError),'cuda_peak':_cuda_peak(cfg['execution']['device']),
            'partial_results_are_not_final':True,'optimizer_steps':0});raise
    print(json.dumps({'status':'SUCCESS','root':str(root),'results_sha256':sha(root/'results.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--output-root',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
