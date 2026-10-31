"""Approximate offline quality-target study: frozen OCR/DeepSolo, no training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from recognition.f1_quality_study import (NEW_FAMILIES,proxy_scores,threshold_curve,
    choose_cut,evaluate,aggregate,good,sorted_ids)

REPO=Path(__file__).resolve().parents[1]
CODE=('scripts/run_yolo_f1_quality_study.py','recognition/f1_quality_study.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def payload(v):
    return hashlib.sha256(json.dumps({k:x for k,x in v.items()if k!='payload_sha256'},
        sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def seal(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=v:raise ValueError('Immutable artifact changed: '+str(path))
    else:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def checked(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen input changed: '+str(path))
    return read(path)


def checked_feature(path,ph):
    v=read(path)
    if v['plan_sha256']!=ph or v['payload_sha256']!=payload(v):raise ValueError('Feature seal mismatch: '+str(path))
    return v


def prepare(args):
    if args.mode!='approximate':raise ValueError('Only approximate reproduction is supported')
    for path in (args.config,args.paths):
        if path.resolve().parent!=REPO/'configs/approximate':raise ValueError('Isolated approximate namespace required')
    cfg=read(args.config);paths=read(args.paths)
    if cfg['reproduction_mode']!='approximate'or paths['reproduction_mode']!='approximate'or cfg['table_5_claim']is not False:
        raise ValueError('Approximate metadata required')
    for k in ('quality_f1_thresholds','target_good_fractions','bypass_fractions'):
        if not cfg[k]or any(not 0<v<=1 for v in cfg[k]):raise ValueError('Invalid explicit fraction grid')
    if not 0<cfg['minimum_bypass_fraction']<=1 or cfg['minimum_calibration_accepted']<1:
        raise ValueError('Invalid bypass/calibration constraints')
    if set(cfg['objectives'])!={'max_f1','max_bypass'}:raise ValueError('Unknown scalar objective')
    roots={k:Path(v).resolve()for k,v in paths['sources'].items()};plans={};results={};inputs={}
    output=(args.output_root or Path(paths['output_root'])).resolve()
    if output.is_relative_to(REPO)or any(output.is_relative_to(p)or p.is_relative_to(output)for p in roots.values()):
        raise ValueError('Keep outputs external and separate from frozen sources')
    for name,root in roots.items():
        pins=cfg['sources'][name]
        plans[name]=checked(root/'resolved_configuration.json',pins['plan'])
        results[name]=checked(root/'results.json',pins['results'])
        audit=checked(root/pins['audit_file'],pins['audit'])
        if results[name]['status']!='SUCCESS'or audit['status']!='PASS':raise ValueError('Unaudited/incomplete frozen source')
        if audit['results_sha256']!=pins['results']:raise ValueError('Audit names a different source result')
        # Frozen snapshots, rather than mutable working-tree comments, qualify historical code.
        for file,d in plans[name]['code_sha256'].items():
            if sha(root/'code_snapshot'/file)!=d:raise ValueError('Historical code snapshot differs')
    cons=plans['consistency'];selection=cons['selection_ids'];calibration=cons['threshold_calibration_ids']
    samples=cons['samples']
    if args.smoke:
        n=cfg['smoke_size_per_partition'];selection=selection[:n];calibration=calibration[:n]
        samples={'train':[s for s in samples['train']if s['sample_id']in selection+calibration],
                 'test':samples['test'][:n]}
    if set(selection)&set(calibration)or set(selection+calibration)!={s['sample_id']for s in samples['train']}:
        raise ValueError('Selection/calibration must partition TRAIN without overlap')
    names=[]
    for name in ('consistency','context','vqa'):
        for family in plans[name]['config']['families']:
            if family not in names:names.append(family)
    if set(names)&set(NEW_FAMILIES):raise ValueError('New family collides with old definition')
    names.extend(NEW_FAMILIES);data={};branches={}
    for split,rows in samples.items():
        data[split]={};branches[split]={}
        for sample in rows:
            sid=sample['sample_id'];key=sid.replace(':','__')+'.json';records={}
            for name,root in roots.items():
                path=root/('640/'if name=='coverage'else'')/split/'features'/key
                records[name]=checked_feature(path,cfg['sources'][name]['plan']);inputs[str(path)]=sha(path)
                v=records[name]
                if v['sample_id']!=sid or v['image_sha256']!=sample['image_sha256']:raise ValueError('Feature/image mismatch')
                if v['ground_truth_used']is not False:raise ValueError('Ground truth in inference feature')
            data[split][sid]=records
            branches[split][sid]=results['coverage']['resolutions']['640']['branch_metrics'][split][sid]
            b=branches[split][sid]
            if b['light']['gt_words']!=b['deepsolo']['gt_words']:raise ValueError('Branch GT populations differ')
    if set(data['train'])&set(data['test']):raise ValueError('TRAIN/TEST ID overlap')
    plan={'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction','table_5_claim':False,
        'result_class':'SMOKE_TEST'if args.smoke else'APPROXIMATE_EXPLORATORY_COMPARISON',
        'config':cfg,'paths':paths,'config_sha256':sha(args.config),'paths_sha256':sha(args.paths),
        'samples':samples,'selection_ids':selection,'threshold_calibration_ids':calibration,'families':names,
        'input_sha256':inputs,'clip_reference':cons['source_selector']['confidence_reference']['word_advantage'],
        'metric':plans['coverage']['metric']if 'metric'in plans['coverage']else results['coverage']['metric'],
        'code_sha256':{f:sha(REPO/f)for f in CODE},'source_deepsolo':'frozen ViTAEv2-S+TextOCR complete425 branches',
        'software':{k:importlib.metadata.version(k)for k in ('numpy','scipy','matplotlib')},'python':sys.version,
        'network_training':False,'new_model_inference':False,'optimizer_steps':0,'test_labels_used_for_selection':False,
        'test_history':'Historically explored TEST and repeatedly reused TRAIN; exploratory, not untouched validation'}
    tag=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg,plan,output/('smoke'if args.smoke else'full')/tag,data,branches


def feature_split(cfg,plan,root,split,data):
    out={};ph=sha(root/'resolved_configuration.json')
    for sid,parts in data.items():
        scores={}
        for name in ('consistency','context','vqa'):
            for k,v in parts[name]['scores'].items():
                if k in scores and scores[k]!=v:raise ValueError('Shared frozen score differs: '+k)
                scores[k]=v
        scores.update(proxy_scores(parts['coverage']['rows'],parts['consistency'],parts['context'],
            plan['clip_reference'],cfg['new_feature_settings']))
        if set(scores)!=set(plan['families']):raise ValueError('Score-family population mismatch')
        out[sid]={'sample_id':sid,'image_sha256':parts['coverage']['image_sha256'],
            'scores':{k:scores[k]for k in plan['families']},'ground_truth_used':False,
            'network_training':False,'original_words_changed':False,'plan_sha256':ph}
        out[sid]['payload_sha256']=payload(out[sid]);seal(root/split/'features'/(sid.replace(':','__')+'.json'),out[sid])
    return out


def minimum(cfg,n,smoke):
    return max(math.ceil(n*cfg['minimum_bypass_fraction']),
               cfg['smoke_minimum_accepted']if smoke else cfg['minimum_calibration_accepted'])


def freeze(cfg,plan,root,train,branches):
    selection=plan['selection_ids'];calibration=plan['threshold_calibration_ids'];qs=cfg['quality_f1_thresholds']
    curves={part:{}for part in ('selection','calibration')};smoke=plan['result_class']=='SMOKE_TEST'
    for name,ids in [('selection',selection),('calibration',calibration)]:
        for family in plan['families']:
            curves[name][family]=threshold_curve({s:train[s]['scores'][family]for s in ids},branches,ids,qs)
    frozen={'policies':{},'budget_selected_families':{},'selection_ids':selection,'threshold_calibration_ids':calibration,
        'test_labels_used':False,'test_features_present_at_freeze':0,'minimum_accepted':{
            name:minimum(cfg,len(ids),smoke)for name,ids in [('selection',selection),('calibration',calibration)]}}
    for q in qs:
        for purity in cfg['target_good_fractions']:
            for objective in cfg['objectives']:
                key=f'{q}:{purity}:{objective}';fits={};trials=[]
                for i,family in enumerate(plan['families']):
                    kw=dict(q=q,purity=purity,objective=objective)
                    trial=choose_cut(curves['selection'][family],minimum_accepted=frozen['minimum_accepted']['selection'],**kw)
                    fits[family]=choose_cut(curves['calibration'][family],minimum_accepted=frozen['minimum_accepted']['calibration'],**kw)
                    trial['family']=family;trials.append(trial)
                def winner_key(t):
                    c=t['calibration']
                    if c is None:return (-1,-1,-1,-plan['families'].index(t['family']))
                    g=c['good_images'][str(q)]/c['accepted_images']
                    return ((c['accepted_images'],g,c['combined_f1'],-plan['families'].index(t['family']))if objective=='max_bypass'else
                        (c['combined_f1'],c['accepted_images'],g,-plan['families'].index(t['family'])))
                winner=max(trials,key=winner_key)
                frozen['policies'][key]={'quality':q,'purity':purity,'objective':objective,'families':fits,
                    'selected_family':winner['family'],'selection_trials':trials,
                    'selection_qualified':winner['calibration']is not None}
        frozen['budget_selected_families'][str(q)]={}
        for fraction in cfg['bypass_fractions']:
            n=round(len(selection)*fraction);trials=[]
            for i,family in enumerate(plan['families']):
                values={s:train[s]['scores'][family]for s in selection};accepted=sorted_ids(values,selection)[:n]
                if len(accepted)!=n:continue
                point=evaluate(branches,selection,accepted,q)
                trials.append((point['good_images'],point['combined']['f1'],-i,family))
            frozen['budget_selected_families'][str(q)][str(fraction)]=max(trials)[3]
    path=root/'frozen_thresholds.json'
    if not path.exists()and list((root/'test/features').glob('*.json')):raise ValueError('Refuse fresh calibration after TEST features')
    frozen['plan_sha256']=sha(root/'resolved_configuration.json');seal(path,frozen)
    seal(root/'train_threshold_curves.json',curves)
    print('[frozen] all TRAIN family/cut choices before new TEST derived features',flush=True)
    return frozen


def split_evaluate(cfg,plan,features,branches,ids,frozen):
    out={'images':len(ids),'light_only':aggregate(branches[s]['light']for s in ids),
        'deepsolo_only':aggregate(branches[s]['deepsolo']for s in ids),'quality_oracles':{},'policies':{},'budgets':{}}
    for q in cfg['quality_f1_thresholds']:
        accepted={s for s in ids if good(branches[s]['light'],q)}
        out['quality_oracles'][str(q)]={'uses_ground_truth':True,**evaluate(branches,ids,accepted,q)}
    for key,p in frozen['policies'].items():
        families={}
        for family,fit in p['families'].items():
            cut=fit['threshold'];accepted={s for s in ids if cut is not None and features[s]['scores'][family]is not None
                and features[s]['scores'][family]>=cut}
            point=evaluate(branches,ids,accepted,p['quality'])
            families[family]={'threshold':cut,'calibration_status':fit['status'],
                'meets_minimum_bypass':len(accepted)>=math.ceil(len(ids)*cfg['minimum_bypass_fraction']),**point}
        # A cutoff on TRAIN45 cannot rescue a failed family selection on TRAIN80.
        selected=families[p['selected_family']]if p['selection_qualified']else{
            'threshold':None,'calibration_status':'ABSTAIN_ALL_NO_QUALIFYING_SELECTION_FAMILY',
            'meets_minimum_bypass':False,**evaluate(branches,ids,set(),p['quality'])}
        out['policies'][key]={'selected_family':p['selected_family'],
            'selection_qualified':p['selection_qualified'],'selected':selected,'families':families}
    for q in cfg['quality_f1_thresholds']:
        out['budgets'][str(q)]={}
        for b in cfg['bypass_fractions']:
            family=frozen['budget_selected_families'][str(q)][str(b)];values={s:features[s]['scores'][family]for s in ids}
            n=round(len(ids)*b);accepted=sorted_ids(values,ids)[:n]
            out['budgets'][str(q)][str(b)]={'selected_family':family,'requested_images':n,
                'unfilled_budget':n-len(accepted),**evaluate(branches,ids,accepted,q)}
    return out


def bootstrap_selected(cfg,plan,result,branches):
    ids=sorted(branches);rng=np.random.default_rng(cfg['seed']);iters=cfg['bootstrap_iterations']
    weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=iters).astype(float)
    deep=np.array([[branches[s]['deepsolo'][k]for k in ('tp','fp','fn')]for s in ids]);d=weights@deep
    deepf=2*d[:,0]/(2*d[:,0]+d[:,1]+d[:,2]);out={}
    for q in cfg['quality_f1_thresholds']:
        key=f"{q}:{cfg['primary_purity']}:{cfg['primary_objective']}";point=result['splits']['test']['policies'][key]['selected']
        accepted=set(point['accepted_ids']);rows=np.array([[branches[s]['light'if s in accepted else'deepsolo'][k]
            for k in ('tp','fp','fn')]for s in ids]);c=weights@rows;f=2*c[:,0]/(2*c[:,0]+c[:,1]+c[:,2])
        out[str(q)]={'delta_vs_deepsolo':point['combined']['f1']-result['splits']['test']['deepsolo_only']['f1'],
            'delta_vs_deepsolo_ci95':np.quantile(f-deepf,[.025,.975]).tolist(),
            'conditional_f1_ci95':np.quantile(f,[.025,.975]).tolist(),
            'iterations':iters,'gates_fixed':True,'multiplicity_correction':False}
    return out


def export(root,cfg,result):
    rows=[]
    for split,data in result['splits'].items():
        for key,p in data['policies'].items():
            q,purity,objective=key.split(':')
            for family,v in p['families'].items():
                rows.append({'split':split,'quality_f1':q,'target_good_fraction':purity,'objective':objective,
                    'family':family,'train_selected':family==p['selected_family'],'score_cut':v['threshold'],
                    'accepted':v['accepted_images'],'fallback':v['fallback_images'],'good':v['good_images'],
                    'good_fraction':v['good_fraction'],'accepted_word_f1':v['accepted_words']['f1'],
                    'rejected_deepsolo_f1':v['deepsolo_rejected_words']['f1'],
                    'predicted_words':v['combined']['detected_words'],'gt_words':v['combined']['gt_words'],
                    'tp':v['combined']['tp'],'fp':v['combined']['fp'],'fn':v['combined']['fn'],
                    'final_f1':v['combined']['f1'],'tn0':v['combined']['accuracy_tn0']})
    with(root/'policies.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Per-image F1 quality threshold study','','Approximate reproduction; exploratory lexical metric, not official paper equivalence.',
        'No network training or new model inference. All DeepSolo scores are actual subset counts.','',
        'Primary policy: TRAIN-only maximum combined F1 subject to empirical 90% good images and 20% bypass.',
        'No qualifying cut abstains on all images; zero-bypass is not a successful routing policy.','',
        '| Quality target | Family selected on TRAIN80 | TRAIN accepted | TRAIN purity | TEST accepted | TEST purity | Accepted wordF1 | Final TEST F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:f'{100*x:.2f}%'if x is not None else'undefined'
    for q in cfg['quality_f1_thresholds']:
        key=f"{q}:{cfg['primary_purity']}:{cfg['primary_objective']}";t=result['splits']['test']['policies'][key]
        a=result['splits']['train']['policies'][key]['selected'];b=t['selected']
        name=t['selected_family']if t['selection_qualified']else'No qualifying TRAIN80 family'
        lines.append(f"|{fmt(q)}|{name}|{a['accepted_images']}|{fmt(a['good_fraction'])}|{b['accepted_images']}|{fmt(b['good_fraction'])}|{fmt(b['accepted_words']['f1'])}|{fmt(b['combined']['f1'])}|")
    lines+=['','GT-only quality threshold oracle (not an inference method):','',
        '|Quality target|TEST accepted|Accepted wordF1|Rejected DeepSolo F1|Final TEST F1|','|---|---:|---:|---:|---:|']
    for q,v in result['splits']['test']['quality_oracles'].items():
        lines.append(f"|{fmt(float(q))}|{v['accepted_images']}|{fmt(v['accepted_words']['f1'])}|{fmt(v['deepsolo_rejected_words']['f1'])}|{fmt(v['combined']['f1'])}|")
    lines+=['','All families, both scalar objectives,80/90/95% purity targets, split counts and actual word totals are in policies.csv/results.json.',
        'Only TRAIN-selected policies support transferred comparisons; TEST maxima are exploratory. Historically explored TEST is not untouched.',
        'Paper nominal78.4/79.8 and standalone80.3571 local DeepSolo have distinct provenance; numerical crossing is not a reproduced benchmark win.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    start=time.monotonic();cfg,plan,root,data,branches=prepare(args)
    if args.preflight:print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root),'families':len(plan['families'])}));return
    seal(root/'resolved_configuration.json',plan)
    for f,d in plan['code_sha256'].items():
        target=root/'code_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists()and sha(target)!=d:raise ValueError('Code snapshot changed')
        if not target.exists():target.write_bytes((REPO/f).read_bytes())
    train=feature_split(cfg,plan,root,'train',data['train']);frozen=freeze(cfg,plan,root,train,branches['train'])
    test=feature_split(cfg,plan,root,'test',data['test'])
    splits=[('train',train,branches['train'],list(train)),('selection',train,branches['train'],plan['selection_ids']),
        ('calibration',train,branches['train'],plan['threshold_calibration_ids']),('test',test,branches['test'],list(test))]
    result={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction','table_5_claim':False,
        'plan_sha256':sha(root/'resolved_configuration.json'),'thresholds_sha256':sha(root/'frozen_thresholds.json'),
        'new_model_inference':False,'network_training':False,'optimizer_steps':0,'result_class':plan['result_class'],
        'metric':plan['metric'],'splits':{name:split_evaluate(cfg,plan,features,b,ids,frozen)for name,features,b,ids in splits}}
    result['primary_paired_intervals']=bootstrap_selected(cfg,plan,result,branches['test'])
    seal(root/'results.json',result);export(root,cfg,result)
    if not(root/'runtime.json').exists():seal(root/'runtime.json',{'seconds':time.monotonic()-start,
        'execution':'CPU offline cached-output study','new_gpu_inference':False,'new_network_training':False,
        'peak_allocated_vram_bytes':None,'peak_reserved_vram_bytes':None})
    print(json.dumps({'status':'SUCCESS','root':str(root),'results_sha256':sha(root/'results.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--paths',required=True,type=Path)
    p.add_argument('--output-root',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--preflight',action='store_true')
    run(p.parse_args())
