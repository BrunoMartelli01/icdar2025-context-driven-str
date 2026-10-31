"""Train one shared APPROXIMATE correction/risk/count-utility procedure."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import (normalize,counts,micro,choose_light,route,
    grouped_folds,map_features,word_features,risk_features,image_features,
    fit_count_head,predict_counts,fit_word_risk,validation_roles)
from recognition.lexical_budget_router import correct_rows
from recognition.deepsolo_lexicon import DeepSoloLexicon
from scripts.icdar_budget_data import read,sha,key,predictions,native_mask
from scripts.run_yolo_f1_quality_study import seal

REPO=Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode!='approximate' or args.config.resolve().parent!=REPO/'configs/approximate':
        raise ValueError('Explicit APPROXIMATE configuration required')
    c=read(args.config)
    if c['mode']!=args.mode:raise ValueError('Mode mismatch')
    for p,h in c['pinned_inputs'].items():
        if sha(p)!=h:raise ValueError('Changed input: '+p)
    out=Path(c['output_root'])/args.run_name;out.mkdir(parents=True,exist_ok=False)
    source={str(p.relative_to(REPO)):sha(p) for folder in c['snapshot_packages'] for p in (REPO/folder).rglob('*.py')}
    plan=dict(mode='approximate',claim_label='approximate reproduction',config=c,config_sha256=sha(args.config),
        smoke=args.smoke,result_class='smoke-test result' if args.smoke else 'approximate reproduction',
        source_sha256=source,historical_test_exposure=True,test_used_for_selection=False,
        runtime=dict(host=platform.node(),python=platform.python_version(),numpy=np.__version__,
            device='cpu',feature_precision='float64',actual_new_gpu_inference=0,actual_new_deepsolo_inference=0))
    seal(out/'resolved_configuration.json',plan)
    for p,h in source.items():
        dest=out/'code_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/p,dest)
        if sha(dest)!=h:raise ValueError('Source changed during snapshot')
    shutil.copytree(REPO/'configs',out/'code_snapshot/configs')
    return out


def no_correction(rows):
    return [dict(r,source_row=i) for i,r in enumerate(rows)],[dict(source_row=i,original_text=r['raw_text'],active=False,
        accepted=True,match=None,potential_match=None,changed=False,confidence=r['confidence']) for i,r in enumerate(rows)]


def make_lexicon(profile,split):
    if profile is None:return None
    vocabulary=profile[split+'_vocabulary'];pairs=profile[split+'_pairs']
    if profile.get(split+'_pair_format')=='json':return DeepSoloLexicon(Path(vocabulary).read_text().splitlines(),read(pairs),distance_exclusive=profile['distance_exclusive'])
    return DeepSoloLexicon.from_files(vocabulary,pairs,distance_exclusive=profile['distance_exclusive'])


def correct(scene,lex,policy):
    rows,trace=no_correction(scene['rows']) if lex is None else correct_rows(scene['rows'],lex,policy)
    ct=counts(rows,scene['ground_truth'],[scene['ignored'][r['source_row']] for r in rows])
    return rows,trace,ct


def totaltext_inputs(c):
    paths=c['sources']['totaltext'];base=Path(paths['features']);p=read(base/'resolved_configuration.json');choice=read(paths['essential_choices'])
    parts=dict(fit=sorted(r['sample_id'] for role in ('fit','calibration') for r in p['records'][role]),
        selection=choice['selection_ids'],calibration=choice['calibration_ids'],test=[r['sample_id'] for r in p['records']['test']])
    records={r['sample_id']:r for role in p['records'].values() for r in role}
    trace=read(base/'train_feature_inputs.json');trace.update(read(base/'test_feature_inputs.json'))
    source=Path(p['source_config']['paths']['source_run']);native=Path(p['source_config']['paths']['native_fallback_run'])
    rows={};gt={};teacher={};expected={};source_pins={}
    for split in ('train','test'):
        path=source/split/'corrected.json';rows.update(read(path));source_pins[str(path)]=sha(path)
        path=native/split/'branches.json';d=read(path);source_pins[str(path)]=sha(path)
        gt.update({s:v['ground_truth'] for s,v in d['records'].items()})
        teacher.update({s:v['deepsolo'] for s,v in d['branches']['yolo'].items()})
    for role in ('fit','selection','calibration','test'):expected.update(read(base/(role+'_branches.json')))
    return parts,records,trace,rows,gt,teacher,expected,source_pins


def prepare_data(plan,root):
    c=plan['config'];out=root/'data';out.mkdir(exist_ok=False)
    tt=totaltext_inputs(c) if 'totaltext' in c['datasets'] else None
    cases={};pinned={};map_cache={}
    for dataset in c['datasets']:
        for arm in c['experts']:
            name=dataset+'__'+arm;scenes={};features={};test_labels={};inputs={}
            if dataset=='totaltext':
                parts,records,trace,source_rows,gt,teacher,expected,sp=tt;pinned.update(sp)
            else:
                source=c['sources'][dataset];ic=read(Path(source['original'])/'resolved_configuration.json')
                parts=ic['partitions'][dataset];records={r['sample_id']:r for group in ic['records'][dataset].values() for r in group}
                path=Path(source['prepared_files'][arm]);prepared=read(path);pinned[str(path)]=sha(path)
                teacher_scenes=None
                if source.get('teacher_scenes'):
                    path=Path(source['teacher_scenes']);teacher_scenes=read(path)['scenes'];pinned[str(path)]=sha(path)
            parts=validation_roles({k:sorted(v) for k,v in parts.items()},c['validation_policy'])
            if plan['smoke']:parts={k:v[:(40 if k=='fit' else 8)] for k,v in parts.items()}
            ids=[s for values in parts.values() for s in values]
            for index,sid in enumerate(ids):
                if dataset=='totaltext':
                    r=records[sid];original=r['token_sample'];split='test' if sid in parts['test'] else 'train'
                    if arm=='unet':raw=source_rows[sid]
                    else:
                        path=Path(c['sources']['totaltext']['yolo'])/split/'light'/(key(sid)+'.json');d=read(path);inputs[str(path)]=sha(path)
                        if d['ground_truth_used'] or d['contract']['image_sha256']!=r['image_sha256']:raise ValueError('Changed YOLO/RGB binding')
                        raw=d['rows']
                    if sha(original['validity_path'])!=original['validity_sha256']:raise ValueError('Changed TT validity')
                    valid=native_mask(Image.open(original['validity_path']),256);ignore=[p['ignored'] for p in predictions(raw,r,valid)]
                    scene=dict(rows=raw,ground_truth=gt[sid],ignored=ignore,image_wh=r['image_wh'],tie_key=original['rgb_sha256'],teacher=teacher[sid])
                    maps=trace[sid]['maps_path'];digest=trace[sid]['maps_sha256']
                    if arm=='unet' and counts(raw,gt[sid],ignore)!=expected[sid]['light']:raise ValueError('TT corrected count protocol mismatch: '+sid)
                else:
                    r=records[sid];scene=dict(prepared['scenes'][sid])
                    if teacher_scenes is not None:scene['teacher']=teacher_scenes[sid]['teacher']
                    if scene['teacher']['gt_words']!=len(scene['ground_truth']):raise ValueError('Teacher care population mismatch')
                    model=source['map_expert'];path=Path(source['original'])/'experts'/model/(key(sid)+'.json');d=read(path)
                    maps=d['maps_path'];digest=d['maps_sha256'];inputs[str(path)]=sha(path)
                if maps not in map_cache:
                    if sha(maps)!=digest:raise ValueError('Changed map input')
                    with np.load(maps,allow_pickle=False) as z:map_cache[maps]=z['probabilities']
                # Only observed prediction maps enter this feature.
                maps_vector=map_features(map_cache[maps],scene['image_wh'],scene['rows'],c['map_features'])
                if sid in parts['test']:
                    test_labels[sid]=dict(ground_truth=scene['ground_truth'],ignored=scene['ignored'],teacher=scene['teacher'])
                scenes[sid]=dict(scene,source_image_sha256=r['image_sha256'])
                features[sid]=dict(maps=maps_vector.tolist())
                inputs[maps]=digest
                if index and index%400==0:print(json.dumps(dict(stage='prepare',case=name,done=index,total=len(ids))),flush=True)
            keys={s:scenes[s]['tie_key'] for s in ids};rgb=[{keys[s] for s in part} for part in parts.values()]
            if any(a&b for i,a in enumerate(rgb) for b in rgb[i+1:]):raise ValueError('RGB overlap across scientific roles')
            if len(ids)!=len(set(ids)):raise ValueError('Duplicated scientific role')
            # Label files are separate: training later receives only non-TEST labels.
            training={s:v for s,v in scenes.items() if s not in parts['test']}
            observations={s:{k:v for k,v in scene.items() if k not in ('ground_truth','ignored','teacher')} for s,scene in scenes.items()}
            seal(out/(name+'__observations.json'),observations);seal(out/(name+'__train_labels.json'),training)
            seal(out/(name+'__test_labels.json'),test_labels);seal(out/(name+'__maps.json'),features)
            seal(out/(name+'__inputs.json'),inputs)
            cases[name]=dict(dataset=dataset,expert=arm,partitions=parts,tie_keys=keys,
                fallback_count=round(len(parts['test'])*c['budgets'][dataset]['fallback_images']/c['budgets'][dataset]['images']) if plan['smoke'] else c['budgets'][dataset]['fallback_images'])
    seal(root/'cases.json',cases);seal(root/'source_input_sha256.json',pinned)
    print(json.dumps(dict(stage='data_prepared',cases={k:{r:len(v) for r,v in x['partitions'].items()} for k,x in cases.items()})),flush=True)


def train_case(plan,root,name,case):
    c=plan['config'];out=root/'models'/name;out.mkdir(parents=True,exist_ok=False)
    observations=read(root/'data'/(name+'__observations.json'));labels=read(root/'data'/(name+'__train_labels.json'));maps=read(root/'data'/(name+'__maps.json'))
    p=case['partitions'];lex=make_lexicon(c['lexicons'].get(case['dataset']),'train')
    policies=[dict(enabled=False,confidence_cap=0.,protect_numeric=True,unmatched_policy='keep_unmatched')]
    if lex is not None:policies.extend(dict(enabled=True,confidence_cap=cap,protect_numeric=True,unmatched_policy=u) for cap in c['correction_confidence_caps'] for u in c['unmatched_policies'])
    grid=[]
    for policy in policies:
        values=[];changes=0
        for s in p['selection']:
            _,trace,ct=correct(labels[s],lex,policy);values.append(ct);changes+=sum(t['changed'] or not t['accepted'] for t in trace)
        grid.append(dict(policy=policy,selection=micro(values),changes=changes))
    correction=max(grid,key=lambda x:(x['selection']['f1'],-x['changes'],not x['policy']['enabled']))
    seal(out/'correction_frozen.json',dict(winner=correction,trials=grid,test_used=False))
    # Predictions for TEST use its allowed vocabulary but never its labels.
    test_lex=make_lexicon(c['lexicons'].get(case['dataset']),'test')
    groups={};word_x={};traces={};corrected_rows={};branches={}
    for s,obs in observations.items():
        locallex=test_lex if s in p['test'] else lex
        rows,trace=no_correction(obs['rows']) if locallex is None else correct_rows(obs['rows'],locallex,correction['policy'])
        corrected_rows[s]=rows;traces[s]=trace;word_x[s]=word_features(obs['rows'],trace,obs['image_wh'])
        groups[s]=image_features(obs['rows'],rows,trace,obs['image_wh'],np.array(maps[s]['maps']))
        if s in labels:
            ct=counts(rows,labels[s]['ground_truth'],[labels[s]['ignored'][r['source_row']] for r in rows]);branches[s]=dict(light=ct,deepsolo=labels[s]['teacher'])
    fitids=p['fit'];folds=grouped_folds(fitids,case['tie_keys'],c['folds'],c['seed']);word_targets={};care={}
    for s in fitids:
        truth={normalize(g['text']) for g in labels[s]['ground_truth'] if normalize(g['text'])}
        word_targets[s]=np.array([bool(normalize(r['raw_text'])) and normalize(r['raw_text']) in truth for r in observations[s]['rows']],int)
        care[s]=np.array([not x for x in labels[s]['ignored']],bool)
    def word_fit(ids,seed):
        xx=np.concatenate([word_x[s][care[s]] for s in ids]);yy=np.concatenate([word_targets[s][care[s]] for s in ids])
        return fit_word_risk(xx,yy,seed,c['word_risk'])
    risk={};audits=[]
    for i,held in enumerate(folds):
        ids=sorted(set(fitids)-set(held));model=word_fit(ids,c['seed']+i)
        joblib.dump(model,out/f'word_fold_{i}.joblib')
        for s in held:risk[s]=model.predict_proba(word_x[s])[:,1] if len(word_x[s]) else np.empty(0)
        audits.append(dict(fit_ids=ids,heldout_ids=held,model_sha256=sha(out/f'word_fold_{i}.joblib')))
    word_model=word_fit(fitids,c['seed']);joblib.dump(word_model,out/'word_full.joblib')
    for s in observations:
        if s not in risk:risk[s]=word_model.predict_proba(word_x[s])[:,1] if len(word_x[s]) else np.empty(0)
        groups[s]['risk']=risk_features(risk[s],traces[s])
    seal(out/'word_oof_audit.json',dict(folds=audits,label_scope='FIT normalized set-membership proxy; ignored words excluded from fitting',upstream_expert_oof=False))
    ids=list(observations);np.savez_compressed(out/'features.npz',ids=np.array(ids),**{g:np.stack([groups[s][g] for s in ids]) for g in groups[ids[0]]})
    seal(out/'observable_outputs.json',dict(corrected_rows=corrected_rows,traces=traces,word_risk={s:v.tolist() for s,v in risk.items()}))
    seal(out/'train_branches.json',branches)
    scale={s:len(observations[s]['rows'])+1 for s in observations};indices={s:i for i,s in enumerate(ids)}
    def matrix(samples,features):return np.stack([np.concatenate([groups[s][g] for g in features]) for s in samples])
    y=np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fitids])/np.array([scale[s] for s in fitids])[:,None]
    w=np.array([scale[s] for s in fitids]);reference=micro(branches[s]['deepsolo'] for s in fitids)['f1'];refs=sorted(set(c['reference_grid']+[reference]))
    b={s:branches[s] for s in p['selection']};n=round(len(b)*c['budgets'][case['dataset']]['fallback_images']/c['budgets'][case['dataset']]['images'])
    keys={s:case['tie_keys'][s] for s in b};trials=[];predictions_by_model={};modelmeta={}
    def evaluate_prediction(prediction,modelpaths,groups_used,scalars):
        best=None
        for ref in refs:
            scores={s:float(2*v[0]-ref*v[1])*scale[s] for s,v in zip(p['selection'],prediction)}
            for guard in c['guards']:
                eligible=[s for s in b if observations[s]['rows'] and all(r['raw_text'].strip() for r in observations[s]['rows'])] if guard else list(b)
                if len(eligible)<len(b)-n:continue
                accepted=choose_light(scores,keys,n,eligible);result=route(b,accepted)
                trial=dict(models=modelpaths,groups=groups_used,reference=ref,guard=guard,selection=result,scalars=scalars)
                trials.append(trial)
                if best is None or result['combined']['f1']>best['selection']['combined']['f1']:best=trial
        return best
    for gi,features in enumerate(c['feature_groups']):
        x=matrix(fitids,features);val=matrix(p['selection'],features)
        for ri,recipe in enumerate(c['head_recipes']):
            name_model=f'head_{gi}_{ri}.joblib';bundle=fit_count_head(x,y,w,recipe,c['seed']);bundle['groups']=features
            joblib.dump(bundle,out/name_model);pr=predict_counts(bundle,val);predictions_by_model[name_model]=pr
            scalars=sum(v.size for m in bundle['models'] for v in [getattr(m,'coef_',np.array([]))])+2*x.shape[1]+2
            # Stored size supplies a consistent complexity proxy for tree heads.
            if recipe['kind']!='ridge':scalars=(out/name_model).stat().st_size//8
            winner=evaluate_prediction(pr,[name_model],[features],scalars);modelmeta[name_model]=winner
    ordered=sorted(modelmeta,key=lambda s:(-modelmeta[s]['selection']['combined']['f1'],modelmeta[s]['scalars'],s))
    for k in c['ensemble_sizes']:
        paths=ordered[:k];pred=np.mean([predictions_by_model[s] for s in paths],axis=0)
        evaluate_prediction(pred,paths,[read_group for s in paths for read_group in modelmeta[s]['groups']],sum(modelmeta[s]['scalars'] for s in paths))
    maximum=max(t['selection']['combined']['f1'] for t in trials)
    winner=min((t for t in trials if t['selection']['combined']['f1']>=maximum-c['selection_tolerance']),key=lambda t:(t['scalars'],-t['selection']['combined']['f1'],len(t['models']),t['models'],t['reference'],t['guard']))
    seal(out/'trials.json',trials)
    seal(out/'selection_frozen.json',dict(winner=winner,fit_ids=fitids,selection_ids=p['selection'],calibration_ids=p['calibration'],test_used_for_selection=False,
        model_sha256={path:sha(out/path) for path in winner['models']},source_features_sha256=sha(out/'features.npz'),word_model_sha256=sha(out/'word_full.joblib'),reference_grid=refs))
    # The same fitted heads score every population. No refit follows selection.
    all_predictions=[]
    for path in winner['models']:
        bundle=joblib.load(out/path);all_predictions.append(predict_counts(bundle,matrix(ids,bundle['groups'])))
    pair=np.mean(all_predictions,axis=0);scores={s:float(2*v[0]-winner['reference']*v[1])*scale[s] for s,v in zip(ids,pair)}
    seal(out/'scores.json',dict(scores=scores,eligible={s:bool(observations[s]['rows']) and all(r['raw_text'].strip() for r in observations[s]['rows']) for s in ids},tie_keys=case['tie_keys'],ground_truth_used=False))
    print(json.dumps(dict(stage='case_trained',case=name,selection_f1=winner['selection']['combined']['f1'],models=winner['models'],reference=winner['reference'],guard=winner['guard'],trials=len(trials))),flush=True)


def evaluate(plan,root):
    cases=read(root/'cases.json');frozen=read(root/'training_complete.json');results={}
    for name,case in cases.items():
        out=root/'models'/name;selection=read(out/'selection_frozen.json')
        if frozen['frozen_sha256'][name]!=sha(out/'selection_frozen.json'):raise ValueError('Frozen scientific choice changed')
        values=read(out/'scores.json');test=read(root/'data'/(name+'__test_labels.json'));obs=read(out/'observable_outputs.json');branches={}
        for s in case['partitions']['test']:
            rows=obs['corrected_rows'][s];ct=counts(rows,test[s]['ground_truth'],[test[s]['ignored'][r['source_row']] for r in rows]);branches[s]=dict(light=ct,deepsolo=test[s]['teacher'])
        scores={s:values['scores'][s] for s in branches};keys={s:values['tie_keys'][s] for s in branches}
        eligible=[s for s in branches if values['eligible'][s]] if selection['winner']['guard'] else list(branches)
        accepted=choose_light(scores,keys,case['fallback_count'],eligible);point=route(branches,accepted)
        baseline=plan['config']['budgets'][case['dataset']]['baseline_f1'][case['expert']]
        target=plan['config']['budgets'][case['dataset']].get('minimum_f1',baseline)
        results[name]=dict(primary=point,baseline_f1=baseline,minimum_f1=target,delta_f1_points=100*(point['combined']['f1']-baseline),target_met=point['combined']['f1']>=target-1e-12,
            all_light=route(branches,list(branches)),all_deepsolo=route(branches,[]))
        seal(root/'test'/(name+'.json'),dict(branches=branches,scores=scores,eligible=eligible,accepted_ids=accepted))
        print(json.dumps(dict(stage='test_evaluated',case=name,f1=point['combined']['f1'],deepsolo_images=point['deepsolo_images'],total_images=point['total_images'],deepsolo_percentage=point['deepsolo_percentage'],target_met=results[name]['target_met'])),flush=True)
    seal(root/'results.json',dict(mode='approximate',claim_label='approximate reproduction',status='COMPLETE',all_targets_met=all(v['target_met'] for v in results.values()),results=results,
        actual_new_deepsolo_inference=0,actual_new_neural_inference=0,test_used_for_selection=False,historical_test_exposure=True))


def execute(root,stage):
    plan=read(root/'resolved_configuration.json')
    if plan['mode']!='approximate':raise ValueError('APPROXIMATE only')
    for p,h in plan['source_sha256'].items():
        if sha(REPO/p)!=h:raise ValueError('Scientific source changed: '+p)
    with threadpool_limits(limits=plan['config']['cpu_threads']):
        if stage in ('prepare_data','all') and not (root/'cases.json').exists():prepare_data(plan,root)
        if stage in ('train','all'):
            cases=read(root/'cases.json')
            for name,case in cases.items():
                if not (root/'models'/name/'selection_frozen.json').exists():train_case(plan,root,name,case)
            seal(root/'training_complete.json',dict(status='COMPLETE',mode='approximate',test_labels_opened=False,
                frozen_sha256={name:sha(root/'models'/name/'selection_frozen.json') for name in cases}))
        if stage in ('evaluate','all') and not (root/'results.json').exists():evaluate(plan,root)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',choices=['approximate'],required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--run-root',type=Path);parser.add_argument('--run-name',default='full');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stage',choices=['prepare','prepare_data','train','evaluate','all'],default='all');a=parser.parse_args()
    if not a.run_root and not a.config:parser.error('Provide explicit configuration or frozen run root')
    root=a.run_root or prepare(a)
    try:
        if a.stage!='prepare':execute(root,a.stage)
    except BaseException as exc:
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
    print(json.dumps(dict(status='SUCCESS',stage=a.stage,root=str(root))),flush=True)


if __name__=='__main__':main()
