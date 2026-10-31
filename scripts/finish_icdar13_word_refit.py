"""Validation-only policy/gate selection followed by complete IC13 TEST evaluation."""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import joblib
import numpy as np
from PIL import Image
import torch

from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.run_yolo_f1_quality_study import seal
from scripts.train_icdar13_word_refit import recognizer,read_rows
from scripts.icdar_budget_stages import score,route,fallback_count
from recognition.icdar_budget_router import observation,lexical_counts,rank,calibration_cut,accept
from recognition.lightweight_unet_router import fit_head,paired_utility
from recognition.unet_token_routing import selected_rows
from recognition.f1_quality_study import aggregate
from recognition.trainable_image_router import oracle
from recognition.fallback_feasibility import target_minimum,json_point
from recognition.icdar_budget_deployment import predict as single_predict
from segmentation.icdar13_word_refit import WordRefitUNet,strict_fallback_count
from segmentation.icdar2015_improvements import make_transform


def full_maps(net,r,config):
    if sha(r['image_path'])!=r['image_sha256']:raise ValueError('Changed inference RGB')
    with Image.open(r['image_path']) as p:image=p.convert('RGB')
    transform=make_transform(config,config['image_hw'],augment=False)
    tensor,_,_=transform.transform_with_validity(image,Image.new('L',image.size),Image.new('L',image.size,255),0)
    with torch.inference_mode():return net(tensor[None].to(config['device']))[0].sigmoid().cpu().numpy()


def load_model(root,branch,config,*,allow_smoke):
    manifest=read(root/branch/'training.json')
    if manifest['status']!='SUCCESS' or (not manifest['training_complete'] and not allow_smoke) or sha(root/branch/'best.pt')!=manifest['checkpoint_sha256']:raise ValueError('Incomplete/changed qualified branch')
    state=torch.load(root/branch/'best.pt',map_location='cpu',weights_only=False)
    if state['plan_sha256']!=sha(root/'resolved_configuration.json'):raise ValueError('Different training contract')
    net=WordRefitUNet(partial_encoder=state['partial_encoder'],skip_floor=config['skip_floor'],initial_gap_bias=config['initial_gap_bias'])
    net.load_state_dict(state['model_state_dict'],strict=True);return net.requires_grad_(False).eval().to(config['device'])


def emit(raw,settings,finish):
    rows=selected_rows(raw,settings)
    return [r for r in rows if r['raw_text'].strip()] if finish['reject_empty_transcriptions'] else rows


def choose_expert(root,plan,finish):
    c=plan['config'];lookup={r['sample_id']:r for r in plan['records']};records=[lookup[s] for s in plan['partitions']['selection']]
    candidates=[];ocr=recognizer(plan)
    try:
        for branch in ['frozen','partial']:
            # Reuse only the strictly replayed, chosen checkpoint's maps.
            for text,padding in itertools.product(c['policy_grid']['text_threshold'],c['policy_grid']['padding']):
                settings=dict(c['inference'],text_threshold=text,padding=padding,minimum_transcript_probability=0.)
                raw={};maps={}
                for r in records:
                    source=root/branch/'selected_validation_replay/samples'/(key(r['sample_id'])+'.json');saved=read(source)
                    if sha(saved['maps_path'])!=saved['maps_sha256']:raise ValueError('Validation probability changed')
                    with np.load(saved['maps_path'],allow_pickle=False) as z:prob=z['probabilities']
                    _,raw[r['sample_id']],_=read_rows(prob,r,settings,ocr,c['parseq_batch_size'])
                for cut in c['policy_grid']['minimum_transcript_probability']:
                    policy=dict(settings,minimum_transcript_probability=cut);counts=[];eligible=0
                    for r in records:
                        rows=emit(raw[r['sample_id']],policy,finish);gt,valid=annotation(r)
                        counts.append(lexical_counts(predictions(rows,r,valid),gt));eligible+=bool(rows) and all(v['raw_text'].strip() for v in rows)
                    value=aggregate(counts)
                    candidates.append(dict(branch=branch,policy=policy,selection=value,eligible_images=eligible,
                        deepsolo_images=0,total_images=len(records),deepsolo_percentage=0.,checkpoint_sha256=sha(root/branch/'best.pt')))
            print(json.dumps(dict(stage='policy_selection',branch=branch,candidates=len(candidates))),flush=True)
    finally:ocr.release();torch.cuda.empty_cache()
    needed=len(records)-strict_fallback_count(len(records),c['maximum_deepsolo_fraction'])
    eligible=[v for v in candidates if v['eligible_images']>=needed]
    if not eligible:raise ValueError('No validation-qualified expert can satisfy the declared fallback budget')
    chosen=max(eligible,key=lambda v:(v['selection']['f1'],-candidates.index(v)))
    seal(root/'selected_expert.json',dict(choice=chosen,candidates=candidates,selection_ids=plan['partitions']['selection'],test_used=False,finish_profile=finish))
    return chosen


def infer(plan,root,choice,records,finish):
    c=plan['config'];net=load_model(root,choice['branch'],c,allow_smoke=plan['smoke']);ocr=recognizer(plan)
    try:
        for i,r in enumerate(records,1):
            destination=root/'experts/refit'/(key(r['sample_id'])+'.json')
            if destination.exists():
                old=read(destination)
                if old['choice']!=choice or old['image_sha256']!=r['image_sha256'] or sha(old['maps_path'])!=old['maps_sha256']:raise ValueError('Changed completed expert output')
                continue
            torch.cuda.reset_peak_memory_stats();prob=full_maps(net,r,c)
            _,raw,audit=read_rows(prob,r,choice['policy'],ocr,c['parseq_batch_size']);rows=emit(raw,choice['policy'],finish)
            mp=root/'maps/refit'/(key(r['sample_id'])+'.npz');mp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mp,probabilities=prob)
            seal(destination,dict(choice=choice,image_sha256=r['image_sha256'],rows=rows,raw_rows=raw,frame='square_letterbox',
                maps_path=str(mp),maps_sha256=sha(mp),ground_truth_used=False,expansion=audit,
                gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved()))
            if i%50==0:print(json.dumps(dict(stage='expert_inference',images=i,total=len(records))),flush=True)
    finally:ocr.release();del net;torch.cuda.empty_cache()


def collect(root,plan,records):
    parent=read(plan['paths']['parent_plan']);parent_root=Path(plan['paths']['parent_root']);features={};branches={};keys={}
    for r in records:
        sid=r['sample_id'];d=read(root/'experts/refit'/(key(sid)+'.json'))
        if sha(d['maps_path'])!=d['maps_sha256'] or d['ground_truth_used'] or d['image_sha256']!=r['image_sha256']:raise ValueError('Expert evidence changed')
        with np.load(d['maps_path'],allow_pickle=False) as z:maps=z['probabilities']
        f=observation(d['rows'],r['image_wh'],maps,d['frame'],settings=plan['config']['features'])
        features[sid]=dict(groups=f,scale=len(d['rows'])+1,eligible=bool(d['rows']) and all(v['raw_text'].strip() for v in d['rows']))
        gt,valid=annotation(r);light=lexical_counts(predictions(d['rows'],r,valid),gt);fallback=fallback_count(parent,parent_root,r)
        if light['gt_words']!=fallback['gt_words']:raise ValueError('Different expert care denominators')
        branches[sid]=dict(light=light,deepsolo=fallback);keys[sid]=r['rgb_sha256']
    return features,branches,keys


def audit_heads(root,plan,features,branches,trials):
    p=plan['partitions'];reference=aggregate(branches[s]['deepsolo'] for s in p['fit'])['f1'];weights=np.array([features[s]['scale'] for s in p['fit']],float)
    y=np.array([paired_utility(branches[s],reference)/features[s]['scale'] for s in p['fit']]);checked=0
    for trial in trials:
        b=joblib.load(root/trial['model_path'])
        if sha(root/trial['model_path'])!=trial['model_sha256'] or b['training_ids']!=p['fit']:raise ValueError('Changed trained gate')
        x=np.stack([np.concatenate([features[s]['groups'][g] for g in b['groups']]) for s in p['fit']])
        if not np.allclose(x.mean(0),b['scaler'].mean_,atol=1e-12,rtol=0):raise ValueError('Scaling saw non-fit images')
        if not np.allclose(x.var(0),b['scaler'].var_,atol=1e-12,rtol=1e-8):raise ValueError('Scale variance saw non-fit images')
        if trial['recipe']['kind']=='ridge':
            z=b['scaler'].transform(x);mu=np.average(z,axis=0,weights=weights);my=np.average(y,weights=weights);center=z-mu
            coef=np.linalg.solve((center*weights[:,None]).T@center+trial['recipe']['alpha']*np.eye(x.shape[1]),(center*weights[:,None]).T@(y-my))
            if not np.allclose(coef,b['regressor'].coef_,rtol=1e-7,atol=1e-9):raise ValueError('Independent Ridge solution differs')
        else:
            replay=fit_head(b['groups'],x,y,weights,trial['recipe'],plan['config']['seed'])
            if not np.array_equal(replay['regressor'].predict(replay['scaler'].transform(x)),b['regressor'].predict(b['scaler'].transform(x))):raise ValueError('Tree fitting replay differs')
        checked+=1
    return checked


def finish(root,profile,*,smoke=False):
    plan=read(root/'resolved_configuration.json');c=plan['config'];settings=read(profile);p=plan['partitions']
    if plan['mode']!='approximate' or settings['mode']!='approximate' or plan['smoke']!=smoke:raise ValueError('Explicit matching approximate full/smoke mode required')
    torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    for file,digest in c['pinned_inputs'].items():
        if sha(file)!=digest:raise ValueError('Changed pinned input')
    seal(root/'finish_resolved_configuration.json',dict(config=settings,profile_sha256=sha(profile),training_plan_sha256=sha(root/'resolved_configuration.json'),code_sha256=sha(__file__)))
    choice=choose_expert(root,plan,settings);lookup={r['sample_id']:r for r in plan['records']};ids=p['fit']+p['selection']+p['calibration'];records=[lookup[s] for s in ids]
    infer(plan,root,choice,records,settings);features,branches,keys=collect(root,plan,records)
    seal(root/'train_branches.json',dict(branches=branches,tie_keys=keys))
    np.savez_compressed(root/'training_features.npz',ids=np.array(ids),word_scale=np.array([features[s]['scale'] for s in ids]),eligible=np.array([features[s]['eligible'] for s in ids]),
        **{g:np.stack([features[s]['groups'][g] for s in ids]) for g in features[ids[0]]['groups']})
    reference=aggregate(branches[s]['deepsolo'] for s in p['fit'])['f1'];weights=np.array([features[s]['scale'] for s in p['fit']],float)
    target=np.array([paired_utility(branches[s],reference)/features[s]['scale'] for s in p['fit']]);trials=[]
    q=(len(p['selection'])-strict_fallback_count(len(p['selection']),c['maximum_deepsolo_fraction']))/len(p['selection'])
    for groups in [['ocr'],['ocr','tokens'],['ocr','maps'],['ocr','tokens','maps']]:
        x=np.stack([np.concatenate([features[s]['groups'][g] for g in groups]) for s in p['fit']])
        for index,recipe in enumerate(c['heads']):
            b=fit_head(groups,x,target,weights,recipe,c['seed']);b.update(training_ids=p['fit'],reference_f1=reference)
            path=root/'routers'/('_'.join(groups)+'_'+str(index)+'.joblib');path.parent.mkdir(exist_ok=True)
            if path.exists():raise ValueError('Refuse to refit an existing gate')
            joblib.dump(b,path);values=score(b,features,p['selection'])
            point=route(branches,p['selection'],rank(values,p['selection'],q,keys))
            if point['deepsolo_images']>strict_fallback_count(len(p['selection']),c['maximum_deepsolo_fraction']):raise ValueError('Selection budget infeasible')
            trials.append(dict(groups=groups,recipe=recipe,scalar_count=b['stored_scalar_count'],serialized_bytes=path.stat().st_size,
                model_path=str(path.relative_to(root)),model_sha256=sha(path),selection=point))
    best=max(t['selection']['combined']['f1'] for t in trials);winner=min([t for t in trials if t['selection']['combined']['f1']>=best-c['selection_tolerance']],key=lambda t:(t['scalar_count'],t['serialized_bytes'],-t['selection']['combined']['f1'],t['model_path']))
    b=joblib.load(root/winner['model_path']);values=score(b,features,p['calibration']);policies=[]
    for cap in settings['fallback_fraction_controls']:
        q=(len(p['calibration'])-strict_fallback_count(len(p['calibration']),cap))/len(p['calibration'])
        cut=calibration_cut(values,p['calibration'],q,keys)
        policies.append(dict(cut,maximum_deepsolo_fraction=cap,model_sha256=winner['model_sha256'],calibration=route(branches,p['calibration'],accept(values,cut,keys))))
    seal(root/'trials.json',trials);seal(root/'frozen_router.json',dict(choice=winner,policies=policies,fit_ids=p['fit'],selection_ids=p['selection'],calibration_ids=p['calibration'],test_used=False,no_refit=True))
    audited=audit_heads(root,plan,features,branches,trials)
    parent=read(plan['paths']['parent_plan']);test_records=[r for r in parent['records']['icdar13']['test'] if not smoke or r['sample_id'] in p['test']]
    infer(plan,root,choice,test_records,settings);tf,tb,tk=collect(root,plan,test_records);test_ids=[r['sample_id'] for r in test_records];scores=score(b,tf,test_ids)
    expected_images=len(p['test']) if smoke else settings['expected_test_images']
    expected_words=settings['smoke_expected_test_care_words'] if smoke else settings['expected_test_care_words']
    if len(test_ids)!=expected_images or sum(v['light']['gt_words'] for v in tb.values())!=expected_words:raise ValueError('Incomplete declared TEST denominator')
    cut_points=[dict(policy=cut,**route(tb,test_ids,accept(scores,cut,tk))) for cut in policies];quota_points={};uppers={}
    for cap in settings['fallback_fraction_controls']:
        keep_count=len(test_ids)-strict_fallback_count(len(test_ids),cap);keep=rank(scores,test_ids,keep_count/len(test_ids),tk)
        point=route(tb,test_ids,keep)
        if point['deepsolo_images']!=strict_fallback_count(len(test_ids),cap):raise ValueError('TEST budget is infeasible with current eligibility guard')
        quota_points[str(cap)]=point;upper=oracle(tb,test_ids,[s for s in test_ids if scores[s] is not None],len(keep))
        uppers[str(cap)]=dict(**route(tb,test_ids,upper['accepted_ids']),certificate=upper['certificate_max_residual'])
    maximum_error=0.;checked=0
    for r in test_records:
        d=read(root/'experts/refit'/(key(r['sample_id'])+'.json'))
        with np.load(d['maps_path'],allow_pickle=False) as z:maps=z['probabilities']
        value=single_predict(b,c['features'],d['rows'],r['image_wh'],tie_key=r['rgb_sha256'],policies=policies,probabilities=maps,frame=d['frame'])
        expected=scores[r['sample_id']]
        if (value['score'] is None)!=(expected is None):raise ValueError('Single-image eligibility differs')
        if expected is not None:maximum_error=max(maximum_error,abs(expected-value['score']))
        for point in cut_points:
            if value['accept_light'][str(point['policy']['target_acceptance'])]!=(r['sample_id'] in point['accepted_ids']):raise ValueError('Single-image decision differs')
            checked+=1
    if maximum_error>1e-9:raise ValueError('Single-image gate score changed')
    seal(root/'test_scores.json',dict(scores=scores,tie_keys=tk,branches=tb))
    result=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',result_class=plan['result_class'],expert=choice,router=winner,thresholds=cut_points,
        strict_budget_quotas=quota_points,conditional_gt_oracles=uppers,all_light=route(tb,test_ids,test_ids),all_deepsolo=route(tb,test_ids,[]),
        test_used_for_selection=False,no_refit=True,actual_new_deepsolo_calls=0,scope='Logical routing on cached fallback; fresh native expert inference',
        target_f1=c['target_f1'],primary_target_met=quota_points[str(c['maximum_deepsolo_fraction'])]['combined']['f1']>=c['target_f1'])
    seal(root/'results.json',result)
    seal(root/'independent_audit.json',dict(status='PASS',results_sha256=sha(root/'results.json'),candidate_fit_checks=audited,
        single_image_decisions=checked,maximum_single_batch_score_error=maximum_error,fit_ids=p['fit'],selection_ids=p['selection'],calibration_ids=p['calibration'],
        training_features_sha256=sha(root/'training_features.npz'),script_sha256=sha(__file__),scope='Source-bound expert inference, fit-only scaling/Ridge equations/tree replay, official TEST denominators, independent route sums, single-image deployment decisions and conditional exact oracle certificates'))
    joblib.dump(dict(bundle=b,features=c['features'],policies=policies,expert=choice),root/'deployment.joblib')
    seal(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),audit_sha256=sha(root/'independent_audit.json')))
    print(json.dumps(dict(status='SUCCESS',target_met=result['primary_target_met'],quotas={k:dict(f1=v['combined']['f1'],deepsolo_images=v['deepsolo_images'],total_images=v['total_images'],deepsolo_percentage=v['deepsolo_percentage']) for k,v in quota_points.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.profile.resolve().parent.name!='approximate':raise ValueError('Explicit approximate finish profile required')
    try:finish(a.run_root,a.profile,smoke=a.smoke)
    except BaseException as error:
        import traceback
        seal(a.run_root/('finish_failure_'+str(time.time_ns())+'.json'),dict(error=str(error),traceback=traceback.format_exc(),oom=isinstance(error,torch.cuda.OutOfMemoryError)));raise


if __name__=='__main__':main()
