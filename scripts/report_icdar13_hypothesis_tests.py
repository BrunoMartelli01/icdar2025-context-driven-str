"""Independent fit/count audits and paired summaries for the completed IC13 tests."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from recognition.icdar13_hypothesis_tests import normalize,gt_crop_diagnosis
from recognition.f1_quality_study import aggregate
from recognition.trainable_image_router import oracle
from scripts.icdar_budget_stages import route
from scripts.run_yolo_f1_quality_study import seal
from segmentation.icdar13_matched_training import emit_words
from scripts.run_icdar13_hypothesis_tests import cached
from recognition.icdar13_matched_router import exact_accepted


def csv_recount(rows,record):
    """Read official transcription CSV directly, independent of the model evaluator."""
    p=Path(record['annotation_path'])
    if sha(p)!=record['annotation_sha256']:raise ValueError('Changed official annotation')
    with p.open(encoding='utf-8-sig',newline='') as f:labels=list(csv.reader(f,skipinitialspace=True))
    if any(len(row)!=5 for row in labels):raise ValueError('Expected five-column official Task1 CSV')
    _,valid=annotation(record)
    all_predictions=predictions(rows,record,valid)
    pp=[r for r in all_predictions if not r['ignored']]
    pred=Counter(normalize(r['text']) for r in pp if normalize(r['text']))
    truth=Counter(normalize(r[4]) for r in labels if normalize(r[4]))
    tp=sum((pred&truth).values())
    return dict(tp=tp,fp=len(pp)-tp,fn=len(labels)-tp,detected_words=len(pp),gt_words=len(labels),
        all_outputs=len(all_predictions),ignored_outputs=len(all_predictions)-len(pp))


def compare_counts(actual,expected):
    if any(expected[k]!=v for k,v in actual.items()):raise ValueError('Independent CSV/count replay failed')


def audit_heads(root,plan):
    verified=0
    for arm in ['seen_control','crossfit_control','crossfit_raw_no_guard']:
        folder=root/'routers'/arm;obs=read(folder/'observations.json');trials=read(folder/'selection_frozen.json')['trials']
        with np.load(folder/'features.npz') as z:
            ids=list(z['ids']);indices=[ids.index(s) for s in obs['fit_ids']];y=z['targets'];w=z['weights']
            deep=[obs['branches'][s]['deepsolo'] for s in obs['fit_ids']]
            reference=2*sum(v['tp'] for v in deep)/sum(v['detected_words']+v['gt_words'] for v in deep)
            expected=[]
            for index,sid in zip(indices,obs['fit_ids']):
                pair=obs['branches'][sid];a,b=pair['light'],pair['deepsolo']
                expected.append((2*(a['tp']-b['tp'])-reference*(a['detected_words']-b['detected_words']))/z['scale'][index])
            if not np.allclose(y,expected,rtol=1e-12,atol=1e-12) or not np.array_equal(w,z['scale'][indices]):raise ValueError('Independent FIT utility/weight reconstruction differs')
            for trial in trials:
                bundle=joblib.load(folder/trial['path'])
                if sha(folder/trial['path'])!=trial['sha256']:raise ValueError('Changed fitted weights')
                x=np.concatenate([z[g] for g in trial['groups']],axis=1)[indices]
                mean=x.mean(0);scale=x.std(0);scale[scale==0]=1
                if not np.allclose(mean,bundle['scaler'].mean_,rtol=1e-12,atol=1e-12):raise ValueError('Non-FIT scaler mean')
                if not np.allclose(scale,bundle['scaler'].scale_,rtol=1e-12,atol=1e-12):raise ValueError('Non-FIT scaler scale')
                xx=(x-mean)/scale;r=trial['recipe'];model=bundle['regressor']
                if r['kind']=='ridge':
                    # Independent weighted centered normal equations, including intercept.
                    xm=np.average(xx,axis=0,weights=w);ym=np.average(y,weights=w);xc=xx-xm
                    coef=np.linalg.solve((xc*w[:,None]).T@xc+r['alpha']*np.eye(xx.shape[1]),(xc*w[:,None]).T@(y-ym))
                    expected=(xx-xm)@coef+ym
                    if not np.allclose(coef,model.coef_,rtol=1e-8,atol=1e-9):raise ValueError('Ridge normal equations differ')
                else:
                    alternate=HistGradientBoostingRegressor(max_iter=r['iterations'],max_leaf_nodes=r['leaves'],l2_regularization=r['l2'],min_samples_leaf=r['minimum_leaf'],early_stopping=False,random_state=plan['config']['seed'])
                    alternate.fit(xx,y,sample_weight=w);expected=alternate.predict(xx)
                if not np.allclose(expected,model.predict(xx),rtol=1e-8,atol=1e-9):raise ValueError('Independent fitted head differs')
                verified+=1
    return verified


def paired_interval(before,after,seed,draws):
    a=np.array([[r[k] for k in ['tp','fp','fn']] for r in before]);b=np.array([[r[k] for k in ['tp','fp','fn']] for r in after])
    if a.shape!=b.shape:raise ValueError('Paired populations differ')
    rng=np.random.default_rng(seed);indices=rng.integers(len(a),size=(draws,len(a)))
    def f(x):
        total=x[indices].sum(1);den=2*total[:,0]+total[:,1]+total[:,2]
        return np.divide(200*total[:,0],den,out=np.zeros(len(den)),where=den!=0)
    delta=f(b)-f(a)
    return dict(draws=draws,seed=seed,delta_f1_percentage_points_95_interval=np.quantile(delta,[.025,.975]).tolist(),
        scope='Conditional paired image bootstrap; fixed fitted models/selection/GT exposure; omits model-selection uncertainty')


def ranking_diagnosis(data,point,ideal_ids):
    """GT-only explanation of a frozen score order, never a deployable feature."""
    from scipy.stats import spearmanr
    accepted=set(point['accepted_ids']);fallback=set(data['scores'])-accepted
    reference=point['combined']['f1'];benefit={}
    for sid,b in data['branches'].items():
        a,d=b['light'],b['deepsolo']
        benefit[sid]=2*(d['tp']-a['tp'])-reference*(d['detected_words']-a['detected_words'])
    eligible=[s for s in data['scores'] if data['scores'][s] is not None]
    scores=[data['scores'][s] for s in eligible];actual=[-benefit[s] for s in eligible]
    rho=float(spearmanr(scores,actual).statistic) if len(set(scores))>1 and len(set(actual))>1 else None
    missed=sorted(accepted,key=lambda s:(-benefit[s],s))[:10]
    return dict(scope='TEST-descriptive frozen-order diagnosis; GT is never fed to the gate',
        deepsolo_images=len(fallback),total_images=len(data['scores']),deepsolo_percentage=100*len(fallback)/len(data['scores']),
        score_vs_true_utility_spearman=rho,undefined_correlation_reason='constant score or utility' if rho is None else None,
        positive_benefit_fallbacks=sum(benefit[s]>0 for s in fallback),negative_benefit_fallbacks=sum(benefit[s]<0 for s in fallback),
        zero_benefit_fallbacks=sum(benefit[s]==0 for s in fallback),accepted_images_with_positive_fallback_benefit=sum(benefit[s]>0 for s in accepted),
        overlap_with_one_optimal_subset=len(fallback&(set(data['scores'])-set(ideal_ids))),
        counterfactual_examples=[dict(sample_id=s,light=data['branches'][s]['light'],deepsolo=data['branches'][s]['deepsolo'],fallback_benefit=benefit[s],score=data['scores'][s]) for s in missed])


def selection_stability(folder,plan):
    """Leave-one-image-out selection replay with fixed fits; no TEST selection."""
    obs=read(folder/'observations.json');saved=read(folder/'selection_frozen.json');all_scores={}
    with np.load(folder/'features.npz') as z:
        ids=list(z['ids']);indices=[ids.index(s) for s in obs['selection_ids']]
        for trial in saved['trials']:
            bundle=joblib.load(folder/trial['path']);x=np.concatenate([z[g] for g in trial['groups']],axis=1)[indices]
            values=bundle['regressor'].predict(bundle['scaler'].transform(x))
            all_scores[trial['path']]={s:float(v)*z['scale'][i] if z['eligible'][i] else None for s,i,v in zip(obs['selection_ids'],indices,values)}
    outcomes=[]
    for excluded in obs['selection_ids']:
        remaining=[s for s in obs['selection_ids'] if s!=excluded];fallback=round(len(remaining)*plan['config']['primary_deepsolo_images']/233)
        candidates=[]
        for trial in saved['trials']:
            scores={s:all_scores[trial['path']][s] for s in remaining};accept=exact_accepted(scores,{s:obs['tie_keys'][s] for s in remaining},fallback)
            point=route(obs['branches'],remaining,accept);candidates.append(dict(trial=trial,point=point))
        best=max(v['point']['combined']['f1'] for v in candidates)
        winner=min([v for v in candidates if v['point']['combined']['f1']>=best-plan['config']['selection_tolerance']],key=lambda v:(v['trial']['scalars'],v['trial']['bytes'],-v['point']['combined']['f1'],v['trial']['path']))
        outcomes.append(dict(excluded=excluded,winner=winner['trial']['path'],f1=winner['point']['combined']['f1'],deepsolo_images=fallback,total_images=len(remaining),deepsolo_percentage=100*fallback/len(remaining)))
    counts=Counter(v['winner'] for v in outcomes)
    return dict(scope='Descriptive validation sensitivity, fixed candidate fits; no new fitting, deployment changes or TEST selection',
        full_selection_winner=saved['winner']['path'],same_winner_replays=counts[saved['winner']['path']],replays=len(outcomes),distinct_winners=len(counts),winner_counts=dict(counts),outcomes=outcomes)


def report(root,report_name='report'):
    if Path(report_name).name!=report_name or report_name in ('.','..'):
        raise ValueError('Report name must be a single directory name')
    plan=read(root/'resolved_configuration.json');c=plan['config'];ids=plan['partitions']['test'];records={r['sample_id']:r for r in plan['records']}
    import torch
    torch.set_num_threads(c['cpu_threads'])
    if plan['smoke']:raise ValueError('Scientific summary requires full run')
    if len(ids)!=233:raise ValueError('Incomplete TEST')
    vocabulary=read(root/'vocabulary_results.json');details=read(root/'vocabulary_details.json');routers=read(root/'router_results.json');rows=[];contrasts={};audited=0
    baseline=[details[s]['none']['counts'] for s in ids]
    def add(name,counts,fallback,kind):
        totals=aggregate(counts);rows.append(dict(name=name,scope=kind,tp=totals['tp'],fp=totals['fp'],fn=totals['fn'],f1_percent=100*totals['f1'],deepsolo_images=fallback,total_images=len(ids),deepsolo_percentage=100*fallback/len(ids)))
    for policy in c['generic_policies']:
        cc=[]
        for s in ids:
            data=details[s][policy];ct=csv_recount(data['rows'],records[s]);compare_counts(ct,data['counts']);cc.append(ct);audited+=1
        compare_counts({k:sum(v[k] for v in cc) for k in cc[0]},vocabulary['results']['test'][policy]['counts'])
        add('light_'+policy,cc,0,'Expert control; vocabulary chosen on selection only')
        contrasts['light_'+policy]=paired_interval(baseline,cc,c['seed'],c['bootstrap_draws'])
    raw_control=[]
    for sid in ids:
        raw=cached(plan,sid)['raw_rows'];raw_control.append(csv_recount(emit_words(raw,0.),records[sid]));audited+=1
    add('raw_no_probability_filter',raw_control,0,'TEST-descriptive diagnostic; no operational threshold selected')
    covered=[];steps=0
    for i,fold in enumerate(plan['folds']):
        done=read(root/f'fold_{i}/complete.json');epochs=read(root/f'fold_{i}/epochs.json')
        if done['excluded_ids']!=fold or set(done['fit_ids'])!=set(plan['partitions']['fit'])-set(fold):raise ValueError('OOF contamination')
        if (len(epochs)!=c['epochs'] or not done['frozen_encoder_tensors_verified']
            or sha(root/f'fold_{i}/last.pt')!=done['checkpoint_sha256']):raise ValueError('Incomplete fold/encoder/weight qualification')
        if any(v['fit_samples']!=len(done['fit_ids']) for v in epochs):raise ValueError('Missing fitting images')
        covered+=fold;steps+=sum(v['optimizer_steps'] for v in epochs)
    if sorted(covered)!=sorted(plan['partitions']['fit']):raise ValueError('OOF population incomplete or duplicated')
    independent_fits=audit_heads(root,plan)
    with np.load(root/'routers/seen_control/features.npz') as seen,np.load(root/'routers/crossfit_control/features.npz') as oof:
        if not np.array_equal(seen['ids'],oof['ids']):raise ValueError('Seen/OOF populations differ')
        outside_fit=len(plan['partitions']['fit'])
        for group in ['ocr','tokens','maps','raw','scale','eligible']:
            if not np.array_equal(seen[group][outside_fit:],oof[group][outside_fit:]):raise ValueError('Seen/OOF changed heldout expert observations')
    teacher=read(c['teacher_counts'])['samples'];old=read(Path(c['router_root'])/'results.json');old_accepted=set(old['primary']['accepted_ids']);route_baseline=[]
    add('all_deepsolo',[teacher[s]['counts'] for s in ids],len(ids),'Authenticated cached standalone fallback expert')
    original_branches={s:dict(light=details[s]['none']['counts'],deepsolo=teacher[s]['counts']) for s in ids}
    older=old['frozen_old_gate_control'];older_accepted=set(older['accepted_ids'])
    if not older_accepted<=set(ids) or len(ids)-len(older_accepted)!=c['primary_deepsolo_images']:
        raise ValueError('Older frozen gate does not use the same exact budget')
    older_counts=[original_branches[s]['light' if s in older_accepted else 'deepsolo'] for s in ids]
    compare_counts({k:sum(v[k] for v in older_counts) for k in ['tp','fp','fn','gt_words','detected_words','all_outputs','ignored_outputs']},older['combined'])
    add('older_frozen_gate',older_counts,c['primary_deepsolo_images'],'Previously trained frozen gate on the same current expert outputs')
    original_oracle=oracle(original_branches,ids,ids,len(ids)-c['primary_deepsolo_images'])
    original_point=route(original_branches,ids,original_oracle['accepted_ids'])
    add('current_gt_oracle',[original_branches[s]['light' if s in set(original_oracle['accepted_ids']) else 'deepsolo'] for s in ids],c['primary_deepsolo_images'],'Privileged GT upper bound; not a deployable gate')
    hash_scores={s:0. if details[s]['none']['rows'] else None for s in ids}
    hash_accept=set(exact_accepted(hash_scores,{s:records[s]['rgb_sha256'] for s in ids},c['primary_deepsolo_images']))
    hash_counts=[original_branches[s]['light' if s in hash_accept else 'deepsolo'] for s in ids]
    add('hash_only_guarded',hash_counts,c['primary_deepsolo_images'],'Post-hoc descriptive GT-blind rank reference; no learned parameters')
    for sid in ids:
        route_baseline.append(details[sid]['none']['counts'] if sid in old_accepted else teacher[sid]['counts'])
    ranking={};stability={}
    for arm,v in routers['results'].items():
        data=read(root/'routers'/arm/'test_scores.json');eligible=[s for s in ids if data['scores'][s] is not None]
        rank=sorted(eligible,key=lambda s:(-data['scores'][s],data['tie_keys'][s],s));accepted=set(rank[:len(ids)-c['primary_deepsolo_images']])
        if accepted!=set(v['test']['accepted_ids']):raise ValueError('Independent exact rank differs')
        cc=[]
        for sid in ids:
            compare_counts(csv_recount(details[sid]['none']['rows'],records[sid]),data['branches'][sid]['light'])
            ct=data['branches'][sid]['light' if sid in accepted else 'deepsolo'];cc.append(ct);audited+=1
        compare_counts({k:sum(v[k] for v in cc) for k in ['tp','fp','fn','gt_words','detected_words']},v['test']['combined'])
        if len(ids)-len(accepted)!=61:raise ValueError('Exact61 budget violated')
        add(arm,cc,61,'NEW trained router; logical cached routing')
        contrasts[arm]=paired_interval(route_baseline,cc,c['seed'],c['bootstrap_draws'])
        contrasts[arm+'_vs_older_frozen_gate']=paired_interval(older_counts,cc,c['seed'],c['bootstrap_draws'])
        ranking[arm]=ranking_diagnosis(data,v['test'],original_oracle['accepted_ids'])
        stability[arm]=selection_stability(root/'routers'/arm,plan)
    # The identical seen-control experiment must reproduce the previously qualified gate.
    compare_counts({k:routers['results']['seen_control']['test']['combined'][k] for k in ['tp','fp','fn']},old['primary']['combined'])
    mixed=root/'mixed_totaltext';completion=read(mixed/'complete.json');history=read(mixed/'epochs.json')
    if sha(mixed/'best.pt')!=completion['checkpoint_sha256']:raise ValueError('Mixed weights changed')
    original=read(Path(c['light_root'])/'totaltext_bce_dice/epochs.json')
    if len(history)!=len(original) or any(a['fit_order_sha256']!=b['sample_order_sha256'] for a,b in zip(history,original)):raise ValueError('Mixed IC exposure/order differs from matched control')
    if any(v['fit_samples']!=183 or v['tt_samples']!=183 or v['optimizer_steps']!=23 for v in history):raise ValueError('Mixed matched-step contract failed')
    mixed_counts=[];branches={};geometry=Counter();gt_readings={r['sample_id']:r['gt_readings'] for r in read(root/'gt_crops/test.json') if r['padding']==c['gt_crop_padding'][0]}
    for sid in ids:
        sample=read(mixed/'test/samples'/(key(sid)+'.json'));ct=csv_recount(sample['rows'],records[sid]);compare_counts(ct,sample['counts']);mixed_counts.append(ct);audited+=1
        branches[sid]=dict(light=sample['counts'],deepsolo=teacher[sid]['counts'])
        gt,_=annotation(records[sid])
        for diagnosis in gt_crop_diagnosis(gt,sample['raw_rows'],sample['rows'],gt_readings[sid],c['crop_iou_threshold']):
            geometry.update({diagnosis['group']:1,diagnosis['group']+'_gt_readable':int(diagnosis['gt_crop_correct'])})
    add('mixed_totaltext_light',mixed_counts,0,'NEW neural training; selection-chosen epoch')
    contrasts['mixed_totaltext_light']=paired_interval(baseline,mixed_counts,c['seed'],c['bootstrap_draws'])
    fixed=read(root/'mixed_fixed_epoch_control/results.json')
    if fixed['epoch']!=c['epochs'] or fixed['mode']!='approximate':raise ValueError('Incomplete fixed30 control')
    fixed_folder=Path(fixed['results']['test']['source']);fixed_counts=[]
    for sid in ids:
        sample=read(fixed_folder/'samples'/(key(sid)+'.json'));ct=csv_recount(sample['rows'],records[sid]);compare_counts(ct,sample['counts']);fixed_counts.append(ct);audited+=1
    add('mixed_fixed_epoch_light',fixed_counts,0,'Predeclared equal30-epoch secondary control; no TEST selection')
    contrasts['mixed_fixed_epoch_light']=paired_interval(baseline,fixed_counts,c['seed'],c['bootstrap_draws'])
    frozen=route(branches,ids,old_accepted);upper=oracle(branches,ids,ids,172)
    seal(root/'mixed_router_controls.json',dict(frozen_gate_replay=frozen,unrestricted_gt_oracle=route(branches,ids,upper['accepted_ids']),certificate=upper['certificate_max_residual'],scope='Whole-image cached replay; mixed expert has no newly fitted router'))
    selected_vocab=vocabulary['selected_policy']
    for label,point in [('selected_vocabulary_frozen_gate',vocabulary['routes'][selected_vocab]['frozen_gate_control']),('mixed_frozen_gate',frozen),('mixed_gt_oracle',route(branches,ids,upper['accepted_ids']))]:
        v=point['combined'];rows.append(dict(name=label,scope='GT diagnostic' if 'oracle' in label else 'Frozen gate replay control',tp=v['tp'],fp=v['fp'],fn=v['fn'],f1_percent=100*v['f1'],deepsolo_images=point['deepsolo_images'],total_images=233,deepsolo_percentage=point['deepsolo_percentage']))
    out=root/report_name;out.mkdir(exist_ok=False)
    with (out/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    seal(out/'paired_bootstrap.json',contrasts)
    seal(out/'ranking_diagnosis.json',ranking)
    seal(out/'selection_stability.json',stability)
    seal(out/'mixed_crop_diagnosis.json',dict(scope='Spatial fixed-GT diagnostic, distinct from lexical F1',counts=dict(geometry),gt_words=1095,deepsolo_images=0,total_images=233,deepsolo_percentage=0.))
    seal(out/'independent_audit.json',dict(status='PASS',mode='approximate',claim_label='approximate reproduction',full_test_images=233,test_gt_words=1095,official_csv_recounts=audited,independent_head_fits=independent_fits,folds=len(plan['folds']),oof_images=len(covered),fold_optimizer_steps=steps,mixed_optimizer_steps=sum(v['optimizer_steps'] for v in history),mixed_selected_epoch=completion['selected_epoch'],matched_ic_sample_orders_verified=True,frozen_encoders_verified=True,actual_new_deepsolo_calls=0,auditor_sha256=sha(__file__)))
    seal(out/'summary.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',results=rows,selected_vocabulary=selected_vocab,
        crop_diagnosis=read(root/'crop_results.json')['results'],router_population_diagnostics={k:v['light_populations'] for k,v in routers['results'].items()},paired_bootstrap=contrasts))
    lines=['# ICDAR2013 hypothesis tests','', 'Approximate reproduction. Full233 TEST images /1095 care words; independent count/fit audit PASS. Logical cached routing; no new DeepSolo inference.','', '| Method | TP | FP | FN | F1 % | DeepSolo images/total (%) |','| --- | ---: | ---: | ---: | ---: | --- |']
    for r in rows:lines.append(f"| {r['name']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['f1_percent']:.4f} | {r['deepsolo_images']}/{r['total_images']} ({r['deepsolo_percentage']:.2f}%) |")
    lines+=['', 'Vocabulary winner from selection only: '+selected_vocab+'.', 'Mixed selected epoch: '+str(completion['selected_epoch'])+'.', '', 'GT crops and GT router oracle are privileged diagnostics. They are not deployment scores or recognition ceilings. The mixed frozen-gate replay is not a newly trained gate. Bootstrap intervals condition on fixed models and omit historical TEST exposure and selection uncertainty. Source-selection/strict metric ambiguities remain.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',report=str(out),rows=rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--report-name',default='report');a=p.parse_args();report(a.run_root,a.report_name)
