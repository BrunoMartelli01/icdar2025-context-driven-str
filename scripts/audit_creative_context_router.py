"""Replay six context signals, independent regressors/retrieval and routing counts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from scripts import run_creative_context_router as run
from scripts.analyze_trainable_image_router import independent_fit,optimum
from scripts.audit_yolo_f1_quality_study import aggregate,check_point,f1,point
from scripts.run_yolo_f1_quality_study import read,seal,sha


def audit(root):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];result=read(root/'results.json')
    frozen=read(root/'frozen_choices.json');cv=read(root/'cv_results.json');ph=sha(root/'resolved_configuration.json')
    assert result['status']=='SUCCESS' and result['plan_sha256']==ph and result['base_models_frozen']
    assert not result['original_word_outputs_changed'] and result['new_visual_model_inference']
    assert result['frozen_choices_sha256']==sha(root/'frozen_choices.json') and frozen['cv_results_sha256']==sha(root/'cv_results.json')
    for path,digest in plan['input_sha256'].items():assert sha(Path(path))==digest,path
    for path,digest in plan['code_sha256'].items():assert sha(root/'code_snapshot'/path)==digest,path
    reuse=plan.get('reused_neural_probes')
    if reuse:
        prior_root=Path(reuse['root']);prior_plan=read(prior_root/'resolved_configuration.json')
        assert sha(prior_root/'resolved_configuration.json')==reuse['plan_sha256']
        for key in ('config','samples','donor_ids','input_sha256'):assert plan[key]==prior_plan[key]
        prior_frozen=read(prior_root/'frozen_choices.json')
        for key in ('models_sha256','budgets','cv_results_sha256'):assert frozen[key]==prior_frozen[key]
        for file,digest in reuse['files_sha256'].items():
            assert sha(prior_root/file)==digest
            if '/permutation/' in file or '/swap/' in file:
                before=read(prior_root/file);after=read(root/file)
                assert after['reused_source_sha256']==digest
                ignored=('plan_sha256','frozen_choices_sha256','reused_source_sha256')
                assert {k:v for k,v in before.items() if k not in ignored}=={k:v for k,v in after.items() if k not in ignored}
        ignored=('plan_sha256','frozen_choices_sha256');prior_result=read(prior_root/'results.json')
        assert {k:v for k,v in prior_result.items() if k not in ignored}=={k:v for k,v in result.items() if k not in ignored}
    source=Path(plan['paths']['source_root']);coverage=Path(plan['source_coverage_root'])
    sample_lookup={s['sample_id']:s for ss in plan['samples'].values() for s in ss}
    branches=read(coverage/'results.json')['resolutions']['640']['branch_metrics'];observed={};counts={}
    for split,samples in plan['samples'].items():
        observed[split]={};counts[split]={'images':len(samples),'predicted_words':0,'conditional_abstentions':0,'glyph_windows':0,'swap_words':0}
        for sample in samples:
            sid=sample['sample_id'];key=sid.replace(':','__')+'.json'
            parts={stage:read(source/split/stage/key) for stage in ('features','inventory','verification')}
            rows=read(coverage/'640'/split/'features'/key)['rows'];observed[split][sid]={'sample':sample,'rows':rows,'source':parts}
            perm=read(root/split/'permutation'/key);swap=read(root/split/'swap'/key)
            binding=sha(root/'frozen_choices.json') if split=='test' else None
            for record in (perm,swap):
                assert record['sample_id']==sid and record['image_sha256']==sample['image_sha256'] and record['plan_sha256']==ph
                assert not record['ground_truth_used'] and record['frozen_choices_sha256']==binding
            assert len(perm['words'])==len(rows)
            for word,row in zip(perm['words'],rows):
                assert word['candidate_id']==row['candidate_id'] and word['text']==row['raw_text']
                if word.get('abstention'):
                    assert word['abstention'] in ('empty_prediction','prediction_exceeds_conditional_probe_capacity')
                    counts[split]['conditional_abstentions']+=1;continue
                assert not word['ground_truth_used'] and not word['original_word_output_changed']
                assert word['crop_rgb_sha256']==row['crop_rgb_sha256']
                p=np.asarray(word['target_probabilities']);assert p.shape==(4,len(word['text'])+1)
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                assert len(word['characters']) in (0,len(word['text']))
                counts[split]['glyph_windows']+=len(word['characters'])
            counts[split]['predicted_words']+=len(rows)
            assert len(swap['probes'])==len(parts['verification']['probes'])
            for probe,prior in zip(swap['probes'],parts['verification']['probes']):
                assert probe['candidate_id']==prior['candidate_id'] and probe['alternatives']==prior['alternatives']
                donor=plan['donor_ids'][split][sid]
                assert donor in plan['fit_ids'] and probe['donor_id']==donor
                donor_sample=sample_lookup[donor]
                assert donor_sample['image_sha256']!=sample['image_sha256']
                assert probe['donor_image_sha256']==donor_sample['image_sha256']
                job=run.expanded_job({k:prior[k] for k in ('candidate_id','target_box','context_box','line_candidate_ids','alternatives','line_transcript')},sample['image_wh'],cfg['swap_settings'])
                assert all(probe[k]==v for k,v in job.items())
                views=run.context_views(run.image(sample),job,run.image(donor_sample),cfg['swap_settings'])
                assert probe['views_rgb_sha256']=={k:run.array_sha(v) for k,v in views.items()}
                for answers in probe['answers'].values():
                    for answer,word in zip(answers,probe['alternatives']):
                        assert answer['prompt']==cfg['question'].format(word=json.dumps(word,ensure_ascii=False))
                        y,n=answer['yes_logprob'],answer['no_logprob'];m=max(y,n)
                        assert math.isclose(answer['yes_probability'],math.exp(y-m)/(math.exp(y-m)+math.exp(n-m)),rel_tol=2e-6,abs_tol=2e-7)
                counts[split]['swap_words']+=1
        for kind in ('permutation','swap'):
            receipt=read(root/split/(kind+'_receipt.json'));model=receipt['model']
            assert receipt['plan_sha256']==ph and receipt['images']==len(samples)
            assert model['optimizer_steps']==0 and model['model_state_sha256_before']==model['model_state_sha256_after']
            assert model['precision']=='float32' and model['cuda_peak']['peak_allocated_bytes']>0
            if kind=='permutation':assert model['attention_capture_qualification']['status']=='PASS'
            else:assert model['native_forward_check']['status']=='PASS'
    # Recompute all CPU geometry/glyph/font features, checking sealed vectors.
    features={s:run.features(cfg,plan,root,s,items) for s,items in observed.items()}
    frozen_ns=(root/'frozen_choices.json').stat().st_mtime_ns
    for kind in ('permutation','swap','features'):
        assert frozen_ns<=min(p.stat().st_mtime_ns for p in (root/'test'/kind).glob('*.json'))
    fit,val=plan['fit_ids'],plan['validation_ids'];assert set(fit+val)==set(features['train']) and not set(fit)&set(val)
    populations={'fit':('train',fit),'validation':('train',val),'train':('train',sorted(features['train'])),'test':('test',sorted(features['test']))}
    predictions={s:read(root/'predictions'/(s+'.json')) for s in populations};cache={};refits=0;held_count=0
    def matrix(split,ids,names):return np.array([[features[split][s]['values'][n] for n in names] for s in ids])
    def unit(x):
        x=np.asarray(x,float);length=np.linalg.norm(x,axis=1,keepdims=True);return x/np.maximum(length,1e-12)
    def reference(ids):return float(f1(aggregate(branches['train'][s]['deepsolo'] for s in ids)))
    def independent(ids,recipe,family):
        nonlocal refits
        assert set(ids)<=set(fit);kind=recipe['kind'];key=(tuple(ids),json.dumps(recipe,sort_keys=True),family)
        if key in cache:return cache[key]
        names=plan['feature_schemas']['core' if kind=='memory' else family.rsplit('_',1)[1]]
        y=np.array([[branches['train'][s]['light']['tp']-branches['train'][s]['deepsolo']['tp'],
            branches['train'][s]['light']['detected_words']-branches['train'][s]['deepsolo']['detected_words']] for s in ids])
        if kind=='memory':
            x=matrix('train',ids,names);mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1.
            vectors=unit([features['train'][s]['visual_embedding'] for s in ids])
            if recipe['variant']=='joint':vectors=unit(np.concatenate([vectors,unit((x-mean)/scale)],axis=1))
            model={'ids':ids,'vectors':vectors,'y':y,'mean':mean,'scale':scale,'names':names}
        else:model=independent_fit(matrix('train',ids,names),y,recipe,cfg)
        cache[key]=model;refits+=1;return model
    def scores(model,recipe,family,split,ids,ref):
        kind=recipe['kind']
        if kind=='control':return {s:features[split][s]['controls'][recipe['field']] if features[split][s]['eligible'] else None for s in ids}
        if kind=='memory':
            q=unit([features[split][s]['visual_embedding'] for s in ids])
            if recipe['variant']=='joint':q=unit(np.concatenate([q,unit((matrix(split,ids,model['names'])-model['mean'])/model['scale'])],axis=1))
            distance=cdist(q,model['vectors'],metric='cosine');estimates=[]
            for s,dd in zip(ids,distance):
                available=[i for i,t in enumerate(model['ids']) if t!=s and features['train'][t]['image_sha256']!=features[split][s]['image_sha256']]
                ix=sorted(available,key=lambda i:(dd[i],model['ids'][i]))[:recipe['k']]
                w=1/np.maximum(dd[ix],1e-6);w=w/w.sum();estimates.append((w[:,None]*model['y'][ix]).sum(axis=0))
            yy=np.asarray(estimates)
        else:
            estimator,mean,scale=model;names=plan['feature_schemas'][family.rsplit('_',1)[1]]
            yy=estimator.predict((matrix(split,ids,names)-mean)/scale)
        return {s:float(2*p[0]-ref*p[1]) if features[split][s]['eligible'] else None for s,p in zip(ids,yy)}
    def compare(a,b):
        assert set(a)==set(b)
        for s,v in b.items():assert a[s] is None if v is None else math.isclose(a[s],v,rel_tol=1e-9,abs_tol=1e-9),(s,a[s],v)
    rank=run.old.top
    for budget,families in cv.items():
        fraction=float(budget)
        for family,data in families.items():
            for trial in data['trials']:
                recipe=trial['recipe'];visited=[]
                for fold in trial['fold_trace']:
                    tr,held=fold['training_ids'],fold['held_out_ids'];assert set(tr+held)==set(fit) and not set(tr)&set(held)
                    assert not {features['train'][s]['image_sha256'] for s in tr}&{features['train'][s]['image_sha256'] for s in held}
                    ref=None if recipe['kind']=='control' else reference(tr)
                    assert fold['reference_f1']==ref
                    model=None if recipe['kind']=='control' else independent(tr,recipe,family)
                    compare({s:trial['oof_scores'][s] for s in held},scores(model,recipe,family,'train',held,ref))
                    visited+=held;held_count+=len(held)
                assert sorted(visited)==sorted(fit)
                check_point(trial['oof_point'],point(branches['train'],fit,rank(trial['oof_scores'],fit,fraction),cfg['quality_f1_threshold']))
            assert data['selected_trial']==max(range(len(data['trials'])),key=lambda i:(f1(data['trials'][i]['oof_point']['combined']),-i))
    for budget,choice in frozen['budgets'].items():
        trials=[];fraction=float(budget)
        for index,family in enumerate(plan['families']):
            relative='models/'+budget+'__'+family+'.json';saved=read(root/relative);assert sha(root/relative)==frozen['models_sha256'][relative]
            recipe=cv[budget][family]['trials'][cv[budget][family]['selected_trial']]['recipe'];assert saved['recipe']==recipe
            assert saved['training_ids']==([] if recipe['kind']=='control' else fit)
            model=None if recipe['kind']=='control' else independent(fit,recipe,family)
            ref=None if recipe['kind']=='control' else reference(fit)
            if recipe['kind']!='control':assert saved['reference_f1']==ref
            if recipe['kind']=='memory':
                assert saved['training_rgb_sha256']==[features['train'][s]['image_sha256'] for s in fit]
                np.testing.assert_allclose(saved['vectors'],model['vectors'],rtol=0,atol=1e-12)
                np.testing.assert_array_equal(saved['targets'],model['y'])
            for split,(origin,ids) in populations.items():compare(predictions[split][budget][family],scores(model,recipe,family,origin,ids,ref))
            ss=predictions['validation'][budget][family];accepted=rank(ss,val,fraction)
            assert choice['scalar_cuts'][family]==min((ss[s] for s in accepted),default=None)
            value=f1(point(branches['train'],val,accepted,cfg['quality_f1_threshold'])['combined'])
            assert choice['validation_f1'][family]==float(value);trials.append((value,-index,family))
        assert choice['selected_family']==max(trials)[2]
    points=0
    for split,(origin,ids) in populations.items():
        b=branches[origin];data=result['splits'][split];eligible=[s for s in ids if features[origin][s]['eligible']]
        assert data['light_only']==aggregate(b[s]['light'] for s in ids) and data['deepsolo_only']==aggregate(b[s]['deepsolo'] for s in ids)
        refs={}
        for budget,value in data['budgets'].items():
            fraction=float(budget);k=round(len(ids)*fraction)
            if k not in refs:refs[k]=optimum(b,ids,eligible,k)
            assert value['oracle']==refs[k]
            for kind in ('families','scalar_thresholds'):
                for family,actual in value[kind].items():
                    ss=predictions[split][budget][family];accepted=rank(ss,ids,fraction)
                    if kind=='scalar_thresholds':
                        cut=frozen['budgets'][budget]['scalar_cuts'][family];accepted=[s for s in ids if cut is not None and ss[s] is not None and ss[s]>=cut]
                    check_point(actual,point(b,ids,accepted,cfg['quality_f1_threshold']))
                    n=len(accepted)
                    if n not in refs:refs[n]=optimum(b,ids,eligible,n)
                    assert actual['oracle_reference_f1']==refs[n]['f1'];points+=1
            assert value['selected']==value['families'][frozen['budgets'][budget]['selected_family']]
    assert not frozen['test_labels_used']
    if plan['result_class']!='SMOKE_TEST':assert (len(fit),len(val),len(features['test']))==(80,45,300)
    out={'status':'PASS','plan_sha256':ph,'results_sha256':sha(root/'results.json'),'audit_code_sha256':sha(__file__),
        'evidence_counts':counts,'routing_points_verified':points,'held_out_predictions_replayed':held_count,
        'independent_fit_calls':refits,'test_labels_used_for_fitting_or_selection':False,'base_models_frozen':True,
        'reused_neural_inputs_verified':bool(reuse),'cache_fix_preserves_all_models_choices_and_numerical_results':bool(reuse)}
    seal(root/'independent_audit.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run.resolve())
