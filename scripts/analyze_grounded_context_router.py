"""Audit grounded probes, independent model refits and full-word routing metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from recognition.grounded_context_router import feature_sets, local_views, verification_jobs, verification_features
from scripts import run_grounded_context_router as runner
from scripts.analyze_trainable_image_router import independent_fit, optimum
from scripts.audit_yolo_f1_quality_study import aggregate, check_point, f1, point
from scripts.crop_clip_runtime import array_sha
from scripts.run_yolo_f1_quality_study import checked, read, seal, sha


def ranked(scores, ids, fraction):
    return sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(ids)*fraction)]


def audit(root):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];result=read(root/'results.json')
    frozen=read(root/'frozen_choices.json');cv=read(root/'cv_results.json');ph=sha(root/'resolved_configuration.json')
    assert result['status']=='SUCCESS' and result['plan_sha256']==ph and result['new_visual_model_inference']
    assert result['base_models_frozen'] and not result['original_word_outputs_changed']
    assert result['frozen_choices_sha256']==sha(root/'frozen_choices.json')
    assert frozen['cv_results_sha256']==sha(root/'cv_results.json')
    for file,digest in plan['code_sha256'].items(): assert sha(root/'code_snapshot'/file)==digest
    for file,digest in plan['input_sha256'].items(): assert sha(Path(file))==digest
    reuse=plan.get('reused_train_inventory')
    if reuse:
        source=Path(reuse['root']);original=checked(source/'resolved_configuration.json',reuse['plan_sha256'])
        receipt=checked(source/'train/inventory_receipt.json',reuse['receipt_sha256'])
        assert receipt['model']['model_state_sha256_before']==receipt['model']['model_state_sha256_after']
        assert original['code_sha256']['recognition/grounded_context_runtime.py']==plan['code_sha256']['recognition/grounded_context_runtime.py']
        for key in ('inventory_settings','inventory_identity','execution'): assert original['config'][key]==cfg[key]
        for file,digest in reuse['files_sha256'].items():
            old=checked(source/file,digest);new=read(root/file)
            assert new['reused_source_sha256']==digest
            assert {k:v for k,v in new.items() if k not in ('plan_sha256','reused_source_sha256')}=={k:v for k,v in old.items() if k!='plan_sha256'}
    branches=read(Path(plan['source_coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']
    from transformers import AutoProcessor
    from recognition.grounded_context_runtime import florence_parser_input
    processor=AutoProcessor.from_pretrained(plan['paths']['inventory_snapshot'],local_files_only=True,trust_remote_code=True)
    features={};probe_count=0;inventory_count=0;region_count=0
    for split,samples in plan['samples'].items():
        features[split]={}
        binding=sha(root/'frozen_choices.json') if split=='test' else None
        for stage in ('inventory','verification'):
            receipt=read(root/split/(stage+'_receipt.json'));model=receipt['model']
            assert receipt['plan_sha256']==ph and receipt['images']==len(samples)
            assert model['model_state_sha256_before']==model['model_state_sha256_after'] and model['optimizer_steps']==0
            assert model['precision']=='float32' and model['device']==cfg['execution']['device']
            assert model['cuda_peak']['peak_allocated_bytes']>0 and model['cuda_peak']['peak_reserved_bytes']>0
            if stage=='verification': assert model['native_forward_check']['status']=='PASS'
        for sample in samples:
            sid=sample['sample_id'];name=sid.replace(':','__')+'.json'
            value=read(root/split/'features'/name);inv=read(root/split/'inventory'/name);local=read(root/split/'verification'/name)
            for record in (value,inv,local):
                assert record['sample_id']==sid and record['image_sha256']==sample['image_sha256'] and record['plan_sha256']==ph
                assert record['frozen_choices_sha256']==binding
            assert not value['per_image_ground_truth_used'] and not value['actual_deepsolo_outputs_used']
            assert not inv['ground_truth_used'] and not inv['ocr_predictions_given_to_model']
            imodel=read(root/split/'inventory_receipt.json')['model']
            for task,output in inv['tasks'].items():
                tokens=np.array([output['token_ids']],dtype=np.int64)
                assert processor.batch_decode(tokens,skip_special_tokens=False)[0]==output['raw']
                clean=florence_parser_input(processor,tokens,imodel['decoder_start_token_id'])
                assert clean==output['parser_input']
                assert processor.post_process_generation(clean,task=task,image_size=tuple(sample['image_wh']))[task]==output['parsed']
                assert output['ended_with_eos']==(output['token_ids'][-1] in imodel['decoder_eos_token_ids'])
            rows=read(Path(plan['source_coverage_root'])/'640'/split/'features'/name)['rows']
            parts={k:read(Path(plan['paths'][k+'_root'])/split/'features'/name) for k in ('residual','quality')}
            observed={'sample':sample,'rows':rows,'parts':parts}
            fields,evidence=runner.derived_inventory(cfg,observed,inv)
            assert evidence==value['inventory_evidence']
            expected_jobs=verification_jobs(rows,parts['residual']['word_evidence']['words'],evidence,sample['image_wh'],cfg['geometry_settings'])
            assert len(local['probes'])==len(expected_jobs)<=cfg['geometry_settings']['maximum_verified_words']
            image=runner.rgb(sample);assert array_sha(image)==inv['rgb_sha256']
            for job,probe in zip(expected_jobs,local['probes']):
                assert {k:probe[k] for k in job}==job and not probe['ground_truth_used']
                views=local_views(image,job,cfg['geometry_settings']['mask_value'])
                assert probe['views_rgb_sha256']=={k:array_sha(v) for k,v in views.items()}
                for answers in probe['answers'].values():
                    assert len(answers)==len(job['alternatives'])
                    for answer,word in zip(answers,job['alternatives']):
                        assert answer['prompt']==cfg['verification_question'].format(word=json.dumps(word,ensure_ascii=False))
                        yes,no=answer['yes_logprob'],answer['no_logprob'];m=max(yes,no)
                        expected=math.exp(yes-m)/(math.exp(yes-m)+math.exp(no-m))
                        assert math.isclose(answer['yes_probability'],expected,rel_tol=2e-6,abs_tol=2e-7)
                probe_count+=1
            base,schema,eligible=runner.source_features(cfg,observed)
            vector,names=feature_sets(base,schema,fields,verification_features(local['probes'],len(rows)),inv['embedding'])
            assert names==plan['feature_schemas'] and vector==value['values'] and eligible==value['eligible']
            for stage in ('inventory','verification'): assert value['source_probe_sha256'][stage]==sha(root/split/stage/name)
            features[split][sid]=value;inventory_count+=1;region_count+=len(evidence['ocr_regions'])
            if split=='test':
                assert (root/'frozen_choices.json').stat().st_mtime_ns <= (root/split/'inventory'/name).stat().st_mtime_ns
    fit_ids,val_ids=plan['fit_ids'],plan['validation_ids']
    assert not set(fit_ids)&set(val_ids) and set(fit_ids+val_ids)==set(features['train'])
    assert not set(features['train'])&set(features['test']) and not frozen['test_labels_used']
    predictions={s:read(root/'predictions'/(s+'.json')) for s in result['splits']}
    populations={'fit':('train',fit_ids),'validation':('train',val_ids),'train':('train',sorted(features['train'])),
                 'test':('test',sorted(features['test']))}
    cache={};refits=0;held_verified=0
    def matrix(split,ids,names): return np.array([[features[split][s]['values'][n] for n in names] for s in ids])
    def refit(ids,recipe,names):
        nonlocal refits
        assert set(ids)<=set(fit_ids)
        key=(tuple(ids),json.dumps(recipe,sort_keys=True),tuple(names))
        if key not in cache:
            y=np.array([[branches['train'][s]['light']['tp']-branches['train'][s]['deepsolo']['tp'],
                branches['train'][s]['light']['detected_words']-branches['train'][s]['deepsolo']['detected_words']] for s in ids])
            cache[key]=independent_fit(matrix('train',ids,names),y,recipe,cfg);refits+=1
        return cache[key]
    def scores(model,split,ids,names,reference):
        estimator,mean,scale=model;y=estimator.predict((matrix(split,ids,names)-mean)/scale)
        return {s:float(2*p[0]-reference*p[1]) if features[split][s]['eligible'] else None for s,p in zip(ids,y)}
    def compare(actual,expected):
        assert set(actual)==set(expected)
        for s,v in expected.items(): assert actual[s] is None if v is None else math.isclose(actual[s],v,rel_tol=1e-10,abs_tol=1e-10)
    for budget,families in cv.items():
        fraction=float(budget)
        for family,data in families.items():
            names=plan['feature_schemas'][family.rsplit('_',1)[1]]
            for trial in data['trials']:
                visited=[]
                for fold in trial['fold_trace']:
                    training,held=fold['training_ids'],fold['held_out_ids'];assert not set(training)&set(held)
                    assert set(training+held)==set(fit_ids) and fold['teacher_oracle'] is None
                    reference=float(f1(aggregate(branches['train'][s]['deepsolo'] for s in training)))
                    assert reference==fold['reference_f1']
                    compare({s:trial['oof_scores'][s] for s in held},scores(refit(training,trial['recipe'],names),'train',held,names,reference))
                    visited+=held;held_verified+=len(held)
                assert sorted(visited)==sorted(fit_ids)
                check_point(trial['oof_point'],point(branches['train'],fit_ids,ranked(trial['oof_scores'],fit_ids,fraction),cfg['quality_f1_threshold']))
            assert data['selected_trial']==max(range(len(data['trials'])),key=lambda i:(f1(data['trials'][i]['oof_point']['combined']),-i))
    for budget,choices in frozen['budgets'].items():
        val_trials=[];fraction=float(budget)
        for index,family in enumerate(plan['families']):
            relative='models/'+budget+'__'+family+'.json';saved=checked(root/relative,frozen['models_sha256'][relative])
            assert saved['training_ids']==fit_ids
            names=plan['feature_schemas'][family.rsplit('_',1)[1]];assert names==saved['feature_names']
            trial=cv[budget][family]['trials'][cv[budget][family]['selected_trial']];assert saved['recipe']==trial['recipe']
            model=refit(fit_ids,saved['recipe'],names);_,mean,scale=model
            np.testing.assert_allclose(saved['mean'],mean,rtol=0,atol=1e-12);np.testing.assert_allclose(saved['scale'],scale,rtol=0,atol=1e-12)
            reference=float(f1(aggregate(branches['train'][s]['deepsolo'] for s in fit_ids)));assert saved['reference_f1']==reference
            for split,(origin,ids) in populations.items(): compare(predictions[split][budget][family],scores(model,origin,ids,names,reference))
            vs=predictions['validation'][budget][family];accepted=ranked(vs,val_ids,fraction)
            assert choices['scalar_cuts'][family]==min((vs[s] for s in accepted),default=None)
            vf1=f1(point(branches['train'],val_ids,accepted,cfg['quality_f1_threshold'])['combined'])
            assert choices['validation_f1'][family]==float(vf1);val_trials.append((vf1,-index,family))
        assert choices['selected_family']==max(val_trials)[2]
    points=0
    for split,(origin,ids) in populations.items():
        data=result['splits'][split];b=branches[origin];eligible=[s for s in ids if features[origin][s]['eligible']]
        assert data['light_only']==aggregate(b[s]['light'] for s in ids) and data['deepsolo_only']==aggregate(b[s]['deepsolo'] for s in ids)
        for budget,value in data['budgets'].items():
            fraction=float(budget);reference=optimum(b,ids,eligible,round(len(ids)*fraction))
            assert {k:value['oracle'][k] for k in reference}==reference
            for kind in ('families','controls','scalar_thresholds'):
                for family,actual in value[kind].items():
                    ss={s:features[origin][s]['controls'][family] for s in ids} if kind=='controls' else predictions[split][budget][family]
                    accepted=ranked(ss,ids,fraction)
                    if kind=='scalar_thresholds':
                        cut=frozen['budgets'][budget]['scalar_cuts'][family]
                        accepted=[s for s in ids if cut is not None and ss[s] is not None and ss[s]>=cut]
                    check_point(actual,point(b,ids,accepted,cfg['quality_f1_threshold']));points+=1
            assert value['selected']==value['families'][frozen['budgets'][budget]['selected_family']]
            assert value['primary']==value['families'][cfg['primary_family']]
    if plan['result_class']!='SMOKE_TEST':
        assert (len(fit_ids),len(val_ids),len(features['test']))==(80,45,300)
        assert result['splits']['train']['light_only']['gt_words']==983 and result['splits']['test']['light_only']['gt_words']==2204
    report={'status':'PASS','plan_sha256':ph,'results_sha256':sha(root/'results.json'),'audit_code_sha256':sha(__file__),
        'features_verified':inventory_count,'independent_regions_verified':region_count,'local_probes_verified':probe_count,
        'routing_points_verified':points,'held_out_predictions_replayed':held_verified,'independent_fit_calls':refits,
        'frozen_before_test':True,'base_models_frozen':True,'test_labels_used_for_selection':False}
    seal(root/'independent_audit.json',report)
    export(root,plan,result)
    print(json.dumps(report),flush=True)


def export(root,plan,result):
    rows=[];lines=['# Grounded context router: approximate reproduction','','Exploratory local word multiset F1, not paper recognition accuracy.','',
        '|Bypass|VAL-selected family|TRAIN125 F1|TEST F1|All DeepSolo TEST F1|Eligible oracle TEST F1|',
        '|---|---|---:|---:|---:|---:|']
    for budget,data in result['splits']['test']['budgets'].items():
        chosen=data['selected_family'];train=result['splits']['train']['budgets'][budget]['families'][chosen]
        lines.append(f'|{100*float(budget):.0f}%|{chosen}|{100*train["combined"]["f1"]:.3f}%|{100*data["selected"]["combined"]["f1"]:.3f}%|{100*result["splits"]["test"]["deepsolo_only"]["f1"]:.3f}%|{100*data["oracle"]["f1"]:.3f}%|')
    for split,data in result['splits'].items():
        for budget,value in data['budgets'].items():
            for kind in ('families','controls','scalar_thresholds'):
                for family,p in value[kind].items():
                    rows.append({'split':split,'budget':budget,'kind':kind,'family':family,'validation_selected':family==value['selected_family'],
                        'accepted':p['accepted_images'],'good80':p['good_images'],**{k:p['combined'][k] for k in ('tp','fp','fn','detected_words','gt_words','f1','accuracy_tn0')}})
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True)
    audit(parser.parse_args().run.resolve())
