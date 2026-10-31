"""Independent fitting, validation-budget and complete ICDAR TEST route audit."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import joblib
import numpy as np

from scripts.icdar_budget_data import read,sha

FIELDS=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')


def counts(branches,ids,keep):
    keep=set(keep)
    assert keep<=set(ids)
    assert all(branches[s]['light']['gt_words']==branches[s]['deepsolo']['gt_words'] for s in ids)
    c={k:sum(branches[s]['light' if s in keep else 'deepsolo'][k] for s in ids) for k in FIELDS}
    assert c['tp']+c['fp']==c['detected_words'] and c['tp']+c['fn']==c['gt_words']
    assert c['detected_words']+c['ignored_outputs']==c['all_outputs']
    return dict(c,f1=2*c['tp']/(c['detected_words']+c['gt_words']),
        accuracy_tn0=c['tp']/(c['tp']+c['fp']+c['fn']))


def verify_route(branches,ids,keep,point):
    assert sorted(keep)==point['accepted_ids']
    assert len(keep)==point['accepted_images']
    assert len(ids)-len(keep)==point['deepsolo_images'] and len(ids)==point['total_images']
    assert point['deepsolo_percentage']==100*(len(ids)-len(keep))/len(ids)
    c=counts(branches,ids,keep)
    for k,v in c.items():assert v==point['combined'][k],(k,v,point['combined'][k])


def ranked(values,ids,q,keys):
    return sorted((s for s in ids if values[s] is not None),key=lambda s:(-values[s],keys[s],s))[:round(len(ids)*q)]


def cut_accept(values,policy,keys):
    return sorted(s for s,v in values.items() if v is not None and policy['threshold'] is not None and
        (v>policy['threshold'] or v==policy['threshold'] and keys[s]<=policy['tie_key']))


def oracle_certificate(branches,ids,eligible,accepted):
    selected=set(accepted);tp=sum(branches[s]['light' if s in selected else 'deepsolo']['tp'] for s in ids)
    den=sum(branches[s]['light' if s in selected else 'deepsolo']['detected_words']+branches[s]['light']['gt_words'] for s in ids)
    f=Fraction(2*tp,den)
    gains=[2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-f*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in eligible]
    residual=2*sum(branches[s]['deepsolo']['tp'] for s in ids)-f*sum(branches[s]['deepsolo']['detected_words']+branches[s]['light']['gt_words'] for s in ids)+sum(sorted(gains,reverse=True)[:len(accepted)])
    assert residual==0


def audit(root,recheck_observations=True):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];frozen=read(root/'frozen_choices.json')
    results=read(root/'results.json');trials=read(root/'trials.json');completion=read(root/'completion.json');checks=Counter()
    assert results['status']=='SUCCESS' and plan['mode']=='approximate'
    assert not results['test_used_for_selection'] and frozen['no_refit'] and not frozen['test_used_for_selection']
    assert frozen['plan_sha256']==sha(root/'resolved_configuration.json')
    assert results['frozen_choices_sha256']==sha(root/'frozen_choices.json') and completion['results_sha256']==sha(root/'results.json')
    for relative,digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/relative)==digest
        checks['immutable_code_files']+=1
    for path,digest in cfg['pinned_inputs'].items():
        assert sha(path)==digest
        checks['pinned_input_files']+=1
    selected=read(root/'unet_selection.json')
    records={s['sample_id']:s for data in plan['records'].values() for rows in data.values() for s in rows}
    for ds,p in plan['partitions'].items():
        populations=[{records[s]['rgb_sha256'] for s in p[role]} for role in ('fit','selection','calibration','test')]
        assert all(not populations[a]&populations[b] for a in range(4) for b in range(a+1,4))
        assert selected['selection_ids'][ds]==p['selection'] and not selected['test_used']
        winner=max(plan['candidates'][ds],key=lambda n:(selected['trials'][ds][n]['f1'],-plan['candidates'][ds].index(n)))
        assert winner==selected['choices'][ds]
        checks['expert_and_rgb_scope']+=1
        q=cfg['primary_acceptance'][ds]
        for arm,choice in frozen['choices'][ds].items():
            assert choice['primary_acceptance']==q
            assert choice['training_ids']==p['fit'] and choice['selection_ids']==p['selection'] and choice['calibration_ids']==p['calibration']
            archive=root/'training_features'/(ds+'__'+arm+'.npz');schema=read(archive.with_suffix('.json'))
            assert schema['npz_sha256']==sha(archive)
            with np.load(archive,allow_pickle=False) as z:arrays={name:z[name] for name in z.files}
            ids=arrays['ids'].tolist();assert ids==p['fit']+p['selection']+p['calibration']
            positions={s:i for i,s in enumerate(ids)}
            evidence=read(root/'train_branches'/(ds+'__'+arm+'.json'));branches=evidence['branches'];keys=evidence['tie_keys']
            assert set(branches)==set(ids) and all(keys[s]==records[s]['rgb_sha256'] for s in ids)
            reference=counts(branches,p['fit'],[])['f1'];assert reference==schema['actual_reference_f1']
            ix=[positions[s] for s in p['fit']];weights=arrays['word_scale'][ix]
            y=np.array([2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-reference*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in p['fit']])/weights
            def matrix(part,groups):return np.concatenate([arrays[g][[positions[s] for s in part]] for g in groups],axis=1)
            def scores(bundle,part):
                values=bundle['regressor'].predict(bundle['scaler'].transform(matrix(part,bundle['groups'])))
                return {s:float(v)*arrays['word_scale'][positions[s]] if arrays['eligible'][positions[s]] else None for s,v in zip(part,values)}
            for trial in trials[ds][arm]:
                path=root/trial['model_path'];assert sha(path)==trial['model_sha256']
                b=joblib.load(path);assert b['training_ids']==p['fit'] and b['reference_f1']==reference
                x=matrix(p['fit'],b['groups']);mean=x.mean(0);variance=x.var(0);eps=np.finfo(float).eps
                constant=variance<=len(x)*eps*variance+(len(x)*mean*eps)**2
                scale=np.sqrt(variance);scale[constant]=1
                assert np.allclose(b['scaler'].mean_,mean,atol=1e-12,rtol=0)
                assert np.allclose(b['scaler'].scale_,scale,atol=1e-10,rtol=0)
                assert np.allclose(b['scaler'].var_,variance,atol=1e-12,rtol=1e-8)
                z=b['scaler'].transform(x);recipe=trial['recipe']
                if recipe['kind']=='ridge':
                    mu=np.average(z,axis=0,weights=weights);my=np.average(y,weights=weights);center=z-mu
                    coef=np.linalg.solve((center*weights[:,None]).T@center+recipe['alpha']*np.eye(z.shape[1]),(center*weights[:,None]).T@(y-my))
                    assert np.allclose(coef,b['regressor'].coef_,atol=1e-8,rtol=1e-7)
                    assert abs(my-mu@coef-b['regressor'].intercept_)<1e-8
                    checks['independent_ridge_normal_equations']+=1
                else:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    r=HistGradientBoostingRegressor(max_iter=recipe['iterations'],max_leaf_nodes=recipe['leaves'],l2_regularization=recipe['l2'],min_samples_leaf=recipe['minimum_leaf'],early_stopping=False,random_state=cfg['seed']).fit(z,y,sample_weight=weights)
                    assert np.allclose(r.predict(z),b['regressor'].predict(z),atol=1e-12,rtol=0)
                    assert abs(float(r._baseline_prediction[0,0])-np.average(y,weights=weights))<1e-12
                    checks['independent_boosted_tree_replay']+=1
                values=scores(b,p['selection']);verify_route(branches,p['selection'],ranked(values,p['selection'],q,keys),trial['selection'])
                checks['candidate_scopes_and_selection_routes']+=1
            options=trials[ds][arm];best=max(t['selection']['combined']['f1'] for t in options)
            expected=min([t for t in options if t['selection']['combined']['f1']>=best-cfg['selection_tolerance']],key=lambda t:(t['scalar_count'],t['serialized_bytes'],-t['selection']['combined']['f1'],t['name']))
            assert expected==choice['choice']
            b=joblib.load(root/expected['model_path']);values=scores(b,p['calibration'])
            assert values==choice['calibration_scores']
            for policy in choice['policies']:
                assert policy['model_sha256']==expected['model_sha256'] and policy['selected_on']=='VALIDATION_CALIBRATION'
                keep=ranked(values,p['calibration'],policy['target_acceptance'],keys);boundary=keep[-1]
                assert policy['threshold']==values[boundary] and policy['tie_key']==keys[boundary]
                verify_route(branches,p['calibration'],cut_accept(values,policy,keys),policy['calibration']);checks['calibration_cuts']+=1
            report=results['datasets'][ds][arm];e=read(root/'test_scores'/(ds+'__'+arm+'.json'));ts=e['scores'];tb=e['branches'];tk=e['tie_keys']
            assert set(ts)==set(p['test']) and all(tk[s]==records[s]['rgb_sha256'] for s in p['test'])
            if plan['result_class']!='smoke-test result':
                assert len(p['test'])==cfg['expected_test_images'][ds]
                assert counts(tb,p['test'],[])['gt_words']==cfg['expected_test_gt'][ds]
            if recheck_observations:
                # Rebuild observation arrays and branch counts from the frozen run's
                # original expert JSON/NPZ and annotations, outside saved head arrays.
                from scripts.icdar_budget_stages import collect
                for part in (ids,p['test']):
                    features,actual,actual_keys=collect(plan,root,ds,choice['expert'],part)
                    assert actual==(branches if part is ids else tb) and actual_keys==(keys if part is ids else tk)
                    if part is ids:
                        for s in part:
                            assert features[s]['scale']==arrays['word_scale'][positions[s]]
                            assert features[s]['eligible']==arrays['eligible'][positions[s]]
                            for g in schema['groups']:assert np.array_equal(features[s]['groups'][g],arrays[g][positions[s]])
                    else:
                        x_test=np.stack([np.concatenate([features[s]['groups'][g] for g in b['groups']]) for s in part])
                        predicted=b['regressor'].predict(b['scaler'].transform(x_test))
                        recomputed={s:float(v)*features[s]['scale'] if features[s]['eligible'] else None for s,v in zip(part,predicted)}
                        assert recomputed==ts
                    checks['rebuilt_expert_observations']+=len(part)
            for point in report['thresholds']:
                verify_route(tb,p['test'],cut_accept(ts,point['policy'],tk),point);checks['complete_test_routes']+=1
            for quota,point in report['predeclared_batch_quotas'].items():
                keep=ranked(ts,p['test'],float(quota),tk);verify_route(tb,p['test'],keep,point)
                upper=report['conditional_gt_oracles'][quota];verify_route(tb,p['test'],upper['accepted_ids'],upper)
                assert len(upper['accepted_ids'])==len(keep)
                oracle_certificate(tb,p['test'],[s for s in p['test'] if ts[s] is not None],upper['accepted_ids']);checks['independent_fractional_oracle_certificates']+=1
            verify_route(tb,p['test'],p['test'],report['all_light']);verify_route(tb,p['test'],[],report['all_deepsolo'])
    if 'replacement_fallback_counts' in cfg:
        for dataset,spec in cfg['replacement_fallback_counts'].items():
            assert sha(spec['path'])==spec['sha256']
            archive=read(spec['path'])
            expected={r['sample_id'] for rows in plan['records'][dataset].values() for r in rows}
            assert set(archive['samples'])==expected
            checks['replacement_teacher_complete_populations']+=1
        assert results['actual_new_deepsolo_calls']==plan['actual_new_deepsolo_calls']
    else:
        receipt=read(root/'fallback_train/receipt_limit_0.json')
        expected_calls=sum(len(d['train']) for d in plan['records'].values())
        assert receipt['new_model_inferences']+receipt['reused']==expected_calls==results['actual_new_deepsolo_calls']
    return dict(status='PASS',checks=dict(checks),results_sha256=sha(root/'results.json'),plan_sha256=sha(root/'resolved_configuration.json'),
        frozen_choices_sha256=sha(root/'frozen_choices.json'),analysis_code_sha256=sha(__file__),
        scope='Independent normal equations, tree refits, actual archived observations and branch counts, disjoint RGB scopes, validation selection/calibration, complete TEST denominators, exact-cardinality oracle certificates. Cached routing; no online latency claim.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.run_root)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
