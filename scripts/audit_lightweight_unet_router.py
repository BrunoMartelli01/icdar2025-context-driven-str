"""Independent small-gate fitting-scope, selection, calibration and count replay."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def counts(branches,ids,keep):
    total=Counter();keep=set(keep)
    for sid in ids:total.update(branches[sid]['light' if sid in keep else 'deepsolo'])
    return dict(total,f1=2*total['tp']/(2*total['tp']+total['fp']+total['fn']),
                deepsolo_images=len(ids)-len(keep),total_images=len(ids),
                deepsolo_percent=100*(len(ids)-len(keep))/len(ids),accepted_images=len(keep),accepted_ids=sorted(keep))


def assert_counts(actual,saved):
    for key in ('tp','fp','fn','detected_words','gt_words','f1','deepsolo_images','total_images','deepsolo_percent','accepted_ids'):
        assert actual[key]==saved[key],key


def audit(root):
    plan=read(root/'resolved_configuration.json');c=plan['source_config'];frozen=read(root/'frozen_choices.json');trials=read(root/'trials.json')
    result=read(root/'results.json');receipt=read(root/'completion.json');checks=Counter()
    assert plan['mode']=='approximate' and not frozen['test_used'] and not frozen['refit_after_calibration']
    assert receipt['results_sha256']==sha(root/'results.json') and receipt['frozen_choices_sha256']==sha(root/'frozen_choices.json')
    assert frozen['trials_sha256']==sha(root/'trials.json')
    for path,digest in c['input_sha256'].items():assert sha(path)==digest;checks['source_files']+=1
    arrays={};schemas={};branches={};rgb=set()
    for role,records in plan['records'].items():
        for record in records:
            value=record['token_sample']['rgb_sha256'];assert value not in rgb;rgb.add(value)
        schemas[role]=read(root/(role+'_feature_schema.json'));ids=sorted(r['sample_id'] for r in records)
        assert ids==schemas[role]['ids'];assert sha(root/(role+'_features.npz'))==schemas[role]['arrays_sha256']
        with np.load(root/(role+'_features.npz'),allow_pickle=False) as z:arrays[role]={k:z[k] for k in z.files}
        branches[role]=read(root/(role+'_branches.json'));assert set(branches[role])==set(ids)
        assert not any('label' in name or 'ground_truth' in name for name in arrays[role])
        checks['role_feature_populations']+=1
    fit,select,cal,test=[schemas[r]['ids'] for r in ['fit','selection','calibration','test']]
    assert fit==frozen['fit_ids'] and select==frozen['selection_ids'] and cal==frozen['calibration_ids']
    projections=joblib.load(root/'fit_only_projections.joblib');assert sha(root/'fit_only_projections.joblib')==frozen['projections_sha256']
    for name,p in projections.items():
        assert p.n_samples_==len(fit) and np.allclose(p.mean_,arrays['fit'][name].mean(0),rtol=0,atol=1e-12)
        checks['fit_only_projections']+=1
    def matrix(role,groups):
        parts=[]
        for group in groups:
            x=arrays[role][group];parts.append(projections[group].transform(x) if group in projections else x)
        return np.concatenate(parts,axis=1)
    def scores(model,role):
        predicted=model['regressor'].predict(model['scaler'].transform(matrix(role,model['groups'])))
        ids=schemas[role]['ids'];normalized={s:float(v) if schemas[role]['eligible'][s] else None for s,v in zip(ids,predicted)}
        return normalized,{s:v*schemas[role]['word_scale'][s] if v is not None else None for s,v in normalized.items()},predicted
    reference=counts(branches['fit'],fit,[])['f1'];assert reference==frozen['reference_f1']
    weights=np.asarray([schemas['fit']['word_scale'][s] for s in fit]);y=np.asarray([2*(branches['fit'][s]['light']['tp']-branches['fit'][s]['deepsolo']['tp'])-reference*(branches['fit'][s]['light']['detected_words']-branches['fit'][s]['deepsolo']['detected_words']) for s in fit])/weights
    for trial in trials:
        path=root/trial['model_path'];assert sha(path)==trial['model_sha256'];model=joblib.load(path)
        assert model['training_ids']==fit and model['plan_sha256']==sha(root/'resolved_configuration.json')
        x=matrix('fit',model['groups']);assert np.allclose(model['scaler'].mean_,x.mean(0),rtol=0,atol=1e-12)
        if trial['head']['kind']=='ridge':
            z=model['scaler'].transform(x);mu=np.average(z,axis=0,weights=weights);my=np.average(y,weights=weights);center=z-mu
            coefficient=np.linalg.solve((center*weights[:,None]).T@center+trial['head']['alpha']*np.eye(z.shape[1]),(center*weights[:,None]).T@(y-my))
            assert np.allclose(coefficient,model['regressor'].coef_,rtol=1e-7,atol=1e-9)
            assert abs(my-mu@coefficient-model['regressor'].intercept_)<1e-9
            checks['independent_weighted_ridge_solutions']+=1
        else:
            assert model['regressor'].n_iter_==trial['head']['iterations']
            assert abs(float(model['regressor']._baseline_prediction.ravel()[0])-np.average(y,weights=weights))<1e-12
            checks['boosted_tree_initialization_and_budget']+=1
        _,values,_=scores(model,'selection');points=[]
        for p in trial['selection_points']:
            order=sorted((s for s in select if values[s] is not None),key=lambda s:(-values[s],s));keep=order[:round(len(select)*p['fraction'])]
            actual=counts(branches['selection'],select,keep);assert_counts(actual,p);points.append(actual['f1']);checks['selection_route_points']+=1
        assert float(np.mean(points))==trial['selection_mean_f1'];checks['fitting_scopes']+=1
    winners={}
    for name in c['feature_sets']:
        indices=[i for i,t in enumerate(trials) if t['name']==name]
        winners[name]=max(indices,key=lambda i:(trials[i]['selection_mean_f1'],-trials[i]['stored_scalar_count'],-i))
    assert winners==frozen['winner_indices']
    best=max(trials[i]['selection_mean_f1'] for i in winners.values());eligible=[i for i in winners.values() if trials[i]['selection_mean_f1']>=best-c['selection_f1_tolerance']]
    selected=min(eligible,key=lambda i:(trials[i]['cost']['additional_encoder_families'],trials[i]['stored_scalar_count'],trials[i]['feature_dimension'],-trials[i]['selection_mean_f1'],i))
    assert selected==frozen['selected_trial'];assert result['selected_name']==trials[selected]['name']
    saved_predictions=read(root/'predictions.json')
    for name,index in winners.items():
        model=joblib.load(root/trials[index]['model_path']);cn,cv,cp=scores(model,'calibration');tn,tv,_=scores(model,'test')
        assert tn==saved_predictions[name]['normalized'] and tv==saved_predictions[name]['absolute']
        for key,p in frozen['policies'][name].items():
            assert p['calibration_ids']==([] if key=='raw_zero' else cal)
            if key.startswith('conformal'):
                actual=np.asarray([2*(branches['calibration'][s]['light']['tp']-branches['calibration'][s]['deepsolo']['tp'])-reference*(branches['calibration'][s]['light']['detected_words']-branches['calibration'][s]['deepsolo']['detected_words']) for s in cal])/np.asarray([schemas['calibration']['word_scale'][s] for s in cal])
                k=math.ceil((len(cal)+1)*(1-p['alpha']));q=float(sorted(cp-actual)[k-1]) if k<=len(cal) else None
                assert q==p['correction'];keep=[] if q is None else [s for s in cal if cn[s] is not None and cn[s]>q]
                test_keep=[] if q is None else [s for s in test if tn[s] is not None and tn[s]>q]
                checks['conformal_corrections']+=1
            else:
                t=p['threshold'];keep=[] if t is None else [s for s in cal if cv[s] is not None and cv[s]>=t]
                test_keep=[] if t is None else [s for s in test if tv[s] is not None and tv[s]>=t]
                if key!='raw_zero':
                    base=counts(branches['calibration'],cal,[])['f1'];valid=[(0,base,None)]
                    for t2 in sorted(set(v for v in cv.values() if v is not None),reverse=True):
                        candidate=[s for s in cal if cv[s] is not None and cv[s]>=t2];score=counts(branches['calibration'],cal,candidate)['f1']
                        if score+p['maximum_f1_loss']+1e-15>=base:valid.append((len(candidate),score,t2))
                    assert max(valid,key=lambda x:x[:2])[2]==t;checks['empirical_calibration_optima']+=1
            assert_counts(counts(branches['calibration'],cal,keep),p['summary']);checks['calibration_routes']+=1
            assert_counts(counts(branches['test'],test,test_keep),result['variants'][name]['policies'][key]);checks['test_routes']+=1
        for p in result['variants'][name]['quotas']:
            order=sorted((s for s in test if tv[s] is not None),key=lambda s:(-tv[s],s));keep=order[:round(len(test)*p['fraction'])]
            assert_counts(counts(branches['test'],test,keep),p);checks['test_routes']+=1
    assert receipt['selected_deployment_sha256']==sha(root/'selected_deployment.joblib')
    return dict(status='PASS',checks=dict(checks),results_sha256=sha(root/'results.json'),analysis_code_sha256=sha(Path(__file__)),
                scope='Independent counts, selection, fit-only preprocessing, weighted ridge normal equations, tree budget, frozen-head CAL empirical optima and finite-sample conformal order statistics. No independent boosted-tree refit or untouched-dataset guarantee.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=audit(args.run)
    with args.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
