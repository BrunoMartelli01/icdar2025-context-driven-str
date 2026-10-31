"""Independent word, partition, head-fit and exact-budget replay checks."""

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor

from recognition.lexical_budget_router import features,predict
from scripts.run_lexical_budget_router import lexicon,corrected
from scripts.icdar_budget_data import read,sha
from scripts.run_yolo_f1_quality_study import seal


FIELDS=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')


def recount(rows,flags,gt):
    norm=lambda s:''.join(c for c in s.casefold() if c.isalnum())
    words=[norm(r['raw_text']) for r in rows if not flags[r['source_row']]];truth=[norm(g['text']) for g in gt]
    a,b=Counter(s for s in words if s),Counter(s for s in truth if s);tp=sum((a&b).values())
    return dict(tp=tp,fp=len(words)-tp,fn=len(truth)-tp,detected_words=len(words),gt_words=len(truth),all_outputs=len(rows),ignored_outputs=len(rows)-len(words))


def route_check(branches,accepted,p):
    accepted=set(accepted)
    for k in FIELDS:assert sum(b['light' if s in accepted else 'deepsolo'][k] for s,b in branches.items())==p[k]
    assert p['f1']==2*p['tp']/(p['detected_words']+p['gt_words'])
    assert p['deepsolo_images']==len(branches)-len(accepted) and p['total_images']==len(branches)
    assert p['deepsolo_percent']==100*(len(branches)-len(accepted))/len(branches)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);plan=read(Path(c['original_router_root'])/'resolved_configuration.json')
    frozen=read(root/'frozen.json');policies=read(root/'correction_frozen.json');grids=read(root/'correction_grid.json');trials=read(root/'trials.json');results=read(root/'results.json')
    assert not frozen['test_used_for_selection'] and not policies['test_used_for_selection'] and frozen['no_refit'];assert results['frozen_sha256']==sha(root/'frozen.json')
    counts=Counter();maxerror=0.
    for ds,t in c['targets'].items():
        part=plan['partitions'][ds]
        for arm in c.get('experts',('unet','yolo')):
            scene=read(root/'prepared'/(ds+'__'+arm+'.json'));data=read(root/'branches'/(ds+'__'+arm+'.json'));branches=data['branches'];keys=data['tie_keys'];chosen=policies['choices'][ds][arm]['policy'];lexicons={s:lexicon(c,ds,s) for s in ('train','test')}
            for path,h in scene['input_sha256'].items():assert sha(path)==h
            for point in grids[ds][arm]:
                total=Counter();changes=0
                for sid in part['selection']:
                    raw=scene['scenes'][sid];rows,trace,_=corrected(raw,lexicons['train'],point['policy']);total.update(recount(rows,raw['ignored'],raw['ground_truth']));changes+=sum(p['changed'] or not p['accepted'] for p in trace)
                for k in FIELDS:assert total[k]==point['metrics'][k]
                assert changes==point['changes'];counts['correction_grid_points']+=1
            best=max(grids[ds][arm],key=lambda x:(2*x['metrics']['tp']/(x['metrics']['detected_words']+x['metrics']['gt_words']),-x['changes'],not x['policy']['enabled']))
            assert best==policies['choices'][ds][arm]
            with np.load(root/'models'/(ds+'__'+arm+'__features.npz'),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            ids=arrays['ids'].tolist();positions={s:i for i,s in enumerate(ids)}
            for sid,raw in scene['scenes'].items():
                rows,trace,_=corrected(raw,lexicons['train' if ':train:' in sid else 'test'],chosen);value=recount(rows,raw['ignored'],raw['ground_truth']);assert value==branches[sid]['light']
                observed=features(raw['rows'],rows,trace,raw['image_wh'],use_consensus=c.get('consensus_features',False))
                if c.get('extra_feature_archive'):observed['word_risk']=np.asarray(read(c['extra_feature_archive'])['values'][sid],np.float64)
                for g,v in observed.items():assert np.array_equal(v,arrays[g][positions[sid]])
                assert arrays['scale'][positions[sid]]==len(raw['rows'])+1;counts['observation_replays']+=1
            fitids=part['fit'];ix=[positions[s] for s in fitids];weight=arrays['scale'][ix]
            y=np.array([2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-t['f1']*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in fitids])/weight
            def matrix(ids,groups):return np.concatenate([arrays[g][[positions[s] for s in ids]] for g in groups],axis=1)
            def scores(bundle,ids):return dict(zip(ids,map(float,predict(bundle,matrix(ids,bundle['groups']),arrays['scale'][[positions[s] for s in ids]]))))
            acceptsel=len(part['selection'])-round(len(part['selection'])*t['fallback_images']/t['test_images'])
            for trial in trials[ds][arm]:
                assert sha(trial['path'])==trial['sha256'];bundle=joblib.load(trial['path']);assert bundle['fit_ids']==fitids
                x=matrix(fitids,bundle['groups']);assert np.allclose(bundle['scaler'].mean_,x.mean(0),atol=1e-12,rtol=1e-12)
                z=bundle['scaler'].transform(x);recipe=bundle['recipe']
                if recipe['kind']=='ridge':
                    aug=np.c_[z,np.ones(len(z))];penalty=np.diag([recipe['alpha']]*z.shape[1]+[0]);beta=np.r_[bundle['model'].coef_,bundle['model'].intercept_]
                    residual=(aug.T@(weight[:,None]*aug)+penalty)@beta-aug.T@(weight*y);assert np.max(np.abs(residual))<1e-7;counts['ridge_normal_equations']+=1
                else:
                    if recipe['kind']=='hgb':model=HistGradientBoostingRegressor(max_iter=recipe['iterations'],max_leaf_nodes=recipe['leaves'],l2_regularization=recipe['l2'],min_samples_leaf=recipe['minimum_leaf'],early_stopping=False,random_state=c['seed'])
                    else:model=ExtraTreesRegressor(n_estimators=recipe['trees'],max_depth=recipe['depth'],min_samples_leaf=recipe['minimum_leaf'],random_state=c['seed'],n_jobs=1)
                    model.fit(z,y,sample_weight=weight);assert np.array_equal(model.predict(z),bundle['model'].predict(z));counts['tree_refits']+=1
                values=scores(bundle,part['selection']);keep=sorted(values,key=lambda s:(-values[s],keys[s],s))[:acceptsel]
                route_check({s:branches[s] for s in values},keep,trial['selection']);counts['candidate_selection_routes']+=1
            options=trials[ds][arm];maxf=max(x['selection']['f1'] for x in options);winner=min((x for x in options if x['selection']['f1']>=maxf-c['selection_tolerance']),key=lambda x:(x['bytes'],-x['selection']['f1']))
            choice=frozen['choices'][ds][arm];assert choice['winner']==winner;bundle=joblib.load(winner['path']);cal=scores(bundle,part['calibration']);assert cal==choice['calibration_scores']
            order=sorted(cal,key=lambda s:(-cal[s],keys[s],s));k=len(order)-round(len(order)*t['fallback_images']/t['test_images']);boundary=order[k-1];cut=choice['calibration_cut'];assert cut['threshold']==cal[boundary] and cut['tie_key']==keys[boundary]
            ids=part['test'];values=scores(bundle,ids);saved=read(root/'test_scores'/(ds+'__'+arm+'.json'));assert values==saved['scores']
            selected=sorted(values,key=lambda s:(-values[s],keys[s],s))[:len(ids)-t['fallback_images']];result=results['datasets'][ds][arm];bb={s:branches[s] for s in ids};route_check(bb,selected,result['primary']);assert result['primary']['deepsolo_images']==t['fallback_images']
            keep=[s for s in ids if values[s]>cut['threshold'] or values[s]==cut['threshold'] and keys[s]<=cut['tie_key']];route_check(bb,keep,result['validation_cut']);counts['test_routes']+=2
            print(ds,arm,'audit PASS',flush=True)
    seal(root/'independent_audit.json',dict(status='PASS',checks=dict(counts),results_sha256=sha(root/'results.json'),code_sha256=sha(__file__),scope='Every corrected word count and archived descriptor, all correction-grid choices and112 head fits, fit-only normalization/targets, disjoint role IDs, calibration and exact TEST budgets'))
    print(dict(counts),flush=True)


if __name__=='__main__':main()
