"""Independent fitting, selection, count and deployment audit of the shared gate."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from recognition.common_budget_deployment import CommonBudgetPipeline
from recognition.common_budget_pipeline import word_features,normalize


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def recount(rows,truth,ignored):
    remaining=[''.join(c for c in r['text'].casefold() if c.isalnum()) for r in truth];tp=0;n=0
    for row,skip in zip(rows,ignored):
        if skip:continue
        n+=1;text=''.join(c for c in row['raw_text'].casefold() if c.isalnum())
        if text and text in remaining:remaining.remove(text);tp+=1
    return dict(tp=tp,fp=n-tp,fn=len(truth)-tp,detected_words=n,gt_words=len(truth),all_outputs=len(rows),ignored_outputs=sum(ignored))


def totals(counts):
    rows=list(counts);v={k:sum(r[k] for r in rows) for k in ['tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs']}
    den=v['detected_words']+v['gt_words'];return dict(v,f1=2*v['tp']/den if den else 0.)


def rank(scores,keys,k,eligible):
    keep=len(scores)-k
    if len(eligible)<keep:raise ValueError('Impossible independently replayed guard')
    return sorted(eligible,key=lambda s:(-scores[s],keys[s],s))[:keep]


def paired_bootstrap(a,b,seed,draws):
    aa=np.array([[v[k] for k in ['tp','fp','fn']] for v in a]);bb=np.array([[v[k] for k in ['tp','fp','fn']] for v in b])
    rng=np.random.default_rng(seed);ix=rng.integers(len(a),size=(draws,len(a)))
    def f(v):
        x=v[ix].sum(1);return 200*x[:,0]/(2*x[:,0]+x[:,1]+x[:,2])
    return dict(delta_f1_points_95_interval=np.quantile(f(bb)-f(aa),[.025,.975]).tolist(),draws=draws,seed=seed,
        scope='Conditional paired image bootstrap; excludes training/selection variance and historical TEST exposure')


def audit_head(bundle,x,y,w,seed):
    mean=x.mean(0);variance=x.var(0);scale=np.sqrt(variance)
    # Independently reproduce StandardScaler's documented floating-point
    # constant-feature bound; exact equality to zero is insufficient for a
    # constant aspect ratio represented in binary floating point.
    eps=np.finfo(np.float64).eps
    constant=variance<=len(x)*eps*variance+(len(x)*mean*eps)**2
    scale[constant]=1
    np.testing.assert_allclose(mean,bundle['scaler'].mean_,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(scale,bundle['scaler'].scale_,rtol=1e-10,atol=1e-12)
    xx=(x-mean)/scale;r=bundle['recipe'];actual=bundle['models']
    if r['kind']=='ridge':
        xm=np.average(xx,axis=0,weights=w);ym=np.average(y,axis=0,weights=w);xc=xx-xm
        coef=np.linalg.solve((xc*w[:,None]).T@xc+r['alpha']*np.eye(x.shape[1]),(xc*w[:,None]).T@(y-ym)).T
        np.testing.assert_allclose(coef,actual[0].coef_,rtol=1e-8,atol=1e-9)
        expected=(xx-xm)@coef.T+ym;got=actual[0].predict(xx)
    elif r['kind']=='hgb':
        models=[HistGradientBoostingRegressor(max_iter=r['iterations'],max_leaf_nodes=r['leaves'],min_samples_leaf=r['minimum_leaf'],l2_regularization=r['l2'],early_stopping=False,random_state=seed).fit(xx,y[:,i],sample_weight=w) for i in range(2)]
        expected=np.stack([m.predict(xx) for m in models],axis=1);got=np.stack([m.predict(xx) for m in actual],axis=1)
    else:
        model=ExtraTreesRegressor(n_estimators=r['trees'],max_depth=r['depth'],min_samples_leaf=r['minimum_leaf'],random_state=seed,n_jobs=1).fit(xx,y,sample_weight=w)
        expected=model.predict(xx);got=actual[0].predict(xx)
    np.testing.assert_allclose(expected,got,rtol=1e-8,atol=1e-9)


def baseline(root,case,config,new_branches):
    spec=config['baseline_artifacts'][case];result=read(spec['results'])
    if spec['kind']=='essential_exact_budget':point=result['exact_budget_control'];branches=new_branches
    elif spec['kind']=='token_exact_budget':point=result['splits']['test']['yolo']['equal_quota_results']['0.5'];branches=new_branches
    else:
        dataset,expert=case.split('__',1)
        point=result['datasets'][spec.get('dataset',dataset)][spec.get('expert',expert)]['primary'];branches=read(spec['branches'])['branches']
    accepted=set(point['accepted_ids']);rows=[branches[s]['light' if s in accepted else 'deepsolo'] for s in sorted(new_branches)]
    value=totals(rows);expected=point.get('combined',point)
    if value['f1']!=expected['f1']:raise ValueError('Qualified baseline does not replay')
    return rows


def audit(root,report_name):
    plan=read(root/'resolved_configuration.json');config=plan['config'];cases=read(root/'cases.json');results=read(root/'results.json');frozen=read(root/'training_complete.json')
    if plan['mode']!='approximate' or not frozen['test_labels_opened'] is False:raise ValueError('Wrong scientific boundary')
    out=root/report_name;out.mkdir(exist_ok=False);summary={};comparisons=[];recounts=0;trial_replays=0;fit_audits=0;word_audits=0;deployment_replays=0;artifact_hashes={}
    for case,meta in cases.items():
        folder=root/'models'/case;p=meta['partitions'];selection=read(folder/'selection_frozen.json');saved=read(folder/'scores.json');obs=read(root/'data'/(case+'__observations.json'));labels=read(root/'data'/(case+'__train_labels.json'));testlabels=read(root/'data'/(case+'__test_labels.json'));outputs=read(folder/'observable_outputs.json');alllabels=dict(labels,**testlabels)
        if sha(folder/'selection_frozen.json')!=frozen['frozen_sha256'][case] or selection['test_used_for_selection']:raise ValueError('Frozen selection changed')
        keys=meta['tie_keys'];role_groups=[{keys[s] for s in v} for v in p.values()]
        if any(a&b for i,a in enumerate(role_groups) for b in role_groups[i+1:]):raise ValueError('RGB role overlap')
        branches={}
        for s,gt in alllabels.items():
            rows=outputs['corrected_rows'][s];ct=recount(rows,gt['ground_truth'],[gt['ignored'][r['source_row']] for r in rows]);branches[s]=dict(light=ct,deepsolo=gt['teacher']);recounts+=1
        train_branches=read(folder/'train_branches.json')
        for s,v in train_branches.items():
            if v!=branches[s]:raise ValueError('Independent train label/count replay differs')
        with np.load(folder/'features.npz') as arrays:
            ids=arrays['ids'].tolist();pos={s:i for i,s in enumerate(ids)};groups={k:arrays[k] for k in arrays.files if k!='ids'}
        def matrix(samples,names):return np.concatenate([groups[g][[pos[s] for s in samples]] for g in names],axis=1)
        scale={s:len(obs[s]['rows'])+1 for s in obs};fitids=p['fit'];w=np.array([scale[s] for s in fitids]);y=np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fitids])/w[:,None]
        for name in selection['winner']['models']:
            if sha(folder/name)!=selection['model_sha256'][name]:raise ValueError('Changed fitted image head')
            bundle=joblib.load(folder/name);audit_head(bundle,matrix(fitids,bundle['groups']),y,w,config['seed']);fit_audits+=1
        # Refit the FIT-only word-risk folds independently, including word labels.
        xword={s:word_features(obs[s]['rows'],outputs['traces'][s],obs[s]['image_wh']) for s in obs};care={s:np.array([not q for q in labels[s]['ignored']],bool) for s in fitids};yt={}
        for s in fitids:
            texts={normalize(g['text']) for g in labels[s]['ground_truth'] if normalize(g['text'])};yt[s]=np.array([bool(normalize(r['raw_text'])) and normalize(r['raw_text']) in texts for r in obs[s]['rows']],int)
        folds=read(folder/'word_oof_audit.json')['folds'];covered=[]
        for i,f in enumerate(folds):
            if set(f['fit_ids'])!=set(fitids)-set(f['heldout_ids']) or set(f['fit_ids'])&set(f['heldout_ids']):raise ValueError('Word OOF contamination')
            covered+=f['heldout_ids'];r=config['word_risk'];xx=np.concatenate([xword[s][care[s]] for s in f['fit_ids']]);yy=np.concatenate([yt[s][care[s]] for s in f['fit_ids']]);model=HistGradientBoostingClassifier(max_iter=r['iterations'],max_leaf_nodes=r['leaves'],min_samples_leaf=r['minimum_leaf'],l2_regularization=r['l2'],early_stopping=False,random_state=config['seed']+i).fit(xx,yy)
            if sha(folder/f'word_fold_{i}.joblib')!=f['model_sha256']:raise ValueError('Changed OOF model')
            for s in f['heldout_ids']:
                value=model.predict_proba(xword[s])[:,1] if len(xword[s]) else np.empty(0);np.testing.assert_allclose(value,outputs['word_risk'][s],rtol=1e-12,atol=1e-12)
            word_audits+=1
        if sorted(covered)!=sorted(fitids):raise ValueError('Incomplete FIT OOF coverage')
        # Independent replay of every validation/reference/guard/ensemble trial.
        trials=read(folder/'trials.json');valids=p['selection'];vb={s:branches[s] for s in valids};n=round(len(valids)*config['budgets'][meta['dataset']]['fallback_images']/config['budgets'][meta['dataset']]['images']);prediction={}
        for name in {m for t in trials for m in t['models']}:
            bundle=joblib.load(folder/name);xx=bundle['scaler'].transform(matrix(valids,bundle['groups']));models=bundle['models'];prediction[name]=models[0].predict(xx) if len(models)==1 else np.stack([m.predict(xx) for m in models],axis=1)
        for t in trials:
            pred=np.mean([prediction[m] for m in t['models']],axis=0);scores={s:float(2*v[0]-t['reference']*v[1])*scale[s] for s,v in zip(valids,pred)};eligible=[s for s in valids if saved['eligible'][s]] if t['guard'] else valids;accepted=set(rank(scores,keys,n,eligible));ct=totals(v['light' if s in accepted else 'deepsolo'] for s,v in vb.items())
            if any(ct[k]!=t['selection']['combined'][k] for k in ct):raise ValueError('Selection trial replay differs')
            trial_replays+=1
        best=max(t['selection']['combined']['f1'] for t in trials);winner=min((t for t in trials if t['selection']['combined']['f1']>=best-config['selection_tolerance']),key=lambda t:(t['scalars'],-t['selection']['combined']['f1'],len(t['models']),t['models'],t['reference'],t['guard']))
        if winner!=selection['winner']:raise ValueError('Selection criterion replay differs')
        inputs=read(root/'data'/(case+'__inputs.json'));map_paths={Path(name).stem:name for name in inputs if name.endswith('.npz')};inference={};max_error=0.
        for split in ('selection','calibration','test'):
            pipeline=CommonBudgetPipeline(root,case,vocabulary_split='test' if split=='test' else 'train',verify_qualification=False)
            for s in p[split]:
                path=map_paths[s.replace(':','__')]
                if sha(path)!=inputs[path]:raise ValueError('Changed prediction map')
                with np.load(path) as z:maps=z['probabilities']
                value=pipeline.score(obs[s]['rows'],obs[s]['image_wh'],maps);max_error=max(max_error,abs(value['score']-saved['scores'][s]));inference[s]=value
                for g in groups:np.testing.assert_allclose(value['groups'][g],groups[g][pos[s]],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(value['score'],saved['scores'][s],rtol=1e-9,atol=1e-9);deployment_replays+=1
        bp={s:branches[s] for s in p['test']};scores={s:saved['scores'][s] for s in bp};eligible=[s for s in bp if saved['eligible'][s]] if winner['guard'] else list(bp);accepted=rank(scores,keys,meta['fallback_count'],eligible);new=[bp[s]['light' if s in accepted else 'deepsolo'] for s in sorted(bp)];aggregate=totals(new);primary=results['results'][case]['primary']
        if set(accepted)!=set(primary['accepted_ids']) or any(aggregate[k]!=primary['combined'][k] for k in aggregate):raise ValueError('Independent TEST route replay differs')
        deployed=pipeline.route_batch({s:inference[s] for s in bp},{s:keys[s] for s in bp},fallback_images=meta['fallback_count'])
        if set(deployed['accepted_ids'])!=set(accepted):raise ValueError('Deployment exact-budget replay differs')
        if not plan['smoke']:
            old=baseline(root,case,config,bp);baseline_f1=totals(old)['f1']
            if abs(baseline_f1-results['results'][case]['baseline_f1'])>1e-12:raise ValueError('Baseline contract differs')
            interval=paired_bootstrap(old,new,config['seed'],2000)
        else:baseline_f1=None;interval=None
        summary[case]=dict(counts=aggregate,deepsolo_images=meta['fallback_count'],total_images=len(bp),deepsolo_percentage=100*meta['fallback_count']/len(bp),baseline_f1=baseline_f1,bootstrap=interval,deployment_score_max_absolute_error=max_error,models=winner['models'],groups=winner['groups'],reference=winner['reference'],guard=winner['guard'])
        comparisons.append(dict(dataset=meta['dataset'],version=meta['expert'],f1_percent=100*aggregate['f1'],baseline_f1_percent=100*baseline_f1 if baseline_f1 is not None else None,delta_f1_points=100*(aggregate['f1']-baseline_f1) if baseline_f1 is not None else None,deepsolo_images=meta['fallback_count'],total_images=len(bp),deepsolo_percentage=100*meta['fallback_count']/len(bp)))
        for path in [folder/'selection_frozen.json',folder/'correction_frozen.json',folder/'word_full.joblib',*[folder/name for name in winner['models']]]:artifact_hashes[str(path.relative_to(root))]=sha(path)
        print(json.dumps(dict(case=case,status='AUDIT_PASS',deployment_score_max_absolute_error=max_error)),flush=True)
    write(out/'comparison.json',comparisons)
    with (out/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    write(out/'summary.json',summary)
    for name in ['resolved_configuration.json','cases.json','training_complete.json']:
        artifact_hashes[name]=sha(root/name)
    write(out/'independent_audit.json',dict(status='PASS',mode='approximate',claim_label='approximate reproduction',smoke=plan['smoke'],per_image_count_replays=recounts,selection_trial_replays=trial_replays,independent_selected_image_head_fits=fit_audits,independent_word_oof_fits=word_audits,annotation_free_deployment_replays=deployment_replays,actual_new_neural_inference=0,artifact_sha256=artifact_hashes,auditor_sha256=sha(__file__),deployment_code_sha256=sha(Path(__file__).resolve().parents[1]/'recognition/common_budget_deployment.py')))
    write(root/'deployment_qualification.json',dict(status='PASS',audit_path=str((out/'independent_audit.json').relative_to(root)),audit_sha256=sha(out/'independent_audit.json'),artifact_sha256=artifact_hashes))
    lines=['# Common budget pipeline — '+('smoke test' if plan['smoke'] else 'approximate reproduction'),'','Logical routing over real authenticated cached expert outputs. Independent audit PASS.','', '| Dataset | Version | F1 % | Baseline F1 % | Difference, points | DeepSolo images/total (%) |','| --- | --- | ---: | ---: | ---: | --- |']
    for v in comparisons:lines.append(f"| {v['dataset']} | {v['version']} | {v['f1_percent']:.4f} | {v['baseline_f1_percent'] if v['baseline_f1_percent'] is not None else 'N/A'} | {v['delta_f1_points'] if v['delta_f1_points'] is not None else 'N/A'} | {v['deepsolo_images']}/{v['total_images']} ({v['deepsolo_percentage']:.2f}%) |")
    lines+=['','Historical TEST exposure, upstream expert fitting and benchmark vocabulary side information remain. Exact batch quota is transductive and does not measure online latency. Paired bootstrap conditions on fixed selected models.'];(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',report=str(out),rows=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--report-name',default='audit_report_v1');a=p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):audit(a.run_root,a.report_name)
