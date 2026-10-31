"""Independent replay of cheap-gate fitting, augmentation labels and deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
import numpy as np
from sklearn.decomposition import PCA

from recognition.budget_distilled_router import observation_features,fit_small_head,predict_small_head,paired_target,rank_accept,predict_observation
from recognition.router_validation import require_same_model
from recognition.unet_token_routing import apply_threshold
from scripts.token_context_data import key
from scripts.unet_token_router_stages import light_predictions
from scripts.run_yolo_f1_quality_study import read,sha,seal
from scripts.replay_router_improved_deepsolo import independent_counts,independent_route


def audit(root,output):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];source=Path(plan['paths']['source_root'])
    complete=read(root/'completion.json');results=read(root/'results.json');frozen=read(root/'frozen_choices.json')
    if complete['results_sha256']!=sha(root/'results.json') or complete['audit_sha256']!=sha(root/'independent_audit.json'):raise ValueError('Completion binding changed')
    for name,digest in plan['code_sha256'].items():
        if sha(root/'code_snapshot'/name)!=digest:raise ValueError('Executed snapshot changed')
    original=read(source/'resolved_configuration.json');training=read(source/'train/branches.json')
    branches=training['branches']['unet'];samples={s['sample_id']:s for s in original['samples']['train']}
    metadata=read(root/'training_observations.json');archive=root/'training_observations.npz'
    if sha(archive)!=metadata['npz_sha256']:raise ValueError('Training observation archive changed')
    with np.load(archive,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    fit=plan['partitions']['fit'];validation=plan['partitions']['validation'];checks=0
    if set(a['parent_ids'])!=set(fit) or set(a['parent_ids'])&set(validation+plan['partitions']['test']):raise ValueError('Actual fitting observations overlap held-out population')
    if sum(a['real'])!=len(fit) or len(a['parent_ids'])!=len(fit)*(1+len(cfg['augmentation']['drop_probabilities'])):raise ValueError('Training view counts differ')
    previous=None
    for index,view in enumerate(metadata['views']):
        sid=view['sid']
        if sid!=a['parent_ids'][index]:raise ValueError('Parent order changed')
        if sid!=previous:
            d=read(source/'train/unet'/(key(sid)+'.json'));raw=light_predictions(d['candidates'],samples[sid],original['config']['metric']['native_ignore_size'])
            lookup={r['candidate_id']:p for r,p in zip(d['candidates'],raw)};previous=sid
        counts=independent_counts([lookup[k] for k in view['candidate_ids']],training['records'][sid]['ground_truth'],'casefold_alnum')
        if counts!=view['light_counts']:raise ValueError('Independent counterfactual count mismatch')
        target=paired_target(counts,branches[sid]['deepsolo'],metadata['reference_f1'])
        if target!=a['target'][index] or target!=view['target']:raise ValueError('Teacher/GT target mismatch')
        checks+=1
    population=list(a['population_ids']);positions={s:i for i,s in enumerate(population)}
    pca=PCA(n_components=min(cfg['shared_context']['pca_dimensions'],len(fit)-1),svd_solver='full').fit(a['shared'][[positions[s] for s in fit]])
    projection=frozen['projection']
    if not np.allclose(pca.mean_,projection['mean'],rtol=0,atol=1e-12) or not np.allclose(pca.components_,projection['components'],rtol=1e-10,atol=1e-10):raise ValueError('Projection was not fitted on exactly FIT')
    groups=read(root/'feature_schema.json')['groups'];trials=read(root/'trials.json')['trials'];best=None
    for i,t in enumerate(trials):
        path=root/t['path'];bundle=read(path)
        if sha(path)!=t['sha256'] or bundle['training_ids']!=fit:raise ValueError('Model artifact or training scope changed')
        indices=[j for j,g in enumerate(groups) if g=='ocr' or t['feature_set']!='ocr']
        choose=np.ones(len(a['real']),bool) if t['augmented'] else a['real']
        xx=a['x'][choose][:,indices];parents=a['parent_ids'][choose]
        if t['feature_set']=='shared_context':xx=np.c_[xx,np.stack([(a['shared'][positions[s]]-pca.mean_)@pca.components_.T for s in parents])]
        # Original fitting uses np.stack (C order). Match its reduction order;
        # advanced column indexing otherwise creates a different memory layout.
        xx=np.ascontiguousarray(xx)
        target=a['target'][choose]/a['scale'][choose]
        fitted=fit_small_head(xx,target,a['weights'][choose],t['recipe'],cfg['seed'])
        if json.dumps(fitted,sort_keys=True)!=json.dumps(bundle['head'],sort_keys=True):
            errors={k:float(np.max(np.abs(np.asarray(fitted[k])-np.asarray(bundle['head'][k])))) for k in ('coef','mean','scale','intercept') if k in fitted}
            raise ValueError('Independent deterministic fit differs: '+t['name']+' '+str(errors))
        xv=a['population_x'][[positions[s] for s in validation]][:,indices]
        if t['feature_set']=='shared_context':xv=np.c_[xv,np.stack([(a['shared'][positions[s]]-pca.mean_)@pca.components_.T for s in validation])]
        xv=np.ascontiguousarray(xv)
        predicted=predict_small_head(bundle['head'],xv)
        for sid,value in zip(validation,predicted):
            rows=read(source/'train/unet'/(key(sid)+'.json'))['rows']
            actual=float(value*math.sqrt(len(rows)+1)) if rows and all(r['raw_text'].strip() for r in rows) else None
            expected=t['validation_scores'][sid]
            if (actual is None)!=(expected is None) or (actual is not None and not np.isclose(actual,expected,rtol=1e-11,atol=1e-11)):raise ValueError('Validation prediction mismatch')
        objective=[]
        for quota,p in t['validation_points'].items():
            selected=rank_accept(t['validation_scores'],validation,float(quota));ind=independent_route(branches,validation,selected)
            if p['accepted_ids']!=sorted(selected) or any(p['combined'][k]!=v for k,v in ind.items()):raise ValueError('Validation selection route mismatch')
            objective.append(ind['f1']);checks+=1
        if not np.isclose(np.mean(objective),t['selection_objective'],atol=1e-15):raise ValueError('Selection objective mismatch')
        quality=(t['selection_objective'],-int(t['feature_set']=='shared_context'),-path.stat().st_size,-(i+1))
        if best is None or quality>best[0]:best=(quality,t['name'])
        checks+=len(validation)+1
    if best[1]!=frozen['choice']['name']:raise ValueError('Selected candidate is not validation winner')
    deployment=read(root/'deployment.json')
    if sha(root/'deployment.json')!=frozen['deployment_sha256']:raise ValueError('Deployment bytes changed')
    for p in frozen['policies']:require_same_model(p,frozen['candidate_sha256']);checks+=1
    test=read(source/'test/branches.json')['branches']['unet'];test_samples={s['sample_id']:s for s in original['samples']['test']}
    predictions=read(root/'test_predictions.json');shared={}
    if deployment['shared_projection'] is not None:
        with np.load(root/'test_shared.npz',allow_pickle=False) as z:shared={k:z[k] for k in z.files}
    for sid in plan['partitions']['test']:
        d=read(source/'test/unet'/(key(sid)+'.json'))
        with np.load(source/'test/unet_maps'/(key(sid)+'.npz'),allow_pickle=False) as z:p=z['probabilities']
        actual=predict_observation(deployment,d['rows'],d['candidates'],test_samples[sid]['image_wh'],p,shared.get(sid))
        if actual!=predictions['per_image'][sid]:raise ValueError('Single-image deployment replay mismatch')
        checks+=1
    for split,data,ids,scores in [('validation',branches,validation,frozen['choice']['validation_scores']),
            ('test',test,plan['partitions']['test'],predictions['scores'])]:
        for p in results['splits'][split]['thresholds']:
            accepted=apply_threshold(scores,p['policy']);ind=independent_route(data,ids,accepted)
            if any(p['combined'][k]!=v for k,v in ind.items()) or p['deepsolo_images']!=len(ids)-len(accepted):raise ValueError('Frozen scalar route mismatch')
            checks+=1
        for q,p in results['splits'][split]['predeclared_batch_quotas'].items():
            accepted=rank_accept(scores,ids,float(q));ind=independent_route(data,ids,accepted)
            if any(p['combined'][k]!=v for k,v in ind.items()) or p['deepsolo_images']!=len(ids)-len(accepted):raise ValueError('Fixed budget route mismatch')
            checks+=1
    seal(output,dict(status='PASS',checks=checks,results_sha256=sha(root/'results.json'),deployment_sha256=sha(root/'deployment.json'),
        scope='Independent deterministic refit of all24 heads; actual observation parents, augmentation counts/targets, FIT-only PCA, validation selection, frozen cuts and all TEST single-image deployments/routes',
        actual_new_deepsolo_inference=0))
    print(json.dumps(dict(status='PASS',checks=checks)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit(args.run_root,args.output)
