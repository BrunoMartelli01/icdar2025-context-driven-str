"""Exploratory fallback fitting on the 125 images excluded from light-model gradients."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import pickle
import traceback

import numpy as np

from recognition.deepsolo_word_f1 import apply_policy,fuse_views
from recognition.trainable_image_router import fit_model,predict
from recognition.unet_fallback_coverage import coverage_features
from recognition.unet_minimum_fallback import exact_counts,independent_counts,routing_curve,minimum_oracle

REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def message(root,text,**values):
    (root/'status.json').write_text(json.dumps(dict(stage=text,**values),indent=2)+'\n');print(text,json.dumps(values),flush=True)

def run(args):
    if args.mode!='approximate' or REPO/'configs/approximate' not in args.config.resolve().parents:raise ValueError('Explicit approximate namespace required')
    c=read(args.config)
    if c['reproduction_mode']!='approximate' or c['claim_label']!='approximate reproduction':raise ValueError('Wrong scientific mode')
    out=args.output;out.mkdir(exist_ok=False,parents=True)
    for p,h in c['input_sha256'].items():
        if sha(p)!=h:raise ValueError('Changed qualified input: '+p)
    write(out/'resolved_config.json',c)
    r=Path(c['paths']['source_run']);g=Path(c['paths']['grounded_run']);h=Path(c['paths']['improved_run'])
    data=pickle.loads((r/'prepared.pkl').read_bytes())['data'];features=np.load(r/'features.npz')
    ocr=pickle.loads(Path(c['paths']['native_ocr_cache']).read_bytes())
    profiles=read(r/'resolved_config.json')['heavy_profiles'];profile_names=list(profiles)+list(c['additional_profiles'])
    improved=read(h/'processed_predictions.json')['splits'];policies=read(h/'deployment_profiles.json')['profiles']
    paired={};xx={};extra_names=None;input_maps={}
    for phase,split in [('evaluation','train'),('test','test')]:
        rows=read(r/f'{phase}_paired_rows.json');extra=[]
        for sample in data[phase]:
            sid=sample['sample_id'];pred=rows[sid]['light']['predictions'];path=Path(sample['maps_path'])
            if sha(path)!=sample['maps_sha256']:raise ValueError('Predicted maps changed')
            input_maps[str(path)]=sample['maps_sha256']
            with np.load(path) as z:
                value,names=coverage_features(z['probabilities'],z['canvas'],sample['pool'],pred,ocr,c['coverage'])
            if extra_names is None:extra_names=names
            if names!=extra_names:raise ValueError('Coverage schema changed')
            extra.append(value)
        base=features[phase+'_tabular'];vision=features[phase+'_vision'];extra=np.asarray(extra)
        xx[phase]=dict(tabular=base,coverage=np.c_[base,extra],vision_coverage=np.c_[vision,extra])
        paired[phase]=rows
    write(out/'input_maps.json',input_maps);write(out/'coverage_schema.json',dict(names=extra_names,GT_used=False))
    np.savez_compressed(out/'features.npz',**{phase+'_'+name:value for phase,branch in xx.items() for name,value in branch.items()})
    def heavy_counts(phase):
        rows=paired[phase];raw={v['sample_id']:v for v in improved['train' if phase=='evaluation' else 'test']}
        for sid,row in rows.items():
            if raw[sid]['image_sha256']!=row['image_sha256']:raise ValueError('Heavy RGB differs')
            for name in c['additional_profiles']:
                p=policies[name]['policy'];value=raw[sid]
                pp=fuse_views(value['native'],value['large'],p) if p['resolution']=='fused' else apply_policy(value['native'],p)
                actual=exact_counts(pp,row['ground_truth'],c['matching_iou']);ind=independent_counts(pp,row['ground_truth'],c['matching_iou'])
                if any(actual[k]!=ind[k] for k in ind):raise ValueError('Heavy count audit differs')
                row['heavy'][name]=dict(predictions=pp,counts=actual)
        return rows
    train=heavy_counts('evaluation');ids=[s['sample_id'] for s in data['evaluation']]
    if {train[s]['pixel_group'] for s in ids}&{r['pixel_group'] for r in paired['test'].values()}:raise ValueError('Router fitting TEST pixel overlap')
    y=np.asarray([[v for p in profile_names for v in (train[s]['heavy'][p]['counts']['tp']-train[s]['light']['counts']['tp'],
        train[s]['heavy'][p]['counts']['predictions']-train[s]['light']['counts']['predictions'])] for s in ids],float)
    order=sorted(ids,key=lambda s:hashlib.sha256((str(c['seed'])+train[s]['pixel_group']).encode()).hexdigest())
    folds={s:i%c['folds'] for i,s in enumerate(order)};fold=np.asarray([folds[s] for s in ids])
    write(out/'folds.json',folds);trials=[];catalog={};oofs={}
    for feature_set in c['feature_sets']:
        x=xx['evaluation'][feature_set]
        for i,recipe in enumerate(c['recipes']):
            name=feature_set+'_'+str(i);oof=np.zeros_like(y);settings=dict(seed=c['seed'],trees=c['trees'])
            for f in range(c['folds']):
                fit=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f)
                model=fit_model(x[fit],y[fit],[ids[i] for i in fit],recipe,settings)
                oof[val]=predict(model,x[val])
            model=fit_model(x,y,ids,recipe,settings);write(out/'models'/(name+'.json'),model)
            catalog[name]=dict(feature_set=feature_set,recipe=recipe);oofs[name]=oof
            for j,profile in enumerate(profile_names):
                light={s:train[s]['light']['counts'] for s in ids};heavy={s:train[s]['heavy'][profile]['counts'] for s in ids}
                for target in c['targets']:
                    scores={s:float(2*v[2*j]-target*v[2*j+1]) for s,v in zip(ids,oof)}
                    _,curve=routing_curve(light,heavy,scores,ids)
                    for point in curve:
                        trials.append(dict(model=name,profile=profile,target=target,fraction=point['deepsolo_images']/len(ids),oof=point))
            message(out,'FIT',model=name)
    selected={}
    for target in c['targets']:
        eligible=[p for p in trials if p['target']==target and p['oof']['f1']>target]
        if eligible:chosen=min(eligible,key=lambda p:(p['fraction'],-p['oof']['f1'],p['model'],p['profile']))
        else:chosen=max([p for p in trials if p['target']==target],key=lambda p:(p['oof']['f1'],-p['fraction']))
        selected[str(target)]=dict(qualified_on_OOF=bool(eligible),**chosen)
    write(out/'development_trials.json',trials);write(out/'model_catalog.json',catalog)
    write(out/'selection_frozen.json',dict(selected=selected,TEST_used_to_select_this_round=False,
        historical_TEST_informed_research_direction=True,router_fit='125 images excluded from light-model gradient training; now explicitly development data'))
    # TEST heavy counts and final routing are evaluated only after new choices freeze.
    test=heavy_counts('test');test_ids=[s['sample_id'] for s in data['test']];results={};curves={};oracles={}
    for target,p in selected.items():
        model=read(out/'models'/(p['model']+'.json'));v=predict(model,xx['test'][catalog[p['model']]['feature_set']]);j=profile_names.index(p['profile'])
        scores={s:float(2*q[2*j]-p['target']*q[2*j+1]) for s,q in zip(test_ids,v)}
        light={s:test[s]['light']['counts'] for s in test_ids};heavy={s:test[s]['heavy'][p['profile']]['counts'] for s in test_ids}
        order,curve=routing_curve(light,heavy,scores,test_ids);n=int(math.ceil(len(test_ids)*p['fraction']-1e-12));chosen=set(order[:n])
        result=dict(policy=p,metrics=curve[n],TEST_target_exceeded=curve[n]['f1']>p['target'])
        cross=[v for v in curve if v['f1']>p['target']];result['posthoc_frozen_ranking_minimum']=cross[0] if cross else None
        rows=[];predictions={};rec=lost=0
        for sid in test_ids:
            branch=test[sid]['heavy'][p['profile']] if sid in chosen else test[sid]['light']
            a=set(light[sid]['matched_gt']);b=set(branch['counts']['matched_gt']);rec+=len(b-a);lost+=len(a-b)
            rows.append(dict(sample_id=sid,action='DeepSolo' if sid in chosen else 'light',counts=branch['counts'],recovered_gt=sorted(b-a),lost_gt=sorted(a-b)))
            predictions[sid]=branch['predictions']
        result.update(recovered=rec,lost=lost,actual_new_DeepSolo_inference_images=0)
        aa=np.asarray([[row['counts'][k] for k in ('tp','predictions','gt')] for row in rows],float)
        rng=np.random.default_rng(c['seed']);boot=np.asarray([aa[rng.integers(0,len(aa),len(aa))].sum(0) for _ in range(c['bootstrap_replicates'])])
        result['conditional_image_bootstrap_95_f1']=np.quantile(2*boot[:,0]/(boot[:,1]+boot[:,2]),[.025,.975]).tolist()
        results[target]=result;curves[target]=dict(order=order,points=curve,scores=scores)
        oracles[target]=minimum_oracle(light,heavy,test_ids,p['target'])
        write(out/('test_'+target+'_predictions.json'),predictions);write(out/('test_'+target+'_routes.json'),dict(DeepSolo_ids=order[:n],per_image=rows))
    write(out/'results.json',dict(claim_label='approximate reproduction',historically_explored_TEST=True,results=results))
    write(out/'test_curves.json',curves);write(out/'test_oracles.json',oracles)
    write(out/'artifact_manifest.json',dict(artifacts={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in out.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','status.json')}))
    message(out,'COMPLETE',status='SUCCESS');print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['approximate']);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    try:run(args)
    except BaseException as e:
        args.output.mkdir(parents=True,exist_ok=True);write(args.output/'failure.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
