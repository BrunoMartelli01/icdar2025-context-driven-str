"""Post-hoc approximate frozen U-Net router grid; explicitly TEST-dependent cuts."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np

from recognition.deepsolo_word_f1 import apply_policy,fuse_views
from recognition.trainable_image_router import predict
from recognition.unet_minimum_fallback import (exact_counts,independent_counts,routing_curve,minimum_oracle)

REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')

def aligned_sample_ids(samples,rows,feature_count):
    ids=[s['sample_id'] for s in samples]
    if len(ids)!=len(set(ids)) or len(ids)!=feature_count or set(ids)!=set(rows):
        raise ValueError('Feature/paired population differs')
    return ids

def run(args):
    if args.mode!='approximate' or REPO/'configs/approximate' not in args.config.resolve().parents:raise ValueError('Explicit approximate namespace required')
    c=read(args.config);r=Path(c['paths']['source_run']);h=Path(c['paths']['improved_run']);out=args.output
    out.mkdir(exist_ok=False,parents=True)
    for p,v in c['input_sha256'].items():
        if sha(p)!=v:raise ValueError('Input changed: '+p)
    if c['claim_label']!='approximate reproduction':raise ValueError('Wrong result label')
    manifest=read(r/'artifact_manifest.json')['artifacts']
    for p,v in manifest.items():
        if sha(r/p)!=v['sha256']:raise ValueError('Primary source artifact changed')
    catalog={v['name']:v for v in read(r/'model_catalog.json')};features=np.load(r/'features.npz')
    processed=read(h/'processed_predictions.json')['splits'];profiles=read(h/'deployment_profiles.json')['profiles']
    populations=pickle.loads((r/'prepared.pkl').read_bytes())['data']
    all_results={};audits=0
    write(out/'resolved_config.json',c)
    for phase,split in [('evaluation','train'),('test','test')]:
        rows=read(r/f'{phase}_paired_rows.json')
        ids=aligned_sample_ids(populations[phase],rows,len(features[phase+'_tabular']))
        raw={v['sample_id']:v for v in processed[split]}
        light={s:rows[s]['light']['counts'] for s in ids};heavies={};predictions={}
        for name in read(r/'resolved_config.json')['heavy_profiles']:
            heavies[name]={s:rows[s]['heavy'][name]['counts'] for s in ids}
            predictions[name]={s:rows[s]['heavy'][name]['predictions'] for s in ids}
        for name in c['additional_profiles']:
            heavies[name]={};predictions[name]={};policy=profiles[name]['policy']
            for sid in ids:
                row=raw[sid]
                if row['image_sha256']!=rows[sid]['image_sha256']:raise ValueError('Different expert RGB')
                if [(w['bbox'],w['text']) for w in row['ground_truth']]!=[(w['bbox'],w['text']) for w in rows[sid]['ground_truth']]:
                    raise ValueError('Expert GT/coordinate denominator differs')
                pp=fuse_views(row['native'],row['large'],policy) if policy['resolution']=='fused' else apply_policy(row['native'],policy)
                value=exact_counts(pp,rows[sid]['ground_truth'],c['matching_iou']);ind=independent_counts(pp,rows[sid]['ground_truth'],c['matching_iou'])
                if any(value[k]!=ind[k] for k in ind):raise ValueError('Independent matching differs')
                heavies[name][sid]=value;predictions[name][sid]=pp;audits+=1
        results=[];curve_data={};native_profiles=list(read(r/'resolved_config.json')['heavy_profiles'])
        for model_name in c['model_names']:
            info=catalog[model_name];model=read(r/'models'/(model_name+'.json'))
            values=predict(model,features[phase+'_'+info['feature_set']])
            for profile,heavy in heavies.items():
                source=c['profile_source_index'].get(profile,profile);j=native_profiles.index(source)
                for target in c['targets']:
                    scores={s:float(2*v[2*j]-target*v[2*j+1]) for s,v in zip(ids,values)}
                    order,curve=routing_curve(light,heavy,scores,ids)
                    crossings=[v for v in curve if v['f1']>target]
                    key=model_name+'/'+profile+'/'+str(target)
                    record=dict(model=model_name,profile=profile,source_predicted_profile=source,target=target,
                        first_crossing=crossings[0] if crossings else None,best_f1_point=max(curve,key=lambda x:x['f1']),
                        TEST_selected_operating_point=phase=='test',new_online_DeepSolo_calls=0)
                    results.append(record);curve_data[key]=dict(order=order,points=curve,scores=scores)
        selected={}
        for target in c['targets']:
            candidates=[v for v in results if v['target']==target and v['first_crossing']]
            if candidates:
                chosen=min(candidates,key=lambda v:(v['first_crossing']['deepsolo_images'],-v['first_crossing']['f1'],v['model'],v['profile']))
                key=chosen['model']+'/'+chosen['profile']+'/'+str(target);order=curve_data[key]['order'];n=chosen['first_crossing']['deepsolo_images']
                chosen=dict(chosen,DeepSolo_ids=order[:n]);selected[str(target)]=chosen
                ss=set(order[:n]);pp={s:predictions[chosen['profile']][s] if s in ss else rows[s]['light']['predictions'] for s in ids}
                recovered=lost=0;per_image=[]
                for sid in ids:
                    actual=exact_counts(pp[sid],rows[sid]['ground_truth'],c['matching_iou']);before=light[sid]
                    rec=set(actual['matched_gt'])-set(before['matched_gt']);damage=set(before['matched_gt'])-set(actual['matched_gt'])
                    recovered+=len(rec);lost+=len(damage);per_image.append(dict(sample_id=sid,action='DeepSolo' if sid in ss else 'light',
                        counts=actual,recovered_gt_indices=sorted(rec),lost_gt_indices=sorted(damage)))
                chosen.update(recovered=recovered,lost=lost)
                x=np.asarray([[v['counts'][k] for k in ('tp','predictions','gt')] for v in per_image],float)
                rng=np.random.default_rng(c['seed']);boot=np.asarray([x[rng.integers(0,len(x),len(x))].sum(0) for _ in range(c['bootstrap_replicates'])])
                chosen['conditional_selected_cut_95_f1']=np.quantile(2*boot[:,0]/(boot[:,1]+boot[:,2]),[.025,.975]).tolist()
                chosen['interval_omits_TEST_model_and_cut_selection']=True
                write(out/(phase+'_target_'+str(target)+'_predictions.json'),pp)
                write(out/(phase+'_target_'+str(target)+'_routes.json'),dict(DeepSolo_ids=order[:n],per_image=per_image,
                    warning='Exploratory cut chosen on this split; not an independently calibrated operating point'))
        oracles={p:{str(t):minimum_oracle(light,heavy,ids,t) for t in c['targets']} for p,heavy in heavies.items()}
        all_results[phase]=dict(points=results,selected_descriptive=selected,
            all_DeepSolo={p:routing_curve(light,heavy,dict.fromkeys(ids,0.),ids)[1][-1] for p,heavy in heavies.items()})
        write(out/(phase+'_curves.json'),curve_data);write(out/(phase+'_oracle_diagnostics.json'),oracles)
    write(out/'results.json',dict(claim_label='approximate reproduction',posthoc=True,independent_validation=False,results=all_results))
    write(out/'independent_audit.json',dict(status='PASS',additional_image_profile_counts=audits,source_artifacts_verified=len(manifest),
        logical_cached_routing=True,actual_online_DeepSolo_calls=0))
    write(out/'artifact_manifest.json',dict(artifacts={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}))
    print(json.dumps({p:v['selected_descriptive'] for p,v in all_results.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['approximate'],required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
