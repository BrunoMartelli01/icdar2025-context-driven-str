"""APPROXIMATE feature-cost ablations and calibration of a frozen small U-Net gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import time
import traceback

import joblib
import numpy as np
from sklearn.decomposition import PCA

from recognition.lightweight_unet_router import (ocr_summary,token_summary,mask_summary,context_summary,
    paired_utility,ranked,conformal_correction,lower_bound_accept,calibration_threshold,threshold_accept,
    fit_head,predict_head,predict_observation_gate)
from recognition.unet_corrected_token_interfaces import route_summary

REPO=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def log(root,stage,**extra):
    write(root/'status.json',dict(stage=stage,**extra));print(stage,json.dumps(extra),flush=True)


def fallback_counts(rows,gt):
    from collections import Counter
    normalize=lambda s:''.join(c for c in s.casefold() if c.isalnum())
    a=Counter(normalize(r['text']) for r in rows if not r['ignored']);b=Counter(normalize(g['text']) for g in gt)
    a.pop('',None);b.pop('',None);tp=sum((a&b).values());n=sum(not r['ignored'] for r in rows)
    return dict(tp=tp,fp=n-tp,fn=len(gt)-tp,detected_words=n,gt_words=len(gt),all_outputs=len(rows),ignored_outputs=sum(r['ignored'] for r in rows))


def prepare(c,root,smoke):
    for path,digest in c['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed scientific input: '+path)
    source=Path(c['paths']['source_run']);native=Path(c['paths']['native_fallback_run'])
    source_plan=read(source/'resolved_configuration.json')
    for base in (source,native):
        audit=read(base/'independent_audit.json')
        if audit['status']!='PASS' or audit['results_sha256']!=sha(base/'results.json'):raise ValueError('Unqualified source result')
    native_plan=read(native/'resolved_configuration.json');records={k:[] for k in c['expected_images']}
    for split in ('train','test'):
        paired={s['sample_id']:s for s in native_plan['samples'][split]}
        for record in source_plan['records'][split]:
            sid=record['sample_id']
            if record['token_sample']['rgb_sha256']!=paired[sid]['rgb_sha256']:raise ValueError('Fallback source RGB mismatch')
            role=c['split_roles'][record['phase']];records[role].append(record)
    if smoke:records={k:v[:{'fit':24,'selection':12,'calibration':12,'test':8}[k]] for k,v in records.items()}
    elif {k:len(v) for k,v in records.items()}!=c['expected_images']:raise ValueError('Changed fitting/evaluation population')
    groups=[r['token_sample']['rgb_sha256'] for values in records.values() for r in values]
    if len(groups)!=len(set(groups)):raise ValueError('RGB groups must be disjoint across all declared roles')
    plan=dict(mode='approximate',claim_label='approximate reproduction',smoke=smoke,source_config=c,records=records,
              roles='FIT fits transforms/heads; selection chooses features/hyperparameters; calibration freezes thresholds; TEST evaluates only.',
              expert_exposure='Historical expert TRAIN/calibration exposure and explored TEST remain; clean new gate roles are not independent end-to-end validation.')
    import sklearn
    from threadpoolctl import threadpool_info
    write(root/'runtime.json',dict(host=platform.node(),python=platform.python_version(),numpy=np.__version__,
        sklearn=sklearn.__version__,joblib=joblib.__version__,device='cpu',feature_dtype='float64',
        cpu_threads=c['cpu_threads'],threadpools=threadpool_info(),new_gpu_inference=False))
    write(root/'resolved_configuration.json',plan);log(root,'PREPARED',images={k:len(v) for k,v in records.items()})
    return plan


def feature_stage(c,root,plan,roles):
    source=Path(c['paths']['source_run']);native=Path(c['paths']['native_fallback_run'])
    with Path(c['paths']['prepared']).open('rb') as f:prepared=pickle.load(f)
    metadata={s['sample_id']:s for phase in prepared['data'].values() for s in phase}
    features={};branches={};raw_rows={};trace={};started=time.monotonic()
    for split in ('train','test'):
        requested=[role for role in roles if (role=='test')==(split=='test')]
        if not requested:continue
        observations=read(source/split/'corrected.json');paired=read(native/split/'branches.json')
        old=read(source/split/('branch_metrics_original.json' if split=='train' else 'branch_metrics_native_word.json'))['corrected']
        for role in requested:
            features[role]={};branches[role]={};raw_rows[role]={}
            for number,record in enumerate(plan['records'][role],1):
                sid=record['sample_id'];rows=observations[sid];s=metadata[sid]
                if s['image_sha256']!=record['image_sha256']:raise ValueError('Map and corrected image differ')
                if sha(s['maps_path'])!=s['maps_sha256']:raise ValueError('Changed predicted map')
                path=source/split/'features/corrected'/(sid.replace(':','__')+'.npz');meta=read(path.with_suffix('.json'))
                if sha(path)!=meta['arrays_sha256']:raise ValueError('Changed qualified context arrays')
                # Exact allowlists exclude any GT/owner/validity arrays.
                with np.load(s['maps_path'],allow_pickle=False) as z:probabilities=z['probabilities']
                with np.load(path,allow_pickle=False) as z:
                    arrays={k:z[k] for k in ('word_text','word_image','patch_valid','patches','queries')}
                if len(arrays['word_text'])!=len(rows):raise ValueError('Context does not describe current corrected words')
                context=context_summary(arrays,c['global_patch_count'])
                groups=dict(ocr=ocr_summary(rows,record['image_wh']),tokens=token_summary(rows,c['token_probability_cutoffs']),
                    mask=mask_summary(probabilities,record['image_wh'],rows,c['map_features']),
                    local=context['local'],global_=context['global_'],blip=context['blip'])
                groups['global']=groups.pop('global_')
                if not all(np.isfinite(v).all() for v in groups.values()):raise ValueError('Nonfinite inference features')
                features[role][sid]=dict(groups=groups,eligible=bool(rows) and all(r['raw_text'].strip() for r in rows),word_scale=len(rows)+1)
                raw_rows[role][sid]=rows
                actual=fallback_counts(paired['records'][sid]['deepsolo'],paired['records'][sid]['ground_truth'])
                if actual!=paired['branches']['unet'][sid]['deepsolo']:raise ValueError('Native fallback count replay failed')
                if actual['gt_words']!=old[sid]['light']['gt_words']:raise ValueError('Care denominator changed')
                if split=='test' and actual!=old[sid]['deepsolo']:raise ValueError('TEST native comparator differs')
                branches[role][sid]=dict(light=old[sid]['light'],deepsolo=actual)
                trace[sid]=dict(maps_path=s['maps_path'],maps_sha256=s['maps_sha256'],
                               context_path=str(path),context_sha256=meta['arrays_sha256'],
                               map_members_accessed=['probabilities'],ground_truth_used_for_features=False)
                if number%200==0:log(root,'FEATURES',role=role,done=number,total=len(plan['records'][role]))
            ids=sorted(features[role]);np.savez_compressed(root/(role+'_features.npz'),
                **{k:np.stack([features[role][s]['groups'][k] for s in ids]) for k in features[role][ids[0]]['groups']})
            write(root/(role+'_feature_schema.json'),dict(ids=ids,eligible={s:features[role][s]['eligible'] for s in ids},
                word_scale={s:features[role][s]['word_scale'] for s in ids},arrays_sha256=sha(root/(role+'_features.npz'))))
            write(root/(role+'_branches.json'),branches[role])
    write(root/('test_feature_inputs.json' if roles==['test'] else 'train_feature_inputs.json'),trace)
    log(root,'FEATURE_STAGE_COMPLETE',roles=roles,seconds=time.monotonic()-started)
    return features,branches


def matrix(features,ids,groups,projections):
    parts=[]
    for group in groups:
        x=np.stack([features[s]['groups'][group] for s in ids])
        parts.append(projections[group].transform(x) if group in projections else x)
    return np.concatenate(parts,axis=1)


def scores_for(bundle,features,ids,projections):
    x=matrix(features,ids,bundle['groups'],projections);p=predict_head(bundle,x)
    normalized={s:float(v) if features[s]['eligible'] else None for s,v in zip(ids,p)}
    absolute={s:v*features[s]['word_scale'] if v is not None else None for s,v in normalized.items()}
    return normalized,absolute


def policies_for(c,bundle,features,branches,projections,reference):
    ids=sorted(features);normalized,absolute=scores_for(bundle,features,ids,projections);policies={}
    for loss in c['f1_loss_budgets']:
        policy=calibration_threshold(absolute,branches,ids,loss,route_summary)
        policies['calibration_f1_budget_'+str(loss)]=policy
    actual=np.asarray([paired_utility(branches[s],reference)/features[s]['word_scale'] for s in ids])
    predicted=predict_head(bundle,matrix(features,ids,bundle['groups'],projections))
    for alpha in c['conformal_alphas']:
        correction=conformal_correction(predicted,actual,alpha)
        policies['conformal_'+str(alpha)]=dict(alpha=alpha,correction=correction if np.isfinite(correction) else None,
            calibration_ids=ids,summary=route_summary(branches,ids,lower_bound_accept(normalized,correction)),
            guarantee='Nominal one-sided marginal normalized-utility coverage under exchangeability; not conditional accepted risk or global micro-F1 preservation.')
    policies['raw_zero']=dict(threshold=0.,calibration_ids=[],summary=route_summary(branches,ids,threshold_accept(absolute,0.)),guarantee='Uncalibrated control')
    return policies


def fit_and_freeze(c,root,features,branches):
    if (root/'test_features.npz').exists():raise ValueError('Do not fit after TEST feature extraction')
    fit=sorted(features['fit']);selection=sorted(features['selection']);projections={}
    for group in ('global','blip'):
        x=np.stack([features['fit'][s]['groups'][group] for s in fit]);n=min(c['pca_components'],len(fit),x.shape[1])
        projections[group]=PCA(n_components=n,svd_solver='full').fit(x)
    joblib.dump(projections,root/'fit_only_projections.joblib')
    reference=route_summary(branches['fit'],fit,[])['f1'];weights=np.asarray([features['fit'][s]['word_scale'] for s in fit],float)
    y=np.asarray([paired_utility(branches['fit'][s],reference) for s in fit])/weights
    trials=[];winners={};started=time.monotonic();(root/'models').mkdir()
    for name,groups in c['feature_sets'].items():
        x=matrix(features['fit'],fit,groups,projections);indices=[]
        projection_count=sum(p.components_.size+p.mean_.size for g,p in projections.items() if g in groups)
        for index,settings in enumerate(c['heads']):
            bundle=fit_head(groups,x,y,weights,settings,c['seed']);bundle.update(training_ids=fit,reference_f1=reference,
                target='(2 deltaTP - FIT native DeepSolo F1 deltaN)/(observed U-Net word count+1)',projection_scalar_count=projection_count,
                plan_sha256=sha(root/'resolved_configuration.json'))
            path=root/'models'/(name+'__'+str(index)+'.joblib');joblib.dump(bundle,path)
            _,scores=scores_for(bundle,features['selection'],selection,projections)
            points=[dict(fraction=f,**route_summary(branches['selection'],selection,ranked(scores,selection,f))) for f in c['selection_quotas']]
            trial=dict(name=name,groups=groups,head=settings,selection_points=points,
                       selection_mean_f1=float(np.mean([p['f1'] for p in points])),stored_scalar_count=bundle['stored_scalar_count']+projection_count,
                       model_path=str(path.relative_to(root)),model_sha256=sha(path),model_bytes=path.stat().st_size,
                       feature_dimension=bundle['feature_dimension'],cost=bundle['cost'],training_images=len(fit),selection_images=len(selection))
            indices.append(len(trials));trials.append(trial)
        best=max(indices,key=lambda i:(trials[i]['selection_mean_f1'],-trials[i]['stored_scalar_count'],-i));winners[name]=best
        log(root,'ABLATION_FIT',name=name,best_head=trials[best]['head'],validation_mean_f1=trials[best]['selection_mean_f1'])
    top=max(trials[i]['selection_mean_f1'] for i in winners.values());eligible=[i for i in winners.values() if trials[i]['selection_mean_f1']>=top-c['selection_f1_tolerance']]
    chosen=min(eligible,key=lambda i:(trials[i]['cost']['additional_encoder_families'],trials[i]['stored_scalar_count'],trials[i]['feature_dimension'],-trials[i]['selection_mean_f1'],i))
    policies={}
    for name,index in winners.items():
        bundle=joblib.load(root/trials[index]['model_path'])
        policies[name]=policies_for(c,bundle,features['calibration'],branches['calibration'],projections,reference)
    frozen=dict(selected_name=trials[chosen]['name'],selected_trial=chosen,winner_indices=winners,policies=policies,
        trials_sha256=None,reference_f1=reference,fit_ids=fit,selection_ids=selection,calibration_ids=sorted(features['calibration']),
        test_used=False,refit_after_calibration=False,selected_model_sha256=trials[chosen]['model_sha256'],
        projections_sha256=sha(root/'fit_only_projections.joblib'),selection_rule=c['selection_rule'])
    write(root/'trials.json',trials);frozen['trials_sha256']=sha(root/'trials.json');write(root/'frozen_choices.json',frozen)
    selected=joblib.load(root/trials[chosen]['model_path'])
    deployment=dict(bundle=selected,projections={g:v for g,v in projections.items() if g in selected['groups']},
        policies={key:{k:v for k,v in value.items() if k not in ['summary','calibration_ids']} for key,value in policies[frozen['selected_name']].items()},
        feature_settings={k:c[k] for k in ['token_probability_cutoffs','global_patch_count','map_features']},
        plan_sha256=sha(root/'resolved_configuration.json'),frozen_choices_sha256=sha(root/'frozen_choices.json'))
    joblib.dump(deployment,root/'selected_deployment.joblib')
    log(root,'FROZEN',selected=frozen['selected_name'],feature_dimension=trials[chosen]['feature_dimension'],seconds=time.monotonic()-started)
    return frozen,trials,projections


def evaluate(c,root,plan,features,branches,frozen,trials,projections):
    ids=sorted(features);output={};predictions={};timings={}
    for name,index in frozen['winner_indices'].items():
        trial=trials[index];path=root/trial['model_path']
        if sha(path)!=trial['model_sha256']:raise ValueError('Fitted gate changed after calibration')
        bundle=joblib.load(path);normalized,absolute=scores_for(bundle,features,ids,projections)
        predictions[name]=dict(normalized=normalized,absolute=absolute)
        points=[dict(fraction=f,**route_summary(branches,ids,ranked(absolute,ids,f))) for f in c['report_quotas']]
        policies={}
        for key,policy in frozen['policies'][name].items():
            if key.startswith('conformal'):
                correction=math.inf if policy['correction'] is None else policy['correction'];accepted=lower_bound_accept(normalized,correction)
            else:accepted=threshold_accept(absolute,policy['threshold'])
            policies[key]=route_summary(branches,ids,accepted)
        x=matrix(features,ids,bundle['groups'],projections);predict_head(bundle,x)
        elapsed=[]
        for _ in range(c['benchmark_repeats']):
            start=time.perf_counter();predict_head(bundle,x);elapsed.append(time.perf_counter()-start)
        timings[name]=dict(cpu_head_batch_images=len(ids),median_seconds=float(np.median(elapsed)),
                           repeats=c['benchmark_repeats'],scope='Head plus frozen scalar standardization only; feature/encoder/OCR costs excluded')
        output[name]=dict(trial=trial,quotas=points,policies=policies)
    result=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',
        result_class='smoke-test result' if plan['smoke'] else 'approximate reproduction',test_images=len(ids),
        care_words=route_summary(branches,ids,[])['gt_words'],selected_name=frozen['selected_name'],primary_policy=c['primary_policy'],
        variants=output,all_deepsolo=route_summary(branches,ids,[]),standalone_unet=route_summary(branches,ids,ids),
        cost_scope='Logical cached expert decisions. New router training/inference uses CPU; selected extra encoder dependencies are explicit. No online speedup claim.',
        new_deepsolo_calls=0,new_expert_inference_calls=0,no_refit_after_calibration=True,
        test_used_for_selection=False,historically_explored_test=True,independent_end_to_end_validation=False)
    write(root/'predictions.json',predictions);write(root/'head_timings.json',timings);write(root/'results.json',result)
    write(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),frozen_choices_sha256=sha(root/'frozen_choices.json'),
                                     selected_deployment_sha256=sha(root/'selected_deployment.joblib')))
    log(root,'COMPLETE',selected=frozen['selected_name'],test_images=len(ids))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['approximate'],required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    c=read(args.config)
    if args.config.resolve().parent!=REPO/'configs/approximate' or c['reproduction_mode']!=args.mode:raise ValueError('Explicit isolated approximate configuration required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    try:
        plan=prepare(c,root,args.smoke);features,branches=feature_stage(c,root,plan,['fit','selection','calibration'])
        frozen,trials,projections=fit_and_freeze(c,root,features,branches)
        features,branches=feature_stage(c,root,plan,['test'])
        evaluate(c,root,plan,features['test'],branches['test'],frozen,trials,projections)
    except BaseException as error:
        write(root/'failure.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
