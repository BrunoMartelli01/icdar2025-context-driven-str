"""APPROXIMATE budget-specific small-head selection and expert-unseen calibration."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from recognition.lightweight_unet_router import predict_head
from recognition.unet_corrected_token_interfaces import route_summary

REPO=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def split_holdout(ids,selection_images,seed):
    order=sorted(ids,key=lambda s:(hashlib.sha256((str(seed)+'|'+s).encode()).hexdigest(),s))
    if not 0<selection_images<len(ids):raise ValueError('Disjoint nonempty selection/calibration populations required')
    return sorted(order[:selection_images]),sorted(order[selection_images:])


def rank(scores,ids,fraction):
    return sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(ids)*fraction)]


def scalar_cut(scores,ids,fraction):
    order=sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s));n=round(len(ids)*fraction)
    if not 0<n<=len(order):raise ValueError('Insufficient eligible observations for requested CAL acceptance')
    threshold=scores[order[n-1]]
    # Preserve a finite tie cutoff, including the same deterministic sample-ID rule.
    return dict(threshold=threshold,tie_sample_id=order[n-1])


def threshold_ids(scores,cut):
    t=cut['threshold'];tie=cut['tie_sample_id']
    return sorted(s for s,v in scores.items() if v is not None and (v>t or v==t and s<=tie))


def load_features(root,role):
    schema=read(root/(role+'_feature_schema.json'));path=root/(role+'_features.npz')
    if sha(path)!=schema['arrays_sha256']:raise ValueError('Changed precomputed feature data')
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    return schema,a


def score(model,schema,arrays):
    x=np.concatenate([arrays[g] for g in model['groups']],axis=1)
    p=predict_head(model,x)
    return {s:float(v)*schema['word_scale'][s] if schema['eligible'][s] else None for s,v in zip(schema['ids'],p)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['approximate'],required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();c=read(args.config)
    if args.config.resolve().parent!=REPO/'configs/approximate' or c['mode']!=args.mode:raise ValueError('Explicit approximate budget configuration required')
    root=args.output.resolve();root.mkdir(exist_ok=False)
    for path,digest in c['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed source: '+path)
    source=Path(c['paths']['lightweight_run']);audit=read(source/'independent_audit.json');assert audit['status']=='PASS'
    expert=read(c['expert_holdout_plan']);schema,arrays=load_features(source,'selection');holdout=schema['ids']
    assert set(holdout)==set(expert['partitions']['validation'])
    selection,calibration=split_holdout(holdout,c['selection_images'],c['partition_seed'])
    assert len(calibration)==c['calibration_images'];branches=read(source/'selection_branches.json');trials=read(source/'trials.json')
    rows=[];models={};selection_scores={}
    for index,trial in enumerate(trials):
        if trial['cost']['additional_encoder_families']>c['maximum_additional_encoders'] or trial['model_bytes']>c['maximum_model_file_bytes']:continue
        path=source/trial['model_path']
        if sha(path)!=trial['model_sha256']:raise ValueError('Changed frozen candidate')
        model=joblib.load(path)
        assert not set(model['training_ids'])&set(holdout) and set(model['training_ids'])<=set(expert['partitions']['fit'])
        values=score(model,schema,arrays);summary=route_summary(branches,selection,rank(values,selection,c['selection_acceptance']))
        rows.append(dict(trial_index=index,name=trial['name'],groups=trial['groups'],head=trial['head'],stored_scalar_count=trial['stored_scalar_count'],
                         model_path=trial['model_path'],model_sha256=trial['model_sha256'],model_bytes=trial['model_bytes'],selection_summary=summary))
        models[index]=model;selection_scores[index]=values
    best=max(r['selection_summary']['f1'] for r in rows);eligible=[r for r in rows if r['selection_summary']['f1']>=best-c['selection_f1_tolerance']]
    chosen=min(eligible,key=lambda r:(r['stored_scalar_count'],r['model_bytes'],-r['selection_summary']['f1'],r['trial_index']))
    model=models[chosen['trial_index']];values=selection_scores[chosen['trial_index']];policies=[]
    for fraction in c['validation_acceptance_controls']:
        cut=scalar_cut(values,calibration,fraction);cal_scores={s:values[s] for s in calibration};accepted=threshold_ids(cal_scores,cut)
        policies.append(dict(target_calibration_acceptance=fraction,**cut,calibration_summary=route_summary(branches,calibration,accepted)))
    frozen=dict(mode='approximate',claim_label='approximate reproduction',source_config=c,chosen=chosen,
        selection_ids=selection,calibration_ids=calibration,policies=policies,candidate_selection=rows,
        no_refit=True,test_used=False,expert_gradient_holdout_verified=True,source_audit_sha256=sha(source/'independent_audit.json'))
    write(root/'frozen_choices.json',frozen)
    joblib.dump(dict(bundle=model,policies=policies,feature_settings=read(source/'resolved_configuration.json')['source_config'],
                     frozen_choices_sha256=sha(root/'frozen_choices.json')),root/'deployment.joblib')
    # TEST data and outcomes are accessed only after model and cuts are frozen.
    test_schema,test_arrays=load_features(source,'test');test_branches=read(source/'test_branches.json');ids=test_schema['ids']
    values=score(model,test_schema,test_arrays);results=[]
    for policy in policies:
        summary=route_summary(test_branches,ids,threshold_ids(values,policy))
        results.append(dict(target_calibration_acceptance=policy['target_calibration_acceptance'],threshold=policy['threshold'],
            tie_sample_id=policy['tie_sample_id'],**summary,nominal_target_met=summary['f1']>=c['target']['nominal_metric_value'],
            budget_range_met=c['target']['deepsolo_fraction_range'][0]<=summary['deepsolo_images']/len(ids)<=c['target']['deepsolo_fraction_range'][1]))
    batch=route_summary(test_branches,ids,rank(values,ids,c['batch_acceptance_control']))
    result=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',chosen=chosen,policies=results,
        exact_budget_control=batch,primary_calibration_acceptance=c['selection_acceptance'],test_images=len(ids),care_words=batch['gt_words'],
        frozen_choices_sha256=sha(root/'frozen_choices.json'),new_training=False,new_expert_inference=False,
        paper_comparison=c['target']['paper_comparison'],historically_explored_test=True)
    write(root/'test_scores.json',values);write(root/'results.json',result)
    write(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),deployment_sha256=sha(root/'deployment.joblib')))
    print(json.dumps(dict(chosen={k:v for k,v in chosen.items() if k!='selection_summary'},policies=[dict(f=p['target_calibration_acceptance'],f1=p['f1'],D=p['deepsolo_images'],N=p['total_images'],pct=p['deepsolo_percent'],target=p['nominal_target_met'],budget=p['budget_range_met']) for p in results],exact50=dict(f1=batch['f1'],D=batch['deepsolo_images'],N=batch['total_images'],pct=batch['deepsolo_percent']))))


if __name__=='__main__':main()
