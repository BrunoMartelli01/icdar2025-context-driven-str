"""Independent budget-head selection/count replay and single-image deployment audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from recognition.essential_budget_router import predict_essential_budget
from scripts.audit_lightweight_unet_router import counts, assert_counts

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.run
    frozen=read(root/'frozen_choices.json');c=frozen['source_config'];source=Path(c['paths']['lightweight_run'])
    result=read(root/'results.json');receipt=read(root/'completion.json');checks=Counter()
    assert receipt['results_sha256']==sha(root/'results.json') and receipt['deployment_sha256']==sha(root/'deployment.joblib')
    assert result['frozen_choices_sha256']==sha(root/'frozen_choices.json') and frozen['no_refit'] and not frozen['test_used']
    for path,digest in c['input_sha256'].items():assert sha(path)==digest;checks['pinned_sources']+=1
    expert=read(c['expert_holdout_plan']);select=frozen['selection_ids'];cal=frozen['calibration_ids']
    order=sorted(expert['partitions']['validation'],key=lambda s:(hashlib.sha256((str(c['partition_seed'])+'|'+s).encode()).hexdigest(),s))
    assert select==sorted(order[:c['selection_images']]) and cal==sorted(order[c['selection_images']:])
    assert not set(select)&set(cal) and not set(select+cal)&set(expert['partitions']['fit']);checks['expert_heldout_partition']+=1
    sb=read(source/'selection_branches.json');ss=read(source/'selection_feature_schema.json')
    with np.load(source/'selection_features.npz',allow_pickle=False) as z:sa={k:z[k] for k in z.files}
    trial_summaries=[];score_sets={}
    for row in frozen['candidate_selection']:
        model=joblib.load(source/row['model_path']);assert sha(source/row['model_path'])==row['model_sha256']
        assert not set(model['training_ids'])&set(select+cal)
        x=np.concatenate([sa[g] for g in model['groups']],1)
        prediction=model['regressor'].predict((x-model['scaler'].mean_)/model['scaler'].scale_)
        scores={s:float(v)*ss['word_scale'][s] if ss['eligible'][s] else None for s,v in zip(ss['ids'],prediction)}
        keep=sorted((s for s in select if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(select)*.5)]
        measured=counts(sb,select,keep);assert_counts(measured,row['selection_summary']);trial_summaries.append((row,measured));score_sets[row['trial_index']]=scores
        checks['candidate_selection_replays']+=1
    best=max(m['f1'] for _,m in trial_summaries)
    chosen=min((r for r,m in trial_summaries if m['f1']>=best-c['selection_f1_tolerance']),key=lambda r:(r['stored_scalar_count'],r['model_bytes'],-r['selection_summary']['f1'],r['trial_index']))
    assert chosen==frozen['chosen'];selected_scores=score_sets[chosen['trial_index']]
    for p in frozen['policies']:
        order=sorted((s for s in cal if selected_scores[s] is not None),key=lambda s:(-selected_scores[s],s))
        boundary=order[round(len(cal)*p['target_calibration_acceptance'])-1]
        assert p['threshold']==selected_scores[boundary] and p['tie_sample_id']==boundary
        keep=[s for s in cal if selected_scores[s] is not None and (selected_scores[s]>p['threshold'] or selected_scores[s]==p['threshold'] and s<=p['tie_sample_id'])]
        assert_counts(counts(sb,cal,keep),p['calibration_summary']);checks['independent_calibration_cuts']+=1
    deployment=joblib.load(root/'deployment.joblib');model=deployment['bundle'];test_scores=read(root/'test_scores.json')
    tb=read(source/'test_branches.json');ids=sorted(test_scores)
    for policy in result['policies']:
        keep=[s for s,v in test_scores.items() if v is not None and (v>policy['threshold'] or v==policy['threshold'] and s<=policy['tie_sample_id'])]
        assert_counts(counts(tb,ids,keep),policy);checks['independent_test_counts']+=1
    keep=sorted((s for s in ids if test_scores[s] is not None),key=lambda s:(-test_scores[s],s))[:round(len(ids)*.5)]
    assert_counts(counts(tb,ids,keep),result['exact_budget_control']);checks['independent_exact50_counts']+=1
    plan=read(source/'resolved_configuration.json');records={r['sample_id']:r for r in plan['records']['test']}
    rows=read(Path(plan['source_config']['paths']['source_run'])/'test/corrected.json');trace=read(source/'test_feature_inputs.json')
    maximum=0
    for sid in ids:
        metadata=trace[sid];path=Path(metadata['maps_path']);assert sha(path)==metadata['maps_sha256']
        with np.load(path,allow_pickle=False) as z:probabilities=z['probabilities']
        out=predict_essential_budget(deployment,sid,rows[sid],records[sid]['image_wh'],probabilities=probabilities)
        expected=test_scores[sid]
        if expected is None:assert out['predicted_utility'] is None
        else:maximum=max(maximum,abs(out['predicted_utility']-expected));assert abs(out['predicted_utility']-expected)<1e-10
        for policy in result['policies']:
            assert out['use_unet'][str(policy['target_calibration_acceptance'])]==(sid in policy['accepted_ids'])
            checks['single_image_decisions']+=1
        checks['single_image_feature_replays']+=1
    rng=np.random.default_rng(c['partition_seed']);primary=next(p for p in result['policies'] if p['target_calibration_acceptance']==.5)
    keep=set(primary['accepted_ids']);triples=np.asarray([[tb[s]['light' if s in keep else 'deepsolo'][k] for k in ['tp','fp','fn']] for s in ids])
    values=[]
    for _ in range(2000):
        tp,fp,fn=triples[rng.integers(len(ids),size=len(ids))].sum(0);values.append(2*tp/(2*tp+fp+fn))
    report=dict(status='PASS',mode='approximate',checks=dict(checks),results_sha256=sha(root/'results.json'),
        max_single_batch_score_error=maximum,primary_f1_bootstrap95=np.quantile(values,[.025,.975]).tolist(),
        bootstrap_scope='Image bootstrap, fixed frozen policy; excludes fitting/selection uncertainty and historical TEST exposure.',
        primary_deepsolo_images=primary['deepsolo_images'],total_images=len(ids),deepsolo_percent=primary['deepsolo_percent'],
        analysis_code_sha256=sha(__file__),scope='Independent selection and CAL cuts, paired count sums, authenticated predicted maps, all300 single-image deployment decisions; original fitting was independently audited upstream.')
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
