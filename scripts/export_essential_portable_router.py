"""Export the existing qualified OCR18+mask12 Ridge without refitting."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np

from recognition.essential_portable_router import predict_portable_essential
from scripts.run_yolo_f1_quality_study import read,sha,seal
from scripts.audit_lightweight_unet_router import counts,assert_counts


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--source-audit',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();root=a.source_root
    completed=read(root/'completion.json');result=read(root/'results.json');audit=read(a.source_audit)
    if completed['results_sha256']!=sha(root/'results.json') or audit['results_sha256']!=sha(root/'results.json') or audit['status']!='PASS':
        raise ValueError('Unqualified source completion or independent audit')
    if completed['deployment_sha256']!=sha(root/'deployment.joblib'):raise ValueError('Original weights changed')
    d=joblib.load(root/'deployment.joblib');b=d['bundle']
    if b['groups']!=['ocr','mask'] or b['regressor'].coef_.shape!=(30,):raise ValueError('Expected exactly the qualified30-feature Ridge')
    deployment=dict(schema_version=1,reproduction_mode='approximate',claim_label='approximate reproduction',
        mean=b['scaler'].mean_.tolist(),scale=b['scaler'].scale_.tolist(),coef=b['regressor'].coef_.tolist(),intercept=float(b['regressor'].intercept_),
        map_features=d['feature_settings']['map_features'],policies=[{k:p[k] for k in ['target_calibration_acceptance','threshold','tie_sample_id']} for p in d['policies']],
        source_deployment_sha256=sha(root/'deployment.joblib'),source_results_sha256=sha(root/'results.json'),
        selected_on='62 expert-unseen selection images; cuts on63 separate expert-unseen calibration images; unchanged weights',
        expert='Qualified corrected U-Net/PARSeq and existing upstream correctors; do not substitute uncorrected rows')
    out=a.output_root;out.mkdir(parents=True,exist_ok=False);seal(out/'deployment.json',deployment)
    source=Path(read(root/'frozen_choices.json')['source_config']['paths']['lightweight_run']);plan=read(source/'resolved_configuration.json')
    records={v['sample_id']:v for v in plan['records']['test']};rows=read(Path(plan['source_config']['paths']['source_run'])/'test/corrected.json')
    trace=read(source/'test_feature_inputs.json');expected=read(root/'test_scores.json');branches=read(source/'test_branches.json')
    scores={};decisions={};times=[];maximum=0.
    for sid in sorted(expected):
        file=Path(trace[sid]['maps_path'])
        if sha(file)!=trace[sid]['maps_sha256']:raise ValueError('Predicted maps changed')
        with np.load(file,allow_pickle=False) as z:maps=z['probabilities']
        tick=time.perf_counter();v=predict_portable_essential(deployment,sid,rows[sid],records[sid]['image_wh'],maps);times.append(time.perf_counter()-tick)
        if (v['score'] is None)!=(expected[sid] is None):raise ValueError('Eligibility changed')
        if v['score'] is not None:
            maximum=max(maximum,abs(v['score']-expected[sid]))
            if abs(v['score']-expected[sid])>=1e-10:raise ValueError('Portable score differs')
        scores[sid]=v['score'];decisions[sid]=v['use_unet']
    for policy in result['policies']:
        accepted=[s for s in scores if decisions[s][str(policy['target_calibration_acceptance'])]]
        assert_counts(counts(branches,sorted(scores),accepted),policy)
    seal(out/'export_audit.json',dict(status='PASS',source_audit_sha256=sha(a.source_audit),source_results_sha256=sha(root/'results.json'),
        deployment_sha256=sha(out/'deployment.json'),maximum_score_error=maximum,single_image_replays=len(scores),
        decisions=len(scores)*len(result['policies']),numeric_scalars=91,coefficient_count=31,parameter_bytes=728,
        median_observations_head_ms=1000*float(np.median(times)),p95_observations_head_ms=1000*float(np.quantile(times,.95)),
        timing_scope='CPU observations and head only; upstream experts and cold imports excluded',new_training=False,new_expert_inference=0,
        code_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS',deployment=str(out/'deployment.json'),images=len(scores),maximum_score_error=maximum)))


if __name__=='__main__':main()
