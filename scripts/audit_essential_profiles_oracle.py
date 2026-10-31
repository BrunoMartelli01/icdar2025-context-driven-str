"""Replay every exported validation profile and certify privileged routing bounds."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from recognition.essential_threshold_deployment import predict_validated_gate

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIELDS=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')

def selected_counts(branches,accepted):
    accepted=set(accepted);total=Counter()
    for sid,b in branches.items():total.update(b['light' if sid in accepted else 'deepsolo'])
    return dict(total,f1=2*total['tp']/(total['detected_words']+total['gt_words']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.root;checks=Counter();maximum=0.
    plan=read(root/'resolved_configuration.json');source_root=Path(read(root/'threshold_analysis_v1/validation_choices_frozen.json')['config']['paths']['totaltext_features'])
    source_plan=read(source_root/'resolved_configuration.json');ttmeta={r['sample_id']:r for r in source_plan['records']['test']}
    trace=read(source_root/'test_feature_inputs.json');corrected=read(Path(source_plan['source_config']['paths']['source_run'])/'test/corrected.json')
    yolo_root=Path(read(root/'threshold_analysis_v1/validation_choices_frozen.json')['config']['paths']['totaltext_yolo'])
    observations=read(root/'evaluation_records.json');oracle=read(root/'conditional_oracle_analysis.json');source_hashes={}
    for expert,study in [('unet','threshold_analysis_v1'),('yolo','threshold_analysis_yolo_matched_v1')]:
        folder=root/study;report=read(folder/'results.json');branches=read(folder/'test_branches.json');scores=read(folder/'test_scores.json')
        for ds,models in branches.items():
            b=models[expert];profile_path=root/'validation_deployments'/(ds+'__'+expert+'.json');profile=read(profile_path);source_hashes[str(profile_path)]=sha(profile_path)
            accepted={name:[] for name in profile['policies']}
            for sid in b:
                if ds=='total_text':
                    rows=corrected[sid] if expert=='unet' else read(yolo_root/'test/light'/(sid.replace(':','__')+'.json'))['rows']
                    wh=ttmeta[sid]['image_wh'];mp=Path(trace[sid]['maps_path'])
                else:
                    rows=observations[ds][sid]['light'][expert]['rows'];meta=next(r for r in plan['population']['test'][ds] if r['sample_id']==sid);wh=meta['image_wh']
                    u=read(root/'unet'/(sid.replace(':','__')+'.json'));mp=root/u['maps_path']
                with np.load(mp,allow_pickle=False) as z:prob=z['probabilities']
                result=predict_validated_gate(profile,sid,rows,wh,prob);expected=scores[ds][expert][sid]
                if expected is None:
                    if result['score'] is not None:raise ValueError('Deployment eligibility differs')
                else:
                    error=abs(result['score']-expected);maximum=max(maximum,error)
                    if error>1e-10:raise ValueError('Deployment score replay differs')
                for name,decision in result['use_light'].items():
                    if decision:accepted[name].append(sid)
                    checks['exported_policy_single_image_decisions']+=1
                checks['exported_profile_image_score_replays']+=1
            for name,ids in accepted.items():
                counts=selected_counts(b,ids);expected=report['datasets'][ds][expert]['validation_selected'][name]['test']
                if len(b)-len(ids)!=expected['deepsolo_images'] or any(counts[k]!=expected[k] for k in FIELDS+('f1',)):raise ValueError('Exported validation profile aggregate differs')
                checks['exported_profile_aggregate_metrics']+=1
            eligible=[s for s,v in scores[ds][expert].items() if v is not None]
            for kind in ['half','minimum_fallback_no_loss']:
                o=oracle['datasets'][ds][expert][kind];ids=o['accepted_ids'];counts=selected_counts(b,ids)
                if any(counts[k]!=o[k] for k in FIELDS+('f1',)) or not set(ids)<=set(eligible):raise ValueError('Privileged oracle count replay failed')
                if kind=='half':
                    value=Fraction(2*o['tp'],o['detected_words']+o['gt_words']);k=round(len(b)*.5)
                    utility={s:2*(b[s]['light']['tp']-b[s]['deepsolo']['tp'])*value.denominator-value.numerator*(b[s]['light']['detected_words']-b[s]['deepsolo']['detected_words']) for s in eligible}
                    base=selected_counts(b,[]);constant=2*base['tp']*value.denominator-value.numerator*(base['detected_words']+base['gt_words'])
                    optimum=sum(sorted(utility.values(),reverse=True)[:k])+constant
                    if len(ids)!=k or optimum!=0:raise ValueError('Exact fractional oracle certificate failed')
                else:
                    base=selected_counts(b,[]);value=Fraction(2*base['tp'],base['detected_words']+base['gt_words'])
                    utility=sorted([2*(b[s]['light']['tp']-b[s]['deepsolo']['tp'])*value.denominator-value.numerator*(b[s]['light']['detected_words']-b[s]['deepsolo']['detected_words']) for s in eligible],reverse=True)
                    k=len(ids)
                    if sum(utility[:k])<0 or k<len(utility) and sum(utility[:k+1])>=0:raise ValueError('Minimum-fallback oracle certificate failed')
                checks['exact_privileged_oracle_certificates']+=1
    result=dict(status='PASS',mode='approximate',checks=dict(checks),max_deployment_score_error=maximum,exported_profiles_sha256=source_hashes,
        oracle_sha256=sha(root/'conditional_oracle_analysis.json'),analysis_code_sha256=sha(__file__),
        scope='All2066 dataset/expert TEST combinations, all24 exported validation policies, and12 exact conditional GT-oracle optimality certificates. GT-oracle outputs remain nondeployable.')
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
