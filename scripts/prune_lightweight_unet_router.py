"""Exploratory validation-only evidence check before retaining extra gate signals."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def choose_on_validation(trials,winner_indices,branches,*,draws,seed,family_error):
    """No CAL/TEST outcomes are arguments; intervals condition on fitted candidates."""
    ids=sorted(branches);base=trials[winner_indices['ocr']]
    def vector(trial):
        values=[]
        for point in trial['selection_points']:
            keep=set(point['accepted_ids']);values.append([[branches[s]['light' if s in keep else 'deepsolo']['tp'],
                branches[s]['light' if s in keep else 'deepsolo']['detected_words']+branches[s]['light' if s in keep else 'deepsolo']['gt_words']] for s in ids])
        return np.asarray(values,float)
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(ids),(draws,len(ids)))
    ref=vector(base)[:,indices,:].sum(2);ref_f1=(2*ref[:,:,0]/ref[:,:,1]).mean(0);points={};supported=[]
    comparisons=len(winner_indices)-1
    for name,index in winner_indices.items():
        trial=trials[index];x=vector(trial)[:,indices,:].sum(2)
        delta=100*((2*x[:,:,0]/x[:,:,1]).mean(0)-ref_f1)
        lower,upper=np.quantile(delta,[family_error/(2*comparisons),1-family_error/(2*comparisons)])
        points[name]=dict(selection_delta_f1_points=100*(trial['selection_mean_f1']-base['selection_mean_f1']),
            conditional95_interval=np.quantile(delta,[.025,.975]).tolist(),
            simultaneous_interval=[float(lower),float(upper)],supported_extra_signal=bool(lower>0),
            selection_route_costs=[{k:p[k] for k in ['deepsolo_images','total_images','deepsolo_percent']} for p in trial['selection_points']])
        if name!='ocr' and lower>0:supported.append(index)
    chosen=max(supported,key=lambda i:(trials[i]['selection_mean_f1'],-trials[i]['cost']['additional_encoder_families'],-trials[i]['stored_scalar_count'])) if supported else winner_indices['ocr']
    return dict(selected_name=trials[chosen]['name'],selected_trial=chosen,baseline_name='ocr',
        comparisons=points,draws=draws,seed=seed,family_error=family_error,
        rule='Retain extra signals only if the validation paired-bootstrap simultaneous lower interval exceeds zero; otherwise keep OCR-only. Conditional exploratory heuristic after candidate hyperparameter selection, not a confirmatory multiplicity guarantee.',
        test_used=False,calibration_used_for_feature_selection=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['approximate'],required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--family-error',type=float,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False);audit=read(args.run/'independent_audit.json');assert audit['status']=='PASS'
    frozen=read(args.run/'frozen_choices.json');trials=read(args.run/'trials.json')
    choice=choose_on_validation(trials,frozen['winner_indices'],read(args.run/'selection_branches.json'),
                                draws=args.draws,seed=args.seed,family_error=args.family_error)
    choice.update(mode=args.mode,claim_label='approximate reproduction',source_run=str(args.run),
        source_trial_sha256=sha(args.run/'trials.json'),source_frozen_choices_sha256=sha(args.run/'frozen_choices.json'),
        source_selection_sha256=sha(args.run/'selection_branches.json'),analysis_code_sha256=sha(Path(__file__)),
        exploratory_followup=True,new_training=False,primary_original_run_preserved=True)
    write(args.output/'choice_frozen.json',choice)
    # TEST metrics are read only after the validation-based feature choice is saved.
    result=read(args.run/'results.json');assert sha(args.run/'results.json')==audit['results_sha256']
    selected=choice['selected_name'];name=frozen['selected_name'];source=joblib.load(args.run/'selected_deployment.joblib')
    bundle=joblib.load(args.run/trials[choice['selected_trial']]['model_path']);projections=joblib.load(args.run/'fit_only_projections.joblib')
    deployment=dict(source,bundle=bundle,projections={g:x for g,x in projections.items() if g in bundle['groups']},
        policies={k:{a:b for a,b in v.items() if a not in ['summary','calibration_ids']} for k,v in frozen['policies'][selected].items()},
        validation_pruning_choice_sha256=sha(args.output/'choice_frozen.json'))
    joblib.dump(deployment,args.output/'selected_deployment.joblib')
    out=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',exploratory_followup=True,
        selected_name=selected,selected=result['variants'][selected],original_primary_name=name,original_primary=result['variants'][name],
        all_deepsolo=result['all_deepsolo'],test_images=result['test_images'],care_words=result['care_words'],
        choice_frozen_sha256=sha(args.output/'choice_frozen.json'),original_results_sha256=sha(args.run/'results.json'),
        calibration_policies_unchanged=True,refit=False,new_deepsolo_calls=0,new_gpu_calls=0,
        selected_deployment_sha256=sha(args.output/'selected_deployment.joblib'))
    write(args.output/'results.json',out)
    print(json.dumps(dict(status='SUCCESS',selected=selected,extra_signals_supported=[k for k,v in choice['comparisons'].items() if v['supported_extra_signal']])))


if __name__=='__main__':main()
