"""Predeclared CAL-only threshold/objective sensitivity for fixed selectors."""
from __future__ import annotations

import argparse
from pathlib import Path

from recognition.contextual_boundary_selection import select
from scripts.run_contextual_boundary_selection import bootstrap, evaluate
from scripts.run_text_reconstruction_tests import read, sha, write


def choose(curve, cap, objective):
    valid=[v for v in curve if v['damaged']<=cap]
    if objective=='correct_then_f1':
        return max(valid,key=lambda r:(r['correct'],r['f1'],r['threshold']))
    if objective=='f1_then_correct':
        return max(valid,key=lambda r:(r['f1'],r['correct'],r['threshold']))
    raise ValueError('Unknown calibration objective')


def calibrate(data, scores, arms, caps, root):
    choices={};curves={}
    for name in arms:
        values=scores[name]['calibration']
        thresholds={1.01}
        for s in data['calibration']:
            thresholds.update(max(v) for v in values[s['sample_id']] if v)
        curve=[]
        for threshold in sorted(thresholds):
            pred={s['sample_id']:select(s,values[s['sample_id']],threshold)[0] for s in data['calibration']}
            curve.append(dict(threshold=threshold,**evaluate(data['calibration'],pred)['aggregate']))
        curves[name]=curve
        for cap in caps:
            for objective in ('correct_then_f1','f1_then_correct'):
                choices[f'{name}_cap{cap}_{objective}']=dict(source_arm=name,cap=cap,objective=objective,
                    choice=choose(curve,cap,objective))
    # This function does not access evaluation/TEST annotations.
    write(root/'selection_frozen.json',dict(choices=choices,curves=curves,evaluation_test_used_for_selection=False))
    return choices


def run(config,root):
    config=Path(config).resolve();repo=Path(__file__).resolve().parents[1]
    if repo/'configs/approximate' not in config.parents:raise ValueError('Explicit APPROXIMATE namespace required')
    c=read(config);source=Path(c['source_run']);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    if c['reproduction_mode']!='approximate' or sha(source/'protocol.json')!=c['source_protocol_sha256']:
        raise ValueError('Supplementary experiment source changed')
    data=read(source/'candidates.json');scores=read(source/'scores.json')
    protocol=dict(resolved_config=c,config_sha256=sha(config),source_scores_sha256=sha(source/'scores.json'),
                  source_candidates_sha256=sha(source/'candidates.json'),code_sha256=sha(__file__))
    write(root/'protocol.json',protocol)
    choices=calibrate(data,scores,c['arms'],c['damage_limits'],root)
    results={};predictions={};comparisons={}
    for phase in ('calibration','evaluation','test'):
        baseline={s['sample_id']:[b['text'] for b in s['blocks']] for s in data[phase]}
        predictions[phase]={'baseline':baseline};results[phase]={'baseline':evaluate(data[phase],baseline)}
        for name,choice in choices.items():
            arm=choice['source_arm'];threshold=choice['choice']['threshold']
            pred={s['sample_id']:select(s,scores[arm][phase][s['sample_id']],threshold)[0] for s in data[phase]}
            predictions[phase][name]=pred;results[phase][name]=evaluate(data[phase],pred)
        if phase in ('evaluation','test'):
            comparisons[phase]={name:bootstrap(results[phase]['baseline'],v,c['bootstrap_draws'],c['seed'])
                for name,v in results[phase].items() if name!='baseline'}
    write(root/'predictions.json',predictions)
    write(root/'results.json',dict(status='SUCCESS',claim_label='approximate reproduction / predeclared calibration sensitivity',
        protocol_sha256=sha(root/'protocol.json'),selection_sha256=sha(root/'selection_frozen.json'),
        results=results,comparisons=comparisons))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config,a.output);print('SUCCESS')


if __name__=='__main__':main()
