"""Fixed-policy paired evidence for keeping one frozen encoder or both."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(branches,ids,first,second,draws,seed):
    arrays=[]
    for accepted in [set(first['accepted_ids']),set(second['accepted_ids'])]:
        arrays.append(np.asarray([[branches[s]['light' if s in accepted else 'deepsolo'][k] for k in ['tp','fp','fn']] for s in ids]))
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        indices=rng.integers(len(ids),size=len(ids));scores=[]
        for a in arrays:
            tp,fp,fn=a[indices].sum(0);scores.append(2*tp/(2*tp+fp+fn))
        values.append(scores[0]-scores[1])
    return dict(f1_difference=first['f1']-second['f1'],paired_bootstrap95=np.quantile(values,[.025,.975]).tolist(),
        first_deepsolo_images=first['deepsolo_images'],second_deepsolo_images=second['deepsolo_images'],total_images=len(ids),
        first_deepsolo_percent=first['deepsolo_percent'],second_deepsolo_percent=second['deepsolo_percent'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.run;plan=read(root/'resolved_configuration.json');c=plan['source_config'];result=read(root/'results.json')
    frozen=read(root/'frozen_choices.json');source=Path(c['paths']['lightweight_run']);comparisons={}
    for phase in ['selection','test_exact50']:
        branches=read(source/('selection_branches.json' if phase=='selection' else 'test_branches.json'))
        if phase=='selection':points={a:v['selection_summary'] for a,v in frozen['arm_choices'].items()};ids=plan['selection_ids']
        else:points={a:v['exact50'] for a,v in result['arms'].items()};ids=sorted(branches)
        comparisons[phase]={}
        for first,second in [('clip','none'),('blip','none'),('both','clip'),('both','blip')]:
            comparisons[phase][first+'_minus_'+second]=paired(branches,ids,points[first],points[second],2000,c['seed'])
    report=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',comparisons=comparisons,
        primary_arm_frozen=frozen['primary_arm'],results_sha256=sha(root/'results.json'),analysis_code_sha256=sha(__file__),
        interpretation='Conditional exploratory paired image bootstrap. Selected checkpoints are fixed; model selection, fitting and multiple comparisons are not included. TEST rank controls are descriptive and cannot select an encoder. Intervals crossing zero do not establish an encoder gain.')
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
