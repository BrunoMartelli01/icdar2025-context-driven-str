"""Post-evaluation paired intervals for every frozen family, without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def compare(root,iterations,seed):
    plan=read(root/'resolved_configuration.json');result=read(root/'results.json')
    base=plan.get('input_plan',plan)
    source=Path(base['paths']['coverage_run']);source_result=read(source/'results.json')
    branches=source_result['resolutions']['640']['branch_metrics']['test'];ids=sorted(s['sample_id']for s in plan['samples']['test'])
    if result['status']!='SUCCESS' or len(ids)!=300:raise ValueError('Only completed full TEST comparisons are eligible')
    labels=np.array([2*branches[s]['light']['tp']/(2*branches[s]['light']['tp']+branches[s]['light']['fp']+branches[s]['light']['fn'])>=.8 for s in ids])
    rng=np.random.default_rng(seed);weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=iterations).astype(np.float64)
    points=result['splits']['test']['families'];statistics={};ties={}
    features={s:read(root/'test/features'/(s.replace(':','__')+'.json'))for s in ids}
    for family,operating in points.items():
        for fraction in (.2,.3,.4,.5):
            name='budget_'+str(fraction);point=operating[name];accepted=set(point['accepted_ids'])
            mask=np.array([s in accepted for s in ids],np.float64)
            rows=np.array([[branches[s]['light'if s in accepted else'deepsolo'][k]for k in ('tp','fp','fn')]for s in ids],np.float64)
            sampled=weights@rows;tp,fp,fn=sampled.T
            accepted_count=weights@mask;good_count=weights@(mask*labels)
            statistics[(family,fraction)]={'f1':2*tp/(2*tp+fp+fn),'tn0':tp/(tp+fp+fn),
                'good_fraction':np.divide(good_count,accepted_count,out=np.full(iterations,np.nan),where=accepted_count>0)}
            score=[features[s]['scores'][family]for s in accepted]
            cut=min(score)if score else None
            ties[family+':'+str(fraction)]={'cut_score':cut,
                'at_cut_total':sum(features[s]['scores'][family]==cut for s in ids)if cut is not None else 0,
                'at_cut_accepted':sum(features[s]['scores'][family]==cut for s in accepted)if cut is not None else 0,
                'requested_images':round(len(ids)*fraction),'accepted_images':len(accepted)}
    comparisons=[]
    for family in points:
        for fraction in (.2,.3,.4,.5):
            for control in ('coverage_baseline','all_checks_softcrop'):
                left=points[family]['budget_'+str(fraction)];right=points[control]['budget_'+str(fraction)]
                comparisons.append({'family':family,'control':control,'bypass_fraction':fraction,
                    'fixed_primary':family==plan['config']['primary_family'],
                    'train_selected_for_budget':family==result['selected_budget_families'][str(fraction)],
                    'delta_f1':left['combined']['f1']-right['combined']['f1'],
                    'delta_tn0':left['combined']['accuracy_tn0']-right['combined']['accuracy_tn0'],
                    'delta_good_fraction':(left['good_fraction_among_accepted']-right['good_fraction_among_accepted'])
                        if left['good_fraction_among_accepted']is not None else None,
                    'ci95':{metric:np.nanquantile(statistics[(family,fraction)][metric]-statistics[(control,fraction)][metric],[.025,.975]).tolist()
                            for metric in ('f1','tn0','good_fraction')}})
    output={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'plan_sha256':sha(root/'resolved_configuration.json'),'results_sha256':sha(root/'results.json'),
        'analysis_code_sha256':sha(__file__),'iterations':iterations,'seed':seed,
        'unit':'paired image bootstrap; fixed gates; no gate/model fitting',
        'multiplicity_correction':False,'test_hypotheses_are_exploratory':True,
        'comparisons':comparisons,'budget_tie_diagnostics':ties}
    target=root/'analysis/paired_comparisons.json';target.parent.mkdir(exist_ok=True)
    if target.exists()and read(target)!=output:raise ValueError('Immutable analysis differs; choose a new name/version')
    target.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'SUCCESS','path':str(target),'comparisons':len(comparisons)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=10000);p.add_argument('--seed',type=int,default=20261021)
    args=p.parse_args();compare(args.run,args.iterations,args.seed)
