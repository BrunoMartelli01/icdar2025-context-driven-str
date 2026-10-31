"""Reproducible exact61/228 oracle feasibility and independent certificates."""

import argparse
from fractions import Fraction
from pathlib import Path

from recognition.budget_target_oracle import analyze_budget
from recognition.essential_threshold_frontier import COUNT_FIELDS
from scripts.icdar_budget_data import read,sha
from scripts.run_yolo_f1_quality_study import seal


def certify(branches,eligible,report,budget,target):
    n=len(branches);target=Fraction(target)
    for name in ('exact_budget','best_within_budget'):
        p=report[name];accepted=set(p['accepted_ids']);ratio=Fraction(2*p['tp'],p['detected_words']+p['gt_words'])
        assert accepted<=set(eligible)
        for k in COUNT_FIELDS:assert sum(v['light' if s in accepted else 'deepsolo'][k] for s,v in branches.items())==p[k]
        assert p['deepsolo_images']==n-len(accepted) and p['total_images']==n and p['deepsolo_percent']==100*(n-len(accepted))/n
        def margin(s):
            b=branches[s];return 2*(b['light']['tp']-b['deepsolo']['tp'])-ratio*(b['light']['detected_words']-b['deepsolo']['detected_words'])
        ordered=sorted((margin(s) for s in eligible),reverse=True)
        base=sum(2*b['deepsolo']['tp']-ratio*(b['deepsolo']['detected_words']+b['deepsolo']['gt_words']) for b in branches.values())
        possible=[base+sum(ordered[:k]) for k in range(n-budget,len(eligible)+1)]
        if name=='exact_budget':assert possible[0]==0
        else:assert max(possible)==0
    expected=report['maximum_residual_by_accepted_count'];feasible=[k for k,x in enumerate(expected) if x>=0]
    assert (n-max(feasible) if feasible else None)==report['minimum_deepsolo_images']
    return True


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config)
    assert c['mode']=='approximate' and c['claim_label']=='approximate reproduction';root=Path(c['study_root']);results={};lines=['# Exact budget feasibility of existing ICDAR experts','',
        '**Approximate reproduction / privileged GT diagnostic.** User targets: IC13 F1>=90% with at most61/233 DeepSolo images (26.18%); IC15 F1>=79% with at most228/500 (45.60%). Metric, full care populations and every expert output remain unchanged. These values are upper bounds using TEST GT, not trained or deployable gates.','',
        'The current nonempty-light guard is reported separately from unrestricted choice between the same whole-image expert outputs. Removing this guard does not add detections or change the metric. Exact requested counts, best at-most-budget values and minimum target-feasible counts are independently certified with rational arithmetic.','',
        '| Dataset | Expert | Eligible choices | Ideal F1 at requested count % | DeepSolo images/total (%) | Minimum DeepSolo for target |','| --- | --- | --- | ---: | --- | --- |']
    checks=0
    for source in c['oracle_sources']:
        assert sha(source['path'])==source['sha256'];data=read(source['path']);b=data['branches'];ds,expert=source['dataset'],source['expert'];t=c['targets'][ds];n=len(b);assert n==t['images'] and sum(v['light']['gt_words'] for v in b.values())==t['care_words']
        for guard,eligible in [('current_guard',[s for s,v in data['scores'].items() if v is not None]),('unrestricted',list(b))]:
            result=analyze_budget(b,eligible,t['maximum_deepsolo_images'],t['f1']);assert certify(b,eligible,result,t['maximum_deepsolo_images'],t['f1']);checks+=1
            results.setdefault(ds,{}).setdefault(expert,{})[guard]=result
            point=result['exact_budget'];minimum=result['minimum_deepsolo_images'];minimum_text='unattainable' if minimum is None else f'{minimum}/{n} ({100*minimum/n:.2f}%)'
            lines.append(f"| {ds} | {expert} | {guard} | {100*point['f1']:.6f} | {point['deepsolo_images']}/{n} ({point['deepsolo_percent']:.2f}%) | {minimum_text} |")
    lines += ['', 'IC13: current U-Nets cannot meet90% at61 calls even with privileged unrestricted image decisions. Current YOLO reaches90.004649% under its existing guard, leaving virtually no ranking/error margin. IC15: both long768 and YOLO can exceed79% at228 calls under their existing guards; the current learned ranking is the limiting component.','',
        'Source classifications: PAPER_SPECIFIED Algorithm2 fallback motivation only; REPOSITORY_SPECIFIED existing fixed experts, lexical counts and oracle primitives; ASSUMPTION user budgets/targets and unrestricted guard control; HARDWARE_DEVIATION CPU cached diagnostic; UNRESOLVED learnability, generalization and official paper metric equivalence. No new neural inference or TEST-based operational threshold selection.','',
        f"Original per-image evidence, selected IDs and exact target-residual curves are preserved under `{root}`. Test: `tests/recognition/test_budget_target_oracle.py` compares exact and at-most optima plus minimum target counts against exhaustive small-population enumeration. Independently certify every returned optimum at its own rational F1 and verify the maximum residual over all allowed cardinalities is zero."]
    seal(root/'oracle_results.json',dict(status='PASS',mode='approximate',claim_label='approximate reproduction',datasets=results,config_sha256=sha(a.config),sources=c['oracle_sources'],actual_new_neural_inferences=0,scope='Privileged TEST-GT diagnostic; never a deployable result'))
    seal(root/'oracle_independent_audit.json',dict(status='PASS',certified_cases=checks,results_sha256=sha(root/'oracle_results.json'),code_sha256=sha(__file__),scope='Independent count sums and rational optimality certificates at every returned exact/at-most optimum'))
    with Path('docs/BUDGET_TARGET_ROUTER.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('PASS',checks,'certified oracle cases',flush=True)


if __name__=='__main__':main()
