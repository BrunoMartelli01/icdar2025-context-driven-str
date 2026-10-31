"""Privileged GT-routing bounds; these are not deployable threshold methods."""
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction

from recognition.essential_threshold_frontier import COUNT_FIELDS,point,fraction

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def aggregate_choice(branches,accepted,threshold=None):
    keep=set(accepted);counts=[sum(b['light' if s in keep else 'deepsolo'][k] for s,b in branches.items()) for k in COUNT_FIELDS]
    return point(counts,len(branches),len(keep),threshold)

def fixed_acceptance_oracle(branches,eligible,accepted_count):
    if not 0<=accepted_count<=len(eligible):raise ValueError('Oracle budget exceeds declared eligibility')
    value=fraction(aggregate_choice(branches,[]));iterations=0
    for iterations in range(1,101):
        def utility(s):
            b=branches[s];return 2*(b['light']['tp']-b['deepsolo']['tp'])*value.denominator-value.numerator*(b['light']['detected_words']-b['deepsolo']['detected_words'])
        order=sorted(eligible,key=lambda s:(-utility(s),s));accepted=order[:accepted_count];p=aggregate_choice(branches,accepted);updated=fraction(p)
        if updated==value:return dict(**p,iterations=iterations,optimality_residual=0,accepted_ids=sorted(accepted),privileged_gt=True)
        value=updated
    raise RuntimeError('Exact fractional oracle did not converge within100 iterations')

def minimum_deepsolo_oracle(branches,eligible):
    base=aggregate_choice(branches,[]);value=fraction(base)
    def utility(s):
        b=branches[s];return 2*(b['light']['tp']-b['deepsolo']['tp'])*value.denominator-value.numerator*(b['light']['detected_words']-b['deepsolo']['detected_words'])
    order=sorted(eligible,key=lambda s:(-utility(s),s));cumulative=0;best=0;sums=[]
    for i,s in enumerate(order,1):
        cumulative+=utility(s);sums.append(cumulative)
        if cumulative>=0:best=i
    p=aggregate_choice(branches,order[:best])
    return dict(**p,privileged_gt=True,accepted_ids=sorted(order[:best]),
        next_acceptance_best_utility=sums[best] if best<len(sums) else None,
        proof='For any fixed number of accepted images, the largest sum of exact baseline-utility values is its sorted prefix. The last nonnegative prefix is the maximum feasible acceptance.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result={};inputs={}
    for expert,study in [('unet','threshold_analysis_v1'),('yolo','threshold_analysis_yolo_matched_v1')]:
        source=args.root/study;branches=read(source/'test_branches.json');scores=read(source/'test_scores.json')
        inputs[str(source/'test_branches.json')]=sha(source/'test_branches.json');inputs[str(source/'test_scores.json')]=sha(source/'test_scores.json')
        for ds,models in branches.items():
            b=models[expert];eligible=[s for s,v in scores[ds][expert].items() if v is not None];k=round(len(b)*.5)
            result.setdefault(ds,{})[expert]=dict(half=fixed_acceptance_oracle(b,eligible,k),minimum_fallback_no_loss=minimum_deepsolo_oracle(b,eligible))
    report=dict(status='SUCCESS',mode='approximate',claim_label='GT-oracle diagnostic, not deployable',datasets=result,input_sha256=inputs,analysis_code_sha256=sha(__file__),
        scope='Fixed measured expert outputs and original observable eligibility; privileged image choices use GT. This bounds conditional selection, not future generalization or a realizable router.')
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({d:{e:dict(f1=v['half']['f1'],D=v['half']['deepsolo_images'],N=v['half']['total_images'],percent=v['half']['deepsolo_percent']) for e,v in models.items()} for d,models in result.items()}))

if __name__=='__main__':main()
