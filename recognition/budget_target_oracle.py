"""Exact finite-population target feasibility; every decision here is privileged."""

from fractions import Fraction

from scripts.analyze_essential_oracle import fixed_acceptance_oracle,aggregate_choice


def target_residual(branches,eligible,target):
    target=Fraction(target)
    if not 0<target<=1 or not set(eligible)<=set(branches):raise ValueError('Valid target and eligible population required')
    def gain(s):
        b=branches[s]
        return 2*(b['light']['tp']-b['deepsolo']['tp'])*target.denominator-target.numerator*(b['light']['detected_words']-b['deepsolo']['detected_words'])
    residual=sum(2*b['deepsolo']['tp']*target.denominator-target.numerator*(b['deepsolo']['detected_words']+b['deepsolo']['gt_words']) for b in branches.values())
    order=sorted(eligible,key=lambda s:(-gain(s),s));values=[residual]
    for sid in order:residual+=gain(sid);values.append(residual)
    feasible=[k for k,v in enumerate(values) if v>=0]
    k=max(feasible) if feasible else None
    witness=None if k is None else fixed_acceptance_oracle(branches,eligible,k)
    return dict(target=str(target),maximum_light_acceptance=k,minimum_deepsolo_images=None if k is None else len(branches)-k,
                witness=witness,maximum_residual_by_accepted_count=values,
                proof='For fixed accepted count, sorted integer target margins maximize 2TP - target*(predictions+GT). Nonnegative maximum is equivalent to attaining target F1.')


def analyze_budget(branches,eligible,deepsolo_budget,target):
    n=len(branches)
    if not branches or not 0<=deepsolo_budget<=n:raise ValueError('Valid nonempty population and budget required')
    if any(b['light']['gt_words']!=b['deepsolo']['gt_words'] for b in branches.values()):raise ValueError('Experts must share every GT denominator')
    target_result=target_residual(branches,eligible,target)
    first=n-deepsolo_budget
    if first>len(eligible):return dict(guard_budget_feasible=False,**target_result)
    exact=fixed_acceptance_oracle(branches,eligible,first)
    choices=[exact]+[fixed_acceptance_oracle(branches,eligible,k) for k in range(first+1,len(eligible)+1)]
    best=max(choices,key=lambda p:(Fraction(2*p['tp'],p['detected_words']+p['gt_words']),-p['deepsolo_images']))
    return dict(guard_budget_feasible=True,exact_budget=exact,best_within_budget=best,
        target_feasible_within_budget=Fraction(2*best['tp'],best['detected_words']+best['gt_words'])>=Fraction(target),**target_result)
