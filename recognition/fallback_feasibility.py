"""Exact target-F1 feasibility for two fixed image-level experts; GT diagnostics only."""
from fractions import Fraction


def totals(branches,accepted):
    accepted=set(accepted);counts={k:sum(v['light' if s in accepted else 'deepsolo'][k] for s,v in branches.items()) for k in ['tp','detected_words','gt_words']}
    den=counts['detected_words']+counts['gt_words']
    if not den:raise ValueError('A nonempty F1 denominator is required')
    return dict(counts,fp=counts['detected_words']-counts['tp'],fn=counts['gt_words']-counts['tp'],f1=Fraction(2*counts['tp'],den),
        accepted_ids=sorted(accepted),deepsolo_images=len(branches)-len(accepted),total_images=len(branches),deepsolo_percentage=100*(len(branches)-len(accepted))/len(branches))


def target_minimum(branches,target,eligible=None):
    """Integer residual proves feasibility at every cardinality without searching labels."""
    ratio=Fraction(str(target));ids=sorted(branches);eligible=ids if eligible is None else list(eligible)
    if not 0<ratio<=1 or not set(eligible)<=set(ids):raise ValueError('Invalid target or eligible population')
    for s,v in branches.items():
        if v['light']['gt_words']!=v['deepsolo']['gt_words']:raise ValueError('Expert GT denominators differ')
        for b in v.values():
            if not 0<=b['tp']<=min(b['detected_words'],b['gt_words']):raise ValueError('Invalid branch counts')
    p,q=ratio.numerator,ratio.denominator;base=totals(branches,[])
    constant=2*q*base['tp']-p*(base['detected_words']+base['gt_words'])
    gains={s:2*q*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-p*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in eligible}
    order=sorted(eligible,key=lambda s:(-gains[s],s));residuals=[constant]
    for s in order:residuals.append(residuals[-1]+gains[s])
    feasible=[k for k,r in enumerate(residuals) if r>=0]
    witness=None if not feasible else totals(branches,order[:max(feasible)])
    return dict(target_fraction=str(ratio),feasible=bool(feasible),minimum_deepsolo_images=None if witness is None else witness['deepsolo_images'],
        witness=witness,residual_by_light_count=residuals,largest_target_residual=max(residuals),uses_ground_truth=True)


def global_oracle(branches,eligible=None):
    eligible=sorted(branches) if eligible is None else list(eligible);ratio=Fraction(0)
    while True:
        gains={s:2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-ratio*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in eligible}
        # Zero-gain images also use the light expert: minimize fallback among
        # solutions attaining the same globally optimal F1.
        keep=[s for s in eligible if gains[s]>=0];result=totals(branches,keep)
        if result['f1']==ratio:
            base=totals(branches,[]);certificate=2*base['tp']-ratio*(base['detected_words']+base['gt_words'])+sum(max(g,Fraction(0)) for g in gains.values())
            if certificate:raise ValueError('Global oracle certificate failed')
            return dict(result,certificate=str(certificate),uses_ground_truth=True)
        if result['f1']<ratio:raise ValueError('Oracle objective decreased')
        ratio=result['f1']


def ranking_frontier(branches,scores,tie_keys):
    order=sorted((s for s in branches if scores[s] is not None),key=lambda s:(-scores[s],tie_keys[s],s))
    return [totals(branches,order[:k]) for k in range(len(order)+1)]


def json_point(point):
    return {k:(float(v) if isinstance(v,Fraction) else v) for k,v in point.items()}
