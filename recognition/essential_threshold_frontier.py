"""Exact cached-expert threshold frontiers and explicitly conditional uncertainty.

High scores accept the light expert. Equal scores are accepted together, so
each point is an attainable scalar cutoff rather than an arbitrary GT ordering.
"""
from bisect import bisect_right
from fractions import Fraction
import math

import numpy as np

COUNT_FIELDS=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')


def observable_length_rescale(score,raw_rows):
    """Recover the trained normalized utility using observed OCR count only."""
    return None if score is None else score/(len(raw_rows)+1)


def point(counts,total,accepted,threshold):
    c=dict(zip(COUNT_FIELDS,map(int,counts)));tp=c['tp'];n=c['detected_words'];g=c['gt_words']
    if c['fp']!=n-tp or c['fn']!=g-tp or c['all_outputs']-c['ignored_outputs']!=n:
        raise ValueError('Inconsistent expert count vector')
    return dict(**c,f1=2*tp/(n+g) if n+g else None,threshold=threshold,accepted_images=accepted,
        deepsolo_images=total-accepted,total_images=total,deepsolo_percent=100*(total-accepted)/total,
        routing_scope='logical cached-output decisions')


def threshold_curve(branches,scores):
    if not branches or set(branches)!=set(scores):raise ValueError('Exactly matched nonempty populations required')
    if any(v is not None and not math.isfinite(v) for v in scores.values()):raise ValueError('Nonfinite routing score')
    ids=sorted(branches);total=np.sum([[branches[s]['deepsolo'][k] for k in COUNT_FIELDS] for s in ids],axis=0,dtype=np.int64)
    if any(branches[s]['light']['gt_words']!=branches[s]['deepsolo']['gt_words'] for s in ids):raise ValueError('Expert GT denominators differ')
    result=[point(total,len(ids),0,None)];eligible=sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))
    accepted=0
    while accepted<len(eligible):
        threshold=scores[eligible[accepted]];end=accepted
        while end<len(eligible) and scores[eligible[end]]==threshold:
            s=eligible[end];total+=np.asarray([branches[s]['light'][k]-branches[s]['deepsolo'][k] for k in COUNT_FIELDS],np.int64);end+=1
        accepted=end;result.append(point(total,len(ids),accepted,threshold))
    return result


def apply_cut(curve,threshold):
    if threshold is None:return dict(curve[0])
    index=bisect_right([-p['threshold'] for p in curve[1:]],-threshold)
    return dict(curve[index],threshold=threshold)


def fraction(p):
    denominator=p['detected_words']+p['gt_words']
    if denominator<=0:raise ValueError('F1 objective needs a positive denominator')
    return Fraction(2*p['tp'],denominator)


def maximum_f1(curve):
    return max(curve,key=lambda p:(fraction(p),-p['deepsolo_images']))


def minimum_cost(curve,reference,allowed_loss=0.):
    if allowed_loss<0:raise ValueError('Nonnegative F1 loss budget required')
    allowed=[p for p in curve if (fraction(p)>=fraction(reference) if allowed_loss==0 else p['f1']>=reference['f1']-allowed_loss)]
    if not allowed:raise ValueError('No threshold satisfies this empirical constraint')
    return min(allowed,key=lambda p:(p['deepsolo_images'],-fraction(p)))


def pareto(curve):
    result=[];best=None
    for p in reversed(curve):
        f=fraction(p)
        if best is None or f>best:result.append(p);best=f
    return result


def gain_identity(branches,accepted):
    """Exact positive-sum condition for hybrid F1 to exceed all DeepSolo."""
    ids=sorted(branches);tp=sum(branches[s]['deepsolo']['tp'] for s in ids)
    den=sum(branches[s]['deepsolo']['detected_words']+branches[s]['deepsolo']['gt_words'] for s in ids)
    dt=sum(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'] for s in accepted)
    dn=sum(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words'] for s in accepted)
    # Multiply through the baseline denominator, avoiding floating comparison.
    return dict(baseline_f1=2*tp/den,delta_tp=dt,delta_predictions=dn,
        scaled_advantage_numerator=2*dt*den-2*tp*dn,positive=2*dt*den-2*tp*dn>0)


def simultaneous_bootstrap(curve,branches,scores,draws,seed,alpha=.05):
    """Approximate simultaneous band for fixed-head image resampling, not a theorem.

    The maximum centered bootstrap error covers the complete searched cutoff
    family. Model fitting and historical TEST/search exposure remain excluded.
    """
    ids=sorted(branches);position={s:i for i,s in enumerate(ids)};n=len(ids)
    eligible=sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))
    order=np.asarray([position[s] for s in eligible],int)
    d=np.asarray([[branches[s]['deepsolo'][k] for k in ['tp','detected_words','gt_words']] for s in ids],float)
    delta=np.asarray([[branches[s]['light'][k]-branches[s]['deepsolo'][k] for k in ['tp','detected_words']] for s in eligible],float).reshape(-1,2)
    endpoints=np.asarray([p['accepted_images'] for p in curve],int)
    empirical=np.asarray([p['f1']-curve[0]['f1'] for p in curve]);rng=np.random.default_rng(seed)
    values=[];maximum_errors=[]
    for start in range(0,draws,100):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(100,draws-start));base=weights@d
        if len(eligible):cumulative=np.cumsum(weights[:,order,None]*delta[None],axis=1)
        else:cumulative=np.empty((len(weights),0,2))
        cumulative=np.concatenate([np.zeros((len(weights),1,2)),cumulative],axis=1)[:,endpoints]
        denominator=base[:,None,1]+base[:,None,2]+cumulative[:,:,1]
        if np.any(denominator<=0) or np.any(base[:,1]+base[:,2]<=0):raise ValueError('Undefined bootstrap F1 draw')
        baseline=2*base[:,0]/(base[:,1]+base[:,2])
        difference=2*(base[:,None,0]+cumulative[:,:,0])/denominator-baseline[:,None]
        maximum_errors.extend(np.max(difference-empirical[None],axis=1).tolist());values.append(difference)
    values=np.concatenate(values);critical=float(np.quantile(maximum_errors,1-alpha));lower=empirical-critical;lower[0]=0.
    return dict(draws=draws,seed=seed,alpha=alpha,simultaneous_critical_value=critical,
        lower_gain=lower.tolist(),pointwise_gain95=np.quantile(values,[.025,.975],axis=0).T.tolist(),
        scope='Conditional fixed-head image bootstrap; complete-cutoff max statistic. Approximate, not a finite-sample population F1 guarantee.')


def common_thresholds(curves):
    candidates=[None]+sorted({p['threshold'] for curve in curves.values() for p in curve[1:]},reverse=True)
    result=[]
    for t in candidates:
        points={ds:apply_cut(curve,t) for ds,curve in curves.items()}
        result.append(dict(threshold=t,datasets=points,deepsolo_images=sum(p['deepsolo_images'] for p in points.values()),
            total_images=sum(p['total_images'] for p in points.values()),
            deepsolo_percent=100*sum(p['deepsolo_images'] for p in points.values())/sum(p['total_images'] for p in points.values()),
            no_loss_all=all(fraction(points[ds])>=fraction(curves[ds][0]) for ds in points),
            strict_gain_all=all(fraction(points[ds])>fraction(curves[ds][0]) for ds in points)))
    return result
