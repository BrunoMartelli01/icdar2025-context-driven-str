"""Training-free quality proxies and TRAIN-only cuts for approximate OCR routing."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
import math
import numpy as np


NEW_FAMILIES = ('native_mean','native_lower_quantile','native_harmonic','clip_mean',
    'word_joint_mean','word_joint_lower_quantile','proxy_f1_native','proxy_f1_stable',
    'proxy_f1_visual','base_proxy_f1','scale_native_proxy','context_proxy_f1',
    'proxy_f1_duplicate_penalty')
COUNT_FIELDS = ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')


def image_f1(c):
    den=2*c['tp']+c['fp']+c['fn']
    return Fraction(2*c['tp'],den) if den else Fraction(0)


def good(c,q):
    den=2*c['tp']+c['fp']+c['fn']
    return den>0 and image_f1(c)>=Fraction(str(q))


def aggregate(rows):
    rows=list(rows);v={k:sum(r[k]for r in rows)for k in COUNT_FIELDS}
    if v['tp']+v['fp']!=v['detected_words']or v['tp']+v['fn']!=v['gt_words']:
        raise ValueError('Word-count denominators differ')
    den=2*v['tp']+v['fp']+v['fn'];union=v['tp']+v['fp']+v['fn']
    return {**v,'images':len(rows),'f1':2*v['tp']/den if den else None,
        'accuracy_tn0':v['tp']/union if union else None,
        'precision':v['tp']/v['detected_words']if v['detected_words']else None,
        'recall':v['tp']/v['gt_words']if v['gt_words']else None}


def bbox_iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=w*h;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def proxy_scores(rows,consistency,context,clip_reference,settings):
    """Only predicted words, geometry, frozen signals and label-free ranks enter."""
    if not rows:return dict.fromkeys(NEW_FAMILIES)
    if any(not r['raw_text'].strip()for r in rows):return dict.fromkeys(NEW_FAMILIES)
    stability={r['candidate_id']:r['minimum_edit_agreement']for r in consistency['crop_stability']['words']}
    if set(stability)!={r['candidate_id']for r in rows}:raise ValueError('Word stability population differs')
    native=np.array([r['confidence']for r in rows],np.float64)
    clip=np.array([(bisect_left(clip_reference,r['clip_word_advantage'])+
        bisect_right(clip_reference,r['clip_word_advantage']))/(2*len(clip_reference))for r in rows])
    crop=np.array([stability[r['candidate_id']]for r in rows]);joint=(native*clip*crop)**(1/3)
    if any(not np.isfinite(x).all()or np.any(x<0)or np.any(x>1)for x in (native,clip,crop)):
        raise ValueError('Invalid frozen word evidence')
    mean=float(native.mean());vis=float(joint.mean());soft=float(crop.mean())
    cov=min(consistency['coverage']['text_coverage'],consistency['coverage']['kernel_coverage'])
    harmonic=len(native)/float(np.sum(1/native))if np.all(native>0)else 0.
    proxy=lambda precision:2*precision*cov/(1+cov)
    dup={i for i,r in enumerate(rows)for j,s in enumerate(rows)if i!=j and
         bbox_iou(r['bbox'],s['bbox'])>=settings['duplicate_iou']}
    base=consistency['scores']['all_checks_softcrop'];mp=context['scores']['base_mpnet']
    values=[mean,float(np.quantile(native,settings['lower_quantile'])),harmonic,float(clip.mean()),
        vis,float(np.quantile(joint,settings['lower_quantile'])),proxy(mean),proxy(mean*soft),proxy(vis),
        math.sqrt(base*proxy(vis)),mean*consistency['scale_agreement']['score'],
        math.sqrt(proxy(vis)*mp)if mp is not None else None,proxy(vis)*(1-len(dup)/len(rows))]
    return dict(zip(NEW_FAMILIES,values))


def sorted_ids(scores,ids):
    if len(ids)!=len(set(ids))or set(scores)!=set(ids):raise ValueError('Scores must cover the declared split exactly')
    if any(x is not None and not math.isfinite(x)for x in scores.values()):raise ValueError('Nonfinite routing score')
    return sorted((s for s in ids if scores[s]is not None),key=lambda s:(-scores[s],s))


def threshold_curve(scores,branches,ids,qs):
    """Inclusive, tie-preserving thresholds; cumulative counts, no candidate leakage."""
    order=sorted_ids(scores,ids);total=aggregate(branches[s]['deepsolo']for s in ids)
    combined={k:total[k]for k in COUNT_FIELDS};accepted_counts=dict.fromkeys(COUNT_FIELDS,0)
    good_counts=dict.fromkeys(map(str,qs),0);out=[]
    for i,sid in enumerate(order):
        light=branches[sid]['light'];deep=branches[sid]['deepsolo']
        for k in COUNT_FIELDS:
            combined[k]+=light[k]-deep[k];accepted_counts[k]+=light[k]
        for q in qs:good_counts[str(q)]+=int(good(light,q))
        if i+1<len(order)and scores[order[i+1]]==scores[sid]:continue
        den=2*combined['tp']+combined['fp']+combined['fn']
        out.append({'threshold':scores[sid],'accepted_images':i+1,'good_images':dict(good_counts),
            'combined_counts':dict(combined),'accepted_counts':dict(accepted_counts),
            'combined_f1':2*combined['tp']/den if den else None})
    return out


def choose_cut(curve,*,q,purity,minimum_accepted,objective):
    """Scalar empirical calibration only; never a fitted network/classifier."""
    if objective not in ('max_bypass','max_f1')or not 0<purity<=1 or minimum_accepted<1:
        raise ValueError('Invalid TRAIN scalar calibration request')
    valid=[v for v in curve if v['accepted_images']>=minimum_accepted and
        Fraction(v['good_images'][str(q)],v['accepted_images'])>=Fraction(str(purity))]
    def key(v):
        c=v['combined_counts'];f=image_f1(c);p=Fraction(v['good_images'][str(q)],v['accepted_images'])
        return ((v['accepted_images'],p,f,v['threshold'])if objective=='max_bypass'else
                (f,v['accepted_images'],p,v['threshold']))
    best=max(valid,key=key)if valid else None
    return {'status':'EMPIRICAL_CUT'if best else'ABSTAIN_ALL_NO_QUALIFYING_CUT',
        'threshold':best['threshold']if best else None,'calibration':best,
        'quality_f1_threshold':q,'target_good_fraction':purity,'minimum_accepted':minimum_accepted,
        'objective':objective,'formal_risk_guarantee':False}


def evaluate(branches,ids,accepted,q):
    accepted=set(accepted)
    if not accepted<=set(ids):raise ValueError('Unknown accepted image')
    true={s for s in ids if good(branches[s]['light'],q)};g=len(accepted&true)
    rejected=set(ids)-accepted;tn=len(rejected-true);n=len(ids)
    return {'images':n,'accepted_images':len(accepted),'fallback_images':len(rejected),
        'accepted_ids':sorted(accepted),'good_images':g,'bad_images':len(accepted)-g,'available_good_images':len(true),
        'good_fraction':g/len(accepted)if accepted else None,'good_recall':g/len(true)if true else None,
        'classification_tp':g,'classification_fp':len(accepted)-g,'classification_fn':len(true)-g,'classification_tn':tn,
        'classification_accuracy':(g+tn)/n if n else None,
        'accepted_macro_image_f1':float(sum((image_f1(branches[s]['light'])for s in accepted),
            Fraction(0))/len(accepted))if accepted else None,
        'accepted_words':aggregate(branches[s]['light']for s in accepted),
        'deepsolo_rejected_words':aggregate(branches[s]['deepsolo']for s in rejected),
        'combined':aggregate(branches[s]['light'if s in accepted else'deepsolo']for s in ids)}
