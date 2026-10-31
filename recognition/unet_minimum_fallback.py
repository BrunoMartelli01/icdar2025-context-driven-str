"""APPROXIMATE fixed-ranking fallback budgets and observable U-Net evidence."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from recognition.text_reconstruction import box_iou


def exact_counts(predictions, words, overlap):
    """Keep every output; null geometry/empty text can never match a care word."""
    edges = [[j for j, w in enumerate(words) if p.get('bbox') is not None
              and p['text'] and p['text'].casefold() == w['text'].casefold()
              and box_iou(p['bbox'], w['bbox']) >= overlap] for p in predictions]
    owner = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen): owner[j] = i; return True
        return False
    for i in range(len(predictions)): augment(i, set())
    return dict(tp=len(owner), predictions=len(predictions), gt=len(words),
                matched_gt=sorted(owner), matched_predictions=sorted(owner.values()))


def independent_counts(predictions, words, overlap):
    matrix = np.asarray([[p.get('bbox') is not None and bool(p['text'])
        and p['text'].casefold()==w['text'].casefold()
        and box_iou(p['bbox'],w['bbox'])>=overlap for w in words]
        for p in predictions], bool).reshape(len(predictions),len(words))
    a,b=linear_sum_assignment(matrix.astype(np.int8),maximize=True)
    return dict(tp=int(matrix[a,b].sum()),predictions=len(predictions),gt=len(words))


def aggregate_counts(rows, heavy_images, total_images):
    rows=list(rows)
    tp=sum(r['tp'] for r in rows);pred=sum(r['predictions'] for r in rows);gt=sum(r['gt'] for r in rows)
    if not 0<=heavy_images<=total_images or len(rows)!=total_images:raise ValueError('Invalid route denominator')
    return dict(tp=tp,fp=pred-tp,fn=gt-tp,predictions=pred,gt_words=gt,images=total_images,
        precision=tp/pred if pred else 0.,recall=tp/gt if gt else 0.,
        f1=2*tp/(pred+gt) if pred+gt else 0.,deepsolo_images=heavy_images,
        deepsolo_fraction=heavy_images/total_images if total_images else 0.)


def routing_curve(light, heavy, scores, ids):
    """All frozen-ranking prefixes, with deterministic sample-ID tie resolution."""
    if set(ids)!=set(light) or set(ids)!=set(heavy) or set(ids)!=set(scores) or len(ids)!=len(set(ids)):
        raise ValueError('Incomplete/duplicate paired population')
    if any(not math.isfinite(float(v)) for v in scores.values()):raise ValueError('Nonfinite routing score')
    if any(light[s]['gt']!=heavy[s]['gt'] for s in ids):raise ValueError('Paired GT denominator differs')
    order=sorted(ids,key=lambda s:(-float(scores[s]),s))
    totals={k:sum(light[s][k] for s in ids) for k in ('tp','predictions','gt')}
    def point(k):
        tp,pred,gt=(totals[n] for n in ('tp','predictions','gt'))
        return dict(tp=tp,fp=pred-tp,fn=gt-tp,predictions=pred,gt_words=gt,images=len(ids),
            f1=2*tp/(pred+gt) if pred+gt else 0.,deepsolo_images=k,
            deepsolo_fraction=k/len(ids) if ids else 0.)
    curve=[point(0)]
    for k,s in enumerate(order,1):
        for name in totals:totals[name]+=heavy[s][name]-light[s][name]
        curve.append(point(k))
    return order,curve


def minimum_oracle(light,heavy,ids,target):
    """Exact minimum-cardinality target crossing; privileged diagnosis only.

    At fixed target t, 2TP-t(Npred+Ngt)>0 is additive. Sorting each image's
    heavy-minus-light contribution maximizes this quantity for every cardinality.
    """
    if not 0<target<1 or len(ids)!=len(set(ids)) or set(ids)!=set(light) or set(ids)!=set(heavy):
        raise ValueError('Invalid oracle target or paired population')
    if any(light[s]['gt']!=heavy[s]['gt'] for s in ids):raise ValueError('Paired GT denominator differs')
    t=Fraction(str(target))
    scores={s:2*(heavy[s]['tp']-light[s]['tp'])-t*(heavy[s]['predictions']-light[s]['predictions']) for s in ids}
    order=sorted(ids,key=lambda s:(-scores[s],s))
    tp=sum(light[s]['tp'] for s in ids);pred=sum(light[s]['predictions'] for s in ids);gt=sum(light[s]['gt'] for s in ids)
    best=2*tp-t*(pred+gt)
    certificate=[]
    for k in range(len(ids)+1):
        certificate.append(dict(deepsolo_images=k,target_margin_numerator=best.numerator,
                                target_margin_denominator=best.denominator))
        if best>0:
            return dict(feasible=True,deepsolo_images=k,images=len(ids),deepsolo_fraction=k/len(ids) if ids else 0.,
                f1=float(Fraction(2*tp,pred+gt)),selected_ids=order[:k],certificate=certificate,
                ground_truth_privileged=True,claim='diagnostic lower bound, not deployable routing')
        if k<len(ids):
            s=order[k];best+=scores[s];tp+=heavy[s]['tp']-light[s]['tp'];pred+=heavy[s]['predictions']-light[s]['predictions']
    return dict(feasible=False,deepsolo_images=None,images=len(ids),selected_ids=[],certificate=certificate,
                ground_truth_privileged=True,claim='No image-wise selection reaches the target for these frozen experts')


def fallback_ids(scores,fraction):
    if not 0<=fraction<=1:raise ValueError('Invalid fallback fraction')
    return sorted(scores,key=lambda s:(-scores[s],s))[:int(math.ceil(len(scores)*fraction-1e-12))]


def observable_features(image_hw,pool,predictions,scores,visual_image,clip,quantiles):
    """No annotations, expert outcomes, or DeepSolo observations are accepted."""
    h,w=image_hw
    if min(h,w)<=0 or len(pool)!=len(visual_image):raise ValueError('Unaligned image evidence')
    by={p['id']:i for i,p in enumerate(pool)}
    if len(by)!=len(pool):raise ValueError('Duplicate source proposal')
    features={}
    def add(name,value):
        if not math.isfinite(float(value)):raise ValueError('Nonfinite observable feature '+name)
        features[name]=float(value)
    def stats(name,values):
        x=np.asarray(values,float)
        add(name+'_empty',len(x)==0);add(name+'_mean',x.mean() if len(x) else 0.)
        add(name+'_std',x.std() if len(x) else 0.)
        for q in quantiles:add(name+'_q'+str(q),np.quantile(x,q) if len(x) else 0.)
    add('image_log_aspect',math.log(w/h));add('image_log_pixels',math.log(h*w))
    add('pool_count',len(pool));add('selected_count',len(predictions))
    add('selected_to_pool',len(predictions)/max(1,len(pool)))
    source_ids=[]
    for p in predictions:
        sid=p.get('source_proposal_id',p['id'])
        if sid not in by:raise ValueError('Final output is not bound to original pool')
        source_ids.append(sid)
    selected=set(source_ids)
    for name,rows,indices in [('pool',pool,[p['id'] for p in pool]),('selected',predictions,source_ids)]:
        stats(name+'_length',[len(p['text']) for p in rows])
        stats(name+'_area',[(p['bbox'][2]-p['bbox'][0])*(p['bbox'][3]-p['bbox'][1])/(w*h) for p in rows])
        stats(name+'_height',[(p['bbox'][3]-p['bbox'][1])/h for p in rows])
        stats(name+'_aspect',[(p['bbox'][2]-p['bbox'][0])/max(1,p['bbox'][3]-p['bbox'][1]) for p in rows])
        stats(name+'_joint',[scores.get(p['id'],scores[sid]) for p,sid in zip(rows,indices)])
        add(name+'_empty_readings',sum(not p['text'] for p in rows))
        add(name+'_punctuation',sum(sum(not c.isalnum() for c in p['text']) for p in rows))
        add(name+'_duplicates',len(rows)-len(set(p['text'].casefold() for p in rows)))
    stats('unselected_joint',[scores[p['id']] for p in pool if p['id'] not in selected])
    add('edited_count',sum(p['text']!=pool[by[sid]]['text'] for p,sid in zip(predictions,source_ids)))
    add('removed_characters',sum(len(pool[by[sid]]['text'])-len(p['text']) for p,sid in zip(predictions,source_ids)))
    stats('pool_overlap_with_selected',[max((box_iou(p['bbox'],q['bbox']) for q in predictions),default=0.) for p in pool])
    tabular=np.asarray(list(features.values()),np.float64)
    # Original cached image/tile CLIP patches depend only on RGB. Word-image and
    # word-text arrays are intentionally excluded: those belong to another OCR.
    patches=clip['patches'];valid=clip['patch_valid']
    if patches.ndim!=2 or valid.shape!=(len(patches),) or not valid.any():raise ValueError('Invalid frozen RGB CLIP patches')
    image_mean=np.asarray(patches[valid],np.float64).mean(0)
    mean=np.mean(visual_image,axis=(0,1)) if len(pool) else np.zeros(visual_image.shape[-1])
    selected_visual=np.mean([visual_image[by[sid]].mean(0) for sid in source_ids],axis=0) if source_ids else np.zeros_like(mean)
    vision=np.r_[tabular,image_mean,mean,selected_visual-mean]
    if not np.isfinite(vision).all():raise ValueError('Invalid visual routing vector')
    return dict(tabular=tabular,vision=vision,names=list(features))
