"""Approximate native-dataset low-cost routing, with explicit map frames."""
from __future__ import annotations
from collections import Counter
import math
import cv2
import numpy as np
from PIL import Image,ImageDraw
from recognition.lightweight_unet_router import ocr_summary,token_summary


def budget_policy(config,dataset):
    """Explicit dataset-specific quotas; never inherit another dataset's budget."""
    primary=float(config['primary_acceptance'][dataset])
    targets=[float(q) for q in config['acceptance_targets'][dataset]]
    if not 0<primary<1 or not targets or any(not 0<q<1 for q in targets) or len(set(targets))!=len(targets) or primary not in targets:
        raise ValueError('Explicit primary quota and distinct calibration controls required')
    return primary,targets


def map_summary(probabilities,image_wh,rows,frame,settings):
    """Six descriptors per real output channel; never synthesize a kernel head."""
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0] not in (1,2) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Finite native one/two-head predicted probabilities required')
    if frame not in ('square_letterbox','aspect_letterbox'):raise ValueError('Explicit native geometry required')
    side=(p.shape[1],p.shape[2]);w,h=image_wh
    def project(boxes,fill):
        image=Image.new('L',(w,h),fill);draw=ImageDraw.Draw(image)
        for a,b,c,d in boxes:
            if not 0<=a<c<=w or not 0<=b<d<=h:raise ValueError('Original RGB exclusive bbox required')
            draw.rectangle([a,b,c-1,d-1],fill=255)
        if frame=='square_letterbox':
            square=Image.new('L',(max(w,h),max(w,h)),0);square.paste(image,((max(w,h)-w)//2,(max(w,h)-h)//2));image=square
        else:
            from segmentation.icdar2015_improvements import pad_to_aspect
            image=pad_to_aspect(image,(2*side[0],2*side[1]),fill=0)
        return np.asarray(image.resize((side[1],side[0]),Image.Resampling.NEAREST))>0
    canvas=project([],255);covered=project([r['bbox'] for r in rows],0);result=[];area=p.shape[1]*p.shape[2]
    for values in p:
        foreground=(values>=settings['probability_threshold'])&canvas;weights=np.where(foreground,values,0).astype(float);mass=weights.sum()
        missing=foreground&~covered;n,_,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),connectivity=8)
        areas=[int(a) for a in stats[1:,cv2.CC_STAT_AREA] if a>=settings['minimum_component_area']]
        result.extend([weights[covered].sum()/mass if mass else 1.,mass/area,missing.sum()/area,
                       math.log1p(len(areas)),max(areas,default=0)/area,float(mass==0)])
    return np.asarray(result,float)


def observation(rows,image_wh,probabilities=None,frame=None,*,settings):
    result=dict(ocr=ocr_summary(rows,image_wh),tokens=token_summary(rows,settings['token_probability_cutoffs']))
    if probabilities is not None:result['maps']=map_summary(probabilities,image_wh,rows,frame,settings['maps'])
    return result


def lexical_counts(predictions,truth):
    def norm(s):return ''.join(c for c in s.casefold() if c.isalnum())
    care=[p for p in predictions if not p['ignored']]
    a=Counter(norm(p['text']) for p in care if norm(p['text']));b=Counter(norm(g['text']) for g in truth if norm(g['text']))
    tp=sum(min(n,b[s]) for s,n in a.items())
    return dict(tp=tp,fp=len(care)-tp,fn=len(truth)-tp,detected_words=len(care),gt_words=len(truth),
                all_outputs=len(predictions),ignored_outputs=len(predictions)-len(care))


def rank(scores,ids,fraction,tie_keys=None):
    keys=tie_keys or {s:s for s in ids}
    return sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],keys[s],s))[:round(len(ids)*fraction)]


def calibration_cut(scores,ids,fraction,tie_keys=None):
    keys=tie_keys or {s:s for s in ids};order=rank(scores,ids,1.,keys);n=round(len(ids)*fraction)
    if n==0:return dict(threshold=None,tie_key=None,target_acceptance=fraction)
    if n>len(order):raise ValueError('Too few eligible calibration images for declared budget')
    boundary=order[n-1]
    return dict(threshold=scores[boundary],tie_key=keys[boundary],target_acceptance=fraction)


def accept(scores,policy,tie_keys=None):
    t=policy['threshold'];tie=policy['tie_key'];keys=tie_keys or {s:s for s in scores}
    return sorted(s for s,v in scores.items() if t is not None and v is not None and (v>t or v==t and keys[s]<=tie))


def predict(deployment,rows,image_wh,probabilities=None,frame=None):
    """Portable Ridge deployment; no annotations or actual DeepSolo outputs."""
    f=observation(rows,image_wh,probabilities,frame,settings=deployment['features'])
    x=np.concatenate([f[g] for g in deployment['groups']]);head=deployment['head']
    if len(x)!=len(head['coef']) or not np.isfinite(x).all():raise ValueError('Changed deployment feature schema')
    score=float(((x-head['mean'])/head['scale'])@np.asarray(head['coef'])+head['intercept'])*(len(rows)+1)
    if not rows or any(not r['raw_text'].strip() for r in rows):score=None
    return dict(score=score)
