"""Harmonized bbox/text word counts and deployable, calibration-only routers."""
from __future__ import annotations

import math
import numpy as np
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from recognition.word_slot_text import normalize_word_text,CASEFOLD


def measure(predictions,truth,iou_threshold,normalization=CASEFOLD):
    normalize_word_text('',normalization)
    predicted_text=[normalize_word_text(p['text'],normalization) for p in predictions]
    true_text=[normalize_word_text(g['text'],normalization) for g in truth]
    edges=[(i,j) for i,p in enumerate(predictions) for j,g in enumerate(truth)
           if p.get('bbox') is not None and predicted_text[i] and predicted_text[i]==true_text[j]
           and box_iou(p['bbox'],g['bbox'])>=iou_threshold]
    matched=maximum_matching(len(predictions),len(truth),edges)
    return {'tp':len(matched),'fp':len(predictions)-len(matched),'fn':len(truth)-len(matched),
            'predictions':len(predictions),'gt_words':len(truth),'matched_gt_indices':sorted(matched)}


def aggregate(rows):
    totals={key:sum(r[key] for r in rows) for key in ('tp','fp','fn','predictions','gt_words')}
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    return {**totals,'images':len(rows),'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def nms(boxes,scores,threshold,limit):
    ordered=sorted(range(len(boxes)),key=lambda i:(-float(scores[i]),i));chosen=[]
    for i in ordered:
        # Reference boxes are normalized floats. The inherited metric helper
        # intentionally truncates ORIGINAL pixel edges and cannot be used here.
        def overlap(a,b):
            intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
            area_a=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);area_b=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
            return intersection/max(area_a+area_b-intersection,1e-12)
        if not any(overlap(boxes[i],boxes[j])>threshold for j in chosen):chosen.append(i)
        if len(chosen)==limit:break
    return chosen


def select(predictions,presence,confidence):
    return [p for p in predictions if p.get('presence',1)>=presence and p['confidence']>=confidence]


def image_features(predictions,image_wh):
    width,height=image_wh;n=len(predictions)
    confidence=np.array([p['confidence'] for p in predictions])
    presence=np.array([p.get('presence',1) for p in predictions])
    punctuation=sum(any(not c.isalnum() for c in p['text']) for p in predictions)
    areas=[max(0,p['bbox'][2]-p['bbox'][0])*max(0,p['bbox'][3]-p['bbox'][1])/(width*height)
           for p in predictions if p.get('bbox') is not None]
    return [math.log1p(n),float(confidence.mean()) if n else 0,float(confidence.min()) if n else 0,
            float(np.quantile(confidence,.25)) if n else 0,float(presence.mean()) if n else 0,
            float(presence.min()) if n else 0,punctuation/max(n,1),
            sum(len(p['text']) for p in predictions)/max(n,1),sum(areas),max(areas,default=0),
            math.log(width/height),sum(not p['text'] for p in predictions)/max(n,1)]


def word_features(prediction,other,image_wh,expert,normalization=CASEFOLD):
    width,height=image_wh;box=prediction.get('bbox');text=prediction['text']
    if box is None:box=[0,0,0,0]
    w,h=max(box[2]-box[0],1),max(box[3]-box[1],1)
    overlaps=[box_iou(box,p['bbox']) if p.get('bbox') else 0 for p in other]
    best=max(range(len(other)),key=lambda i:overlaps[i]) if other else None
    compared=normalize_word_text(text,normalization)
    agreement=bool((compared or normalization==CASEFOLD) and best is not None and overlaps[best]>=.3 and normalize_word_text(other[best]['text'],normalization)==compared)
    return [expert,prediction['confidence'],prediction.get('presence',1),math.log1p(len(text)),
            float(any(not c.isalnum() for c in text)),math.log(w/h),w*h/(width*height),
            (box[0]+box[2])/(2*width),(box[1]+box[3])/(2*height),max(overlaps,default=0),
            float(agreement),other[best]['confidence'] if best is not None and overlaps[best]>=.3 else 0]


def merge_words(student,deep,student_probability,deep_probability,*,append_threshold,replace_threshold,margin,conflict_iou,normalization=CASEFOLD):
    """Keep DeepSolo, replace a single conflict, or admit a nonconflicting word.

    Many-to-many replacement is deliberately undefined and does not occur.
    Predictions and calibrated probabilities are the only inference inputs.
    """
    result=[dict(p,route_expert='deepsolo',router_probability=float(prob)) for p,prob in zip(deep,deep_probability)]
    decisions=[]
    for i in sorted(range(len(student)),key=lambda i:(-float(student_probability[i]),i)):
        p=student[i];prob=float(student_probability[i])
        overlap=[j for j,q in enumerate(result) if p.get('bbox') is not None and q.get('bbox') is not None and box_iou(p['bbox'],q['bbox'])>=conflict_iou]
        if not overlap and prob>=append_threshold:
            result.append(dict(p,route_expert='unet',router_probability=prob));decisions.append({'student_index':i,'action':'append'})
        elif len(overlap)==1:
            j=overlap[0];q=result[j]
            if normalize_word_text(p['text'],normalization)==normalize_word_text(q['text'],normalization):continue
            if prob>=replace_threshold and prob-q['router_probability']>=margin:
                result[j]=dict(p,route_expert='unet',router_probability=prob);decisions.append({'student_index':i,'action':'replace','replaced_text':q['text']})
    return result,decisions
