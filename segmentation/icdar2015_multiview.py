"""GT-blind merging of hypotheses from one U-Net at several input views."""

import math


def confidence(prediction,field):
    if field=='sequence_confidence':return prediction['sequence_confidence']
    if field=='mean_token_confidence':
        value=prediction['mean_token_log_probability']
        return math.exp(value) if value is not None else 0.
    raise ValueError('Unknown confidence field')


def normalized(text):return ''.join(c for c in text.casefold() if c.isalnum())


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    if union<=0:raise ValueError('Invalid original-coordinate hypothesis box')
    return overlap/union


def emitted(predictions,*,field,threshold,minimum_length,nms_iou):
    if not 0<nms_iou<=1 or not 0<=threshold<=1 or minimum_length<0:
        raise ValueError('Invalid explicit multiview operating policy')
    ranked=[]
    for index,p in enumerate(predictions):
        score=confidence(p,field)
        if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid OCR confidence')
        if score>=threshold and len(normalized(p['text']))>=minimum_length:
            ranked.append((score,index,p))
    kept=[]
    for _,_,p in sorted(ranked,key=lambda x:(-x[0],x[1])):
        # A single original stream retains every admitted component, including
        # overlapping component boxes. Suppression only removes cross-stream
        # hypotheses; boxes/text are never averaged, padded or changed.
        if not any(p['stream_id']!=q['stream_id'] and iou(p['bbox'],q['bbox'])>=nms_iou for q in kept):
            kept.append(p)
    return kept
