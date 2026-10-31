"""Explicit approximate corrected-expert counting and token routing boundaries."""
from collections import Counter
import math
import numpy as np


def normalized(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def verified_source_crop(rgb,row):
    from scripts.evaluate_cached_totaltext_padding import crop_hash
    a,b,x,y=row.get('crop_bbox',row['bbox'])
    if not 0<=a<x<=rgb.shape[1] or not 0<=b<y<=rgb.shape[0]:
        raise ValueError('Source crop outside original raster')
    crop=np.ascontiguousarray(rgb[b:y,a:x])
    if crop_hash(crop)!=row['crop_rgb_sha256']:
        raise ValueError('Source crop shape or RGB bytes changed')
    return crop


def native_observation(row,parent,reading,confidence,verifier_score):
    if reading['raw_text']!=parent['text']:
        raise ValueError('Native confidence belongs to another transcript')
    if row['bbox']!=parent['bbox'] or row['crop_array_sha256']!=parent['crop_array_sha256']:
        raise ValueError('Text-only edits must preserve original geometry and RGB')
    if any(not math.isfinite(v) or not 0<=v<=1 for v in (confidence,verifier_score)):
        raise ValueError('Finite original confidence and verifier score required')
    return dict(candidate_id=row['id'],raw_text=row['text'],bbox=list(row['bbox']),
        crop_bbox=list(row.get('padded_bbox_original',row['bbox'])),
        crop_rgb_sha256=row['crop_array_sha256'],token_probabilities=list(reading['token_probabilities']),
        confidence=confidence,detector_score=verifier_score,original_raw_text=parent['text'],
        inherited_box_and_confidence=bool(row.get('source_proposal_id')),
        detector_score_origin='frozen RGB-verifier joint score',confidence_origin='native original PARSeq')


def lexical_counts(rows,ground_truth,ignored):
    if len(rows)!=len(ignored):raise ValueError('Output and ignore counts differ')
    words=[normalized(r['raw_text']) for r,flag in zip(rows,ignored) if not flag]
    targets=[normalized(g['text']) for g in ground_truth]
    a,b=Counter(words),Counter(targets);a.pop('',None);b.pop('',None)
    tp=sum((a&b).values());n,g=len(words),len(targets)
    return dict(tp=tp,fp=n-tp,fn=g-tp,detected_words=n,gt_words=g,all_outputs=len(rows),ignored_outputs=sum(ignored))


def summarize(rows):
    rows=list(rows);counts={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')}
    tp,n,g=counts['tp'],counts['detected_words'],counts['gt_words']
    return dict(images=len(rows),**counts,precision=tp/n if n else None,recall=tp/g if g else None,f1=2*tp/(n+g) if n+g else None)


def ranked_ids(scores,ids,fraction):
    if not 0<=fraction<=1 or len(set(ids))!=len(ids) or set(scores)!=set(ids):raise ValueError('Invalid routing population')
    if any(v is not None and not math.isfinite(v) for v in scores.values()):raise ValueError('Nonfinite score')
    return sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(ids)*fraction)]


def route_summary(branches,ids,accepted):
    accepted=set(accepted)
    if not accepted<=set(ids) or not ids:raise ValueError('Invalid accepted population')
    total=len(ids)
    return dict(**summarize(branches[s]['light' if s in accepted else 'deepsolo'] for s in ids),
        accepted_ids=sorted(accepted),accepted_images=len(accepted),deepsolo_images=total-len(accepted),
        total_images=total,deepsolo_percent=100*(total-len(accepted))/total,routing_scope='logical cached-output decisions')


def weak_word_labels(rows,ground_truth,iou_threshold):
    def iou(a,b):
        overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
        area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
        return overlap/area if area>0 else 0.
    return np.asarray([float(any(normalized(r['raw_text'])==normalized(g['text']) and iou(r['bbox'],g['bbox'])>=iou_threshold
                                for g in ground_truth)) if normalized(r['raw_text']) else -1. for r in rows],np.float32)
