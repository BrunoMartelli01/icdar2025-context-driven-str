"""Shared prediction-only word union, learned correctness and spatial conflicts."""

from collections import Counter
import math

import numpy as np
from rapidfuzz.distance import Levenshtein

from recognition.deepsolo_word_f1 import box_iou
from recognition.uniform_lexical_risk import prepare_words
from recognition.fixed_expert_budget import fixed_words_with_risk
from segmentation.uniform_unet_fusion import normalize

FEATURE_SCHEMA='uniform_detector_confidence_v2'


def add_union_rows(observation, raw, dictionary, lexical_policy):
    emitted, _ = prepare_words(raw, dictionary, lexical_policy)
    observation['union_light_rows'] = fixed_words_with_risk(emitted, observation['probabilities'])
    return observation


def candidates(light, heavy, image_wh, matching_iou):
    rows = [dict(r, union_source='deepsolo', union_id='d:'+str(i)) for i,r in enumerate(heavy)]
    for i,r in enumerate(light):
        word = normalize(r['raw_text'])
        if word and any(normalize(h['raw_text']) == word and box_iou(r['bbox'],h['bbox']) >= matching_iou for h in heavy):
            continue
        rows.append(dict(r, union_source='light', union_id='u:'+str(i), fixed_light_complement=True))
    vectors = []
    w,h = image_wh
    for r in rows:
        is_light = r['union_source']=='light'; other = heavy if is_light else light
        text = normalize(r['raw_text']); box = r['bbox'] or [0.,0.,0.,0.]
        x1,y1,x2,y2 = box; bw,bh = max(0.,x2-x1),max(0.,y2-y1)
        overlaps = [box_iou(r['bbox'],p['bbox']) for p in other]
        j = int(np.argmax(overlaps)) if overlaps else None
        partner = other[j] if j is not None else None
        same = [box_iou(r['bbox'],p['bbox']) for p in other if text and normalize(p['raw_text'])==text]
        # DeepSolo recognition confidence is absent from both ICDAR TRAIN
        # caches. Exclude it uniformly, even when a fresh response supplies it.
        confidence = r.get('sequence_confidence_native',0.) if is_light else 0.
        digits = sum(c.isdigit() for c in text)/max(1,len(text))
        vector = [float(is_light),float(r.get('word_probability',0.)),float(r.get('detector_score',r.get('score',0.))),
            float(confidence),float(is_light and 'sequence_confidence_native' in r),
            x1/w,y1/h,x2/w,y2/h,bw/w,bh/h,bw*bh/(w*h),math.log1p(bw/max(bh,1.)),
            min(len(text),50)/50,digits,float(not text),float(r['bbox'] is None),
            max(overlaps,default=0.),max(same,default=0.),
            Levenshtein.normalized_distance(text,normalize(partner['raw_text'])) if partner is not None else 1.,
            float(partner.get('word_probability',partner.get('score',0.))) if partner is not None else 0.,
            len(light)/50,len(heavy)/50,sum(normalize(p['raw_text'])==text for p in rows)/10]
        if not np.isfinite(vector).all(): raise ValueError('Nonfinite observed word-union feature')
        vectors.append(vector)
    return rows,np.asarray(vectors,dtype=np.float64).reshape(len(rows),24)


def training_labels(rows, truth, ignored):
    remaining = Counter(normalize(g['text']) for g in truth if normalize(g['text']))
    def alignment(i):
        r=rows[i];word=normalize(r['raw_text'])
        return max((box_iou(r['bbox'],g['bbox']) for g in truth if normalize(g['text'])==word),default=0.)
    labels = np.zeros(len(rows),np.int64)
    for i in sorted(range(len(rows)),key=lambda i:(-alignment(i),rows[i]['union_id'])):
        word=normalize(rows[i]['raw_text'])
        if not ignored[i] and word and remaining[word]: labels[i]=1;remaining[word]-=1
    return labels


def select(rows, probabilities, threshold, conflict_iou):
    if len(rows)!=len(probabilities) or not 0<=threshold<=1 or not 0<conflict_iou<=1:
        raise ValueError('Explicit aligned word-union selection required')
    chosen=[]
    for i in sorted(range(len(rows)),key=lambda i:(-probabilities[i],rows[i]['union_source']!='deepsolo',rows[i]['union_id'])):
        p=float(probabilities[i]);r=rows[i]
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid union correctness probability')
        if p<threshold:continue
        if any(box_iou(r['bbox'],q['bbox'])>=conflict_iou for q in chosen):continue
        chosen.append(dict(r,union_probability=p))
    return chosen


def combine(observation, heavy, image_wh, model, policy, settings):
    rows,x=candidates(observation['union_light_rows'],heavy,image_wh,settings['matching_iou'])
    probabilities=model.predict_proba(x)[:,1] if len(x) else []
    return select(rows,probabilities,policy['learned_union_threshold'],settings['conflict_iou'])
