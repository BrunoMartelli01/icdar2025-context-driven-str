"""APPROXIMATE frozen-model image gates; labels enter calibration/evaluation only."""
from __future__ import annotations

import math
import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

from localize.independent_word_slots import overlap_iou
from recognition.clip_crop_admission import percentile
from recognition.clip_router_evaluation import fixed_image_score
from recognition.yolo_router_comparison import aggregate

FAMILIES = ('native_clip', 'coverage_baseline', 'native_only', 'clip_only',
    'scale_only', 'crop_only', 'coverage_only', 'confidence_scale', 'confidence_crop',
    'confidence_coverage', 'all_checks', 'all_checks_min', 'all_checks_softcrop')


def scale_agreement(left, right, min_iou):
    """Maximum-cardinality one-to-one exact-casefold+IoU matching; no GT words."""
    if not 0 < min_iou <= 1:
        raise ValueError('Invalid scale-pair IoU')
    for rows in (left,right):
        if len({r['candidate_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate scale candidate IDs')
    left=sorted(left,key=lambda r:r['candidate_id']);right=sorted(right,key=lambda r:r['candidate_id'])
    if not left or not right:
        return {'score':0.,'pairs':[],'left_words':len(left),'right_words':len(right),'empty_scale':True}
    weights=np.full((len(left),len(right)+len(left)),-np.inf)
    bonus=1+min(len(left),len(right))
    for i,a in enumerate(left):
        weights[i,len(right)+i]=0
        for j,b in enumerate(right):
            iou=overlap_iou(a['bbox'],b['bbox'])
            if a['raw_text'] and a['raw_text'].casefold()==b['raw_text'].casefold() and iou>=min_iou:
                weights[i,j]=bonus+iou
    ii,jj=linear_sum_assignment(weights,maximize=True)
    pairs=[{'left':left[i]['candidate_id'],'right':right[j]['candidate_id'],
        'iou':overlap_iou(left[i]['bbox'],right[j]['bbox'])} for i,j in zip(ii,jj) if j<len(right)]
    return {'score':2*len(pairs)/(len(left)+len(right)),'pairs':pairs,
        'left_words':len(left),'right_words':len(right),'empty_scale':False}


def crop_stability(rows, views, fractions):
    expected={(r['candidate_id'],f) for r in rows for f in fractions}
    bykey={(v['original_candidate_id'],v['fraction']):v for v in views}
    if len(bykey)!=len(views) or set(bykey)!=expected or len(set(fractions))!=len(fractions) or not fractions:
        raise ValueError('Crop views must cover each original word and each declared fraction exactly once')
    words=[]
    for r in rows:
        original=r['raw_text'].casefold()
        strings=[bykey[(r['candidate_id'],f)]['raw_text'].casefold() for f in fractions]
        words.append({'candidate_id':r['candidate_id'],
            'all_views_exact':bool(original) and all(s==original for s in strings),
            'minimum_edit_agreement':min(float(Levenshtein.normalized_similarity(original,s))
                if original and s else 0. for s in strings)})
    return {'exact_fraction':sum(w['all_views_exact'] for w in words)/len(words) if words else 0.,
        'edit_agreement':sum(w['minimum_edit_agreement'] for w in words)/len(words) if words else 0.,
        'words':words,'fractions':fractions}


def make_features(base, high, views, *, reference, previous_selector, fractions, min_iou):
    rows=base['rows'];pair=scale_agreement(rows,high['rows'],min_iou)
    stability=crop_stability(rows,views,fractions)
    confidence=fixed_image_score(rows,reference)
    if confidence is None:
        scores=dict.fromkeys(FAMILIES)
    else:
        native=min(percentile(reference['native'],r['confidence']) for r in rows)
        clip=min(percentile(reference['word_advantage'],r['clip_word_advantage']) for r in rows)
        cov=min(base['coverage']['text_coverage'],base['coverage']['kernel_coverage'])
        scale=pair['score'];crop=stability['exact_fraction'];softcrop=stability['edit_agreement']
        previous=(confidence if previous_selector['weight']==0 else
            (1-previous_selector['weight'])*confidence+previous_selector['weight']*
            percentile(previous_selector['reference'],base['coverage'][previous_selector['coverage_field']]))
        scores=dict(zip(FAMILIES,[confidence,previous,native,clip,scale,crop,cov,
            math.sqrt(confidence*scale),math.sqrt(confidence*crop),math.sqrt(confidence*cov),
            (confidence*scale*crop*cov)**.25,min(confidence,scale,crop,cov),
            (confidence*scale*softcrop*cov)**.25]))
        if any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
            raise ValueError('Invalid frozen-model consistency score')
    return {'scores':scores,'scale_agreement':pair,'crop_stability':stability,
        'coverage':base['coverage'],'ground_truth_used':False,'models_trained':False,
        'original_word_outputs_changed':False}


def good_image(counts, threshold):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'));den=2*tp+fp+fn
    if min(tp,fp,fn)<0 or not 0<threshold<=1:
        raise ValueError('Invalid quality counts/threshold')
    # Avoid binary rounding at rational F1 boundaries such as exactly4/5.
    from fractions import Fraction
    return bool(den) and Fraction(2*tp,den)>=Fraction(str(threshold))


def image_precision_interval(good, total):
    if not 0<=good<=total:raise ValueError('Invalid image-quality denominator')
    return ([float(beta.ppf(.025,good,total-good+1)) if good else 0.,
             float(beta.ppf(.975,good+1,total-good)) if good<total else 1.] if total else [None,None])


def calibrate(scores, branches, ids, *, target, quality_threshold, minimum_accepted):
    """Empirical threshold calibration, not a certified risk bound or NN training."""
    if (not ids or len(set(ids))!=len(ids) or set(scores)!=set(ids) or set(branches)!=set(ids)
            or not 0<target<=1 or minimum_accepted<1):
        raise ValueError('Threshold calibration must receive exactly its declared TRAIN IDs')
    trials=[]
    for threshold in sorted({v for v in scores.values() if v is not None}):
        accepted={s for s in ids if scores[s] is not None and scores[s]>=threshold}
        good=sum(good_image(branches[s]['light'],quality_threshold) for s in accepted)
        if len(accepted)>=minimum_accepted and good/len(accepted)>=target:
            trials.append({'threshold':threshold,'accepted_images':len(accepted),'good_images':good,
                'good_fraction':good/len(accepted)})
    best=max(trials,key=lambda t:(t['accepted_images'],t['good_fraction'],t['threshold'])) if trials else None
    return {'status':'EMPIRICALLY_CALIBRATED' if best else 'ABSTAIN_ALL_NO_EMPIRICAL_THRESHOLD',
        'threshold':best['threshold'] if best else None,'calibration':best,
        'calibration_ids':list(ids),'target_good_fraction':target,
        'quality_f1_threshold':quality_threshold,'minimum_accepted':minimum_accepted,
        'formal_risk_guarantee':False,'test_labels_used':False}


def evaluate(branches, ids, accepted, quality_threshold):
    accepted=set(accepted)
    if not accepted<=set(ids) or len(set(ids))!=len(ids):raise ValueError('Invalid evaluation population')
    good={s for s in ids if good_image(branches[s]['light'],quality_threshold)}
    n=len(accepted);g=len(accepted & good)
    return {'images':len(ids),'accepted_images':n,'fallback_images':len(ids)-n,
        'bypass_fraction':n/len(ids) if ids else None,'accepted_ids':sorted(accepted),
        'accepted_good_images':g,'accepted_bad_images':n-g,'available_good_images':len(good),
        'good_fraction_among_accepted':g/n if n else None,
        'good_fraction_ci95':image_precision_interval(g,n),
        'good_image_recall':g/len(good) if good else None,
        'combined':aggregate(branches[s]['light' if s in accepted else 'deepsolo'] for s in ids),
        'accepted_words':aggregate(branches[s]['light'] for s in ids if s in accepted)}
