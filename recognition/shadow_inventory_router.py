"""Approximate whole-image routing from an independent text-region inventory.

No annotation, fallback reading, or evaluation mask is accepted by this module.
Capture-recapture is an uncalibrated diagnostic: detector independence is not
established and neither its count nor OCR confidence is a probability guarantee.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def area(box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Finite positive exclusive XYXY envelope required')
    return (box[2]-box[0])*(box[3]-box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def union_coverage(box, others):
    """Exact rectangle-union intersection; overlaps never count twice."""
    total = area(box)
    clipped = [[max(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), min(box[3], b[3])]
               for b in others if intersection(box, b) > 0]
    xs = sorted({v for b in clipped for v in (b[0], b[2])})
    covered = 0.
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in clipped if b[0] < right and b[2] > left)
        length = 0.; end = -math.inf
        for bottom, top in intervals:
            length += max(0., top-max(bottom, end)); end = max(end, top)
        covered += (right-left)*length
    return covered/total


def geometric_pairs(light, shadow, threshold):
    """Maximum cardinality, then maximum IoU; one-to-one deterministic input order."""
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    if not light or not shadow:
        return []
    values = np.array([[intersection(a['bbox'], b['bbox'])/
                       (area(a['bbox'])+area(b['bbox'])-intersection(a['bbox'], b['bbox']))
                       for b in shadow] for a in light])
    bonus = min(len(light), len(shadow))+1
    weights = np.where(values >= threshold, bonus+values, 0.)
    ii, jj = linear_sum_assignment(weights, maximize=True)
    return [{'light_index': int(i), 'shadow_index': int(j), 'iou': float(values[i, j])}
            for i, j in zip(ii, jj) if values[i, j] >= threshold]


def probe_indices(light, shadow, settings):
    """Select at most K suspects BEFORE observing any shadow OCR output."""
    for r in light+shadow:
        area(r['bbox'])
    boxes = [r['bbox'] for r in light]
    risks = []
    for i, r in enumerate(shadow):
        coverage = union_coverage(r['bbox'], boxes)
        near = [a['confidence'] for a in light if intersection(a['bbox'], r['bbox']) > 0]
        risk = r['score']*((1-coverage)+settings['covered_uncertainty_weight']*(1-min(near, default=0.)))
        risks.append((risk, i))
    return [i for _, i in sorted(risks, key=lambda p: (-p[0], p[1]))[:settings['maximum_probes']]]


def quality(row):
    """Explicit balanced per-character evidence, retaining punctuation/symbols."""
    if not row['raw_text'].strip():
        return 0.
    p = row['token_probabilities']
    if not p or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p):
        raise ValueError('Finite native character probabilities required')
    return math.sqrt(min(p)*math.exp(sum(math.log(max(v, 1e-300)) for v in p)/len(p)))


def inventory_features(light, regions, ocr_rows, settings):
    """Four prespecified rules plus compact inputs for a benefit regressor."""
    shadow = [r for r in regions if r['valid']]
    if len({r['source_index'] for r in shadow}) != len(shadow):
        raise ValueError('Duplicate shadow region identity')
    by_id = {}
    for row in ocr_rows:
        by_id.setdefault(row['source_index'], []).append(row)
    if set(by_id) != {r['source_index'] for r in shadow}:
        raise ValueError('Every valid independent region requires both frozen OCR views')
    for group in by_id.values():
        if [r['view'] for r in group] != ['axis_aligned', 'rectified']:
            raise ValueError('Expected original/rectified views in declared tie order')
    readings = [max(by_id[r['source_index']], key=quality) for r in shadow]
    pairs = geometric_pairs(light, shadow, settings['matching_iou'])
    li = {p['light_index']: p['shadow_index'] for p in pairs}
    sj = {p['shadow_index']: p['light_index'] for p in pairs}
    selected = probe_indices(light, shadow, settings)
    cover = [union_coverage(r['bbox'], [a['bbox'] for a in light]) for r in shadow]
    reverse = [union_coverage(a['bbox'], [r['bbox'] for r in shadow]) for a in light]
    q = [math.sqrt(r['score']*quality(s)) for r, s in zip(shadow, readings)]
    if any(not math.isfinite(a['confidence']) or not 0 <= a['confidence'] <= 1 for a in light):
        raise ValueError('Light OCR confidence must be finite in [0,1]')
    native_error = sum(1-a['confidence'] for a in light)
    missing_geometry = sum(r['score']*(1-c) for r, c in zip(shadow, cover))
    unsupported = sum(a['confidence']*(1-c) for a, c in zip(light, reverse))
    def equal(i, j):
        return bool(light[i]['raw_text'].strip()) and light[i]['raw_text'].casefold() == readings[j]['raw_text'].casefold()
    agreements = [p for p in pairs if equal(p['light_index'], p['shadow_index'])]
    disagreements = sum(min(light[p['light_index']]['confidence'], q[p['shadow_index']])
                        for p in pairs if not equal(p['light_index'], p['shadow_index']))
    confirmed_missing = sum(q[j]*(1-cover[j]) for j in selected)
    probe_disagreement = sum(min(light[sj[j]]['confidence'], q[j]) for j in selected if j in sj and not equal(sj[j], j))
    n, m, matched = len(light), len(shadow), len(pairs)
    # Chapman count is a heuristic here: correlated false negatives and false
    # positives violate its classical capture assumptions.
    chapman = (n+1)*(m+1)/(matched+1)-1
    unseen = max(0., chapman-(n+m-matched))
    y = Counter(r['raw_text'].casefold() for r in light if r['raw_text'].strip())
    z = Counter(r['raw_text'].casefold() for r in readings if r['raw_text'].strip())
    bag_matches = sum((y & z).values())
    bag_dice = 2*bag_matches/(n+m) if n+m else 0.
    fields = {'shadow_log_count': math.log1p(m), 'shadow_log_matches': math.log1p(matched),
        'shadow_coverage': sum(cover)/m if m else 0., 'light_shadow_coverage': sum(reverse)/n if n else 0.,
        'shadow_log_missing_geometry': math.log1p(missing_geometry), 'shadow_log_unsupported': math.log1p(unsupported),
        'shadow_log_confirmed_missing': math.log1p(confirmed_missing),
        'shadow_log_probe_disagreement': math.log1p(probe_disagreement),
        'shadow_log_disagreement': math.log1p(disagreements), 'shadow_bag_dice': bag_dice,
        'shadow_log_recapture_unseen': math.log1p(unseen), 'shadow_log_recapture_count': math.log1p(chapman),
        'shadow_spatial_lexical_agreement': len(agreements)/max(1, matched),
        'shadow_mean_ocr_evidence': sum(q)/m if m else 0.,
        'shadow_probe_count': float(len(selected)), 'shadow_pair_count_difference': float(m-n)}
    scores = {
        'geometry': -(2*native_error+missing_geometry+unsupported),
        'probe': -(2*native_error+confirmed_missing+2*probe_disagreement+unsupported),
        'recapture': -(2*native_error+max(0., chapman-n)),
        'agreement': bag_dice*sum(a['confidence'] for a in light)/max(1, n),
    }
    eligible = bool(light) and all(r['raw_text'].strip() for r in light)
    if not eligible:
        scores = dict.fromkeys(scores)
    if any(not math.isfinite(v) for v in fields.values()):
        raise ValueError('Nonfinite shadow-inventory feature')
    return fields, scores, {'probe_region_ids': [shadow[j]['source_index'] for j in selected],
        'pairs': pairs, 'agreements': agreements, 'region_coverage': cover,
        'valid_regions': m, 'rejected_regions': len(regions)-m,
        'selected_ocr_candidates': [r['candidate_id'] for r in readings],
        'ground_truth_used': False, 'chapman_is_calibrated_count': False}
