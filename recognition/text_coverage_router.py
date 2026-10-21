"""APPROXIMATE image routing: reading confidence plus inference-only text coverage."""
from __future__ import annotations

import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

from recognition.clip_crop_admission import percentile
from recognition.clip_router_prompt_ablation import budget_accept, coverage_threshold
from recognition.yolo_router_comparison import aggregate


def projected_support(image_wh, boxes, side):
    """Exclusive original-frame rectangles, same center-pad/nearest frame as the U-Net."""
    w, h = image_wh
    if min(w, h, side) < 1:
        raise ValueError('Invalid coverage dimensions')
    original = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(original)
    for box in boxes:
        if len(box) != 4 or any(not math.isfinite(v) for v in box):
            raise ValueError('Invalid coverage box')
        x1, y1, x2, y2 = box
        if any(int(v) != v for v in box) or not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
            raise ValueError('Coverage boxes must be clipped, exclusive integer rectangles')
        draw.rectangle((x1, y1, x2-1, y2-1), fill=255)
    square = Image.new('L', (max(w, h), max(w, h)), 0)
    square.paste(original, ((max(w, h)-w)//2, (max(w, h)-h)//2))
    return np.asarray(square.resize((side, side), Image.Resampling.NEAREST)) > 0


def coverage_features(probabilities, image_wh, boxes, *, probability_threshold, minimum_component_area):
    """Read only predicted Text/Kernel maps; never read GT/validity/owners from an NPZ."""
    p = np.asarray(probabilities)
    if (p.ndim != 3 or p.shape[0] != 2 or p.shape[1] != p.shape[2]
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not 0 < probability_threshold < 1 or minimum_component_area < 1):
        raise ValueError('Invalid predicted Text/Kernel probabilities or coverage configuration')
    side = p.shape[1]
    w, h = image_wh
    canvas = projected_support(image_wh, [[0, 0, w, h]], side)
    covered = projected_support(image_wh, boxes, side)
    result = {'probability_threshold': probability_threshold, 'map_side': side,
              'boxes': len(boxes), 'ground_truth_used': False}
    for name, values in zip(('text', 'kernel'), p):
        foreground = (values >= probability_threshold) & canvas
        weights = np.where(foreground, values, 0).astype(np.float64)
        mass = float(weights.sum())
        missing = foreground & ~covered
        n, _, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= minimum_component_area]
        result.update({name+'_coverage': float(weights[covered].sum()/mass) if mass else 1.,
            name+'_foreground_mass': mass, name+'_foreground_pixels': int(foreground.sum()),
            name+'_uncovered_pixels': int(missing.sum()), name+'_uncovered_components': len(areas),
            name+'_largest_uncovered_component': max(areas, default=0), name+'_mask_empty': mass == 0})
    return result


def scores(features, reference, weight, coverage_feature):
    if not 0 <= weight <= 1 or not reference:
        raise ValueError('Invalid coverage mixture/reference')
    result = {}
    for sid, value in features.items():
        base = value['confidence_score']
        coverage = value['coverage'][coverage_feature]
        if not math.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError('Invalid image coverage')
        if base is None:
            result[sid] = None
        elif not math.isfinite(base) or not 0 <= base <= 1:
            raise ValueError('Invalid image reading score')
        else:
            result[sid] = base if weight == 0 else (1-weight)*base+weight*percentile(reference, coverage)
    return result


def fit_selector(features, branches, fit_ids, *, weights, coverage_fields, coverage_targets, primary_coverage):
    """Select a mixture using only the explicitly supplied TRAIN-calibration population."""
    if (not fit_ids or len(set(fit_ids)) != len(fit_ids)
            or set(features) != set(branches) or set(features) != set(fit_ids)):
        raise ValueError('Fit must receive exactly the TRAIN calibration population')
    trials = []
    for field in coverage_fields:
        reference = sorted(features[s]['coverage'][field] for s in fit_ids)
        for weight in weights:
            values = scores(features, reference, weight, field)
            accepted = budget_accept(values, fit_ids, round(len(fit_ids)*primary_coverage))
            if len(accepted) != round(len(fit_ids)*primary_coverage):
                raise ValueError('Insufficient nonempty readings to meet the bypass budget')
            combined = aggregate([branches[s]['light' if s in accepted else 'deepsolo'] for s in fit_ids])
            trials.append({'coverage_field': field, 'weight': weight, 'reference': reference,
                           'combined': combined, 'accepted_ids': sorted(accepted)})
    # A tie retains the simpler confidence-only gate; field order is predeclared.
    winner = min(trials, key=lambda t: (-t['combined']['f1'], t['weight'], coverage_fields.index(t['coverage_field'])))
    values = scores(features, winner['reference'], winner['weight'], winner['coverage_field'])
    return {'fit_ids': list(fit_ids), 'winner': winner, 'trials': trials,
        'scalar_thresholds': {str(f): coverage_threshold(values, fit_ids, f) for f in coverage_targets},
        'primary_coverage': primary_coverage, 'test_labels_used': False,
        'objective': 'TRAIN-calibration micro F1 at fixed bypass budget',
        'batch_budget_rule': 'descending score then sample ID; no GT; scalar thresholds reported separately'}
