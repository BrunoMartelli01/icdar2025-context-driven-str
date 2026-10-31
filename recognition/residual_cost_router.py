"""Approximate, training-free routing costs from uncovered predicted text regions."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import math

import cv2
import numpy as np

from recognition.text_coverage_router import projected_support


NEW_FAMILIES = (
    'residual_only', 'risk_native_count', 'risk_visual_count', 'risk_visual_mass',
    'risk_visual_consensus', 'risk_base_consensus', 'f1_consensus_proxy',
    'base_residual_penalty',
)


def predicted_probabilities(path):
    """The NPZ may contain annotations; access exactly the prediction member."""
    with np.load(path, allow_pickle=False) as archive:
        return archive['probabilities']


def residual_regions(probabilities, image_wh, boxes, settings):
    """Use full foreground components to avoid counting small exposed box edges."""
    p = np.asarray(probabilities)
    threshold = settings['probability_threshold']
    area = settings['minimum_component_area']
    fraction = settings['minimum_uncovered_fraction']
    corroboration = settings['minimum_other_head_support']
    if (p.ndim != 3 or p.shape[0] != 2 or p.shape[1] != p.shape[2]
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not 0 < threshold < 1 or area < 1 or not 0 < fraction <= 1
            or not 0 <= corroboration <= 1):
        raise ValueError('Invalid prediction maps or explicit residual configuration')
    side = p.shape[1]
    w, h = image_wh
    canvas = projected_support(image_wh, [[0, 0, w, h]], side)
    covered = projected_support(image_wh, boxes, side)
    foreground = (p >= threshold) & canvas[None]
    word_supports = [projected_support(image_wh, [box], side) for box in boxes]
    out = {'map_side': side, 'map_shape': list(p.shape), 'map_dtype': str(p.dtype),
           'map_members_accessed': ['probabilities'], 'ground_truth_used': False,
           'canvas_pixels': int(canvas.sum()), 'covered_canvas_pixels': int(covered.sum()),
           'heads': {}}
    for index, name in enumerate(('text', 'kernel')):
        values = p[index].astype(np.float64)
        fg = foreground[index]
        missing = fg & ~covered
        other = foreground[1-index] & ~covered
        weights = np.where(fg, values, 0.)
        word_masses = [float(weights[support].sum()) for support in word_supports]
        positive = [x for x in word_masses if x > 0]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        components = []
        for label in range(1, count):
            x, y, width, height, pixels = map(int, stats[label])
            if pixels < area:
                continue
            component = labels == label
            exposed = component & missing
            exposed_pixels = int(exposed.sum())
            exposed_fraction = exposed_pixels / pixels
            residual = exposed_pixels >= area and exposed_fraction >= fraction
            support = float(other[exposed].mean()) if exposed_pixels else 0.
            components.append({'bbox': [x, y, x+width, y+height], 'pixels': pixels,
                'mass': float(values[component].sum()), 'uncovered_pixels': exposed_pixels,
                'uncovered_fraction': exposed_fraction, 'uncovered_mass': float(values[exposed].sum()),
                'other_head_support': support, 'residual': residual,
                'corroborated': residual and support >= corroboration})
        components.sort(key=lambda c: (c['bbox'][1], c['bbox'][0], c['bbox'][3], c['bbox'][2]))
        # If all word boxes miss predicted foreground, component mass supplies a
        # declared fallback scale. No GT word size or annotation mask is used.
        denominator = (float(np.median(positive)) if positive else
                       float(np.median([c['mass'] for c in components])) if components else 1.)
        selected = [c for c in components if c['residual']]
        supported = [c for c in components if c['corroborated']]
        total_mass = float(weights.sum())
        out['heads'][name] = {
            'components': components, 'foreground_pixels': int(fg.sum()),
            'foreground_mass': total_mass, 'coverage': float(weights[covered].sum()/total_mass) if total_mass else 1.,
            'uncovered_pixels': int(missing.sum()), 'word_foreground_masses': word_masses,
            'word_mass_scale': denominator, 'word_mass_scale_source':
                'positive_predicted_word_median' if positive else 'predicted_component_median' if components else 'empty_neutral',
            'missing_regions': len(selected), 'corroborated_missing_regions': len(supported),
            'missing_word_mass_equivalents': sum(c['uncovered_mass'] for c in selected)/denominator,
            'corroborated_word_mass_equivalents': sum(c['uncovered_mass'] for c in supported)/denominator,
        }
    heads = list(out['heads'].values())
    out['missing_count'] = max(h['missing_regions'] for h in heads)
    out['missing_mass'] = max(h['missing_word_mass_equivalents'] for h in heads)
    out['missing_consensus'] = sum(h['corroborated_missing_regions'] for h in heads)/2
    return out


def word_evidence(rows, consistency, reference):
    """Uncalibrated word evidence: no inference labels or fitted weights."""
    if not reference or sorted(reference) != reference:
        raise ValueError('A sorted frozen TRAIN CLIP reference is required')
    stability = {w['candidate_id']: w['minimum_edit_agreement']
                 for w in consistency['crop_stability']['words']}
    if len(stability) != len(rows) or set(stability) != {r['candidate_id'] for r in rows}:
        raise ValueError('Word/stability population mismatch')
    words = []
    for row in rows:
        native = row['confidence']
        rank = (bisect_left(reference, row['clip_word_advantage'])
                + bisect_right(reference, row['clip_word_advantage']))/(2*len(reference))
        agreement = stability[row['candidate_id']]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (native, rank, agreement)):
            raise ValueError('Invalid uncalibrated word evidence')
        words.append({'candidate_id': row['candidate_id'], 'native': native,
                      'clip_rank': rank, 'crop_agreement': agreement,
                      'visual': (native*rank*agreement)**(1/3)})
    return {'words': words, 'predicted_word_count': len(rows),
            'has_empty_reading': any(not r['raw_text'].strip() for r in rows),
            'native_error_proxy': math.fsum(1-w['native'] for w in words),
            'visual_error_proxy': math.fsum(1-w['visual'] for w in words),
            'probabilities_calibrated': False, 'ground_truth_used': False}


def routing_scores(regions, evidence, baseline):
    """Higher is safer. A misreading contributes both one FP and one FN proxy."""
    n = evidence['predicted_word_count']
    if not n or evidence['has_empty_reading']:
        return dict.fromkeys(NEW_FAMILIES)
    native = evidence['native_error_proxy']; visual = evidence['visual_error_proxy']
    count = regions['missing_count']; mass = regions['missing_mass']; consensus = regions['missing_consensus']
    base = baseline['all_checks_softcrop']
    values = [-count, -(2*native+count), -(2*visual+count), -(2*visual+mass),
              -(2*visual+consensus), -(2*n*(1-base)+consensus),
              2*(n-visual)/(2*n+consensus), baseline['base_proxy_f1']/(1+consensus)]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite residual routing score')
    return dict(zip(NEW_FAMILIES, values))
