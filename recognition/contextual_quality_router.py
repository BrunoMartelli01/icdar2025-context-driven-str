"""APPROXIMATE, frozen contextual probes for whole-image OCR routing.

No labels enter these features; TRAIN-only empirical references rescale units.
None means unavailable evidence, never a fabricated confidence value.
"""
from __future__ import annotations

import math
import numpy as np

from recognition.clip_crop_admission import percentile
from recognition.clip_router_prompt_ablation import mutations


FAMILIES = (
    'coverage_baseline', 'all_checks_softcrop', 'paper_caption_control',
    'literal_only', 'base_literal', 'mpnet_contrast', 'base_mpnet',
    'clip_plain_margin', 'clip_context_margin', 'clip_context_gain',
    'base_clip_plain', 'base_clip_context', 'base_clip_gain',
    'lm_literal', 'lm_coherence', 'base_lm_literal', 'base_lm_coherence',
)
REFERENCE_FIELDS = ('mpnet_contrast', 'clip_plain_margin', 'clip_context_margin',
                    'clip_context_gain', 'lm_literal', 'lm_coherence')


def word_variants(word):
    """Original plus fixed annotation-free near-spelling controls."""
    return [word, *mutations(word)] if word.strip() else []


def lower_quantile(values, q):
    if not values:
        return None
    if not 0 <= q <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError('Invalid contextual quantile')
    return float(np.quantile(values, q, method='linear'))


def spelling_margin(similarities):
    if len(similarities) < 2 or any(not math.isfinite(v) for v in similarities):
        raise ValueError('Spelling margin requires original and at least one control')
    return float(similarities[0] - max(similarities[1:]))


def fit_reference(raw_features, ids):
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(raw_features):
        raise ValueError('Invalid TRAIN-only reference population')
    ref = {k: sorted(raw_features[s]['signals'][k] for s in ids
                    if raw_features[s]['signals'][k] is not None)
           for k in REFERENCE_FIELDS}
    if any(not values for values in ref.values()):
        raise ValueError('Missing TRAIN evidence for a predeclared reference field')
    return {'reference_ids': list(ids), 'fields': ref, 'labels_used': False}


def image_scores(raw, reference):
    if not raw['eligible']:
        return dict.fromkeys(FAMILIES)
    base = raw['source_scores']['all_checks_softcrop']
    if base is None:
        raise ValueError('Eligible OCR lacks baseline consistency score')
    signals = raw['signals']
    scaled = {k: percentile(reference['fields'][k], signals[k])
              if signals[k] is not None else None for k in REFERENCE_FIELDS}
    mix = lambda value: math.sqrt(base * value) if value is not None else None
    literal = signals['literal_recall']
    result = {
        'coverage_baseline': raw['source_scores']['coverage_baseline'],
        'all_checks_softcrop': base,
        'paper_caption_control': signals['paper_caption_control'],
        'literal_only': literal,
        # Caption omission is not a contradiction: absence leaves this control neutral.
        'base_literal': base * (literal if literal is not None else 1.),
        **{k: scaled[k] for k in REFERENCE_FIELDS},
        'base_mpnet': mix(scaled['mpnet_contrast']),
        'base_clip_plain': mix(scaled['clip_plain_margin']),
        'base_clip_context': mix(scaled['clip_context_margin']),
        'base_clip_gain': mix(scaled['clip_context_gain']),
        'base_lm_literal': mix(scaled['lm_literal']),
        'base_lm_coherence': mix(scaled['lm_coherence']),
    }
    if set(result) != set(FAMILIES) or any(v is not None and not math.isfinite(v)
                                          for v in result.values()):
        raise ValueError('Invalid contextual image scores')
    return {name: result[name] for name in FAMILIES}
