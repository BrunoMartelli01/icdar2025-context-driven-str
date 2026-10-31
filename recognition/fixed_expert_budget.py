"""Shared count-utility gate around fixed observable expert word emissions.

APPROXIMATE extension: the gate estimates risk without editing or dropping
the expert's words. Predicted-map summaries are optional and explicitly marked.
No annotation or actual fallback output is an input to this feature boundary.
"""

import numpy as np

from localize.independent_word_slots import overlap_iou
from recognition.common_budget_pipeline import (
    normalize, word_features, risk_features, image_features, predict_counts,
)


def fixed_trace(raw, emitted, supplied=None):
    indices = [int(r['source_row']) for r in emitted]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(raw) for i in indices):
        raise ValueError('Fixed emissions require distinct valid raw source indices')
    if any(list(r['bbox']) != list(raw[int(r['source_row'])]['bbox']) for r in emitted):
        raise ValueError('Fixed emissions must retain their raw geometry')
    if supplied is not None:
        if len(supplied) != len(raw) or {i for i, t in enumerate(supplied) if t['accepted']} != set(indices):
            raise ValueError('Fixed expert trace and emissions differ')
        return supplied
    kept = {int(r['source_row']): r for r in emitted}
    return [dict(source_row=i, original_text=r['raw_text'], active=False,
        accepted=i in kept, match=None, potential_match=None,
        changed=i in kept and normalize(kept[i]['raw_text']) != normalize(r['raw_text']),
        confidence=r['confidence']) for i, r in enumerate(raw)]


def training_word_labels(raw, emitted, truth):
    """Correctness of emitted text, or raw counterfactual for rejected words.

    This is a lexical membership training proxy. Evaluation always retains
    multiset multiplicity, ignored outputs and all care denominators.
    """
    fixed_trace(raw, emitted)
    chosen = {int(r['source_row']): r for r in emitted}
    target = {normalize(g['text']) for g in truth if normalize(g['text'])}
    return np.asarray([int(bool(normalize(chosen.get(i, row)['raw_text'])) and
        normalize(chosen.get(i, row)['raw_text']) in target) for i, row in enumerate(raw)])


def observed_features(raw, emitted, trace, image_wh, map_summary):
    fixed_trace(raw, emitted, trace)
    # Zero summaries mean a missing sensor, with BOTH availability coordinates
    # zero; an observed empty map instead has its text-available coordinate one.
    if map_summary is None:
        maps = np.zeros(14, dtype=np.float64)
    else:
        maps = np.asarray(map_summary, dtype=np.float64)
        if maps.shape != (14,) or not np.isfinite(maps).all() or maps[-2] != 1:
            raise ValueError('A supplied map summary must represent a real predicted text map')
    x = word_features(raw, trace, image_wh)
    groups = image_features(raw, emitted, trace, image_wh, maps)
    if not np.isfinite(x).all() or any(not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('Nonfinite observable fixed-expert features')
    return x, groups


def with_word_risk(groups, probabilities, trace):
    return dict(groups, risk=risk_features(probabilities, trace))


def fixed_words_with_risk(emitted, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid fixed-word correctness probabilities')
    if any(not 0 <= int(r['source_row']) < len(p) for r in emitted):
        raise ValueError('Fixed word outside risk source population')
    return [dict(r, word_probability=float(p[int(r['source_row'])])) for r in emitted]


def conditional_complement(light, fallback, policy):
    """Consume fallback words only after routing; preserve all fallback instances."""
    result = [dict(r) for r in fallback]
    threshold = policy['confidence']
    if threshold is None:
        return result
    if not 0 <= threshold <= 1 or not 0 < policy['maximum_overlap'] <= 1:
        raise ValueError('Explicit valid complement policy required')
    for row in light:
        if row['word_probability'] < threshold:
            continue
        if any(other['bbox'] is not None and overlap_iou(row['bbox'], other['bbox']) >=
                policy['maximum_overlap'] for other in result):
            continue
        result.append(dict(row, fixed_light_complement=True))
    return result


def fixed_heuristic(raw, emitted, probabilities, name):
    p = np.asarray(probabilities)
    kept = {int(r['source_row']) for r in emitted}
    if name == 'expected_errors':
        return -float(sum((1-v) if i in kept else v for i, v in enumerate(p)))
    if name == 'weakest_word':
        return min((float(p[i]) for i in kept), default=0.)
    if name == 'weakest_sequence':
        return min((r['sequence_confidence_native'] for r in emitted), default=0.)
    raise ValueError('Unknown explicit fixed-expert heuristic')


def count_scores(bundles, matrices, scales, reference):
    pairs = np.mean([predict_counts(b, matrices[tuple(b['groups'])]) for b in bundles], axis=0)
    return (2*pairs[:, 0]-reference*pairs[:, 1])*np.asarray(scales)
