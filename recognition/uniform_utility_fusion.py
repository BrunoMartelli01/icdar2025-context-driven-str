"""One lexical/word-risk/count-utility model family for all three domains."""

import numpy as np

from recognition.uniform_lexical_risk import prepare_words
from recognition.common_budget_pipeline import (
    word_features, image_features, geometry_features, risk_features, map_features, predict_counts,
)
from recognition.fixed_expert_budget import fixed_words_with_risk


GROUP_ORDER = ('raw_ocr', 'tokens', 'post_ocr', 'lexical', 'maps', 'geometry', 'risk')


def word_observation(raw, image_wh, dictionary, lexical_policy, model, admission):
    if not 0 <= admission <= 1:
        raise ValueError('Explicit valid word-admission probability required')
    emitted, trace = prepare_words(raw, dictionary, lexical_policy)
    x = word_features(raw, trace, image_wh)
    p = model.predict_proba(x)[:, 1] if len(x) else np.empty(0)
    rows = [r for r in fixed_words_with_risk(emitted, p) if r['word_probability'] >= admission]
    kept = {int(r['source_row']) for r in rows}
    trace = [dict(t, accepted=i in kept, changed=i in kept and t['changed']) for i, t in enumerate(trace)]
    return dict(rows=rows, probabilities=p.tolist(), trace=trace)


def image_vector(raw, image_wh, probabilities, observation, map_settings, context=None):
    rows, trace = observation['rows'], observation['trace']
    maps = map_features(probabilities, image_wh, rows, map_settings)
    groups = image_features(raw, rows, trace, image_wh, maps)
    groups['geometry'] = geometry_features(rows, image_wh)
    groups['risk'] = risk_features(observation['probabilities'], trace)
    result = np.concatenate([groups[name] for name in GROUP_ORDER])
    if context is not None:
        result = np.concatenate((result, np.asarray(context)))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite observable utility-router feature')
    return result


def score(raw, image_wh, probabilities, dictionary, lexical_policy, word_model,
          count_head, admission, reference, map_settings, context=None):
    observation = word_observation(raw, image_wh, dictionary, lexical_policy, word_model, admission)
    vector = image_vector(raw, image_wh, probabilities, observation, map_settings, context)
    delta = predict_counts(count_head, vector[None])[0]
    observation['score'] = float((2*delta[0]-reference*delta[1])*(len(raw)+1))
    return observation


def paired_f1_lower(base, candidate, indices, alpha):
    """Conditional paired-image bootstrap; not a post-search risk guarantee."""
    a = np.asarray(base, np.int64)
    b = np.asarray(candidate, np.int64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Aligned per-image TP/FP/FN triples required')
    aa, bb = a[indices].sum(axis=1), b[indices].sum(axis=1)
    da, db = 2*aa[:, 0]+aa[:, 1]+aa[:, 2], 2*bb[:, 0]+bb[:, 1]+bb[:, 2]
    fa = np.divide(2*aa[:, 0], da, out=np.zeros(len(aa)), where=da != 0)
    fb = np.divide(2*bb[:, 0], db, out=np.zeros(len(bb)), where=db != 0)
    return float(np.quantile(fb-fa, alpha))
