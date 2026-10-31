"""One learned word-risk, expected-error gate and conditional fusion for all domains."""

import numpy as np

from recognition.common_budget_pipeline import choose_light
from recognition.fixed_expert_budget import (
    fixed_trace, observed_features, fixed_words_with_risk, fixed_heuristic,
    conditional_complement,
)
from recognition.fallback_word_fusion import replace_fallback_transcripts


def features(raw, emitted, image_wh):
    trace = fixed_trace(raw, emitted)
    # The same 18 observable word features for every dataset. No predicted-map
    # model, image-wide encoder, fallback output or GT is consumed here.
    values, _ = observed_features(raw, emitted, trace, image_wh, None)
    return values


def observe(raw, emitted, image_wh, model, admission_threshold):
    if not 0 <= admission_threshold <= 1:
        raise ValueError('Explicit valid admission threshold required')
    x = features(raw, emitted, image_wh)
    p = model.predict_proba(x)[:, 1] if len(x) else np.empty(0)
    words = [r for r in fixed_words_with_risk(emitted, p)
             if r['word_probability'] >= admission_threshold]
    return dict(rows=words, probabilities=p.tolist(),
                score=fixed_heuristic(raw, words, p, 'expected_errors'))


def route(observations, rgb_hashes, fallback_images, guard):
    scores = {s: o['score'] for s, o in observations.items()}
    eligible = [s for s, o in observations.items() if o['rows']] if guard else None
    accepted = choose_light(scores, rgb_hashes, fallback_images, eligible)
    fallback = sorted(set(scores)-set(accepted))
    return dict(accepted_ids=sorted(accepted), fallback_ids=fallback,
                deepsolo_images=len(fallback), total_images=len(scores),
                deepsolo_percentage=100*len(fallback)/len(scores),
                ground_truth_used=False, fallback_outputs_loaded=False)


def combine(observation, fallback_rows, fusion, complement):
    replaced = replace_fallback_transcripts(observation['rows'], fallback_rows, fusion)
    return conditional_complement(observation['rows'], replaced, complement)
