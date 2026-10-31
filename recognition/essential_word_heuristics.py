"""Small GT-free routing controls using the already executed word U-Net."""

import math
import numpy as np

from recognition.budget_distilled_router import observation_features


def risk_scores(rows, raw, probabilities, image_wh, lexicon, settings):
    values, names, _ = observation_features(rows, raw, image_wh, probabilities,
                                             settings['observations'])
    features = dict(zip(names, values))
    sequence = [row['sequence_confidence_native'] for row in rows]
    matches = [lexicon.nearest(row['raw_text']) for row in rows]
    missing = float(features['text_uncovered_soft'])
    result = dict(
        weakest_sequence=1-min(sequence) if sequence else 1.,
        raw_surprisal=sum(-math.log(max(row['sequence_confidence_native'],
            np.finfo(np.float64).tiny)) for row in raw),
        rejected_candidates=float(len(raw)-len(rows)),
        word_count=float(len(rows)),
        uncovered_text_mass=missing,
        uncovered_kernel_components=float(features[
            'kernel_'+str(settings['kernel_cut'])+'_missing_components']),
        unmatched_words=float(sum(match is None for match in matches)),
        lexical_edits=float(sum(match is not None and match['distance'] > 0 for match in matches)),
        estimated_word_risk=sum(1-value for value in sequence)+(len(raw)+1)*missing)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Finite inference-only risks required')
    return result


def select_deepsolo(scores, tie_keys, budget):
    if set(scores) != set(tie_keys) or not 0 <= budget <= len(scores):
        raise ValueError('Explicit valid population and fallback budget required')
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('Nonfinite router score')
    return sorted(scores, key=lambda sid: (-scores[sid], tie_keys[sid], sid))[:budget]
