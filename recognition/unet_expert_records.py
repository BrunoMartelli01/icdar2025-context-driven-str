"""Prediction-only U-Net/PARSeq rows in the common fixed-expert schema."""

import math

from recognition.adapted_word_policy import emit
from recognition.common_budget_pipeline import map_features
from recognition.group_word_selection import reading_confidence


def decorate_unet_rows(rows, image_wh):
    width, height = image_wh; result = []; identifiers = set()
    for row in rows:
        row = dict(row); a, b, c, d = row['bbox']; identifier = row['candidate_id']
        if (identifier in identifiers or not 0 <= a < c <= width or not 0 <= b < d <= height or
                not math.isfinite(row['detector_score']) or not 0 <= row['detector_score'] <= 1):
            raise ValueError('Invalid or duplicate native U-Net/PARSeq candidate')
        identifiers.add(identifier)
        row['confidence'] = reading_confidence(row)['confidence_balanced']
        result.append(row)
    return result


def observation(rows, probabilities, image_wh, initial_policy, map_settings):
    raw = decorate_unet_rows(rows, image_wh)
    if initial_policy['lexical'] != 'none':
        raise ValueError('The initial fixed expert emission must be vocabulary-free')
    emitted, _ = emit(raw, None, initial_policy)
    maps = map_features(probabilities, image_wh, raw, map_settings).tolist()
    return raw, emitted, maps
