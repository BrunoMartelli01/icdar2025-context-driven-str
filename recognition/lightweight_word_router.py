"""Small pre-fallback features for a word-admission image router."""

import numpy as np

from recognition.common_budget_pipeline import (
    choose_light, geometry_features, predict_counts,
)
from recognition.lightweight_unet_router import ocr_summary, token_summary
from recognition.lightweight_word_selection import risk_summary


def image_groups(proposed, selected, probabilities, image_wh, token_cutoffs):
    raw = [r for r in proposed if not r['lexical_alternative']]
    return dict(raw=ocr_summary(raw, image_wh), post=ocr_summary(selected, image_wh),
        tokens=token_summary(raw, token_cutoffs),
        risk=risk_summary(proposed, selected, probabilities),
        geometry=geometry_features(selected, image_wh))


def image_score(groups, bundle, reference, scale):
    x = np.concatenate([groups[g] for g in bundle['groups']])[None]
    predicted = predict_counts(bundle, x)[0]
    return float(2*predicted[0]-reference*predicted[1])*scale


def exact_plan(scores, rgb_hashes, fallback_images):
    accepted = choose_light(scores, rgb_hashes, fallback_images)
    fallback = sorted(set(scores)-set(accepted))
    return dict(accepted_ids=accepted, fallback_ids=fallback,
        deepsolo_images=len(fallback), total_images=len(scores),
        deepsolo_percentage=100*len(fallback)/len(scores),
        scope='Pre-fallback exact batch plan; logical cache use is not online inference')
