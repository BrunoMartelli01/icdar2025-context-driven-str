"""Single-image interface for the frozen, validation-budgeted essential gate."""
from recognition.lightweight_unet_router import predict_observation_gate


def predict_essential_budget(deployment, sample_id, rows, image_wh, *, probabilities=None):
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError('Stable nonempty sample ID required for deterministic cutoff ties')
    observation = predict_observation_gate(dict(bundle=deployment['bundle'],
        feature_settings=deployment['feature_settings'], projections={}, policies={}), rows, image_wh,
        probabilities=probabilities)
    score = observation['predicted_utility']
    observation['use_unet'] = {str(p['target_calibration_acceptance']): bool(score is not None and
        (score > p['threshold'] or score == p['threshold'] and sample_id <= p['tie_sample_id']))
        for p in deployment['policies']}
    return observation
