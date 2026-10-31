"""GT-blind single-image interface for exported validation-selected gate profiles."""
from recognition.essential_portable_router import predict_portable_essential
from recognition.essential_threshold_frontier import observable_length_rescale

def predict_validated_gate(profile,sample_id,rows,image_wh,probabilities):
    if profile['mode']!='approximate' or not profile['validation_selected']:
        raise ValueError('Explicit approximate validation-selected profile required')
    head=dict(profile['head'],policies=[])
    score=predict_portable_essential(head,sample_id,rows,image_wh,probabilities)['score']
    if profile['score_mode']=='normalized_utility':score=observable_length_rescale(score,rows)
    elif profile['score_mode']!='absolute_utility':raise ValueError('Unknown explicit score convention')
    return dict(score=score,use_light={name:bool(score is not None and policy['threshold'] is not None and score>=policy['threshold'])
        for name,policy in profile['policies'].items()},ground_truth_used=False)
