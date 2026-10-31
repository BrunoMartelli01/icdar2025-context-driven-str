import numpy as np
import pytest
from recognition.essential_threshold_deployment import predict_validated_gate


def profile(mode='absolute_utility'):
    return dict(mode='approximate',validation_selected=True,score_mode=mode,
        head=dict(mean=[0.]*30,scale=[1.]*30,coef=[0.]*30,intercept=1.,map_features=dict(probability_threshold=.5,minimum_component_area=4)),
        policies=dict(yes=dict(threshold=2.),no=dict(threshold=4.),abstain=dict(threshold=None)))


def observations():
    return [dict(raw_text='ABC',bbox=[0,0,4,5],confidence=.9,detector_score=.8),dict(raw_text='D',bbox=[4,0,8,5],confidence=.9,detector_score=.8)]


def test_profile_observation_scores_and_all_fallback_endpoint():
    result=predict_validated_gate(profile(),'x',observations(),[10,10],np.zeros((2,16,16),np.float32))
    assert result['score']==3. and result['use_light']==dict(yes=True,no=False,abstain=False)
    assert not result['ground_truth_used']
    normalized=predict_validated_gate(profile('normalized_utility'),'x',observations(),[10,10],np.zeros((2,16,16),np.float32))
    assert normalized['score']==1. and not any(normalized['use_light'].values())


def test_empty_or_unqualified_profile_cannot_accept():
    result=predict_validated_gate(profile(),'x',[],[10,10],np.zeros((2,16,16),np.float32))
    assert result['score'] is None and not any(result['use_light'].values())
    p=profile();p['validation_selected']=False
    with pytest.raises(ValueError,match='validation-selected'):predict_validated_gate(p,'x',[],[10,10],np.zeros((2,16,16),np.float32))
