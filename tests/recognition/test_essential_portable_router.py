import numpy as np
import pytest
from recognition.essential_portable_router import predict_portable_essential


def bundle():
    return dict(mean=[0.]*30,scale=[1.]*30,coef=[0.]*30,intercept=1.,
        map_features=dict(probability_threshold=.5,minimum_component_area=4),
        policies=[dict(target_calibration_acceptance=.5,threshold=2.,tie_sample_id='b')])


def rows(text='abc'):
    return [dict(raw_text=text,bbox=[2,2,8,8],confidence=.98,detector_score=.9)]


@pytest.mark.parametrize('sid,accepted',[('a',True),('b',True),('c',False)])
def test_normalization_and_stable_cutoff_ties(sid,accepted):
    result=predict_portable_essential(bundle(),sid,rows(),[16,16],np.zeros((2,16,16),np.float32))
    assert result['score']==2. and result['use_unet']['0.5']==accepted


@pytest.mark.parametrize('selected',[[],rows('')])
def test_empty_outputs_always_defer(selected):
    result=predict_portable_essential(bundle(),'a',selected,[16,16],np.zeros((2,16,16),np.float32))
    assert result['score'] is None and result['use_unet']['0.5'] is False
