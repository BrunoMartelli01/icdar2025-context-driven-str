"""Single-image deployment and annotation-free input-boundary tests."""
import numpy as np
import pytest
from recognition.icdar_budget_deployment import predict
from recognition.icdar_budget_router import observation
from recognition.lightweight_unet_router import fit_head,predict_head


def test_single_image_deployment_matches_training_score_and_ties():
    settings=dict(token_probability_cutoffs=[.5,.9],maps=dict(probability_threshold=.5,minimum_component_area=1))
    rows=[dict(bbox=[1,1,8,6],raw_text='word',confidence=.9,detector_score=.8,token_probabilities=[.9,.8])]
    x=observation(rows,[20,10],settings=settings)['ocr'];training=np.stack([x,x+1,x+2,x+3])
    b=fit_head(['ocr'],training,np.arange(4.),np.ones(4),dict(kind='ridge',alpha=1.),seed=1)
    score=float(predict_head(b,x[None])[0])*2
    policy=dict(target_acceptance=.7,threshold=score,tie_key='b')
    result=predict(b,settings,rows,[20,10],tie_key='a',policies=[policy])
    assert result['score']==score and result['accept_light']['0.7']
    assert not predict(b,settings,rows,[20,10],tie_key='c',policies=[policy])['accept_light']['0.7']
    empty=predict(b,settings,[],[20,10],tie_key='a',policies=[policy])
    assert empty['score'] is None and empty['requires_deepsolo']['0.7']
    b['groups']=['ocr','maps']
    with pytest.raises(ValueError):predict(b,settings,rows,[20,10],tie_key='a',policies=[policy])
