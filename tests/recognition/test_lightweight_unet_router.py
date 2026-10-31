import math

import numpy as np
import pytest

from recognition.lightweight_unet_router import (ocr_summary,token_summary,mask_summary,feature_cost,
    conformal_correction,lower_bound_accept,calibration_threshold,threshold_accept,fit_head,predict_head,predict_observation_gate,context_summary)
from recognition.unet_corrected_token_interfaces import route_summary


def row(text='A',box=(0,0,10,10),confidence=.9):
    return dict(raw_text=text,bbox=list(box),confidence=confidence,detector_score=.8,token_probabilities=[.7,.9])


def metric(tp,n,g):
    return dict(tp=tp,fp=n-tp,fn=g-tp,detected_words=n,gt_words=g,all_outputs=n,ignored_outputs=0)


def test_ocr_features_exactly_match_qualified_eighteen_scalars():
    from recognition.token_context_router import base_features
    for rows in [[],[row()],[row(),row('B!',(20,10,40,20),.4)]]:
        assert np.allclose(ocr_summary(rows,(60,30)),base_features(rows,(60,30)),rtol=0,atol=1e-7)


def test_token_features_validate_parent_probabilities_and_empty_words():
    assert token_summary([],[.5,.9]).tolist()==[0.]*8
    x=token_summary([row()],[.5,.9]);assert x.shape==(8,) and x[4]==0 and x[5]==.5
    with pytest.raises(ValueError):token_summary([dict(row(),token_probabilities=[np.nan])],[.5,.9])


def test_mask_features_detect_predicted_text_outside_emitted_boxes():
    probabilities=np.zeros((2,32,32),np.float32);probabilities[:,4:12,4:12]=.9
    settings=dict(probability_threshold=.5,minimum_component_area=4)
    covered=mask_summary(probabilities,(32,32),[row(box=(4,4,12,12))],settings)
    missing=mask_summary(probabilities,(32,32),[],settings)
    assert covered[0]==covered[6]==1 and covered[2]==0
    assert missing[0]==missing[6]==0 and missing[2]>0 and missing[3]>0


def test_conformal_uses_finite_sample_order_statistic_and_small_sample_infinity():
    predicted=np.arange(9,dtype=float);actual=np.zeros(9)
    assert conformal_correction(predicted,actual,.2)==7
    assert math.isinf(conformal_correction(predicted,actual,.01))
    assert lower_bound_accept({'a':7.,'b':7.1,'c':None},7.)==['b']
    assert lower_bound_accept({'a':100.,'b':None},math.inf)==[]
    with pytest.raises(ValueError):conformal_correction([np.nan],[0],.1)


def test_calibration_maximizes_actual_acceptance_under_f1_budget_with_ties():
    branches={'good':dict(light=metric(2,2,2),deepsolo=metric(1,2,2)),
              'bad':dict(light=metric(0,2,2),deepsolo=metric(2,2,2)),
              'none':dict(light=metric(0,0,2),deepsolo=metric(2,2,2))}
    scores={'good':.8,'bad':.2,'none':None};ids=list(branches)
    p=calibration_threshold(scores,branches,ids,0.,route_summary)
    assert p['threshold']==.8 and p['summary']['accepted_ids']==['good']
    assert p['summary']['deepsolo_images']==2 and p['summary']['total_images']==3
    assert threshold_accept(scores,None)==[]
    tied=dict(scores,bad=.8)
    assert calibration_threshold(tied,branches,ids,0.,route_summary)['threshold'] is None


def test_calibration_refuses_a_fitting_or_test_population_mixed_into_cal():
    with pytest.raises(ValueError):calibration_threshold({'a':1},{},['a'],0.,route_summary)


def test_encoder_dependencies_are_not_hidden_in_small_head_size():
    assert feature_cost(['ocr','tokens','mask'])['additional_encoder_families']==0
    assert feature_cost(['ocr','local','global'])['additional_encoder_families']==1
    assert feature_cost(['global','blip'])['additional_encoder_families']==2


def test_head_scaler_and_weights_fit_only_supplied_rows():
    x=np.arange(60,dtype=float).reshape(20,3);y=np.linspace(-1,1,20);w=np.ones(20)
    model=fit_head(['ocr'],x,y,w,dict(kind='ridge',alpha=1.),23)
    assert np.array_equal(model['scaler'].mean_,x.mean(0))
    assert np.isfinite(predict_head(model,np.array([[999.,999.,999.]]))).all()
    assert np.array_equal(model['scaler'].mean_,x.mean(0))
    assert model['stored_scalar_count']==10


def test_native_fallback_replay_preserves_empty_symbol_and_ignore_denominators():
    from scripts.run_lightweight_unet_router import fallback_counts
    rows=[dict(text='A!',ignored=False),dict(text='?',ignored=False),dict(text='A',ignored=True)]
    assert fallback_counts(rows,[dict(text='a'),dict(text='!')])==dict(tp=1,fp=1,fn=1,detected_words=2,gt_words=2,all_outputs=3,ignored_outputs=1)


def test_cheap_online_boundary_does_not_require_or_access_foundation_context():
    rows=[row()];x=np.stack([ocr_summary(rows,(32,32)) for _ in range(10)])
    model=fit_head(['ocr'],x,np.ones(10),np.ones(10),dict(kind='ridge',alpha=1.),23)
    deployment=dict(bundle=model,projections={},feature_settings={},policies={'raw_zero':dict(threshold=0.)})
    class Forbidden:
        def __getitem__(self,key):raise AssertionError('Unselected context must not be accessed')
    out=predict_observation_gate(deployment,rows,(32,32),context_arrays=Forbidden())
    assert out['use_unet']['raw_zero'] and out['predicted_utility']==pytest.approx(2.)
    assert out['additional_encoder_cost']['additional_encoder_families']==0


def test_global_context_does_not_silently_require_word_crop_or_blip_encoders():
    values=context_summary({'patches':np.ones((4,3)), 'patch_valid':np.ones(4,bool)},4,['global'])
    assert list(values)==['global_'] and values['global_'].tolist()==[1.,1.,1.]
