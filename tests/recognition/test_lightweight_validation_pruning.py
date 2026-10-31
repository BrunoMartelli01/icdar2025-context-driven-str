from copy import deepcopy

from scripts.prune_lightweight_unet_router import choose_on_validation


def test_uncertain_or_identical_extra_features_are_not_retained():
    branch=dict(light=dict(tp=1,detected_words=1,gt_words=1),deepsolo=dict(tp=1,detected_words=1,gt_words=1))
    branches={str(i):deepcopy(branch) for i in range(20)}
    base=dict(name='ocr',selection_mean_f1=1.,stored_scalar_count=55,cost=dict(additional_encoder_families=0),
              selection_points=[dict(accepted_ids=['0'],deepsolo_images=19,total_images=20,deepsolo_percent=95.)])
    extra=dict(base,name='extra')
    result=choose_on_validation([base,extra],{'ocr':0,'extra':1},branches,draws=30,seed=1,family_error=.05)
    assert result['selected_name']=='ocr'
    assert not result['comparisons']['extra']['supported_extra_signal']
    assert not result['test_used'] and not result['calibration_used_for_feature_selection']


def test_large_consistent_validation_benefit_retains_extra_signal():
    branches={str(i):dict(light=dict(tp=1,detected_words=1,gt_words=1),deepsolo=dict(tp=0,detected_words=1,gt_words=1)) for i in range(30)}
    point=lambda keep:dict(accepted_ids=keep,deepsolo_images=30-len(keep),total_images=30,deepsolo_percent=100*(30-len(keep))/30)
    base=dict(name='ocr',selection_mean_f1=0.,stored_scalar_count=55,cost=dict(additional_encoder_families=0),selection_points=[point([])])
    extra=dict(name='extra',selection_mean_f1=1.,stored_scalar_count=100,cost=dict(additional_encoder_families=1),selection_points=[point(list(branches))])
    result=choose_on_validation([base,extra],{'ocr':0,'extra':1},branches,draws=30,seed=1,family_error=.05)
    assert result['selected_name']=='extra' and result['comparisons']['extra']['supported_extra_signal']
