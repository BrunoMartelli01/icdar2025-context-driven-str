import numpy as np
import pytest
from recognition.icdar_budget_router import map_summary,lexical_counts,calibration_cut,accept,rank,budget_policy
from recognition.lightweight_unet_router import mask_summary


def test_two_channel_square_replays_existing_essential_mask_features():
    rng=np.random.default_rng(3);p=rng.random((2,32,32),dtype=np.float32)
    rows=[dict(bbox=[2,3,18,14])];settings=dict(probability_threshold=.5,minimum_component_area=4)
    assert np.array_equal(map_summary(p,[40,20],rows,'square_letterbox',settings),mask_summary(p,[40,20],rows,settings))


def test_single_rectangular_output_uses_only_six_real_descriptors():
    p=np.zeros((1,8,16),np.float32);p[0,2:6,4:12]=1
    value=map_summary(p,[32,16],[dict(bbox=[8,4,24,12])],'aspect_letterbox',dict(probability_threshold=.5,minimum_component_area=1))
    assert value.shape==(6,) and value[0]==1 and value[2]==0 and value[3]==0


def test_symbol_only_and_empty_outputs_remain_in_denominator():
    pp=[dict(text='WORD!',ignored=False),dict(text='!!!',ignored=False),dict(text='',ignored=False),dict(text='bad',ignored=True)]
    got=lexical_counts(pp,[dict(text='word'),dict(text='?')])
    assert (got['tp'],got['fp'],got['fn'],got['gt_words'],got['ignored_outputs'])==(1,2,1,2,1)


def test_calibration_ties_use_rgb_identity_instead_of_train_test_name_prefix():
    scores={'train:a':2.,'train:b':2.,'train:c':1.,'train:d':0.};keys={s:k for s,k in zip(scores,['d','b','a','c'])}
    cut=calibration_cut(scores,list(scores),.25,keys)
    assert accept(scores,cut,keys)==['train:b']
    assert rank(scores,list(scores),.5,keys)==['train:b','train:a']
    test={'test:x':2.,'test:y':2.};test_keys={'test:x':'a','test:y':'z'}
    assert accept(test,cut,test_keys)==['test:x']


def test_ineligible_calibration_shortfall_fails_explicitly():
    with pytest.raises(ValueError):calibration_cut({'a':None,'b':None},['a','b'],.5)


def test_dataset_budget_changes_selection_and_calibration_population():
    c=dict(primary_acceptance={'icdar13':.7,'icdar2015':.5},acceptance_targets={'icdar13':[.65,.7,.75],'icdar2015':[.45,.5,.55]})
    scores={str(i):float(i) for i in range(10)}
    for dataset,n in [('icdar13',7),('icdar2015',5)]:
        primary,targets=budget_policy(c,dataset)
        assert primary in targets
        assert len(rank(scores,list(scores),primary))==n
        assert len(accept(scores,calibration_cut(scores,list(scores),primary)))==n
    with pytest.raises(KeyError):budget_policy(c,'unknown')
    c['primary_acceptance']['icdar13']=.8
    with pytest.raises(ValueError):budget_policy(c,'icdar13')


@pytest.mark.parametrize('value',[np.nan,1.1,-.1])
def test_invalid_predicted_maps_rejected(value):
    p=np.zeros((1,8,16),np.float32);p[0,0,0]=value
    with pytest.raises(ValueError):map_summary(p,[32,16],[],'aspect_letterbox',dict(probability_threshold=.5,minimum_component_area=1))
