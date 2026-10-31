"""Scientific boundaries for the cheap, paired-expert supervised gate."""
import copy
import inspect
import math

import numpy as np
import pytest

from recognition.budget_distilled_router import (observation_features,paired_target,
    fit_small_head,predict_small_head,rank_accept,predict_observation)


SETTINGS=dict(sequence_cuts=[.5,.7,.9],map_cuts=[.5,.65],grid_side=2)


def row():
    return dict(candidate_id='word0',bbox=[4,4,12,12],bbox_native=[4,4,12,12],
        raw_text='Ab7',sequence_confidence_native=.9,token_probabilities=[.95,.98,.99],
        detector_score=.96,foreground_area=60)


def test_removed_word_exposes_uncovered_predicted_support():
    p=np.zeros((2,16,16),np.float32);p[:,4:12,4:12]=.99
    full,names,groups=observation_features([row()],[row()],[16,16],p,SETTINGS)
    missing,names2,_=observation_features([],[row()],[16,16],p,SETTINGS)
    assert names==names2 and len(names)==len(set(names))==len(groups)
    assert full[names.index('text_uncovered_soft')]==0
    assert missing[names.index('text_uncovered_soft')]==1
    assert missing[names.index('text_0.65_missing_components')]>0


def test_observation_ignores_extra_scientific_outcomes():
    p=np.zeros((2,16,16),np.float32);r=row()
    expected=observation_features([r],[r],[16,16],p,SETTINGS)
    r=copy.deepcopy(r);r.update(ground_truth='forbidden',deepsolo_tp=999,correctness=1)
    actual=observation_features([r],[r],[16,16],p,SETTINGS)
    assert np.array_equal(expected[0],actual[0])
    arguments=set(inspect.signature(predict_observation).parameters)
    assert not arguments&{'annotations','ground_truth','heavy','deepsolo','validity'}


def test_targets_penalize_errors_and_reward_added_correct_words():
    heavy=dict(tp=4,detected_words=5)
    assert paired_target(dict(tp=4,detected_words=6),heavy,.8)==pytest.approx(-.8)
    assert paired_target(dict(tp=5,detected_words=6),heavy,.8)==pytest.approx(1.2)


@pytest.mark.parametrize('recipe',[
    dict(kind='ridge',alpha=10.),dict(kind='extra_trees',trees=8,depth=3,minimum_leaf=2),
    dict(kind='boosted',trees=8,depth=2,minimum_leaf=2,learning_rate=.05)])
def test_portable_weighted_fit_and_json_roundtrip(recipe):
    import json
    rng=np.random.default_rng(17);x=rng.normal(size=(60,5));y=x[:,0]**2-x[:,1];weights=np.linspace(.2,2,60)
    model=fit_small_head(x,y,weights,recipe,23)
    before=predict_small_head(model,x)
    after=predict_small_head(json.loads(json.dumps(model)),x)
    assert np.array_equal(before,after) and np.isfinite(after).all()


def test_budget_has_deterministic_ties_and_preserves_ineligible_images():
    scores={'b':1.,'a':1.,'c':None,'d':-.4}
    assert rank_accept(scores,list(scores),.5)==['a','b']
    assert len(rank_accept(scores,list(scores),1))==3
    with pytest.raises(ValueError):rank_accept(scores,['a'],.5)


def test_invalid_observations_fail_instead_of_silently_substituting():
    p=np.zeros((2,16,16),np.float32)
    with pytest.raises(ValueError):observation_features([row()],[],[16,16],p,SETTINGS)
    p[0,0,0]=math.nan
    with pytest.raises(ValueError):observation_features([],[],[16,16],p,SETTINGS)


@pytest.mark.parametrize('override',[dict(grid_side=0),dict(map_cuts=[1.])])
def test_invalid_scientific_feature_settings_fail(override):
    with pytest.raises(ValueError):
        observation_features([],[],[16,16],np.zeros((2,16,16),np.float32),dict(SETTINGS,**override))
