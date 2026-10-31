"""Validation must stay outside fitting and bind cuts to identical weights."""
import hashlib

import pytest

from recognition.router_validation import expert_holdout_ids,independent_partitions,validation_cuts,require_same_model
from recognition.unet_token_routing import apply_threshold


def test_qualified_expert_holdout_matches_ordered_identity():
    ids=['a','b','c','d'];seed=17
    chosen=set(sorted(ids,key=lambda s:(hashlib.sha256(f'{seed}:{s}'.encode()).digest(),s))[:2])
    expected=[s for s in ids if s in chosen]
    digest=hashlib.sha256(''.join(s+'\n' for s in expected).encode()).hexdigest()
    assert expert_holdout_ids(ids,seed,2,digest)==expected
    with pytest.raises(ValueError):expert_holdout_ids(ids,seed,2,'changed')


def test_fit_validation_and_test_stay_separate():
    samples=[dict(sample_id=s,rgb_sha256=s) for s in ['a','b','c']]
    test=[dict(sample_id='test',rgb_sha256='different')]
    assert independent_partitions(samples,['c'],test)==dict(fit=['a','b'],validation=['c'],test=['test'])
    with pytest.raises(ValueError):independent_partitions(samples,['missing'],test)
    samples[0]['rgb_sha256']='c'
    with pytest.raises(ValueError,match='RGB leakage'):independent_partitions(samples,['c'],test)


def test_cross_split_rgb_copy_cannot_enter_validation_or_fit():
    samples=[dict(sample_id='a',rgb_sha256='A'),dict(sample_id='b',rgb_sha256='B')]
    with pytest.raises(ValueError,match='RGB leakage'):
        independent_partitions(samples,['b'],[dict(sample_id='test',rgb_sha256='A')])


def test_actual_validation_scores_choose_cut_and_test_does_not_recalibrate():
    policy=validation_cuts({'v1':3.,'v2':2.,'v3':1.,'v4':None},[.5],'frozen-checkpoint')[0]
    assert policy['selected_on']=='VALIDATION' and policy['threshold']==2.
    assert policy['validation_target_fraction']==.5 and 'train_target_fraction' not in policy
    assert apply_threshold({'t1':2.1,'t2':1.9,'t3':None},policy)==['t1']
    require_same_model(policy,'frozen-checkpoint')
    with pytest.raises(ValueError,match='refitted'):require_same_model(policy,'refitted-checkpoint')


def test_validation_reject_all_and_eligible_limit_are_explicit():
    values={'a':1.,'b':None}
    policies=validation_cuts(values,[0.,1.],'weights')
    assert apply_threshold(values,policies[0])==[]
    assert apply_threshold(values,policies[1])==['a']
