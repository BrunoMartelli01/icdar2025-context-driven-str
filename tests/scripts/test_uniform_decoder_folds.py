import copy

import pytest

from scripts.uniform_decoder_folds import check_fold


def plan():
    return dict(partitions=dict(fit=['a','b'],decoder_heldout=['c','d'],selection=['v'],test=['t']),
        train_records=[dict(sample_id=s,rgb_sha256=s) for s in ['a','b','c','d','v']],
        test_records=[dict(sample_id='t',rgb_sha256='t')],
        fold_contract=dict(original_fit_ids=['a','b','c','d'],fixed_epochs=25),
        config=dict(training=dict(epochs=25)))


def test_fold_preserves_complete_native_fit_and_frozen_epoch_budget():
    check_fold(plan())


def test_fold_rejects_same_rgb_under_different_sample_names():
    p=plan();p['train_records'][2]['rgb_sha256']='a'
    with pytest.raises(ValueError,match='RGB'):check_fold(p)


def test_fold_rejects_missing_native_fit_image_and_heldout_checkpoint_tuning():
    p=plan();p['partitions']['fit']=['a']
    with pytest.raises(ValueError,match='partition'):check_fold(p)
    p=plan();p['config']['training']['epochs']=30
    with pytest.raises(ValueError,match='epoch'):check_fold(p)
