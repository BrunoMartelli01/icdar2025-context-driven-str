import pytest

from scripts.prepare_ic13_unet_oof import check_fold


def test_conditional_decoder_fold_rejects_fit_validation_and_rgb_leaks():
    records = {s: {'rgb_sha256': s} for s in ('a', 'b', 'v', 't')}
    parent = dict(partitions=dict(fit=['a', 'b'], selection=['v'], calibration=[]),
                  records=records, tt_ids=['t'])
    manifest = dict(source_fit_ids=['a', 'b'], validation_ids=['v'])
    fold = dict(fit_ids=['a'], heldout_ids=['b'])
    assert check_fold(parent, manifest, fold)['heldout'] == ['b']
    with pytest.raises(ValueError, match='preserve'):
        check_fold(parent, manifest, dict(fit_ids=['a', 'b'], heldout_ids=['b']))
    records['b']['rgb_sha256'] = 't'
    with pytest.raises(ValueError, match='duplicates'):
        check_fold(parent, manifest, fold)
