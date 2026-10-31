from copy import deepcopy

import pytest

from segmentation.uniform_unet_fusion import (
    word_counts, aggregate_counts, validate_roles, checked_mode,
)


def test_metric_keeps_duplicates_empty_care_and_ignore_denominators():
    rows = [{'raw_text': s} for s in ['SHOP!', 'shop', '', '###', 'ignored']]
    truth = [{'text': s} for s in ['Shop', '', 'missing']]
    count = word_counts(rows, truth, [False, False, False, False, True])
    assert (count['tp'], count['fp'], count['fn']) == (1, 3, 2)
    result = aggregate_counts([count])
    assert result['f1'] == pytest.approx(2/7)
    assert result['images'] == 1 and result['gt_words'] == 3


def test_all_empty_predictions_keep_false_negatives():
    count = word_counts([], [{'text': 'word'}], [])
    assert count['fn'] == 1 and aggregate_counts([count])['f1'] == 0


def test_empty_aggregate_has_explicit_zero_f1():
    assert aggregate_counts([])['f1'] == 0


def test_ignore_cardinality_is_not_silently_truncated():
    with pytest.raises(ValueError, match='cardinality'):
        word_counts([{'raw_text': 'word'}], [], [])


def test_roles_reject_duplicate_rgb_under_different_names():
    records = {'a': {'rgb_sha256': 'same'}, 'b': {'rgb_sha256': 'same'}}
    parts = dict(fit=['a'], selection=['b'], calibration=[], test=[])
    with pytest.raises(ValueError, match='RGB overlap'):
        validate_roles(records, parts)


def test_roles_reject_duplicate_sample_and_accept_disjoint_population():
    records = {s: {'rgb_sha256': s} for s in 'abc'}
    parts = dict(fit=['a'], selection=['b'], calibration=[], test=['c'])
    validate_roles(records, parts)
    bad = deepcopy(parts); bad['selection'] = ['a']
    with pytest.raises(ValueError, match='overlapping roles'):
        validate_roles(records, bad)


def test_new_experiment_cannot_enter_strict_namespace():
    config = dict(mode='approximate', claim_label='approximate reproduction')
    checked_mode('/some/configs/approximate/x.json', config)
    with pytest.raises(ValueError, match='approximate'):
        checked_mode('/some/configs/strict/x.json', config)
