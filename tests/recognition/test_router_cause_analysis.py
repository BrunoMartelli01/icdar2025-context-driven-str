"""Independent numerical guards for matched cause-test reporting."""
import numpy as np
import pytest

from scripts.analyze_router_cause_tests import paired_interval, mean_interval, direction_interval, ordered_likelihood


SETTINGS = dict(seed=17, bootstrap_replicates=100, confidence_level=.95)


def test_identical_routes_have_exact_zero_paired_difference():
    counts = np.array([[[1, 2, 3], [2, 3, 4], [3, 4, 5]]], dtype=float)
    x = paired_interval(counts, counts, ['a', 'b', 'c'], SETTINGS)
    assert x['difference_pp'] == x['lower95_pp'] == x['upper95_pp'] == 0
    assert x['support'] == 'inconclusive'


def test_uniformly_better_routes_have_exact_positive_effect_for_each_seed():
    a = np.array([[[1, 1, 1]]*4]*3, dtype=float); b = a.copy(); b[..., 0] = 0
    value = paired_interval(a, b, ['same', 'same', 'other', 'third'], SETTINGS)
    assert value['difference_pp'] == value['lower95_pp'] == value['upper95_pp'] == 100
    assert value['matched_seeds'] == 3 and value['support'] == 'positive'


def test_unpaired_seed_populations_fail_explicitly():
    with pytest.raises(ValueError):
        paired_interval(np.zeros((1, 3, 3)), np.zeros((2, 3, 3)), ['a', 'b', 'c'], SETTINGS)


def test_direction_interval_states_uncertainty_without_fabricating_samples():
    assert direction_interval([], 1e-6, .95) is None
    value = direction_interval([1.]*100, 1e-6, .95)
    assert value['positive_direction_fraction'] == 1 and value['lower95'] > .95
    assert mean_interval([], SETTINGS) is None


def test_likelihood_order_is_restored_after_json_key_sorting():
    value = ordered_likelihood({'a': 2., 'b': 3.}, ['b', 'a'])
    np.testing.assert_array_equal(value, [3., 2.])
    assert value.dtype == np.float32
    with pytest.raises(ValueError):
        ordered_likelihood({'a': 2.}, ['a', 'a'])
