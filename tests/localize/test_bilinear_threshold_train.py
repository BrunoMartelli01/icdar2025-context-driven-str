import numpy as np
import pytest

from diagnostics.bilinear_mask_backprojection_train import bilinear_binary
from diagnostics.bilinear_threshold_train import interpolate_binary, threshold_binary


def test_control_matches_previous_and_higher_threshold_only_removes_pixels():
    mask = np.random.default_rng(5).integers(0, 2, (128, 128), dtype=np.uint8)
    original = mask.copy()
    values = interpolate_binary(mask, (319, 521))
    previous = bilinear_binary(mask, (319, 521), threshold=.5, align_corners=False, antialias=False)
    assert np.array_equal(previous, threshold_binary(values, .5))
    for threshold in (.55, .6, .65, .7):
        current = threshold_binary(values, threshold)
        assert np.all(current <= previous)
        previous = current
    assert np.array_equal(mask, original)


def test_threshold_inclusive_and_float_preserved():
    values = np.array([[.5, .55, .6, .65, .7]], dtype=np.float64)
    np.testing.assert_array_equal(threshold_binary(values, .6), [[0, 0, 1, 1, 1]])


@pytest.mark.parametrize("threshold", [.49, 1.01, np.nan])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        threshold_binary(np.zeros((2, 2)), threshold)


@pytest.mark.parametrize("values", [np.array([[np.nan]]), np.array([[-.1]]), np.array([[1.1]]), np.array([.5])])
def test_invalid_interpolated_values(values):
    with pytest.raises(ValueError):
        threshold_binary(values, .6)


@pytest.mark.parametrize("fill", [0, 1])
def test_constant_masks_and_border(fill):
    values = interpolate_binary(np.full((128, 128), fill, dtype=np.uint8), (333, 257))
    assert np.all(threshold_binary(values, .7) == fill)


def test_reject_probabilities_and_invalid_shape():
    with pytest.raises(ValueError):
        interpolate_binary(np.full((128, 128), .6, dtype=np.float32), (256, 256))
    with pytest.raises(ValueError):
        interpolate_binary(np.zeros((128, 128), dtype=np.uint8), (0, 256))


def test_lower_thresholds_only_add_foreground_and_preserve_exact_ties():
    mask = np.random.default_rng(7).integers(0, 2, (128, 128), dtype=np.uint8)
    values = interpolate_binary(mask, (319, 521))
    previous = threshold_binary(values, .5)
    for t in (.45, .4, .35, .3):
        current = threshold_binary(values, t, minimum=.3)
        assert np.all(current >= previous)
        previous = current
    np.testing.assert_array_equal(
        threshold_binary(np.array([[.29, .3, .31]]), .3, minimum=.3), [[0, 1, 1]])


def test_lower_threshold_must_be_explicit_and_in_declared_range():
    values = np.ones((2, 2))
    with pytest.raises(ValueError):
        threshold_binary(values, .3)
    with pytest.raises(ValueError):
        threshold_binary(values, .29, minimum=.3)
    with pytest.raises(ValueError):
        threshold_binary(values, 0, minimum=0)
