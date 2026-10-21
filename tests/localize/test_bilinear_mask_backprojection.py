import numpy as np
import pytest

from diagnostics.bilinear_mask_backprojection_train import bilinear_binary


def resize(mask, shape, **kwargs):
    return bilinear_binary(mask, shape, **{
        "threshold": .5, "align_corners": False, "antialias": False, **kwargs})


@pytest.mark.parametrize("fill", [0, 1])
def test_constant_and_border(fill):
    mask = np.full((128, 128), fill, dtype=np.uint8)
    result = resize(mask, (257, 511))
    assert result.dtype == np.uint8
    assert result.shape == (257, 511)
    assert np.all(result == fill)


def test_identity_and_input_unchanged():
    mask = np.random.default_rng(0).integers(0, 2, (128, 128), dtype=np.uint8)
    saved = mask.copy()
    np.testing.assert_array_equal(resize(mask, mask.shape), saved)
    resize(mask, (300, 500))
    np.testing.assert_array_equal(mask, saved)


def test_half_pixel_bilinear_corner_not_nearest_or_uint8_truncation():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:64, :64] = 1
    result = resize(mask, (512, 512))
    # At (255,255), input position is (63.375,63.375): .625**2 < .5.
    assert result[255, 255] == 0  # nearest would retain this corner
    # At (254,254), .875**2 >= .5: don't truncate float before threshold.
    assert result[254, 254] == 1


def test_exact_half_threshold_included():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:, :64] = 1
    result = resize(mask, (129, 129))
    assert np.all(result[:, 64] == 1)  # x=63.5 gives exactly .5
    assert np.all(result[:, 65] == 0)


@pytest.mark.parametrize("kwargs", [{"threshold": .4}, {"align_corners": True}, {"antialias": True}])
def test_reject_protocol_drift(kwargs):
    with pytest.raises(ValueError):
        resize(np.zeros((128, 128), dtype=np.uint8), (256, 256), **kwargs)


@pytest.mark.parametrize("mask", [np.zeros((64, 64), dtype=np.uint8),
    np.full((128, 128), .5, dtype=np.float32), np.full((128, 128), 255, dtype=np.uint8)])
def test_reject_nonbinary_native_input(mask):
    with pytest.raises(ValueError):
        resize(mask, (256, 256))


@pytest.mark.parametrize("shape", [(0, 128), (128,), (128, 1.5)])
def test_reject_invalid_dimensions(shape):
    with pytest.raises(ValueError):
        resize(np.zeros((128, 128), dtype=np.uint8), shape)
