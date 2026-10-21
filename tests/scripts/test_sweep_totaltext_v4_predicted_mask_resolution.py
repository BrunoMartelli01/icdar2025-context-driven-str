import numpy as np
import pytest

from scripts.sweep_totaltext_v4_gt_mask_resolution import _proposals
from scripts.sweep_totaltext_v4_predicted_mask_resolution import resize_predicted_binary


def test_native128_binary_prediction_is_byte_identical_and_copied():
    native = np.zeros((128, 128), dtype=np.uint8)
    native[4:12, 6:20] = 1
    result = resize_predicted_binary(native, 128)
    assert np.array_equal(result, native)
    assert result.ctypes.data != native.ctypes.data


def test_bilinear_binary_upscale_keeps_small_isolated_component_without_filters():
    native = np.zeros((128, 128), dtype=np.uint8)
    native[20, 20] = 1
    enlarged = resize_predicted_binary(native, 256)
    assert enlarged.shape == (256, 256)
    assert enlarged.dtype == np.uint8
    assert np.isin(enlarged, (0, 1)).all()
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    audit, proposals = _proposals(enlarged, 256, image, filter_mode="none")
    assert audit["raw_external_contours"] == len(proposals) == 1
    assert proposals[0]["bbox_area_mask"] < 150


def test_predicted_resolution_rejects_nonbinary_or_unscheduled_input():
    native = np.zeros((128, 128), dtype=np.uint8)
    native[0, 0] = 2
    with pytest.raises(ValueError, match="canonical"):
        resize_predicted_binary(native, 256)
    native[0, 0] = 1
    with pytest.raises(ValueError, match="scheduled"):
        resize_predicted_binary(native, 192)
