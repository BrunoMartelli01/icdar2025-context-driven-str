from __future__ import annotations

import numpy as np
import pytest

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam
from localize.native_algorithm1 import localize_native_algorithm1


def params(*, min_area: int = 1, pad: int = 0) -> BlockParam:
    return BlockParam(min_area, pad, 8, APPROXIMATE_TIE_BREAK, 10)


def test_native_area_padding_mapping_and_original_crop() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:12, 20:23] = 1
    image = np.arange(256 * 512 * 3, dtype=np.uint32).reshape(256, 512, 3)

    result = localize_native_algorithm1(mask, image, params(min_area=6, pad=2))

    assert len(result.retained) == 1
    block = result.retained[0]
    assert block.component_area_native == 6
    assert block.bbox_native == (20, 10, 23, 12)
    assert block.padded_bbox_native == (18, 8, 25, 14)
    assert block.bbox_original == (80, 20, 92, 24)
    assert block.padded_bbox_original == (72, 16, 100, 28)
    np.testing.assert_array_equal(block.crop, image[16:28, 72:100])


def test_native_path_filters_by_foreground_not_bbox_area_and_sorts() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[2:7, 2] = 1
    mask[6, 2:7] = 1  # foreground 9, bbox area 25: rejected by Amin=10
    mask[20:23, 30:34] = 1  # foreground 12
    image = np.zeros((300, 400, 3), dtype=np.uint8)

    result = localize_native_algorithm1(mask, image, params(min_area=10))

    assert [block.component_area_native for block in result.retained] == [12]
    assert result.audit["rejected"][0]["component_area"] == 9
    assert result.audit["rejected"][0]["reason"] == "below_min_area"


def test_native_path_rejects_non_native_mask() -> None:
    with pytest.raises(ValueError, match="128x128"):
        localize_native_algorithm1(
            np.zeros((64, 64), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            params(),
        )
