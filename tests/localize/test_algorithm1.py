from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from localize.mask_to_blocks import (
    ALGORITHM1_NMAX,
    APPROXIMATE_TIE_BREAK,
    BlockParam,
    localize_mask,
    mask_to_blocks,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "localize"
ROOT = Path(__file__).parents[2]


def approximate_params(*, min_area: int = 1, pad: int = 0) -> BlockParam:
    return BlockParam(
        min_area=min_area,
        pad=pad,
        connectivity=8,
        tie_break=APPROXIMATE_TIE_BREAK,
    )


def test_irregular_component_uses_pixel_area_not_bounding_box_area() -> None:
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[2:7, 2] = 1
    mask[6, 2:7] = 1  # sparse L: true area 9, bbox area 25
    mask[15:18, 20:24] = 1  # compact rectangle: true/bbox area 12
    image = np.zeros((30, 40, 3), dtype=np.uint8)

    result = localize_mask(mask, image, approximate_params(min_area=10))

    assert [block.component_area for block in result.retained] == [12]
    assert result.retained[0].bbox == (20, 15, 24, 18)
    assert [(item.component_area, item.bbox, item.reason) for item in result.rejected] == [
        (9, (2, 2, 7, 7), "below_min_area")
    ]


def test_minimum_area_is_inclusive_and_rejects_smaller_components() -> None:
    mask = np.zeros((7, 10), dtype=np.uint8)
    mask[1, 1:4] = 1
    mask[5, 7:9] = 1
    image = np.zeros((7, 10, 3), dtype=np.uint8)

    result = localize_mask(mask, image, approximate_params(min_area=3))

    assert [block.component_area for block in result.retained] == [3]
    assert [(item.component_area, item.reason) for item in result.rejected] == [
        (2, "below_min_area")
    ]


def test_empty_mask_returns_no_retained_or_rejected_components() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    result = localize_mask(np.zeros((5, 7), dtype=np.uint8), image, approximate_params())
    assert result.retained == ()
    assert result.rejected == ()
    assert mask_to_blocks(
        np.zeros((5, 7), dtype=np.uint8), image, approximate_params()
    ) == []


def test_padding_uses_explicit_exclusive_crop_coordinates() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[4:6, 5:8] = 1
    image = np.arange(10 * 12 * 3, dtype=np.uint16).reshape(10, 12, 3)

    block = mask_to_blocks(mask, image, approximate_params(pad=2))[0]

    assert block.bbox == (5, 4, 8, 6)
    assert block.padded_bbox == (3, 2, 10, 8)
    np.testing.assert_array_equal(block.crop, image[2:8, 3:10])


def test_padding_clamps_components_touching_both_image_borders() -> None:
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[0:2, 0:3] = 1
    mask[6:8, 7:9] = 1
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)

    blocks = mask_to_blocks(mask, image, approximate_params(pad=4))

    assert [block.bbox for block in blocks] == [(0, 0, 3, 2), (7, 6, 9, 8)]
    assert [block.padded_bbox for block in blocks] == [(0, 0, 7, 6), (3, 2, 9, 8)]
    np.testing.assert_array_equal(blocks[0].crop, image[0:6, 0:7])
    np.testing.assert_array_equal(blocks[1].crop, image[2:8, 3:9])


def test_equal_area_ties_are_top_to_bottom_then_left_to_right() -> None:
    mask = np.zeros((14, 18), dtype=np.uint8)
    mask[8:10, 2:4] = 1
    mask[2:4, 11:13] = 1
    mask[2:4, 5:7] = 1
    image = np.zeros((14, 18, 3), dtype=np.uint8)

    blocks = mask_to_blocks(mask, image, approximate_params())

    assert [block.component_area for block in blocks] == [4, 4, 4]
    assert [block.bbox[:2] for block in blocks] == [(5, 2), (11, 2), (2, 8)]
    assert all(block.tie_break == APPROXIMATE_TIE_BREAK for block in blocks)


def test_components_are_ordered_by_descending_true_area() -> None:
    mask = np.zeros((15, 24), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # area 4
    mask[6:9, 8:11] = 1  # area 9
    mask[11:13, 18:21] = 1  # area 6
    image = np.zeros((15, 24, 3), dtype=np.uint8)

    blocks = mask_to_blocks(mask, image, approximate_params())

    assert [block.component_area for block in blocks] == [9, 6, 4]
    assert [block.rank for block in blocks] == [1, 2, 3]


def test_top_ten_cap_records_overflow_component() -> None:
    mask = np.zeros((3, 45), dtype=np.uint8)
    for index in range(11):
        mask[1, 1 + index * 4] = 1
    image = np.zeros((3, 45, 3), dtype=np.uint8)

    result = localize_mask(mask, image, approximate_params())

    assert len(result.retained) == ALGORITHM1_NMAX
    assert [block.bbox[0] for block in result.retained] == [
        1 + index * 4 for index in range(10)
    ]
    assert len(result.rejected) == 1
    assert result.rejected[0].bbox == (41, 1, 42, 2)
    assert result.rejected[0].padded_bbox == (41, 1, 42, 2)
    assert result.rejected[0].reason == "top_10_limit"


def test_connectivity_and_tie_policy_are_explicit_validated_inputs() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2, 2] = 1
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    blocks = mask_to_blocks(mask, image, approximate_params())
    assert len(blocks) == 1
    assert (blocks[0].component_area, blocks[0].connectivity) == (2, 8)
    four_connected = mask_to_blocks(
        mask,
        image,
        BlockParam(1, 0, connectivity=4, tie_break=APPROXIMATE_TIE_BREAK),
    )
    assert [block.component_area for block in four_connected] == [1, 1]
    with pytest.raises(TypeError, match="connectivity"):
        BlockParam(1, 0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="connectivity"):
        BlockParam(1, 0, connectivity=6, tie_break=APPROXIMATE_TIE_BREAK)
    with pytest.raises(ValueError, match="tie policy"):
        BlockParam(1, 0, connectivity=8, tie_break="source_order")
    with pytest.raises(ValueError, match="nmax=10"):
        BlockParam(1, 0, connectivity=8, tie_break=APPROXIMATE_TIE_BREAK, nmax=9)


def test_input_shapes_must_match() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="identical"):
        mask_to_blocks(np.zeros((4, 7), dtype=np.uint8), image, approximate_params())


def test_strict_localization_configuration_remains_fail_closed() -> None:
    strict = OmegaConf.load(ROOT / "configs" / "strict" / "pipeline.yaml")
    assert strict.segmenter.binarize_thresh is None
    assert strict.localize.min_area is None
    assert strict.localize.pad is None
    assert strict.localize.connectivity is None
    assert strict.localize.equal_area_tie_break is None
    assert {"A-005", "L-001", "L-003", "L-005"}.issubset(
        set(strict.blocking_ambiguity_ids)
    )


def test_golden_fixture_exposes_algorithm1_audit_and_final_crops() -> None:
    expected = json.loads((FIXTURE_DIR / "algorithm1_golden.json").read_text())
    mask = np.asarray(Image.open(FIXTURE_DIR / expected["mask_file"]).convert("L"))
    image = np.asarray(Image.open(FIXTURE_DIR / expected["image_file"]).convert("RGB"))
    parameters = expected["localization"]["parameters"]

    result = localize_mask(
        mask,
        image,
        BlockParam(
            min_area=parameters["min_area"],
            pad=parameters["pad"],
            connectivity=parameters["connectivity"],
            tie_break=parameters["tie_break"],
            nmax=parameters["nmax"],
        ),
    )

    assert result.as_dict() == expected["localization"]
    assert len(result.retained) == len(expected["retained_crop_files"])
    for block, crop_file in zip(result.retained, expected["retained_crop_files"]):
        expected_crop = np.asarray(Image.open(FIXTURE_DIR / crop_file).convert("RGB"))
        np.testing.assert_array_equal(block.crop, expected_crop)
