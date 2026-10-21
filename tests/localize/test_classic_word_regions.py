from __future__ import annotations

import numpy as np

from localize.classic_word_regions import (
    ClassicMaskGroups,
    global_horizontal_closing,
    line_closing_projection_split,
    linewise_adaptive_closing,
    lowres_binary,
    lowres_connected_components,
    padded_bbox_overlap_grouping,
)
from scripts.diagnose_paperlike_localization_classic_train import _to_original_groups


def _three_glyph_mask() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:52, 10:15] = 1
    mask[40:52, 18:23] = 1
    mask[40:52, 70:75] = 1
    return mask


def test_global_and_linewise_closing_merge_nearby_glyphs_only() -> None:
    mask = _three_glyph_mask()
    global_result = global_horizontal_closing(mask, 0.5)
    line_result = linewise_adaptive_closing(mask, 0.5)
    assert [group["raw_component_ids"] for group in global_result.groups] == [(1, 2), (3,)]
    assert [group["raw_component_ids"] for group in line_result.groups] == [(1, 2), (3,)]


def test_projection_split_recovers_large_whitespace_boundary() -> None:
    mask = _three_glyph_mask()
    result = line_closing_projection_split(
        mask, closing_height_ratio=6.0, split_gap_height_ratio=1.0
    )
    assert {group["raw_component_ids"] for group in result.groups} == {(1, 2), (3,)}


def test_padded_bbox_overlap_uses_unexpanded_union_box() -> None:
    mask = _three_glyph_mask()
    result = padded_bbox_overlap_grouping(mask, horizontal_expansion_ratio=0.2)
    by_ids = {group["raw_component_ids"]: group for group in result.groups}
    assert by_ids[(1, 2)]["bbox_grid"] == (10, 40, 23, 52)
    assert by_ids[(3,)]["bbox_grid"] == (70, 40, 75, 52)


def test_lowres_modes_are_binary_and_record_component_counts() -> None:
    mask = _three_glyph_mask()
    for interpolation in ("bilinear", "area"):
        resized = lowres_binary(mask, 64, interpolation)
        assert resized.shape == (64, 64)
        assert set(np.unique(resized)).issubset({0, 1})
        result = lowres_connected_components(mask, 64, interpolation)
        assert result.raw_component_count == 3
        assert result.processed_mask.shape == (64, 64)
        assert result.processed_component_count == len(result.groups)


def test_lowres_128_is_an_exact_binary_control_for_both_interpolations() -> None:
    mask = _three_glyph_mask()
    assert np.array_equal(lowres_binary(mask, 128, "bilinear"), mask)
    assert np.array_equal(lowres_binary(mask, 128, "area"), mask)


def test_resampling_split_keeps_duplicate_component_associations_as_distinct_proposals() -> None:
    result = ClassicMaskGroups(
        groups=(
            {"group_id": 1, "raw_component_ids": (7,), "foreground_area": 4,
             "bbox_grid": (1, 1, 3, 3), "grid_shape": (32, 32)},
            {"group_id": 2, "raw_component_ids": (7,), "foreground_area": 3,
             "bbox_grid": (4, 1, 6, 3), "grid_shape": (32, 32)},
        ),
        processed_mask=np.zeros((32, 32), dtype=np.uint8),
        grid_shape=(32, 32),
        raw_component_count=1,
        processed_component_count=2,
    )
    groups, retained = _to_original_groups(result, 128, 128)
    assert groups[0]["raw_component_ids"] == (7, -1)
    assert groups[1]["raw_component_ids"] == (7, -2)
    assert len(retained) == 2
