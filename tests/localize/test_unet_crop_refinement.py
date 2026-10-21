from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from localize.unet_crop_refinement import (
    ContourConfig, CropRefinementConfig, component_contours,
    mask_diagnostics, refine_contour_crops,
)


def settings(**changes):
    return replace(CropRefinementConfig(ContourConfig(0, 0, 10), ContourConfig(0, 0, 10),
                                       0.5, 4, 10, "approximate"), **changes)


def coordinate_rgb(height=50, width=80):
    y, x = np.indices((height, width))
    return np.stack((x, y, np.zeros_like(x)), axis=2).astype(np.uint8)


def test_filter_and_sort_use_component_pixels_including_holes_not_box_area():
    mask = np.zeros((40, 40), np.uint8)
    mask[1, 1:21] = 1
    mask[1:21, 1] = 1  # 39 pixels, bbox area 400
    mask[25:32, 25:32] = 1  # 49 pixels, bbox area 49
    result = component_contours(mask, ContourConfig(40, 0, 10))
    assert [r["component_area"] for r in result["retained"]] == [49]
    assert result["rejected"][0]["component_area"] == 39
    result = component_contours(mask, ContourConfig(0, 0, 1))
    assert result["retained"][0]["bbox"] == [25, 25, 32, 32]
    ring = np.ones((5, 5), np.uint8)
    ring[1:4, 1:4] = 0
    assert component_contours(ring, ContourConfig(0, 0, 1))["retained"][0]["component_area"] == 16


def test_area_boundary_ties_clipping_and_input_immutability():
    mask = np.zeros((10, 15), np.uint8)
    mask[:2, :2] = 255
    mask[8:, 13:] = 255
    before = mask.copy()
    result = component_contours(mask, ContourConfig(4, 3, 10))
    assert [r["padded_bbox"] for r in result["retained"]] == [[0, 0, 5, 5], [10, 5, 15, 10]]
    np.testing.assert_array_equal(mask, before)
    assert result == component_contours(mask, ContourConfig(4, 3, 10))


def test_crops_reuse_rgb_callable_once_and_remap_exclusive_boxes():
    rgb = coordinate_rgb()
    original = rgb.copy()
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[20:26, 30:40] = 1
    calls = []

    def segmenter(crop):
        calls.append(crop.copy())
        local = np.zeros(crop.shape[:2], np.uint8)
        local[2:8, 3:15] = 1
        return {"mask": local}

    result = refine_contour_crops(rgb, mask, segmenter, settings())
    assert len(calls) == result["report"]["inference_calls_second_pass"] == 1
    assert result["report"]["crops"][0]["crop_bbox"] == [21, 13, 49, 33]
    np.testing.assert_array_equal(calls[0], original[13:33, 21:49])
    assert result["report"]["refined_proposals"][0]["bbox"] == [24, 15, 36, 21]
    assert result["refined_mask"].sum() == 72
    assert not result["refined_mask"][:13].any()
    np.testing.assert_array_equal(rgb, original)


def test_zoom_can_join_fragments_with_context_and_exact_duplicate_is_audited():
    rgb = coordinate_rgb(30, 40)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[10:15, 10:16] = mask[10:15, 18:24] = 1

    def segmenter(crop):
        x, y = crop[:, :, 0], crop[:, :, 1]
        return {"mask": ((x >= 10) & (x < 24) & (y >= 10) & (y < 15)).astype(np.uint8)}

    result = refine_contour_crops(rgb, mask, segmenter, settings(context_margin_pixels=12))
    assert result["report"]["inference_calls_second_pass"] == 2
    assert [r["padded_bbox"] for r in result["report"]["refined_proposals"]] == [[10, 10, 24, 15]]
    assert result["report"]["rejected_proposals"][0]["reason"] == "exact_duplicate_bbox"
    for row in result["report"]["crops"]:
        assert row["diagnostics"]["before_components"] == 2
        assert row["diagnostics"]["after_components"] == 1
        assert row["diagnostics"]["seed_associated_after_components"] == 1
        assert row["diagnostics"]["seed_support_retained_fraction"] == 1
        assert row["diagnostics"]["added_foreground_pixels"] == 10


def test_second_pass_can_split_and_clamp_padding_without_recursive_inference():
    rgb = coordinate_rgb(20, 30)
    mask = np.ones((20, 30), np.uint8)
    calls = []

    def segmenter(crop):
        calls.append(1)
        after = np.ones(crop.shape[:2], np.uint8)
        after[:, 12:18] = 0
        return {"mask": after}

    config = settings(second_pass=ContourConfig(0, 4, 10), max_final_blocks=1)
    result = refine_contour_crops(rgb, mask, segmenter, config)
    assert calls == [1]
    row = result["report"]["crops"][0]
    assert row["diagnostics"]["component_count_change"] == 1
    assert row["diagnostics"]["seed_associated_after_components"] == 2
    assert all(row["diagnostics"]["foreground_at_crop_border"].values())
    assert len(result["report"]["refined_proposals"]) == 1
    assert result["report"]["refined_proposals"][0]["padded_bbox"] == [0, 0, 16, 20]
    assert result["report"]["rejected_proposals"][0]["reason"] == "max_final_blocks"


def test_empty_output_is_preserved_and_empty_initial_mask_skips_inference():
    rgb = coordinate_rgb()
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[1:4, 1:4] = 1
    result = refine_contour_crops(rgb, mask, lambda c: {"mask": np.zeros(c.shape[:2], np.uint8)}, settings())
    assert not result["report"]["refined_proposals"]
    assert result["report"]["crops"][0]["diagnostics"]["empty_second_pass"]
    mask[:] = 0

    def forbidden(crop):
        raise AssertionError("No crop should be inferred")

    result = refine_contour_crops(rgb, mask, forbidden, settings())
    assert result["report"]["inference_calls_second_pass"] == 0
    assert not result["refined_mask"].any()


def test_errors_propagate_without_baseline_or_dummy_substitution():
    rgb = coordinate_rgb()
    mask = np.ones(rgb.shape[:2], np.uint8)

    def failed(crop):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        refine_contour_crops(rgb, mask, failed, settings())
    for bad in (np.zeros((1, 1), np.uint8), np.full(mask.shape, np.nan), np.full(mask.shape, 0.2)):
        with pytest.raises(ValueError):
            refine_contour_crops(rgb, mask, lambda c: {"mask": bad}, settings())


@pytest.mark.parametrize("changes", [{"reproduction_mode": "strict"},
                                    {"context_margin_fraction": float("nan")},
                                    {"context_margin_fraction": -0.1},
                                    {"context_margin_pixels": True}, {"max_final_blocks": 0}])
def test_invalid_or_strict_configs_fail(changes):
    with pytest.raises(ValueError):
        settings(**changes)


def test_diagnostic_empty_masks_are_defined_and_overlap_does_not_imply_words():
    empty = np.zeros((3, 4), np.uint8)
    assert mask_diagnostics(empty, empty)["foreground_iou"] == 1
    assert "not word correctness" in mask_diagnostics(empty, empty)["interpretation"]
