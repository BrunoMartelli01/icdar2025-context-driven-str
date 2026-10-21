from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_cached_totaltext_padding import proposals_for_padding
from scripts.evaluate_totaltext_compact_v2_selected_ocr import distribution, mask_summary, measure_mask
from scripts.audit_totaltext_compact_v2_selected_ocr import independent_metrics


PLAN = {"padding_native_pixels": [0, 1, 2, 4, 6, 8],
        "min_unpadded_bbox_area_native": 150, "max_proposals_per_image": 10}


def measured(mask):
    variants = proposals_for_padding(mask, (384, 640), PLAN)
    return variants, measure_mask(mask, variants)


def test_pixel_area_counts_holes_and_rejected_fragments_separately_from_bbox():
    mask = np.zeros((128, 128), np.uint8)
    mask[10:20, 15:35] = 1
    mask[12:18, 20:30] = 0
    mask[80:82, 90:92] = 1
    variants, measurements = measured(mask)
    assert measurements["foreground_pixels"] == 144
    assert measurements["foreground_fraction"] == 144 / (128 * 128)
    assert sorted(c["foreground_area_native"] for c in measurements["components_8_connected"]) == [4, 140]
    for v in variants.values():
        assert len(v["proposals"]) == 1
        p = v["proposals"][0]
        assert p["foreground_area_native"] == 140
        assert p["bbox_area_native"] == 200
        assert p["bbox_native"] == [15, 10, 35, 20]
    shard = {"mask_measurements": measurements, "variants": variants}
    summary = mask_summary([shard], PLAN["padding_native_pixels"])
    assert summary["all_component_foreground_area_native"]["count"] == 2
    assert summary["retained_by_padding"]["0"]["component_foreground_area_native"]["median"] == 140
    assert summary["retained_by_padding"]["0"]["bbox_width_native"]["median"] == 20
    assert summary["retained_by_padding"]["8"]["padded_crop_width_native"]["median"] == 36
    assert summary["retained_by_padding"]["0"]["foreground_to_bbox_area_ratio"]["median"] == .7


def test_eight_connected_diagonal_pixels_partition_foreground_without_inflation():
    mask = np.zeros((128, 128), np.uint8)
    for i in range(20):
        mask[i, i] = 1
    variants, measurements = measured(mask)
    assert len(measurements["components_8_connected"]) == 1
    assert measurements["foreground_pixels"] == 20
    assert variants["0"]["proposals"][0]["foreground_area_native"] == 20
    assert variants["0"]["proposals"][0]["bbox_area_native"] == 400


def test_empty_mask_and_empty_distributions_are_explicit():
    variants, measurements = measured(np.zeros((128, 128), np.uint8))
    assert measurements["foreground_pixels"] == 0
    assert measurements["components_8_connected"] == []
    summary = mask_summary([{"variants": variants, "mask_measurements": measurements}], [0])
    assert summary["foreground_fraction_per_image"]["mean"] == 0
    assert summary["retained_by_padding"]["0"]["proposals"] == 0
    assert distribution([])["median"] is None


def test_pooled_component_distribution_does_not_average_image_medians():
    a, b = np.zeros((128, 128), np.uint8), np.zeros((128, 128), np.uint8)
    a[:10, :20] = 1
    b[20:30, 20:40] = 1
    b[60:70, 60:80] = 1
    shards = []
    for mask in [a, b]:
        variants, measurements = measured(mask)
        shards.append({"variants": variants, "mask_measurements": measurements})
    summary = mask_summary(shards, [0])
    assert summary["foreground_pixels_per_image"]["mean"] == 300
    assert summary["all_component_foreground_area_native"]["count"] == 3
    assert summary["retained_by_padding"]["0"]["component_foreground_area_native"]["count"] == 3


@pytest.mark.parametrize("mask", [np.ones((32, 32), np.uint8), np.full((128, 128), 2, np.uint8)])
def test_wrong_native_shape_or_nonbinary_input_fails(mask):
    with pytest.raises(ValueError, match="binary native128"):
        measure_mask(mask, {})


def test_independent_audit_counts_duplicate_predictions_once_and_retains_punctuation():
    predictions = [{"bbox": [0, 0, 20, 10], "crop_array_sha256": s} for s in ["a", "b"]]
    words = [{"bbox": [0, 0, 20, 10], "text": "Book"}, {"bbox": [30, 0, 50, 10], "text": "Book!"}]
    ocr = {"a": {"raw_text": "BOOK"}, "b": {"raw_text": "book"}}
    values = independent_metrics(predictions, words, "bbox", ocr)
    assert values["gt_instances"] == 2
    assert values["retained_proposals"] == 2
    assert values["exact_casefold_spatial_one_to_one"] == 1
    assert values["exact_case_sensitive_spatial_one_to_one"] == 0


def test_independent_audit_empty_outputs_still_include_all_gt():
    words = [{"bbox": [0, 0, 10, 10], "text": "Book"}]
    values = independent_metrics([], words, "bbox", {})
    assert values["gt_instances"] == 1
    assert values["exact_casefold_spatial_one_to_one"] == 0
    assert values["retained_proposals"] == 0


def test_independent_audit_iou_half_is_inclusive():
    values = independent_metrics([{"bbox": [0, 0, 10, 10], "crop_array_sha256": "a"}],
        [{"bbox": [0, 0, 20, 10], "text": "Book"}], "bbox", {"a": {"raw_text": "book"}})
    assert values["iou_ge_0_50_one_to_one"] == 1
    assert values["exact_casefold_spatial_one_to_one"] == 1
