import inspect

import numpy as np
import pytest

from localize.contiguous_span_candidates import (
    enumerate_contiguous_span_candidates,
    remap_and_deduplicate_candidates,
)
from localize.line_word_blocks import RawComponent
from localize.multihypothesis_word_proposals import (
    _robust_line_orders,
    deduplicate_candidates,
    deduplicate_candidates_with_provenance,
    enumerate_native_hypotheses,
    materialize_bbox_variants,
)


def _scene(boxes):
    mask = np.zeros((128, 128), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    return mask


def test_same_line_max12_reproduces_old_contiguous_span_boxes() -> None:
    mask = _scene([(10, 20, 13, 40), (18, 20, 21, 40), (26, 20, 29, 40)])
    old = enumerate_contiguous_span_candidates(mask)
    old_boxes, _ = remap_and_deduplicate_candidates(old, 256, 384)
    new = enumerate_native_hypotheses(mask, max_components=12)
    materialized = materialize_bbox_variants(new, threshold=0.5, width=256, height=384)
    actual = {
        tuple(row["bbox_original"])
        for row in materialized
        if row["source_kind"] == "same_line" and row["bbox_variant"] == "standard_pad6"
    }
    assert actual == {tuple(row["padded_bbox_original"]) for row in old_boxes}


def test_span_limit_twenty_and_width_pruning() -> None:
    mask = _scene([(2 + 3 * index, 10, 4 + 3 * index, 30) for index in range(20)])
    result = enumerate_native_hypotheses(mask, max_components=20)
    assert max(len(row["raw_component_ids"]) for row in result.spans) == 20
    with pytest.raises(ValueError):
        enumerate_native_hypotheses(mask, max_components=21)


def test_cross_line_robust_order_uses_frozen_component_compatibility() -> None:
    raw = (
        RawComponent(1, 30, (10, 20, 13, 30), (11.0, 25.0)),
        RawComponent(2, 30, (15, 20, 18, 30), (16.0, 25.0)),
    )
    lines = (
        {"line_id": 1, "raw_component_ids": (1,)},
        {"line_id": 2, "raw_component_ids": (2,)},
    )
    robust, edges = _robust_line_orders(raw, lines)
    assert robust == ({"robust_line_id": 1, "source_line_ids": (1, 2),
                       "raw_component_ids": (1, 2)},)
    assert len(edges) == 1


def test_bbox_variants_use_original_union_and_adaptive_padding() -> None:
    result = enumerate_native_hypotheses(_scene([(10, 20, 20, 40)]), max_components=20)
    rows = materialize_bbox_variants(result, threshold=0.5, width=256, height=512)
    by_variant = {row["bbox_variant"]: row for row in rows if row["source_kind"] == "same_line"}
    assert by_variant["tight"]["bbox_original"] == (20, 80, 40, 160)
    assert by_variant["standard_pad6"]["bbox_original"] == (14, 74, 46, 166)
    assert by_variant["adaptive"]["adaptive_pad_original"] == 12
    assert by_variant["adaptive"]["bbox_original"] == (8, 68, 52, 172)


def test_iou_deduplication_is_stable_and_keeps_first_candidate() -> None:
    rows = [
        {"name": "first", "bbox_original": (0, 0, 100, 100)},
        {"name": "near", "bbox_original": (1, 1, 100, 100)},
        {"name": "different", "bbox_original": (120, 0, 150, 30)},
    ]
    assert [row["name"] for row in deduplicate_candidates(rows)] == ["first", "different"]


def test_provenance_dedup_keeps_output_boxes_and_all_threshold_sources() -> None:
    def row(name, threshold, box):
        return {
            "name": name, "bbox_original": box, "bbox_native": box,
            "threshold": threshold, "bbox_variant": "standard_pad6",
            "source_kind": "same_line", "source_line_ids": (1,),
            "raw_component_ids": (1,),
        }
    rows = [
        row("first", 0.5, (0, 0, 100, 100)),
        row("same-at-lower-threshold", 0.4, (1, 1, 100, 100)),
        row("different", 0.5, (120, 0, 150, 30)),
    ]
    plain = deduplicate_candidates(rows)
    preserved = deduplicate_candidates_with_provenance(rows)
    assert [value["bbox_original"] for value in preserved] == [value["bbox_original"] for value in plain]
    assert {value["threshold"] for value in preserved[0]["dedup_sources"]} == {0.4, 0.5}


def test_generator_is_deterministic_binary_and_has_no_gt_api() -> None:
    mask = _scene([(10, 20, 13, 40), (18, 20, 21, 40)])
    assert enumerate_native_hypotheses(mask) == enumerate_native_hypotheses(mask)
    assert set(inspect.signature(enumerate_native_hypotheses).parameters) == {"mask", "max_components"}
    with pytest.raises(ValueError):
        enumerate_native_hypotheses(mask.astype(np.float32))
