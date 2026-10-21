import inspect

import numpy as np
import pytest

from localize.contiguous_span_candidates import (
    MAX_COMPONENTS_PER_SPAN, enumerate_contiguous_span_candidates,
    remap_and_deduplicate_candidates,
)


def scene(boxes):
    mask = np.zeros((128, 128), np.uint8)
    for box in boxes:
        mask[box[1]:box[3], box[0]:box[2]] = 1
    return mask


def test_enumerates_every_contiguous_span_and_raw_union_bbox():
    result = enumerate_contiguous_span_candidates(scene([
        (10, 20, 13, 40), (18, 20, 21, 40), (26, 20, 29, 40),
    ]))
    assert len(result.lines) == 1 and len(result.candidates_native) == 6
    assert {row["raw_component_ids"] for row in result.candidates_native} == {
        (1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3),
    }
    whole = next(row for row in result.candidates_native if row["raw_component_ids"] == (1, 2, 3))
    assert whole["bbox_native"] == (10, 20, 29, 40)


def test_lines_do_not_mix_and_span_length_is_capped_at_twelve():
    boxes = [(2 + 6 * index, 10, 4 + 6 * index, 20) for index in range(13)]
    boxes += [(10, 60, 13, 70), (18, 60, 21, 70)]
    result = enumerate_contiguous_span_candidates(scene(boxes))
    assert len(result.lines) == 2
    assert max(len(row["raw_component_ids"]) for row in result.candidates_native) == MAX_COMPONENTS_PER_SPAN
    line_id_sets = [set(line["raw_component_ids"]) for line in result.lines]
    assert all(
        sum(set(row["raw_component_ids"]) <= line_ids for line_ids in line_id_sets) == 1
        for row in result.candidates_native
    )


def test_width_pruning_and_final_bbox_remap_padding():
    result = enumerate_contiguous_span_candidates(scene([(0, 20, 2, 21), (30, 20, 32, 21)]))
    assert all(len(row["raw_component_ids"]) == 1 for row in result.candidates_native)
    candidates, counts = remap_and_deduplicate_candidates(result, 256, 256)
    assert candidates[0]["bbox_original"] == (0, 40, 4, 42)
    assert candidates[0]["padded_bbox_original"] == (0, 34, 10, 48)
    assert counts["after_exact_final_bbox_deduplication"] == 2


def test_empty_deterministic_and_no_gt_api():
    empty = enumerate_contiguous_span_candidates(scene([]))
    assert not empty.lines and not empty.candidates_native
    mask = scene([(10, 20, 13, 40), (18, 20, 21, 40)])
    first = enumerate_contiguous_span_candidates(mask)
    assert first == enumerate_contiguous_span_candidates(mask)
    assert set(inspect.signature(enumerate_contiguous_span_candidates).parameters) == {"mask"}


def test_binary_contract():
    with pytest.raises(ValueError):
        enumerate_contiguous_span_candidates(np.zeros((128, 128), np.float32))
    with pytest.raises(ValueError):
        enumerate_contiguous_span_candidates(np.full((128, 128), 255, np.uint8))
