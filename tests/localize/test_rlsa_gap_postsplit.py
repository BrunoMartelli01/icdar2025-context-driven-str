from __future__ import annotations

import numpy as np

from localize.rlsa_gap_postsplit import (
    RLSAGapPostSplitParams,
    _recursive_split,
    rlsa_original_gap_postsplit,
)


def _raw_box(left: int, height: int = 20) -> dict[str, object]:
    return {"bbox": (left, 10, left + 4, 10 + height), "foreground_area": 4 * height}


def test_recursive_split_uses_original_largest_gap_and_both_gates() -> None:
    raw = {1: _raw_box(0), 2: _raw_box(5), 3: _raw_box(10), 4: _raw_box(18)}
    params = RLSAGapPostSplitParams(
        max_gap_to_median_gap=2.0,
        max_gap_to_median_component_height=0.15,
    )
    groups, splits = _recursive_split((1, 2, 3, 4), raw, params)
    assert groups == ((1, 2, 3), (4,))
    assert splits == 1


def test_too_few_gaps_and_zero_median_gap_do_not_split() -> None:
    raw = {1: _raw_box(0), 2: _raw_box(20)}
    params = RLSAGapPostSplitParams(2.0, 0.1)
    assert _recursive_split((1, 2), raw, params) == (((1, 2),), 0)

    touching = {
        1: _raw_box(0), 2: _raw_box(4), 3: _raw_box(8), 4: _raw_box(16)
    }
    assert _recursive_split((1, 2, 3, 4), touching, params) == (
        ((1, 2, 3, 4),), 0
    )


def test_rlsa_initial_blob_is_split_but_output_uses_original_component_bbox() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    for left in (10, 15, 20, 28):
        mask[40:60, left:left + 4] = 1
    result = rlsa_original_gap_postsplit(
        mask,
        RLSAGapPostSplitParams(2.0, 0.15),
        frozen_run_height_ratio=0.25,
    )
    assert result.initial_rlsa_blobs == 1
    assert result.rlsa_blobs_split == 1
    assert result.total_splits == 1
    assert [group["raw_component_ids"] for group in result.groups.groups] == [
        (1, 2, 3), (4,),
    ]
    assert result.groups.groups[0]["bbox_grid"] == (10, 40, 24, 60)
