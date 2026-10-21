"""Deterministic original-gap post-split for the frozen TRAIN RLSA diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Final, Sequence

import numpy as np

from localize.classic_word_regions import ClassicMaskGroups, _components
from localize.gap_word_regions import linewise_rlsa


@dataclass(frozen=True)
class RLSAGapPostSplitParams:
    max_gap_to_median_gap: float
    max_gap_to_median_component_height: float

    def __post_init__(self) -> None:
        if self.max_gap_to_median_gap <= 0:
            raise ValueError("max_gap_to_median_gap must be positive")
        if self.max_gap_to_median_component_height <= 0:
            raise ValueError("max_gap_to_median_component_height must be positive")


@dataclass(frozen=True)
class RLSAGapPostSplitResult:
    groups: ClassicMaskGroups
    initial_rlsa_blobs: int
    initial_multicomponent_blobs: int
    rlsa_blobs_split: int
    total_splits: int


def _recursive_split(
    raw_component_ids: Sequence[int],
    raw_by_id: dict[int, dict[str, object]],
    params: RLSAGapPostSplitParams,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    ordered = tuple(sorted(raw_component_ids, key=lambda component_id: (
        raw_by_id[component_id]["bbox"][0],
        raw_by_id[component_id]["bbox"][1],
        component_id,
    )))
    # One or two components provide fewer than two internal gaps and therefore
    # cannot establish that one gap is anomalous relative to the others.
    if len(ordered) < 3:
        return (ordered,), 0
    gaps = []
    for left_id, right_id in zip(ordered, ordered[1:]):
        left, right = raw_by_id[left_id]["bbox"], raw_by_id[right_id]["bbox"]
        gaps.append(float(max(0, right[0] - left[2])))
    median_gap = float(median(gaps))
    if median_gap <= 0:
        return (ordered,), 0
    component_heights = [
        float(raw_by_id[component_id]["bbox"][3] - raw_by_id[component_id]["bbox"][1])
        for component_id in ordered
    ]
    median_height = float(median(component_heights))
    largest_gap = max(gaps)
    largest_index = gaps.index(largest_gap)  # deterministic leftmost tie
    if (
        largest_gap / median_gap < params.max_gap_to_median_gap
        or largest_gap / median_height < params.max_gap_to_median_component_height
    ):
        return (ordered,), 0
    left_groups, left_splits = _recursive_split(
        ordered[:largest_index + 1], raw_by_id, params
    )
    right_groups, right_splits = _recursive_split(
        ordered[largest_index + 1:], raw_by_id, params
    )
    return (*left_groups, *right_groups), 1 + left_splits + right_splits


def rlsa_original_gap_postsplit(
    mask: np.ndarray,
    params: RLSAGapPostSplitParams,
    *,
    frozen_run_height_ratio: float = 0.25,
) -> RLSAGapPostSplitResult:
    """Use RLSA only for initial blobs, then split from original raw geometry."""
    binary = (mask > 0).astype(np.uint8)
    _, raw = _components(binary)
    raw_by_id = {int(value["component_id"]): value for value in raw}
    rlsa = linewise_rlsa(binary, frozen_run_height_ratio)
    partitions = []
    blobs_split = total_splits = multicomponent = 0
    for blob in rlsa.groups:
        raw_ids = tuple(int(value) for value in blob["raw_component_ids"] if value > 0)
        if not raw_ids:
            continue
        multicomponent += int(len(raw_ids) >= 2)
        subgroups, split_count = _recursive_split(raw_ids, raw_by_id, params)
        partitions.extend(subgroups)
        blobs_split += int(split_count > 0)
        total_splits += split_count
    groups = []
    for raw_ids in partitions:
        boxes = [raw_by_id[value]["bbox"] for value in raw_ids]
        groups.append({
            "raw_component_ids": tuple(raw_ids),
            "foreground_area": sum(int(raw_by_id[value]["foreground_area"]) for value in raw_ids),
            "bbox_grid": (
                min(value[0] for value in boxes), min(value[1] for value in boxes),
                max(value[2] for value in boxes), max(value[3] for value in boxes),
            ),
            "grid_shape": (128, 128),
        })
    groups.sort(key=lambda value: (
        -int(value["foreground_area"]), value["bbox_grid"][1], value["bbox_grid"][0],
        value["raw_component_ids"],
    ))
    groups = tuple({"group_id": index, **value} for index, value in enumerate(groups, start=1))
    return RLSAGapPostSplitResult(
        groups=ClassicMaskGroups(
            groups=groups,
            processed_mask=rlsa.processed_mask,
            grid_shape=(128, 128),
            raw_component_count=rlsa.raw_component_count,
            processed_component_count=len(groups),
        ),
        initial_rlsa_blobs=len(rlsa.groups),
        initial_multicomponent_blobs=multicomponent,
        rlsa_blobs_split=blobs_split,
        total_splits=total_splits,
    )


__all__: Final = [
    "RLSAGapPostSplitParams", "RLSAGapPostSplitResult",
    "rlsa_original_gap_postsplit",
]
