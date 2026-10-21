"""TRAIN-diagnostic classical gap grouping on native binary masks.

These localizers are intentionally separate from Algorithm 1 and every
practical-remap pipeline mode.  They do not accept GT, text, or OCR inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Final, Literal, Sequence

import cv2
import numpy as np

from localize.classic_word_regions import ClassicMaskGroups, _components, _contour_groups, _line_params
from localize.line_word_blocks import prepare_word_blocks_v2


GapMethod = Literal["otsu", "two_means"]


@dataclass(frozen=True)
class GapSegmentationResult:
    groups: ClassicMaskGroups
    line_statistics: dict[str, Any]


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    return (mask > 0).astype(np.uint8)


def _raw_lines(
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, object], ...], dict[int, dict[str, object]]]:
    binary = _binary(mask)
    labels, raw = _components(binary)
    raw_by_id = {int(value["component_id"]): value for value in raw}
    prepared = prepare_word_blocks_v2(binary, _line_params())
    lines = []
    for value in prepared.line_candidates:
        raw_ids = tuple(sorted(
            (int(component_id) for component_id in value["raw_component_ids"]),
            key=lambda component_id: (
                raw_by_id[component_id]["bbox"][0],
                raw_by_id[component_id]["bbox"][1],
                component_id,
            ),
        ))
        heights = [
            raw_by_id[component_id]["bbox"][3] - raw_by_id[component_id]["bbox"][1]
            for component_id in raw_ids
        ]
        lines.append({
            "line_id": int(value["line_id"]),
            "raw_component_ids": raw_ids,
            "median_component_height": float(median(heights)),
        })
    return labels, tuple(lines), raw_by_id


def _rlsa_row(row: np.ndarray, maximum_background_run: int) -> None:
    foreground = np.flatnonzero(row)
    for left, right in zip(foreground, foreground[1:]):
        run_length = int(right - left - 1)
        if 0 < run_length <= maximum_background_run:
            row[left + 1:right] = 1


def linewise_rlsa(mask: np.ndarray, run_height_ratio: float) -> ClassicMaskGroups:
    """Fill short bounded horizontal background runs independently per text line."""
    if run_height_ratio < 0:
        raise ValueError("run_height_ratio must be non-negative")
    binary = _binary(mask)
    raw_labels, lines, _ = _raw_lines(binary)
    processed = np.zeros_like(binary)
    for line in lines:
        line_mask = np.isin(raw_labels, line["raw_component_ids"]).astype(np.uint8)
        maximum_run = max(1, int(round(run_height_ratio * line["median_component_height"])))
        for row in line_mask:
            _rlsa_row(row, maximum_run)
        processed |= line_mask
    groups = _contour_groups(processed, raw_labels, grid_shape=(128, 128))
    return ClassicMaskGroups(
        groups=groups,
        processed_mask=processed,
        grid_shape=(128, 128),
        raw_component_count=int(raw_labels.max()),
        processed_component_count=len(groups),
    )


def _normalized_gaps(
    raw_ids: Sequence[int], raw_by_id: dict[int, dict[str, object]]
) -> tuple[float, ...]:
    values = []
    for left_id, right_id in zip(raw_ids, raw_ids[1:]):
        left, right = raw_by_id[left_id]["bbox"], raw_by_id[right_id]["bbox"]
        gap = max(0, right[0] - left[2])
        reference_height = ((left[3] - left[1]) + (right[3] - right[1])) / 2
        values.append(gap / reference_height)
    return tuple(values)


def _otsu_1d(values: Sequence[float]) -> float | None:
    unique = sorted(set(float(value) for value in values))
    if len(values) < 2 or len(unique) < 2:
        return None
    candidates = [(left + right) / 2 for left, right in zip(unique, unique[1:])]
    best_threshold, best_score = None, -1.0
    array = np.asarray(values, dtype=np.float64)
    for threshold in candidates:
        low, high = array[array <= threshold], array[array > threshold]
        if not len(low) or not len(high):
            continue
        low_weight, high_weight = len(low) / len(array), len(high) / len(array)
        score = low_weight * high_weight * float(low.mean() - high.mean()) ** 2
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold


def _two_means_1d(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < 2 or min(values) == max(values):
        return None
    array = np.asarray(values, dtype=np.float64)
    means = np.asarray([array.min(), array.max()], dtype=np.float64)
    for _ in range(50):
        distances = np.abs(array[:, None] - means[None, :])
        labels = np.argmin(distances, axis=1)
        if len(set(labels.tolist())) < 2:
            return None
        updated = np.asarray([array[labels == index].mean() for index in (0, 1)])
        if np.allclose(updated, means, rtol=0.0, atol=1e-12):
            means = updated
            break
        means = updated
    return float(min(means)), float(max(means))


def _segments(raw_ids: Sequence[int], boundary_indices: set[int]) -> tuple[tuple[int, ...], ...]:
    start, values = 0, []
    for gap_index in range(len(raw_ids) - 1):
        if gap_index in boundary_indices:
            values.append(tuple(raw_ids[start:gap_index + 1]))
            start = gap_index + 1
    values.append(tuple(raw_ids[start:]))
    return tuple(values)


def line_gap_segmentation(
    mask: np.ndarray,
    *,
    method: GapMethod,
    fallback_normalized_gap: float = 0.35,
    minimum_cluster_separation: float = 0.0,
) -> GapSegmentationResult:
    """Split left-to-right raw components using one line-local gap classifier."""
    if method not in ("otsu", "two_means"):
        raise ValueError("method must be otsu or two_means")
    if fallback_normalized_gap < 0 or minimum_cluster_separation < 0:
        raise ValueError("gap thresholds must be non-negative")
    binary = _binary(mask)
    raw_labels, lines, raw_by_id = _raw_lines(binary)
    groups = []
    lines_two_or_more_gaps = lines_two_groups = fallback_lines = 0
    lines_without_gaps = boundary_count = merge_count = 0
    resulting_thresholds = []
    for line in lines:
        raw_ids = line["raw_component_ids"]
        gaps = _normalized_gaps(raw_ids, raw_by_id)
        if not gaps:
            lines_without_gaps += 1
            boundary_indices: set[int] = set()
        else:
            if len(gaps) >= 2:
                lines_two_or_more_gaps += 1
            threshold = None
            if method == "otsu":
                threshold = _otsu_1d(gaps)
            else:
                means = _two_means_1d(gaps)
                if means is not None and means[1] - means[0] >= minimum_cluster_separation:
                    threshold = (means[0] + means[1]) / 2
            if threshold is None:
                threshold = fallback_normalized_gap
                fallback_lines += 1
            else:
                lines_two_groups += 1
            resulting_thresholds.append(float(threshold))
            boundary_indices = {index for index, gap in enumerate(gaps) if gap > threshold}
            boundary_count += len(boundary_indices)
            merge_count += len(gaps) - len(boundary_indices)
        for raw_group in _segments(raw_ids, boundary_indices):
            boxes = [raw_by_id[value]["bbox"] for value in raw_group]
            groups.append({
                "raw_component_ids": raw_group,
                "foreground_area": sum(int(raw_by_id[value]["foreground_area"]) for value in raw_group),
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
    classic = ClassicMaskGroups(
        groups=groups, processed_mask=binary, grid_shape=(128, 128),
        raw_component_count=int(raw_labels.max()), processed_component_count=len(groups),
    )
    return GapSegmentationResult(
        groups=classic,
        line_statistics={
            "lines_total": len(lines),
            "lines_with_at_least_two_gaps": lines_two_or_more_gaps,
            "lines_with_real_two_group_separation": lines_two_groups,
            "lines_using_fallback": fallback_lines,
            "lines_without_gaps": lines_without_gaps,
            "median_resulting_threshold": (
                float(median(resulting_thresholds)) if resulting_thresholds else None
            ),
            "boundary_count": boundary_count,
            "merge_count": merge_count,
            "_threshold_values": tuple(resulting_thresholds),
        },
    )


__all__: Final = [
    "GapSegmentationResult", "linewise_rlsa", "line_gap_segmentation",
]
