"""Classical mask-to-word diagnostics kept separate from all pipeline modes.

Every function consumes only a binary native mask.  The returned groups retain
the original raw-component IDs so that GT association can remain an evaluation
step outside these localizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Final, Literal

import cv2
import numpy as np

from localize.line_word_blocks import V2WordBlockParams, prepare_word_blocks_v2


Interpolation = Literal["bilinear", "area"]


@dataclass(frozen=True)
class ClassicMaskGroups:
    groups: tuple[dict[str, object], ...]
    processed_mask: np.ndarray
    grid_shape: tuple[int, int]
    raw_component_count: int
    processed_component_count: int


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    return (mask > 0).astype(np.uint8)


def _components(mask: np.ndarray) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(_binary(mask), 8)
    rows = []
    for component_id in range(1, count):
        x, y = int(stats[component_id, 0]), int(stats[component_id, 1])
        width, height = int(stats[component_id, 2]), int(stats[component_id, 3])
        rows.append({
            "component_id": component_id,
            "bbox": (x, y, x + width, y + height),
            "foreground_area": int(stats[component_id, 4]),
        })
    return labels, tuple(rows)


def _odd_kernel_width(reference_height: float, ratio: float) -> int:
    if ratio < 0:
        raise ValueError("kernel ratio must be non-negative")
    width = max(1, int(round(reference_height * ratio)))
    return width if width % 2 else width + 1


def _close(mask: np.ndarray, kernel_width: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    return cv2.morphologyEx(_binary(mask), cv2.MORPH_CLOSE, kernel)


def _contour_groups(
    processed: np.ndarray,
    raw_labels_native: np.ndarray,
    *,
    grid_shape: tuple[int, int],
) -> tuple[dict[str, object], ...]:
    contours, _ = cv2.findContours(_binary(processed), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_rows = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        region = np.zeros(grid_shape, dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 1, thickness=cv2.FILLED)
        if grid_shape != (128, 128):
            native_region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_NEAREST)
        else:
            native_region = region
        raw_ids = tuple(sorted(int(value) for value in np.unique(raw_labels_native[native_region > 0]) if value))
        contour_rows.append({
            "raw_component_ids": raw_ids,
            "foreground_area": int(cv2.countNonZero(region)),
            "bbox_grid": (x, y, x + width, y + height),
            "grid_shape": grid_shape,
        })
    contour_rows.sort(key=lambda value: (
        -int(value["foreground_area"]), value["bbox_grid"][1], value["bbox_grid"][0],
        value["raw_component_ids"],
    ))
    return tuple({"group_id": index, **value} for index, value in enumerate(contour_rows, start=1))


def _line_params() -> V2WordBlockParams:
    return V2WordBlockParams(
        family="absolute", absolute_gap_floor=0.35, local_window=2,
        local_gap_ratio=2.0, local_fallback_absolute_floor=0.35,
        line_vertical_overlap_min=0.7, line_max_center_y_ratio=0.5,
        line_max_height_ratio=2.5, fragment_x_overlap_min=0.5,
        fragment_max_vertical_gap_ratio=0.5, fragment_small_height_ratio=0.45,
        pad=6,
    )


def global_horizontal_closing(mask: np.ndarray, kernel_height_ratio: float) -> ClassicMaskGroups:
    binary = _binary(mask)
    raw_labels, raw = _components(binary)
    reference = median(row["bbox"][3] - row["bbox"][1] for row in raw) if raw else 1.0
    processed = _close(binary, _odd_kernel_width(reference, kernel_height_ratio))
    groups = _contour_groups(processed, raw_labels, grid_shape=(128, 128))
    return ClassicMaskGroups(groups, processed, (128, 128), len(raw), len(groups))


def _line_masks(mask: np.ndarray) -> tuple[np.ndarray, tuple[dict[str, object], ...], dict[int, dict[str, object]]]:
    binary = _binary(mask)
    raw_labels, raw = _components(binary)
    prepared = prepare_word_blocks_v2(binary, _line_params())
    raw_by_id = {int(row["component_id"]): row for row in raw}
    lines = []
    for line in prepared.line_candidates:
        raw_ids = tuple(int(value) for value in line["raw_component_ids"])
        line_mask = np.isin(raw_labels, raw_ids).astype(np.uint8)
        raw_heights = [raw_by_id[value]["bbox"][3] - raw_by_id[value]["bbox"][1] for value in raw_ids]
        lines.append({
            "line_id": int(line["line_id"]), "raw_component_ids": raw_ids,
            "mask": line_mask, "median_raw_height": float(median(raw_heights)),
        })
    return raw_labels, tuple(lines), raw_by_id


def linewise_adaptive_closing(mask: np.ndarray, kernel_height_ratio: float) -> ClassicMaskGroups:
    binary = _binary(mask)
    raw_labels, lines, _ = _line_masks(binary)
    processed = np.zeros_like(binary)
    groups = []
    for line in lines:
        closed = _close(line["mask"], _odd_kernel_width(line["median_raw_height"], kernel_height_ratio))
        processed |= closed
        groups.extend(_contour_groups(closed, raw_labels, grid_shape=(128, 128)))
    groups.sort(key=lambda value: (
        -int(value["foreground_area"]), value["bbox_grid"][1], value["bbox_grid"][0],
        value["raw_component_ids"],
    ))
    groups = tuple({"group_id": index, **value} for index, value in enumerate(groups, start=1))
    return ClassicMaskGroups(groups, processed, (128, 128), int(raw_labels.max()), len(groups))


def _zero_runs(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    runs, start = [], None
    for index, value in enumerate(values.tolist() + [1]):
        if value == 0 and start is None:
            start = index
        elif value != 0 and start is not None:
            runs.append((start, index))
            start = None
    return tuple(runs)


def line_closing_projection_split(
    mask: np.ndarray,
    *,
    closing_height_ratio: float,
    split_gap_height_ratio: float,
) -> ClassicMaskGroups:
    binary = _binary(mask)
    raw_labels, lines, raw_by_id = _line_masks(binary)
    processed = np.zeros_like(binary)
    groups = []
    for line in lines:
        closed = _close(line["mask"], _odd_kernel_width(line["median_raw_height"], closing_height_ratio))
        processed |= closed
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            column_foreground = line["mask"][y:y + height, x:x + width].sum(axis=0)
            minimum_gap = max(1, int(round(split_gap_height_ratio * line["median_raw_height"])))
            cuts = [x + (left + right) // 2 for left, right in _zero_runs(column_foreground) if right - left >= minimum_gap]
            boundaries = [x, *cuts, x + width]
            for left, right in zip(boundaries, boundaries[1:]):
                raw_ids = tuple(sorted(
                    component_id for component_id in line["raw_component_ids"]
                    if (
                        left
                        <= (
                            raw_by_id[component_id]["bbox"][0]
                            + raw_by_id[component_id]["bbox"][2]
                        ) / 2
                        < right
                        and y
                        <= (
                            raw_by_id[component_id]["bbox"][1]
                            + raw_by_id[component_id]["bbox"][3]
                        ) / 2
                        < y + height
                    )
                ))
                if not raw_ids:
                    continue
                boxes = [raw_by_id[value]["bbox"] for value in raw_ids]
                bbox = (
                    min(value[0] for value in boxes), min(value[1] for value in boxes),
                    max(value[2] for value in boxes), max(value[3] for value in boxes),
                )
                groups.append({
                    "raw_component_ids": raw_ids,
                    "foreground_area": sum(int(raw_by_id[value]["foreground_area"]) for value in raw_ids),
                    "bbox_grid": bbox, "grid_shape": (128, 128),
                })
    groups.sort(key=lambda value: (
        -int(value["foreground_area"]), value["bbox_grid"][1], value["bbox_grid"][0],
        value["raw_component_ids"],
    ))
    groups = tuple({"group_id": index, **value} for index, value in enumerate(groups, start=1))
    return ClassicMaskGroups(groups, processed, (128, 128), int(raw_labels.max()), len(groups))


def padded_bbox_overlap_grouping(mask: np.ndarray, horizontal_expansion_ratio: float) -> ClassicMaskGroups:
    binary = _binary(mask)
    _, raw = _components(binary)
    parent = {int(row["component_id"]): int(row["component_id"]) for row in raw}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    expanded = {}
    for row in raw:
        x1, y1, x2, y2 = row["bbox"]
        amount = horizontal_expansion_ratio * (y2 - y1)
        expanded[int(row["component_id"])] = (x1 - amount, y1, x2 + amount, y2)
    for index, left in enumerate(raw):
        left_id = int(left["component_id"])
        for right in raw[index + 1:]:
            right_id = int(right["component_id"])
            a, b = expanded[left_id], expanded[right_id]
            if min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                join(left_id, right_id)
    partitions: dict[int, list[int]] = {}
    for component_id in parent:
        partitions.setdefault(find(component_id), []).append(component_id)
    raw_by_id = {int(row["component_id"]): row for row in raw}
    groups = []
    for raw_ids in partitions.values():
        raw_ids = tuple(sorted(raw_ids))
        boxes = [raw_by_id[value]["bbox"] for value in raw_ids]
        groups.append({
            "raw_component_ids": raw_ids,
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
    return ClassicMaskGroups(groups, binary, (128, 128), len(raw), len(groups))


def lowres_binary(mask: np.ndarray, resolution: int, interpolation: Interpolation) -> np.ndarray:
    if resolution not in (128, 96, 64, 48, 32):
        raise ValueError("unsupported diagnostic resolution")
    interpolation_code = {"bilinear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}.get(interpolation)
    if interpolation_code is None:
        raise ValueError("interpolation must be bilinear or area")
    resized = cv2.resize(_binary(mask).astype(np.float32), (resolution, resolution), interpolation=interpolation_code)
    return (resized >= 0.5).astype(np.uint8)


def lowres_connected_components(mask: np.ndarray, resolution: int, interpolation: Interpolation) -> ClassicMaskGroups:
    binary = _binary(mask)
    raw_labels, raw = _components(binary)
    processed = lowres_binary(binary, resolution, interpolation)
    groups = _contour_groups(processed, raw_labels, grid_shape=(resolution, resolution))
    return ClassicMaskGroups(groups, processed, (resolution, resolution), len(raw), len(groups))


__all__: Final = [
    "ClassicMaskGroups", "global_horizontal_closing", "linewise_adaptive_closing",
    "line_closing_projection_split", "padded_bbox_overlap_grouping",
    "lowres_binary", "lowres_connected_components",
]
