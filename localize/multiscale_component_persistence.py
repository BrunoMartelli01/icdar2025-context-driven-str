"""TRAIN-diagnostic multiscale persistence grouping of native-mask components.

This is a separate PRACTICAL_APPROXIMATION. Horizontal dilation establishes
connectivity only; every reported/final box and area comes from original raw CCs.
No GT, text, OCR, learned model, or score-selected parameter enters inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Final, Sequence

import cv2
import numpy as np


MODE: Final = "MULTISCALE_COMPONENT_PERSISTENCE_V1"
SCALES: Final = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
MAX_SCALE: Final = SCALES[-1]
LINE_VERTICAL_OVERLAP_MIN: Final = 0.7
LINE_MAX_CENTER_Y_RATIO: Final = 0.5
LINE_MAX_HEIGHT_RATIO: Final = 2.5
PAD_ORIGINAL: Final = 6
NMAX: Final = 10


@dataclass(frozen=True)
class RawComponent:
    component_id: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


@dataclass(frozen=True)
class PersistenceResult:
    raw_components: tuple[RawComponent, ...]
    lines: tuple[dict[str, Any], ...]
    nodes: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    selected: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": MODE,
            "classification": "PRACTICAL_APPROXIMATION",
            "scales": list(SCALES),
            "morphology_role": "horizontal dilation for connectivity only",
            "selection": "weighted interval DP; weight=(persistence/0.60)*raw_component_count",
            "raw_components": [vars(value) for value in self.raw_components],
            "lines": self.lines,
            "nodes": self.nodes,
            "candidates": self.candidates,
            "selected": self.selected,
        }


def _extract(mask: np.ndarray) -> tuple[np.ndarray, tuple[RawComponent, ...]]:
    if mask.dtype != np.uint8 or mask.shape != (128, 128) or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be uint8 binary 0/1 with shape 128x128")
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    rows = []
    for component_id in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        rows.append(RawComponent(
            component_id, area, (x, y, x + width, y + height),
            (float(centroids[component_id, 0]), float(centroids[component_id, 1])),
        ))
    return labels, tuple(rows)


def _height(component: RawComponent) -> int:
    return component.bbox[3] - component.bbox[1]


def _line_compatible(left: RawComponent, right: RawComponent) -> bool:
    lh, rh = _height(left), _height(right)
    overlap = max(0, min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]))
    reference = (lh + rh) / 2.0
    return (
        overlap / min(lh, rh) >= LINE_VERTICAL_OVERLAP_MIN
        and abs(left.centroid[1] - right.centroid[1]) / reference <= LINE_MAX_CENTER_Y_RATIO
        and max(lh, rh) / min(lh, rh) <= LINE_MAX_HEIGHT_RATIO
    )


def _lines(components: tuple[RawComponent, ...]) -> tuple[tuple[RawComponent, ...], ...]:
    parent = {value.component_id: value.component_id for value in components}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for index, left in enumerate(components):
        for right in components[index + 1:]:
            if _line_compatible(left, right):
                a, b = find(left.component_id), find(right.component_id)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    grouped: dict[int, list[RawComponent]] = {}
    for component in components:
        grouped.setdefault(find(component.component_id), []).append(component)
    lines = [tuple(sorted(values, key=lambda value: (value.bbox[0], value.bbox[1], value.component_id)))
             for values in grouped.values()]
    return tuple(sorted(lines, key=lambda values: (
        min(value.bbox[1] for value in values), min(value.bbox[0] for value in values),
        values[0].component_id,
    )))


def _radius(scale: float, reference_height: float) -> int:
    """Nearest integer, half upward; scale zero is the exact raw mask."""
    return int(math.floor(scale * reference_height + 0.5))


def _partition_at_radius(
    line: Sequence[RawComponent], raw_labels: np.ndarray, radius: int,
) -> tuple[tuple[int, int], ...]:
    ids = tuple(value.component_id for value in line)
    line_mask = np.isin(raw_labels, ids).astype(np.uint8)
    expanded = (line_mask if radius == 0 else cv2.dilate(
        line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 1))
    ))
    _, connected = cv2.connectedComponents(expanded, 8)
    memberships = []
    for component_id in ids:
        labels = np.unique(connected[raw_labels == component_id])
        labels = labels[labels > 0]
        if len(labels) != 1:
            raise RuntimeError("raw component must map to exactly one morphology component")
        memberships.append(int(labels[0]))
    spans, start = [], 0
    for index in range(1, len(ids)):
        if memberships[index] != memberships[index - 1]:
            spans.append((start, index))
            start = index
    spans.append((start, len(ids)))
    return tuple(spans)


def _union_box(values: Sequence[RawComponent]) -> tuple[int, int, int, int]:
    return (min(v.bbox[0] for v in values), min(v.bbox[1] for v in values),
            max(v.bbox[2] for v in values), max(v.bbox[3] for v in values))


def _gap(left: RawComponent, right: RawComponent) -> int:
    return max(0, right.bbox[0] - left.bbox[2])


def _nodes_for_line(
    line_id: int, line: tuple[RawComponent, ...], raw_labels: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_height = float(median(_height(value) for value in line))
    partitions = []
    previous = None
    for scale in SCALES:
        radius = _radius(scale, reference_height)
        spans = _partition_at_radius(line, raw_labels, radius)
        # Horizontal dilation is monotone. Every old span must be contained in a new span.
        if previous is not None and any(
            not any(now[0] <= old[0] and old[1] <= now[1] for now in spans)
            for old in previous
        ):
            raise RuntimeError("scale-space partition is not a merge tree")
        partitions.append({"scale": scale, "radius_pixels": radius, "spans": spans})
        previous = spans
    unique = sorted({span for row in partitions for span in row["spans"]}, key=lambda v: (v[0], v[1]))
    nodes = []
    for start, end in unique:
        exact_scales = [row["scale"] for row in partitions if (start, end) in row["spans"]]
        birth = min(exact_scales)
        later = [row["scale"] for row in partitions if row["scale"] > birth and (start, end) not in row["spans"]]
        death = min(later) if later else MAX_SCALE
        persistence = max(0.0, death - birth)
        members = line[start:end]
        internal = [_gap(a, b) for a, b in zip(members, members[1:])]
        nodes.append({
            "line_id": line_id, "start": start, "end": end,
            "raw_component_ids": tuple(value.component_id for value in members),
            "birth_scale": birth, "death_scale": death, "persistence": persistence,
            "normalized_persistence": persistence / MAX_SCALE,
            "raw_component_count": len(members), "bbox_native": _union_box(members),
            "foreground_area": sum(value.area for value in members),
            "median_internal_gap_pixels": float(median(internal)) if internal else None,
            "max_internal_gap_pixels": max(internal) if internal else None,
            "left_external_gap_pixels": _gap(line[start - 1], line[start]) if start else None,
            "right_external_gap_pixels": _gap(line[end - 1], line[end]) if end < len(line) else None,
        })
    return nodes, {
        "line_id": line_id, "raw_component_ids": tuple(value.component_id for value in line),
        "median_raw_component_height": reference_height,
        "scale_partitions": tuple(partitions),
    }


def _better(left: tuple[float, int, int, tuple], right: tuple[float, int, int, tuple]) -> bool:
    """Exact score tie: cover more CCs, then use fewer/lexicographically earlier spans."""
    if not math.isclose(left[0], right[0], rel_tol=0.0, abs_tol=1e-12):
        return left[0] > right[0]
    return (left[1], -left[2], tuple(reversed(left[3]))) > (
        right[1], -right[2], tuple(reversed(right[3])))


def _select_line(nodes: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    ordered = sorted(nodes, key=lambda row: (row["end"], row["start"], row["raw_component_ids"]))
    compatible = [max((j for j in range(i) if ordered[j]["end"] <= row["start"]), default=-1)
                  for i, row in enumerate(ordered)]
    states: list[tuple[float, int, int, tuple[int, ...]]] = [(0.0, 0, 0, ())]
    for index, row in enumerate(ordered):
        skip = states[-1]
        base = states[compatible[index] + 1]
        weight = row["normalized_persistence"] * row["raw_component_count"]
        take = (base[0] + weight, base[1] + row["raw_component_count"],
                base[2] + 1, (*base[3], index))
        states.append(take if _better(take, skip) else skip)
    return tuple(ordered[index] for index in states[-1][3])


def group_multiscale_component_persistence(mask: np.ndarray) -> PersistenceResult:
    """Group native raw CCs without accepting GT or image/text inputs."""
    labels, raw = _extract(mask)
    lines = _lines(raw)
    nodes, line_records, selected = [], [], []
    for line_id, line in enumerate(lines, 1):
        line_nodes, record = _nodes_for_line(line_id, line, labels)
        chosen = _select_line(line_nodes)
        nodes.extend(line_nodes)
        line_records.append(record)
        selected.extend(chosen)
    candidates = tuple(sorted(nodes, key=lambda row: (
        row["line_id"], row["start"], row["end"], row["birth_scale"]
    )))
    selected = tuple(sorted(selected, key=lambda row: (
        -row["foreground_area"], row["bbox_native"][1], row["bbox_native"][0],
        row["raw_component_ids"],
    )))
    flat = [value for row in selected for value in row["raw_component_ids"]]
    if sorted(flat) != sorted(value.component_id for value in raw) or len(flat) != len(set(flat)):
        raise RuntimeError("selected persistence spans must partition all raw components")
    return PersistenceResult(raw, tuple(line_records), tuple(nodes), candidates, selected)


def groups_in_original_coordinates(
    result: PersistenceResult, width: int, height: int,
) -> tuple[tuple[dict[str, Any], ...], frozenset[tuple[int, ...]]]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    groups = []
    for group_id, source in enumerate(result.selected, 1):
        native = source["bbox_native"]
        box = (math.floor(native[0] * width / 128), math.floor(native[1] * height / 128),
               math.ceil(native[2] * width / 128), math.ceil(native[3] * height / 128))
        groups.append({
            "group_id": group_id, "line_id": source["line_id"],
            "raw_component_ids": source["raw_component_ids"],
            "foreground_area": source["foreground_area"], "bbox": box,
            "padded_bbox": (max(0, box[0] - PAD_ORIGINAL), max(0, box[1] - PAD_ORIGINAL),
                            min(width, box[2] + PAD_ORIGINAL), min(height, box[3] + PAD_ORIGINAL)),
            "birth_scale": source["birth_scale"], "death_scale": source["death_scale"],
            "persistence": source["persistence"],
            "normalized_persistence": source["normalized_persistence"],
        })
    keys = [tuple(row["raw_component_ids"]) for row in groups]
    return tuple(groups), frozenset(keys[:NMAX])


__all__ = ["MODE", "SCALES", "PersistenceResult", "group_multiscale_component_persistence",
           "groups_in_original_coordinates"]
