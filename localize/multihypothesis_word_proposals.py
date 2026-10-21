"""Inference-safe multi-hypothesis word proposals from native probabilities.

This module is a separate PRACTICAL_APPROXIMATION diagnostic.  It does not use
GT, text, OCR, or learned proposal parameters and does not alter Algorithm1 or
practical-remap-v1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Final, Iterable, Sequence

import numpy as np

from .contiguous_span_candidates import (
    MAX_WIDTH_OVER_MEDIAN_HEIGHT,
    PAD_ORIGINAL,
    frozen_line_params,
)
from .grouped_word_blocks import (
    GroupedWordBlockParams,
    RawComponent as V1RawComponent,
    compatible_components,
)
from .line_word_blocks import RawComponent, prepare_word_blocks_v2


MODE: Final = "MULTIHYPOTHESIS_WORD_PROPOSAL_ORACLE_TRAIN_V1"
THRESHOLDS: Final = (0.20, 0.30, 0.40, 0.50)
MAX_COMPONENTS_FULL: Final = 20
DEDUPLICATION_IOU: Final = 0.95
BBOX_VARIANTS: Final = ("standard_pad6", "tight", "adaptive")


@dataclass(frozen=True)
class NativeHypotheses:
    raw_components: tuple[RawComponent, ...]
    base_lines: tuple[dict[str, Any], ...]
    robust_lines: tuple[dict[str, Any], ...]
    cross_line_edges: tuple[dict[str, Any], ...]
    spans: tuple[dict[str, Any], ...]


def frozen_cross_line_params() -> GroupedWordBlockParams:
    """The already selected v1 gates, reused without alteration."""
    return GroupedWordBlockParams(
        horizontal_gap_ratio=0.35,
        vertical_overlap_min=0.7,
        max_center_y_ratio=0.5,
        max_height_ratio=2.5,
        pad=PAD_ORIGINAL,
    )


def _union_boxes(boxes: Iterable[Sequence[int]]) -> tuple[int, int, int, int]:
    values = tuple(boxes)
    if not values:
        raise ValueError("cannot union no boxes")
    return (
        min(int(value[0]) for value in values),
        min(int(value[1]) for value in values),
        max(int(value[2]) for value in values),
        max(int(value[3]) for value in values),
    )


def _robust_line_orders(
    raw_components: Sequence[RawComponent],
    base_lines: Sequence[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Join existing line IDs only through frozen-v1 cross-line raw-CC edges."""
    if not base_lines:
        return (), ()
    raw_by_id = {value.component_id: value for value in raw_components}
    line_by_raw = {
        int(raw_id): int(line["line_id"])
        for line in base_lines
        for raw_id in line["raw_component_ids"]
    }
    if set(line_by_raw) != set(raw_by_id):
        raise RuntimeError("base lines must partition every raw component")
    parent = {int(line["line_id"]): int(line["line_id"]) for line in base_lines}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    params = frozen_cross_line_params()
    edges = []
    raw = tuple(raw_components)
    for index, left in enumerate(raw):
        for right in raw[index + 1 :]:
            left_line, right_line = line_by_raw[left.component_id], line_by_raw[right.component_id]
            if left_line == right_line:
                continue
            edge = compatible_components(
                V1RawComponent(left.component_id, left.foreground_area, left.bbox, left.centroid),
                V1RawComponent(right.component_id, right.foreground_area, right.bbox, right.centroid),
                params,
            )
            if edge is None:
                continue
            union(left_line, right_line)
            edges.append({**edge, "left_line_id": left_line, "right_line_id": right_line})

    grouped: dict[int, list[int]] = {}
    for line_id in parent:
        grouped.setdefault(find(line_id), []).append(line_id)
    robust = []
    for robust_id, line_ids in enumerate(
        (tuple(sorted(values)) for _, values in sorted(grouped.items()) if len(values) >= 2),
        start=1,
    ):
        ids = tuple(sorted(
            (raw_id for raw_id, line_id in line_by_raw.items() if line_id in line_ids),
            key=lambda raw_id: (
                raw_by_id[raw_id].bbox[0], raw_by_id[raw_id].bbox[1], raw_id,
            ),
        ))
        robust.append({
            "robust_line_id": robust_id,
            "source_line_ids": line_ids,
            "raw_component_ids": ids,
        })
    return tuple(robust), tuple(edges)


def _spans_for_order(
    *, raw_by_id: dict[int, RawComponent], raw_ids: tuple[int, ...],
    source_kind: str, source_line_ids: tuple[int, ...], max_components: int,
) -> list[dict[str, Any]]:
    if not raw_ids:
        return []
    reference_height = float(median(
        raw_by_id[value].bbox[3] - raw_by_id[value].bbox[1] for value in raw_ids
    ))
    values = []
    for start in range(len(raw_ids)):
        stop = min(len(raw_ids), start + max_components)
        for end in range(start + 1, stop + 1):
            ids = raw_ids[start:end]
            box = _union_boxes(raw_by_id[value].bbox for value in ids)
            width_ratio = (box[2] - box[0]) / reference_height
            if width_ratio > MAX_WIDTH_OVER_MEDIAN_HEIGHT:
                break
            values.append({
                "source_kind": source_kind,
                "source_line_ids": source_line_ids,
                "start": start,
                "end": end,
                "raw_component_ids": ids,
                "bbox_native": box,
                "foreground_area_native": sum(raw_by_id[value].foreground_area for value in ids),
                "median_component_height_native": reference_height,
                "width_over_median_component_height": width_ratio,
            })
    return values


def enumerate_native_hypotheses(
    mask: np.ndarray, *, max_components: int = MAX_COMPONENTS_FULL,
) -> NativeHypotheses:
    if (
        mask.dtype != np.uint8
        or mask.shape != (128, 128)
        or not np.isin(mask, (0, 1)).all()
    ):
        raise ValueError("mask must be binary uint8 0/1 at native 128x128")
    if not 1 <= max_components <= MAX_COMPONENTS_FULL:
        raise ValueError("max_components must be between 1 and 20")
    prepared = prepare_word_blocks_v2(mask, frozen_line_params())
    raw_by_id = {value.component_id: value for value in prepared.raw_components}
    base_lines = tuple({
        "line_id": int(line["line_id"]),
        "raw_component_ids": tuple(sorted(
            (int(value) for value in line["raw_component_ids"]),
            key=lambda value: (raw_by_id[value].bbox[0], raw_by_id[value].bbox[1], value),
        )),
    } for line in prepared.line_candidates)
    robust_lines, cross_edges = _robust_line_orders(prepared.raw_components, base_lines)
    spans = []
    for line in base_lines:
        spans.extend(_spans_for_order(
            raw_by_id=raw_by_id,
            raw_ids=line["raw_component_ids"],
            source_kind="same_line",
            source_line_ids=(int(line["line_id"]),),
            max_components=max_components,
        ))
    for line in robust_lines:
        spans.extend(_spans_for_order(
            raw_by_id=raw_by_id,
            raw_ids=line["raw_component_ids"],
            source_kind="line_robust",
            source_line_ids=tuple(line["source_line_ids"]),
            max_components=max_components,
        ))
    return NativeHypotheses(
        raw_components=prepared.raw_components,
        base_lines=base_lines,
        robust_lines=robust_lines,
        cross_line_edges=cross_edges,
        spans=tuple(spans),
    )


def _remap_native_box(
    box: Sequence[int], width: int, height: int,
) -> tuple[int, int, int, int]:
    return (
        math.floor(int(box[0]) * width / 128),
        math.floor(int(box[1]) * height / 128),
        math.ceil(int(box[2]) * width / 128),
        math.ceil(int(box[3]) * height / 128),
    )


def _pad_box(
    box: Sequence[int], pad: int, width: int, height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, int(box[0]) - pad), max(0, int(box[1]) - pad),
        min(width, int(box[2]) + pad), min(height, int(box[3]) + pad),
    )


def component_union_bbox_variants(
    raw_by_id: dict[int, RawComponent], raw_component_ids: Sequence[int],
    *, width: int, height: int,
) -> tuple[dict[str, tuple[int, int, int, int]], int]:
    if not raw_component_ids:
        raise ValueError("at least one raw component is required")
    tight = _remap_native_box(
        _union_boxes(raw_by_id[value].bbox for value in raw_component_ids), width, height
    )
    original_heights = [
        _remap_native_box(raw_by_id[value].bbox, width, height)[3]
        - _remap_native_box(raw_by_id[value].bbox, width, height)[1]
        for value in raw_component_ids
    ]
    adaptive_pad = max(PAD_ORIGINAL, round(0.15 * float(median(original_heights))))
    return {
        "standard_pad6": _pad_box(tight, PAD_ORIGINAL, width, height),
        "tight": tight,
        "adaptive": _pad_box(tight, adaptive_pad, width, height),
    }, adaptive_pad


def materialize_bbox_variants(
    hypotheses: NativeHypotheses, *, threshold: float, width: int, height: int,
) -> tuple[dict[str, Any], ...]:
    if threshold not in THRESHOLDS:
        raise ValueError("threshold is outside the frozen ensemble")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    raw_by_id = {value.component_id: value for value in hypotheses.raw_components}
    values = []
    for span_index, source in enumerate(hypotheses.spans):
        boxes, adaptive_pad = component_union_bbox_variants(
            raw_by_id, source["raw_component_ids"], width=width, height=height
        )
        for variant in BBOX_VARIANTS:
            values.append({
                **source,
                "span_index": span_index,
                "threshold": threshold,
                "bbox_variant": variant,
                "bbox_original": boxes[variant],
                "adaptive_pad_original": adaptive_pad,
            })
    return tuple(values)


def box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def deduplicate_candidates(
    candidates: Sequence[dict[str, Any]], *, threshold: float = DEDUPLICATION_IOU,
) -> tuple[dict[str, Any], ...]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("deduplication threshold must be in [0,1]")
    boxes = np.asarray([candidate["bbox_original"] for candidate in candidates], dtype=np.float64)
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
             if len(boxes) else np.empty(0, dtype=np.float64))
    retained_indices: list[int] = []
    for index, candidate in enumerate(candidates):
        if retained_indices:
            previous = boxes[retained_indices]
            intersection_width = np.maximum(
                0.0, np.minimum(boxes[index, 2], previous[:, 2])
                - np.maximum(boxes[index, 0], previous[:, 0]),
            )
            intersection_height = np.maximum(
                0.0, np.minimum(boxes[index, 3], previous[:, 3])
                - np.maximum(boxes[index, 1], previous[:, 1]),
            )
            intersection = intersection_width * intersection_height
            union = areas[index] + areas[retained_indices] - intersection
            ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            duplicate = bool(np.any(ious >= threshold))
        else:
            duplicate = False
        if duplicate:
            continue
        retained_indices.append(index)
    return tuple(candidates[index] for index in retained_indices)


def deduplicate_candidates_with_provenance(
    candidates: Sequence[dict[str, Any]], *, threshold: float = DEDUPLICATION_IOU,
) -> tuple[dict[str, Any], ...]:
    """Deduplicate boxes while retaining every contributing hypothesis.

    The retained boxes and their order are identical to
    :func:`deduplicate_candidates`.  Only provenance metadata is added, so
    frozen crop/OCR artifacts can be reused byte-for-byte.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("deduplication threshold must be in [0,1]")
    retained: list[dict[str, Any]] = []
    for candidate in candidates:
        matches = [
            (box_iou(candidate["bbox_original"], row["bbox_original"]), index)
            for index, row in enumerate(retained)
            if box_iou(candidate["bbox_original"], row["bbox_original"]) >= threshold
        ]
        source = {
            "threshold": float(candidate["threshold"]),
            "bbox_variant": str(candidate["bbox_variant"]),
            "source_kind": str(candidate["source_kind"]),
            "source_line_ids": tuple(int(value) for value in candidate["source_line_ids"]),
            "raw_component_ids": tuple(int(value) for value in candidate["raw_component_ids"]),
            "bbox_native": tuple(int(value) for value in candidate["bbox_native"]),
            "bbox_original": tuple(int(value) for value in candidate["bbox_original"]),
        }
        if matches:
            _, index = max(matches, key=lambda value: (value[0], -value[1]))
            retained[index]["dedup_sources"].append(source)
            continue
        row = dict(candidate)
        row["dedup_sources"] = [source]
        retained.append(row)
    return tuple(retained)


__all__ = [
    "BBOX_VARIANTS", "DEDUPLICATION_IOU", "MAX_COMPONENTS_FULL", "MODE", "THRESHOLDS",
    "NativeHypotheses", "box_iou", "component_union_bbox_variants", "deduplicate_candidates",
    "deduplicate_candidates_with_provenance",
    "enumerate_native_hypotheses", "frozen_cross_line_params", "materialize_bbox_variants",
]
