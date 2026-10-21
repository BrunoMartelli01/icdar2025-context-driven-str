"""Inference-safe contiguous raw-component span enumeration.

This TRAIN diagnostic is separate from Algorithm1 and practical-remap-v1/v2.
It accepts only a native binary mask; GT and recognition cannot affect candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Final

import numpy as np

from .line_word_blocks import V2WordBlockParams, prepare_word_blocks_v2


MODE: Final = "CONTIGUOUS_SPAN_ORACLE_COVERAGE_V1"
MAX_COMPONENTS_PER_SPAN: Final = 12
MAX_WIDTH_OVER_MEDIAN_HEIGHT: Final = 15.0
PAD_ORIGINAL: Final = 6


@dataclass(frozen=True)
class SpanCandidateResult:
    raw_component_count: int
    lines: tuple[dict[str, Any], ...]
    candidates_native: tuple[dict[str, Any], ...]


def frozen_line_params() -> V2WordBlockParams:
    """The already frozen v2 preparation geometry; boundary values are unused."""
    return V2WordBlockParams(
        family="hybrid", absolute_gap_floor=.35, local_window=2,
        local_gap_ratio=1.5, local_fallback_absolute_floor=.35,
        line_vertical_overlap_min=.7, line_max_center_y_ratio=.5,
        line_max_height_ratio=2.5, fragment_x_overlap_min=.5,
        fragment_max_vertical_gap_ratio=.5, fragment_small_height_ratio=.45,
        pad=6,
    )


def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (min(v[0] for v in boxes), min(v[1] for v in boxes),
            max(v[2] for v in boxes), max(v[3] for v in boxes))


def enumerate_contiguous_span_candidates(mask: np.ndarray) -> SpanCandidateResult:
    if mask.dtype != np.uint8 or mask.shape != (128, 128) or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must be binary uint8 0/1 at native 128x128")
    prepared = prepare_word_blocks_v2(mask, frozen_line_params())
    raw = {value.component_id: value for value in prepared.raw_components}
    lines, candidates = [], []
    seen_ids = []
    for line in prepared.line_candidates:
        ordered = tuple(sorted((int(value) for value in line["raw_component_ids"]),
                               key=lambda value: (raw[value].bbox[0], raw[value].bbox[1], value)))
        seen_ids.extend(ordered)
        heights = [raw[value].bbox[3] - raw[value].bbox[1] for value in ordered]
        reference_height = float(median(heights))
        accepted = 0
        for start in range(len(ordered)):
            for end in range(start + 1, min(len(ordered), start + MAX_COMPONENTS_PER_SPAN) + 1):
                ids = ordered[start:end]
                box = _union([raw[value].bbox for value in ids])
                width_ratio = (box[2] - box[0]) / reference_height
                if width_ratio > MAX_WIDTH_OVER_MEDIAN_HEIGHT:
                    # Width is monotone for a fixed start, so longer spans also fail.
                    break
                candidates.append({
                    "line_id": int(line["line_id"]), "start": start, "end": end,
                    "raw_component_ids": ids, "bbox_native": box,
                    "foreground_area_native": sum(raw[value].foreground_area for value in ids),
                    "median_component_height_native": reference_height,
                    "width_over_median_component_height": width_ratio,
                })
                accepted += 1
        lines.append({"line_id": int(line["line_id"]), "raw_component_ids": ordered,
                      "raw_component_count": len(ordered), "candidate_span_count": accepted,
                      "median_component_height_native": reference_height})
    if sorted(seen_ids) != sorted(raw) or len(seen_ids) != len(set(seen_ids)):
        raise RuntimeError("frozen line formation must partition raw components")
    return SpanCandidateResult(len(raw), tuple(lines), tuple(candidates))


def remap_and_deduplicate_candidates(
    result: SpanCandidateResult, width: int, height: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, int]]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    unique: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for source in result.candidates_native:
        native = source["bbox_native"]
        box = (math.floor(native[0] * width / 128), math.floor(native[1] * height / 128),
               math.ceil(native[2] * width / 128), math.ceil(native[3] * height / 128))
        padded = (max(0, box[0] - PAD_ORIGINAL), max(0, box[1] - PAD_ORIGINAL),
                  min(width, box[2] + PAD_ORIGINAL), min(height, box[3] + PAD_ORIGINAL))
        row = {**source, "bbox_original": box, "padded_bbox_original": padded}
        current = unique.get(padded)
        canonical = (len(row["raw_component_ids"]), row["raw_component_ids"], row["line_id"])
        if current is None or canonical < (
            len(current["raw_component_ids"]), current["raw_component_ids"], current["line_id"]
        ):
            unique[padded] = row
    values = tuple(sorted(unique.values(), key=lambda row: (
        row["padded_bbox_original"][1], row["padded_bbox_original"][0],
        row["padded_bbox_original"][3], row["padded_bbox_original"][2],
        row["raw_component_ids"],
    )))
    return values, {"before_exact_final_bbox_deduplication": len(result.candidates_native),
                    "after_exact_final_bbox_deduplication": len(values),
                    "removed": len(result.candidates_native) - len(values)}


__all__ = ["MODE", "MAX_COMPONENTS_PER_SPAN", "MAX_WIDTH_OVER_MEDIAN_HEIGHT",
           "enumerate_contiguous_span_candidates", "remap_and_deduplicate_candidates"]
