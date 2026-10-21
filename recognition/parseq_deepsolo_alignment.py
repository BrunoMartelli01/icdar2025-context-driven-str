"""Spatial/text diagnostics between PARSeq crop outputs and DeepSolo detections.

This module does not define recognition accuracy or a paper matching protocol.  It
validates an existing APPROXIMATE trace ledger, retains every crop/detection pair, and
reports explicitly named axis-aligned-envelope observations in source-image coordinates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from rapidfuzz.fuzz import ratio

from recognition.icdar13_annotations import ICDAR13RecognitionSample
from recognition.icdar13_trace_diagnostics import diagnose_trace_index


class ParseqDeepSoloAlignmentError(ValueError):
    """Raised when model outputs cannot satisfy the diagnostic contract."""


def _box(value: Any, *, label: str, image_size_wh: tuple[int, int]) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ParseqDeepSoloAlignmentError("{} must be an xyxy box".format(label))
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ParseqDeepSoloAlignmentError("{} coordinates must be numeric".format(label)) from error
    if not all(math.isfinite(item) for item in box):
        raise ParseqDeepSoloAlignmentError("{} coordinates must be finite".format(label))
    x1, y1, x2, y2 = box
    width, height = image_size_wh
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ParseqDeepSoloAlignmentError(
            "{} must be a positive box inside the source image".format(label)
        )
    return box


def _detection_envelope(
    detection: dict[str, Any], *, label: str, image_size_wh: tuple[int, int]
) -> tuple[float, ...]:
    geometry = detection.get("geometry")
    if not isinstance(geometry, dict):
        raise ParseqDeepSoloAlignmentError("{} geometry is missing".format(label))
    points = geometry.get("boundary_points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) < 3:
        raise ParseqDeepSoloAlignmentError(
            "{} boundary_points must contain at least three points".format(label)
        )
    normalized: list[tuple[float, float]] = []
    for point in points:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            raise ParseqDeepSoloAlignmentError(
                "{} contains an invalid boundary point".format(label)
            )
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as error:
            raise ParseqDeepSoloAlignmentError(
                "{} boundary coordinates must be numeric".format(label)
            ) from error
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ParseqDeepSoloAlignmentError(
                "{} boundary coordinates must be finite".format(label)
            )
        normalized.append((x, y))
    return _box(
        (
            min(point[0] for point in normalized),
            min(point[1] for point in normalized),
            max(point[0] for point in normalized),
            max(point[1] for point in normalized),
        ),
        label="{} boundary envelope".format(label),
        image_size_wh=image_size_wh,
    )


def _overlap(left: tuple[float, ...], right: tuple[float, ...]) -> dict[str, float]:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return {
        "intersection_area": intersection,
        "bbox_iou": intersection / union,
        "deepsolo_envelope_coverage": intersection / left_area,
        "parseq_crop_coverage": intersection / right_area,
    }


def _text_metrics(parseq_text: str, deepsolo_text: str) -> dict[str, Any]:
    both_nonempty = bool(parseq_text) and bool(deepsolo_text)
    return {
        "case_sensitive_exact": both_nonempty and parseq_text == deepsolo_text,
        "casefold_exact": both_nonempty and parseq_text.casefold() == deepsolo_text.casefold(),
        "rapidfuzz_ratio_raw": ratio(parseq_text, deepsolo_text) / 100.0,
        "rapidfuzz_ratio_casefold": ratio(
            parseq_text.casefold(), deepsolo_text.casefold()
        ) / 100.0,
    }


def _fraction(numerator: int, denominator: int, *, available: bool) -> float | None:
    if not available or denominator == 0:
        return None
    return numerator / denominator


def _range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _best_pair(
    pairs: list[dict[str, Any]], *, primary: str, secondary: str, tertiary: str,
    final_tie_key: str,
) -> dict[str, Any] | None:
    if not pairs:
        return None
    best = max(
        pairs,
        key=lambda item: (
            float(item[primary]),
            float(item[secondary]),
            float(item[tertiary]),
            -int(item[final_tie_key]),
        ),
    )
    return best if float(best[primary]) > 0 else None


def diagnose_parseq_deepsolo_alignment(
    index_path: Path,
    samples: Iterable[ICDAR13RecognitionSample],
) -> dict[str, Any]:
    """Validate traces and report non-evaluative PARSeq/DeepSolo alignment evidence."""

    samples = tuple(samples)
    base = diagnose_trace_index(index_path, samples)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    index_path = index_path.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    fractions_available = base["population"]["failed_sample_count"] == 0

    counters = {
        "fallback_samples": 0,
        "parseq_crops": 0,
        "deepsolo_detections": 0,
        "detections_positive_overlap": 0,
        "detections_center_inside_crop": 0,
        "detections_coverage_ge_0_5": 0,
        "detections_coverage_ge_0_8": 0,
        "crops_positive_overlap": 0,
        "crops_coverage_ge_0_5": 0,
        "crops_coverage_ge_0_8": 0,
        "spatial_pair_raw_exact": 0,
        "spatial_pair_casefold_exact": 0,
        "spatial_pair_casefold_ratio_ge_0_8": 0,
        "t1_any_raw_exact": 0,
        "t1_any_casefold_exact": 0,
        "selected_t3_any_raw_exact": 0,
        "selected_t3_any_casefold_exact": 0,
        "detections_any_parseq_casefold_exact": 0,
    }
    detection_best_coverages: list[float] = []
    detection_best_ious: list[float] = []
    crop_best_coverages: list[float] = []
    spatial_pair_casefold_ratios: list[float] = []
    per_sample: list[dict[str, Any]] = []

    for record in index["samples"]:
        sample_id = str(record["sample_id"])
        if not record["success"]:
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "success": False,
                    "alignment_evaluable": False,
                    "reason": "source trace failed",
                }
            )
            continue
        sample = sample_by_id[sample_id]
        result = json.loads(Path(record["result"]).read_text(encoding="utf-8"))
        fallback = result["fallback"]
        if not fallback["invoked"]:
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "success": True,
                    "alignment_evaluable": False,
                    "reason": "DeepSolo was not invoked, so no DeepSolo output exists",
                    "T1": result["T1"]["raw_text"],
                    "selected_T3": result["T3"]["selected"]["raw_text"],
                }
            )
            continue

        counters["fallback_samples"] += 1
        blocks = result["segmentation"].get("localized_blocks")
        t3_all = result["T3"].get("all")
        detections = fallback.get("raw_outputs")
        if not isinstance(blocks, list) or not isinstance(t3_all, list):
            raise ParseqDeepSoloAlignmentError(
                "Sample {} PARSeq crop records are missing".format(sample_id)
            )
        if len(blocks) != len(t3_all):
            raise ParseqDeepSoloAlignmentError(
                "Sample {} crop/T3 cardinality mismatch".format(sample_id)
            )
        if not isinstance(detections, list):
            raise ParseqDeepSoloAlignmentError(
                "Sample {} DeepSolo detections are missing".format(sample_id)
            )

        crop_records: list[dict[str, Any]] = []
        ranks: set[int] = set()
        for block, prediction in zip(blocks, t3_all):
            if not isinstance(block, dict) or not isinstance(prediction, dict):
                raise ParseqDeepSoloAlignmentError(
                    "Sample {} contains an invalid crop/T3 record".format(sample_id)
                )
            rank = block.get("rank")
            if not isinstance(rank, int) or rank < 1 or rank in ranks:
                raise ParseqDeepSoloAlignmentError(
                    "Sample {} PARSeq ranks must be unique positive integers".format(sample_id)
                )
            ranks.add(rank)
            text = prediction.get("raw_text")
            if not isinstance(text, str):
                raise ParseqDeepSoloAlignmentError(
                    "Sample {} PARSeq text must be a string".format(sample_id)
                )
            crop_records.append(
                {
                    "rank": rank,
                    "padded_bbox": list(
                        _box(
                            block.get("padded_bbox"),
                            label="Sample {} PARSeq rank {}".format(sample_id, rank),
                            image_size_wh=sample.image_size_wh,
                        )
                    ),
                    "text": text,
                    "confidence": prediction.get("confidence"),
                }
            )

        detection_records: list[dict[str, Any]] = []
        source_indices: set[int] = set()
        for offset, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ParseqDeepSoloAlignmentError(
                    "Sample {} contains an invalid DeepSolo detection".format(sample_id)
                )
            source_index = detection.get("source_index")
            if (
                not isinstance(source_index, int)
                or source_index < 0
                or source_index in source_indices
            ):
                raise ParseqDeepSoloAlignmentError(
                    "Sample {} DeepSolo source indices must be unique nonnegative integers".format(
                        sample_id
                    )
                )
            source_indices.add(source_index)
            text = detection.get("text")
            if not isinstance(text, str):
                raise ParseqDeepSoloAlignmentError(
                    "Sample {} DeepSolo text must be a string".format(sample_id)
                )
            envelope = _detection_envelope(
                detection,
                label="Sample {} DeepSolo detection {}".format(sample_id, offset),
                image_size_wh=sample.image_size_wh,
            )
            detection_records.append(
                {
                    "source_index": source_index,
                    "boundary_envelope": list(envelope),
                    "text": text,
                    "score": detection.get("score"),
                }
            )

        counters["parseq_crops"] += len(crop_records)
        counters["deepsolo_detections"] += len(detection_records)
        pairs: list[dict[str, Any]] = []
        for crop in crop_records:
            for detection in detection_records:
                overlap = _overlap(
                    tuple(detection["boundary_envelope"]),
                    tuple(crop["padded_bbox"]),
                )
                envelope = detection["boundary_envelope"]
                center_x = (envelope[0] + envelope[2]) / 2.0
                center_y = (envelope[1] + envelope[3]) / 2.0
                crop_box = crop["padded_bbox"]
                pairs.append(
                    {
                        "parseq_rank": crop["rank"],
                        "deepsolo_source_index": detection["source_index"],
                        **overlap,
                        "deepsolo_envelope_center_inside_parseq_crop": (
                            crop_box[0] <= center_x <= crop_box[2]
                            and crop_box[1] <= center_y <= crop_box[3]
                        ),
                        **_text_metrics(crop["text"], detection["text"]),
                    }
                )

        best_by_detection: list[dict[str, Any]] = []
        for detection in detection_records:
            candidates = [
                pair
                for pair in pairs
                if pair["deepsolo_source_index"] == detection["source_index"]
            ]
            best = _best_pair(
                candidates,
                primary="deepsolo_envelope_coverage",
                secondary="bbox_iou",
                tertiary="parseq_crop_coverage",
                final_tie_key="parseq_rank",
            )
            any_casefold_exact = any(
                pair["casefold_exact"] for pair in candidates
            )
            counters["detections_any_parseq_casefold_exact"] += int(any_casefold_exact)
            if best is not None:
                coverage = float(best["deepsolo_envelope_coverage"])
                detection_best_coverages.append(coverage)
                detection_best_ious.append(float(best["bbox_iou"]))
                spatial_pair_casefold_ratios.append(
                    float(best["rapidfuzz_ratio_casefold"])
                )
                counters["detections_positive_overlap"] += 1
                counters["detections_coverage_ge_0_5"] += int(coverage >= 0.5)
                counters["detections_coverage_ge_0_8"] += int(coverage >= 0.8)
                counters["spatial_pair_raw_exact"] += int(best["case_sensitive_exact"])
                counters["spatial_pair_casefold_exact"] += int(best["casefold_exact"])
                counters["spatial_pair_casefold_ratio_ge_0_8"] += int(
                    best["rapidfuzz_ratio_casefold"] >= 0.8
                )
            center_inside = any(
                pair["deepsolo_envelope_center_inside_parseq_crop"]
                for pair in candidates
            )
            counters["detections_center_inside_crop"] += int(center_inside)
            best_by_detection.append(
                {
                    "deepsolo_source_index": detection["source_index"],
                    "best_positive_spatial_pair": best,
                    "center_inside_any_parseq_crop": center_inside,
                    "casefold_exact_with_any_parseq_crop": any_casefold_exact,
                }
            )

        best_by_crop: list[dict[str, Any]] = []
        for crop in crop_records:
            candidates = [pair for pair in pairs if pair["parseq_rank"] == crop["rank"]]
            best = _best_pair(
                candidates,
                primary="parseq_crop_coverage",
                secondary="bbox_iou",
                tertiary="deepsolo_envelope_coverage",
                final_tie_key="deepsolo_source_index",
            )
            if best is not None:
                coverage = float(best["parseq_crop_coverage"])
                crop_best_coverages.append(coverage)
                counters["crops_positive_overlap"] += 1
                counters["crops_coverage_ge_0_5"] += int(coverage >= 0.5)
                counters["crops_coverage_ge_0_8"] += int(coverage >= 0.8)
            best_by_crop.append(
                {
                    "parseq_rank": crop["rank"],
                    "best_positive_spatial_pair": best,
                }
            )

        deepsolo_texts = [record["text"] for record in detection_records]
        t1 = result["T1"]["raw_text"]
        selected_t3 = result["T3"]["selected"]["raw_text"]
        t1_raw_exact = bool(t1) and t1 in deepsolo_texts
        t1_casefold_exact = bool(t1) and t1.casefold() in {
            text.casefold() for text in deepsolo_texts
        }
        selected_raw_exact = bool(selected_t3) and selected_t3 in deepsolo_texts
        selected_casefold_exact = bool(selected_t3) and selected_t3.casefold() in {
            text.casefold() for text in deepsolo_texts
        }
        counters["t1_any_raw_exact"] += int(t1_raw_exact)
        counters["t1_any_casefold_exact"] += int(t1_casefold_exact)
        counters["selected_t3_any_raw_exact"] += int(selected_raw_exact)
        counters["selected_t3_any_casefold_exact"] += int(selected_casefold_exact)

        per_sample.append(
            {
                "sample_id": sample_id,
                "success": True,
                "alignment_evaluable": True,
                "image_size_wh": list(sample.image_size_wh),
                "T1": {
                    "text": t1,
                    "spatial_scope": "full source image; no unique DeepSolo instance association",
                    "raw_exact_with_any_deepsolo_detection": t1_raw_exact,
                    "casefold_exact_with_any_deepsolo_detection": t1_casefold_exact,
                    "maximum_casefold_rapidfuzz_ratio": max(
                        (ratio(t1.casefold(), text.casefold()) / 100.0 for text in deepsolo_texts),
                        default=None,
                    ),
                },
                "selected_T3": {
                    "text": selected_t3,
                    "rank": result["T3"].get("selected_rank"),
                    "raw_exact_with_any_deepsolo_detection": selected_raw_exact,
                    "casefold_exact_with_any_deepsolo_detection": selected_casefold_exact,
                    "maximum_casefold_rapidfuzz_ratio": max(
                        (
                            ratio(selected_t3.casefold(), text.casefold()) / 100.0
                            for text in deepsolo_texts
                        ),
                        default=None,
                    ),
                },
                "parseq_crops": crop_records,
                "deepsolo_detections": detection_records,
                "pairwise_alignment": pairs,
                "best_by_deepsolo_detection": best_by_detection,
                "best_by_parseq_crop": best_by_crop,
            }
        )

    detection_count = counters["deepsolo_detections"]
    crop_count = counters["parseq_crops"]
    positive_pair_count = counters["detections_positive_overlap"]
    fallback_samples = counters["fallback_samples"]
    return {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "PARSeq/DeepSolo alignment diagnostics; not recognition accuracy",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "accuracy": None,
        "evaluation_status": "UNRESOLVED",
        "scientific_boundary": {
            "classification": "ASSUMPTION",
            "spatial_method": (
                "axis-aligned envelope of DeepSolo boundary_points versus the exact "
                "padded source-image bbox supplied to each PARSeq T3 call"
            ),
            "matching_method": (
                "retain every pair; best-pair summaries maximize asymmetric coverage, "
                "then IoU, opposite coverage, and lowest source rank/index"
            ),
            "coordinate_scope": "original source-image coordinates",
            "text_methods": (
                "raw exact, Unicode casefold exact, and RapidFuzz ratio; no punctuation, "
                "whitespace, lexicon, or geometry-aware transcription normalization"
            ),
            "diagnostic_thresholds": {
                "spatial_coverage": [0.5, 0.8],
                "casefold_rapidfuzz_ratio": 0.8,
                "classification": "ASSUMPTION",
            },
            "T1_limitation": (
                "PARSeq T1 consumes the full image and cannot be assigned to one "
                "DeepSolo instance from the recorded outputs"
            ),
            "blocking_ambiguity_ids": [
                "A-008", "L-004", "R-003", "D-001", "D-002", "E-001"
            ],
            "prohibition": "Alignment diagnostics must not be reported as Table 5 accuracy.",
        },
        "source": base["source"],
        "population": {
            **base["population"],
            "fallback_alignment_sample_count": fallback_samples,
            "samples_without_deepsolo_output": (
                base["population"]["successful_sample_count"] - fallback_samples
            ),
            "parseq_crop_count": crop_count,
            "deepsolo_detection_count": detection_count,
        },
        "aggregation_guard": {
            "fractions_available": fractions_available,
            "rule": "Fractions are suppressed when any source trace failed.",
        },
        "spatial_envelope_diagnostics": {
            "deepsolo_detection_denominator": detection_count,
            "detections_with_positive_overlap": counters["detections_positive_overlap"],
            "detections_with_positive_overlap_fraction": _fraction(
                counters["detections_positive_overlap"], detection_count,
                available=fractions_available,
            ),
            "detections_with_center_inside_any_crop": counters[
                "detections_center_inside_crop"
            ],
            "detections_with_center_inside_any_crop_fraction": _fraction(
                counters["detections_center_inside_crop"], detection_count,
                available=fractions_available,
            ),
            "detections_best_coverage_ge_0_5": counters["detections_coverage_ge_0_5"],
            "detections_best_coverage_ge_0_8": counters["detections_coverage_ge_0_8"],
            "positive_detection_best_coverage_range": _range(detection_best_coverages),
            "positive_detection_best_iou_range": _range(detection_best_ious),
            "parseq_crop_denominator": crop_count,
            "crops_with_positive_overlap": counters["crops_positive_overlap"],
            "crops_with_positive_overlap_fraction": _fraction(
                counters["crops_positive_overlap"], crop_count,
                available=fractions_available,
            ),
            "crops_best_coverage_ge_0_5": counters["crops_coverage_ge_0_5"],
            "crops_best_coverage_ge_0_8": counters["crops_coverage_ge_0_8"],
            "positive_crop_best_coverage_range": _range(crop_best_coverages),
        },
        "text_diagnostics": {
            "spatially_best_positive_pairs": {
                "denominator": positive_pair_count,
                "raw_exact_count": counters["spatial_pair_raw_exact"],
                "casefold_exact_count": counters["spatial_pair_casefold_exact"],
                "casefold_ratio_ge_0_8_count": counters[
                    "spatial_pair_casefold_ratio_ge_0_8"
                ],
                "casefold_ratio_range": _range(spatial_pair_casefold_ratios),
            },
            "deepsolo_detections_casefold_exact_with_any_parseq_crop": counters[
                "detections_any_parseq_casefold_exact"
            ],
            "T1_vs_any_deepsolo_detection": {
                "sample_denominator": fallback_samples,
                "raw_exact_count": counters["t1_any_raw_exact"],
                "casefold_exact_count": counters["t1_any_casefold_exact"],
            },
            "selected_T3_vs_any_deepsolo_detection": {
                "sample_denominator": fallback_samples,
                "raw_exact_count": counters["selected_t3_any_raw_exact"],
                "casefold_exact_count": counters["selected_t3_any_casefold_exact"],
            },
        },
        "per_sample": per_sample,
    }


__all__ = [
    "ParseqDeepSoloAlignmentError",
    "diagnose_parseq_deepsolo_alignment",
]
