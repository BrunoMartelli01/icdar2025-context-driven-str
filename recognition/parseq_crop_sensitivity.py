"""GT-bridged PARSeq crop sensitivity diagnostics for APPROXIMATE traces.

The paper fixes component-area filtering/sorting, padded source-image crops, and
``nmax=10`` but not the numeric area/padding values or connectivity.  This module
reuses a frozen segmentation mask, varies only those unresolved inputs, runs the
qualified approximate PARSeq candidate, and reports geometry-first associations
to ICDAR13 Task 1 word boxes.  It deliberately does not define recognition
accuracy or select a scientifically preferred parameter variant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from recognition.icdar13_annotations import ICDAR13RecognitionSample
from recognition.icdar13_trace_diagnostics import diagnose_trace_index


BASELINE_MIN_AREA = 150
BASELINE_PADDING = 6
BASELINE_CONNECTIVITY = 8


class ParseqCropSensitivityError(ValueError):
    """Raised when source traces or generated crop records fail closed."""


@dataclass(frozen=True, order=True)
class LocalizationVariant:
    """One explicitly named localization sensitivity point."""

    min_area: int
    padding: int
    connectivity: int

    def __post_init__(self) -> None:
        # Reuse the production validation rather than maintaining a second contract.
        BlockParam(
            min_area=self.min_area,
            pad=self.padding,
            connectivity=self.connectivity,
            tie_break=APPROXIMATE_TIE_BREAK,
            nmax=10,
        )

    @property
    def variant_id(self) -> str:
        return "amin-{}__pad-{}__conn-{}".format(
            self.min_area, self.padding, self.connectivity
        )

    @property
    def is_baseline(self) -> bool:
        return (
            self.min_area == BASELINE_MIN_AREA
            and self.padding == BASELINE_PADDING
            and self.connectivity == BASELINE_CONNECTIVITY
        )


def build_variant_grid(
    min_areas: Sequence[int], paddings: Sequence[int], connectivities: Sequence[int]
) -> tuple[LocalizationVariant, ...]:
    """Build a deterministic Cartesian grid and require the frozen baseline."""

    variants = tuple(
        LocalizationVariant(int(min_area), int(padding), int(connectivity))
        for min_area in min_areas
        for padding in paddings
        for connectivity in connectivities
    )
    if not variants:
        raise ParseqCropSensitivityError("The localization variant grid is empty")
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise ParseqCropSensitivityError("The localization variant grid contains duplicates")
    if not any(variant.is_baseline for variant in variants):
        raise ParseqCropSensitivityError(
            "The grid must include baseline min_area=150, padding=6, connectivity=8"
        )
    return variants


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pixel_sha256(value: np.ndarray) -> str:
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _exclusive_gt_box(
    value: tuple[int, int, int, int], image_size_wh: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert acquired ICDAR13 inclusive maxima to half-open coordinates."""

    x1, y1, x2, y2 = value
    width, height = image_size_wh
    converted = (float(x1), float(y1), float(x2 + 1), float(y2 + 1))
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ParseqCropSensitivityError("Ground-truth box is outside the source image")
    return converted


def _overlap(
    crop_box: tuple[float, ...], gt_box: tuple[float, ...]
) -> dict[str, float | bool]:
    cx1, cy1, cx2, cy2 = crop_box
    gx1, gy1, gx2, gy2 = gt_box
    intersection = max(0.0, min(cx2, gx2) - max(cx1, gx1)) * max(
        0.0, min(cy2, gy2) - max(cy1, gy1)
    )
    crop_area = (cx2 - cx1) * (cy2 - cy1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = crop_area + gt_area - intersection
    center_x = (gx1 + gx2) / 2.0
    center_y = (gy1 + gy2) / 2.0
    return {
        "intersection_area": intersection,
        "bbox_iou": intersection / union,
        "crop_coverage": intersection / crop_area,
        "gt_coverage": intersection / gt_area,
        "gt_center_inside_crop": cx1 <= center_x < cx2 and cy1 <= center_y < cy2,
    }


def _text_metrics(prediction: str, ground_truth: str) -> dict[str, Any]:
    both_nonempty = bool(prediction) and bool(ground_truth)
    return {
        "case_sensitive_exact": both_nonempty and prediction == ground_truth,
        "casefold_exact": (
            both_nonempty and prediction.casefold() == ground_truth.casefold()
        ),
        "rapidfuzz_ratio_raw": ratio(prediction, ground_truth) / 100.0,
        "rapidfuzz_ratio_casefold": (
            ratio(prediction.casefold(), ground_truth.casefold()) / 100.0
        ),
    }


def _best_positive_pair(
    pairs: Sequence[dict[str, Any]], *, final_tie_key: str
) -> dict[str, Any] | None:
    if not pairs:
        return None
    best = max(
        pairs,
        key=lambda item: (
            float(item["bbox_iou"]),
            float(item["gt_coverage"]),
            float(item["crop_coverage"]),
            -int(item[final_tie_key]),
        ),
    )
    return best if float(best["intersection_area"]) > 0 else None


def _range(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _load_rgb(path: Path, expected_size_wh: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"))
    if value.dtype != np.uint8 or value.shape[:2] != (
        expected_size_wh[1], expected_size_wh[0]
    ):
        raise ParseqCropSensitivityError("Decoded source image geometry changed")
    return value


def _load_binary_mask(path: Path, expected_shape_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"))
    if value.shape != expected_shape_hw:
        raise ParseqCropSensitivityError("Recorded segmentation mask geometry changed")
    unique = set(np.unique(value).tolist())
    if not unique.issubset({0, 255}):
        raise ParseqCropSensitivityError(
            "Recorded segmentation mask is not a lossless 0/255 binary artifact"
        )
    return (value > 0).astype(np.uint8)


def _prediction_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
    elif isinstance(value, dict):
        result = dict(value)
    else:
        raise ParseqCropSensitivityError("PARSeq recognizer returned an invalid record")
    if not isinstance(result.get("raw_text"), str):
        raise ParseqCropSensitivityError("PARSeq prediction is missing raw_text")
    # Crop-level output retains the decoded sequence evidence. Runtime identity and
    # preprocessing are frozen once at report level instead of being duplicated for
    # every variant reference to the same crop.
    return {
        key: result.get(key)
        for key in (
            "raw_text",
            "token_ids",
            "token_confidences",
            "confidence",
            "confidence_aggregation",
            "input_role",
        )
    }


def _variant_sample(
    *,
    sample: ICDAR13RecognitionSample,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    variant: LocalizationVariant,
    recognize_crop: Callable[[np.ndarray], Any],
    prediction_cache: dict[tuple[str, tuple[int, ...]], dict[str, Any]],
) -> dict[str, Any]:
    params = BlockParam(
        min_area=variant.min_area,
        pad=variant.padding,
        connectivity=variant.connectivity,
        tie_break=APPROXIMATE_TIE_BREAK,
        nmax=10,
    )
    localization = localize_mask(mask, image_rgb, params)
    gt_records = [
        {
            "gt_index": index,
            "bbox_xyxy_inclusive_acquired": list(word.bbox_xyxy),
            "bbox_xyxy_exclusive_diagnostic": list(
                _exclusive_gt_box(word.bbox_xyxy, sample.image_size_wh)
            ),
            "transcription": word.transcription,
        }
        for index, word in enumerate(sample.words, start=1)
    ]

    crop_records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for block in localization.retained:
        cache_key = (sample.sample_id, tuple(block.padded_bbox))
        cache_hit = cache_key in prediction_cache
        if not cache_hit:
            prediction_cache[cache_key] = _prediction_dict(recognize_crop(block.crop))
        prediction = prediction_cache[cache_key]
        crop_record = {
            "rank": block.rank,
            "component_id": block.component_id,
            "component_area": block.component_area,
            "bbox_xyxy_exclusive": list(block.bbox),
            "padded_bbox_xyxy_exclusive": list(block.padded_bbox),
            "crop_shape_hwc": list(block.crop.shape),
            "crop_pixel_sha256": _pixel_sha256(block.crop),
            "parseq": prediction,
            "prediction_cache_hit": cache_hit,
        }
        crop_records.append(crop_record)
        crop_box = tuple(float(value) for value in block.padded_bbox)
        for gt in gt_records:
            metrics = _overlap(
                crop_box, tuple(gt["bbox_xyxy_exclusive_diagnostic"])
            )
            pairs.append(
                {
                    "crop_rank": block.rank,
                    "gt_index": gt["gt_index"],
                    "parseq_text": prediction["raw_text"],
                    "gt_text": gt["transcription"],
                    **metrics,
                    **_text_metrics(prediction["raw_text"], gt["transcription"]),
                }
            )

    crop_summaries: list[dict[str, Any]] = []
    for crop in crop_records:
        crop_pairs = [pair for pair in pairs if pair["crop_rank"] == crop["rank"]]
        best = _best_positive_pair(crop_pairs, final_tie_key="gt_index")
        crop_summaries.append(
            {
                "rank": crop["rank"],
                "parseq_text": crop["parseq"]["raw_text"],
                "best_spatial_gt": dict(best) if best is not None else None,
                "casefold_exact_with_any_overlapping_gt": any(
                    pair["intersection_area"] > 0 and pair["casefold_exact"]
                    for pair in crop_pairs
                ),
            }
        )

    gt_summaries: list[dict[str, Any]] = []
    for gt in gt_records:
        gt_pairs = [pair for pair in pairs if pair["gt_index"] == gt["gt_index"]]
        best = _best_positive_pair(gt_pairs, final_tie_key="crop_rank")
        gt_summaries.append(
            {
                **gt,
                "best_spatial_crop": dict(best) if best is not None else None,
                "center_inside_any_crop": any(
                    bool(pair["gt_center_inside_crop"]) for pair in gt_pairs
                ),
                "casefold_exact_with_any_overlapping_crop": any(
                    pair["intersection_area"] > 0 and pair["casefold_exact"]
                    for pair in gt_pairs
                ),
                "casefold_ratio_ge_0_8_with_any_overlapping_crop": any(
                    pair["intersection_area"] > 0
                    and pair["rapidfuzz_ratio_casefold"] >= 0.8
                    for pair in gt_pairs
                ),
                "casefold_exact_with_any_crop_regardless_of_geometry": any(
                    pair["casefold_exact"] for pair in gt_pairs
                ),
            }
        )

    return {
        "sample_id": sample.sample_id,
        "ground_truth_word_count": len(gt_records),
        "localization": localization.as_dict(),
        "crops": crop_records,
        "crop_spatial_summaries": crop_summaries,
        "ground_truth_spatial_summaries": gt_summaries,
        "pairwise_crop_gt": pairs,
    }


def _aggregate_variant(
    variant: LocalizationVariant, samples: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    gt_summaries = [
        gt for sample in samples for gt in sample["ground_truth_spatial_summaries"]
    ]
    crop_summaries = [
        crop for sample in samples for crop in sample["crop_spatial_summaries"]
    ]
    positive_best = [
        gt["best_spatial_crop"]
        for gt in gt_summaries
        if gt["best_spatial_crop"] is not None
    ]
    iou_half = [pair for pair in positive_best if pair["bbox_iou"] >= 0.5]
    return {
        "variant_id": variant.variant_id,
        "parameters": asdict(variant),
        "is_frozen_trace_baseline": variant.is_baseline,
        "population": {
            "sample_count": len(samples),
            "ground_truth_word_denominator": len(gt_summaries),
            "parseq_crop_count": len(crop_summaries),
        },
        "spatial_gt_diagnostics": {
            "gt_words_with_positive_crop_overlap": len(positive_best),
            "gt_words_with_center_inside_any_crop": sum(
                bool(gt["center_inside_any_crop"]) for gt in gt_summaries
            ),
            "gt_best_coverage_ge_0_5": sum(
                pair["gt_coverage"] >= 0.5 for pair in positive_best
            ),
            "gt_best_coverage_ge_0_8": sum(
                pair["gt_coverage"] >= 0.8 for pair in positive_best
            ),
            "gt_best_iou_ge_0_25": sum(
                pair["bbox_iou"] >= 0.25 for pair in positive_best
            ),
            "gt_best_iou_ge_0_5": len(iou_half),
            "positive_best_gt_coverage_range": _range(
                [float(pair["gt_coverage"]) for pair in positive_best]
            ),
            "positive_best_iou_range": _range(
                [float(pair["bbox_iou"]) for pair in positive_best]
            ),
            "crops_with_positive_gt_overlap": sum(
                crop["best_spatial_gt"] is not None for crop in crop_summaries
            ),
        },
        "text_on_geometry_first_gt_pairs": {
            "positive_overlap_denominator": len(positive_best),
            "case_sensitive_exact_count": sum(
                bool(pair["case_sensitive_exact"]) for pair in positive_best
            ),
            "casefold_exact_count": sum(
                bool(pair["casefold_exact"]) for pair in positive_best
            ),
            "casefold_ratio_ge_0_8_count": sum(
                pair["rapidfuzz_ratio_casefold"] >= 0.8 for pair in positive_best
            ),
            "casefold_ratio_range": _range(
                [float(pair["rapidfuzz_ratio_casefold"]) for pair in positive_best]
            ),
            "iou_ge_0_5_denominator": len(iou_half),
            "iou_ge_0_5_casefold_exact_count": sum(
                bool(pair["casefold_exact"]) for pair in iou_half
            ),
            "iou_ge_0_5_casefold_ratio_ge_0_8_count": sum(
                pair["rapidfuzz_ratio_casefold"] >= 0.8 for pair in iou_half
            ),
        },
        "optimistic_named_text_diagnostics": {
            "gt_casefold_exact_with_any_overlapping_crop": sum(
                bool(gt["casefold_exact_with_any_overlapping_crop"])
                for gt in gt_summaries
            ),
            "gt_casefold_ratio_ge_0_8_with_any_overlapping_crop": sum(
                bool(gt["casefold_ratio_ge_0_8_with_any_overlapping_crop"])
                for gt in gt_summaries
            ),
            "gt_casefold_exact_with_any_crop_regardless_of_geometry": sum(
                bool(gt["casefold_exact_with_any_crop_regardless_of_geometry"])
                for gt in gt_summaries
            ),
        },
        "per_sample": list(samples),
    }


def _baseline_source_records(index_path: Path) -> dict[str, dict[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for record in index["samples"]:
        if not record["success"]:
            raise ParseqCropSensitivityError(
                "Source traces must all succeed before parameter aggregation"
            )
        result[str(record["sample_id"])] = json.loads(
            Path(record["result"]).read_text(encoding="utf-8")
        )
    return result


def _verify_baseline(
    generated: Sequence[dict[str, Any]], source: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    verified = 0
    for sample_result in generated:
        sample_id = sample_result["sample_id"]
        source_result = source[sample_id]
        source_blocks = source_result["segmentation"]["localized_blocks"]
        source_predictions = source_result["T3"]["all"]
        generated_crops = sample_result["crops"]
        if len(source_blocks) != len(source_predictions) or len(source_blocks) != len(
            generated_crops
        ):
            raise ParseqCropSensitivityError(
                "Baseline crop cardinality drifted for sample {}".format(sample_id)
            )
        for block, old_prediction, new_crop in zip(
            source_blocks, source_predictions, generated_crops
        ):
            if list(block["padded_bbox"]) != new_crop["padded_bbox_xyxy_exclusive"]:
                raise ParseqCropSensitivityError(
                    "Baseline crop box drifted for sample {} rank {}".format(
                        sample_id, new_crop["rank"]
                    )
                )
            old_crop_path = Path(block["crop_artifact"]["path"])
            with Image.open(old_crop_path) as image:
                old_crop = np.asarray(image.convert("RGB"))
            if _pixel_sha256(old_crop) != new_crop["crop_pixel_sha256"]:
                raise ParseqCropSensitivityError(
                    "Baseline crop pixels drifted for sample {} rank {}".format(
                        sample_id, new_crop["rank"]
                    )
                )
            if old_prediction["raw_text"] != new_crop["parseq"]["raw_text"]:
                raise ParseqCropSensitivityError(
                    "Baseline PARSeq text drifted for sample {} rank {}: {!r} != {!r}".format(
                        sample_id,
                        new_crop["rank"],
                        old_prediction["raw_text"],
                        new_crop["parseq"]["raw_text"],
                    )
                )
            verified += 1
    return {
        "status": "passed",
        "verified_crop_box_count": verified,
        "verified_crop_pixel_count": verified,
        "verified_parseq_text_count": verified,
    }


def diagnose_parseq_crop_sensitivity(
    *,
    index_path: Path,
    samples: Iterable[ICDAR13RecognitionSample],
    variants: Sequence[LocalizationVariant],
    recognize_crop: Callable[[np.ndarray], Any],
    parseq_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Run a predeclared localization grid without producing an accuracy claim."""

    samples = tuple(samples)
    variants = tuple(variants)
    if not variants or not any(variant.is_baseline for variant in variants):
        raise ParseqCropSensitivityError("A frozen baseline variant is required")
    base = diagnose_trace_index(index_path, samples)
    if base["population"]["failed_sample_count"] != 0:
        raise ParseqCropSensitivityError(
            "Source traces contain failures; refusing parameter aggregation"
        )
    index_path = index_path.resolve()
    source = _baseline_source_records(index_path)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    selected_samples = [sample_by_id[str(sample_id)] for sample_id in source]

    prediction_cache: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    variant_reports: list[dict[str, Any]] = []
    baseline_generated: list[dict[str, Any]] | None = None
    started = time.perf_counter()
    for variant in variants:
        generated: list[dict[str, Any]] = []
        for sample in selected_samples:
            source_result = source[sample.sample_id]
            image_rgb = _load_rgb(sample.image_path, sample.image_size_wh)
            mask_path = Path(
                source_result["segmentation"]["mask_artifact"]["path"]
            )
            mask = _load_binary_mask(mask_path, image_rgb.shape[:2])
            generated.append(
                _variant_sample(
                    sample=sample,
                    image_rgb=image_rgb,
                    mask=mask,
                    variant=variant,
                    recognize_crop=recognize_crop,
                    prediction_cache=prediction_cache,
                )
            )
        if variant.is_baseline:
            baseline_generated = generated
        variant_reports.append(_aggregate_variant(variant, generated))
    if baseline_generated is None:
        raise ParseqCropSensitivityError("Frozen baseline was not evaluated")
    baseline_verification = _verify_baseline(baseline_generated, source)
    source_parseq_placements = sorted(
        {
            str(event["placement"])
            for result in source.values()
            for event in result.get("lifecycle", [])
            if event.get("event") == "stage_completed"
            and event.get("stage") == "parseq"
        }
    )
    baseline_verification["source_trace_parseq_placements"] = source_parseq_placements
    baseline_verification["diagnostic_parseq_device"] = parseq_provenance.get("device")
    baseline_verification["cross_device_scope"] = (
        "All frozen-baseline crop boxes, pixels, and decoded texts were reproduced; "
        "new sensitivity crops were not independently repeated on the source GPU."
    )

    def ordering(metric_path: tuple[str, ...]) -> list[dict[str, Any]]:
        def metric(report: dict[str, Any]) -> int:
            value: Any = report
            for key in metric_path:
                value = value[key]
            return int(value)

        return [
            {"variant_id": report["variant_id"], "count": metric(report)}
            for report in sorted(
                variant_reports,
                key=lambda report: (-metric(report), report["variant_id"]),
            )
        ]

    index = json.loads(index_path.read_text(encoding="utf-8"))
    contract = index["batch_contract"]
    return {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "PARSeq crop/GT sensitivity diagnostics; not recognition accuracy",
        "accuracy": None,
        "evaluation_status": "UNRESOLVED",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "source": {
            "index_path": str(index_path),
            "index_sha256": _sha256_file(index_path),
            "batch_contract_path": contract["path"],
            "batch_contract_sha256": contract["sha256"],
            "recorded_segmentation_masks_reused": True,
        },
        "parseq": parseq_provenance,
        "grid": {
            "variant_count": len(variants),
            "min_areas": sorted({variant.min_area for variant in variants}),
            "paddings": sorted({variant.padding for variant in variants}),
            "connectivities": sorted({variant.connectivity for variant in variants}),
            "classification": "ASSUMPTION diagnostic grid fixed before inference",
        },
        "scientific_boundary": {
            "paper_fixed": [
                "true foreground-pixel component area",
                "descending component-area ordering",
                "padding and source-bound clamping",
                "original-image RGB crop",
                "nmax=10",
            ],
            "varied": {
                "min_area": "REPOSITORY_SPECIFIED baseline plus ASSUMPTION sensitivity points",
                "padding": "REPOSITORY_SPECIFIED baseline plus ASSUMPTION sensitivity points",
                "connectivity": "ASSUMPTION sensitivity points",
            },
            "held_fixed": [
                "recorded segmentation mask",
                "exclusive xyxy crop convention",
                "top-to-bottom then left-to-right equal-area tie rule",
                "qualified approximate PARSeq checkpoint/runtime and decoding",
            ],
            "hardware_deviation": (
                "Diagnostic PARSeq inference ran on {}; the source traces used {}. "
                "The baseline equivalence gate passed, but new variant crops have no "
                "independent cross-device replication."
            ).format(
                parseq_provenance.get("device"),
                ", ".join(source_parseq_placements) or "an unrecorded placement",
            ),
            "hardware_deviation_classification": "HARDWARE_DEVIATION",
            "gt_matching": (
                "ASSUMPTION: acquired inclusive GT maxima converted to exclusive by +1; "
                "geometry-first best pair maximizes bbox IoU, then GT coverage, crop "
                "coverage, then lowest source rank/index; text never selects the pair"
            ),
            "prohibition": (
                "Variant diagnostics and their orderings must not be called accuracy or "
                "used to tune a Table 5 result."
            ),
            "blocking_ambiguity_ids": [
                "A-008", "L-001", "L-003", "L-005", "R-001", "R-003", "E-001"
            ],
        },
        "baseline_reproduction": baseline_verification,
        "runtime": {
            "unique_crop_inference_count": len(prediction_cache),
            "total_variant_crop_references": sum(
                report["population"]["parseq_crop_count"] for report in variant_reports
            ),
            "prediction_cache_reuse_count": sum(
                report["population"]["parseq_crop_count"] for report in variant_reports
            )
            - len(prediction_cache),
            "wall_seconds": time.perf_counter() - started,
        },
        "diagnostic_orderings": {
            "by_geometry_first_casefold_exact_count": ordering(
                ("text_on_geometry_first_gt_pairs", "casefold_exact_count")
            ),
            "by_geometry_first_casefold_ratio_ge_0_8_count": ordering(
                ("text_on_geometry_first_gt_pairs", "casefold_ratio_ge_0_8_count")
            ),
            "by_gt_best_iou_ge_0_5_count": ordering(
                ("spatial_gt_diagnostics", "gt_best_iou_ge_0_5")
            ),
            "by_gt_positive_overlap_count": ordering(
                ("spatial_gt_diagnostics", "gt_words_with_positive_crop_overlap")
            ),
        },
        "variants": variant_reports,
    }


__all__ = [
    "BASELINE_CONNECTIVITY",
    "BASELINE_MIN_AREA",
    "BASELINE_PADDING",
    "LocalizationVariant",
    "ParseqCropSensitivityError",
    "build_variant_grid",
    "diagnose_parseq_crop_sensitivity",
]
