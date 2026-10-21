"""Native target/prediction component topology diagnostics.

This module is deliberately read-only with respect to the frozen Table-5 trace.
It compares the original ICDAR13 Task 2.2 target, its author-specified native
128x128 target conversion, and the recorded native prediction for a fixed first
ten sample diagnostic.  Ground truth is used solely for measurement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

from pipeline.frozen_segmentation import FROZEN_LOGITS_SHAPE
from recognition.icdar13_annotations import ICDAR13RecognitionSample
from recognition.segmentation_word_topology import (
    EXPECTED_SEGMENTATION_CHECKPOINT_SHA256,
    SegmentationWordTopologyError,
    recover_native_mask_from_projected,
)
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest
from segmentation.evaluate import resize_binary_target_to_native_logits


COMPONENT_CONNECTIVITY = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_projected_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"))
    if values.shape != expected_shape or not set(np.unique(values).tolist()).issubset({0, 255}):
        raise SegmentationWordTopologyError("Projected prediction must be a binary HxW artifact")
    return (values > 0).astype(np.uint8)


def _binary_target(path: Path) -> tuple[Image.Image, np.ndarray]:
    with Image.open(path) as image:
        binary = icdar13_task22_rgb_to_binary_mask(image)
    values = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8)
    return binary, values


def _components(mask: np.ndarray) -> dict[str, Any]:
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise SegmentationWordTopologyError("Component masks must be uint8 HxW")
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, COMPONENT_CONNECTIVITY)
    records: dict[int, dict[str, Any]] = {}
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        records[component_id] = {
            "component_id": component_id,
            "area": int(stats[component_id, cv2.CC_STAT_AREA]),
            "bbox_xyxy_exclusive": [x, y, x + width, y + height],
            "centroid": [float(centroids[component_id][0]), float(centroids[component_id][1])],
        }
    return {"labels": labels, "records": records}


def _exclusive_box(
    box_inclusive: tuple[int, int, int, int], image_size_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_inclusive
    width, height = image_size_wh
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise SegmentationWordTopologyError("Recognition GT box is outside its image")
    return x1, y1, x2 + 1, y2 + 1


def map_original_box_to_mask(
    box: tuple[int, int, int, int],
    *,
    image_size_wh: tuple[int, int],
    mask_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a half-open image box to a nonempty half-open mask box.

    Floor lower / ceil upper is an explicitly diagnostic-only coverage-preserving
    convention. It cannot affect any pipeline inference or reported metric.
    """

    x1, y1, x2, y2 = box
    width, height = image_size_wh
    mask_h, mask_w = mask_shape
    mx1 = max(0, min(mask_w - 1, math.floor(x1 * mask_w / width)))
    my1 = max(0, min(mask_h - 1, math.floor(y1 * mask_h / height)))
    mx2 = max(mx1 + 1, min(mask_w, math.ceil(x2 * mask_w / width)))
    my2 = max(my1 + 1, min(mask_h, math.ceil(y2 * mask_h / height)))
    return mx1, my1, mx2, my2


def _region(mask_shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    region = np.zeros(mask_shape, dtype=bool)
    region[y1:y2, x1:x2] = True
    return region


def _ids_in_region(labels: np.ndarray, region: np.ndarray) -> list[int]:
    return sorted(int(value) for value in np.unique(labels[region]) if value)


def _union_bbox_coverage(
    components: dict[str, Any], ids: list[int], region: np.ndarray
) -> float:
    """Area of the union of raw component bounding rectangles within the word ROI."""

    coverage = np.zeros_like(region, dtype=bool)
    for component_id in ids:
        x1, y1, x2, y2 = components["records"][component_id]["bbox_xyxy_exclusive"]
        coverage[y1:y2, x1:x2] = True
    return float((coverage & region).sum() / max(1, int(region.sum())))


def level_word_metrics(
    *,
    mask: np.ndarray,
    components: dict[str, Any],
    native_or_source_box: tuple[int, int, int, int],
    transcription: str,
) -> tuple[dict[str, Any], list[int], np.ndarray]:
    """Measure component topology inside one GT-derived region."""

    region = _region(mask.shape, native_or_source_box)
    labels = components["labels"]
    ids = _ids_in_region(labels, region)
    foreground = int((mask & region).sum())
    centers = []
    x1, y1, x2, y2 = native_or_source_box
    for component_id in ids:
        cx, cy = components["records"][component_id]["centroid"]
        if x1 <= cx < x2 and y1 <= cy < y2:
            centers.append(component_id)
    largest = max(
        (int(((labels == component_id) & region).sum()) for component_id in ids),
        default=0,
    )
    character_count = len(transcription)
    return (
        {
            "foreground_pixels": foreground,
            "foreground_coverage_of_word_region": foreground / max(1, int(region.sum())),
            "component_count_intersecting_word": len(ids),
            "component_count_with_center_inside_word": len(centers),
            "largest_component_to_total_foreground_ratio": largest / foreground if foreground else None,
            "union_component_bbox_coverage_of_word_region": _union_bbox_coverage(components, ids, region),
            "component_count_per_transcription_character": len(ids) / character_count if character_count else None,
        },
        ids,
        region,
    )


def _best_match(
    source_mask: np.ndarray,
    candidate_ids: list[int],
    candidate_labels: np.ndarray,
    region: np.ndarray,
) -> tuple[int | None, int, float, float]:
    source_pixels = int(source_mask.sum())
    best_id: int | None = None
    best_intersection = 0
    best_iou = 0.0
    best_fraction = 0.0
    for candidate_id in candidate_ids:
        candidate_mask = (candidate_labels == candidate_id) & region
        candidate_pixels = int(candidate_mask.sum())
        intersection = int((source_mask & candidate_mask).sum())
        union = source_pixels + candidate_pixels - intersection
        iou = intersection / union if union else 0.0
        fraction = intersection / source_pixels if source_pixels else 0.0
        if (intersection, iou, -candidate_id) > (best_intersection, best_iou, -(best_id or 10**9)):
            best_id, best_intersection, best_iou, best_fraction = (
                candidate_id,
                intersection,
                iou,
                fraction,
            )
    return best_id, best_intersection, best_iou, best_fraction


def match_target_prediction_components(
    *,
    target_components: dict[str, Any],
    prediction_components: dict[str, Any],
    target_ids: list[int],
    prediction_ids: list[int],
    region: np.ndarray,
) -> dict[str, Any]:
    """Pixel-geometric target/prediction matching inside one word region."""

    target_labels = target_components["labels"]
    prediction_labels = prediction_components["labels"]
    x1, y1, x2, y2 = np.where(region)[1].min(), np.where(region)[0].min(), np.where(region)[1].max() + 1, np.where(region)[0].max() + 1
    diagonal = math.hypot(x2 - x1, y2 - y1)
    prediction_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pred_best: dict[int, int | None] = {}
    target_best: dict[int, int | None] = {}

    for prediction_id in prediction_ids:
        prediction_mask = (prediction_labels == prediction_id) & region
        best_id, overlap, iou, fraction = _best_match(
            prediction_mask, target_ids, target_labels, region
        )
        pixels_in_target_foreground = int(
            (prediction_mask & (target_labels > 0) & region).sum()
        )
        pred_best[prediction_id] = best_id
        if best_id is None:
            distance = None
        else:
            px, py = prediction_components["records"][prediction_id]["centroid"]
            tx, ty = target_components["records"][best_id]["centroid"]
            distance = math.hypot(px - tx, py - ty) / diagonal if diagonal else 0.0
        prediction_rows.append(
            {
                "predicted_component_id": prediction_id,
                "pixels_inside_word": int(prediction_mask.sum()),
                "pixels_inside_target_foreground": pixels_in_target_foreground,
                "fraction_pixels_inside_target_foreground": (
                    pixels_in_target_foreground / int(prediction_mask.sum())
                    if int(prediction_mask.sum()) else 0.0
                ),
                "best_target_component_id": best_id,
                "best_target_component_overlap_pixels": overlap,
                "best_target_component_overlap_fraction_of_predicted": fraction,
                "best_target_component_iou": iou,
                "centroid_distance_normalized_to_word_diagonal": distance,
            }
        )

    for target_id in target_ids:
        target_mask = (target_labels == target_id) & region
        best_id, overlap, iou, fraction = _best_match(
            target_mask, prediction_ids, prediction_labels, region
        )
        target_best[target_id] = best_id
        target_rows.append(
            {
                "target_component_id": target_id,
                "pixels_inside_word": int(target_mask.sum()),
                "best_predicted_component_id": best_id,
                "best_predicted_component_overlap_pixels": overlap,
                "best_predicted_component_iou": iou,
                "fraction_target_component_covered_by_best_prediction": fraction,
            }
        )

    target_to_predictions = {
        target_id: [
            prediction_id
            for prediction_id in prediction_ids
            if int(((target_labels == target_id) & (prediction_labels == prediction_id) & region).sum()) > 0
        ]
        for target_id in target_ids
    }
    prediction_to_targets = {
        prediction_id: [
            target_id
            for target_id in target_ids
            if int(((target_labels == target_id) & (prediction_labels == prediction_id) & region).sum()) > 0
        ]
        for prediction_id in prediction_ids
    }
    mutual_pairs = sorted(
        (prediction_id, target_id)
        for prediction_id, target_id in pred_best.items()
        if target_id is not None and target_best.get(target_id) == prediction_id
        and int(((prediction_labels == prediction_id) & (target_labels == target_id) & region).sum()) > 0
    )
    return {
        "predicted_components": prediction_rows,
        "target_components": target_rows,
        "one_to_one_plausible_matches": [
            {"predicted_component_id": prediction_id, "target_component_id": target_id}
            for prediction_id, target_id in mutual_pairs
        ],
        "split_target_components": [
            {"target_component_id": target_id, "overlapping_predicted_component_ids": ids}
            for target_id, ids in target_to_predictions.items() if len(ids) >= 2
        ],
        "merge_predicted_components": [
            {"predicted_component_id": prediction_id, "overlapping_target_component_ids": ids}
            for prediction_id, ids in prediction_to_targets.items() if len(ids) >= 2
        ],
    }


def _component_distribution(words: list[dict[str, Any]], level: str) -> dict[str, Any]:
    counts = [word["levels"][level]["component_count_intersecting_word"] for word in words]
    char_lengths = [len(word["gt_text"]) for word in words]
    correlation = None
    if len(counts) > 1 and np.std(counts) > 0 and np.std(char_lengths) > 0:
        correlation = float(np.corrcoef(counts, char_lengths)[0, 1])
    return {
        "words_with_0_components": sum(count == 0 for count in counts),
        "words_with_1_component": sum(count == 1 for count in counts),
        "words_with_2_components": sum(count == 2 for count in counts),
        "words_with_3_or_more_components": sum(count >= 3 for count in counts),
        "median_component_count": float(np.median(counts)),
        "pearson_component_count_vs_transcription_length": correlation,
    }


def _matching_aggregate(words: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_rows = [
        row for word in words for row in word["target_prediction_matching"]["predicted_components"]
    ]
    target_rows = [
        row for word in words for row in word["target_prediction_matching"]["target_components"]
    ]
    return {
        "predicted_component_instances": len(prediction_rows),
        "predicted_components_fraction_on_target_foreground_ge_0_8": sum(
            row["fraction_pixels_inside_target_foreground"] >= 0.8 for row in prediction_rows
        ),
        "predicted_components_fraction_on_target_foreground_ge_0_5": sum(
            row["fraction_pixels_inside_target_foreground"] >= 0.5 for row in prediction_rows
        ),
        "target_component_instances": len(target_rows),
        "target_components_covered_by_best_prediction_ge_0_8": sum(
            row["fraction_target_component_covered_by_best_prediction"] >= 0.8 for row in target_rows
        ),
        "target_components_covered_by_best_prediction_ge_0_5": sum(
            row["fraction_target_component_covered_by_best_prediction"] >= 0.5 for row in target_rows
        ),
        "mutual_best_nonzero_pixel_one_to_one_matches": sum(
            len(word["target_prediction_matching"]["one_to_one_plausible_matches"])
            for word in words
        ),
        "target_component_instances_with_split": sum(
            len(word["target_prediction_matching"]["split_target_components"])
            for word in words
        ),
        "predicted_component_instances_with_merge": sum(
            len(word["target_prediction_matching"]["merge_predicted_components"])
            for word in words
        ),
    }


def diagnose_target_component_topology(
    *,
    index_path: Path,
    samples: Iterable[ICDAR13RecognitionSample],
    segmentation_manifest_path: Path,
    data_root: Path,
    segmentation_checkpoint: Path,
) -> dict[str, Any]:
    """Return the fixed-population source/target/prediction topology diagnostic."""

    samples = tuple(samples)
    if len(samples) != 10:
        raise SegmentationWordTopologyError("This diagnostic is fixed to exactly 10 samples")
    checkpoint = segmentation_checkpoint.resolve()
    checkpoint_sha = _sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_SEGMENTATION_CHECKPOINT_SHA256:
        raise SegmentationWordTopologyError("Segmentation checkpoint SHA256 mismatch")
    manifest = load_segmentation_manifest(segmentation_manifest_path)
    seg_by_id = {sample.sample_id: sample for sample in manifest.samples}
    sample_by_id = {sample.sample_id: sample for sample in samples}
    index = json.loads(index_path.resolve().read_text(encoding="utf-8"))
    records = [entry for entry in index["samples"] if str(entry["sample_id"]) in sample_by_id]
    if len(records) != 10:
        raise SegmentationWordTopologyError("Trace index does not contain all first-ten samples")

    words: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    native_hash_verifications = 0
    for record in records:
        sample_id = str(record["sample_id"])
        sample = sample_by_id[sample_id]
        seg_sample = seg_by_id.get(sample_id)
        if seg_sample is None:
            raise SegmentationWordTopologyError("Segmentation test manifest lacks a trace sample")
        result = json.loads(Path(record["result"]).read_text(encoding="utf-8"))
        inference = result["segmentation"]["inference"]
        if inference["checkpoint"]["sha256"] != checkpoint_sha:
            raise SegmentationWordTopologyError("Frozen trace checkpoint SHA mismatch")
        target_path = (data_root.resolve() / Path(seg_sample.mask_path)).resolve()
        if _sha256_file(target_path) != seg_sample.mask_sha256:
            raise SegmentationWordTopologyError("Segmentation source target SHA mismatch")
        source_target_image, source_target = _binary_target(target_path)
        target_128 = resize_binary_target_to_native_logits(
            source_target_image, size=FROZEN_LOGITS_SHAPE, threshold=0.5
        ).numpy().astype(np.uint8)
        projected = _load_projected_mask(
            Path(result["segmentation"]["mask_artifact"]["path"]), source_target.shape
        )
        prediction_128 = recover_native_mask_from_projected(projected)
        native_sha = _sha256_bytes(prediction_128.tobytes())
        if native_sha != inference["native_mask_sha256"]:
            raise SegmentationWordTopologyError("Recovered native mask differs from frozen adapter output")
        native_hash_verifications += 1
        source_components = _components(source_target)
        target_components = _components(target_128)
        prediction_components = _components(prediction_128)
        sample_words: list[dict[str, Any]] = []
        for word_index, word in enumerate(sample.words, start=1):
            original_box = _exclusive_box(word.bbox_xyxy, sample.image_size_wh)
            source_box = map_original_box_to_mask(
                original_box, image_size_wh=sample.image_size_wh, mask_shape=source_target.shape
            )
            native_box = map_original_box_to_mask(
                original_box, image_size_wh=sample.image_size_wh, mask_shape=FROZEN_LOGITS_SHAPE
            )
            source_metric, _, _ = level_word_metrics(
                mask=source_target, components=source_components,
                native_or_source_box=source_box, transcription=word.transcription,
            )
            target_metric, target_ids, native_region = level_word_metrics(
                mask=target_128, components=target_components,
                native_or_source_box=native_box, transcription=word.transcription,
            )
            prediction_metric, prediction_ids, _ = level_word_metrics(
                mask=prediction_128, components=prediction_components,
                native_or_source_box=native_box, transcription=word.transcription,
            )
            matching = match_target_prediction_components(
                target_components=target_components,
                prediction_components=prediction_components,
                target_ids=target_ids,
                prediction_ids=prediction_ids,
                region=native_region,
            )
            word_record = {
                "sample_id": sample_id,
                "gt_index": word_index,
                "gt_text": word.transcription,
                "transcription_length": len(word.transcription),
                "gt_box_original_xyxy_exclusive": list(original_box),
                "gt_box_source_target_xyxy_exclusive": list(source_box),
                "gt_box_native_xyxy_exclusive": list(native_box),
                "levels": {
                    "SOURCE_TARGET_FULLRES": source_metric,
                    "TARGET_128": target_metric,
                    "PREDICTION_128": prediction_metric,
                },
                "target_prediction_matching": matching,
            }
            words.append(word_record)
            sample_words.append(word_record)
        per_sample.append(
            {
                "sample_id": sample_id,
                "source_target_path": str(target_path),
                "source_target_sha256": seg_sample.mask_sha256,
                "source_target_shape": list(source_target.shape),
                "source_target_raw_component_count": len(source_components["records"]),
                "target_128_raw_component_count": len(target_components["records"]),
                "prediction_128_raw_component_count": len(prediction_components["records"]),
                "words": sample_words,
            }
        )
    examples = {}
    for name in ("Tiredness", "GREEK", "intelligent", "Harwich", "Ipswich"):
        selected = next((word for word in words if word["gt_text"].casefold() == name.casefold()), None)
        if selected is None:
            raise SegmentationWordTopologyError("Required example is absent: {}".format(name))
        matching = selected["target_prediction_matching"]
        examples[name] = {
            "sample_id": selected["sample_id"],
            "transcription": selected["gt_text"],
            "transcription_length": selected["transcription_length"],
            "source_component_count": selected["levels"]["SOURCE_TARGET_FULLRES"]["component_count_intersecting_word"],
            "target_128_component_count": selected["levels"]["TARGET_128"]["component_count_intersecting_word"],
            "prediction_128_component_count": selected["levels"]["PREDICTION_128"]["component_count_intersecting_word"],
            "mutual_best_nonzero_pixel_matches": len(matching["one_to_one_plausible_matches"]),
            "target_component_coverage_by_best_prediction": {
                "median": float(np.median([
                    item["fraction_target_component_covered_by_best_prediction"]
                    for item in matching["target_components"]
                ])) if matching["target_components"] else None,
                "minimum": min((item["fraction_target_component_covered_by_best_prediction"] for item in matching["target_components"]), default=None),
            },
            "predicted_component_purity_on_target_foreground": {
                "median": float(np.median([
                    item["fraction_pixels_inside_target_foreground"]
                    for item in matching["predicted_components"]
                ])) if matching["predicted_components"] else None,
                "minimum": min((item["fraction_pixels_inside_target_foreground"] for item in matching["predicted_components"]), default=None),
            },
            "split_target_component_count": len(matching["split_target_components"]),
            "merge_predicted_component_count": len(matching["merge_predicted_components"]),
        }
    return {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "APPROXIMATE_DIAGNOSTIC; target/prediction component topology, not a Table 3 or Table 5 result",
        "accuracy": None,
        "claims": {"table_3": False, "table_5": False, "table_6": False, "table_7": False},
        "source_target_provenance": {
            "paper": "PAPER_SPECIFIED: binary text-foreground segmentation; input 256x256 and masks 128x128.",
            "author_clarification": "AUTHOR_CONFIRMED: training GT is bilinearly reduced to 128x128 then re-binarized at 0.5; training Dont Care has zero BCE validity weight.",
            "training_source": "AUTHOR_CONFIRMED: ICDAR13 Task 2.2 color-coded BMP with paired TXT RGB records; TXT records enumerate character glyph colors and coordinates, while # records mark Dont Care colors.",
            "test_source": "REPOSITORY_SPECIFIED/ASSUMPTION for M10: official Task 2.2 color-coded PNG; exact RGB white is background and every non-white color is foreground; Dont Care remains foreground with all pixels valid.",
            "source_semantics_inference": "DIAGNOSTIC_ONLY: color-coded source targets are glyph/pixel masks, not filled word rectangles. Character-level TXT records in the training source support this conclusion.",
            "target_path": "binary non-white Task 2.2 target -> bilinear 128x128 -> >=0.5 re-binarization; test target has no augmentation.",
            "word_region_mapping": "DIAGNOSTIC_ONLY ASSUMPTION: recognition box inclusive maxima -> exclusive +1; floor lower and ceil upper scaling into each measured mask.",
            "component_connectivity": COMPONENT_CONNECTIVITY,
        },
        "qualification": {
            "sample_count": len(per_sample),
            "word_count": len(words),
            "checkpoint_sha256": checkpoint_sha,
            "native_prediction_hash_verifications": native_hash_verifications,
            "no_inference_rerun": True,
            "no_parseq_call": True,
        },
        "phase_b_topology": {
            "aggregate": {
                level: _component_distribution(words, level)
                for level in ("SOURCE_TARGET_FULLRES", "TARGET_128", "PREDICTION_128")
            },
            "per_word": words,
        },
        "phase_c_target_prediction_matching": _matching_aggregate(words),
        "required_examples": examples,
        "per_sample": per_sample,
        "scientific_boundary": {
            "classification": "DIAGNOSTIC_ONLY",
            "prohibition": "Ground truth is never used to alter prediction, localization, crop selection, threshold, or recognition.",
            "unresolved": ["Whether paper Algorithm 1 includes an unreported grouping/connection stage before component crops."],
        },
    }


def write_target_component_per_word_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for word in report["phase_b_topology"]["per_word"]:
        matching = report_word_matching = word["target_prediction_matching"]
        row: dict[str, Any] = {
            "sample_id": word["sample_id"],
            "gt_index": word["gt_index"],
            "gt_text": word["gt_text"],
            "transcription_length": word["transcription_length"],
            "mutual_best_nonzero_pixel_matches": len(report_word_matching["one_to_one_plausible_matches"]),
            "split_target_component_count": len(report_word_matching["split_target_components"]),
            "merge_predicted_component_count": len(report_word_matching["merge_predicted_components"]),
        }
        for level in ("SOURCE_TARGET_FULLRES", "TARGET_128", "PREDICTION_128"):
            for key, value in word["levels"][level].items():
                row["{}_{}".format(level, key)] = value
        prediction_rows = matching["predicted_components"]
        target_rows = matching["target_components"]
        row["predicted_component_purity_median"] = (
            float(np.median([item["fraction_pixels_inside_target_foreground"] for item in prediction_rows]))
            if prediction_rows else None
        )
        row["target_component_coverage_median"] = (
            float(np.median([item["fraction_target_component_covered_by_best_prediction"] for item in target_rows]))
            if target_rows else None
        )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "diagnose_target_component_topology",
    "level_word_metrics",
    "map_original_box_to_mask",
    "match_target_prediction_components",
    "write_target_component_per_word_csv",
]
