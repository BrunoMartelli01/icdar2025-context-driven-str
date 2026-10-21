"""Train-only failure analysis for frozen practical-remap-v1.

No result here changes graph parameters, segmentation, or recognition.  The
oracle component-union uses GT boxes only to quantify a localization ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.practical_remap_protocol import TrainWordBox, _box_iou, train_task22_word_boxes
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest


FOREGROUND_RECALL_INSUFFICIENT = 0.1
VERTICAL_OVERLAP_SAME_LINE = 0.5


@dataclass(frozen=True)
class RawComponent:
    component_id: int
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, RawComponent]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    records = {}
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        records[component_id] = RawComponent(
            component_id, (x, y, x + width, y + height),
            int(stats[component_id, cv2.CC_STAT_AREA]),
            (float(centroids[component_id][0]), float(centroids[component_id][1])),
        )
    return labels, records


def _box_union(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )


def _region(shape: tuple[int, int], box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    value = np.zeros(shape, dtype=bool)
    value[y1:y2, x1:x2] = True
    return value


def _pair_metrics(left: Sequence[int], right: Sequence[int]) -> dict[str, float]:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    left_height, right_height = ly2 - ly1, ry2 - ry1
    reference_height = (left_height + right_height) / 2.0
    horizontal_gap = max(0, max(lx1, rx1) - min(lx2, rx2))
    vertical_overlap = max(0, min(ly2, ry2) - max(ly1, ry1))
    min_height = min(left_height, right_height)
    left_center = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
    right_center = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
    return {
        "horizontal_gap_over_reference_height": horizontal_gap / reference_height,
        "vertical_overlap_over_min_height": vertical_overlap / min_height,
        "center_y_distance_over_reference_height": abs(left_center[1] - right_center[1]) / reference_height,
        "height_ratio": max(left_height, right_height) / min_height,
    }


def _quantiles(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows]
    if not values:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def classify_failure(
    *,
    foreground_recall: float,
    best_retained_group_iou: float,
    pre_cap_has_valid_group: bool,
    oracle_union_iou: float,
    associated_group_count: int,
    best_group_gt_word_count: int,
) -> str:
    """Single primary cause, with cap checked before graph-shape causes."""

    if foreground_recall < FOREGROUND_RECALL_INSUFFICIENT:
        return "NO_FOREGROUND"
    if pre_cap_has_valid_group:
        return "TOP10_TRUNCATION"
    if oracle_union_iou >= 0.5 and associated_group_count >= 2:
        return "GROUPING_UNDERMERGE"
    if oracle_union_iou >= 0.5 and best_group_gt_word_count >= 2:
        return "GROUPING_OVERMERGE"
    if oracle_union_iou < 0.5:
        return "GEOMETRY_LIMIT"
    return "OTHER"


def _gate_violations(metrics: dict[str, float], params: GroupedWordBlockParams) -> list[str]:
    violations = []
    if metrics["horizontal_gap_over_reference_height"] > params.horizontal_gap_ratio:
        violations.append("horizontal_gap")
    if metrics["vertical_overlap_over_min_height"] < params.vertical_overlap_min:
        violations.append("vertical_overlap")
    if metrics["center_y_distance_over_reference_height"] > params.max_center_y_ratio:
        violations.append("center_y_distance")
    if metrics["height_ratio"] > params.max_height_ratio:
        violations.append("height_ratio")
    return violations


def _source_binary(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        binary = icdar13_task22_rgb_to_binary_mask(opened)
    return (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8)


def analyze_train_failures(
    *,
    train_manifest_path: Path,
    data_root: Path,
    params: GroupedWordBlockParams,
    predict: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Measure raw/practical/oracle geometry on train, without parameter choice."""

    manifest = load_segmentation_manifest(train_manifest_path)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("Failure analysis requires only the 229-image train manifest")
    per_word: list[dict[str, Any]] = []
    undermerge_pairs: list[dict[str, Any]] = []
    overmerge_pairs: list[dict[str, Any]] = []
    cross_word_horizontal_bridge_proxies: list[dict[str, Any]] = []
    seen_overmerge_groups: set[tuple[str, tuple[int, ...]]] = set()
    source_records = []

    for sample in manifest.samples:
        image_path = (data_root / sample.image_path).resolve()
        mask_path = (data_root / sample.mask_path).resolve()
        annotation_path = (data_root / sample.validity_annotation_path).resolve()
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        target = _source_binary(mask_path)
        prediction = predict(image)
        if prediction.shape != image.shape[:2] or prediction.dtype != np.uint8:
            raise ValueError("Frozen segmentation prediction geometry mismatch")
        words = train_task22_word_boxes(mask_path, annotation_path)
        labels, raw = _raw_components(prediction)
        practical = group_word_blocks(prediction, image, params)
        groups = [dict(value) for value in practical.groups_before_cap]
        retained_group_ids = {
            tuple(block.raw_component_ids) for block in practical.retained
        }
        component_to_words: dict[int, set[int]] = {component_id: set() for component_id in raw}
        word_raw_ids: list[list[int]] = []
        word_regions: list[np.ndarray] = []
        for word_index, word in enumerate(words, start=1):
            region = _region(prediction.shape, word.bbox)
            ids = sorted(int(value) for value in np.unique(labels[region]) if value)
            word_regions.append(region)
            word_raw_ids.append(ids)
            for component_id in ids:
                component_to_words[component_id].add(word_index)
        group_to_words = {
            tuple(group["raw_component_ids"]): sorted(
                set().union(*(component_to_words[component_id] for component_id in group["raw_component_ids"]))
            )
            for group in groups
        }
        # This is not a morphology simulation.  It is a deliberately coarse
        # group-bbox proxy for the newly admitted cross-word links that a
        # wider horizontal-only bridge could make possible.
        spatial_groups = sorted(
            groups, key=lambda group: (group["bbox"][0], group["bbox"][1], group["group_id"])
        )
        for left_index, left_group in enumerate(spatial_groups):
            left_words = set(group_to_words[tuple(left_group["raw_component_ids"])])
            if not left_words:
                continue
            for right_group in spatial_groups[left_index + 1 :]:
                right_words = set(group_to_words[tuple(right_group["raw_component_ids"])])
                if not right_words or not left_words.isdisjoint(right_words):
                    continue
                metrics = _pair_metrics(left_group["bbox"], right_group["bbox"])
                if _gate_violations(metrics, params) == ["horizontal_gap"]:
                    cross_word_horizontal_bridge_proxies.append({
                        "sample_id": sample.sample_id,
                        "left_gt_word_indices": sorted(left_words),
                        "right_gt_word_indices": sorted(right_words),
                        **metrics,
                    })

        for word_index, (word, region, raw_ids) in enumerate(zip(words, word_regions, word_raw_ids), start=1):
            target_pixels = int((target & region).sum())
            overlap_pixels = int((prediction & target & region).sum())
            foreground_recall = overlap_pixels / target_pixels if target_pixels else 0.0
            raw_boxes = [raw[component_id].bbox for component_id in raw_ids]
            raw_ious = [_box_iou(box, word.bbox) for box in raw_boxes]
            oracle_box = _box_union(raw_boxes)
            oracle_iou = _box_iou(oracle_box, word.bbox) if oracle_box is not None else 0.0
            associated_groups = [
                group for group in groups
                if set(group["raw_component_ids"]).intersection(raw_ids)
            ]
            retained_groups = [
                group for group in groups
                if tuple(group["raw_component_ids"]) in retained_group_ids
            ]
            best_retained = max(
                retained_groups, key=lambda group: _box_iou(group["padded_bbox"], word.bbox), default=None
            )
            best_retained_iou = (
                _box_iou(best_retained["padded_bbox"], word.bbox) if best_retained else 0.0
            )
            pre_cap_valid = any(
                _box_iou(group["padded_bbox"], word.bbox) >= 0.5 for group in groups
            )
            best_group_word_count = (
                len(group_to_words[tuple(best_retained["raw_component_ids"])])
                if best_retained is not None else 0
            )
            status = "SUCCESS" if best_retained_iou >= 0.5 else classify_failure(
                foreground_recall=foreground_recall,
                best_retained_group_iou=best_retained_iou,
                pre_cap_has_valid_group=pre_cap_valid,
                oracle_union_iou=oracle_iou,
                associated_group_count=len(associated_groups),
                best_group_gt_word_count=best_group_word_count,
            )
            record = {
                "sample_id": sample.sample_id,
                "gt_word_index": word_index,
                "gt_bbox": list(word.bbox),
                "segmentation_foreground_recall_in_gt_word": foreground_recall,
                "best_raw_component_iou": max(raw_ious, default=0.0),
                "best_practical_v1_group_iou": best_retained_iou,
                "oracle_component_union_bbox": list(oracle_box) if oracle_box else None,
                "oracle_component_union_iou": oracle_iou,
                "raw_component_count_associated": len(raw_ids),
                "practical_group_count_associated": len(associated_groups),
                "best_practical_group_gt_word_count": best_group_word_count,
                "failure_category": status,
            }
            per_word.append(record)
            if status == "GROUPING_UNDERMERGE":
                ordered = sorted(
                    associated_groups,
                    key=lambda group: (group["bbox"][0], group["bbox"][1], group["group_id"]),
                )
                target_component_heights = [
                    raw[component_id].bbox[3] - raw[component_id].bbox[1]
                    for component_id in raw_ids
                ]
                median_raw_height = float(np.median(target_component_heights)) if target_component_heights else 0.0
                for left, right in zip(ordered, ordered[1:]):
                    metrics = _pair_metrics(left["bbox"], right["bbox"])
                    violations = _gate_violations(metrics, params)
                    left_small = any(
                        raw[component_id].bbox[3] - raw[component_id].bbox[1] <= 0.35 * median_raw_height
                        for component_id in left["raw_component_ids"]
                    )
                    right_small = any(
                        raw[component_id].bbox[3] - raw[component_id].bbox[1] <= 0.35 * median_raw_height
                        for component_id in right["raw_component_ids"]
                    )
                    undermerge_pairs.append({
                        "sample_id": sample.sample_id,
                        "gt_word_index": word_index,
                        **metrics,
                        "left_raw_component_count": len(left["raw_component_ids"]),
                        "right_raw_component_count": len(right["raw_component_ids"]),
                        "small_component_proxy_present": left_small or right_small,
                        "gate_violations": violations,
                        "violation_count": len(violations),
                    })
            if status == "GROUPING_OVERMERGE" and best_retained is not None:
                group_key = (sample.sample_id, tuple(best_retained["raw_component_ids"]))
                if group_key not in seen_overmerge_groups:
                    seen_overmerge_groups.add(group_key)
                    associated_word_ids = group_to_words[group_key[1]]
                    for left_index, left_word_index in enumerate(associated_word_ids):
                        for right_word_index in associated_word_ids[left_index + 1 :]:
                            metrics = _pair_metrics(
                                words[left_word_index - 1].bbox, words[right_word_index - 1].bbox
                            )
                            overmerge_pairs.append({
                                "sample_id": sample.sample_id,
                                "group_raw_component_ids": list(group_key[1]),
                                "gt_word_count_in_group": len(associated_word_ids),
                                "same_line": metrics["vertical_overlap_over_min_height"] >= VERTICAL_OVERLAP_SAME_LINE,
                                **metrics,
                            })
        source_records.append({
            "sample_id": sample.sample_id,
            "image_sha256": _sha256_file(image_path),
            "mask_sha256": _sha256_file(mask_path),
            "annotation_sha256": _sha256_file(annotation_path),
            "gt_word_count": len(words),
        })

    failures = [record for record in per_word if record["failure_category"] != "SUCCESS"]
    categories = (
        "NO_FOREGROUND", "GROUPING_UNDERMERGE", "GROUPING_OVERMERGE",
        "TOP10_TRUNCATION", "GEOMETRY_LIMIT", "OTHER",
    )
    threshold = 0.5
    def _count_rate(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, float | int]:
        count = sum(predicate(record) for record in per_word)
        return {
            "count": count,
            "fraction_of_gt_words": count / len(per_word) if per_word else 0.0,
        }

    ceiling = {
        "gt_word_count": len(per_word),
        "raw_component_best_iou_ge_0_5": _count_rate(
            lambda record: record["best_raw_component_iou"] >= threshold
        ),
        "practical_v1_best_group_iou_ge_0_5": _count_rate(
            lambda record: record["best_practical_v1_group_iou"] >= threshold
        ),
        "oracle_component_union_iou_ge_0_5": _count_rate(
            lambda record: record["oracle_component_union_iou"] >= threshold
        ),
        "segmentation_foreground_recall_in_gt_word": _quantiles(
            per_word, "segmentation_foreground_recall_in_gt_word"
        ),
    }
    under_stats = {
        "undermerge_word_count": sum(record["failure_category"] == "GROUPING_UNDERMERGE" for record in per_word),
        "adjacent_group_pair_count": len(undermerge_pairs),
        "quantiles": {
            key: _quantiles(undermerge_pairs, key)
            for key in (
                "horizontal_gap_over_reference_height", "vertical_overlap_over_min_height",
                "center_y_distance_over_reference_height", "height_ratio",
                "left_raw_component_count", "right_raw_component_count",
            )
        },
        "small_component_proxy_pair_count": sum(row["small_component_proxy_present"] for row in undermerge_pairs),
        "small_component_proxy_fraction": (
            sum(row["small_component_proxy_present"] for row in undermerge_pairs) / len(undermerge_pairs)
            if undermerge_pairs else 0.0
        ),
        "single_gate_violation_pairs": sum(row["violation_count"] == 1 for row in undermerge_pairs),
        "multiple_gate_violation_pairs": sum(row["violation_count"] >= 2 for row in undermerge_pairs),
        "zero_gate_violation_pairs": sum(row["violation_count"] == 0 for row in undermerge_pairs),
        "gate_violation_counts": {
            name: sum(name in row["gate_violations"] for row in undermerge_pairs)
            for name in ("horizontal_gap", "vertical_overlap", "center_y_distance", "height_ratio")
        },
    }
    horizontal_bridge_only = [
        row for row in undermerge_pairs if row["gate_violations"] == ["horizontal_gap"]
    ]
    bridge_gap_quantiles = _quantiles(
        horizontal_bridge_only, "horizontal_gap_over_reference_height"
    )
    minimum_cross_word_gap = min(
        (row["horizontal_gap_over_reference_height"] for row in cross_word_horizontal_bridge_proxies),
        default=None,
    )
    safe_against_observed_proxy = [
        row for row in horizontal_bridge_only
        if minimum_cross_word_gap is not None
        and row["horizontal_gap_over_reference_height"] < minimum_cross_word_gap
    ]
    under_stats["horizontal_bridge_only_gate"] = {
        "pair_count": len(horizontal_bridge_only),
        "fraction_of_adjacent_undermerge_pairs": (
            len(horizontal_bridge_only) / len(undermerge_pairs) if undermerge_pairs else 0.0
        ),
        "word_count": len({(row["sample_id"], row["gt_word_index"]) for row in horizontal_bridge_only}),
        "interpretation": (
            "Pairs satisfy every frozen v1 vertical/height gate and fail only horizontal gap; "
            "they are the geometry-only upper bound for horizontal bridging without relaxing those gates."
        ),
    }
    under_stats["horizontal_bridge_risk_proxy"] = {
        "undermerge_required_gap_quantiles": bridge_gap_quantiles,
        "different_word_group_pair_count_only_horizontal_gate": len(cross_word_horizontal_bridge_proxies),
        "different_word_gap_quantiles_only_horizontal_gate": _quantiles(
            cross_word_horizontal_bridge_proxies, "horizontal_gap_over_reference_height"
        ),
        "minimum_different_word_gap_only_horizontal_gate": minimum_cross_word_gap,
        "undermerge_pair_count_below_that_minimum": len(safe_against_observed_proxy),
        "undermerge_word_count_below_that_minimum": len({
            (row["sample_id"], row["gt_word_index"]) for row in safe_against_observed_proxy
        }),
        "interpretation": (
            "Diagnostic group-bbox proxy only: a horizontal bridge reaching the minimum observed "
            "different-word gap could create a new cross-word link. Counts below that gap are "
            "the narrowest observed no-new-cross-word window, not a guarantee for morphology."
        ),
    }
    over_stats = {
        "overmerge_word_count": sum(record["failure_category"] == "GROUPING_OVERMERGE" for record in per_word),
        "unique_overmerged_group_count": len(seen_overmerge_groups),
        "gt_word_pair_count": len(overmerge_pairs),
        "same_line_pair_count": sum(row["same_line"] for row in overmerge_pairs),
        "different_line_pair_count": sum(not row["same_line"] for row in overmerge_pairs),
        "quantiles": {
            key: _quantiles(overmerge_pairs, key)
            for key in (
                "horizontal_gap_over_reference_height", "vertical_overlap_over_min_height",
                "center_y_distance_over_reference_height", "height_ratio", "gt_word_count_in_group",
            )
        },
        "pair_records": overmerge_pairs,
    }
    return {
        "schema_version": 1,
        "mode": "practical-remap-v1",
        "classification": "PRACTICAL_APPROXIMATION diagnostic; train only",
        "train_only_guard": {
            "manifest": str(train_manifest_path.resolve()),
            "manifest_sha256": _sha256_file(train_manifest_path.resolve()),
            "sample_count": len(source_records),
            "test_data_loaded": False,
            "ocr_called": False,
            "parameter_selection_performed": False,
        },
        "parameters_analyzed_not_changed": asdict(params),
        "diagnostic_definitions": {
            "foreground_recall": "predicted foreground pixels intersecting care target foreground inside GT word / care target foreground pixels inside GT word",
            "no_foreground": "foreground recall < {} (diagnostic threshold; not a model parameter)".format(FOREGROUND_RECALL_INSUFFICIENT),
            "oracle_component_union": "un-padded union bbox of all raw predicted components with at least one foreground pixel inside the GT word box; DIAGNOSTIC_ONLY",
            "top10_primary_priority": "a pre-cap group has IoU>=0.5 but no retained group does",
            "undermerge_pair": "adjacent left-to-right practical groups containing raw foreground associated with the same GT word",
            "small_component_proxy": "either pair side contains raw component height <= 0.35 times the word-associated raw-component median; no lexical punctuation label used",
            "same_line": "GT-word vertical-overlap/min-height >= {}".format(VERTICAL_OVERLAP_SAME_LINE),
            "horizontal_bridge_risk_proxy": "unmerged practical-group bbox pairs tied to disjoint GT words that pass every frozen v1 gate except horizontal gap; DIAGNOSTIC_ONLY, not a morphology simulation",
            "same_line": "GT-word vertical-overlap/min-height >= {}".format(VERTICAL_OVERLAP_SAME_LINE),
        },
        "ceiling_train": ceiling,
        "failure_breakdown": {
            "failure_count": len(failures),
            "success_count": len(per_word) - len(failures),
            "categories": {
                category: {
                    "count": sum(record["failure_category"] == category for record in per_word),
                    "fraction_of_failures": (
                        sum(record["failure_category"] == category for record in per_word) / len(failures)
                        if failures else 0.0
                    ),
                }
                for category in categories
            },
        },
        "undermerge": under_stats,
        "overmerge": over_stats,
        "undermerge_pair_records": undermerge_pairs,
        "per_word": per_word,
        "train_sources": source_records,
        "scientific_boundary": {
            "prohibition": "No train statistic has modified v1 or selected a v2 configuration; test, PARSeq, context scoring, and segmentation changes are absent.",
        },
    }


__all__ = ["analyze_train_failures", "classify_failure"]
