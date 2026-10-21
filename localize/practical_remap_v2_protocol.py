"""Train-only calibration and frozen-test geometry for practical-remap-v2.

The module has no paper-faithful entry point: it exists only for the explicitly
labelled practical approximation.  It shares masks across v1/v2 comparisons and
keeps GT boxes outside every localizer call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.line_word_blocks import (
    PRACTICAL_REMAP_V2,
    V2PreparedMask,
    V2WordBlockParams,
    group_prepared_word_blocks_v2,
    prepare_word_blocks_v2,
)
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from localize.practical_remap_failure_analysis import (
    _box_union,
    _region,
    classify_failure,
)
from localize.practical_remap_protocol import (
    PADDING,
    TrainWordBox,
    _box_iou,
    proposal_geometry,
    train_task22_word_boxes,
)
from recognition.icdar13_annotations import ICDAR13RecognitionSample
from segmentation.data import (
    SegmentationManifest,
    icdar13_task22_rgb_to_binary_mask,
    load_segmentation_manifest,
)


V2_GRID = {
    "absolute": {"absolute_gap_floor": (0.25, 0.35, 0.45)},
    "local_ratio": {"local_window": (2, 3), "local_gap_ratio": (1.5, 2.0, 2.5, 3.0)},
    "hybrid": {
        "absolute_gap_floor": (0.25, 0.35, 0.45),
        "local_window": (2, 3),
        "local_gap_ratio": (1.5, 2.0, 2.5, 3.0),
    },
}
V2_GRID_CONFIGURATION_COUNT = 35
V2_LINE_VERTICAL_OVERLAP_MIN = 0.7
V2_LINE_MAX_CENTER_Y_RATIO = 0.5
V2_LINE_MAX_HEIGHT_RATIO = 2.5
V2_FRAGMENT_X_OVERLAP_MIN = 0.5
V2_FRAGMENT_MAX_VERTICAL_GAP_RATIO = 0.5
V2_FRAGMENT_SMALL_HEIGHT_RATIO = 0.45
V2_LOCAL_FALLBACK_ABSOLUTE_FLOOR = 0.35
V2_PAD = PADDING
V2_CONNECTIVITY = 8
V2_NMAX = 10
V2_F1_ATOL = 0.005


@dataclass(frozen=True)
class V2TrainRecord:
    sample_id: str
    image: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    words: tuple[TrainWordBox, ...]
    raw_labels: np.ndarray
    raw_boxes: dict[int, tuple[int, int, int, int]]
    prepared: V2PreparedMask


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_target(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(icdar13_task22_rgb_to_binary_mask(image), dtype=np.uint8) > 0).astype(np.uint8)


def _raw_label_geometry(mask: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    boxes = {
        component_id: (
            int(stats[component_id, cv2.CC_STAT_LEFT]),
            int(stats[component_id, cv2.CC_STAT_TOP]),
            int(stats[component_id, cv2.CC_STAT_LEFT] + stats[component_id, cv2.CC_STAT_WIDTH]),
            int(stats[component_id, cv2.CC_STAT_TOP] + stats[component_id, cv2.CC_STAT_HEIGHT]),
        )
        for component_id in range(1, count)
    }
    return labels, boxes


def declared_v2_grid() -> tuple[V2WordBlockParams, ...]:
    """The 35 configurations declared before train predictions are inspected."""

    common = {
        "line_vertical_overlap_min": V2_LINE_VERTICAL_OVERLAP_MIN,
        "line_max_center_y_ratio": V2_LINE_MAX_CENTER_Y_RATIO,
        "line_max_height_ratio": V2_LINE_MAX_HEIGHT_RATIO,
        "fragment_x_overlap_min": V2_FRAGMENT_X_OVERLAP_MIN,
        "fragment_max_vertical_gap_ratio": V2_FRAGMENT_MAX_VERTICAL_GAP_RATIO,
        "fragment_small_height_ratio": V2_FRAGMENT_SMALL_HEIGHT_RATIO,
        "pad": V2_PAD,
        "connectivity": V2_CONNECTIVITY,
        "tie_break": APPROXIMATE_TIE_BREAK,
        "nmax": V2_NMAX,
        "local_fallback_absolute_floor": V2_LOCAL_FALLBACK_ABSOLUTE_FLOOR,
    }
    values = []
    for absolute_floor in V2_GRID["absolute"]["absolute_gap_floor"]:
        values.append(V2WordBlockParams(
            family="absolute", absolute_gap_floor=absolute_floor,
            local_window=2, local_gap_ratio=2.0, **common,
        ))
    for window in V2_GRID["local_ratio"]["local_window"]:
        for ratio in V2_GRID["local_ratio"]["local_gap_ratio"]:
            values.append(V2WordBlockParams(
                family="local_ratio", absolute_gap_floor=V2_LOCAL_FALLBACK_ABSOLUTE_FLOOR,
                local_window=window, local_gap_ratio=ratio, **common,
            ))
    for absolute_floor in V2_GRID["hybrid"]["absolute_gap_floor"]:
        for window in V2_GRID["hybrid"]["local_window"]:
            for ratio in V2_GRID["hybrid"]["local_gap_ratio"]:
                values.append(V2WordBlockParams(
                    family="hybrid", absolute_gap_floor=absolute_floor,
                    local_window=window, local_gap_ratio=ratio, **common,
                ))
    if len(values) != V2_GRID_CONFIGURATION_COUNT:
        raise RuntimeError("Declared practical-remap-v2 grid cardinality drifted")
    return tuple(values)


def load_v2_train_records(
    manifest: SegmentationManifest,
    data_root: Path,
    predict: Callable[[np.ndarray], np.ndarray],
    preparation_params: V2WordBlockParams,
    progress: Callable[[str], None] | None = None,
) -> tuple[tuple[V2TrainRecord, ...], dict[str, Any]]:
    """Load exactly train-229 and cache one frozen prediction per image."""

    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("practical-remap-v2 calibration accepts only train-229")
    records = []
    sources = []
    for sample_index, sample in enumerate(manifest.samples, start=1):
        image_path = (data_root / sample.image_path).resolve()
        mask_path = (data_root / sample.mask_path).resolve()
        annotation_path = (data_root / sample.validity_annotation_path).resolve()
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        prediction = predict(rgb)
        if prediction.dtype != np.uint8 or prediction.shape != rgb.shape[:2]:
            raise ValueError("frozen segmentation prediction violates source geometry")
        target = _source_target(mask_path)
        labels, raw_boxes = _raw_label_geometry(prediction)
        words = train_task22_word_boxes(mask_path, annotation_path)
        records.append(V2TrainRecord(
            sample_id=sample.sample_id, image=rgb, prediction=prediction, target=target,
            words=words, raw_labels=labels, raw_boxes=raw_boxes,
            prepared=prepare_word_blocks_v2(prediction, preparation_params),
        ))
        sources.append({
            "sample_id": sample.sample_id,
            "image_sha256": _sha256_file(image_path),
            "mask_sha256": _sha256_file(mask_path),
            "annotation_sha256": _sha256_file(annotation_path),
            "gt_word_count": len(words),
        })
        if progress is not None:
            progress(
                "[train masks] {}/{} sample_id={} raw_components={} lines={}".format(
                    sample_index, len(manifest.samples), sample.sample_id,
                    len(raw_boxes), len(records[-1].prepared.line_candidates),
                )
            )
    return tuple(records), {"sample_count": len(records), "samples": sources}


def _result_groups(result: Any) -> tuple[list[dict[str, Any]], set[tuple[int, ...]]]:
    groups = [dict(value) for value in result.groups_before_cap]
    retained_ids = {tuple(value.raw_component_ids) for value in result.retained}
    return groups, retained_ids


def _failure_and_geometry(
    records: Sequence[V2TrainRecord],
    localize: Callable[[V2TrainRecord], Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Apply the v1 primary-failure definitions without changing their priority."""

    per_word = []
    per_image = []
    categories = (
        "NO_FOREGROUND", "GROUPING_UNDERMERGE", "GROUPING_OVERMERGE",
        "TOP10_TRUNCATION", "GEOMETRY_LIMIT", "OTHER",
    )
    for record in records:
        result = localize(record)
        groups, retained_ids = _result_groups(result)
        component_to_words: dict[int, set[int]] = {component_id: set() for component_id in record.raw_boxes}
        word_raw_ids = []
        word_regions = []
        for word_index, word in enumerate(record.words, start=1):
            region = _region(record.prediction.shape, word.bbox)
            raw_ids = sorted(int(value) for value in np.unique(record.raw_labels[region]) if value)
            word_regions.append(region)
            word_raw_ids.append(raw_ids)
            for raw_id in raw_ids:
                component_to_words[raw_id].add(word_index)
        group_to_words = {
            tuple(group["raw_component_ids"]): sorted(
                set().union(*(component_to_words[component_id] for component_id in group["raw_component_ids"]))
            )
            for group in groups
        }
        for word_index, (word, region, raw_ids) in enumerate(zip(record.words, word_regions, word_raw_ids), start=1):
            target_pixels = int((record.target & region).sum())
            predicted_target_pixels = int((record.target & record.prediction & region).sum())
            foreground_recall = predicted_target_pixels / target_pixels if target_pixels else 0.0
            raw_boxes = [record.raw_boxes[raw_id] for raw_id in raw_ids]
            oracle_box = _box_union(raw_boxes)
            oracle_iou = _box_iou(oracle_box, word.bbox) if oracle_box is not None else 0.0
            raw_iou = max((_box_iou(box, word.bbox) for box in raw_boxes), default=0.0)
            associated = [
                group for group in groups if set(group["raw_component_ids"]).intersection(raw_ids)
            ]
            retained = [
                group for group in groups if tuple(group["raw_component_ids"]) in retained_ids
            ]
            best = max(retained, key=lambda group: _box_iou(group["padded_bbox"], word.bbox), default=None)
            best_iou = _box_iou(best["padded_bbox"], word.bbox) if best is not None else 0.0
            pre_cap_valid = any(_box_iou(group["padded_bbox"], word.bbox) >= 0.5 for group in groups)
            best_group_word_count = len(group_to_words[tuple(best["raw_component_ids"])]) if best is not None else 0
            category = "SUCCESS" if best_iou >= 0.5 else classify_failure(
                foreground_recall=foreground_recall,
                best_retained_group_iou=best_iou,
                pre_cap_has_valid_group=pre_cap_valid,
                oracle_union_iou=oracle_iou,
                associated_group_count=len(associated),
                best_group_gt_word_count=best_group_word_count,
            )
            per_word.append({
                "sample_id": record.sample_id,
                "gt_word_index": word_index,
                "gt_bbox": list(word.bbox),
                "segmentation_foreground_recall_in_gt_word": foreground_recall,
                "best_raw_component_iou": raw_iou,
                "best_{}_group_iou".format(label): best_iou,
                "oracle_component_union_iou": oracle_iou,
                "raw_component_count_associated": len(raw_ids),
                "group_count_associated": len(associated),
                "best_group_gt_word_count": best_group_word_count,
                "failure_category": category,
            })
        boxes = [tuple(value.padded_bbox) for value in result.retained]
        merges = sum(len(value["raw_component_ids"]) - 1 for value in groups)
        multi = sum(len(value["raw_component_ids"]) > 1 for value in groups)
        per_image.append((boxes, [word.bbox for word in record.words], len(groups), merges, multi))
    failures = [value for value in per_word if value["failure_category"] != "SUCCESS"]
    ceiling = {
        "gt_word_count": len(per_word),
        "raw_component_best_iou_ge_0_5": sum(value["best_raw_component_iou"] >= 0.5 for value in per_word),
        "best_group_iou_ge_0_5": sum(value["failure_category"] == "SUCCESS" for value in per_word),
        "oracle_component_union_iou_ge_0_5": sum(value["oracle_component_union_iou"] >= 0.5 for value in per_word),
    }
    return {
        "label": label,
        "geometry": proposal_geometry(per_image).as_dict(),
        "ceiling": ceiling,
        "failure_breakdown": {
            "success_count": len(per_word) - len(failures),
            "failure_count": len(failures),
            "categories": {
                category: {
                    "count": sum(value["failure_category"] == category for value in per_word),
                    "fraction_of_failures": (
                        sum(value["failure_category"] == category for value in per_word) / len(failures)
                        if failures else 0.0
                    ),
                }
                for category in categories
            },
        },
        "per_word": per_word,
    }


def _v1_localizer(params: GroupedWordBlockParams) -> Callable[[V2TrainRecord], Any]:
    return lambda record: group_word_blocks(record.prediction, record.image, params)


def _v2_localizer(params: V2WordBlockParams) -> Callable[[V2TrainRecord], Any]:
    return lambda record: group_prepared_word_blocks_v2(record.prepared, record.image, params)


def _complexity_rank(params: V2WordBlockParams) -> int:
    return {"absolute": 0, "local_ratio": 1, "hybrid": 2}[params.family]


def _compare_changes(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, int]:
    v1_by_key = {(value["sample_id"], value["gt_word_index"]): value for value in v1["per_word"]}
    v2_by_key = {(value["sample_id"], value["gt_word_index"]): value for value in v2["per_word"]}
    if v1_by_key.keys() != v2_by_key.keys():
        raise ValueError("v1/v2 train comparison word populations differ")
    values = list(v1_by_key)
    return {
        "v1_undermerge_count": sum(v1_by_key[key]["failure_category"] == "GROUPING_UNDERMERGE" for key in values),
        "v1_undermerge_recovered_by_v2": sum(
            v1_by_key[key]["failure_category"] == "GROUPING_UNDERMERGE"
            and v2_by_key[key]["failure_category"] == "SUCCESS" for key in values
        ),
        "v1_overmerge_count": sum(v1_by_key[key]["failure_category"] == "GROUPING_OVERMERGE" for key in values),
        "v1_overmerge_resolved_by_v2": sum(
            v1_by_key[key]["failure_category"] == "GROUPING_OVERMERGE"
            and v2_by_key[key]["failure_category"] != "GROUPING_OVERMERGE" for key in values
        ),
        "v2_new_overmerge_count": sum(
            v1_by_key[key]["failure_category"] != "GROUPING_OVERMERGE"
            and v2_by_key[key]["failure_category"] == "GROUPING_OVERMERGE" for key in values
        ),
    }


def calibrate_v2_on_train(
    records: Sequence[V2TrainRecord],
    v1_params: GroupedWordBlockParams,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate all predeclared v2 variants and freeze one using train only."""

    v1 = _failure_and_geometry(records, _v1_localizer(v1_params), label="practical_v1")
    variants = []
    grid = declared_v2_grid()
    for variant_index, params in enumerate(grid, start=1):
        evaluation = _failure_and_geometry(records, _v2_localizer(params), label="practical_v2")
        variants.append({
            "parameters": asdict(params),
            "geometry": evaluation["geometry"],
            "failure_breakdown": evaluation["failure_breakdown"],
            "selection_secondary": {
                "overmerge_failure_count": evaluation["failure_breakdown"]["categories"]["GROUPING_OVERMERGE"]["count"],
                "undermerge_failure_count": evaluation["failure_breakdown"]["categories"]["GROUPING_UNDERMERGE"]["count"],
                "algorithm_complexity_rank": _complexity_rank(params),
                "average_proposals_per_image": evaluation["geometry"]["average_proposals_per_image"],
            },
            "_evaluation": evaluation,
        })
        if progress is not None:
            geometry = evaluation["geometry"]
            progress(
                "[train grid] {}/{} family={} floor={} window={} ratio={} F1@.5={:.6f} recall={:.6f} precision={:.6f}".format(
                    variant_index, len(grid), params.family, params.absolute_gap_floor,
                    params.local_window, params.local_gap_ratio,
                    geometry["f1_proposal_level_iou_ge_0_50"],
                    geometry["recall_iou_ge_0_50"],
                    geometry["proposal_precision_iou_ge_0_50"],
                )
            )
    best_f1 = max(value["geometry"]["f1_proposal_level_iou_ge_0_50"] for value in variants)
    near_best = [
        value for value in variants
        if best_f1 - value["geometry"]["f1_proposal_level_iou_ge_0_50"] <= V2_F1_ATOL
    ]
    selected_internal = min(
        near_best,
        key=lambda value: (
            value["selection_secondary"]["overmerge_failure_count"],
            value["selection_secondary"]["undermerge_failure_count"],
            value["selection_secondary"]["algorithm_complexity_rank"],
            value["selection_secondary"]["average_proposals_per_image"],
        ),
    )
    selected = {key: value for key, value in selected_internal.items() if key != "_evaluation"}
    v2 = selected_internal["_evaluation"]
    public_variants = [
        {key: value for key, value in variant.items() if key != "_evaluation"}
        for variant in variants
    ]
    return {
        "mode": PRACTICAL_REMAP_V2,
        "classification": "PRACTICAL_APPROXIMATION",
        "declared_grid": V2_GRID,
        "declared_configuration_count": V2_GRID_CONFIGURATION_COUNT,
        "fixed_line_formation": {
            "vertical_overlap_min": V2_LINE_VERTICAL_OVERLAP_MIN,
            "max_center_y_ratio": V2_LINE_MAX_CENTER_Y_RATIO,
            "max_height_ratio": V2_LINE_MAX_HEIGHT_RATIO,
            "horizontal_gap_used": False,
        },
        "fixed_fragment_consolidation": {
            "x_overlap_min": V2_FRAGMENT_X_OVERLAP_MIN,
            "max_vertical_gap_ratio": V2_FRAGMENT_MAX_VERTICAL_GAP_RATIO,
            "small_height_ratio": V2_FRAGMENT_SMALL_HEIGHT_RATIO,
            "classification": "PRACTICAL_APPROXIMATION fixed geometric pre-stage",
        },
        "selection_rule_declared_before_execution": {
            "primary": "maximize proposal-level F1 at IoU>=0.5",
            "near_tie": "within absolute F1 0.005 of maximum",
            "tie_order": [
                "fewer GROUPING_OVERMERGE failures",
                "fewer GROUPING_UNDERMERGE failures",
                "simpler boundary family: absolute, local_ratio, hybrid",
                "fewer proposals per image",
            ],
        },
        "best_f1": best_f1,
        "near_best_variant_count": len(near_best),
        "selected": selected,
        "grid_results": public_variants,
        "train_comparison": {
            "v1": {key: value for key, value in v1.items() if key != "per_word"},
            "v2": {key: value for key, value in v2.items() if key != "per_word"},
            "oracle_ceiling_shared_raw_components": v2["ceiling"]["oracle_component_union_iou_ge_0_5"],
            "v1_to_v2_changes": _compare_changes(v1, v2),
        },
        "v1_per_word": v1["per_word"],
        "v2_per_word": v2["per_word"],
    }


def evaluate_v1_v2_test_geometry(
    samples: Sequence[ICDAR13RecognitionSample],
    masks: Sequence[np.ndarray],
    v1_params: GroupedWordBlockParams,
    v2_params: V2WordBlockParams,
) -> dict[str, Any]:
    """Evaluate Algorithm1, v1, and v2 from precisely the same masks."""

    if len(samples) != 233 or len(masks) != 233:
        raise ValueError("frozen test geometry requires exactly all 233 test samples and masks")
    alg1_rows = []
    v1_rows = []
    v2_rows = []
    per_sample = []
    alg1_params = BlockParam(150, PADDING, 8, APPROXIMATE_TIE_BREAK, 10)
    for sample, mask in zip(samples, masks):
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        gt_boxes = [
            (word.bbox_xyxy[0], word.bbox_xyxy[1], word.bbox_xyxy[2] + 1, word.bbox_xyxy[3] + 1)
            for word in sample.words
        ]
        alg1 = localize_mask(mask, image, alg1_params)
        v1 = group_word_blocks(mask, image, v1_params)
        prepared = prepare_word_blocks_v2(mask, v2_params)
        v2 = group_prepared_word_blocks_v2(prepared, image, v2_params)
        alg1_boxes = [value.padded_bbox for value in alg1.retained]
        v1_boxes = [value.padded_bbox for value in v1.retained]
        v2_boxes = [value.padded_bbox for value in v2.retained]
        alg1_rows.append((alg1_boxes, gt_boxes, len(alg1.retained) + sum(value.reason == "top_10_limit" for value in alg1.rejected), 0, 0))
        v1_rows.append((v1_boxes, gt_boxes, len(v1.groups_before_cap), sum(len(value["raw_component_ids"]) - 1 for value in v1.groups_before_cap), sum(len(value["raw_component_ids"]) > 1 for value in v1.groups_before_cap)))
        v2_rows.append((v2_boxes, gt_boxes, len(v2.groups_before_cap), sum(len(value["raw_component_ids"]) - 1 for value in v2.groups_before_cap), sum(len(value["raw_component_ids"]) > 1 for value in v2.groups_before_cap)))
        per_sample.append({
            "sample_id": sample.sample_id,
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "algorithm1_groups_before_cap": alg1_rows[-1][2],
            "v1_groups_before_cap": v1_rows[-1][2],
            "v2_groups_before_cap": v2_rows[-1][2],
        })
    return {
        "algorithm1_paper_faithful": proposal_geometry(alg1_rows).as_dict(),
        "practical_remap_v1": proposal_geometry(v1_rows).as_dict(),
        "practical_remap_v2": proposal_geometry(v2_rows).as_dict(),
        "same_mask_contract": {
            "sample_count": len(per_sample),
            "mask_generation_count": len(per_sample),
            "all_three_localizers_consume_each_sample_mask_once": True,
        },
        "per_sample": per_sample,
    }


__all__ = [
    "PRACTICAL_REMAP_V2", "V2TrainRecord", "V2_GRID", "V2_GRID_CONFIGURATION_COUNT",
    "V2_F1_ATOL", "calibrate_v2_on_train", "declared_v2_grid",
    "evaluate_v1_v2_test_geometry", "load_v2_train_records",
]
