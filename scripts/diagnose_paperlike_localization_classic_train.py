#!/usr/bin/env python3
"""TRAIN-only comparison of classical glyph-mask to word-region diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from localize.classic_word_regions import (
    ClassicMaskGroups,
    global_horizontal_closing,
    line_closing_projection_split,
    linewise_adaptive_closing,
    lowres_binary,
    lowres_connected_components,
    padded_bbox_overlap_grouping,
)
from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.practical_remap_protocol import _box_iou
from scripts.evaluate_supervised_word_boundary_train import (
    EXPECTED_CACHE_SHA256,
    SupervisedTrainRecord,
    _area,
    _evaluate_groups,
    _intersection,
    _native_word_box,
    _prepare_record,
    _sha256_file,
    _summary,
)
from segmentation.data import load_segmentation_manifest


GRID = {
    "A_global_horizontal_closing": {"kernel_height_ratio": (0.25, 0.50, 0.75)},
    "B_linewise_adaptive_closing": {"kernel_height_ratio": (0.25, 0.50, 0.75)},
    "C_line_closing_projection_split": {
        "closing_height_ratio": (0.50,),
        "split_gap_height_ratio": (0.25, 0.50, 0.75),
    },
    "D_padded_bbox_overlap": {"horizontal_expansion_ratio": (0.15, 0.25, 0.35, 0.50)},
    "E_lowres_connected_components": {
        "resolution": (128, 96, 64, 48, 32),
        "interpolation": ("bilinear", "area"),
    },
}
GRID_CONFIGURATION_COUNT = 23
PAD_ORIGINAL_PIXELS = 6
NMAX = 10
SMALL_TEXT_NATIVE_HEIGHT_MAX = 8
MAX_OVERMERGE = 45  # baseline 36 + at most 25%
MAX_MULTIWORD = 46  # baseline 37 + at most 25%


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--supervised-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _declared_variants() -> tuple[dict[str, Any], ...]:
    variants = []
    for ratio in GRID["A_global_horizontal_closing"]["kernel_height_ratio"]:
        variants.append({"technique": "A", "parameters": {"kernel_height_ratio": ratio}})
    for ratio in GRID["B_linewise_adaptive_closing"]["kernel_height_ratio"]:
        variants.append({"technique": "B", "parameters": {"kernel_height_ratio": ratio}})
    for closing in GRID["C_line_closing_projection_split"]["closing_height_ratio"]:
        for split in GRID["C_line_closing_projection_split"]["split_gap_height_ratio"]:
            variants.append({
                "technique": "C",
                "parameters": {"closing_height_ratio": closing, "split_gap_height_ratio": split},
            })
    for ratio in GRID["D_padded_bbox_overlap"]["horizontal_expansion_ratio"]:
        variants.append({"technique": "D", "parameters": {"horizontal_expansion_ratio": ratio}})
    for resolution in GRID["E_lowres_connected_components"]["resolution"]:
        for interpolation in GRID["E_lowres_connected_components"]["interpolation"]:
            variants.append({
                "technique": "E",
                "parameters": {"resolution": resolution, "interpolation": interpolation},
            })
    if len(variants) != GRID_CONFIGURATION_COUNT:
        raise RuntimeError("declared classic grid cardinality drifted")
    return tuple(variants)


def _run_localizer(mask: np.ndarray, variant: dict[str, Any]) -> ClassicMaskGroups:
    technique, parameters = variant["technique"], variant["parameters"]
    if technique == "A":
        return global_horizontal_closing(mask, parameters["kernel_height_ratio"])
    if technique == "B":
        return linewise_adaptive_closing(mask, parameters["kernel_height_ratio"])
    if technique == "C":
        return line_closing_projection_split(mask, **parameters)
    if technique == "D":
        return padded_bbox_overlap_grouping(mask, parameters["horizontal_expansion_ratio"])
    if technique == "E":
        return lowres_connected_components(mask, **parameters)
    raise ValueError(f"unknown technique: {technique}")


def _map_group_box(
    box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    grid_height, grid_width = grid_shape
    return (
        math.floor(box[0] * width / grid_width),
        math.floor(box[1] * height / grid_height),
        math.ceil(box[2] * width / grid_width),
        math.ceil(box[3] * height / grid_height),
    )


def _to_original_groups(
    result: ClassicMaskGroups, width: int, height: int
) -> tuple[tuple[dict[str, Any], ...], frozenset[tuple[int, ...]]]:
    groups = []
    for source in result.groups:
        box = _map_group_box(source["bbox_grid"], source["grid_shape"], width, height)
        raw_ids = tuple(int(value) for value in source["raw_component_ids"])
        # A resampling-only island with no surviving original foreground remains
        # an unmatched proposal, represented by a unique negative diagnostic ID.
        association_ids = raw_ids or (-int(source["group_id"]),)
        groups.append({
            "group_id": int(source["group_id"]),
            "raw_component_ids": association_ids,
            "foreground_area": int(source["foreground_area"]),
            "bbox": box,
            "padded_bbox": (
                max(0, box[0] - PAD_ORIGINAL_PIXELS),
                max(0, box[1] - PAD_ORIGINAL_PIXELS),
                min(width, box[2] + PAD_ORIGINAL_PIXELS),
                min(height, box[3] + PAD_ORIGINAL_PIXELS),
            ),
        })
    # The localizers already sort by processed foreground area. Reassert the
    # deterministic rule after coordinate conversion without changing rank.
    groups.sort(key=lambda value: (
        -value["foreground_area"], value["bbox"][1], value["bbox"][0],
        value["raw_component_ids"],
    ))
    key_counts: dict[tuple[int, ...], int] = {}
    for group in groups:
        key = tuple(group["raw_component_ids"])
        key_counts[key] = key_counts.get(key, 0) + 1
    for group in groups:
        key = tuple(group["raw_component_ids"])
        if key_counts[key] > 1:
            # Resampling may split one original component into multiple output
            # contours. The negative token distinguishes proposals only; all
            # geometry/GT association code ignores non-positive IDs.
            group["raw_component_ids"] = (*key, -int(group["group_id"]))
    keys = [tuple(group["raw_component_ids"]) for group in groups]
    if len(keys) != len(set(keys)):
        raise RuntimeError("diagnostic proposal tokens are not unique")
    return tuple(groups), frozenset(keys[:NMAX])


def _small_text_loss(
    mask: np.ndarray,
    processed: np.ndarray,
    record: SupervisedTrainRecord,
) -> tuple[int, int, int]:
    upsampled = (
        cv2.resize(processed, (128, 128), interpolation=cv2.INTER_NEAREST)
        if processed.shape != (128, 128) else processed
    )
    before = after = words = 0
    for word_box in record.word_boxes:
        native_box = _native_word_box(word_box, record.width, record.height)
        if native_box[3] - native_box[1] > SMALL_TEXT_NATIVE_HEIGHT_MAX:
            continue
        region = np.zeros((128, 128), dtype=bool)
        region[native_box[1]:native_box[3], native_box[0]:native_box[2]] = True
        original_foreground = int((mask.astype(bool) & region).sum())
        if original_foreground == 0:
            continue
        before += original_foreground
        after += int((mask.astype(bool) & upsampled.astype(bool) & region).sum())
        words += 1
    return before, after, words


def _e_diagnostics(
    records: list[SupervisedTrainRecord],
    masks: list[np.ndarray],
    native_results: list[ClassicMaskGroups],
    original_groups: list[tuple[tuple[dict[str, Any], ...], frozenset[tuple[int, ...]]]],
) -> dict[str, Any]:
    raw_before = raw_after = merged = disappeared = fragmented_to_single = 0
    erroneous_groups = 0
    erroneous_words: set[tuple[str, int]] = set()
    small_before = small_after = small_words = 0
    best_ious = []
    for record, mask, result, (groups, _) in zip(records, masks, native_results, original_groups):
        raw_before += result.raw_component_count
        raw_after += result.processed_component_count
        positive_ids = {
            component_id for group in groups for component_id in group["raw_component_ids"]
            if component_id > 0
        }
        disappeared += result.raw_component_count - len(positive_ids)
        merged += sum(max(0, len([value for value in group["raw_component_ids"] if value > 0]) - 1) for group in groups)
        for word_evidence in record.supervised_word_evidence:
            associated = [
                group for group in groups
                if set(group["raw_component_ids"]).intersection(word_evidence.raw_component_ids)
            ]
            if len(word_evidence.raw_component_ids) >= 2 and len(associated) == 1:
                fragmented_to_single += 1
        for group in groups:
            significant_words = [
                word_index for word_index, word_box in enumerate(record.word_boxes, start=1)
                if _intersection(group["bbox"], word_box) / max(1, _area(word_box)) >= 0.25
            ]
            if len(significant_words) >= 2:
                erroneous_groups += 1
                erroneous_words.update((record.sample_id, value) for value in significant_words)
        best_ious.extend(
            max((_box_iou(group["padded_bbox"], word_box) for group in groups[:NMAX]), default=0.0)
            for word_box in record.word_boxes
        )
        before, after, words = _small_text_loss(mask, result.processed_mask, record)
        small_before += before
        small_after += after
        small_words += words
    return {
        "raw_component_count_before": raw_before,
        "raw_component_count_after": raw_after,
        "raw_component_count_reduction": raw_before - raw_after,
        "raw_component_merges": merged,
        "raw_components_disappeared": disappeared,
        "fragmented_gt_words_becoming_single_component": fragmented_to_single,
        "processed_components_significantly_fusing_multiple_gt_words": erroneous_groups,
        "distinct_gt_words_in_erroneous_fusions": len(erroneous_words),
        "small_text_definition": f"native GT bbox height <= {SMALL_TEXT_NATIVE_HEIGHT_MAX}px",
        "small_text_words_with_predicted_foreground": small_words,
        "small_text_original_predicted_foreground_pixels": small_before,
        "small_text_retained_predicted_foreground_pixels": small_after,
        "small_text_foreground_loss_fraction": (
            (small_before - small_after) / small_before if small_before else 0.0
        ),
        "median_best_padded_group_iou": float(np.median(best_ious)),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    summary = row["summary"]
    over_excess = max(0, summary["overmerge"] - MAX_OVERMERGE)
    multi_excess = max(0, row["wordness"]["MULTIWORD_MERGE"] - MAX_MULTIWORD)
    return (
        -(over_excess + multi_excess),
        summary["recall_iou_ge_0_50"],
        row["wordness"]["WORD_GOOD"],
        summary["f1_iou_ge_0_50"],
        -summary["overmerge"],
        -row["wordness"]["MULTIWORD_MERGE"],
    )


def _is_interesting(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        row["summary"]["recall_iou_ge_0_50"] > baseline["recall_iou_ge_0_50"]
        and row["wordness"]["WORD_GOOD"] > baseline["wordness"]["WORD_GOOD"]
        and row["summary"]["overmerge"] <= MAX_OVERMERGE
        and row["wordness"]["MULTIWORD_MERGE"] <= MAX_MULTIWORD
    )


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cache_manifest = args.cache_manifest.resolve()
    if _sha256_file(cache_manifest) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache manifest SHA256 mismatch")
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if (
        manifest.split != "official-training-all-229-author-protocol"
        or manifest.split_role != "train" or len(manifest.samples) != 229
        or cache.get("sample_count") != 229 or cache.get("test_data_loaded") is not False
    ):
        raise ValueError("classic diagnostic permits only official TRAIN-229")

    v1_calibration = json.loads(args.v1_calibration.read_text(encoding="utf-8"))
    v1_params = GroupedWordBlockParams(**v1_calibration["calibration"]["selected"]["parameters"])
    supervised = json.loads(args.supervised_report.read_text(encoding="utf-8"))
    baseline_evaluation = supervised["oof_v1"]
    baseline = {**_summary(baseline_evaluation), "wordness": baseline_evaluation["wordness"]}
    expected_baseline = {
        "iou_ge_0_50": 568, "f1_iou_ge_0_50": 0.553606237816764,
        "undermerge": 92, "overmerge": 36,
    }
    for key, expected in expected_baseline.items():
        if baseline[key] != expected:
            raise RuntimeError(f"v1 baseline mismatch for {key}")
    if baseline["wordness"] != {
        "WORD_GOOD": 552, "WORD_PARTIAL": 130,
        "MULTIWORD_MERGE": 37, "UNMATCHED": 484,
    }:
        raise RuntimeError("v1 wordness baseline mismatch")

    cache_by_id = {str(value["sample_id"]): value for value in cache["samples"]}
    if set(cache_by_id) != {sample.sample_id for sample in manifest.samples}:
        raise RuntimeError("cache/manifest sample identities differ")
    records, masks = [], []
    for index, sample in enumerate(manifest.samples, start=1):
        cached = cache_by_id[sample.sample_id]
        mask_path = (cache_manifest.parent / cached["native_mask_path"]).resolve()
        if _sha256_file(mask_path) != cached["native_mask_file_sha256"]:
            raise RuntimeError(f"mask file SHA mismatch: {sample.sample_id}")
        with Image.open(mask_path) as opened:
            mask = (np.asarray(opened.convert("L")) > 0).astype(np.uint8)
        if (
            mask.shape != (128, 128)
            or hashlib.sha256(mask.tobytes()).hexdigest() != cached["native_mask_array_sha256"]
        ):
            raise RuntimeError(f"mask array mismatch: {sample.sample_id}")
        image_path = (args.data_root / sample.image_path).resolve()
        if _sha256_file(image_path) != cached["image_sha256"]:
            raise RuntimeError(f"source image SHA mismatch: {sample.sample_id}")
        records.append(_prepare_record(
            sample.sample_id, mask, image_path,
            (args.data_root / sample.mask_path).resolve(),
            (args.data_root / sample.validity_annotation_path).resolve(), v1_params,
        ))
        masks.append(mask)
        if index % 10 == 0 or index == 229:
            print(f"[verified cache + prepared] {index}/229", flush=True)

    variants = []
    for variant_index, variant in enumerate(_declared_variants(), start=1):
        native_results = [_run_localizer(mask, variant) for mask in masks]
        original_groups = [
            _to_original_groups(result, record.width, record.height)
            for result, record in zip(native_results, records)
        ]
        evaluation = _evaluate_groups(
            records, original_groups, evidence_attribute="supervised_word_evidence"
        )
        row = {
            **variant,
            "summary": _summary(evaluation),
            "wordness": evaluation["wordness"],
            "images_with_more_than_10_candidates": evaluation["geometry"]["images_with_more_than_10_candidates_before_cap"],
        }
        if variant["technique"] == "E":
            row["lowres_diagnostics"] = _e_diagnostics(
                records, masks, native_results, original_groups
            )
        row["interesting"] = _is_interesting(row, baseline)
        variants.append(row)
        print(
            "[grid] {}/{} technique={} params={} F1={:.6f} recall={:.6f} over={} multi={}".format(
                variant_index, GRID_CONFIGURATION_COUNT, variant["technique"],
                variant["parameters"], row["summary"]["f1_iou_ge_0_50"],
                row["summary"]["recall_iou_ge_0_50"], row["summary"]["overmerge"],
                row["wordness"]["MULTIWORD_MERGE"],
            ), flush=True,
        )

    best_by_technique = {}
    for technique in "ABCDE":
        members = [row for row in variants if row["technique"] == technique]
        best_by_technique[technique] = max(members, key=_selection_key)
    interesting = [row for row in variants if row["interesting"]]
    overall = max(interesting or variants, key=_selection_key)
    e_go = any(row["interesting"] for row in variants if row["technique"] == "E")

    report = {
        "schema_version": 1,
        "mode": "paperlike-localization-classic-train-v1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "train_only": {
            "images": 229, "gt_words": 849, "test_loaded": False,
            "ocr_called": False, "training_executed": False,
            "segmentation_executed": False, "device": "cpu",
        },
        "cache": {
            "manifest": str(cache_manifest), "manifest_sha256": EXPECTED_CACHE_SHA256,
            "verified_samples": 229,
        },
        "baseline_v1": baseline,
        "declared_grid": GRID,
        "declared_configuration_count": GRID_CONFIGURATION_COUNT,
        "fixed_postprocessing": {
            "bbox_remap": "direct from method grid to original image, floor/ceil xyxy-exclusive",
            "padding_original_pixels": PAD_ORIGINAL_PIXELS,
            "sort": "processed foreground area descending, then y/x/component IDs",
            "nmax": NMAX,
            "binarization": "greater_than_or_equal_0.5",
        },
        "interest_rule_declared_before_execution": {
            "recall": "strictly greater than v1",
            "WORD_GOOD": "strictly greater than v1",
            "maximum_overmerge": MAX_OVERMERGE,
            "maximum_MULTIWORD_MERGE": MAX_MULTIWORD,
            "strong_increase_definition": "more than 25% above the v1 count, rounded down to an integer cap",
        },
        "selection_rule": {
            "primary": "only interest-rule variants are promotable",
            "within_safe_variants": "recall, WORD_GOOD, F1, fewer overmerge, fewer MULTIWORD",
            "if_none_safe": "least combined overmerge/MULTIWORD cap excess, then the same ordering",
        },
        "best_by_technique": best_by_technique,
        "overall_best": overall,
        "technique_e_decision": "GO" if e_go else "NO-GO",
        "technique_e_complete": [row for row in variants if row["technique"] == "E"],
        "grid_results": variants,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    print(output, flush=True)
    print(_sha256_file(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
