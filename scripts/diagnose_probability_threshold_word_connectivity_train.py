#!/usr/bin/env python3
"""TRAIN-only native probability-threshold connectivity diagnostic.

No segmentation, grouping, morphology, OCR, or TEST data is used.  Every
threshold follows the frozen Algorithm 1 area/filter/pad/sort/top-10 path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from localize.practical_remap_failure_analysis import _box_union, classify_failure
from localize.practical_remap_protocol import _box_iou, proposal_geometry, train_task22_word_boxes
from localize.probability_threshold_word_connectivity import (
    algorithm1_contract, component_ids_intersecting_original_box,
    localize_native_components, nearest_axis_cells,
)
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.diagnose_multihypothesis_word_proposal_oracle_train import (
    EXPECTED_CHECKPOINT_SHA256, EXPECTED_MASK_MANIFEST_SHA256,
    EXPECTED_PROBABILITY_MANIFEST_SHA256, EXPECTED_V1_SHA256,
    _load_probability,
)
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest


THRESHOLDS = (0.50, 0.475, 0.45, 0.425, 0.40, 0.375, 0.35, 0.325, 0.30)
EXPECTED_TRAIN_MANIFEST_SHA256 = "3aeecf40411326f0721e5d4eea1bf74a9357eab1c9da54a77d0624b6e24f7c14"
EXPECTED_SPLIT = "official-training-all-229-author-protocol"
MIN_AREA = 150
PAD = 6
CONNECTIVITY = 8
NMAX = 10
BASELINE_V1 = {
    "iou_ge_0_50": 568,
    "WORD_GOOD": 552,
    "undermerge": 92,
    "overmerge": 36,
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _area(box: Sequence[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _contained(root: Path, relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("cached path must be relative")
    result = (root / candidate).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("cached path escapes declared root")
    return result


def _write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _validate_algorithm1_identity(
    mask: np.ndarray, localization: Any, *, width: int, height: int,
) -> None:
    projected = backproject_binary_mask(mask, (height, width))
    direct = localize_mask(
        projected, projected,
        BlockParam(MIN_AREA, PAD, CONNECTIVITY, APPROXIMATE_TIE_BREAK, NMAX),
    )
    native_values = [
        (row.mapped_area, row.bbox_original, row.padded_bbox_original)
        for row in localization.retained
    ]
    direct_values = [
        (row.component_area, row.bbox, row.padded_bbox) for row in direct.retained
    ]
    if native_values != direct_values:
        raise RuntimeError("native-CC remap differs from frozen backprojected Algorithm1")
    direct_pre_cap = len(direct.retained) + sum(
        row.reason == "top_10_limit" for row in direct.rejected
    )
    if direct_pre_cap != len(localization.candidates_before_cap):
        raise RuntimeError("native-CC area filter differs from frozen Algorithm1")


def _evaluate_one(
    *, sample_id: str, probability: np.ndarray, threshold: float,
    width: int, height: int, word_boxes: tuple[tuple[int, int, int, int], ...],
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask = np.ascontiguousarray((probability >= threshold).astype(np.uint8))
    localized = localize_native_components(
        mask, output_width=width, output_height=height,
        min_area=MIN_AREA, pad=PAD, connectivity=CONNECTIVITY, nmax=NMAX,
    )
    projected = backproject_binary_mask(mask, (height, width))
    x_cells = nearest_axis_cells(128, width)
    y_cells = nearest_axis_cells(128, height)
    raw_by_id = {row.component_id: row for row in localized.raw_components}
    ids_by_word = [
        component_ids_intersecting_original_box(
            localized.labels_native, box, x_cells=x_cells, y_cells=y_cells,
        )
        for box in word_boxes
    ]
    component_to_words: dict[int, set[int]] = {
        row.component_id: set() for row in localized.raw_components
    }
    for word_index, ids in enumerate(ids_by_word, start=1):
        for component_id in ids:
            component_to_words[component_id].add(word_index)

    proposals = []
    fusion_pairs: set[tuple[int, int]] = set()
    wordness = Counter()
    for rank, component in enumerate(localized.retained, start=1):
        box = component.padded_bbox_original
        significant = tuple(
            index for index, word_box in enumerate(word_boxes, start=1)
            if _intersection(box, word_box) / max(1, _area(word_box)) >= 0.25
        )
        for left_offset, left in enumerate(significant):
            for right in significant[left_offset + 1:]:
                fusion_pairs.add((left, right))
        best_iou = max((_box_iou(box, word_box) for word_box in word_boxes), default=0.0)
        category = (
            "MULTIWORD_MERGE" if len(significant) >= 2 else
            "WORD_GOOD" if best_iou >= 0.50 else
            "WORD_PARTIAL" if len(significant) == 1 else "UNMATCHED"
        )
        wordness[category] += 1
        proposals.append({
            "rank": rank,
            "component_id": component.component_id,
            "mapped_area": component.mapped_area,
            "bbox_native": list(component.bbox_native),
            "bbox_original": list(component.bbox_original),
            "padded_bbox_original": list(box),
            "significant_gt_word_indices": list(significant),
            "wordness": category,
        })

    pre_cap_by_id = {row.component_id: row for row in localized.candidates_before_cap}
    retained_by_id = {row.component_id: row for row in localized.retained}
    per_word = []
    failure_counts = Counter()
    topology = Counter()
    best_ious = []
    for word_index, (word_box, raw_ids) in enumerate(zip(word_boxes, ids_by_word), start=1):
        component_count = len(raw_ids)
        topology[
            "no_foreground" if component_count == 0 else
            "one_component" if component_count == 1 else
            "two_components" if component_count == 2 else "three_or_more_components"
        ] += 1
        ranked = sorted(
            proposals,
            key=lambda row: (-_box_iou(row["padded_bbox_original"], word_box), row["rank"]),
        )
        best = ranked[0] if ranked else None
        best_iou = (
            _box_iou(best["padded_bbox_original"], word_box) if best is not None else 0.0
        )
        best_ious.append(best_iou)
        if best_iou >= 0.50:
            failure = "SUCCESS"
        else:
            region_target = target[word_box[1]:word_box[3], word_box[0]:word_box[2]]
            region_prediction = projected[word_box[1]:word_box[3], word_box[0]:word_box[2]]
            target_pixels = int(region_target.sum())
            foreground_recall = int((region_target & region_prediction).sum()) / target_pixels if target_pixels else 0.0
            raw_boxes = [raw_by_id[value].bbox_original for value in raw_ids]
            oracle_box = _box_union(raw_boxes)
            oracle_iou = _box_iou(oracle_box, word_box) if oracle_box is not None else 0.0
            associated_pre_cap = [value for value in raw_ids if value in pre_cap_by_id]
            pre_cap_valid = any(
                _box_iou(pre_cap_by_id[value].padded_bbox_original, word_box) >= 0.50
                for value in associated_pre_cap
            )
            best_component_words = (
                len(component_to_words.get(int(best["component_id"]), ())) if best else 0
            )
            failure = classify_failure(
                foreground_recall=foreground_recall,
                best_retained_group_iou=best_iou,
                pre_cap_has_valid_group=pre_cap_valid,
                oracle_union_iou=oracle_iou,
                associated_group_count=len(associated_pre_cap),
                best_group_gt_word_count=best_component_words,
            )
        failure_counts[failure] += 1
        per_word.append({
            "sample_id": sample_id,
            "gt_word_index": word_index,
            "gt_bbox": list(word_box),
            "raw_component_count": component_count,
            "raw_component_ids": list(raw_ids),
            "best_iou": best_iou,
            "best_retained_component_id": best["component_id"] if best else None,
            "best_proposal_wordness": best["wordness"] if best else None,
            "failure_category": failure,
        })
    return ({
        "sample_id": sample_id,
        "raw_component_count": len(localized.raw_components),
        "candidate_count_before_area_filter": len(localized.raw_components),
        "candidate_count_after_area_filter_before_cap": len(localized.candidates_before_cap),
        "retained_proposal_count": len(localized.retained),
        "topology": dict(topology),
        "wordness": dict(wordness),
        "fusion_pairs": [list(value) for value in sorted(fusion_pairs)],
        "raw_components_touching_multiple_gt_words": sum(
            len(words) >= 2 for words in component_to_words.values()
        ),
        "proposals": proposals,
        "per_word": per_word,
        "best_ious": best_ious,
        "failure_counts": dict(failure_counts),
    }, {
        "mask": mask,
        "localization": localized,
    })


def _aggregate(rows: list[dict[str, Any]], word_count: int) -> dict[str, Any]:
    best = [value for row in rows for value in row["best_ious"]]
    topology = Counter()
    wordness = Counter()
    failures = Counter()
    fusion_pairs = set()
    for row in rows:
        topology.update(row["topology"])
        wordness.update(row["wordness"])
        failures.update(row["failure_counts"])
        fusion_pairs.update(
            (row["sample_id"], pair[0], pair[1]) for pair in row["fusion_pairs"]
        )
    proposals_total = sum(row["retained_proposal_count"] for row in rows)
    per_image = [
        (
            [proposal["padded_bbox_original"] for proposal in row["proposals"]],
            [word["gt_bbox"] for word in row["per_word"]],
            row["candidate_count_after_area_filter_before_cap"], 0, 0,
        )
        for row in rows
    ]
    geometry = proposal_geometry(per_image).as_dict()
    if len(best) != word_count or geometry["gt_words"] != word_count:
        raise RuntimeError("threshold aggregate GT cardinality mismatch")
    return {
        "gt_words": word_count,
        "single_cc_gt_words": topology["one_component"],
        "topology": {
            name: topology[name] for name in (
                "no_foreground", "one_component", "two_components", "three_or_more_components"
            )
        },
        "iou_ge_0_25": sum(value >= 0.25 for value in best),
        "iou_ge_0_50": sum(value >= 0.50 for value in best),
        "iou_ge_0_75": sum(value >= 0.75 for value in best),
        "iou_ge_0_90": sum(value >= 0.90 for value in best),
        "median_best_iou": float(median(best)),
        "WORD_GOOD": wordness["WORD_GOOD"],
        "WORD_PARTIAL": wordness["WORD_PARTIAL"],
        "MULTIWORD_MERGE": wordness["MULTIWORD_MERGE"],
        "UNMATCHED": wordness["UNMATCHED"],
        "undermerge": failures["GROUPING_UNDERMERGE"],
        "overmerge": failures["GROUPING_OVERMERGE"],
        "failure_breakdown": dict(failures),
        "retained_proposals": proposals_total,
        "proposals_per_image": proposals_total / len(rows),
        "images_over_10_candidates_before_cap": sum(
            row["candidate_count_after_area_filter_before_cap"] > 10 for row in rows
        ),
        "proposal_precision_iou_ge_0_50": geometry["proposal_precision_iou_ge_0_50"],
        "proposal_recall_iou_ge_0_50": geometry["recall_iou_ge_0_50"],
        "proposal_f1_iou_ge_0_50": geometry["f1_proposal_level_iou_ge_0_50"],
        "raw_components_touching_multiple_gt_words": sum(
            row["raw_components_touching_multiple_gt_words"] for row in rows
        ),
        "unique_gt_word_pairs_fused_by_retained_crops": len(fusion_pairs),
    }


def _crop_manifest(
    *, threshold: float, prepared: list[dict[str, Any]], output: Path,
    probability_manifest: Path,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite crop artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="threshold-word-crops-", dir=output.parent))
    samples = []
    try:
        crops_root = staging / "crops"
        crops_root.mkdir()
        for image_index, item in enumerate(prepared, 1):
            with Image.open(item["image_path"]) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            mask = np.ascontiguousarray((item["probability"] >= threshold).astype(np.uint8))
            localized = localize_native_components(
                mask, output_width=rgb.shape[1], output_height=rgb.shape[0],
                min_area=MIN_AREA, pad=PAD, connectivity=CONNECTIVITY, nmax=NMAX,
            )
            sample_dir = crops_root / f"{int(item['sample_id']):04d}"
            sample_dir.mkdir()
            crops = []
            for rank, component in enumerate(localized.retained, 1):
                x1, y1, x2, y2 = component.padded_bbox_original
                path = sample_dir / f"crop-{rank:02d}.png"
                Image.fromarray(rgb[y1:y2, x1:x2], mode="RGB").save(path)
                crops.append({
                    "candidate_id": f"{int(item['sample_id']):04d}:{rank:02d}",
                    "rank": rank,
                    "component_id": component.component_id,
                    "mapped_component_area": component.mapped_area,
                    "bbox_native": list(component.bbox_native),
                    "bbox_original": list(component.bbox_original),
                    "padded_bbox_original": list(component.padded_bbox_original),
                    "crop_path": str(path.relative_to(staging)),
                    "crop_sha256": _sha_file(path),
                })
            samples.append({
                "sample_id": item["sample_id"],
                "image_sha256": item["image_sha256"],
                "crop_count": len(crops),
                "crops": crops,
            })
            if image_index % 25 == 0 or image_index == 229:
                print(f"[selected-threshold crop export/no OCR] {image_index}/229", flush=True)
        report = {
            "schema_version": 1,
            "artifact": "paper-like-probability-threshold-train-crops-v1",
            "classification": "APPROXIMATE_DIAGNOSTIC_CROP_CACHE",
            "split": "TRAIN",
            "sample_count": 229,
            "test_loaded": False,
            "ocr_called": False,
            "selected_threshold": threshold,
            "selection_rule": [
                "maximize WORD_GOOD", "then minimize MULTIWORD_MERGE",
                "then minimize overmerge", "then prefer the higher threshold",
            ],
            "algorithm1_contract": algorithm1_contract(),
            "probability_manifest": str(probability_manifest),
            "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
            "samples": samples,
        }
        manifest = staging / "manifest.json"
        manifest.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return output / "manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args(); cv2.setNumThreads(1)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    probability_manifest = args.probability_manifest.resolve()
    train_manifest = args.train_manifest.resolve()
    v1_path = args.v1_calibration.resolve()
    if _sha_file(probability_manifest) != EXPECTED_PROBABILITY_MANIFEST_SHA256:
        raise RuntimeError("TRAIN probability manifest SHA256 mismatch")
    if _sha_file(train_manifest) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("TRAIN split manifest SHA256 mismatch")
    if _sha_file(v1_path) != EXPECTED_V1_SHA256:
        raise RuntimeError("frozen practical-remap-v1 report SHA256 mismatch")
    probability_report = json.loads(probability_manifest.read_text(encoding="utf-8"))
    v1_report = json.loads(v1_path.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(train_manifest)
    if (
        probability_report.get("sample_count") != 229
        or probability_report.get("test_data_loaded") is not False
        or probability_report.get("gt_loaded") is not False
        or probability_report.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or probability_report.get("reference_mask_cache", {}).get("sha256") != EXPECTED_MASK_MANIFEST_SHA256
        or probability_report.get("reference_mask_cache", {}).get("all_thresholded_maps_match") is not True
        or probability_report.get("train_manifest", {}).get("sha256") != EXPECTED_TRAIN_MANIFEST_SHA256
        or manifest.split != EXPECTED_SPLIT or manifest.split_role != "train"
        or len(manifest.samples) != 229
    ):
        raise RuntimeError("requires uncontaminated frozen TRAIN-229 probabilities")
    v1_geometry = v1_report["calibration"]["selected"]["geometry"]
    if (
        v1_geometry["gt_words"] != 849
        or v1_geometry["gt_with_proposal_iou_ge_0_50"] != BASELINE_V1["iou_ge_0_50"]
    ):
        raise RuntimeError("frozen practical-remap-v1 baseline mismatch")

    probability_by_id = {
        str(row["sample_id"]): row for row in probability_report["samples"]
    }
    source_by_id = {
        str(row["sample_id"]): row for row in v1_report["train_source"]["samples"]
    }
    expected_ids = {sample.sample_id for sample in manifest.samples}
    if set(probability_by_id) != expected_ids or set(source_by_id) != expected_ids:
        raise RuntimeError("TRAIN source populations differ")

    threshold_rows: dict[float, list[dict[str, Any]]] = {value: [] for value in THRESHOLDS}
    prepared: list[dict[str, Any]] = []
    gt_words = 0
    algorithm1_identity_samples = 0
    for image_index, sample in enumerate(manifest.samples, 1):
        probability_record = probability_by_id[sample.sample_id]
        source = source_by_id[sample.sample_id]
        probability = _load_probability(probability_manifest.parent, probability_record)
        image_path = _contained(args.data_root.resolve(), sample.image_path)
        target_path = _contained(args.data_root.resolve(), sample.mask_path)
        annotation_path = _contained(args.data_root.resolve(), sample.validity_annotation_path)
        if (
            _sha_file(image_path) != probability_record["image_sha256"]
            or _sha_file(image_path) != source["image_sha256"]
            or _sha_file(target_path) != source["mask_sha256"]
            or _sha_file(annotation_path) != source["annotation_sha256"]
        ):
            raise RuntimeError(f"TRAIN source SHA mismatch: {sample.sample_id}")
        with Image.open(image_path) as opened:
            width, height = opened.size
        with Image.open(target_path) as opened:
            target = np.ascontiguousarray(
                (np.asarray(icdar13_task22_rgb_to_binary_mask(opened)) > 0).astype(np.uint8)
            )
        word_boxes = tuple(
            row.bbox for row in train_task22_word_boxes(target_path, annotation_path)
        )
        if len(word_boxes) != source["word_box_count"]:
            raise RuntimeError(f"TRAIN GT word count mismatch: {sample.sample_id}")
        gt_words += len(word_boxes)
        for threshold in THRESHOLDS:
            row, details = _evaluate_one(
                sample_id=sample.sample_id, probability=probability, threshold=threshold,
                width=width, height=height, word_boxes=word_boxes, target=target,
            )
            if threshold == 0.50:
                if _sha_array(details["mask"]) != probability_record["reference_native_mask_array_sha256"]:
                    raise RuntimeError(f"threshold-.50 native-mask identity mismatch: {sample.sample_id}")
                # For ordinary upsampling, native CC remap and the frozen
                # backproject-then-CC route are provably identical and are
                # checked byte-for-byte. If an original axis is below 128,
                # nearest downsampling can make two native labels touch; the
                # task explicitly fixes CC before remap, so that image remains
                # in the requested native-CC path but not in this equivalence check.
                if width >= 128 and height >= 128:
                    _validate_algorithm1_identity(
                        details["mask"], details["localization"], width=width, height=height,
                    )
                    algorithm1_identity_samples += 1
            threshold_rows[threshold].append(row)
        prepared.append({
            "sample_id": sample.sample_id,
            "image_path": image_path,
            "image_sha256": probability_record["image_sha256"],
            "probability": probability,
        })
        if image_index % 10 == 0 or image_index == 229:
            print(f"[threshold connectivity CPU] {image_index}/229", flush=True)
    if gt_words != 849:
        raise RuntimeError(f"expected 849 TRAIN GT words, got {gt_words}")

    aggregates = {
        threshold: _aggregate(rows, gt_words) for threshold, rows in threshold_rows.items()
    }
    baseline_words = [word for row in threshold_rows[0.50] for word in row["per_word"]]
    baseline_pairs = {
        (row["sample_id"], pair[0], pair[1])
        for row in threshold_rows[0.50] for pair in row["fusion_pairs"]
    }
    transitions = {}
    for threshold in THRESHOLDS:
        words = [word for row in threshold_rows[threshold] for word in row["per_word"]]
        pairs = {
            (row["sample_id"], pair[0], pair[1])
            for row in threshold_rows[threshold] for pair in row["fusion_pairs"]
        }
        fragmented_to_single = [
            current for base, current in zip(baseline_words, words)
            if base["raw_component_count"] >= 2 and current["raw_component_count"] == 1
        ]
        transitions[threshold] = {
            "fragmented_at_0_50_becomes_single_cc": len(fragmented_to_single),
            "fragmented_to_single_and_iou_ge_0_50": sum(
                row["best_iou"] >= 0.50 for row in fragmented_to_single
            ),
            "fragmented_to_single_and_WORD_GOOD": sum(
                row["best_iou"] >= 0.50 and row["best_proposal_wordness"] == "WORD_GOOD"
                for row in fragmented_to_single
            ),
            "baseline_iou_correct_becomes_MULTIWORD": sum(
                base["best_iou"] >= 0.50
                and base["best_proposal_wordness"] != "MULTIWORD_MERGE"
                and current["best_proposal_wordness"] == "MULTIWORD_MERGE"
                for base, current in zip(baseline_words, words)
            ),
            "new_unique_gt_word_pairs_fused_vs_0_50": len(pairs - baseline_pairs),
            "resolved_0_50_fusion_pairs": len(baseline_pairs - pairs),
        }

    selected_threshold = min(
        THRESHOLDS,
        key=lambda value: (
            -aggregates[value]["WORD_GOOD"],
            aggregates[value]["MULTIWORD_MERGE"],
            aggregates[value]["overmerge"],
            -value,
        ),
    )
    crop_manifest = _crop_manifest(
        threshold=selected_threshold, prepared=prepared,
        output=args.crop_output_dir.resolve(), probability_manifest=probability_manifest,
    )
    crop_sha = _sha_file(crop_manifest)
    table = []
    for threshold in THRESHOLDS:
        aggregate = aggregates[threshold]
        table.append({
            "threshold": threshold,
            **aggregate,
            "transitions_from_0_50": transitions[threshold],
        })
    report = {
        "schema_version": 1,
        "mode": "PROBABILITY_THRESHOLD_WORD_CONNECTIVITY_TRAIN_V1",
        "classification": "APPROXIMATE_DIAGNOSTIC",
        "table_5_claim": False,
        "scientific_boundary": {
            "execution_device": "cpu", "cuda_used": False,
            "segmentation_rerun": False, "test_loaded": False,
            "training": False, "ocr_parseq": False,
            "grouping": False, "morphology": False,
            "gt_used_only_for_posthoc_geometry_and_threshold_diagnostic": True,
        },
        "question": (
            "whether a threshold band near 0.5 naturally connects word components "
            "without merging distinct words; not an F1-maximization study"
        ),
        "thresholds_predeclared": list(THRESHOLDS),
        "algorithm1_contract": algorithm1_contract(),
        "algorithm1_identity_check_at_0_50": {
            "samples_verified": algorithm1_identity_samples,
            "samples_not_equivalence_checked_due_to_original_axis_below_128": 229 - algorithm1_identity_samples,
            "reference": "native CC remap equals nearest-backprojected frozen Algorithm1",
            "passed": True,
        },
        "practical_remap_v1_frozen_comparator": BASELINE_V1,
        "selection_rule": {
            "primary": "most WORD_GOOD retained proposals",
            "tie_1": "fewer MULTIWORD_MERGE retained proposals",
            "tie_2": "fewer per-word GROUPING_OVERMERGE failures",
            "tie_3": "higher threshold (conservative deterministic tie-break)",
            "selected_threshold": selected_threshold,
        },
        "selected_crop_manifest": {
            "path": str(crop_manifest), "sha256": crop_sha,
        },
        "provenance": {
            "probability_manifest": str(probability_manifest),
            "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
            "reference_mask_manifest_sha256": EXPECTED_MASK_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": EXPECTED_TRAIN_MANIFEST_SHA256,
            "v1_calibration": str(v1_path),
            "v1_calibration_sha256": EXPECTED_V1_SHA256,
        },
        "source_sha256": {
            str(Path(__file__).resolve()): _sha_file(Path(__file__).resolve()),
            str(Path(__file__).resolve().parents[1] / "localize/probability_threshold_word_connectivity.py"):
                _sha_file(Path(__file__).resolve().parents[1] / "localize/probability_threshold_word_connectivity.py"),
            str(Path(__file__).resolve().parents[1] / "localize/mask_to_blocks.py"):
                _sha_file(Path(__file__).resolve().parents[1] / "localize/mask_to_blocks.py"),
        },
        "threshold_table": table,
        "per_threshold_samples": {
            str(threshold): rows for threshold, rows in threshold_rows.items()
        },
    }
    _write_exclusive(output, report)
    print(json.dumps({
        "threshold_table": table,
        "selected_threshold": selected_threshold,
        "crop_manifest": str(crop_manifest),
        "crop_manifest_sha256": crop_sha,
    }, indent=2, sort_keys=True), flush=True)
    print(f"JSON {output}", flush=True)
    print(f"SHA256 {_sha_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
