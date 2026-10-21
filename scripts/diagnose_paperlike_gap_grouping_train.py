#!/usr/bin/env python3
"""TRAIN-only RLSA/Otsu/2-means gap-grouping diagnostic from cached masks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from localize.gap_word_regions import line_gap_segmentation, linewise_rlsa
from localize.practical_remap_protocol import train_task22_word_boxes
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.diagnose_paperlike_localization_classic_train import _to_original_groups
from scripts.evaluate_supervised_word_boundary_train import (
    EXPECTED_CACHE_SHA256,
    WordEvidence,
    _evaluate_groups,
    _raw_boxes_from_labels,
    _sha256_file,
    _summary,
    _word_evidence,
)
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest


RLSA_RUN_HEIGHT_RATIOS = (0.15, 0.25, 0.35, 0.50, 0.70)
OTSU_FALLBACK_NORMALIZED_GAP = 0.35
TWO_MEANS_MINIMUM_SEPARATIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
GRID_CONFIGURATION_COUNT = 11
MAX_OVERMERGE = 45
MAX_MULTIWORD = 46


@dataclass(frozen=True)
class ClassicTrainRecord:
    sample_id: str
    width: int
    height: int
    word_boxes: tuple[tuple[int, int, int, int], ...]
    supervised_word_evidence: tuple[WordEvidence, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--supervised-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_record(
    sample_id: str,
    mask: np.ndarray,
    image_path: Path,
    target_path: Path,
    annotation_path: Path,
) -> ClassicTrainRecord:
    with Image.open(image_path) as opened:
        width, height = opened.size
    with Image.open(target_path) as opened:
        target = (
            np.asarray(icdar13_task22_rgb_to_binary_mask(opened), dtype=np.uint8) > 0
        ).astype(np.uint8)
    word_boxes = tuple(
        word.bbox for word in train_task22_word_boxes(target_path, annotation_path)
    )
    full_prediction = backproject_binary_mask(mask, (height, width))
    component_count, native_labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    full_labels = cv2.resize(
        native_labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    raw_boxes = _raw_boxes_from_labels(full_labels)
    if len(raw_boxes) != component_count - 1:
        raise RuntimeError(f"native component loss during remap: sample {sample_id}")
    evidence = _word_evidence(
        full_labels, raw_boxes, full_prediction, target, word_boxes
    )
    return ClassicTrainRecord(
        sample_id=sample_id,
        width=width,
        height=height,
        word_boxes=word_boxes,
        supervised_word_evidence=evidence,
    )


def _aggregate_line_statistics(values: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = (
        "lines_total", "lines_with_at_least_two_gaps",
        "lines_with_real_two_group_separation", "lines_using_fallback",
        "lines_without_gaps", "boundary_count", "merge_count",
    )
    thresholds = [
        threshold for value in values for threshold in value["_threshold_values"]
    ]
    return {
        **{key: sum(int(value[key]) for value in values) for key in count_keys},
        "median_resulting_threshold": float(median(thresholds)) if thresholds else None,
    }


def _go(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        row["summary"]["recall_iou_ge_0_50"] >= baseline["recall_iou_ge_0_50"]
        and row["wordness"]["WORD_GOOD"] >= baseline["wordness"]["WORD_GOOD"]
        and row["summary"]["overmerge"] <= MAX_OVERMERGE
        and row["wordness"]["MULTIWORD_MERGE"] <= MAX_MULTIWORD
    )


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["go"]),
        row["summary"]["f1_iou_ge_0_50"],
        row["summary"]["recall_iou_ge_0_50"],
        row["wordness"]["WORD_GOOD"],
        -row["summary"]["overmerge"],
        -row["wordness"]["MULTIWORD_MERGE"],
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
        or manifest.split_role != "train"
        or len(manifest.samples) != 229
        or cache.get("sample_count") != 229
        or cache.get("test_data_loaded") is not False
    ):
        raise ValueError("gap grouping diagnostic permits only official TRAIN-229")

    source_report = json.loads(args.supervised_report.read_text(encoding="utf-8"))
    baseline_evaluation = source_report["oof_v1"]
    baseline = {**_summary(baseline_evaluation), "wordness": baseline_evaluation["wordness"]}
    if (
        baseline["iou_ge_0_50"] != 568
        or round(baseline["precision_iou_ge_0_50"], 4) != 0.4722
        or round(baseline["recall_iou_ge_0_50"], 4) != 0.6690
        or round(baseline["f1_iou_ge_0_50"], 4) != 0.5536
        or baseline["undermerge"] != 92
        or baseline["overmerge"] != 36
        or baseline["wordness"]["WORD_GOOD"] != 552
        or baseline["wordness"]["MULTIWORD_MERGE"] != 37
    ):
        raise RuntimeError("frozen v1 baseline mismatch")

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
            or hashlib.sha256(mask.tobytes()).hexdigest()
            != cached["native_mask_array_sha256"]
        ):
            raise RuntimeError(f"mask array mismatch: {sample.sample_id}")
        image_path = (args.data_root / sample.image_path).resolve()
        if _sha256_file(image_path) != cached["image_sha256"]:
            raise RuntimeError(f"source image SHA mismatch: {sample.sample_id}")
        records.append(_load_record(
            sample.sample_id,
            mask,
            image_path,
            (args.data_root / sample.mask_path).resolve(),
            (args.data_root / sample.validity_annotation_path).resolve(),
        ))
        masks.append(mask)
        if index % 10 == 0 or index == 229:
            print(f"[verified cache + prepared] {index}/229", flush=True)

    variants = []
    for ratio in RLSA_RUN_HEIGHT_RATIOS:
        results = [linewise_rlsa(mask, ratio) for mask in masks]
        groups = [
            _to_original_groups(result, record.width, record.height)
            for result, record in zip(results, records)
        ]
        evaluation = _evaluate_groups(
            records, groups, evidence_attribute="supervised_word_evidence"
        )
        row = {
            "technique": "RLSA",
            "parameters": {"run_height_ratio": ratio},
            "summary": _summary(evaluation),
            "wordness": evaluation["wordness"],
        }
        row["go"] = _go(row, baseline)
        variants.append(row)
        print(
            f"[grid] RLSA ratio={ratio:.2f} F1={row['summary']['f1_iou_ge_0_50']:.6f} "
            f"recall={row['summary']['recall_iou_ge_0_50']:.6f} "
            f"under={row['summary']['undermerge']} over={row['summary']['overmerge']}",
            flush=True,
        )

    for method, separations in (
        ("OTSU", (0.0,)),
        ("TWO_MEANS", TWO_MEANS_MINIMUM_SEPARATIONS),
    ):
        for separation in separations:
            results = [
                line_gap_segmentation(
                    mask,
                    method="otsu" if method == "OTSU" else "two_means",
                    fallback_normalized_gap=OTSU_FALLBACK_NORMALIZED_GAP,
                    minimum_cluster_separation=separation,
                )
                for mask in masks
            ]
            groups = [
                _to_original_groups(result.groups, record.width, record.height)
                for result, record in zip(results, records)
            ]
            evaluation = _evaluate_groups(
                records, groups, evidence_attribute="supervised_word_evidence"
            )
            row = {
                "technique": method,
                "parameters": {
                    "fallback_normalized_gap": OTSU_FALLBACK_NORMALIZED_GAP,
                    **(
                        {"minimum_cluster_separation": separation}
                        if method == "TWO_MEANS" else {}
                    ),
                },
                "summary": _summary(evaluation),
                "wordness": evaluation["wordness"],
                "line_statistics": _aggregate_line_statistics(
                    [result.line_statistics for result in results]
                ),
            }
            row["go"] = _go(row, baseline)
            variants.append(row)
            print(
                f"[grid] {method} separation={separation:.2f} "
                f"F1={row['summary']['f1_iou_ge_0_50']:.6f} "
                f"recall={row['summary']['recall_iou_ge_0_50']:.6f} "
                f"under={row['summary']['undermerge']} over={row['summary']['overmerge']}",
                flush=True,
            )

    if len(variants) != GRID_CONFIGURATION_COUNT:
        raise RuntimeError("executed grid cardinality drifted")
    best = {
        technique: max(
            (row for row in variants if row["technique"] == technique),
            key=_selection_key,
        )
        for technique in ("RLSA", "OTSU", "TWO_MEANS")
    }
    report = {
        "schema_version": 1,
        "mode": "paperlike-gap-grouping-train-v1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "train_only": {
            "images": 229,
            "gt_words": 849,
            "test_loaded": False,
            "ocr_called": False,
            "training_executed": False,
            "segmentation_executed": False,
            "device": "cpu",
            "gt_used_in_inference": False,
        },
        "cache": {
            "manifest": str(cache_manifest),
            "manifest_sha256": EXPECTED_CACHE_SHA256,
            "verified_samples": 229,
        },
        "baseline_v1": baseline,
        "declared_grid": {
            "RLSA_run_height_ratios": RLSA_RUN_HEIGHT_RATIOS,
            "OTSU": {"configuration_count": 1},
            "TWO_MEANS_minimum_cluster_separations": TWO_MEANS_MINIMUM_SEPARATIONS,
            "fallback_normalized_gap": OTSU_FALLBACK_NORMALIZED_GAP,
            "total_configurations": GRID_CONFIGURATION_COUNT,
        },
        "fixed_geometry": {
            "native_mask_shape": [128, 128],
            "line_formation": "existing fixed vertical geometry",
            "within_line_order": "raw components left-to-right",
            "bbox": "union of unpadded raw component bboxes",
            "padding": "6 original-image pixels after native-to-original remap",
            "sort": "foreground area descending with deterministic spatial ties",
            "nmax": 10,
        },
        "go_rule": {
            "recall": "greater than or equal to v1",
            "WORD_GOOD": "greater than or equal to v1",
            "maximum_overmerge": MAX_OVERMERGE,
            "maximum_MULTIWORD_MERGE": MAX_MULTIWORD,
        },
        "best_by_technique": best,
        "decisions": {
            technique: "GO" if row["go"] else "NO-GO"
            for technique, row in best.items()
        },
        "grid_results": variants,
        "provenance": {
            "v1_source_report": str(args.supervised_report.resolve()),
            "v1_source_report_sha256": _sha256_file(args.supervised_report.resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    print(output, flush=True)
    print(_sha256_file(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
