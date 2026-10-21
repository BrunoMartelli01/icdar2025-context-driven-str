#!/usr/bin/env python3
"""TRAIN-only original-gap post-split sweep for frozen RLSA run/height=0.25."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from localize.gap_word_regions import linewise_rlsa
from localize.rlsa_gap_postsplit import RLSAGapPostSplitParams, rlsa_original_gap_postsplit
from scripts.diagnose_paperlike_gap_grouping_train import _load_record
from scripts.diagnose_paperlike_localization_classic_train import _to_original_groups
from scripts.evaluate_supervised_word_boundary_train import (
    EXPECTED_CACHE_SHA256,
    _evaluate_groups,
    _sha256_file,
    _summary,
)
from segmentation.data import load_segmentation_manifest


FROZEN_RLSA_RUN_HEIGHT_RATIO = 0.25
MAX_GAP_TO_MEDIAN_GAPS = (2.0, 2.5, 3.0)
MAX_GAP_TO_MEDIAN_HEIGHTS = (0.30, 0.40, 0.50)
GRID_CONFIGURATION_COUNT = 9
GO_MAX_OVERMERGE = 45


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _transitions(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_by_key = {
        (value["sample_id"], value["gt_word_index"]): value["failure_category"]
        for value in before["per_word"]
    }
    after_by_key = {
        (value["sample_id"], value["gt_word_index"]): value["failure_category"]
        for value in after["per_word"]
    }
    if before_by_key.keys() != after_by_key.keys():
        raise RuntimeError("RLSA/post-split word populations differ")
    keys = tuple(before_by_key)
    return {
        "rlsa_overmerge_resolved": sum(
            before_by_key[key] == "GROUPING_OVERMERGE"
            and after_by_key[key] != "GROUPING_OVERMERGE"
            for key in keys
        ),
        "rlsa_overmerge_recovered_to_success": sum(
            before_by_key[key] == "GROUPING_OVERMERGE"
            and after_by_key[key] == "SUCCESS"
            for key in keys
        ),
        "new_undermerge_introduced": sum(
            before_by_key[key] != "GROUPING_UNDERMERGE"
            and after_by_key[key] == "GROUPING_UNDERMERGE"
            for key in keys
        ),
        "rlsa_undermerge_recovered_to_success": sum(
            before_by_key[key] == "GROUPING_UNDERMERGE"
            and after_by_key[key] == "SUCCESS"
            for key in keys
        ),
    }


def _go(row: dict[str, Any], v1: dict[str, Any]) -> bool:
    return (
        row["summary"]["undermerge"] < v1["undermerge"]
        and row["summary"]["overmerge"] <= GO_MAX_OVERMERGE
        and row["summary"]["f1_iou_ge_0_50"] >= v1["f1_iou_ge_0_50"]
        and row["wordness"]["WORD_GOOD"] >= v1["wordness"]["WORD_GOOD"]
    )


def _best(rows: list[dict[str, Any]], v1: dict[str, Any]) -> dict[str, Any]:
    promotable = [row for row in rows if row["go"]]
    if promotable:
        return max(promotable, key=lambda row: (
            row["summary"]["f1_iou_ge_0_50"],
            row["wordness"]["WORD_GOOD"],
            -row["summary"]["overmerge"],
            -row["summary"]["undermerge"],
        ))
    maintains_undermerge_advantage = [
        row for row in rows if row["summary"]["undermerge"] < v1["undermerge"]
    ]
    pool = maintains_undermerge_advantage or rows
    return min(pool, key=lambda row: (
        row["summary"]["overmerge"],
        -row["summary"]["f1_iou_ge_0_50"],
        -row["wordness"]["WORD_GOOD"],
        row["summary"]["undermerge"],
    ))


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
        raise ValueError("RLSA post-split permits only official TRAIN-229")

    gap_report = json.loads(args.gap_report.read_text(encoding="utf-8"))
    v1 = gap_report["baseline_v1"]
    frozen_rlsa_source = gap_report["best_by_technique"]["RLSA"]
    if (
        frozen_rlsa_source["parameters"] != {"run_height_ratio": 0.25}
        or frozen_rlsa_source["summary"]["undermerge"] != 74
        or frozen_rlsa_source["summary"]["overmerge"] != 95
        or frozen_rlsa_source["wordness"]["WORD_GOOD"] != 509
    ):
        raise RuntimeError("frozen RLSA source does not match the requested baseline")

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

    rlsa_native = [
        linewise_rlsa(mask, FROZEN_RLSA_RUN_HEIGHT_RATIO) for mask in masks
    ]
    rlsa_groups = [
        _to_original_groups(result, record.width, record.height)
        for result, record in zip(rlsa_native, records)
    ]
    rlsa_evaluation = _evaluate_groups(
        records, rlsa_groups, evidence_attribute="supervised_word_evidence"
    )
    if (
        _summary(rlsa_evaluation) != frozen_rlsa_source["summary"]
        or rlsa_evaluation["wordness"] != frozen_rlsa_source["wordness"]
    ):
        raise RuntimeError("recomputed frozen RLSA metrics drifted")

    rows = []
    for median_ratio in MAX_GAP_TO_MEDIAN_GAPS:
        for height_ratio in MAX_GAP_TO_MEDIAN_HEIGHTS:
            params = RLSAGapPostSplitParams(median_ratio, height_ratio)
            results = [rlsa_original_gap_postsplit(mask, params) for mask in masks]
            groups = [
                _to_original_groups(result.groups, record.width, record.height)
                for result, record in zip(results, records)
            ]
            evaluation = _evaluate_groups(
                records, groups, evidence_attribute="supervised_word_evidence"
            )
            row = {
                "parameters": {
                    "max_gap_to_median_internal_gap": median_ratio,
                    "max_gap_to_median_component_height": height_ratio,
                    "gate_combination": "AND",
                    "fallback": "do_not_split_if_fewer_than_two_gaps_or_median_gap_zero",
                    "recursive": True,
                },
                "summary": _summary(evaluation),
                "wordness": evaluation["wordness"],
                "split_diagnostics": {
                    "initial_rlsa_blobs": sum(value.initial_rlsa_blobs for value in results),
                    "initial_multicomponent_rlsa_blobs": sum(
                        value.initial_multicomponent_blobs for value in results
                    ),
                    "rlsa_blobs_split": sum(value.rlsa_blobs_split for value in results),
                    "total_splits": sum(value.total_splits for value in results),
                    **_transitions(rlsa_evaluation, evaluation),
                },
            }
            row["go"] = _go(row, v1)
            rows.append(row)
            print(
                "[grid] median_ratio={} height_ratio={} F1={:.6f} under={} over={} "
                "good={} blobs_split={} total_splits={}".format(
                    median_ratio, height_ratio,
                    row["summary"]["f1_iou_ge_0_50"],
                    row["summary"]["undermerge"], row["summary"]["overmerge"],
                    row["wordness"]["WORD_GOOD"],
                    row["split_diagnostics"]["rlsa_blobs_split"],
                    row["split_diagnostics"]["total_splits"],
                ),
                flush=True,
            )
    if len(rows) != GRID_CONFIGURATION_COUNT:
        raise RuntimeError("post-split grid cardinality drifted")
    best = _best(rows, v1)
    report = {
        "schema_version": 1,
        "mode": "rlsa-original-gap-postsplit-train-v1",
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
        "baseline_v1": v1,
        "frozen_rlsa": {
            "run_height_ratio": FROZEN_RLSA_RUN_HEIGHT_RATIO,
            "summary": _summary(rlsa_evaluation),
            "wordness": rlsa_evaluation["wordness"],
        },
        "declared_grid": {
            "max_gap_to_median_internal_gap": MAX_GAP_TO_MEDIAN_GAPS,
            "max_gap_to_median_component_height": MAX_GAP_TO_MEDIAN_HEIGHTS,
            "gate_combination": "AND",
            "configuration_count": GRID_CONFIGURATION_COUNT,
        },
        "selection_rule_declared_before_execution": {
            "promote": "GO candidates ranked by F1, WORD_GOOD, fewer over/under",
            "fallback_if_no_GO": "among variants retaining undermerge<v1, minimize overmerge then maximize F1 and WORD_GOOD",
        },
        "go_rule": {
            "undermerge": "strictly below v1=92",
            "maximum_overmerge": GO_MAX_OVERMERGE,
            "f1": "greater than or equal to v1",
            "WORD_GOOD": "greater than or equal to v1",
        },
        "grid_results": rows,
        "best": best,
        "decision": "GO" if best["go"] else "NO-GO",
        "provenance": {
            "gap_report": str(args.gap_report.resolve()),
            "gap_report_sha256": _sha256_file(args.gap_report.resolve()),
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
