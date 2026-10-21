#!/usr/bin/env python3
"""Analyze frozen practical-remap-v1 failures on ICDAR13 train-229 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.practical_remap_failure_analysis import analyze_train_failures
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _write_csv(per_word: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_id", "gt_word_index", "gt_bbox",
        "segmentation_foreground_recall_in_gt_word", "best_raw_component_iou",
        "best_practical_v1_group_iou", "oracle_component_union_iou",
        "raw_component_count_associated", "practical_group_count_associated",
        "best_practical_group_gt_word_count", "failure_category",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for value in per_word:
            row = {key: value.get(key) for key in fieldnames}
            row["gt_bbox"] = json.dumps(row["gt_bbox"], separators=(",", ":"))
            writer.writerow(row)


def _selected_parameters(calibration: dict[str, Any]) -> GroupedWordBlockParams:
    guard = calibration.get("train_only_guard", {})
    boundary = calibration.get("scientific_boundary", {})
    if (
        guard.get("split_role") != "train"
        or guard.get("sample_count") != 229
        or guard.get("test_data_loaded") is not False
        or boundary.get("selection_used_test") is not False
        or boundary.get("selection_used_ocr") is not False
    ):
        raise ValueError("Practical-remap configuration was not selected exclusively on train-229")
    return GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-calibration", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output, csv_output = args.output.resolve(), args.csv_output.resolve()
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite a frozen train failure-analysis report")
    calibration_path = args.train_calibration.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    params = _selected_parameters(calibration)
    segmenter = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            evaluation_config=args.evaluation_config.resolve(),
            checkpoint=args.segmentation_checkpoint.resolve(),
            device=args.device,
        )
    )
    checkpoint = dict(segmenter.checkpoint_record)
    try:
        report = analyze_train_failures(
            train_manifest_path=args.train_manifest.resolve(),
            data_root=args.data_root.resolve(),
            params=params,
            predict=lambda image: segmenter(image)["mask"],
        )
    finally:
        release = segmenter.release()
    report["frozen_segmentation"] = {
        "evaluation_config": str(args.evaluation_config.resolve()),
        "checkpoint": checkpoint,
        "device": args.device,
        "hardware_deviation": "CPU failure analysis" if args.device == "cpu" else None,
        "release": release,
    }
    report["practical_remap_v1_provenance"] = {
        "calibration_path": str(calibration_path),
        "calibration_sha256": _sha256(calibration_path),
        "selected_parameters": calibration["calibration"]["selected"]["parameters"],
        "selection_reused_not_repeated": True,
    }
    _write_json_atomic(report, output)
    _write_csv(report["per_word"], csv_output)
    print(output)
    print(csv_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
