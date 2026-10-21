#!/usr/bin/env python3
"""Calibrate practical-remap-v2 only on the frozen ICDAR13 train-229 split."""

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
from localize.practical_remap_v2_protocol import (
    V2_GRID_CONFIGURATION_COUNT,
    calibrate_v2_on_train,
    declared_v2_grid,
    load_v2_train_records,
)
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from segmentation.data import load_segmentation_manifest


EXPECTED_TRAIN_MANIFEST = "manifests/icdar13_segmentation_author_protocol.yaml"
EXPECTED_TRAIN_SPLIT = "official-training-all-229-author-protocol"


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


def _v1_parameters(calibration: dict[str, Any]) -> GroupedWordBlockParams:
    guard = calibration.get("train_only_guard", {})
    boundary = calibration.get("scientific_boundary", {})
    if (
        guard.get("split_role") != "train"
        or guard.get("sample_count") != 229
        or guard.get("test_data_loaded") is not False
        or boundary.get("selection_used_test") is not False
        or boundary.get("selection_used_ocr") is not False
    ):
        raise ValueError("v1 comparison configuration was not selected exclusively on train-229")
    return GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])


def _validate_train_manifest(manifest: Any, manifest_path: Path) -> None:
    """Fail closed unless this is the frozen author-protocol train-229 split."""

    if (
        manifest.split_role != "train"
        or manifest.split != EXPECTED_TRAIN_SPLIT
        or len(manifest.samples) != 229
    ):
        raise ValueError(
            "v2 calibration requires the official author-protocol train-229 manifest "
            f"{EXPECTED_TRAIN_MANIFEST}; got {manifest_path} "
            f"(split={manifest.split!r}, role={manifest.split_role!r}, "
            f"samples={len(manifest.samples)})"
        )


def _write_csv(v1_rows: list[dict[str, Any]], v2_rows: list[dict[str, Any]], path: Path) -> None:
    v2_by_key = {(value["sample_id"], value["gt_word_index"]): value for value in v2_rows}
    rows = []
    for v1 in v1_rows:
        v2 = v2_by_key[(v1["sample_id"], v1["gt_word_index"])]
        rows.append({
            "sample_id": v1["sample_id"],
            "gt_word_index": v1["gt_word_index"],
            "v1_best_group_iou": v1["best_practical_v1_group_iou"],
            "v2_best_group_iou": v2["best_practical_v2_group_iou"],
            "oracle_component_union_iou": v2["oracle_component_union_iou"],
            "v1_failure_category": v1["failure_category"],
            "v2_failure_category": v2["failure_category"],
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
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
        raise FileExistsError("Refusing to overwrite a frozen practical-remap-v2 train report")
    manifest_path = args.train_manifest.resolve()
    manifest = load_segmentation_manifest(manifest_path)
    _validate_train_manifest(manifest, manifest_path)
    v1_path = args.v1_calibration.resolve()
    v1_calibration = json.loads(v1_path.read_text(encoding="utf-8"))
    v1_params = _v1_parameters(v1_calibration)
    grid = declared_v2_grid()
    if len(grid) != V2_GRID_CONFIGURATION_COUNT:
        raise RuntimeError("v2 grid declaration changed unexpectedly")
    print(
        "[setup] train-only samples=229 grid={} device={} checkpoint={}".format(
            len(grid), args.device, args.segmentation_checkpoint.resolve()
        ),
        flush=True,
    )
    segmenter = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            evaluation_config=args.evaluation_config.resolve(),
            checkpoint=args.segmentation_checkpoint.resolve(),
            device=args.device,
        )
    )
    checkpoint = dict(segmenter.checkpoint_record)
    try:
        records, source = load_v2_train_records(
            manifest,
            args.data_root.resolve(),
            lambda image: segmenter(image)["mask"],
            grid[0],
            progress=lambda message: print(message, flush=True),
        )
        print("[train grid] evaluating frozen boundary-rule grid", flush=True)
        calibration = calibrate_v2_on_train(
            records, v1_params, progress=lambda message: print(message, flush=True)
        )
    finally:
        release = segmenter.release()
    report = {
        "schema_version": 1,
        "mode": "practical-remap-v2",
        "classification": "PRACTICAL_APPROXIMATION",
        "train_only_guard": {
            "manifest": str(args.train_manifest.resolve()),
            "manifest_sha256": _sha256(args.train_manifest.resolve()),
            "split_role": manifest.split_role,
            "sample_count": len(manifest.samples),
            "test_data_loaded": False,
            "ocr_called": False,
            "parameter_selection_performed": True,
        },
        "frozen_segmentation": {
            "evaluation_config": str(args.evaluation_config.resolve()),
            "checkpoint": checkpoint,
            "device": args.device,
            "hardware_deviation": "CPU calibration" if args.device == "cpu" else None,
            "release": release,
        },
        "v1_calibration_provenance": {
            "path": str(v1_path), "sha256": _sha256(v1_path),
            "selected_parameters": v1_calibration["calibration"]["selected"]["parameters"],
        },
        "train_source": source,
        "calibration": calibration,
        "scientific_boundary": {
            "selection_used_test": False,
            "selection_used_ocr": False,
            "prohibition": "v2 is a train-calibrated practical approximation; it does not alter Algorithm 1, v1, segmentation, PARSeq, or scoring.",
        },
    }
    _write_json_atomic(report, output)
    _write_csv(calibration["v1_per_word"], calibration["v2_per_word"], csv_output)
    chosen = calibration["selected"]
    print("[selected] {}".format(json.dumps(chosen["parameters"], sort_keys=True)), flush=True)
    print("[summary] {}".format(json.dumps(chosen["geometry"], sort_keys=True)), flush=True)
    print(output, flush=True)
    print(csv_output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
