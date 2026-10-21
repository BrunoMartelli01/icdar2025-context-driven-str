#!/usr/bin/env python3
"""Calibrate practical-remap-v1 geometry exclusively on ICDAR13 train-229."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from localize.practical_remap_protocol import calibrate_on_train, train_records_from_manifest
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from segmentation.data import load_segmentation_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite calibration report")
    manifest = load_segmentation_manifest(args.train_manifest)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("Calibration permits only the author-protocol 229-image train manifest")
    segmenter = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            evaluation_config=args.evaluation_config.resolve(),
            checkpoint=args.segmentation_checkpoint.resolve(),
            device=args.device,
        )
    )
    try:
        records, source = train_records_from_manifest(
            manifest, args.data_root.resolve(), lambda image: segmenter(image)["mask"]
        )
    finally:
        release = segmenter.release()
    calibration = calibrate_on_train(records)
    report = {
        "schema_version": 1,
        "mode": "practical-remap-v1",
        "classification": "PRACTICAL_APPROXIMATION",
        "train_only_guard": {
            "manifest": str(args.train_manifest.resolve()),
            "manifest_sha256": _sha256(args.train_manifest.resolve()),
            "split_role": manifest.split_role,
            "sample_count": len(manifest.samples),
            "test_data_loaded": False,
            "ocr_called": False,
        },
        "frozen_segmentation": {
            "evaluation_config": str(args.evaluation_config.resolve()),
            "checkpoint": dict(segmenter.checkpoint_record),
            "device": args.device,
            "hardware_deviation": "CPU calibration" if args.device == "cpu" else None,
            "release": release,
        },
        "train_source": source,
        "calibration": calibration,
        "scientific_boundary": {
            "prohibition": "The selected graph parameters are a practical approximation, not paper Algorithm 1.",
            "selection_used_test": False,
            "selection_used_ocr": False,
        },
    }
    _write_atomic(report, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
