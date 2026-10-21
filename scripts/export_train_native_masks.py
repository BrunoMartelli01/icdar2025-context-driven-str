#!/usr/bin/env python3
"""Export frozen author-protocol TRAIN-229 native masks for CPU diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from segmentation.data import load_segmentation_manifest


EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
EXPECTED_SPLIT = "official-training-all-229-author-protocol"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite native-mask cache: {output}")
    checkpoint = args.checkpoint.resolve()
    if _sha_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen segmentation checkpoint SHA256 mismatch")
    manifest_path = args.train_manifest.resolve()
    manifest = load_segmentation_manifest(manifest_path)
    if (
        manifest.split != EXPECTED_SPLIT
        or manifest.split_role != "train"
        or len(manifest.samples) != 229
    ):
        raise ValueError("native-mask export requires author-protocol TRAIN-229")

    staging = Path(tempfile.mkdtemp(prefix="train-native-masks-", dir=output.parent))
    masks_dir = staging / "masks"
    masks_dir.mkdir()
    segmenter = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            args.evaluation_config.resolve(), checkpoint, args.device
        )
    )
    records = []
    try:
        for index, sample in enumerate(manifest.samples, start=1):
            image_path = (args.data_root.resolve() / sample.image_path).resolve()
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            result = segmenter(rgb)
            native = result["mask_native"]
            if native.shape != (128, 128) or native.dtype != np.uint8:
                raise RuntimeError(f"invalid native mask for sample {sample.sample_id}")
            mask_path = masks_dir / f"{int(sample.sample_id):04d}.png"
            Image.fromarray(native * 255, mode="L").save(mask_path)
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "image_sha256": _sha_file(image_path),
                    "native_mask_path": str(mask_path.relative_to(staging)),
                    "native_mask_array_sha256": _sha_array(native),
                    "native_mask_file_sha256": _sha_file(mask_path),
                    "foreground_pixels": int(native.sum()),
                    "shape": [128, 128],
                    "dtype": "uint8",
                    "values": [0, 1],
                }
            )
            print(
                f"[native-mask export] {index}/229 sample_id={sample.sample_id} "
                f"foreground={int(native.sum())}",
                flush=True,
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        release = segmenter.release()

    cache_manifest = {
        "schema_version": 1,
        "artifact": "frozen-author-protocol-train-native-masks-epoch0098",
        "classification": "APPROXIMATE_DIAGNOSTIC_CACHE",
        "split": EXPECTED_SPLIT,
        "sample_count": len(records),
        "test_data_loaded": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "evaluation_config": {
            "path": str(args.evaluation_config.resolve()),
            "sha256": _sha_file(args.evaluation_config.resolve()),
        },
        "train_manifest": {
            "path": str(manifest_path),
            "sha256": _sha_file(manifest_path),
        },
        "inference": {
            "device": args.device,
            "native_shape": [128, 128],
            "activation": "sigmoid",
            "threshold": 0.5,
            "comparison": "greater_than_or_equal",
            "backprojection_applied": False,
            "release": release,
        },
        "samples": records,
    }
    manifest_output = staging / "manifest.json"
    manifest_output.write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    final_manifest = output / "manifest.json"
    print(final_manifest, flush=True)
    print(_sha_file(final_manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
