#!/usr/bin/env python3
"""Export frozen epoch-0098 TRAIN-229 native sigmoid probability maps.

This is a GPU export boundary only.  It neither reads GT masks nor performs any
proposal generation.  Every exported map must reproduce the already verified
native binary-mask cache when thresholded at the frozen inclusive threshold 0.5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import torch

from pipeline.frozen_segmentation import (
    FROZEN_LOGITS_SHAPE,
    FROZEN_THRESHOLD,
    FrozenAuthorProtocolSegmenter,
    FrozenSegmentationConfig,
)
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _preprocess


EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
EXPECTED_REFERENCE_CACHE_SHA256 = "e9a7ce5badfbc3d99885fef5e655b20f7f5f167f29defa2d3f3c7e91c6d5851a"
EXPECTED_TRAIN_MANIFEST_SHA256 = "3aeecf40411326f0721e5d4eea1bf74a9357eab1c9da54a77d0624b6e24f7c14"
EXPECTED_EVALUATION_CONFIG_SHA256 = "bf3a4d69ebb0f0407461b32b4e14b5464ca2f197b82abf8e75d87a4aa08a6fa2"
EXPECTED_SPLIT = "official-training-all-229-author-protocol"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _contained(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("cached artifact path must be relative and contained")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("cached artifact path escapes its cache root")
    return resolved


def _load_reference_mask(cache_root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _contained(cache_root, str(record["native_mask_path"]))
    if _sha_file(path) != record["native_mask_file_sha256"]:
        raise RuntimeError(f"reference native-mask file SHA mismatch: {record['sample_id']}")
    with Image.open(path) as opened:
        pixels = np.asarray(opened.convert("L"))
    if pixels.dtype != np.uint8 or pixels.shape != FROZEN_LOGITS_SHAPE:
        raise RuntimeError(f"invalid reference native mask: {record['sample_id']}")
    if not np.isin(pixels, (0, 255)).all():
        raise RuntimeError(f"non-binary reference native mask: {record['sample_id']}")
    mask = np.ascontiguousarray((pixels > 0).astype(np.uint8))
    if (
        _sha_array(mask) != record["native_mask_array_sha256"]
        or int(mask.sum()) != int(record["foreground_pixels"])
    ):
        raise RuntimeError(f"reference native-mask array mismatch: {record['sample_id']}")
    return mask


def _validate_probability_map(value: np.ndarray) -> np.ndarray:
    probability = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    if probability.shape != FROZEN_LOGITS_SHAPE:
        raise RuntimeError(f"expected native probability map 128x128, got {probability.shape}")
    if not np.isfinite(probability).all():
        raise RuntimeError("native probability map contains non-finite values")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RuntimeError("native probability map contains values outside [0,1]")
    return probability


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-mask-cache", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.device != "cuda:0":
        raise ValueError("this qualified export requires exactly --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command on PAPER")

    checkpoint = args.checkpoint.resolve()
    train_manifest = args.train_manifest.resolve()
    evaluation_config = args.evaluation_config.resolve()
    reference_manifest = args.reference_mask_cache.resolve()
    output = args.output_dir.resolve()
    expected_hashes = (
        (checkpoint, EXPECTED_CHECKPOINT_SHA256, "checkpoint"),
        (train_manifest, EXPECTED_TRAIN_MANIFEST_SHA256, "TRAIN manifest"),
        (evaluation_config, EXPECTED_EVALUATION_CONFIG_SHA256, "evaluation config"),
        (reference_manifest, EXPECTED_REFERENCE_CACHE_SHA256, "reference mask cache"),
    )
    for path, expected, label in expected_hashes:
        if _sha_file(path) != expected:
            raise RuntimeError(f"frozen {label} SHA256 mismatch")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probability cache: {output}")

    manifest = load_segmentation_manifest(train_manifest)
    if manifest.split != EXPECTED_SPLIT or manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("probability export requires author-protocol TRAIN-229")
    reference = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if (
        reference.get("sample_count") != 229
        or reference.get("test_data_loaded") is not False
        or reference.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or reference.get("train_manifest", {}).get("sha256") != EXPECTED_TRAIN_MANIFEST_SHA256
    ):
        raise RuntimeError("reference native-mask cache provenance mismatch")
    reference_by_id = {str(row["sample_id"]): row for row in reference["samples"]}
    if set(reference_by_id) != {sample.sample_id for sample in manifest.samples}:
        raise RuntimeError("TRAIN populations differ between manifest and reference cache")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="train-native-probabilities-", dir=output.parent))
    maps_dir = staging / "probabilities"
    maps_dir.mkdir()
    segmenter: FrozenAuthorProtocolSegmenter | None = None
    records: list[dict[str, Any]] = []
    try:
        segmenter = FrozenAuthorProtocolSegmenter(
            FrozenSegmentationConfig(evaluation_config, checkpoint, args.device)
        )
        device = torch.device(args.device)
        for index, sample in enumerate(manifest.samples, start=1):
            reference_record = reference_by_id[sample.sample_id]
            image_path = (args.data_root.resolve() / sample.image_path).resolve()
            if not image_path.is_relative_to(args.data_root.resolve()):
                raise ValueError("TRAIN image path escapes data root")
            if _sha_file(image_path) != reference_record["image_sha256"]:
                raise RuntimeError(f"TRAIN image SHA mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                inputs = _preprocess(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = segmenter.model(inputs)
                if tuple(logits.shape) != (1, 1, *FROZEN_LOGITS_SHAPE):
                    raise RuntimeError(f"invalid native logits shape: {tuple(logits.shape)}")
                probability = _validate_probability_map(
                    logits.sigmoid()[0, 0].to(device="cpu", dtype=torch.float32).numpy()
                )
            reference_mask = _load_reference_mask(reference_manifest.parent, reference_record)
            thresholded = np.ascontiguousarray((probability >= FROZEN_THRESHOLD).astype(np.uint8))
            if not np.array_equal(thresholded, reference_mask):
                changed = int(np.count_nonzero(thresholded != reference_mask))
                raise RuntimeError(
                    f"probability/reference-mask mismatch for {sample.sample_id}: {changed} pixels"
                )

            map_path = maps_dir / f"{int(sample.sample_id):04d}.npy"
            np.save(map_path, probability, allow_pickle=False)
            records.append({
                "sample_id": sample.sample_id,
                "image_sha256": reference_record["image_sha256"],
                "native_probability_path": str(map_path.relative_to(staging)),
                "native_probability_array_sha256": _sha_array(probability),
                "native_probability_file_sha256": _sha_file(map_path),
                "reference_native_mask_array_sha256": reference_record["native_mask_array_sha256"],
                "thresholded_foreground_pixels": int(thresholded.sum()),
                "shape": list(probability.shape),
                "dtype": probability.dtype.str,
                "minimum": float(probability.min()),
                "maximum": float(probability.max()),
                "mean": float(probability.mean(dtype=np.float64)),
            })
            print(
                f"[native-probability export] {index}/229 sample_id={sample.sample_id} "
                f"range=[{probability.min():.6f},{probability.max():.6f}]",
                flush=True,
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        release = segmenter.release() if segmenter is not None else {"model_reference_released": False}

    report = {
        "schema_version": 1,
        "artifact": "frozen-author-protocol-train-native-probabilities-epoch0098",
        "classification": "APPROXIMATE_DIAGNOSTIC_CACHE",
        "split": EXPECTED_SPLIT,
        "sample_count": len(records),
        "test_data_loaded": False,
        "gt_loaded": False,
        "checkpoint": {"path": str(checkpoint), "sha256": EXPECTED_CHECKPOINT_SHA256},
        "evaluation_config": {"path": str(evaluation_config), "sha256": EXPECTED_EVALUATION_CONFIG_SHA256},
        "train_manifest": {"path": str(train_manifest), "sha256": EXPECTED_TRAIN_MANIFEST_SHA256},
        "reference_mask_cache": {
            "path": str(reference_manifest),
            "sha256": EXPECTED_REFERENCE_CACHE_SHA256,
            "all_thresholded_maps_match": True,
        },
        "inference": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "native_shape": list(FROZEN_LOGITS_SHAPE),
            "activation": "sigmoid",
            "stored_dtype": "little-endian float32",
            "reference_threshold": FROZEN_THRESHOLD,
            "reference_comparison": "greater_than_or_equal",
            "backprojection_applied": False,
            "release": release,
        },
        "samples": records,
    }
    if len(records) != 229:
        raise RuntimeError("incomplete probability export")
    manifest_output = staging / "manifest.json"
    manifest_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, output)
    final_manifest = output / "manifest.json"
    print(f"MANIFEST {final_manifest}", flush=True)
    print(f"SHA256 {_sha_file(final_manifest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
