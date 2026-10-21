#!/usr/bin/env python3
"""Persist compact TRAIN practical-remap-v1 crops and PARSeq outputs.

The two GPU models are loaded sequentially.  Candidate and OCR shards are
atomic and per-image, so an interrupted run resumes without repeating verified
images.  GT annotations, TEST, BLIP-2, MPNet and routing are never loaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    PARSEQ_RUNTIME_REVISION, ParseqApproxConfig, ParseqApproxRecognizer,
)
from segmentation.data import load_segmentation_manifest


EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
EXPECTED_TRAIN_MANIFEST_SHA256 = "3aeecf40411326f0721e5d4eea1bf74a9357eab1c9da54a77d0624b6e24f7c14"
EXPECTED_SPLIT = "official-training-all-229-author-protocol"
EXPECTED_V1_CANDIDATES = 1203
EXPECTED_V1 = {
    "horizontal_gap_ratio": 0.35,
    "vertical_overlap_min": 0.7,
    "max_center_y_ratio": 0.5,
    "max_height_ratio": 2.5,
    "pad": 6,
    "connectivity": 8,
    "nmax": 10,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_png_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        Image.fromarray(value, mode="RGB").save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contained(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative contained path required")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes declared root")
    return path


def _params(path: Path) -> GroupedWordBlockParams:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = value["localization"]
    group = root["grouping"]
    resolved = {
        "horizontal_gap_ratio": float(group["horizontal_gap_ratio"]),
        "vertical_overlap_min": float(group["vertical_overlap_min"]),
        "max_center_y_ratio": float(group["max_center_y_ratio"]),
        "max_height_ratio": float(group["max_height_ratio"]),
        "pad": int(group["pad_after_grouping"]),
        "connectivity": int(root["raw_component_connectivity"]),
        "nmax": int(group["nmax"]),
    }
    if resolved != EXPECTED_V1:
        raise RuntimeError(f"practical-remap-v1 config drift: {resolved}")
    return GroupedWordBlockParams(**resolved)


def _candidate_shard_valid(path: Path, *, sample_id: str, image_sha: str, root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("sample_id") != sample_id or value.get("image_sha256") != image_sha:
        raise RuntimeError(f"candidate shard identity mismatch: {sample_id}")
    rows = value.get("candidates", [])
    if len(rows) > 10 or [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"candidate shard ordering mismatch: {sample_id}")
    for row in rows:
        crop = _contained(root, row["crop_path"])
        if _sha(crop) != row["crop_sha256"]:
            raise RuntimeError(f"candidate crop SHA mismatch: {row['candidate_id']}")
    return value


def _ocr_shard_valid(path: Path, *, candidate_ids: list[str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("candidate_ids") != candidate_ids
        or value.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
        or value.get("parseq_runtime_revision") != PARSEQ_RUNTIME_REVISION
        or value.get("parseq_runtime_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or [row["candidate_id"] for row in value.get("results", [])] != candidate_ids
    ):
        raise RuntimeError(f"PARSeq shard contract mismatch: {path}")
    return value


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--localizer-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("compact TRAIN export requires PAPER --device cuda:0")
    cuda_device = torch.device(args.device)
    torch.cuda.set_device(cuda_device)
    train_path, checkpoint = args.train_manifest.resolve(), args.checkpoint.resolve()
    data_root, output = args.data_root.resolve(), args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"segmentation checkpoint file is missing: {checkpoint}")
    if _sha(train_path) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("TRAIN manifest SHA256 mismatch")
    if _sha(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("segmentation checkpoint SHA256 mismatch")
    if _sha(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError("PARSeq checkpoint SHA256 mismatch")
    params = _params(args.localizer_config.resolve())
    manifest = load_segmentation_manifest(train_path)
    if manifest.split != EXPECTED_SPLIT or manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("requires exactly TRAIN-229")
    output.mkdir(parents=True, exist_ok=True)
    candidate_root, ocr_root = output / "candidate-shards", output / "ocr-shards"
    candidate_root.mkdir(exist_ok=True); ocr_root.mkdir(exist_ok=True)
    final_manifest = output / "manifest.json"
    if final_manifest.exists():
        existing = json.loads(final_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("artifact") != "train-practical-remap-v1-parseq-compact-v1"
            or existing.get("scope") != {
                "split": "TRAIN", "images": 229,
                "test_loaded": False, "gt_loaded": False,
            }
            or existing.get("candidate_count") != EXPECTED_V1_CANDIDATES
            or existing.get("localizer", {}).get("parameters") != EXPECTED_V1
            or existing.get("localizer", {}).get("config_sha256") != _sha(args.localizer_config.resolve())
            or existing.get("segmentation", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
            or existing.get("parseq", {}).get("checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
            or len(existing.get("samples", [])) != 229
        ):
            raise RuntimeError("existing compact TRAIN manifest contract mismatch")
        existing_by_id = {str(row["sample_id"]): row for row in existing["samples"]}
        for sample in manifest.samples:
            ref = existing_by_id.get(sample.sample_id)
            if ref is None:
                raise RuntimeError("existing compact TRAIN population mismatch")
            cp = _contained(output, ref["candidate_shard"])
            op = _contained(output, ref["ocr_shard"])
            if _sha(cp) != ref["candidate_shard_sha256"] or _sha(op) != ref["ocr_shard_sha256"]:
                raise RuntimeError(f"existing compact shard SHA mismatch: {sample.sample_id}")
            candidate = _candidate_shard_valid(
                cp, sample_id=sample.sample_id, image_sha=sample.image_sha256, root=output,
            )
            _ocr_shard_valid(op, candidate_ids=[row["candidate_id"] for row in candidate["candidates"]])
        print(f"[compact TRAIN export] verified reusable cache: {final_manifest}", flush=True)
        print(f"SHA256 {_sha(final_manifest)}", flush=True)
        return 0

    pending = []
    candidate_shards: dict[str, dict[str, Any]] = {}
    for sample in manifest.samples:
        path = candidate_root / f"{int(sample.sample_id):04d}.json"
        if path.exists():
            candidate_shards[sample.sample_id] = _candidate_shard_valid(
                path, sample_id=sample.sample_id, image_sha=sample.image_sha256, root=output,
            )
        else:
            pending.append(sample)
    segmenter = None
    segmentation_release = {"model_reference_released": False}
    segmentation_peak = 0
    try:
        if pending:
            initialization = torch.empty(1, device=cuda_device)
            del initialization
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_device)
            segmenter = FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(
                args.evaluation_config.resolve(), checkpoint, args.device,
            ))
        for number, sample in enumerate(pending, start=1):
            image_path = _contained(data_root, sample.image_path)
            if _sha(image_path) != sample.image_sha256:
                raise RuntimeError(f"TRAIN image SHA mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
            segmentation = segmenter(rgb)
            grouped = group_word_blocks(segmentation["mask"], rgb, params)
            rows = []
            for block in grouped.retained:
                candidate_id = f"v1:{int(sample.sample_id):04d}:{block.rank:03d}"
                relative = Path("crops") / f"{int(sample.sample_id):04d}" / f"v1-{block.rank:03d}.png"
                crop_path = _contained(output, relative)
                _save_png_atomic(crop_path, np.ascontiguousarray(block.crop))
                rows.append({
                    "candidate_id": candidate_id, "rank": block.rank,
                    "bbox": list(block.padded_bbox), "tight_bbox": list(block.bbox),
                    "raw_component_ids": list(block.raw_component_ids),
                    "foreground_area": int(block.foreground_area),
                    "crop_path": str(relative), "crop_sha256": _sha(crop_path),
                    "crop_shape_hwc": list(block.crop.shape),
                })
            shard = {
                "schema_version": 1, "sample_id": sample.sample_id,
                "image_path": str(sample.image_path), "image_sha256": sample.image_sha256,
                "native_mask_sha256": segmentation["metadata"]["native_mask_sha256"],
                "projected_mask_sha256": segmentation["metadata"]["output_mask_sha256"],
                "candidate_count": len(rows), "candidates": rows,
            }
            _write_atomic(candidate_root / f"{int(sample.sample_id):04d}.json", shard)
            candidate_shards[sample.sample_id] = shard
            print(f"[v1 segmentation] {number}/{len(pending)} sample_id={sample.sample_id} crops={len(rows)}", flush=True)
        if segmenter is not None:
            segmentation_peak = int(torch.cuda.max_memory_allocated(cuda_device))
    finally:
        if segmenter is not None:
            segmentation_release = segmenter.release()
            del segmenter
        gc.collect(); torch.cuda.empty_cache()

    pending_ocr = []
    ocr_shards: dict[str, dict[str, Any]] = {}
    for sample in manifest.samples:
        candidate_ids = [row["candidate_id"] for row in candidate_shards[sample.sample_id]["candidates"]]
        path = ocr_root / f"{int(sample.sample_id):04d}.json"
        if path.exists():
            ocr_shards[sample.sample_id] = _ocr_shard_valid(path, candidate_ids=candidate_ids)
        else:
            pending_ocr.append(sample)
    recognizer = None
    parseq_release = {"model_reference_released": False}
    parseq_peak = 0
    try:
        if pending_ocr:
            initialization = torch.empty(1, device=cuda_device)
            del initialization
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_device)
            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                runtime_source=args.parseq_source.resolve(),
                checkpoint=args.parseq_checkpoint.resolve(), device=args.device,
                runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=args.parseq_source_archive.resolve(),
            ))
        for number, sample in enumerate(pending_ocr, start=1):
            candidates = candidate_shards[sample.sample_id]["candidates"]
            images = []
            for row in candidates:
                crop = _contained(output, row["crop_path"])
                with Image.open(crop) as opened:
                    images.append(np.ascontiguousarray(np.asarray(opened.convert("RGB"))))
            results = recognizer.recognize_candidate_batch(images) if images else ()
            shard = {
                "schema_version": 1,
                "sample_id": sample.sample_id,
                "candidate_ids": [row["candidate_id"] for row in candidates],
                "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
                "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "results": [
                    {"candidate_id": row["candidate_id"], **result.to_dict()}
                    for row, result in zip(candidates, results)
                ],
            }
            _write_atomic(ocr_root / f"{int(sample.sample_id):04d}.json", shard)
            ocr_shards[sample.sample_id] = shard
            print(f"[v1 PARSeq] {number}/{len(pending_ocr)} sample_id={sample.sample_id} crops={len(images)}", flush=True)
        if recognizer is not None:
            parseq_peak = int(torch.cuda.max_memory_allocated(cuda_device))
    finally:
        if recognizer is not None:
            parseq_release = recognizer.release()
            del recognizer
        gc.collect(); torch.cuda.empty_cache()

    samples = []
    for sample in manifest.samples:
        candidate_path = candidate_root / f"{int(sample.sample_id):04d}.json"
        ocr_path = ocr_root / f"{int(sample.sample_id):04d}.json"
        _candidate_shard_valid(candidate_path, sample_id=sample.sample_id,
                               image_sha=sample.image_sha256, root=output)
        _ocr_shard_valid(ocr_path, candidate_ids=[
            row["candidate_id"] for row in candidate_shards[sample.sample_id]["candidates"]
        ])
        samples.append({
            "sample_id": sample.sample_id,
            "candidate_shard": str(candidate_path.relative_to(output)),
            "candidate_shard_sha256": _sha(candidate_path),
            "ocr_shard": str(ocr_path.relative_to(output)),
            "ocr_shard_sha256": _sha(ocr_path),
            "candidate_count": len(candidate_shards[sample.sample_id]["candidates"]),
        })
    candidate_count = sum(row["candidate_count"] for row in samples)
    if candidate_count != EXPECTED_V1_CANDIDATES:
        raise RuntimeError(
            f"frozen practical-remap-v1 control mismatch: expected "
            f"{EXPECTED_V1_CANDIDATES} crops, got {candidate_count}"
        )
    report = {
        "schema_version": 1, "artifact": "train-practical-remap-v1-parseq-compact-v1",
        "classification": "PRACTICAL_APPROXIMATION_CACHE",
        "scope": {"split": "TRAIN", "images": 229, "test_loaded": False, "gt_loaded": False},
        "localizer": {"mode": "practical-remap-v1", "parameters": EXPECTED_V1,
                      "config_sha256": _sha(args.localizer_config.resolve())},
        "segmentation": {"checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                         "peak_allocated_bytes": segmentation_peak,
                         "release": segmentation_release},
        "parseq": {"checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                   "runtime_revision": PARSEQ_RUNTIME_REVISION,
                   "runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                   "peak_allocated_bytes": parseq_peak, "release": parseq_release},
        "candidate_count": candidate_count,
        "storage": "lossless PNG crops plus compact per-image JSON; resumable atomic shards",
        "samples": samples,
    }
    _write_atomic(final_manifest, report)
    print(f"MANIFEST {final_manifest}", flush=True)
    print(f"SHA256 {_sha(final_manifest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
