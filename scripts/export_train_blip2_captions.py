#!/usr/bin/env python3
"""Export one pinned BLIP-2 caption per frozen ICDAR13 TRAIN image.

This GPU-only boundary reads RGB images but never loads segmentation masks, GT
annotations, proposal crops, or TEST data.  Per-image shards make an interrupted
run safely resumable; the complete manifest is written only after all 229 shards
have passed provenance checks.
"""

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
import torch

from caption.blip2_captioner import (
    BLIP2_MODEL_ID, BLIP2_REVISION, BLIP2ApproxConfig, BLIP2Captioner,
    DEFAULT_PROMPT, GENERATION_PROFILES, stable_hash,
)
from scripts.check_table5_assets import BLIP2_WEIGHT_SHA256, check_blip2
from segmentation.data import load_segmentation_manifest


EXPECTED_SOURCE_REPORT_SHA256 = "532b5da717a5a9ca11ca79b924cfb652a3a57b432931f9ae64f3e98362ea2960"
EXPECTED_GENERATION_MANIFEST_SHA256 = "4ced95784912cc1799146fd934caa6d0e1914fd6230563145e5a662efbcfaed7"
EXPECTED_TRAIN_MANIFEST_SHA256 = "3aeecf40411326f0721e5d4eea1bf74a9357eab1c9da54a77d0624b6e24f7c14"
EXPECTED_SPLIT = "official-training-all-229-author-protocol"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _caption_contract(profile: str = "repository-v1") -> dict[str, Any]:
    generation = dict(GENERATION_PROFILES[profile])
    contract = {
        "model_id": BLIP2_MODEL_ID,
        "revision": BLIP2_REVISION,
        "prompt": DEFAULT_PROMPT,
        "prompt_sha256": stable_hash(DEFAULT_PROMPT),
        "generation": generation,
        "generation_sha256": stable_hash(generation),
        "device": "cuda:0",
        "dtype": "float16",
        "placement": "single_device",
        "quantization": "none",
        "caption_count_per_image": 1,
    }
    if profile != "repository-v1":
        contract["generation_profile"] = profile
    return contract


def _validate_shard(
    value: dict[str, Any], *, sample_id: str, image_sha256: str,
    profile: str = "repository-v1",
) -> None:
    generation = dict(GENERATION_PROFILES[profile])
    result = value.get("result", {})
    if (
        value.get("schema_version") != 1
        or value.get("sample_id") != sample_id
        or value.get("image_sha256") != image_sha256
        or value.get("caption_contract") != _caption_contract(profile)
        or result.get("model_id") != BLIP2_MODEL_ID
        or result.get("revision") != BLIP2_REVISION
        or result.get("prompt_hash") != stable_hash(DEFAULT_PROMPT)
        or result.get("generation_config") != generation
        or result.get("generation_config_hash") != stable_hash(generation)
        or result.get("device") != "cuda:0"
        or result.get("dtype") != "float16"
        or result.get("placement") != "single_device"
        or result.get("quantization") != "none"
        or result.get("success") is not True
        or not isinstance(result.get("raw_caption"), str)
        or not result["raw_caption"].strip()
    ):
        raise RuntimeError(f"BLIP-2 caption shard contract mismatch: {sample_id}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--candidate-generation-manifest", type=Path)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--generation-profile", choices=tuple(GENERATION_PROFILES),
        default="repository-v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.device != "cuda:0":
        raise ValueError("qualified BLIP-2 TRAIN caption export requires --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command on PAPER")
    if "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("qualified BLIP-2 path requires the PAPER RTX 3090")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    configured_cache = os.environ.get("HF_HUB_CACHE")
    if configured_cache is None or Path(configured_cache).resolve() != args.hf_cache.resolve():
        raise RuntimeError("HF_HUB_CACHE must equal --hf-cache")

    train_manifest = args.train_manifest.resolve()
    source_report = args.source_report.resolve() if args.source_report else None
    generation_path = (
        args.candidate_generation_manifest.resolve()
        if args.candidate_generation_manifest else None
    )
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    profile = args.generation_profile
    if _sha_file(train_manifest) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("frozen TRAIN manifest SHA256 mismatch")
    asset = check_blip2(argparse.Namespace(hf_cache=args.hf_cache.resolve()))

    manifest = load_segmentation_manifest(train_manifest)
    if (
        manifest.split != EXPECTED_SPLIT
        or manifest.split_role != "train"
        or len(manifest.samples) != 229
    ):
        raise RuntimeError("caption export requires exactly frozen TRAIN-229")
    if (source_report is None) != (generation_path is None):
        raise ValueError(
            "--source-report and --candidate-generation-manifest must be supplied together"
        )
    if source_report is None:
        if profile == "repository-v1":
            raise ValueError(
                "repository-v1 requires frozen PARSeq/candidate provenance; the independent "
                "TRAIN-image contract is permitted only for the practical caption profile"
            )
        generated_by_id = {
            sample.sample_id: {"image_sha256": sample.image_sha256}
            for sample in manifest.samples
        }
        if any(
            not isinstance(row["image_sha256"], str)
            or len(row["image_sha256"]) != 64
            for row in generated_by_id.values()
        ):
            raise RuntimeError("TRAIN manifest lacks complete image SHA256 identities")
        optional_provenance = None
    else:
        if _sha_file(source_report) != EXPECTED_SOURCE_REPORT_SHA256:
            raise RuntimeError("frozen PARSeq candidate-ranking report SHA256 mismatch")
        if _sha_file(generation_path) != EXPECTED_GENERATION_MANIFEST_SHA256:
            raise RuntimeError("frozen candidate-generation manifest SHA256 mismatch")
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if (
            generation.get("sample_count") != 229
            or generation.get("gt_loaded") is not False
            or generation.get("test_loaded") is not False
        ):
            raise RuntimeError("candidate provenance is not frozen TRAIN-229")
        generated_by_id = {
            str(row["sample_id"]): row for row in generation["samples"]
        }
        if set(generated_by_id) != {sample.sample_id for sample in manifest.samples}:
            raise RuntimeError("candidate and TRAIN image populations differ")
        optional_provenance = {
            "parseq_candidate_ranking_report": str(source_report),
            "parseq_candidate_ranking_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "candidate_generation_manifest": str(generation_path),
            "candidate_generation_manifest_sha256": EXPECTED_GENERATION_MANIFEST_SHA256,
        }

    shards = output / "captions"
    shards.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    pending = []
    for sample in manifest.samples:
        shard_path = shards / f"{int(sample.sample_id):04d}.json"
        image_sha = generated_by_id[sample.sample_id]["image_sha256"]
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            _validate_shard(
                shard, sample_id=sample.sample_id, image_sha256=image_sha,
                profile=profile,
            )
            completed.append(shard)
        else:
            pending.append(sample)
    if (output / "manifest.json").exists() and pending:
        raise RuntimeError("complete caption manifest exists but one or more shards are missing")
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (
            existing.get("sample_count") != 229
            or existing.get("caption_contract") != _caption_contract(profile)
            or len(existing.get("samples", ())) != 229
        ):
            raise RuntimeError("existing complete BLIP-2 caption manifest is invalid")
        print(f"[BLIP2 TRAIN caption] verified reusable cache: 229/229", flush=True)
        print(f"MANIFEST {output / 'manifest.json'}", flush=True)
        print(f"SHA256 {_sha_file(output / 'manifest.json')}", flush=True)
        return 0

    captioner = None
    release = {"model_reference_released": False, "processor_reference_released": False}
    try:
        if pending:
            captioner = BLIP2Captioner(BLIP2ApproxConfig(
                generation=dict(GENERATION_PROFILES[profile]),
                device="cuda:0", dtype="float16", placement="single_device", quantization="none",
            ))
        for offset, sample in enumerate(pending, start=len(completed) + 1):
            image_path = (data_root / Path(sample.image_path)).resolve()
            if not image_path.is_relative_to(data_root):
                raise ValueError("TRAIN RGB path escapes data root")
            image_sha = generated_by_id[sample.sample_id]["image_sha256"]
            if _sha_file(image_path) != image_sha or sample.image_sha256 != image_sha:
                raise RuntimeError(f"TRAIN RGB SHA mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
            result = captioner.caption(rgb)
            shard = {
                "schema_version": 1,
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "image_sha256": image_sha,
                "caption_contract": _caption_contract(profile),
                "result": result.to_dict(),
            }
            _validate_shard(
                shard, sample_id=sample.sample_id, image_sha256=image_sha,
                profile=profile,
            )
            _write_atomic(shards / f"{int(sample.sample_id):04d}.json", shard)
            completed.append(shard)
            print(
                f"[BLIP2 TRAIN caption] {offset}/229 sample_id={sample.sample_id} "
                f"caption={result.raw_caption!r}", flush=True,
            )
    finally:
        if captioner is not None:
            release = captioner.release()
            torch.cuda.empty_cache()

    records = []
    for sample in manifest.samples:
        path = shards / f"{int(sample.sample_id):04d}.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        _validate_shard(
            shard, sample_id=sample.sample_id,
            image_sha256=generated_by_id[sample.sample_id]["image_sha256"],
            profile=profile,
        )
        records.append({
            "sample_id": sample.sample_id,
            "image_sha256": shard["image_sha256"],
            "caption": shard["result"]["raw_caption"],
            "shard_path": str(path.relative_to(output)),
            "shard_sha256": _sha_file(path),
            "result": shard["result"],
        })
    if len(records) != 229 or len({row["sample_id"] for row in records}) != 229:
        raise RuntimeError("incomplete or duplicate TRAIN caption population")
    report = {
        "schema_version": 1,
        "artifact": (
            "icdar13-train-blip2-captions-v1" if profile == "repository-v1"
            else "icdar13-train-blip2-captions-practical-antirepetition-v1"
        ),
        "classification": (
            "APPROXIMATE_DIAGNOSTIC_CACHE" if profile == "repository-v1"
            else "PRACTICAL_APPROXIMATION_CACHE"
        ),
        "sample_count": 229,
        "split": EXPECTED_SPLIT,
        "test_data_loaded": False,
        "gt_loaded": False,
        "one_caption_per_image": True,
        "caption_contract": _caption_contract(profile),
        "blip2_asset": asset,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "hardware_deviation": "HARDWARE_DEVIATION: RTX 3090 FP16 single-device BLIP-2 inference",
        },
        "source": {
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": EXPECTED_TRAIN_MANIFEST_SHA256,
            "image_identity": "per-sample SHA256 from frozen TRAIN manifest",
            "optional_candidate_provenance": optional_provenance,
        },
        "release": release,
        "samples": records,
    }
    _write_atomic(output / "manifest.json", report)
    print(f"MANIFEST {output / 'manifest.json'}", flush=True)
    print(f"SHA256 {_sha_file(output / 'manifest.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
