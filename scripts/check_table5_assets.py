#!/usr/bin/env python3
"""Offline identity audit for artifacts used by the APPROXIMATE Table 5 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from caption.blip2_captioner import BLIP2_MODEL_ID, BLIP2_REVISION
from recognition.parseq import (
    PARSEQ_CHECKPOINT_NAME,
    PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_GIT_SOURCE,
    PARSEQ_SOURCE_TYPES,
    verify_parseq_runtime_source,
)
from scoring.semantic import MPNET_MODEL_ID, MPNET_REVISION, MPNET_WEIGHT_SHA256
from segmentation.config import load_segmentation_config
from scripts.check_main_environment import (
    EXPECTED as EXPECTED_MAIN_ENVIRONMENT,
    EXPECTED_CUDA_RUNTIME,
)


BLIP2_WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "23bb7a7a78644afa9a138dd6c4360edf51ca5fc6da1c5d01f147b964ad7c0c6d",
    "model-00002-of-00002.safetensors": "c77bd5bcdbc0945c57444d34d28bd4452dc715c3d8aef54e2e91459fdafb7e03",
}
COMPONENTS = ("segmentation", "parseq", "blip2", "mpnet", "python")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(cache: Path, model_id: str, revision: str) -> Path:
    encoded = "models--{}".format(model_id.replace("/", "--"))
    candidates = (
        cache / encoded / "snapshots" / revision,
        cache / "hub" / encoded / "snapshots" / revision,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Hugging Face snapshot is missing for {}@{} under {}".format(
            model_id, revision, cache
        )
    )


def check_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    if args.segmentation_config is None or args.segmentation_checkpoint is None:
        raise ValueError("segmentation requires --segmentation-config and --segmentation-checkpoint")
    config = load_segmentation_config(args.segmentation_config)
    checkpoint = args.segmentation_checkpoint.resolve()
    actual = sha256_file(checkpoint)
    expected = str(config.checkpoint.sha256)
    if actual != expected:
        raise RuntimeError("Segmentation checkpoint SHA256 mismatch")
    return {
        "status": "ready",
        "checkpoint": str(checkpoint),
        "sha256": actual,
        "architecture": str(config.architecture.identity),
        "logits_shape": list(config.transforms.mask_size),
        "threshold": float(config.evaluation.probability_threshold),
        "encoder_frozen": bool(config.architecture.freeze_encoder),
        "validity_mask_downstream": False,
    }


def check_parseq(args: argparse.Namespace) -> dict[str, Any]:
    if args.parseq_source is None or args.parseq_checkpoint is None:
        raise ValueError("parseq requires --parseq-source and --parseq-checkpoint")
    source, checkpoint = args.parseq_source.resolve(), args.parseq_checkpoint.resolve()
    if not source.is_dir():
        raise FileNotFoundError("PARSeq source directory is missing: {}".format(source))
    if not checkpoint.is_file():
        raise FileNotFoundError("PARSeq checkpoint is missing: {}".format(checkpoint))
    identity = verify_parseq_runtime_source(
        source,
        source_type=args.parseq_source_type,
        source_archive=args.parseq_source_archive,
    )
    digest = sha256_file(checkpoint)
    if checkpoint.name != PARSEQ_CHECKPOINT_NAME or digest != PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError("PARSeq checkpoint name or SHA256 mismatch")
    return {
        "status": "ready",
        "source": str(source),
        **identity,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
    }


def check_blip2(args: argparse.Namespace) -> dict[str, Any]:
    if args.hf_cache is None:
        raise ValueError("blip2 requires --hf-cache")
    snapshot = _snapshot(args.hf_cache.resolve(), BLIP2_MODEL_ID, BLIP2_REVISION)
    required = ("config.json", "preprocessor_config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError("BLIP-2 snapshot is incomplete: {}".format(missing))
    hashes = {name: sha256_file(snapshot / name) for name in BLIP2_WEIGHT_SHA256}
    if hashes != BLIP2_WEIGHT_SHA256:
        raise RuntimeError("BLIP-2 weight shard SHA256 mismatch")
    return {
        "status": "ready",
        "model_id": BLIP2_MODEL_ID,
        "revision": BLIP2_REVISION,
        "snapshot": str(snapshot),
        "weight_sha256": hashes,
        "processor_files_present": True,
    }


def check_mpnet(args: argparse.Namespace) -> dict[str, Any]:
    if args.hf_cache is None:
        raise ValueError("mpnet requires --hf-cache")
    snapshot = _snapshot(args.hf_cache.resolve(), MPNET_MODEL_ID, MPNET_REVISION)
    required = ("modules.json", "1_Pooling/config.json", "model.safetensors")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError("MPNet snapshot is incomplete: {}".format(missing))
    digest = sha256_file(snapshot / "model.safetensors")
    if digest != MPNET_WEIGHT_SHA256:
        raise RuntimeError("MPNet weight SHA256 mismatch")
    return {
        "status": "ready",
        "model_id": MPNET_MODEL_ID,
        "revision": MPNET_REVISION,
        "snapshot": str(snapshot),
        "weight_sha256": digest,
        "pooling": "mean",
        "normalize_embeddings": True,
    }


def check_python(_: argparse.Namespace) -> dict[str, Any]:
    import platform
    import importlib
    import importlib.metadata

    import_names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "pillow": "PIL",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "sentence-transformers": "sentence_transformers",
        "rapidfuzz": "rapidfuzz",
        "omegaconf": "omegaconf",
        "opencv-python": "cv2",
        "timm": "timm",
        "pytorch-lightning": "pytorch_lightning",
        "torchmetrics": "torchmetrics",
        "nltk": "nltk",
        "lmdb": "lmdb",
        "pyyaml": "yaml",
        "pytest": "pytest",
    }
    versions = {"python": platform.python_version()}
    for distribution, module_name in import_names.items():
        importlib.import_module(module_name)
        versions[distribution] = importlib.metadata.version(distribution)
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in EXPECTED_MAIN_ENVIRONMENT.items()
        if versions.get(name) != expected
    }
    if mismatches:
        raise RuntimeError("Main environment version mismatch: {}".format(mismatches))

    import torch

    if torch.version.cuda != EXPECTED_CUDA_RUNTIME:
        raise RuntimeError(
            "PyTorch CUDA runtime mismatch: expected {}, got {}".format(
                EXPECTED_CUDA_RUNTIME, torch.version.cuda
            )
        )
    return {
        "status": "ready",
        "python": versions.pop("python"),
        "packages": versions,
        "torch_cuda": torch.version.cuda,
    }


CHECKS = {
    "segmentation": check_segmentation,
    "parseq": check_parseq,
    "blip2": check_blip2,
    "mpnet": check_mpnet,
    "python": check_python,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, action="append")
    parser.add_argument("--segmentation-config", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--parseq-source", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path)
    parser.add_argument(
        "--parseq-source-type",
        choices=PARSEQ_SOURCE_TYPES,
        default=PARSEQ_GIT_SOURCE,
    )
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.component or list(COMPONENTS)
    report = {"schema_version": 1, "components": {}}
    failed = False
    for name in selected:
        try:
            report["components"][name] = CHECKS[name](args)
        except Exception as error:
            failed = True
            report["components"][name] = {
                "status": "missing_or_blocked",
                "error": "{}: {}".format(type(error).__name__, error),
            }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
