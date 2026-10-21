#!/usr/bin/env python3
"""Validate the installed ``context-str-main`` runtime without model downloads.

The optional PARSeq check requires an existing, provenance-verified copy of the
qualified official runtime and the already-downloaded v1.0.0 checkpoint.  It
never uses Torch Hub or Hugging Face model loading.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

from recognition.parseq import (
    PARSEQ_GIT_SOURCE,
    PARSEQ_SOURCE_TYPES,
    verify_parseq_runtime_source,
)


EXPECTED = {
    "python": "3.11.11",
    "torch": "2.4.1+cu124",
    "torchvision": "0.19.1+cu124",
    "numpy": "1.26.4",
    "pillow": "10.4.0",
    "opencv-python": "4.10.0.84",
    "transformers": "4.47.1",
    "accelerate": "1.2.1",
    "sentence-transformers": "3.3.1",
    "rapidfuzz": "3.11.0",
    "omegaconf": "2.3.0",
    "pytest": "8.3.4",
    "timm": "0.9.16",
    "pytorch-lightning": "2.2.0.post0",
    "torchmetrics": "1.3.1",
    "nltk": "3.8.1",
    "lmdb": "1.4.1",
    "pyyaml": "6.0.1",
}
EXPECTED_CUDA_RUNTIME = "12.4"
EXPECTED_GPU_FRAGMENT = "RTX 3070"
EXPECTED_PARSEQ_CHECKPOINT_SHA256 = (
    "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00"
)


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parseq_check(
    source: Path,
    checkpoint: Path,
    image: Path | None,
    source_type: str = PARSEQ_GIT_SOURCE,
    source_archive: Path | None = None,
) -> dict[str, Any]:
    if not source.is_dir():
        raise RuntimeError(f"PARSeq source directory does not exist: {source}")
    if not checkpoint.is_file():
        raise RuntimeError(f"PARSeq checkpoint does not exist: {checkpoint}")

    source_identity = verify_parseq_runtime_source(
        source,
        source_type=source_type,
        source_archive=source_archive,
    )
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != EXPECTED_PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError(
            "PARSeq checkpoint SHA256 mismatch: expected "
            f"{EXPECTED_PARSEQ_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    sys.path.insert(0, str(source))
    try:
        from PIL import Image
        import torch
        from strhub.data.module import SceneTextDataModule
        from strhub.models.utils import create_model

        model = create_model("parseq", pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        load_result = model.model.load_state_dict(state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"Strict PARSeq load was incomplete: {load_result}")

        result: dict[str, Any] = {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_tensor_count": len(state),
            **source_identity,
            "strict_load": True,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "inference": "not_requested",
        }
        if image is not None:
            if not image.is_file():
                raise RuntimeError(f"PARSeq smoke image does not exist: {image}")
            transform = SceneTextDataModule.get_transform(tuple(model.hparams.img_size))
            with Image.open(image) as pil_image:
                tensor = transform(pil_image.convert("RGB")).unsqueeze(0)
            device = torch.device("cuda:0")
            model.eval().to(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                logits = model(tensor.to(device))
                decoded, _ = model.tokenizer.decode(logits.softmax(-1))
            torch.cuda.synchronize(device)
            result.update(
                {
                    "inference": "passed",
                    "image": image.name,
                    "input_shape": list(tensor.shape),
                    "logits_shape": list(logits.shape),
                    "decoded_text": decoded[0],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            )
            del logits, model
            torch.cuda.empty_cache()
        return result
    finally:
        sys.path.remove(str(source))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torchvision

    versions = {name: _distribution_version(name) for name in EXPECTED if name != "python"}
    versions["python"] = platform.python_version()
    mismatches = {
        name: {"expected": expected, "actual": versions[name]}
        for name, expected in EXPECTED.items()
        if versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Version mismatch: {mismatches}")

    # Import every requested package explicitly; distribution metadata alone is
    # insufficient to detect broken shared-library or transitive imports.
    for module_name in (
        "cv2",
        "numpy",
        "PIL",
        "transformers",
        "accelerate",
        "sentence_transformers",
        "rapidfuzz",
        "omegaconf",
        "pytest",
        "timm",
        "pytorch_lightning",
        "torchmetrics",
        "nltk",
        "lmdb",
        "yaml",
    ):
        importlib.import_module(module_name)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in context-str-main")
    if torch.version.cuda != EXPECTED_CUDA_RUNTIME:
        raise RuntimeError(
            f"PyTorch CUDA runtime mismatch: expected {EXPECTED_CUDA_RUNTIME}, got {torch.version.cuda}"
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if args.expected_gpu_fragment not in properties.name:
        raise RuntimeError(
            "Expected GPU name containing {!r}, got {!r}".format(
                args.expected_gpu_fragment, properties.name
            )
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cpu")
    round_trip = (cpu_tensor.to(device) * 2).cpu()
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    matrix_result = (left @ right).cpu()
    torch.cuda.synchronize(device)

    report: dict[str, Any] = {
        "status": "passed",
        "platform": platform.platform(),
        "executable": sys.executable,
        "versions": versions,
        "torch_build": torch.__version__,
        "torchvision_build": torchvision.__version__,
        "cuda": {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": properties.name,
            "expected_gpu_fragment": args.expected_gpu_fragment,
            "hardware_classification": (
                "PRIMARY_TARGET"
                if args.expected_gpu_fragment == EXPECTED_GPU_FRAGMENT
                else "HARDWARE_DEVIATION"
            ),
            "total_vram_bytes": properties.total_memory,
            "cpu_gpu_cpu_result": round_trip.tolist(),
            "matrix_result": matrix_result.tolist(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }

    if args.parseq_source or args.parseq_checkpoint or args.parseq_image:
        if not args.parseq_source or not args.parseq_checkpoint:
            raise RuntimeError("PARSeq check requires both --parseq-source and --parseq-checkpoint")
        report["parseq"] = _parseq_check(
            args.parseq_source.resolve(),
            args.parseq_checkpoint.resolve(),
            args.parseq_image.resolve() if args.parseq_image else None,
            args.parseq_source_type,
            args.parseq_source_archive,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parseq-source", type=Path)
    parser.add_argument(
        "--parseq-source-type",
        choices=PARSEQ_SOURCE_TYPES,
        default=PARSEQ_GIT_SOURCE,
    )
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path)
    parser.add_argument("--parseq-image", type=Path)
    parser.add_argument(
        "--expected-gpu-fragment",
        default=EXPECTED_GPU_FRAGMENT,
        help=(
            "Required substring in the CUDA device name. Values other than the "
            "primary RTX 3070 target are recorded as HARDWARE_DEVIATION."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = collect(args)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
