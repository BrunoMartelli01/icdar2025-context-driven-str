#!/usr/bin/env python3
"""Emit the static M8 contract or profile one canonical CUDA forward pass."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from models.ag_unet_approx import ApproximateAGUNet


def _write(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _failure_record(error: Exception, *, device: str, precision: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
    peak_allocated = None
    peak_reserved = None
    device_name = None
    if cuda_available:
        try:
            cuda_device = torch.device(device)
            device_name = torch.cuda.get_device_name(cuda_device)
            peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
        except Exception:
            pass
    return {
        "status": "failed",
        "success": False,
        "oom": is_oom,
        "error_type": type(error).__name__,
        "error": str(error),
        "actionable_oom": (
            "Close other GPU processes and rerun the unchanged float32 batch-1 command; "
            "any reduced precision or offload retry is a separately documented HARDWARE_DEVIATION."
            if is_oom
            else None
        ),
        "traceback": traceback.format_exc(),
        "device": device,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "input_shape": [1, 3, 256, 256],
        "batch_size": 1,
        "precision": precision,
        "encoder_weights": "IMAGENET1K_V2",
        "encoder_weights_artifact": "ART-RESNET50-IMAGENET1K-V2-CANDIDATE",
        "encoder_weights_classification": "REPOSITORY_SPECIFIED",
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def profile_cuda_forward(*, device: str, precision: str) -> dict[str, Any]:
    if device != "cuda:0":
        raise ValueError("M8 physical qualification requires exactly device cuda:0")
    if precision != "float32":
        raise ValueError("M8 selected qualification branch declares float32 precision")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PENDING_PHYSICAL_HOST: CUDA is unavailable in this execution context; "
            "run this command on the physical RTX 3070 host"
        )
    cuda_device = torch.device(device)
    cuda_initializer = torch.empty(1, device=cuda_device)
    del cuda_initializer
    torch.cuda.synchronize(cuda_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_device)
    model = ApproximateAGUNet(
        encoder_weights="IMAGENET1K_V2", freeze_encoder=False
    ).eval().to(device=cuda_device, dtype=torch.float32)
    image = torch.zeros((1, 3, 256, 256), device=cuda_device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(image)
    torch.cuda.synchronize(cuda_device)
    return {
        "status": "passed",
        "success": True,
        "oom": False,
        "error_type": None,
        "error": None,
        "actionable_oom": None,
        "device": device,
        "device_name": torch.cuda.get_device_name(cuda_device),
        "cuda_available": True,
        "input_shape": list(image.shape),
        "output_shape": list(logits.shape),
        "batch_size": 1,
        "precision": precision,
        "encoder_weights": "IMAGENET1K_V2",
        "encoder_weights_artifact": "ART-RESNET50-IMAGENET1K-V2-CANDIDATE",
        "encoder_weights_classification": "REPOSITORY_SPECIFIED",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="float32", choices=("float32",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    static_model = ApproximateAGUNet(encoder_weights=None, freeze_encoder=False)
    architecture = static_model.architecture_summary()
    if args.architecture_only:
        _write(architecture, args.output)
        return 0
    del static_model

    try:
        qualification = profile_cuda_forward(device=args.device, precision=args.precision)
    except Exception as error:
        qualification = _failure_record(error, device=args.device, precision=args.precision)
    report = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "result_class": "smoke-test result",
        "architecture": architecture,
        "cuda_qualification": qualification,
    }
    _write(report, args.output)
    return 0 if qualification["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
