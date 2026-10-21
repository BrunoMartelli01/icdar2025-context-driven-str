#!/usr/bin/env python3
"""Standalone BLIP-2 feasibility profiler; always emits a JSON success/OOM record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from caption.blip2_captioner import (
    BLIP2_REVISION, BLIP2ApproxConfig, BLIP2Captioner, DEFAULT_GENERATION,
    DEFAULT_PROMPT, canonical_torch_device, cpu_peak_rss_bytes, cuda_device_index,
    stable_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--placement", choices=("single_device", "cpu_offload"), default="single_device")
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--model-lifecycle", choices=("standalone_blip2_only", "sequential_pipeline"),
                        default="standalone_blip2_only")
    args = parser.parse_args()
    started = time.perf_counter()
    image_path = args.image.resolve()
    image_sha256 = None
    image_shape = None
    try:
        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        image = np.asarray(Image.open(image_path).convert("RGB"))
        image_shape = list(image.shape)
        result = BLIP2Captioner(BLIP2ApproxConfig(
            device=args.device, dtype=args.dtype, placement=args.placement,
            quantization=args.quantization,
        )).caption(image).to_dict()
        result["model_lifecycle"] = args.model_lifecycle
    except Exception as error:
        import torch
        is_oom = "out of memory" in str(error).lower()
        elapsed = time.perf_counter() - started
        try:
            parsed_device = canonical_torch_device(args.device)
            cuda_index = cuda_device_index(parsed_device)
        except ValueError:
            cuda_index = None
        result = {
            "success": False, "oom": is_oom, "error_type": type(error).__name__,
            "error": str(error), "latency_seconds": elapsed,
            "traceback": traceback.format_exc(),
            "device": args.device, "dtype": args.dtype, "placement": args.placement,
            "quantization": args.quantization,
            "model_lifecycle": args.model_lifecycle,
            "model_id": "Salesforce/blip2-flan-t5-xl", "revision": BLIP2_REVISION,
            "prompt_hash": stable_hash(DEFAULT_PROMPT),
            "generation_config": DEFAULT_GENERATION,
            "generation_config_hash": stable_hash(DEFAULT_GENERATION),
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(cuda_index)
                if cuda_index is not None else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(cuda_index)
                if cuda_index is not None else None
            ),
            "load_time_seconds": None,
            "generation_time_seconds": None,
            "total_latency_seconds": elapsed,
            "cpu_peak_rss_bytes": cpu_peak_rss_bytes(),
            "device_placement_summary": {},
            "hardware_deviation": (
                "HARDWARE_DEVIATION: FP16 Accelerate CPU offload on RTX 3070 8GB"
                if args.placement == "cpu_offload" else None
            ),
        }
    import torch

    hardware = {
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        free_vram, visible_total_vram = torch.cuda.mem_get_info(0)
        hardware.update(
            {
                "cuda_device_index": 0,
                "free_vram_before_profile_bytes": free_vram,
                "gpu": properties.name,
                "total_vram_bytes": properties.total_memory,
                "visible_total_vram_bytes": visible_total_vram,
            }
        )
    result.update(
        {
            "batch_size": 1,
            "hardware": hardware,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "image_shape_hwc": image_shape,
            "runtime": {
                "accelerate": importlib.metadata.version("accelerate"),
                "python": platform.python_version(),
                "transformers": importlib.metadata.version("transformers"),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
