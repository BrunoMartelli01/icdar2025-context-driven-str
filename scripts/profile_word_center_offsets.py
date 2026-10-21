"""Real auxiliary U-Net forward/backward smoke; synthetic input, no trained-result claim."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from models.word_center_offsets import WordCenterOffsetAGUNet
from segmentation.training import seed_everything, runtime_environment
from segmentation.word_center_offsets import (
    load_center_offset_config, make_word_center_offsets, train_word_center_offset_epoch,
)


def profile(config, *, device, batch_size, height, width, seed):
    if batch_size < 1 or min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError("Positive batch and spatial dimensions >=32 divisible by 32 are required")
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run on the physical GPU host to qualify VRAM")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    seed_everything(seed, deterministic_algorithms=True)
    model = WordCenterOffsetAGUNet(skip_floor=config["skip_floor"], encoder_pretrained=False).to(device)
    h, w = height // 2, width // 2
    owners = np.zeros((h, w), np.int32)
    owners[h//4:3*h//4, w//8:w//2] = 1
    owners[h//4:3*h//4, w//2:7*w//8] = 2
    offsets, _ = make_word_center_offsets(owners, np.ones_like(owners, bool))
    batch = {"image": torch.randn(batch_size, 3, height, width),
             "center_offsets": torch.from_numpy(offsets)[None].repeat(batch_size, 1, 1, 1),
             "owners_native": torch.from_numpy(owners)[None].repeat(batch_size, 1, 1),
             "validity": torch.ones(batch_size, 1, h, w),
             "sample_id": ["synthetic:" + str(i) for i in range(batch_size)]}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    training = train_word_center_offset_epoch(model, [batch], optimizer, device, beta=config["loss"]["beta"])
    model.eval()
    with torch.inference_mode():
        predicted = model(batch["image"].to(device)).tanh()
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("Nonfinite center-offset smoke output")
    if not torch.any(model.head.weight != 0):
        raise RuntimeError("Auxiliary offset head did not learn from the synthetic smoke batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"status": "PASS", "output_shape": list(predicted.shape), "training": training,
            "offset_head_updated": True, "environment": runtime_environment(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("approximate", "strict"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing smoke report and select a new output file")
    config = load_center_offset_config(args.config, mode=args.mode)
    root = Path(__file__).resolve().parents[1]
    sources = ("models/word_center_offsets.py", "models/unet_adv.py", "segmentation/word_center_offsets.py",
               "segmentation/totaltext_word_kernel.py", "segmentation/training.py",
               "localize/word_center_offsets.py", "localize/word_multitask.py", "localize/crop_recovery.py",
               "scripts/profile_word_center_offsets.py")
    report = {"result_class": "smoke-test result", "claim_label": "approximate reproduction",
              "reproduction_mode": args.mode, "resolved_config": config, "model_set": "auxiliary offset U-Net only",
              "device": args.device, "precision": "float32", "input_shape": [args.batch_size, 3, args.height, args.width],
              "batch_size": args.batch_size, "seed": args.seed, "encoder_weights": "random; synthetic smoke only",
              "optimizer": "Adam; lr=1e-4; no weight decay; smoke only",
              "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
              "source_sha256": {name: sha256((root / name).read_bytes()).hexdigest() for name in sources}}
    try:
        report.update(profile(config, device=args.device, batch_size=args.batch_size,
                              height=args.height, width=args.width, seed=args.seed))
    except Exception as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error),
                      oom=isinstance(error, torch.cuda.OutOfMemoryError),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device)
                      if args.device == "cuda:0" and torch.cuda.is_available() else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device)
                      if args.device == "cuda:0" and torch.cuda.is_available() else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
