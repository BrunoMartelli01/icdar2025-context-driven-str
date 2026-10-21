"""Tiny synthetic end-to-end smoke for the APPROXIMATE segmentation path."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from localize.mask_to_blocks import BlockParam, localize_mask
from models.ag_unet_approx import ApproximateAGUNet
from segmentation.config import load_segmentation_config
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.metrics import (
    binarize_probabilities,
    foreground_iou_f1_global_micro,
    foreground_iou_f1_per_image_macro,
)
from segmentation.training import (
    SegmentationTrainingConfig,
    build_optimizer,
    seed_everything,
    train_one_epoch,
)
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "approximate" / "segmentation_smoke.yaml"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _make_synthetic_dataset(root: Path, count: int) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    samples = []
    for index in range(count):
        height, width = 96, 144
        mask = np.zeros((height, width), dtype=np.uint8)
        y1 = 15 + index % 7
        x1 = 12 + (index * 5) % 20
        mask[y1 : y1 + 28, x1 : x1 + 65] = 255
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 1] = 30
        image[mask > 0] = (220, 220, 220)
        image_path = root / "images" / f"sample-{index:02d}.png"
        mask_path = root / "masks" / f"sample-{index:02d}.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        samples.append(
            {
                "id": f"synthetic-{index:02d}",
                "image": str(image_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
                "image_sha256": _sha(image_path),
                "mask_sha256": _sha(mask_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "dataset": "synthetic-smoke",
        "split": "smoke",
        "annotation_policy": {
            "classification": "ASSUMPTION",
            "foreground": "synthetic nonzero pixels",
            "ignore": "none in synthetic smoke",
        },
        "samples": samples,
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _transform_from_config(config: Any) -> PairedSegmentationTransform:
    item = config.transforms
    return PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=float(item.horizontal_flip_probability),
            rotation_probability=float(item.rotation_probability),
            rotation_degrees=tuple(float(value) for value in item.rotation_degrees),
            image_interpolation=str(item.image_interpolation),
            mask_interpolation=str(item.mask_interpolation),
            image_fill=tuple(int(value) for value in item.image_fill),
            mask_fill=int(item.mask_fill),
            normalization=str(item.normalization),
            resize_geometry=str(item.resize_geometry),
            operation_order=str(item.operation_order),
        ),
        base_seed=int(config.training.seed),
    )


def _training_config(config: Any) -> SegmentationTrainingConfig:
    item = config.training
    return SegmentationTrainingConfig(
        epochs=int(item.epochs),
        seed=int(item.seed),
        adam_betas=tuple(float(value) for value in item.adam_betas),
        adam_epsilon=float(item.adam_epsilon),
        learning_rate_schedule=str(item.learning_rate_schedule),
        checkpoint_selection=str(item.checkpoint_selection),
        deterministic_algorithms=bool(item.deterministic_algorithms),
    )


def run_smoke(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device_name: str,
    batch_size: int,
    train_step: bool,
) -> dict[str, Any]:
    config = load_segmentation_config(config_path)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "PENDING_PHYSICAL_HOST: CUDA is unavailable in this execution context; "
            "this is not a physical-host validation failure"
        )
    if train_step and batch_size != int(config.training.batch_size):
        raise ValueError("Training profile must use the paper-specified batch_size=8")
    device = torch.device(device_name)
    training_config = _training_config(config)
    seed_everything(training_config.seed, deterministic_algorithms=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with tempfile.TemporaryDirectory(prefix="context-str-seg-smoke-") as directory:
        data_root = Path(directory)
        manifest_path = _make_synthetic_dataset(data_root, batch_size)
        manifest = load_segmentation_manifest(manifest_path)
        transform = _transform_from_config(config)
        dataset = SegmentationDataset(manifest, data_root, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        model = ApproximateAGUNet(encoder_weights=None, freeze_encoder=False).to(device)
        optimizer = build_optimizer(model, training_config)
        training_metrics: dict[str, float | int] | None = None
        if train_step:
            training_metrics = train_one_epoch(model, [batch], optimizer, device=device)

        model.eval()
        with torch.inference_mode():
            logits = model(batch["image"].to(device))
            probabilities = logits.sigmoid().cpu()
        prediction = binarize_probabilities(
            probabilities, threshold=float(config.evaluation.probability_threshold)
        )
        targets = batch["mask"].bool()
        empty_policy = str(config.evaluation.empty_foreground_policy)
        macro = foreground_iou_f1_per_image_macro(
            prediction, targets, empty_foreground_policy=empty_policy
        )
        micro = foreground_iou_f1_global_micro(
            prediction, targets, empty_foreground_policy=empty_policy
        )

        # The transformed tensor is CHW/normalized, so reconstruct the original RGB
        # directly from the deterministic source fixture for crop provenance.
        with Image.open(dataset.paths[0][0]) as opened:
            original_rgb = np.asarray(opened.convert("RGB"))
        original_prediction = (
            F.interpolate(
                prediction[0:1].float(), size=original_rgb.shape[:2], mode="nearest"
            )[0, 0]
            .numpy()
            .astype(np.uint8)
        )
        localization = localize_mask(
            original_prediction,
            original_rgb,
            BlockParam(
                min_area=int(config.localization.min_area),
                pad=int(config.localization.pad),
                connectivity=int(config.localization.connectivity),
                tie_break=str(config.localization.equal_area_tie_break),
                nmax=int(config.localization.nmax),
            ),
        )
        blocks = localization.retained

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        device_name_report = torch.cuda.get_device_name(device)
    else:
        peak_allocated = peak_reserved = None
        device_name_report = platform.processor() or "cpu"
    return {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "result_class": "smoke-test result",
        "device": str(device),
        "device_name": device_name_report,
        "model_set": ["ApproximateAGUNet"],
        "precision": "float32",
        "encoder_weights": None,
        "freeze_encoder": False,
        "batch_size": batch_size,
        "train_step": train_step,
        "input_shape": list(batch["image"].shape),
        "target_shape": list(batch["mask"].shape),
        "logit_shape": list(logits.shape),
        "parameter_count": model.parameter_count,
        "training": training_metrics,
        "metrics": {"per_image_macro": macro.to_dict(), "global_micro": micro.to_dict()},
        "localization_audit": localization.as_dict(),
        "localized_blocks": [
            {
                "rank": block.rank,
                "component_id": block.component_id,
                "component_area": block.component_area,
                "bbox": list(block.bbox),
                "padded_bbox": list(block.padded_bbox),
                "crop_shape": list(block.crop.shape),
            }
            for block in blocks
        ],
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "strict_status": "BLOCKED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    failed = False
    try:
        report = run_smoke(
            config_path=args.config,
            device_name=args.device,
            batch_size=args.batch_size,
            train_step=args.train_step,
        )
    except RuntimeError as error:
        is_oom = "out of memory" in str(error).lower()
        if args.device != "cuda" or not is_oom:
            raise
        report = {
            "schema_version": 1,
            "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction",
            "result_class": "smoke-test result",
            "status": "CUDA_OUT_OF_MEMORY",
            "model_set": ["ApproximateAGUNet"],
            "precision": "float32",
            "input_shape": [args.batch_size, 3, 256, 256],
            "target_shape": [args.batch_size, 1, 128, 128],
            "batch_size": args.batch_size,
            "train_step": args.train_step,
            "error": str(error),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        failed = True
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
