"""Deterministic APPROXIMATE ICDAR13 segmentation evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.ag_unet_approx import ApproximateAGUNet
from models.unet_adv import AuthorProtocolAGUNet
from segmentation.config import load_segmentation_config
from segmentation.data import (
    SegmentationDataset,
    file_sha256,
    load_segmentation_manifest,
)
from segmentation.training import load_checkpoint, seed_everything


EXPECTED_CHECKPOINT_SHA256 = "335a003ae00cf21d159151b9568e10eafde12d6ae524bf7505e1d62ae4c3bfe2"
EXPECTED_SAMPLE_COUNT = 233
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/approximate/segmentation_table3_icdar13.yaml"


class SegmentationEvaluationError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _preprocess(image: Image.Image) -> torch.Tensor:
    resized = TF.resize(
        image.convert("RGB"), [256, 256], interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return TF.normalize(
        TF.to_tensor(resized), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )


def resize_binary_target_to_native_logits(
    target: Image.Image,
    *,
    size: tuple[int, int] = (128, 128),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Bilinearly resize a binary PIL target and re-binarize at ``threshold``."""

    if target.mode != "L":
        raise SegmentationEvaluationError("Native evaluation target must be an L-mode mask")
    resized = TF.resize(
        target,
        list(size),
        interpolation=InterpolationMode.BILINEAR,
    )
    return (
        TF.pil_to_tensor(resized).to(torch.float32) / 255.0 >= threshold
    )[0]


def resize_validity_to_native_logits(
    validity: Image.Image,
    *,
    size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Resize a care/ignore mask with the training protocol's nearest rule."""

    if validity.mode != "L":
        raise SegmentationEvaluationError("Native validity mask must be an L-mode mask")
    resized = TF.resize(
        validity,
        list(size),
        interpolation=InterpolationMode.NEAREST,
    )
    return TF.pil_to_tensor(resized)[0] > 0


def _sample_record(
    sample_id: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor | None = None,
) -> dict[str, Any]:
    prediction = prediction.bool()
    target = target.bool()
    if prediction.shape != target.shape:
        raise SegmentationEvaluationError("Prediction and target shapes differ")
    if validity is None:
        validity = torch.ones_like(target, dtype=torch.bool)
    else:
        validity = validity.bool()
        if validity.shape != target.shape:
            raise SegmentationEvaluationError("Validity and target shapes differ")
    tp = int(((prediction & target) & validity).sum())
    fp = int(((prediction & ~target) & validity).sum())
    fn = int(((~prediction & target) & validity).sum())
    union = tp + fp + fn
    both_empty = union == 0
    return {
        "sample_id": sample_id,
        "status": "success",
        "failure_type": None,
        "failure_message": None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "foreground_iou": 1.0 if both_empty else tp / union,
        "foreground_f1": 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        "both_empty": both_empty,
        "valid_pixel_count": int(validity.sum()),
        "ignored_pixel_count": int((~validity).sum()),
    }


def aggregate_records(records: list[dict[str, Any]], *, expected_count: int) -> dict[str, Any]:
    if len(records) != expected_count:
        raise SegmentationEvaluationError(
            f"No omitted samples allowed: expected {expected_count}, got {len(records)}"
        )
    failures = [record for record in records if record["status"] != "success"]
    base = {
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "both_empty_count": sum(bool(record.get("both_empty")) for record in records),
        "valid_pixel_count": (
            None
            if failures or any(record.get("valid_pixel_count") is None for record in records)
            else sum(int(record["valid_pixel_count"]) for record in records)
        ),
        "ignored_pixel_count": (
            None
            if failures or any(record.get("ignored_pixel_count") is None for record in records)
            else sum(int(record["ignored_pixel_count"]) for record in records)
        ),
        "complete": not failures,
    }
    if failures:
        return {**base, "primary_per_image_macro": None, "diagnostic_global_micro": None}
    tp = sum(int(record["tp"]) for record in records)
    fp = sum(int(record["fp"]) for record in records)
    fn = sum(int(record["fn"]) for record in records)
    union = tp + fp + fn
    return {
        **base,
        "primary_per_image_macro": {
            "foreground_iou": sum(float(record["foreground_iou"]) for record in records)
            / len(records),
            "foreground_f1": sum(float(record["foreground_f1"]) for record in records)
            / len(records),
            "aggregation": "arithmetic_mean_across_all_233_images",
        },
        "diagnostic_global_micro": {
            "foreground_iou": 1.0 if union == 0 else tp / union,
            "foreground_f1": 1.0 if union == 0 else 2 * tp / (2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "diagnostic_only": True,
        },
    }


def _model_from_config(config: Any) -> torch.nn.Module:
    implementation = str(config.architecture.implementation)
    if implementation == "figure_derived_ag_unet":
        return ApproximateAGUNet(
            encoder_weights=None,
            freeze_encoder=bool(config.architecture.freeze_encoder),
        )
    if implementation == "author_clarified_ag_unet":
        return AuthorProtocolAGUNet(
            encoder_pretrained=False,
            freeze_encoder=bool(config.architecture.freeze_encoder),
            bottleneck_channels=int(config.architecture.bottleneck_channels),
            dec_channels=tuple(int(value) for value in config.architecture.decoder_channels),
        )
    raise SegmentationEvaluationError(
        f"Unsupported segmentation architecture implementation: {implementation}"
    )


def _prepare_cuda_memory_tracking(device: torch.device) -> int:
    """Initialize the allocator before resetting peaks on the selected CUDA device."""

    torch.cuda.set_device(device)
    allocator_initialization = torch.empty(1, device=device)
    del allocator_initialization
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)
    return device_index


def validate_checkpoint(
    path: Path,
    model: torch.nn.Module,
    expected_sha256: str,
    expected_architecture: str = "approximate-ag-unet-figure3-v1",
) -> dict[str, Any]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise SegmentationEvaluationError(
            f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    payload = load_checkpoint(
        path, model=model, expected_architecture=expected_architecture
    )
    return {"path": str(path.resolve()), "sha256": actual, "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
            "architecture": payload.get("architecture")}


def _write_outputs(output: Path, records: list[dict[str, Any]], report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_image.jsonl").write_text(
        "".join(_canonical_json(record) + "\n" for record in records), encoding="utf-8"
    )
    fields = list(records[0])
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    (output / "table3.json").write_text(_canonical_json(report) + "\n", encoding="utf-8")
    primary = report["aggregates"]["primary_per_image_macro"] or {}
    micro = report["aggregates"]["diagnostic_global_micro"] or {}
    rows = [
        {"role": "primary", "aggregation": "per_image_macro", **primary},
        {"role": "diagnostic", "aggregation": "global_micro", **micro},
    ]
    fields = sorted({key for row in rows for key in row})
    with (output / "table3.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def evaluate(
    *, config_path: Path, data_root: Path, checkpoint: Path, output_root: Path,
    device_name: str, model_factory: Callable[[], torch.nn.Module] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    config = load_segmentation_config(config_path)
    if config.run.kind != "real_evaluation":
        raise SegmentationEvaluationError("M10 requires run.kind=real_evaluation")
    if ROOT == output_root.resolve() or ROOT in output_root.resolve().parents:
        raise SegmentationEvaluationError("Evaluation output must remain outside the Git worktree")
    manifest_path = ROOT / str(config.data.test.manifest)
    actual_manifest_sha256 = file_sha256(manifest_path)
    if actual_manifest_sha256 != str(config.data.test.manifest_sha256):
        raise SegmentationEvaluationError("Test manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(manifest_path)
    if len(manifest.samples) != EXPECTED_SAMPLE_COUNT or manifest.split_role != "test":
        raise SegmentationEvaluationError("M10 requires exactly 233 ordered test samples")
    if str(config.evaluation.dont_care) == "excluded_by_validity_mask" and any(
        sample.validity_annotation_path is None for sample in manifest.samples
    ):
        raise SegmentationEvaluationError(
            "Don\'t Care exclusion requires paired validity annotations for all 233 test samples"
        )
    dataset = SegmentationDataset(manifest, data_root, verify_checksums=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SegmentationEvaluationError("CUDA is unavailable")
    cuda_device_index = None
    if device.type == "cuda":
        cuda_device_index = _prepare_cuda_memory_tracking(device)
    seed_everything(int(config.evaluation.seed), deterministic_algorithms=True)
    model = (model_factory or (lambda: _model_from_config(config)))()
    checkpoint_record = validate_checkpoint(
        checkpoint,
        model,
        str(config.checkpoint.sha256),
        str(config.architecture.identity),
    )
    model.to(device=device, dtype=torch.float32).eval()
    predictions = output_root / str(config.run.name) / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(manifest.samples):
        try:
            item = dataset[index]
            image, target_image = item["image"], item["mask"]
            validity_image = item.get("validity")
            height, width = target_image.height, target_image.width
            with torch.inference_mode():
                logits = model(_preprocess(image).unsqueeze(0).to(device))
            if tuple(logits.shape) != (1, 1, 128, 128):
                raise SegmentationEvaluationError(f"Expected logits [1,1,128,128], got {list(logits.shape)}")
            native = (
                logits.sigmoid() >= float(config.evaluation.probability_threshold)
            )[0, 0].cpu()
            evaluation_resolution = str(config.evaluation.evaluation_resolution)
            if evaluation_resolution == "native_128x128":
                prediction = native
                target = resize_binary_target_to_native_logits(
                    target_image,
                    size=tuple(int(value) for value in config.transforms.mask_size),
                    threshold=float(config.evaluation.gt_binarization_threshold),
                )
                validity = (
                    resize_validity_to_native_logits(
                        validity_image,
                        size=tuple(int(value) for value in config.transforms.mask_size),
                    )
                    if validity_image is not None
                    else None
                )
            elif evaluation_resolution == "original_gt_hxw":
                prediction = F.interpolate(
                    native[None, None].float(),
                    size=(height, width),
                    mode="nearest",
                ).bool()[0, 0]
                target = torch.from_numpy(
                    np.asarray(target_image, dtype=np.uint8).copy() > 0
                )
                validity = (
                    torch.from_numpy(
                        np.asarray(validity_image, dtype=np.uint8).copy() > 0
                    )
                    if validity_image is not None
                    else None
                )
            else:
                raise SegmentationEvaluationError(
                    "Unsupported evaluation resolution: {}".format(
                        evaluation_resolution
                    )
                )
            dont_care = str(config.evaluation.dont_care)
            if dont_care == "excluded_by_validity_mask" and validity is None:
                raise SegmentationEvaluationError(
                    "Don\'t Care exclusion requires paired test validity annotations"
                )
            if dont_care == "foreground":
                validity = None
            elif dont_care != "excluded_by_validity_mask":
                raise SegmentationEvaluationError(
                    "Unsupported Don\'t Care evaluation policy: {}".format(dont_care)
                )
            record = _sample_record(sample.sample_id, prediction, target, validity)
            Image.fromarray(prediction.numpy().astype(np.uint8) * 255, mode="L").save(
                predictions / f"{sample.sample_id}.png"
            )
        except Exception as error:  # retained as an auditable failed sample
            record = {"sample_id": sample.sample_id, "status": "failure",
                      "failure_type": type(error).__name__, "failure_message": str(error),
                      "tp": None, "fp": None, "fn": None, "foreground_iou": None,
                      "foreground_f1": None, "both_empty": None,
                      "valid_pixel_count": None, "ignored_pixel_count": None}
        records.append(record)
    aggregates = aggregate_records(records, expected_count=EXPECTED_SAMPLE_COUNT)
    resolved = OmegaConf.to_container(config, resolve=True)
    runtime = {
        "device": str(device),
        "torch": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started_at,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(cuda_device_index)
            if cuda_device_index is not None else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(cuda_device_index)
            if cuda_device_index is not None else None
        ),
    }
    if cuda_device_index is not None:
        runtime["gpu"] = torch.cuda.get_device_name(cuda_device_index)
    report = {
        "schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "result_class": "approximate reproduction",
        "dataset": {"manifest": str(config.data.test.manifest),
                    "manifest_sha256": actual_manifest_sha256, "sample_count": len(dataset)},
        "checkpoint": checkpoint_record, "resolved_config": resolved,
        "resolved_config_sha256": sha256(_canonical_json(resolved).encode()).hexdigest(),
        "aggregates": aggregates, "runtime": runtime,
        "strict_blockers": list(config.strict_blockers),
    }
    _write_outputs(output_root / str(config.run.name), records, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = evaluate(config_path=args.config, data_root=args.data_root,
                      checkpoint=args.checkpoint, output_root=args.output_root,
                      device_name=args.device)
    primary = report["aggregates"]["primary_per_image_macro"] or {}
    print(
        "evaluation_complete samples={} failures={} macro_fgIoU={} macro_F1={} "
        "device={} elapsed={:.1f}s".format(
            report["aggregates"]["sample_count"],
            report["aggregates"]["failure_count"],
            primary.get("foreground_iou", "n/a"),
            primary.get("foreground_f1", "n/a"),
            report["runtime"]["device"],
            report["runtime"]["elapsed_seconds"],
        ),
        flush=True,
    )
    if not report["aggregates"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
