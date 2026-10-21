"""Auditable training primitives for APPROXIMATE segmentation runs."""

from __future__ import annotations

import platform
import random
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    epochs: int
    seed: int
    adam_betas: tuple[float, float]
    adam_epsilon: float
    learning_rate_schedule: str
    checkpoint_selection: str
    deterministic_algorithms: bool
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    loss: str = "BCEWithLogitsLoss"
    separation_weight: float = 1.0
    reproduction_mode: str = "approximate"
    claim_label: str = "approximate reproduction"

    def __post_init__(self) -> None:
        if (self.optimizer, self.learning_rate, self.weight_decay, self.batch_size) != (
            "Adam", 1e-4, 0.01, 8
        ):
            raise ValueError("Optimizer, lr, weight decay, and batch size must match paper values")
        if self.loss not in ("BCEWithLogitsLoss", "masked_BCEWithLogitsLoss"):
            raise ValueError("Unsupported explicit BCE loss contract")
        if not (1.0 <= self.separation_weight <= 20.0):
            raise ValueError("Separation weight must be in [1,20]")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        beta1, beta2 = self.adam_betas
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and self.adam_epsilon > 0):
            raise ValueError("Invalid explicit Adam beta/epsilon assumptions")
        if self.learning_rate_schedule != "none" or self.checkpoint_selection not in (
            "last", "training_foreground_iou", "validation_foreground_iou",
            "validation_word_f1",
        ):
            raise ValueError(
                "Training supports schedule=none and checkpoint selection by last or "
                "training_foreground_iou or validation_foreground_iou or validation_word_f1"
            )
        if self.reproduction_mode != "approximate" or self.claim_label != "approximate reproduction":
            raise ValueError("Training engine is not a STRICT segmentation implementation")


def seed_everything(seed: int, *, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def build_optimizer(
    model: nn.Module, config: SegmentationTrainingConfig
) -> torch.optim.Adam:
    return torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
    )


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, validity: torch.Tensor
) -> torch.Tensor:
    """Mean BCE over valid pixels only; invalid pixels have exactly zero gradient."""

    if logits.shape != targets.shape or logits.shape != validity.shape:
        raise ValueError(
            "Masked BCE shape mismatch: logits={}, targets={}, validity={}".format(
                tuple(logits.shape), tuple(targets.shape), tuple(validity.shape)
            )
        )
    if torch.any((validity != 0) & (validity != 1)):
        raise ValueError("Masked BCE validity must be binary")
    valid_count = validity.sum()
    if valid_count.item() <= 0:
        raise ValueError("Masked BCE requires at least one valid pixel")
    per_pixel = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return (per_pixel * validity).sum() / valid_count


def separation_weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
    separation: torch.Tensor,
    *,
    separation_weight: float,
) -> torch.Tensor:
    """Additional false-positive cost on valid inter-word separator pixels."""

    if logits.shape != separation.shape or targets.shape != separation.shape:
        raise ValueError("Separation shape mismatch")
    if torch.any((separation != 0) & (separation != 1)):
        raise ValueError("Separation must be binary")
    if torch.any((separation > 0) & ((targets > 0) | (validity == 0))):
        raise ValueError("Separation must be valid zero-target pixels")
    if separation_weight <= 1.0:
        return masked_bce_with_logits(logits, targets, validity)
    weights = validity * (1.0 + (separation_weight - 1.0) * separation)
    per_pixel = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    return (per_pixel * weights).sum() / weights.sum()


def _to_compute_device(
    tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    moved = tensor.to(
        device=device,
        dtype=dtype,
        non_blocking=device.type == "cuda",
    )
    if moved.device.type != device.type or (
        device.index is not None and moved.device.index != device.index
    ):
        raise RuntimeError(
            "Tensor device placement failed: requested {}, got {}".format(
                device, moved.device
            )
        )
    return moved


def _require_output_device(logits: torch.Tensor, images: torch.Tensor) -> None:
    if logits.device != images.device:
        raise RuntimeError(
            "Segmentation logits left the compute device: images={}, logits={}".format(
                images.device, logits.device
            )
        )


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_steps: int | None = None,
    separation_weight: float = 1.0,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    steps = 0
    ordered_ids: list[str] = []
    image_shape: list[int] | None = None
    target_shape: list[int] | None = None
    for batch in batches:
        if max_steps is not None and steps >= max_steps:
            break
        images = _to_compute_device(
            batch["image"], device=device, dtype=torch.float32
        )
        targets = _to_compute_device(
            batch["mask"], device=device, dtype=torch.float32
        )
        validity = batch.get("validity")
        if validity is None:
            validity = torch.ones_like(targets)
        else:
            validity = _to_compute_device(
                validity, device=device, dtype=torch.float32
            )
        if image_shape is None:
            image_shape = list(images.shape)
            target_shape = list(targets.shape)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        _require_output_device(logits, images)
        if logits.shape != targets.shape:
            raise ValueError(
                f"BCE shape mismatch: logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
            )
        if separation_weight > 1.0:
            if "separation" not in batch:
                raise ValueError("Weighted training requires per-sample separation masks")
            separation = _to_compute_device(
                batch["separation"], device=device, dtype=torch.float32
            )
            loss = separation_weighted_bce_with_logits(
                logits, targets, validity, separation,
                separation_weight=separation_weight,
            )
        else:
            loss = masked_bce_with_logits(logits, targets, validity)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite segmentation loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        steps += 1
        sample_ids = batch.get("sample_id")
        if isinstance(sample_ids, (list, tuple)):
            ordered_ids.extend(str(sample_id) for sample_id in sample_ids)
    if not total_samples:
        raise ValueError("Training loader yielded no samples")
    metrics: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "steps": steps,
        "image_shape": image_shape,
        "mask_shape": target_shape,
        "compute_device": str(logits.device),
    }
    if ordered_ids:
        metrics["sample_order_sha256"] = sha256(
            "".join(f"{sample_id}\n" for sample_id in ordered_ids).encode("utf-8")
        ).hexdigest()
    return metrics


def measure_foreground_iou(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    probability_threshold: float,
    aggregation: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Measure training-set foreground IoU for author-protocol checkpoint selection."""

    if aggregation not in ("global_micro", "per_image_macro"):
        raise ValueError("Unsupported foreground-IoU aggregation")
    model.eval()
    intersections = 0
    unions = 0
    per_image = []
    samples = 0
    steps = 0
    with torch.inference_mode():
        for batch in batches:
            if max_steps is not None and steps >= max_steps:
                break
            images = _to_compute_device(
                batch["image"], device=device, dtype=torch.float32
            )
            targets = _to_compute_device(
                batch["mask"], device=device, dtype=torch.bool
            )
            validity_value = batch.get("validity")
            validity = (
                torch.ones_like(targets)
                if validity_value is None
                else _to_compute_device(
                    validity_value, device=device, dtype=torch.bool
                )
            )
            logits = model(images)
            _require_output_device(logits, images)
            predictions = torch.sigmoid(logits) >= probability_threshold
            if predictions.shape != targets.shape or validity.shape != targets.shape:
                raise ValueError("Foreground-IoU tensor shapes do not match")
            batch_intersections = ((predictions & targets) & validity).flatten(1).sum(1)
            batch_unions = ((predictions | targets) & validity).flatten(1).sum(1)
            batch_counts = torch.stack(
                (batch_intersections, batch_unions), dim=1
            ).cpu().tolist()
            for intersection, union in batch_counts:
                intersections += intersection
                unions += union
                per_image.append(1.0 if union == 0 else intersection / union)
                samples += 1
            steps += 1
    if not samples:
        raise ValueError("Foreground-IoU loader yielded no samples")
    value = (
        (1.0 if unions == 0 else intersections / unions)
        if aggregation == "global_micro"
        else sum(per_image) / len(per_image)
    )
    return {
        "foreground_iou": value,
        "aggregation": aggregation,
        "probability_threshold": probability_threshold,
        "samples": samples,
        "steps": steps,
        "intersection": intersections,
        "union": unions,
        "compute_device": str(logits.device),
    }


def select_checkpoint_record(
    records: Iterable[Mapping[str, Any]], checkpoint_selection: str
) -> Mapping[str, Any]:
    records_list = list(records)
    if not records_list:
        raise ValueError("Checkpoint selection requires epoch records")
    if checkpoint_selection == "last":
        return records_list[-1]
    if checkpoint_selection == "validation_foreground_iou":
        try:
            return max(
                records_list,
                key=lambda record: float(record["validation"]["foreground_iou"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Validation foreground-IoU selection metadata is missing") from error
    if checkpoint_selection != "training_foreground_iou":
        raise ValueError("Unsupported checkpoint selection")
    try:
        return max(
            records_list,
            key=lambda record: float(record["training_selection"]["foreground_iou"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Training foreground-IoU selection metadata is missing") from error


def validate_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Measure validation BCE without changing parameters or BatchNorm state."""

    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0
    steps = 0
    ordered_ids: list[str] = []
    image_shape: list[int] | None = None
    target_shape: list[int] | None = None
    with torch.inference_mode():
        for batch in batches:
            if max_steps is not None and steps >= max_steps:
                break
            images = _to_compute_device(
                batch["image"], device=device, dtype=torch.float32
            )
            targets = _to_compute_device(
                batch["mask"], device=device, dtype=torch.float32
            )
            if image_shape is None:
                image_shape = list(images.shape)
                target_shape = list(targets.shape)
            logits = model(images)
            _require_output_device(logits, images)
            if logits.shape != targets.shape:
                raise ValueError(
                    "Validation BCE shape mismatch: "
                    f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
                )
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")
            batch_size = int(images.shape[0])
            total_loss += float(loss) * batch_size
            total_samples += batch_size
            steps += 1
            sample_ids = batch.get("sample_id")
            if isinstance(sample_ids, (list, tuple)):
                ordered_ids.extend(str(sample_id) for sample_id in sample_ids)
    if not total_samples:
        raise ValueError("Validation loader yielded no samples")
    metrics: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "steps": steps,
        "image_shape": image_shape,
        "mask_shape": target_shape,
    }
    if ordered_ids:
        metrics["sample_order_sha256"] = sha256(
            "".join(f"{sample_id}\n" for sample_id in ordered_ids).encode("utf-8")
        ).hexdigest()
    return metrics


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG used by the declared seed policy."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(state) != required:
        raise ValueError(
            f"Checkpoint RNG state fields are incompatible: expected {sorted(required)}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda_all"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state does not match visible CUDA devices")
        torch.cuda.set_rng_state_all(cuda_state)


def runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: SegmentationTrainingConfig,
    resolved_config: str,
    architecture: str,
    result_class: str = "smoke-test result",
    run_contract: Mapping[str, Any] | None = None,
    train_metrics: Mapping[str, Any] | None = None,
    validation_metrics: Mapping[str, Any] | None = None,
    training_selection_metrics: Mapping[str, Any] | None = None,
) -> str:
    model_architecture = getattr(model, "architecture_identity", None)
    if model_architecture is not None and architecture != model_architecture:
        raise ValueError(
            "Checkpoint architecture identity does not match the model: "
            f"expected {model_architecture!r}, got {architecture!r}"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "reproduction_mode": config.reproduction_mode,
        "claim_label": config.claim_label,
        "result_class": result_class,
        "architecture": architecture,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "training_config": asdict(config),
        "resolved_config": resolved_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "environment": runtime_environment(),
        "run_contract": dict(run_contract or {}),
        "train_metrics": dict(train_metrics or {}),
        "validation_metrics": dict(validation_metrics or {}),
        "training_selection_metrics": dict(training_selection_metrics or {}),
    }
    checkpoint_metadata = getattr(model, "checkpoint_metadata", None)
    if checkpoint_metadata is not None:
        payload.update(checkpoint_metadata())
    torch.save(payload, temporary)
    temporary.replace(destination)
    return _file_sha256(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_architecture: str,
    expected_run_contract: Mapping[str, Any] | None = None,
    expected_training_config: SegmentationTrainingConfig | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported segmentation checkpoint schema")
    if payload.get("reproduction_mode") != "approximate":
        raise ValueError("Checkpoint is not labeled as approximate reproduction")
    if payload.get("claim_label") != "approximate reproduction":
        raise ValueError("Checkpoint claim label is missing or inconsistent")
    if payload.get("architecture") != expected_architecture:
        raise ValueError("Checkpoint architecture identity does not match the model")
    model_architecture = getattr(model, "architecture_identity", None)
    if model_architecture is not None and expected_architecture != model_architecture:
        raise ValueError(
            "Expected architecture identity does not match the model: "
            f"expected {model_architecture!r}, got {expected_architecture!r}"
        )
    if expected_run_contract is not None and payload.get("run_contract") != dict(
        expected_run_contract
    ):
        raise ValueError("Checkpoint run contract is incompatible with this training run")
    if expected_training_config is not None and payload.get("training_config") != asdict(
        expected_training_config
    ):
        raise ValueError("Checkpoint training configuration is incompatible with this run")
    compatible_loader = getattr(model, "load_compatible_checkpoint", None)
    if compatible_loader is not None:
        compatible_loader(payload)
    else:
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint model_state_dict must be a mapping")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f"Strict checkpoint state_dict is incompatible: {error}") from error
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("Checkpoint optimizer_state_dict must be a mapping")
        try:
            optimizer.load_state_dict(optimizer_state)
        except (ValueError, RuntimeError, KeyError) as error:
            raise ValueError(f"Checkpoint optimizer state is incompatible: {error}") from error
        parameter_device = next(
            (parameter.device for group in optimizer.param_groups for parameter in group["params"]),
            torch.device("cpu"),
        )
        for optimizer_value in optimizer.state.values():
            for key, value in optimizer_value.items():
                if isinstance(value, torch.Tensor):
                    optimizer_value[key] = value.to(parameter_device)
    if restore_rng:
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("Checkpoint RNG state is missing or incompatible")
        restore_rng_state(rng_state)
    return payload


__all__ = [
    "SegmentationTrainingConfig",
    "build_optimizer",
    "capture_rng_state",
    "load_checkpoint",
    "masked_bce_with_logits",
    "measure_foreground_iou",
    "restore_rng_state",
    "runtime_environment",
    "save_checkpoint",
    "seed_everything",
    "select_checkpoint_record",
    "train_one_epoch",
    "validate_one_epoch",
]
