"""Real-manifest segmentation training entry point.

This module preserves the historical M9 branch and separately supports the explicitly
labeled author-clarified APPROXIMATE protocol. STRICT configuration remains fail-closed.
It contains no synthetic or demo fallback.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import socket
import sys
import time
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision
from tqdm import tqdm
import yaml

from models.ag_unet_approx import ApproximateAGUNet
from models.unet_adv import AuthorProtocolAGUNet
from segmentation.config import load_segmentation_config
from segmentation.data import (
    ICDAR13_TASK22_MASK_CONVERSION,
    SegmentationDataset,
    SegmentationManifest,
    file_sha256,
    load_segmentation_manifest,
)
from segmentation.training import (
    SegmentationTrainingConfig,
    build_optimizer,
    load_checkpoint,
    runtime_environment,
    save_checkpoint,
    seed_everything,
    measure_foreground_iou,
    select_checkpoint_record,
    train_one_epoch,
    validate_one_epoch,
)
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_REGISTRY = ROOT / "provenance" / "ARTIFACTS.yaml"
ENVIRONMENT_METADATA = ROOT / "environments" / "main" / "environment_metadata.json"


class RealTrainingError(ValueError):
    """Raised when a real training run cannot satisfy its declared contract."""


class EpochShuffleSampler(Sampler[int]):
    """Epoch-addressable permutation independent of process RNG consumption."""

    def __init__(self, length: int, *, seed: int) -> None:
        if length < 1:
            raise ValueError("Training sampler requires at least one sample")
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        yield from torch.randperm(self.length, generator=generator).tolist()

    def __len__(self) -> int:
        return self.length


class DomainBalancedEpochSampler(Sampler[int]):
    """Deterministically oversample each prefixed dataset to the largest domain."""

    def __init__(self, sample_ids: list[str], *, seed: int) -> None:
        groups: dict[str, list[int]] = {}
        for index, sample_id in enumerate(sample_ids):
            domain, separator, _ = sample_id.partition(":")
            if not separator or not domain:
                raise ValueError(
                    "Balanced scene-text sampling requires '<dataset>:<id>' sample IDs"
                )
            groups.setdefault(domain, []).append(index)
        if len(groups) < 2:
            raise ValueError("Balanced scene-text sampling requires multiple datasets")
        self.groups = tuple((name, tuple(indices)) for name, indices in sorted(groups.items()))
        self.target_per_domain = max(len(indices) for _, indices in self.groups)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _domain_epoch(self, indices: tuple[int, ...], domain_index: int) -> list[int]:
        output: list[int] = []
        cycle = 0
        while len(output) < self.target_per_domain:
            generator = torch.Generator().manual_seed(
                self.seed
                + self.epoch * 1_000_003
                + domain_index * 10_007
                + cycle * 101
            )
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            output.extend(indices[position] for position in permutation)
            cycle += 1
        return output[: self.target_per_domain]

    def __iter__(self) -> Iterator[int]:
        per_domain = [
            self._domain_epoch(indices, domain_index)
            for domain_index, (_, indices) in enumerate(self.groups)
        ]
        domain_order = list(range(len(per_domain)))
        rotation = self.epoch % len(domain_order)
        domain_order = domain_order[rotation:] + domain_order[:rotation]
        for position in range(self.target_per_domain):
            for domain_index in domain_order:
                yield per_domain[domain_index][position]

    def __len__(self) -> int:
        return self.target_per_domain * len(self.groups)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(value) + "\n")


def _relative_root_path(root: Path, value: str, *, field: str, root_name: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RealTrainingError(f"{field} must be a relative POSIX path within {root_name}")
    resolved = (root / Path(relative)).resolve()
    if root not in resolved.parents:
        raise RealTrainingError(f"{field} escapes {root_name}")
    return resolved


def _artifact_index() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(ARTIFACT_REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("artifacts"), list):
        raise RealTrainingError("Segmentation artifact registry has an invalid schema")
    return {
        item["id"]: item
        for item in raw["artifacts"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def _validate_dataset_artifact(
    artifact_id: str, artifacts: Mapping[str, Mapping[str, Any]]
) -> None:
    artifact = artifacts.get(artifact_id)
    if artifact is None:
        raise RealTrainingError(f"Dataset artifact '{artifact_id}' is not registered")
    if artifact.get("artifact_type") not in ("dataset", "dataset_split"):
        raise RealTrainingError(
            f"Artifact '{artifact_id}' is not a registered segmentation dataset"
        )
    if artifact.get("evidence_classification") != "UPSTREAM_PRIMARY":
        raise RealTrainingError(f"Dataset artifact '{artifact_id}' lacks upstream-primary evidence")
    if artifact.get("status") not in ("RESOLVED", "PARTIALLY_RESOLVED"):
        raise RealTrainingError(f"Dataset artifact '{artifact_id}' is not sufficiently identified")


def _validate_real_manifest(
    manifest: SegmentationManifest,
    *,
    expected_artifact_id: str,
    expected_role: str,
    expected_split_policy_fingerprint: str,
    expected_mask_policy_fingerprint: str,
) -> None:
    if (
        manifest.reproduction_mode != "approximate"
        or manifest.claim_label != "approximate reproduction"
    ):
        raise RealTrainingError("M9 APPROXIMATE training requires approximate-labeled manifests")
    if manifest.artifact_id != expected_artifact_id:
        raise RealTrainingError(
            f"Manifest artifact_id must match configured artifact '{expected_artifact_id}'"
        )
    if manifest.split_role != expected_role:
        raise RealTrainingError(f"Manifest split_role must be '{expected_role}'")
    if manifest.split_policy is None:
        raise RealTrainingError("Manifest split_policy is required for M9 real training")
    expected_policy = {
        "artifact_id": "ART-ICDAR13-M9-APPROX-SPLIT-POLICY",
        "algorithm_id": "icdar13-task2.2-sha256-ranked-holdout-v1",
        "classification": "ASSUMPTION",
        "seed": 202503,
        "policy_fingerprint_sha256": expected_split_policy_fingerprint,
    }
    if dict(manifest.split_policy) != expected_policy:
        raise RealTrainingError("Manifest split policy does not match the frozen M9 policy")
    lowered = f"{manifest.dataset} {manifest.split}".lower()
    if any(marker in lowered for marker in ("synthetic", "demo", "smoke", "test")):
        raise RealTrainingError("Real training rejects synthetic/demo/smoke/test manifests")
    expected_annotation = {
        "artifact_id": "ART-ICDAR13-M9-APPROX-TRAINING-MASK-POLICY",
        "classification": "ASSUMPTION",
        "source_format": "ICDAR13 Task 2.2 color-coded per-character RGB BMP",
        "target_conversion": ICDAR13_TASK22_MASK_CONVERSION,
        "policy_fingerprint_sha256": expected_mask_policy_fingerprint,
        "foreground_conversion": "exact RGB white -> 0; every other RGB -> 1",
        "conversion_classification": "REPOSITORY_SPECIFIED",
        "dont_care_training_behavior": "foreground",
        "dont_care_classification": "ASSUMPTION",
        "txt_role": "audit/provenance only; does not alter BCE targets",
        "ignore_mask": False,
        "masked_bce": False,
        "ignore_index": None,
        "ambiguity_ids": ["DS-003"],
        "training_ready": True,
    }
    if dict(manifest.annotation_policy) != expected_annotation:
        raise RealTrainingError(
            "Manifest annotation_policy does not match the frozen M9 training-mask policy"
        )


def _manifest_identity(path: Path, manifest: SegmentationManifest) -> dict[str, Any]:
    sample_records = [
        {
            "id": sample.sample_id,
            "image": str(sample.image_path),
            "mask": str(sample.mask_path),
            "image_sha256": sample.image_sha256,
            "mask_sha256": sample.mask_sha256,
            "validity_annotation": (
                str(sample.validity_annotation_path)
                if sample.validity_annotation_path is not None
                else None
            ),
        }
        for sample in manifest.samples
    ]
    return {
        "artifact_id": manifest.artifact_id,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "split_role": manifest.split_role,
        "manifest_sha256": file_sha256(path),
        "sample_count": len(manifest.samples),
        "ordered_samples_sha256": sha256(
            _canonical_json(sample_records).encode("utf-8")
        ).hexdigest(),
        "annotation_policy": _json_safe(manifest.annotation_policy),
        "split_policy": _json_safe(manifest.split_policy),
        "acquisition": _json_safe(manifest.acquisition),
        "source_manifests": list(manifest.source_manifests),
    }


def _validate_author_protocol_manifest(
    manifest: SegmentationManifest, *, expected_artifact_id: str
) -> None:
    if manifest.artifact_id != expected_artifact_id or manifest.split_role != "train":
        raise RealTrainingError("Author protocol requires the explicit ICDAR13 training manifest")
    if len(manifest.samples) != 229:
        raise RealTrainingError("Author protocol requires all 229 ICDAR13 training images")
    if manifest.split_policy is None or dict(manifest.split_policy) != {
        "artifact_id": "ART-ICDAR13-AUTHOR-PROTOCOL-ALL-229",
        "algorithm_id": "icdar13-task2.2-all-229-v1",
        "classification": "PAPER_SPECIFIED",
        "evidence_basis": "AUTHOR_CLARIFIED",
        "validation_count": 0,
    }:
        raise RealTrainingError("Author-protocol all-229 split metadata is incompatible")
    if manifest.annotation_policy.get("target_conversion") != (
        "icdar13-task2.2-author-validity-mask-v1"
    ) or manifest.annotation_policy.get("masked_bce") is not True:
        raise RealTrainingError("Author protocol requires masked ICDAR13 targets")
    if any(sample.validity_annotation_path is None for sample in manifest.samples):
        raise RealTrainingError("Every author-protocol sample requires a validity annotation")


def _validate_scene_text_training_manifest(
    manifest: SegmentationManifest,
    *,
    expected_artifact_id: str,
    expected_sample_count: int,
    expected_datasets: tuple[str, ...],
) -> None:
    if (
        manifest.reproduction_mode != "approximate"
        or manifest.claim_label != "approximate reproduction"
        or manifest.split_role != "train"
    ):
        raise RealTrainingError("Scene-text training requires an approximate TRAIN manifest")
    if manifest.artifact_id != expected_artifact_id:
        raise RealTrainingError("Scene-text manifest artifact ID does not match the config")
    if len(manifest.samples) != expected_sample_count:
        raise RealTrainingError(
            f"Scene-text manifest expected {expected_sample_count} samples; "
            f"resolved {len(manifest.samples)}"
        )
    if manifest.annotation_policy.get("target_conversion") != (
        "binary-mask-with-validity-v1"
    ) or manifest.annotation_policy.get("masked_bce") is not True:
        raise RealTrainingError("Scene-text training requires canonical binary masks and validity")
    if any(sample.validity_annotation_path is None for sample in manifest.samples):
        raise RealTrainingError("Every scene-text sample requires a binary validity mask")
    observed = {sample.sample_id.partition(":")[0] for sample in manifest.samples}
    if observed != set(expected_datasets):
        raise RealTrainingError(
            f"Scene-text manifest datasets differ: expected={sorted(expected_datasets)}, "
            f"observed={sorted(observed)}"
        )


def _assert_validation_separation(
    train_dataset: SegmentationDataset, validation_dataset: SegmentationDataset
) -> None:
    train_ids = {sample.sample_id for sample in train_dataset.manifest.samples}
    validation_ids = {sample.sample_id for sample in validation_dataset.manifest.samples}
    duplicate_ids = sorted(train_ids & validation_ids)
    if duplicate_ids:
        raise RealTrainingError(f"Training/validation sample IDs overlap: {duplicate_ids[:5]}")
    train_paths = {path for pair in train_dataset.paths for path in pair if path is not None}
    validation_paths = {path for pair in validation_dataset.paths for path in pair if path is not None}
    duplicate_paths = sorted(str(path) for path in train_paths & validation_paths)
    if duplicate_paths:
        raise RealTrainingError(
            f"Training/validation resolved paths overlap: {duplicate_paths[:5]}"
        )


def _transform(
    item: Any, *, seed: int, augmentation_enabled: bool = True
) -> PairedSegmentationTransform:
    return PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=(
                float(item.horizontal_flip_probability) if augmentation_enabled else 0.0
            ),
            rotation_probability=(
                float(item.rotation_probability) if augmentation_enabled else 0.0
            ),
            rotation_degrees=(
                tuple(float(value) for value in item.rotation_degrees)
                if augmentation_enabled
                else (0.0, 0.0)
            ),
            image_interpolation=str(item.image_interpolation),
            mask_interpolation=str(item.mask_interpolation),
            image_fill=tuple(int(value) for value in item.image_fill),
            mask_fill=int(item.mask_fill),
            normalization=str(item.normalization),
            resize_geometry=str(item.resize_geometry),
            operation_order=str(item.operation_order),
            target_binarization_threshold=float(
                item.get("target_binarization_threshold", 0.5)
            ),
            validity_interpolation=str(item.get("validity_interpolation", "nearest")),
        ),
        base_seed=seed,
    )


def _training_config(config: DictConfig) -> SegmentationTrainingConfig:
    item = config.training
    return SegmentationTrainingConfig(
        epochs=int(item.epochs),
        seed=int(item.seed),
        adam_betas=tuple(float(value) for value in item.adam_betas),
        adam_epsilon=float(item.adam_epsilon),
        learning_rate_schedule=str(item.learning_rate_schedule),
        checkpoint_selection=str(item.checkpoint_selection),
        deterministic_algorithms=bool(item.deterministic_algorithms),
        loss=str(item.loss),
        separation_weight=float(item.get("separation_weight", 1.0)),
    )


def _prepare_run_directory(output_root: Path, run_name: str, *, resume: Path | None) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise RealTrainingError("run.name must be one safe path component")
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / run_name
    if resume is None and run_directory.exists() and any(run_directory.iterdir()):
        raise RealTrainingError(
            f"Fresh run output already exists and will not be overwritten: {run_directory}"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def _invocation_final_epoch(
    *,
    configured_epochs: int,
    start_epoch: int,
    qualification_max_steps: Optional[int],
    max_epochs_this_invocation: Optional[int],
) -> int:
    """Keep a qualification invocation to one epoch and its explicit step cap."""

    if qualification_max_steps is not None:
        return start_epoch
    if max_epochs_this_invocation is not None:
        return min(configured_epochs, start_epoch + max_epochs_this_invocation - 1)
    return configured_epochs


def _existing_epoch_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RealTrainingError(f"Invalid epoch log line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise RealTrainingError(f"Invalid epoch log record at line {line_number}")
        records.append(record)
    if [record.get("epoch") for record in records] != list(range(1, len(records) + 1)):
        raise RealTrainingError("Existing epoch log is not contiguous from epoch 1")
    return records


def _runtime_identifiers(device: torch.device) -> dict[str, Any]:
    environment = runtime_environment()
    environment.update(
        {
            "torchvision": torchvision.__version__,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "visible_cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            "environment_metadata_sha256": (
                file_sha256(ENVIRONMENT_METADATA) if ENVIRONMENT_METADATA.is_file() else None
            ),
        }
    )
    return environment


def _peak_cuda(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device)), int(
        torch.cuda.max_memory_reserved(device)
    )


def _duration(seconds: float) -> str:
    minutes, remaining = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return "{}h{:02d}m{:04.1f}s".format(hours, minutes, remaining)
    if minutes:
        return "{}m{:04.1f}s".format(minutes, remaining)
    return "{:.1f}s".format(remaining)


def _memory(value: Optional[int]) -> str:
    return "n/a" if value is None else "{:.1f}MiB".format(value / (1024 ** 2))


def _resolution(value: Any) -> str:
    return "{}x{}".format(int(value[0]), int(value[1]))


def _progress_batches(batches: Any, *, epoch: int, epochs: int, phase: str) -> Any:
    """Add a terminal-only progress bar without changing iteration order."""

    return tqdm(
        batches,
        total=len(batches),
        desc="epoch={:03d}/{:03d} {}".format(epoch, epochs, phase),
        unit="batch",
        mininterval=5.0,
        leave=False,
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    )


def _print_training_start(
    *,
    config_path: Path,
    protocol: str,
    device: torch.device,
    gpu_name: Optional[str],
    train_images: int,
    batch_size: int,
    epochs: int,
    architecture: Mapping[str, Any],
    input_resolution: Any,
    logits_resolution: Any,
    checkpoint_selection: str,
    compute_device: str,
) -> None:
    total = int(architecture["parameter_count"])
    trainable = int(architecture["trainable_parameter_count"])
    frozen = total - trainable
    print(
        "training_start device={} gpu={} protocol={} config={} train_images={} "
        "batch_size={} epochs={} compute_device={}".format(
            device,
            gpu_name or "n/a",
            protocol,
            config_path,
            train_images,
            batch_size,
            epochs,
            compute_device,
        ),
        flush=True,
    )
    print(
        "training_model parameters_total={} parameters_trainable={} parameters_frozen={} "
        "encoder_frozen={} input_resolution={} logits_resolution={} checkpoint_selection={}".format(
            total,
            trainable,
            frozen,
            "yes" if bool(architecture["freeze_encoder"]) else "no",
            _resolution(input_resolution),
            _resolution(logits_resolution),
            checkpoint_selection,
        ),
        flush=True,
    )


def _print_epoch_progress(
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    train_fg_iou: Optional[float],
    best_fg_iou: Optional[float],
    learning_rate: float,
    elapsed_seconds: float,
    cuda_allocated: Optional[int],
    cuda_reserved: Optional[int],
    new_best_checkpoint: Optional[str],
    compute_device: str,
) -> None:
    fg_iou = "n/a" if train_fg_iou is None else "{:.6f}".format(train_fg_iou)
    best_iou = "n/a" if best_fg_iou is None else "{:.6f}".format(best_fg_iou)
    checkpoint = (
        " new_best_checkpoint={}".format(new_best_checkpoint)
        if new_best_checkpoint is not None
        else ""
    )
    print(
        "epoch={:03d}/{:03d} train_loss={:.6f} train_fgIoU={} best_fgIoU={} "
        "lr={:.6g} compute_device={} elapsed={} cuda_peak_allocated={} "
        "cuda_peak_reserved={}{}".format(
            epoch,
            epochs,
            train_loss,
            fg_iou,
            best_iou,
            learning_rate,
            compute_device,
            _duration(elapsed_seconds),
            _memory(cuda_allocated),
            _memory(cuda_reserved),
            checkpoint,
        ),
        flush=True,
    )


def _print_training_complete(
    *,
    status: str,
    best_epoch: int,
    best_fg_iou: Optional[float],
    checkpoint_path: Path,
    elapsed_seconds: float,
    cuda_allocated: Optional[int],
    cuda_reserved: Optional[int],
) -> None:
    best_iou = "n/a" if best_fg_iou is None else "{:.6f}".format(best_fg_iou)
    print(
        "training_complete status={} best_epoch={} best_train_fgIoU={} checkpoint={} "
        "elapsed={} cuda_peak_allocated={} cuda_peak_reserved={}".format(
            status,
            best_epoch,
            best_iou,
            checkpoint_path,
            _duration(elapsed_seconds),
            _memory(cuda_allocated),
            _memory(cuda_reserved),
        ),
        flush=True,
    )


def _architecture_record(model: nn.Module, config: DictConfig) -> dict[str, Any]:
    architecture = config.architecture
    implementation = str(architecture.implementation)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    fingerprint = getattr(model, "state_dict_schema_fingerprint", None)

    if implementation == "figure_derived_ag_unet":
        return {
            "identity": str(architecture.identity),
            "contract_schema_version": int(architecture.contract_schema_version),
            "contract_artifact": str(architecture.contract_artifact),
            "state_dict_schema_fingerprint": fingerprint,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "encoder_weights": str(architecture.encoder_weights),
            "encoder_weights_artifact_id": str(architecture.encoder_weights_artifact_id),
            "freeze_encoder": bool(architecture.freeze_encoder),
        }
    if implementation != "author_clarified_ag_unet":
        raise RealTrainingError(
            "Unsupported segmentation architecture implementation: {}".format(implementation)
        )

    record = {
        "implementation": implementation,
        "identity": str(architecture.identity),
        "contract_schema_version": int(architecture.contract_schema_version),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": parameter_count - trainable_parameter_count,
        "encoder": str(architecture.encoder),
        "encoder_weights": str(architecture.encoder_weights),
        "encoder_weights_artifact_id": str(architecture.encoder_weights_artifact_id),
        "freeze_encoder": bool(architecture.freeze_encoder),
        "frozen_batchnorm_state": bool(architecture.frozen_batchnorm_state),
        "bottleneck_channels": int(architecture.bottleneck_channels),
        "bottleneck_width_classification": str(
            architecture.bottleneck_width_classification
        ),
        "bottleneck_evidence_basis": str(architecture.bottleneck_evidence_basis),
        "bottleneck_block": str(architecture.bottleneck_block),
        "bottleneck_block_classification": str(
            architecture.bottleneck_block_classification
        ),
        "decoder_channels": [int(value) for value in architecture.decoder_channels],
        "decoder_operations_classification": str(
            architecture.decoder_operations_classification
        ),
        "attention_intermediate_channels": str(
            architecture.attention_intermediate_channels
        ),
        "attention_intermediate_channels_classification": str(
            architecture.attention_intermediate_channels_classification
        ),
        "output_stride": int(architecture.output_stride),
    }
    if fingerprint is not None:
        record["state_dict_schema_fingerprint"] = fingerprint
    return record


def _verify_model_device(model: nn.Module, device: torch.device) -> str:
    named_tensors = list(model.named_parameters()) + list(model.named_buffers())
    if not named_tensors:
        raise RuntimeError("Segmentation model has no parameters or buffers")
    mismatched = [
        "{}={}".format(name, tensor.device)
        for name, tensor in named_tensors
        if tensor.device.type != device.type
        or (device.index is not None and tensor.device.index != device.index)
    ]
    if mismatched:
        raise RuntimeError(
            "Segmentation model is not fully on {}: {}".format(
                device, ", ".join(mismatched[:5])
            )
        )
    return str(named_tensors[0][1].device)


def _failure_record(
    *,
    error: BaseException,
    run_directory: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    allocated, reserved = (
        _peak_cuda(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else (None, None)
    )
    return {
        "schema_version": 1,
        "status": "CUDA_OUT_OF_MEMORY" if "out of memory" in str(error).lower() else "FAILED",
        "success": False,
        "oom": "out of memory" in str(error).lower(),
        "error_type": type(error).__name__,
        "error": str(error),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "precision": "float32",
        "batch_size": batch_size,
        "image_shape": [batch_size, 3, 256, 256],
        "mask_shape": [batch_size, 1, 128, 128],
        "loss": None,
        "cuda_peak_allocated_bytes": allocated,
        "cuda_peak_reserved_bytes": reserved,
        "checkpoint_output_location": None,
        "run_output_location": str(run_directory),
    }


def run_training(
    *,
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    device_name: str,
    resume: str | Path | None = None,
    qualification_max_steps: int | None = None,
    manifest_root: str | Path = ROOT,
    _model_factory: Callable[[], nn.Module] | None = None,
    _max_epochs_this_invocation: int | None = None,
) -> dict[str, Any]:
    """Run a verified real-manifest training protocol and return its run manifest."""

    config = load_segmentation_config(config_path)
    if config.run.kind != "real_training":
        raise RealTrainingError("The real training entry point requires run.kind=real_training")
    if device_name not in ("cpu", "cuda"):
        raise RealTrainingError("device must be cpu or cuda")
    if qualification_max_steps is not None and qualification_max_steps < 1:
        raise RealTrainingError("qualification_max_steps must be positive")
    if _max_epochs_this_invocation is not None and _max_epochs_this_invocation < 1:
        raise RealTrainingError("_max_epochs_this_invocation must be positive")

    device = torch.device(device_name)
    resume_path = Path(resume).resolve() if resume is not None else None
    resolved_output_root = Path(output_root).resolve()
    resolved_manifest_root = Path(manifest_root).resolve()
    if not resolved_manifest_root.is_dir():
        raise RealTrainingError(f"--manifest-root does not exist: {resolved_manifest_root}")
    if resolved_output_root == ROOT or ROOT in resolved_output_root.parents:
        raise RealTrainingError("--output-root must be outside the repository")
    run_directory = _prepare_run_directory(
        resolved_output_root, str(config.run.name), resume=resume_path
    )
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.update(
        resolved,
        "runtime_overrides.qualification_max_steps",
        qualification_max_steps,
        force_add=True,
    )
    OmegaConf.update(
        resolved,
        "runtime_overrides.manifest_root",
        str(resolved_manifest_root),
        force_add=True,
    )
    resolved_config = OmegaConf.to_yaml(resolved, resolve=True)
    (run_directory / "resolved_config.yaml").write_text(resolved_config, encoding="utf-8")
    training_config = _training_config(config)
    batch_size = training_config.batch_size
    training_started_at = time.monotonic()

    try:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PENDING_PHYSICAL_HOST: CUDA is unavailable in this execution context; "
                "no CUDA result was produced"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            allocator_initialization = torch.empty(1, device=device)
            del allocator_initialization
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        artifacts = _artifact_index()
        protocol = str(config.training.get("protocol", ""))
        author_protocol = protocol == "author_clarified_icdar13_v1"
        scene_text_protocol = protocol in {
            "paper_hyperparameters_scene_text_v1",
            "paper_hyperparameters_multidataset_balanced_v1",
            "practical_totaltext_adaptive_word_boxes_500_v1",
            "practical_totaltext_curved_word_boxes_500_v1",
            "practical_totaltext_separated_words_500_v1",
            "practical_totaltext_tighter_words_500_v2",
        }
        training_iou_selection = author_protocol or scene_text_protocol
        author_architecture = (
            str(config.architecture.implementation) == "author_clarified_ag_unet"
        )
        train_artifact = str(config.data.train.artifact_id)
        if not scene_text_protocol:
            _validate_dataset_artifact(train_artifact, artifacts)
        root = Path(data_root).resolve()
        if not root.is_dir():
            raise RealTrainingError(f"--data-root does not exist or is not a directory: {root}")
        train_manifest_path = _relative_root_path(
            resolved_manifest_root,
            str(config.data.train.manifest),
            field="data.train.manifest",
            root_name="--manifest-root",
        )
        if scene_text_protocol:
            expected_manifest_sha256 = str(config.data.train.manifest_sha256)
            actual_manifest_sha256 = file_sha256(train_manifest_path)
            if actual_manifest_sha256 != expected_manifest_sha256:
                raise RealTrainingError(
                    "Scene-text training manifest SHA256 mismatch: expected {}, got {}".format(
                        expected_manifest_sha256, actual_manifest_sha256
                    )
                )
        train_manifest = load_segmentation_manifest(train_manifest_path, require_checksums=True)
        if (protocol in {"practical_totaltext_separated_words_500_v1", "practical_totaltext_tighter_words_500_v2"}
            and train_manifest.annotation_policy.get("target_source") != config.data.train.target_source):
            raise RealTrainingError("Separated-word training requires its qualified wider-gap target manifest")
        train_transform = _transform(config.transforms, seed=training_config.seed)
        train_dataset = SegmentationDataset(
            train_manifest, root, transform=train_transform, verify_checksums=True
        )
        if training_config.separation_weight > 1.0 and any(
            sample.separation_annotation_path is None for sample in train_manifest.samples
        ):
            raise RealTrainingError("Separation-weighted run requires every TRAIN separation annotation")
        selection_transform = None
        selection_loader = None
        validation_transform = None
        validation_loader = None
        if training_iou_selection:
            if author_protocol:
                _validate_author_protocol_manifest(
                    train_manifest, expected_artifact_id=train_artifact
                )
            else:
                _validate_scene_text_training_manifest(
                    train_manifest,
                    expected_artifact_id=train_artifact,
                    expected_sample_count=int(config.data.train.expected_sample_count),
                    expected_datasets=tuple(str(value) for value in config.data.train.datasets),
                )
            selection_transform = _transform(
                config.transforms,
                seed=training_config.seed,
                augmentation_enabled=False,
            )
            selection_dataset = SegmentationDataset(
                train_manifest,
                root,
                transform=selection_transform,
                verify_checksums=True,
            )
            datasets = {"train": _manifest_identity(train_manifest_path, train_manifest)}
            selection_loader = DataLoader(
                selection_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(config.training.num_workers),
                drop_last=False,
                pin_memory=device.type == "cuda" and training_iou_selection,
            )
        else:
            validation_artifact = str(config.data.validation.artifact_id)
            _validate_dataset_artifact(validation_artifact, artifacts)
            split_policy_artifact = str(config.data.split_policy_artifact_id)
            policy_record = artifacts.get(split_policy_artifact)
            if (
                policy_record is None
                or policy_record.get("artifact_type") != "dataset_split_policy"
                or policy_record.get("decision_classification") != "ASSUMPTION"
            ):
                raise RealTrainingError(
                    "Configured M9 split-policy artifact is missing or invalid"
                )
            split_policy_fingerprint = str(config.data.split_policy_fingerprint_sha256)
            mask_policy_artifact = str(config.data.training_mask_policy_artifact_id)
            mask_policy_record = artifacts.get(mask_policy_artifact)
            if (
                mask_policy_record is None
                or mask_policy_record.get("artifact_type") != "dataset_annotation_policy"
                or mask_policy_record.get("decision_classification") != "ASSUMPTION"
            ):
                raise RealTrainingError(
                    "Configured M9 training-mask-policy artifact is missing or invalid"
                )
            mask_policy_fingerprint = str(config.data.training_mask_policy_fingerprint_sha256)
            validation_manifest_path = _relative_root_path(
                resolved_manifest_root,
                str(config.data.validation.manifest),
                field="data.validation.manifest",
                root_name="--manifest-root",
            )
            if train_manifest_path == validation_manifest_path:
                raise RealTrainingError("Training and validation manifests must be distinct files")
            validation_manifest = load_segmentation_manifest(
                validation_manifest_path, require_checksums=True
            )
            _validate_real_manifest(
                train_manifest,
                expected_artifact_id=train_artifact,
                expected_role="train",
                expected_split_policy_fingerprint=split_policy_fingerprint,
                expected_mask_policy_fingerprint=mask_policy_fingerprint,
            )
            _validate_real_manifest(
                validation_manifest,
                expected_artifact_id=validation_artifact,
                expected_role="validation",
                expected_split_policy_fingerprint=split_policy_fingerprint,
                expected_mask_policy_fingerprint=mask_policy_fingerprint,
            )
            validation_transform = _transform(
                config.validation.transforms, seed=training_config.seed
            )
            validation_dataset = SegmentationDataset(
                validation_manifest,
                root,
                transform=validation_transform,
                verify_checksums=True,
            )
            _assert_validation_separation(train_dataset, validation_dataset)
            datasets = {
                "train": _manifest_identity(train_manifest_path, train_manifest),
                "validation": _manifest_identity(
                    validation_manifest_path, validation_manifest
                ),
            }
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(config.training.num_workers),
                drop_last=False,
                pin_memory=device.type == "cuda" and training_iou_selection,
            )
        if str(config.training.get("sampling", {}).get("strategy", "epoch_shuffle")) == (
            "balanced_by_dataset_oversample_to_largest"
        ):
            sampler: Sampler[int] = DomainBalancedEpochSampler(
                [sample.sample_id for sample in train_manifest.samples],
                seed=training_config.seed,
            )
        else:
            sampler = EpochShuffleSampler(len(train_dataset), seed=training_config.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=int(config.training.num_workers),
            drop_last=bool(config.training.drop_last),
            pin_memory=device.type == "cuda" and training_iou_selection,
        )
        seed_everything(
            training_config.seed,
            deterministic_algorithms=training_config.deterministic_algorithms,
        )
        if _model_factory is not None:
            model_factory = _model_factory
        elif author_architecture:
            model_factory = lambda: AuthorProtocolAGUNet(
                encoder_pretrained=True,
                freeze_encoder=True,
                bottleneck_channels=int(config.architecture.bottleneck_channels),
            )
        else:
            model_factory = lambda: ApproximateAGUNet(
                encoder_weights=str(config.architecture.encoder_weights),
                freeze_encoder=bool(config.architecture.freeze_encoder),
            )
        model = model_factory().to(device=device, dtype=torch.float32)
        compute_device = _verify_model_device(model, device)
        optimizer = build_optimizer(model, training_config)
        architecture = _architecture_record(model, config)
        resolved_sha256 = sha256(resolved_config.encode("utf-8")).hexdigest()
        run_contract = {
            "schema_version": 1,
            "reproduction_mode": str(config.reproduction_mode),
            "claim_label": str(config.claim_label),
            "result_class": (
                "smoke-test result"
                if qualification_max_steps is not None
                else str(config.run.result_class)
            ),
            "architecture": architecture,
            "datasets": datasets,
            "resolved_config_sha256": resolved_sha256,
            "seed": training_config.seed,
            "validation_protocol_classification": (
                None
                if training_iou_selection
                else str(config.validation.protocol_classification)
            ),
            "checkpoint_selection": training_config.checkpoint_selection,
            "qualification_max_steps": qualification_max_steps,
            "scientific_choices": OmegaConf.to_container(
                config.scientific_choices, resolve=True
            ),
            "strict_blockers": list(config.strict_blockers),
        }

        start_epoch = 1
        global_step = 0
        if resume_path is not None:
            if not resume_path.is_file():
                raise RealTrainingError(f"Resume checkpoint does not exist: {resume_path}")
            payload = load_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                expected_architecture=str(config.architecture.identity),
                expected_run_contract=run_contract,
                expected_training_config=training_config,
                restore_rng=True,
            )
            completed_epoch = payload.get("epoch")
            saved_step = payload.get("global_step")
            if not isinstance(completed_epoch, int) or completed_epoch < 1:
                raise RealTrainingError("Resume checkpoint has an invalid completed epoch")
            if not isinstance(saved_step, int) or saved_step < 0:
                raise RealTrainingError("Resume checkpoint has an invalid global step")
            start_epoch = completed_epoch + 1
            global_step = saved_step
            if start_epoch > training_config.epochs:
                raise RealTrainingError("Resume checkpoint already completed configured epochs")

        log_path = run_directory / "epochs.jsonl"
        records = _existing_epoch_records(log_path)
        if len(records) != start_epoch - 1:
            raise RealTrainingError(
                "Epoch log length does not match checkpoint continuation point: "
                f"records={len(records)}, expected={start_epoch - 1}"
            )
        runtime = _runtime_identifiers(device)
        final_epoch = _invocation_final_epoch(
            configured_epochs=training_config.epochs,
            start_epoch=start_epoch,
            qualification_max_steps=qualification_max_steps,
            max_epochs_this_invocation=_max_epochs_this_invocation,
        )

        protocol_value = config.training.get("protocol")
        _print_training_start(
            config_path=Path(config_path).resolve(),
            protocol=(
                str(protocol_value)
                if protocol_value is not None
                else str(config.architecture.implementation)
            ),
            device=device,
            gpu_name=runtime["gpu"],
            train_images=len(train_dataset),
            batch_size=batch_size,
            epochs=training_config.epochs,
            architecture=architecture,
            input_resolution=config.transforms.image_size,
            logits_resolution=config.transforms.mask_size,
            checkpoint_selection=training_config.checkpoint_selection,
            compute_device=compute_device,
        )

        for epoch in range(start_epoch, final_epoch + 1):
            epoch_started_at = time.monotonic()
            epoch_index = epoch - 1
            sampler.set_epoch(epoch_index)
            train_transform.set_epoch(epoch_index)
            if validation_transform is not None:
                validation_transform.set_epoch(epoch_index)
            if selection_transform is not None:
                selection_transform.set_epoch(epoch_index)
            train_metrics = train_one_epoch(
                model,
                _progress_batches(
                    train_loader,
                    epoch=epoch,
                    epochs=training_config.epochs,
                    phase="train",
                ),
                optimizer,
                device=device,
                max_steps=qualification_max_steps,
                separation_weight=training_config.separation_weight,
            )
            global_step += int(train_metrics["steps"])
            if training_iou_selection:
                if selection_loader is None:
                    raise RuntimeError("Author-protocol selection loader is missing")
                training_selection_metrics = measure_foreground_iou(
                    model,
                    _progress_batches(
                        selection_loader,
                        epoch=epoch,
                        epochs=training_config.epochs,
                        phase="train_fgIoU",
                    ),
                    device=device,
                    probability_threshold=float(
                        config.checkpoint_selection.probability_threshold
                    ),
                    aggregation=str(config.checkpoint_selection.aggregation),
                    max_steps=qualification_max_steps,
                )
                validation_metrics = {}
            else:
                if validation_loader is None:
                    raise RuntimeError("M9 validation loader is missing")
                validation_metrics = validate_one_epoch(
                    model,
                    _progress_batches(
                        validation_loader,
                        epoch=epoch,
                        epochs=training_config.epochs,
                        phase="validation",
                    ),
                    device=device,
                    max_steps=qualification_max_steps,
                )
                training_selection_metrics = {}
            checkpoint_path = run_directory / "checkpoints" / f"epoch-{epoch:04d}.pt"
            checkpoint_sha256 = save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                config=training_config,
                resolved_config=resolved_config,
                architecture=str(config.architecture.identity),
                result_class=str(run_contract["result_class"]),
                run_contract=run_contract,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                training_selection_metrics=training_selection_metrics,
            )
            checkpoint_path.with_suffix(".pt.sha256").write_text(
                f"{checkpoint_sha256}  {checkpoint_path.name}\n", encoding="utf-8"
            )
            record = {
                "schema_version": 1,
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "training_selection": training_selection_metrics,
                "checkpoint": str(checkpoint_path.relative_to(run_directory)),
                "checkpoint_sha256": checkpoint_sha256,
            }
            _append_jsonl(log_path, record)
            records.append(record)
            selected_so_far = select_checkpoint_record(
                records, training_config.checkpoint_selection
            )
            train_fg_iou_value = training_selection_metrics.get("foreground_iou")
            best_fg_iou_value = selected_so_far.get("training_selection", {}).get(
                "foreground_iou"
            )
            is_new_best = (
                training_config.checkpoint_selection == "training_foreground_iou"
                and selected_so_far is record
            )
            epoch_allocated, epoch_reserved = _peak_cuda(device)
            _print_epoch_progress(
                epoch=epoch,
                epochs=training_config.epochs,
                train_loss=float(train_metrics["loss"]),
                train_fg_iou=(
                    None if train_fg_iou_value is None else float(train_fg_iou_value)
                ),
                best_fg_iou=(
                    None if best_fg_iou_value is None else float(best_fg_iou_value)
                ),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                elapsed_seconds=time.monotonic() - epoch_started_at,
                cuda_allocated=epoch_allocated,
                cuda_reserved=epoch_reserved,
                new_best_checkpoint=(record["checkpoint"] if is_new_best else None),
                compute_device=str(train_metrics["compute_device"]),
            )

        training_complete = final_epoch == training_config.epochs
        qualification_passed = qualification_max_steps is not None
        if training_complete:
            run_status = "SUCCESS"
        elif qualification_passed:
            run_status = "QUALIFICATION_SUCCESS"
        else:
            run_status = "INCOMPLETE"
        allocated, reserved = _peak_cuda(device)
        last_record = records[-1]
        selected_record = select_checkpoint_record(
            records, training_config.checkpoint_selection
        )
        run_manifest = {
            "schema_version": 1,
            "status": run_status,
            "success": training_complete or qualification_passed,
            "training_complete": training_complete,
            "qualification": {
                "enabled": qualification_max_steps is not None,
                "passed": qualification_passed,
                "max_steps_per_phase": qualification_max_steps,
                "configured_epochs": training_config.epochs,
                "completed_epoch": final_epoch,
                "epochs_executed_this_invocation": final_epoch - start_epoch + 1,
                "training_steps_executed_this_invocation": sum(
                    int(record["train"]["steps"])
                    for record in records[start_epoch - 1 :]
                ),
                "full_training_complete": training_complete,
            },
            "oom": False,
            "reproduction_mode": str(config.reproduction_mode),
            "claim_label": str(config.claim_label),
            "result_class": str(run_contract["result_class"]),
            "architecture": architecture,
            "checkpoint": {
                "selection": training_config.checkpoint_selection,
                "path": selected_record["checkpoint"],
                "sha256": selected_record["checkpoint_sha256"],
                "epoch": selected_record["epoch"],
                "global_step": selected_record["global_step"],
                "training_foreground_iou": selected_record.get(
                    "training_selection", {}
                ).get("foreground_iou"),
                "optimizer_state_included": True,
                "rng_state_included": True,
            },
            "resolved_config": {
                "path": "resolved_config.yaml",
                "sha256": resolved_sha256,
            },
            "datasets": datasets,
            "seed": training_config.seed,
            "epochs": records,
            "validation": (
                {"enabled": False, "held_out_count": 0}
                if training_iou_selection
                else {
                    "source_explicit": True,
                    "leakage_policy": str(config.validation.leakage_policy),
                    "protocol_classification": str(
                        config.validation.protocol_classification
                    ),
                    "checkpoint_semantics": str(config.validation.checkpoint_semantics),
                }
            ),
            "scientific_mode": str(config.reproduction_mode),
            "scientific_choices": OmegaConf.to_container(
                config.scientific_choices, resolve=True
            ),
            "strict_blockers": list(config.strict_blockers),
            "runtime": runtime,
            "precision": "float32",
            "batch_size": batch_size,
            "device": str(device),
            "gpu": runtime["gpu"],
            "image_shape": last_record["train"]["image_shape"],
            "mask_shape": last_record["train"]["mask_shape"],
            "loss": last_record["train"]["loss"],
            "validation_loss": (
                None if training_iou_selection else last_record["validation"]["loss"]
            ),
            "cuda_peak_allocated_bytes": allocated,
            "cuda_peak_reserved_bytes": reserved,
            "checkpoint_output_location": str(
                run_directory / selected_record["checkpoint"]
            ),
            "run_output_location": str(run_directory),
        }
        _write_json(run_directory / "run_manifest.json", run_manifest)
        selected_fg_iou = selected_record.get("training_selection", {}).get(
            "foreground_iou"
        )
        _print_training_complete(
            status=str(run_manifest["status"]),
            best_epoch=int(selected_record["epoch"]),
            best_fg_iou=(None if selected_fg_iou is None else float(selected_fg_iou)),
            checkpoint_path=run_directory / selected_record["checkpoint"],
            elapsed_seconds=time.monotonic() - training_started_at,
            cuda_allocated=allocated,
            cuda_reserved=reserved,
        )
        return run_manifest
    except BaseException as error:
        failure = _failure_record(
            error=error,
            run_directory=run_directory,
            device=device,
            batch_size=batch_size,
        )
        _write_json(run_directory / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=ROOT,
        help="Root for configured manifest paths (defaults to the repository root)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--qualification-max-steps",
        type=int,
        help=(
            "Explicit smoke-only optimizer-step limit for one qualification epoch; "
            "changes result_class to smoke-test result"
        ),
    )
    args = parser.parse_args()
    report = run_training(
        config_path=args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        manifest_root=args.manifest_root,
        device_name=args.device,
        resume=args.resume,
        qualification_max_steps=args.qualification_max_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DomainBalancedEpochSampler",
    "EpochShuffleSampler",
    "RealTrainingError",
    "run_training",
]
