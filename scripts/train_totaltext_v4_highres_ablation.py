"""Isolated TRAIN-only Total-Text v4 native-256 segmentation ablation.

Three predeclared PRACTICAL_APPROXIMATION variants share a deterministic
TRAIN/validation split. This entry point never reads TEST or runs OCR. It keeps
at most best.pt and last.pt during training, then only best.pt on completion.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Subset

from models.unet_adv import PracticalTotalTextHighResAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    SegmentationTrainingConfig,
    build_optimizer,
    load_checkpoint,
    measure_foreground_iou,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_v4_highres_ablation_v1.json"
ARCHITECTURE = PracticalTotalTextHighResAGUNet.PRACTICAL_IDENTITY


def _json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("reproduction_mode") != "approximate"
        or config.get("variant_class") != "PRACTICAL_APPROXIMATION"
        or config.get("image_size") != [512, 512]
        or config.get("native_mask_size") != [256, 256]
        or config.get("architecture", {}).get("identity") != ARCHITECTURE
        or config.get("architecture", {}).get("encoder_weights") != "IMAGENET1K_V2"
        or config.get("architecture", {}).get("freeze_encoder") is not True
        or config.get("training", {}).get("checkpoint_retention")
        != "best_and_latest_during_training_best_only_after_success"
        or set(config.get("variants", {}))
        != {"highres_baseline", "highres_separator", "highres_separator_residual"}
    ):
        raise ValueError("High-resolution ablation configuration changed incompatibly")
    return config


def _split_indices(manifest: object, seed: int, validation_count: int) -> tuple[list[int], list[int], str]:
    samples = manifest.samples
    if len(samples) != 1255 or not 0 < validation_count < len(samples):
        raise ValueError("Expected qualified Total-Text TRAIN-1255 and a nonempty held-out subset")
    ordered = sorted(
        range(len(samples)),
        key=lambda index: (
            sha256(f"{seed}:{samples[index].sample_id}".encode("utf-8")).digest(),
            samples[index].sample_id,
        ),
    )
    validation_set = set(ordered[:validation_count])
    train = [index for index in range(len(samples)) if index not in validation_set]
    validation = [index for index in range(len(samples)) if index in validation_set]
    ids = [samples[index].sample_id for index in validation]
    fingerprint = sha256("".join(f"{value}\n" for value in ids).encode("utf-8")).hexdigest()
    return train, validation, fingerprint


def _transform(config: dict, *, augment: bool, seed: int) -> PairedSegmentationTransform:
    training = config["training"]
    return PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=training["horizontal_flip_probability"] if augment else 0.0,
            rotation_probability=training["rotation_probability"] if augment else 0.0,
            rotation_degrees=tuple(training["rotation_degrees"]),
            image_interpolation="bilinear",
            mask_interpolation="bilinear",
            image_fill=(0, 0, 0),
            mask_fill=0,
            normalization="imagenet",
            resize_geometry="stretch",
            operation_order="flip_rotate_resize",
            image_size=(512, 512),
            mask_size=(256, 256),
            target_binarization_threshold=0.5,
            validity_interpolation="nearest",
            classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION",
        ),
        base_seed=seed,
    )


def _training_config(config: dict, variant: dict) -> SegmentationTrainingConfig:
    item = config["training"]
    return SegmentationTrainingConfig(
        epochs=item["epochs"],
        seed=item["seed"],
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        learning_rate_schedule="none",
        checkpoint_selection="validation_foreground_iou",
        deterministic_algorithms=item["deterministic_algorithms"],
        optimizer=item["optimizer"],
        learning_rate=item["learning_rate"],
        weight_decay=item["weight_decay"],
        batch_size=item["batch_size"],
        loss="masked_BCEWithLogitsLoss",
        separation_weight=variant["separation_weight"],
    )


def _epoch_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_epoch(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _checkpoint_metadata(path: Path, contract: dict) -> dict | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != ARCHITECTURE or payload.get("run_contract") != contract:
        raise ValueError(f"Checkpoint provenance mismatch: {path}")
    return {"epoch": payload["epoch"], "validation": payload["validation_metrics"],
            "sha256": sha256_file(path)}


def _device(value: str) -> torch.device:
    if value not in {"cpu", "cuda:0"}:
        raise ValueError("Device must be cpu or cuda:0")
    if value == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no GPU experiment was started")
    return torch.device(value)


def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    union = area_left + area_right - intersection
    return 0.0 if union <= 0 else intersection / union


def word_topology_metrics(
    prediction: np.ndarray, owners: np.ndarray, *, expected_owner_ids: tuple[int, ...] | None = None
) -> dict:
    """GT used only for held-out TRAIN validation, never to generate groups."""

    if prediction.shape != owners.shape or prediction.shape != (256, 256):
        raise ValueError("Word topology expects paired native-256 arrays")
    if prediction.dtype != np.bool_ or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("Word topology requires bool prediction and integer owner IDs")
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        prediction.astype(np.uint8), connectivity=8
    )
    component_boxes = [
        (int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]),
         int(stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]),
         int(stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT]))
        for index in range(1, component_count)
    ]
    present_owner_ids = {int(value) for value in np.unique(owners) if value > 0}
    owner_ids = (sorted(present_owner_ids) if expected_owner_ids is None
                 else sorted(set(expected_owner_ids)))
    if not present_owner_ids.issubset(owner_ids):
        raise ValueError("Native owner map contains an unrecorded word ID")
    word_components: dict[int, set[int]] = {}
    good = split = 0
    for owner_id in owner_ids:
        owner_pixels = owners == owner_id
        ys, xs = np.where(owner_pixels)
        if ys.size == 0:
            word_components[owner_id] = set()
            continue
        owner_box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        intersecting = {int(value) for value in np.unique(labels[owner_pixels]) if value > 0}
        word_components[owner_id] = intersecting
        split += len(intersecting) >= 2
        good += max((_box_iou(owner_box, box) for box in component_boxes), default=0.0) >= 0.5
    component_owners: dict[int, set[int]] = {index: set() for index in range(1, component_count)}
    for owner_id, component_ids in word_components.items():
        for component_id in component_ids:
            component_owners[component_id].add(owner_id)
    fused = [associated for associated in component_owners.values() if len(associated) >= 2]
    return {
        "gt_words": len(owner_ids),
        "gt_words_lost_in_native_target": len(set(owner_ids) - present_owner_ids),
        "predicted_components": component_count - 1,
        "gt_with_predicted_bbox_iou_ge_0_5": good,
        "gt_words_split_across_predicted_components": split,
        "predicted_components_touching_multiple_gt_words": len(fused),
        "gt_words_touched_by_fused_components": len(set().union(*fused)) if fused else 0,
    }


def _validation_topology(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    manifest: object,
    target_root: Path,
    *,
    device: torch.device,
) -> dict:
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    totals = {name: 0 for name in (
        "gt_words", "gt_words_lost_in_native_target", "predicted_components", "gt_with_predicted_bbox_iou_ge_0_5",
        "gt_words_split_across_predicted_components",
        "predicted_components_touching_multiple_gt_words",
        "gt_words_touched_by_fused_components",
    )}
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in validation_loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[:, 0]
            for index, sample_id in enumerate(batch["sample_id"]):
                sample = by_id[sample_id]
                if sample.word_owner_annotation_path is None:
                    raise ValueError(f"Missing word-owner map for validation sample: {sample_id}")
                owner_path = target_root / sample.word_owner_annotation_path
                with Image.open(owner_path) as opened:
                    expected_owner_ids = tuple(int(value) for value in np.unique(np.asarray(opened)) if value > 0)
                    owner_image = TF.resize(
                        opened.convert("I"), [256, 256], interpolation=InterpolationMode.NEAREST
                    )
                owners = np.asarray(owner_image, dtype=np.int32).copy()
                owners[(batch["mask"][index, 0].numpy() == 0)
                       | (batch["validity"][index, 0].numpy() == 0)] = 0
                values = word_topology_metrics(
                    predictions[index], owners, expected_owner_ids=expected_owner_ids
                )
                for name in totals:
                    totals[name] += values[name]
                rows.append({"sample_id": sample_id, **values})
            if len(rows) % 25 < len(batch["sample_id"]):
                print(f"[validation topology] {len(rows)}/{len(validation_loader.dataset)}", flush=True)
    if len(rows) != len(validation_loader.dataset):
        raise RuntimeError("Validation topology did not cover the complete held-out population")
    return {"schema_version": 1, "split": "TRAIN_heldout_125", "images": len(rows),
            "aggregate": totals, "per_image": rows}


def run(args: argparse.Namespace) -> dict:
    config = _load_config()
    variant = config["variants"][args.variant]
    device = _device(args.device)
    target_root = args.target_root.resolve()
    manifest_path = target_root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text v4 TRAIN manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Expected official Total-Text TRAIN v4 manifest")
    seed = int(config["training"]["seed"])
    train_indices, validation_indices, split_sha = _split_indices(
        manifest, seed, int(config["validation"]["count"])
    )
    seed_everything(seed, deterministic_algorithms=True)
    model = PracticalTotalTextHighResAGUNet(
        skip_floor=variant["skip_floor"], encoder_pretrained=True
    ).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if (total_params, trainable_params) != (58155081, 32598049):
        raise RuntimeError("High-resolution ablation changed parameter counts")
    config_sha = sha256_file(CONFIG_PATH)
    contract = {
        "variant": args.variant,
        "classification": "PRACTICAL_APPROXIMATION",
        "config_sha256": config_sha,
        "manifest_sha256": config["train_manifest_sha256"],
        "validation_ids_sha256": split_sha,
        "input_shape": [3, 512, 512],
        "logits_shape": [1, 256, 256],
        "separation_weight": variant["separation_weight"],
        "skip_floor": variant["skip_floor"],
        "checkpoint_selection": "best_validation_foreground_iou_tie_earliest_epoch",
    }
    if args.check:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            batch_size = int(config["training"]["batch_size"])
            sample_batch = {
                "image": torch.zeros(batch_size, 3, 512, 512),
                "mask": torch.zeros(batch_size, 1, 256, 256),
                "validity": torch.ones(batch_size, 1, 256, 256),
                "separation": torch.zeros(batch_size, 1, 256, 256),
            }
            metrics = train_one_epoch(
                model, [sample_batch], build_optimizer(model, _training_config(config, variant)),
                device=device, separation_weight=variant["separation_weight"],
            )
            if metrics["image_shape"] != [batch_size, 3, 512, 512] or metrics["mask_shape"] != [batch_size, 1, 256, 256]:
                raise RuntimeError("High-resolution GPU batch geometry mismatch")
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
        else:
            with torch.inference_mode():
                output = model.eval()(torch.zeros(1, 3, 512, 512, device=device))
            if tuple(output.shape) != (1, 1, 256, 256):
                raise RuntimeError(f"Native logits geometry mismatch: {tuple(output.shape)}")
            allocated = reserved = None
        return {"status": "CHECK_PASS", "variant": args.variant,
                "train_images": len(train_indices), "validation_images": len(validation_indices),
                "validation_ids_sha256": split_sha, "parameter_count": total_params,
                "trainable_parameter_count": trainable_params,
                "logits_shape": [1, 1, 256, 256], "device": str(device),
                "gpu_peak_allocated_bytes": allocated, "gpu_peak_reserved_bytes": reserved}

    output_root = args.output_root.resolve()
    if output_root == REPO or REPO in output_root.parents:
        raise ValueError("Training outputs must be outside the repository")
    run_dir = output_root / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"Existing run contract differs: {run_dir}")
    else:
        if any(run_dir.iterdir()):
            raise ValueError(f"Nonempty run directory without contract: {run_dir}")
        _json(contract_path, contract)
    result_path = run_dir / "run_manifest.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "SUCCESS" or result.get("run_contract") != contract:
            raise ValueError(f"Existing run manifest is incompatible: {result_path}")
        if sha256_file(run_dir / "best.pt") != result["checkpoint"]["sha256"]:
            raise ValueError("Completed best checkpoint SHA256 mismatch")
        last_path = run_dir / "last.pt"
        if last_path.exists():
            records = _epoch_log(run_dir / "epochs.jsonl")
            if len(records) != config["training"]["epochs"] or sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
                raise ValueError("Completed run has an unverified latest checkpoint")
            last_path.unlink()
        return result

    train_transform = _transform(config, augment=True, seed=seed)
    validation_transform = _transform(config, augment=False, seed=seed)
    train_dataset = Subset(SegmentationDataset(manifest, target_root, transform=train_transform), train_indices)
    validation_dataset = Subset(
        SegmentationDataset(manifest, target_root, transform=validation_transform), validation_indices
    )
    sampler = EpochShuffleSampler(len(train_dataset), seed=seed)
    loader_kwargs = {"num_workers": int(config["training"]["num_workers"]),
                     "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                              sampler=sampler, drop_last=False, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, batch_size=config["training"]["batch_size"],
                                   shuffle=False, drop_last=False, **loader_kwargs)
    training_config = _training_config(config, variant)
    optimizer = build_optimizer(model, training_config)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    log_path = run_dir / "epochs.jsonl"
    records = _epoch_log(log_path)
    start_epoch = 1
    if last_path.exists():
        payload = load_checkpoint(
            last_path, model=model, optimizer=optimizer, expected_architecture=ARCHITECTURE,
            expected_run_contract=contract, expected_training_config=training_config,
            restore_rng=True,
        )
        start_epoch = int(payload["epoch"]) + 1
        if len(records) == start_epoch - 2:
            _append_epoch(log_path, {
                "epoch": int(payload["epoch"]),
                "global_step": int(payload["global_step"]),
                "train": payload["train_metrics"],
                "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last_path),
                "recovered_from_last_checkpoint": True,
            })
            records = _epoch_log(log_path)
        if len(records) != start_epoch - 1 or records[-1]["epoch"] != start_epoch - 1:
            raise ValueError("Resume epoch log is incompatible with last checkpoint")
    elif records or best_path.exists():
        raise ValueError("Run has historical records but no last checkpoint")
    best_meta = _checkpoint_metadata(best_path, contract)
    best_iou = -1.0 if best_meta is None else float(best_meta["validation"]["foreground_iou"])
    if start_epoch > 1:
        last_iou = float(records[-1]["validation"]["foreground_iou"])
        if last_iou > best_iou:
            best_iou = last_iou
            best_sha = save_checkpoint(
                best_path, model=model, optimizer=optimizer, epoch=start_epoch - 1,
                global_step=int(payload["global_step"]), config=training_config,
                resolved_config=CONFIG_PATH.read_text(encoding="utf-8"), architecture=ARCHITECTURE,
                result_class="approximate reproduction", run_contract=contract,
                train_metrics=records[-1]["train"], validation_metrics=records[-1]["validation"],
            )
            best_meta = {"epoch": start_epoch - 1, "sha256": best_sha,
                         "validation": records[-1]["validation"]}
    started = time.monotonic()
    for epoch in range(start_epoch, training_config.epochs + 1):
        sampler.set_epoch(epoch - 1)
        train_transform.set_epoch(epoch - 1)
        epoch_start = time.monotonic()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device=device,
            separation_weight=variant["separation_weight"],
        )
        validation_metrics = measure_foreground_iou(
            model, validation_loader, device=device, probability_threshold=0.5,
            aggregation="per_image_macro",
        )
        # A latest checkpoint is required for interruption-safe resume.
        global_step = (records[-1].get("global_step", 0) if records else 0) + int(train_metrics["steps"])
        last_sha = save_checkpoint(
            last_path, model=model, optimizer=optimizer, epoch=epoch, global_step=global_step,
            config=training_config, resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
            architecture=ARCHITECTURE, result_class="approximate reproduction",
            run_contract=contract, train_metrics=train_metrics, validation_metrics=validation_metrics,
        )
        if float(validation_metrics["foreground_iou"]) > best_iou:
            best_iou = float(validation_metrics["foreground_iou"])
            best_sha = save_checkpoint(
                best_path, model=model, optimizer=optimizer, epoch=epoch, global_step=global_step,
                config=training_config, resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
                architecture=ARCHITECTURE, result_class="approximate reproduction",
                run_contract=contract, train_metrics=train_metrics, validation_metrics=validation_metrics,
            )
            best_meta = {"epoch": epoch, "sha256": best_sha, "validation": validation_metrics}
        record = {"epoch": epoch, "global_step": global_step,
                  "train": train_metrics, "validation": validation_metrics,
                  "last_checkpoint_sha256": last_sha,
                  "best_epoch_so_far": best_meta["epoch"], "best_iou_so_far": best_iou}
        _append_epoch(log_path, record)
        records.append(record)
        print(
            f"[{args.variant}] epoch {epoch}/{training_config.epochs} "
            f"train_loss={train_metrics['loss']:.5f} val_IoU={best_iou:.5f} "
            f"current_val_IoU={validation_metrics['foreground_iou']:.5f} "
            f"seconds={time.monotonic() - epoch_start:.1f}", flush=True,
        )
    if best_meta is None or len(records) != training_config.epochs:
        raise RuntimeError("Training did not produce a complete selected checkpoint")
    load_checkpoint(
        best_path, model=model, expected_architecture=ARCHITECTURE,
        expected_run_contract=contract, expected_training_config=training_config,
    )
    topology = _validation_topology(
        model, validation_loader, manifest, target_root, device=device
    )
    _json(run_dir / "validation_word_topology.json", topology)
    result = {
        "schema_version": 1, "status": "SUCCESS", "training_complete": True,
        "classification": "PRACTICAL_APPROXIMATION", "result_class": "approximate reproduction",
        "run_contract": contract, "train_images": len(train_indices),
        "validation_images": len(validation_indices),
        "selection": "best_validation_foreground_iou_tie_earliest_epoch",
        "checkpoint": {"path": "best.pt", "epoch": best_meta["epoch"],
                       "sha256": best_meta["sha256"],
                       "validation_foreground_iou": best_iou},
        "validation_word_topology": topology["aggregate"],
        "parameter_count": total_params, "trainable_parameter_count": trainable_params,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "epochs": records,
    }
    _json(result_path, result)
    # The selected checkpoint plus its manifest are sufficient for evaluation.
    # Delete only this run's exact, already-verified latest checkpoint.
    if sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest checkpoint changed; refusing automatic retention cleanup")
    last_path.unlink()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(_load_config()["variants"]), required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--check", action="store_true", help="Read-only shape/provenance check")
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error("--output-root is required unless --check is set")
    result = run(args)
    print(json.dumps(result if args.check else {
        "status": result["status"], "variant": args.variant,
        "checkpoint": result["checkpoint"],
        "train_images": result["train_images"], "validation_images": result["validation_images"],
        "parameter_count": result["parameter_count"],
        "cuda_peak_allocated_bytes": result["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": result["cuda_peak_reserved_bytes"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
