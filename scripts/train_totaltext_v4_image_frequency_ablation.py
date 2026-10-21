"""TRAIN-only, restartable practical image/frequency ablations for Total-Text v4.

Four predeclared variants use the frozen 1130/125 TRAIN split. No TEST, OCR,
threshold search, area filter, or top-10 truncation enters checkpoint selection.
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
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.totaltext_wavelet import PracticalTotalTextWaveletHighResAGUNet
from models.unet_adv import PracticalTotalTextHighResAGUNet
from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import (
    _append_epoch, _box_iou, _epoch_log, _json, _split_indices, _training_config,
    word_topology_metrics,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import (
    PracticalTotalTextImageTransform, letterbox_canvas_mask,
)
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    build_optimizer, load_checkpoint, measure_foreground_iou, save_checkpoint,
    seed_everything, train_one_epoch,
)
from segmentation.transforms import SegmentationTransformConfig


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_v4_image_frequency_ablation_v1.json"
VARIANTS = ("photometric", "letterbox", "fourier_amplitude", "wavelet_haar")
TRANSFORM_SOURCE = REPO / "segmentation/totaltext_image_frequency_augment.py"
WAVELET_SOURCE = REPO / "models/totaltext_wavelet.py"
BASE_MODEL_SOURCE = REPO / "models/unet_adv.py"


def _config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (config.get("reproduction_mode") != "approximate"
        or config.get("variant_class") != "PRACTICAL_APPROXIMATION"
        or config.get("input_size") != [512, 512]
        or config.get("native_mask_size") != [256, 256]
        or config.get("probability_threshold") != 0.5
        or config.get("training", {}).get("seed") != 202503
        or config.get("training", {}).get("separation_weight") != 3.0
        or config.get("training", {}).get("skip_floor") != 0.25
        or set(config.get("variants", {})) != set(VARIANTS)):
        raise ValueError("Practical image/frequency ablation config is incompatible")
    return config


def _effective_epoch_budget(configured_epochs: int, stop_after_epoch: int | None) -> int:
    """Limit future execution without changing the frozen checkpoint contract."""
    if configured_epochs < 1:
        raise ValueError("Configured epoch budget must be positive")
    if stop_after_epoch is None:
        return configured_epochs
    if not 1 <= stop_after_epoch <= configured_epochs:
        raise ValueError("--stop-after-epoch must be between 1 and configured epochs")
    return stop_after_epoch


def _completed_run_matches_budget(report: dict, requested_epochs: int,
                                  configured_epochs: int) -> bool:
    """A historical full run may be reused; a short run cannot masquerade as full."""
    completed_epochs = len(report.get("epochs", []))
    recorded = report.get("execution_budget")
    if recorded is None:
        return completed_epochs == configured_epochs
    if (
        recorded.get("configured_epochs") != configured_epochs
        or recorded.get("completed_epochs") != completed_epochs
        or recorded.get("requested_epochs") != completed_epochs
    ):
        return False
    return completed_epochs == configured_epochs or completed_epochs == requested_epochs


def _transform(config: dict, name: str, *, train: bool, donor_paths=(), donor_indices=()):
    training = config["training"]
    intervention = config["variants"][name]["image_intervention"]
    return PracticalTotalTextImageTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=training["horizontal_flip_probability"] if train else 0.0,
            rotation_probability=training["rotation_probability"] if train else 0.0,
            rotation_degrees=tuple(training["rotation_degrees"]),
            image_interpolation="bilinear", mask_interpolation="bilinear",
            image_fill=(0, 0, 0), mask_fill=0, normalization="imagenet",
            resize_geometry="stretch", operation_order="flip_rotate_resize",
            image_size=(512, 512), mask_size=(256, 256),
            target_binarization_threshold=0.5, validity_interpolation="nearest",
            classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION",
        ),
        base_seed=int(training["seed"]), mode=intervention, augment_images=train,
        parameters=config["variants"][name]["parameters"],
        donor_paths=tuple(donor_paths), donor_indices=tuple(donor_indices),
    )


def _model(config: dict, name: str):
    model_kind = config["variants"][name]["model"]
    model_class = (PracticalTotalTextWaveletHighResAGUNet if model_kind == "haar_early_skip_adapter"
                   else PracticalTotalTextHighResAGUNet)
    if model_kind not in {"haar_early_skip_adapter", "frozen_resnet_ag_unet"}:
        raise ValueError("Unexpected model class in predeclared ablation")
    return model_class(skip_floor=float(config["training"]["skip_floor"]), encoder_pretrained=True)


def _owner_map(sample, root: Path, transform: PracticalTotalTextImageTransform) -> tuple[np.ndarray, tuple[int, ...]]:
    if sample.word_owner_annotation_path is None:
        raise ValueError(f"Missing v4 word-owner map: {sample.sample_id}")
    owner_path = root / sample.word_owner_annotation_path
    with Image.open(owner_path) as opened:
        source = opened.convert("I")
        expected_ids = tuple(int(value) for value in np.unique(np.asarray(source)) if value > 0)
        blank = Image.new("RGB", source.size)
        _, transformed, _ = transform._augment(blank, source, None, 0)
    # No augmentation is enabled on validation; sample index cannot affect geometry.
    resized = TF.resize(transformed, [256, 256], interpolation=InterpolationMode.NEAREST)
    return np.asarray(resized, dtype=np.int32).copy(), expected_ids


def proposal_metrics(prediction: np.ndarray, owners: np.ndarray, *, expected_owner_ids=None) -> dict:
    """Native geometry with one-to-one GT/proposal matching and no filtering."""
    base = word_topology_metrics(prediction, owners, expected_owner_ids=expected_owner_ids)
    count, _, stats, _ = cv2.connectedComponentsWithStats(prediction.astype(np.uint8), connectivity=8)
    boxes = [(int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
              int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]),
              int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT])) for s in stats[1:]]
    gt_boxes = []
    owner_ids = (sorted(set(expected_owner_ids)) if expected_owner_ids is not None
                 else sorted(int(x) for x in np.unique(owners) if x > 0))
    for owner_id in owner_ids:
        ys, xs = np.where(owners == owner_id)
        gt_boxes.append(None if ys.size == 0 else
            (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))
    for threshold, key in ((0.25, "25"), (0.5, "50"), (0.75, "75")):
        edges = [(pi, gi) for pi, box in enumerate(boxes) for gi, gt in enumerate(gt_boxes)
                 if gt is not None and _box_iou(box, gt) >= threshold]
        base[f"matched_gt_iou_ge_{key}"] = len(maximum_matching(len(boxes), len(gt_boxes), edges))
    base["raw_proposals"] = count - 1
    return base


def _validation_geometry(model, loader, manifest, root, transform, *, device):
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    fields = (
        "gt_words", "gt_words_lost_in_native_target", "predicted_components",
        "gt_with_predicted_bbox_iou_ge_0_5", "gt_words_split_across_predicted_components",
        "predicted_components_touching_multiple_gt_words", "gt_words_touched_by_fused_components",
        "matched_gt_iou_ge_25", "matched_gt_iou_ge_50", "matched_gt_iou_ge_75", "raw_proposals",
    )
    totals = {key: 0 for key in fields}
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            prediction = (torch.sigmoid(model(batch["image"].to(device))) >= 0.5).cpu().numpy()[:, 0]
            for offset, sample_id in enumerate(batch["sample_id"]):
                sample = by_id[sample_id]
                if transform.mode == "letterbox":
                    with Image.open(root / sample.image_path) as source_rgb:
                        canvas = letterbox_canvas_mask(source_rgb.size).numpy()
                    prediction[offset] &= canvas
                owners, expected_owner_ids = _owner_map(sample, root, transform)
                owners[(batch["mask"][offset, 0].numpy() == 0)
                       | (batch["validity"][offset, 0].numpy() == 0)] = 0
                metrics = proposal_metrics(prediction[offset], owners,
                                           expected_owner_ids=expected_owner_ids)
                for field in fields:
                    totals[field] += metrics[field]
                rows.append({"sample_id": sample_id, **metrics})
            if len(rows) % 25 < len(batch["sample_id"]):
                print(f"[word geometry] {len(rows)}/{len(loader.dataset)}", flush=True)
    if len(rows) != len(loader.dataset):
        raise RuntimeError("Incomplete held-out TRAIN geometry evaluation")
    tp = totals["matched_gt_iou_ge_50"]
    proposals, words = totals["raw_proposals"], totals["gt_words"]
    precision = tp / proposals if proposals else 0.0
    recall = tp / words if words else 0.0
    return {"schema_version": 1, "split": "TRAIN_heldout_125", "images": len(rows),
            "postprocessing": "native_256_8_connected_no_area_no_top10_no_padding",
            "aggregate": {**totals, "proposal_precision_iou_50": precision,
                          "word_recall_iou_50": recall,
                          "proposal_f1_iou_50": (2*precision*recall/(precision+recall)
                                                   if precision+recall else 0.0)},
            "per_image": rows}


def run(args):
    config = _config()
    if args.variant not in VARIANTS:
        raise ValueError("Unknown variant")
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text TRAIN manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This ablation permits only official Total-Text TRAIN")
    seed = config["training"]["seed"]
    train_ids, val_ids, val_sha = _split_indices(manifest, seed, 125)
    if len(train_ids) != 1130 or len(val_ids) != 125:
        raise RuntimeError("TRAIN/validation split drift")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training was not started")
    device = torch.device(args.device)
    seed_everything(seed, deterministic_algorithms=True)
    model = _model(config, args.variant).to(device)
    architecture = model.architecture_identity
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = (58155273, 32598241) if args.variant == "wavelet_haar" else (58155081, 32598049)
    if (parameter_count, trainable_count) != expected:
        raise RuntimeError("High-resolution architecture parameter count drift")
    variant_config = {"separation_weight": config["training"]["separation_weight"]}
    training_config = _training_config(config, variant_config)
    target_epochs = _effective_epoch_budget(training_config.epochs, args.stop_after_epoch)
    donor_paths = tuple(root / manifest.samples[index].image_path for index in train_ids)
    train_transform = _transform(config, args.variant, train=True,
        donor_paths=donor_paths if args.variant == "fourier_amplitude" else (),
        donor_indices=tuple(train_ids) if args.variant == "fourier_amplitude" else ())
    val_transform = _transform(config, args.variant, train=False)
    train_data = Subset(SegmentationDataset(manifest, root, transform=train_transform), train_ids)
    val_data = Subset(SegmentationDataset(manifest, root, transform=val_transform), val_ids)
    loader_args = {"num_workers": config["training"]["num_workers"],
                   "pin_memory": device.type == "cuda"}
    sampler = EpochShuffleSampler(len(train_data), seed=seed)
    train_loader = DataLoader(train_data, batch_size=training_config.batch_size,
        sampler=sampler, **loader_args)
    val_loader = DataLoader(val_data, batch_size=training_config.batch_size,
        shuffle=False, **loader_args)
    contract = {"variant": args.variant, "classification": "PRACTICAL_APPROXIMATION",
        "config_sha256": sha256_file(CONFIG_PATH),
        "transform_source_sha256": sha256_file(TRANSFORM_SOURCE),
        "wavelet_source_sha256": sha256_file(WAVELET_SOURCE),
        "base_model_source_sha256": sha256_file(BASE_MODEL_SOURCE),
        "manifest_sha256": config["train_manifest_sha256"],
        "validation_ids_sha256": val_sha,
        "train_donor_indices_sha256": sha256(
            "".join(f"{i}\n" for i in train_ids).encode()).hexdigest(),
        "input_shape": [3, 512, 512], "logits_shape": [1, 256, 256],
        "architecture": architecture, "separation_weight": 3.0, "skip_floor": 0.25,
        "intervention": config["variants"][args.variant],
        "checkpoint_selection": "best_validation_foreground_iou_tie_earliest_epoch"}
    if args.check:
        sample = train_data[0]
        if sample["image"].shape != (3, 512, 512) or sample["mask"].shape != (1, 256, 256):
            raise RuntimeError("Augmentation output shape drift")
        if args.shape_check:
            with torch.inference_mode():
                logits = model.eval()(sample["image"].unsqueeze(0).to(device))
            if logits.shape != (1, 1, 256, 256):
                raise RuntimeError("Model output shape drift")
        return {"status": "CHECK_PASS", "variant": args.variant, "contract": contract,
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_count}
    output_root = args.output_root.resolve()
    if output_root == REPO or REPO in output_root.parents:
        raise ValueError("Model outputs must be outside the repository")
    run_dir = output_root / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing ablation run contract differs")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("Nonempty run directory without contract")
        _json(contract_path, contract)
    result_path = run_dir / "run_manifest.json"
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    log_path = run_dir / "epochs.jsonl"
    if result_path.exists():
        report = json.loads(result_path.read_text())
        if report.get("status") != "SUCCESS" or report.get("run_contract") != contract:
            raise ValueError("Completed run manifest is incompatible")
        if not _completed_run_matches_budget(report, target_epochs, training_config.epochs):
            raise ValueError("Completed run epoch budget differs from the requested budget")
        if sha256_file(best_path) != report["checkpoint"]["sha256"]:
            raise ValueError("Completed best checkpoint SHA256 mismatch")
        return report
    optimizer = build_optimizer(model, training_config)
    records = _epoch_log(log_path)
    start_epoch = 1
    if last_path.exists():
        payload = load_checkpoint(last_path, model=model, optimizer=optimizer,
            expected_architecture=architecture, expected_run_contract=contract,
            expected_training_config=training_config, restore_rng=True)
        start_epoch = int(payload["epoch"]) + 1
        if len(records) == start_epoch - 2:
            _append_epoch(log_path, {"epoch": payload["epoch"],
                "global_step": payload["global_step"], "train": payload["train_metrics"],
                "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last_path),
                "recovered_from_last_checkpoint": True})
            records = _epoch_log(log_path)
        if len(records) != start_epoch - 1 or records[-1]["epoch"] != start_epoch - 1:
            raise ValueError("Resume epoch log is incompatible with last checkpoint")
    elif records or best_path.exists():
        raise ValueError("Historical training records without latest checkpoint")
    best_meta = None
    best_iou = -1.0
    if best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if best_payload.get("run_contract") != contract:
            raise ValueError("Best checkpoint provenance mismatch")
        best_meta = {"epoch": best_payload["epoch"], "sha256": sha256_file(best_path),
                     "validation": best_payload["validation_metrics"]}
        best_iou = float(best_meta["validation"]["foreground_iou"])
    if start_epoch > target_epochs + 1:
        raise ValueError("Existing checkpoint exceeds the requested stop epoch")
    started = time.monotonic()
    for epoch in range(start_epoch, target_epochs + 1):
        sampler.set_epoch(epoch - 1)
        train_transform.set_epoch(epoch - 1)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device,
            separation_weight=3.0)
        val_metrics = measure_foreground_iou(model, val_loader, device=device,
            probability_threshold=0.5, aggregation="per_image_macro")
        step = (records[-1].get("global_step", 0) if records else 0) + train_metrics["steps"]
        last_sha = save_checkpoint(last_path, model=model, optimizer=optimizer,
            epoch=epoch, global_step=step, config=training_config,
            resolved_config=CONFIG_PATH.read_text(), architecture=architecture,
            result_class="approximate reproduction", run_contract=contract,
            train_metrics=train_metrics, validation_metrics=val_metrics)
        if float(val_metrics["foreground_iou"]) > best_iou:
            best_iou = float(val_metrics["foreground_iou"])
            best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
                epoch=epoch, global_step=step, config=training_config,
                resolved_config=CONFIG_PATH.read_text(), architecture=architecture,
                result_class="approximate reproduction", run_contract=contract,
                train_metrics=train_metrics, validation_metrics=val_metrics)
            best_meta = {"epoch": epoch, "sha256": best_sha, "validation": val_metrics}
        row = {"epoch": epoch, "global_step": step, "train": train_metrics,
               "validation": val_metrics, "last_checkpoint_sha256": last_sha}
        _append_epoch(log_path, row)
        records.append(row)
        print(f"[{args.variant}] epoch {epoch}/{target_epochs} "
              f"loss={train_metrics['loss']:.5f} "
              f"val_fgIoU={val_metrics['foreground_iou']:.5f} "
              f"best_fgIoU={best_iou:.5f}", flush=True)
    if best_meta is None or len(records) != target_epochs:
        raise RuntimeError("Training did not complete")
    load_checkpoint(best_path, model=model, expected_architecture=architecture,
        expected_run_contract=contract, expected_training_config=training_config)
    geometry = _validation_geometry(model, val_loader, manifest, root, val_transform, device=device)
    _json(run_dir / "validation_word_geometry.json", geometry)
    report = {"schema_version": 1, "status": "SUCCESS", "training_complete": True,
        "classification": "PRACTICAL_APPROXIMATION", "result_class": "approximate reproduction",
        "execution_budget": {"configured_epochs": training_config.epochs,
                             "requested_epochs": target_epochs,
                             "completed_epochs": len(records),
                             "configured_budget_completed": target_epochs == training_config.epochs,
                             "classification": "ASSUMPTION"},
        "run_contract": contract, "train_images": 1130, "validation_images": 125,
        "checkpoint": {"path": "best.pt", "epoch": best_meta["epoch"],
                       "sha256": best_meta["sha256"],
                       "validation_foreground_iou": best_iou},
        "validation_word_geometry": geometry["aggregate"],
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "epochs": records}
    _json(result_path, report)
    if sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest checkpoint changed; refusing cleanup")
    last_path.unlink()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Explicit shorter budget; preserves existing checkpoint contract")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--shape-check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error("--output-root is required unless --check is set")
    report = run(args)
    print(json.dumps(report if args.check else {
        "status": report["status"], "variant": args.variant,
        "checkpoint": report["checkpoint"],
        "validation_word_geometry": report["validation_word_geometry"],
        "gpu_peak_allocated_bytes": report["gpu_peak_allocated_bytes"],
        "gpu_peak_reserved_bytes": report["gpu_peak_reserved_bytes"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
