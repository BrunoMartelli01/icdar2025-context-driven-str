"""Two isolated TRAIN-only letterbox runs: word-F1 control and separator head.

Both branches use the same frozen 1130/125 split and RGB/target transform.
TEST and PARSeq are never loaded. Output checkpoints live outside the repo.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.unet_adv import (
    PracticalTotalTextHighResAGUNet, PracticalTotalTextSeparatorHeadAGUNet,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import (
    _append_epoch, _epoch_log, _json, _split_indices, _training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    _config as source_config, _owner_map, _transform, proposal_metrics,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    build_optimizer, load_checkpoint, save_checkpoint, seed_everything,
    separation_weighted_bce_with_logits, train_one_epoch,
)
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_letterbox_word_f1_separator_head_v1.json"
BRANCHES = ("word_f1_control", "word_f1_separator_head")


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (value.get("classification") != "PRACTICAL_APPROXIMATION"
        or value.get("reproduction_mode") != "approximate"
        or value.get("split") != "frozen_TRAIN_1130_validation_125"
        or tuple(value.get("branches", ())) != BRANCHES
        or value.get("epochs") != 100
        or value.get("seed") != 202503
        or value.get("batch_size") != 8
        or value.get("num_workers_per_branch") != 2
        or value.get("input_size") != [512, 512]
        or value.get("native_mask_size") != [256, 256]
        or value.get("foreground_threshold") != 0.5
        or value.get("separator_threshold") != 0.5
        or value.get("separator_auxiliary_loss_weight") != 0.25
        or value.get("postprocessing") != "native256_8_connected_no_area_no_top10_no_padding"):
        raise ValueError("Frozen practical word-F1/separator ablation config changed")
    return value


def balanced_separator_bce(logits: torch.Tensor, target: torch.Tensor,
                           validity: torch.Tensor) -> torch.Tensor:
    """Equal positive/negative contribution; invalid pixels have zero gradient."""
    if logits.shape != target.shape or logits.shape != validity.shape:
        raise ValueError("Separator logits/target/validity shape mismatch")
    if torch.any((target != 0) & (target != 1)) or torch.any((validity != 0) & (validity != 1)):
        raise ValueError("Separator target and validity must be binary")
    if torch.any((target > 0) & (validity == 0)):
        raise ValueError("Separator foreground outside valid region")
    positive = target * validity
    negative = (1 - target) * validity
    positive_count, negative_count = positive.sum(), negative.sum()
    if negative_count.item() == 0:
        raise ValueError("Separator batch has no valid negative pixels")
    positive_loss = (F.softplus(-logits) * positive).sum() / positive_count.clamp_min(1)
    negative_loss = (F.softplus(logits) * negative).sum() / negative_count
    return (positive_loss + negative_loss) / 2 if positive_count.item() else negative_loss


def selection_key(metrics: dict, epoch: int) -> tuple[float, int, float, float, int]:
    """Declared TRAIN-only ordering; earliest epoch wins exact ties."""
    return (float(metrics["proposal_f1_iou_50"]),
            -int(metrics["gt_words_touched_by_fused_components"]),
            float(metrics["word_recall_iou_50"]),
            float(metrics["foreground_iou"]), -epoch)


def _train_separator_epoch(model, batches, optimizer, device: torch.device) -> dict:
    model.train()
    total, count, steps = 0.0, 0, 0
    ordered_ids: list[str] = []
    image_shape = mask_shape = None
    for batch in batches:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["mask"].to(device, non_blocking=device.type == "cuda")
        validity = batch["validity"].to(device, non_blocking=device.type == "cuda")
        separator = batch["separation"].to(device, non_blocking=device.type == "cuda")
        if torch.any((separator > 0) & (target > 0)):
            raise ValueError("Separator and word foreground overlap after transform")
        if image_shape is None:
            image_shape, mask_shape = list(images.shape), list(target.shape)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != (len(images), 2, 256, 256):
            raise ValueError(f"Unexpected two-head output: {tuple(logits.shape)}")
        foreground_loss = separation_weighted_bce_with_logits(
            logits[:, :1], target, validity, separator, separation_weight=3.0)
        separator_loss = balanced_separator_bce(logits[:, 1:2], separator, validity)
        loss = foreground_loss + 0.25 * separator_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite two-head segmentation loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(images)
        count += len(images)
        steps += 1
        ordered_ids.extend(str(value) for value in batch["sample_id"])
    if not count:
        raise ValueError("Empty TRAIN loader")
    return {"loss": total / count, "samples": count, "steps": steps,
            "sample_order_sha256": sha256("".join(f"{v}\n" for v in ordered_ids).encode()).hexdigest(),
            "image_shape": image_shape, "mask_shape": mask_shape,
            "compute_device": str(device)}


def _validation_reference(manifest, root: Path, val_ids: list[int], transform) -> dict:
    reference = {}
    for index in val_ids:
        sample = manifest.samples[index]
        owners, expected = _owner_map(sample, root, transform)
        from PIL import Image
        with Image.open(root / sample.image_path) as rgb:
            canvas = letterbox_canvas_mask(rgb.size).numpy()
        reference[sample.sample_id] = (owners, expected, canvas)
    if len(reference) != 125:
        raise RuntimeError("Frozen validation reference must contain 125 images")
    return reference


def evaluate_word_geometry(model, batches, references: dict, *, device: torch.device,
                           separator_head: bool, return_rows: bool = False) -> dict:
    """Single pass: macro pixel IoU and native one-to-one word-proposal geometry."""
    fields = ("gt_words", "gt_words_lost_in_native_target", "predicted_components",
              "gt_with_predicted_bbox_iou_ge_0_5",
              "gt_words_split_across_predicted_components",
              "predicted_components_touching_multiple_gt_words",
              "gt_words_touched_by_fused_components", "matched_gt_iou_ge_25",
              "matched_gt_iou_ge_50", "matched_gt_iou_ge_75", "raw_proposals")
    totals = {field: 0 for field in fields}
    ious, final_ious, rows = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch["image"].to(device, non_blocking=device.type == "cuda"))
            if logits.shape[1:] != ((2 if separator_head else 1), 256, 256):
                raise ValueError("Validation logits shape drift")
            foreground = (torch.sigmoid(logits[:, :1]) >= 0.5).cpu().numpy()[:, 0]
            separator = ((torch.sigmoid(logits[:, 1:2]) >= 0.5).cpu().numpy()[:, 0]
                         if separator_head else None)
            for offset, sample_id in enumerate(batch["sample_id"]):
                owners_base, expected, canvas = references[sample_id]
                target = batch["mask"][offset, 0].numpy().astype(bool)
                valid = batch["validity"][offset, 0].numpy().astype(bool)
                raw_prediction = foreground[offset] & canvas
                prediction = raw_prediction.copy()
                if separator_head:
                    prediction &= ~separator[offset]
                owners = owners_base.copy()
                owners[~target | ~valid] = 0
                metrics = proposal_metrics(prediction, owners, expected_owner_ids=expected)
                for field in fields:
                    totals[field] += metrics[field]
                intersection = int(np.count_nonzero(raw_prediction & target & valid))
                union = int(np.count_nonzero((raw_prediction | target) & valid))
                ious.append(intersection / union if union else 1.0)
                final_intersection = int(np.count_nonzero(prediction & target & valid))
                final_union = int(np.count_nonzero((prediction | target) & valid))
                final_ious.append(final_intersection / final_union if final_union else 1.0)
                if return_rows:
                    rows.append({"sample_id": sample_id, **metrics,
                                 "foreground_iou": ious[-1],
                                 "final_mask_iou": final_ious[-1]})
    if len(ious) != 125 or totals["gt_words"] != 983:
        raise RuntimeError("TRAIN validation population or GT-word count changed")
    tp = totals["matched_gt_iou_ge_50"]
    proposals, words = totals["raw_proposals"], totals["gt_words"]
    precision = tp / proposals if proposals else 0.0
    recall = tp / words
    aggregate = {**totals, "foreground_iou": float(np.mean(ious)),
                 "final_mask_iou": float(np.mean(final_ious)),
                 "proposal_precision_iou_50": precision, "word_recall_iou_50": recall,
                 "proposal_f1_iou_50": 2 * tp / (proposals + words) if proposals + words else 0.0}
    return {"aggregate": aggregate, "per_image": rows if return_rows else None}


def run(args) -> dict:
    config, source = _config(), source_config()
    if args.branch not in BRANCHES:
        raise ValueError("Unknown ablation branch")
    if sha256_file(REPO / config["source_training_config"]) != config["source_training_config_sha256"]:
        raise ValueError("Source training configuration changed")
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if (sha256_file(manifest_path) != config["train_manifest_sha256"]
        or config["train_manifest_sha256"] != source["train_manifest_sha256"]):
        raise ValueError("Qualified Total-Text v4 TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This ablation permits official Total-Text TRAIN only")
    train_ids, val_ids, val_sha = _split_indices(manifest, config["seed"], 125)
    if len(train_ids) != 1130 or len(val_ids) != 125:
        raise ValueError("Frozen Total-Text TRAIN split changed")
    if args.preflight_only:
        result = {"status": "PREFLIGHT_PASS", "branch": args.branch,
                  "train_images": len(train_ids), "validation_images": len(val_ids),
                  "validation_ids_sha256": val_sha, "device_requested": args.device}
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training was not started")
    output_root = args.output_root.resolve()
    if output_root == REPO or REPO in output_root.parents:
        raise ValueError("Training outputs must be outside repository")
    run_dir = output_root / args.branch
    run_dir.mkdir(parents=True, exist_ok=True)
    separator_head = args.branch == "word_f1_separator_head"
    seed_everything(config["seed"], deterministic_algorithms=True)
    model_type = (PracticalTotalTextSeparatorHeadAGUNet if separator_head
                  else PracticalTotalTextHighResAGUNet)
    model = model_type(skip_floor=0.25, encoder_pretrained=True).to(device)
    architecture = model.architecture_identity
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters()
                               if parameter.requires_grad)
    expected = ((58155146, 32598114) if separator_head else (58155081, 32598049))
    if (total_parameters, trainable_parameters) != expected:
        raise RuntimeError("Ablation architecture parameter count changed")
    training_config = replace(_training_config(source, {"separation_weight": 3.0}),
                              epochs=100, checkpoint_selection="validation_word_f1")
    contract = {"branch": args.branch, "classification": "PRACTICAL_APPROXIMATION",
                "config_sha256": sha256_file(CONFIG_PATH),
                "source_training_config_sha256": sha256_file(REPO / config["source_training_config"]),
                "model_source_sha256": sha256_file(REPO / "models/unet_adv.py"),
                "trainer_source_sha256": sha256_file(Path(__file__)),
                "training_primitives_sha256": sha256_file(REPO / "segmentation/training.py"),
                "transform_source_sha256": sha256_file(REPO / "segmentation/totaltext_image_frequency_augment.py"),
                "manifest_sha256": config["train_manifest_sha256"],
                "validation_ids_sha256": val_sha, "architecture": architecture,
                "parameters": total_parameters, "trainable_parameters": trainable_parameters,
                "output_channels": 2 if separator_head else 1,
                "selection": config["checkpoint_selection"],
                "separator_auxiliary_loss_weight": 0.25 if separator_head else 0.0,
                "foreground_threshold": 0.5, "separator_threshold": 0.5,
                "postprocessing": config["postprocessing"]}
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing run contract differs; refusing overwrite")
    elif any(run_dir.iterdir()):
        raise ValueError("Nonempty run directory without matching contract")
    else:
        _json(contract_path, contract)
    result_path, last_path, best_path = (run_dir / "run_manifest.json",
                                         run_dir / "last.pt", run_dir / "best.pt")
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result.get("status") != "SUCCESS" or result.get("run_contract") != contract
            or sha256_file(best_path) != result["checkpoint"]["sha256"]):
            raise ValueError("Completed run provenance mismatch")
        if last_path.exists():
            records = _epoch_log(run_dir / "epochs.jsonl")
            if len(records) != 100 or sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
                raise ValueError("Completed latest checkpoint is inconsistent")
            last_path.unlink()
        print(f"[{args.branch}] completed cache verified", flush=True)
        return result
    train_transform = _transform(source, "letterbox", train=True)
    val_transform = _transform(source, "letterbox", train=False)
    train_set = Subset(SegmentationDataset(manifest, root, transform=train_transform), train_ids)
    val_set = Subset(SegmentationDataset(manifest, root, transform=val_transform), val_ids)
    loader_options = {"num_workers": config["num_workers_per_branch"],
                      "pin_memory": device.type == "cuda"}
    sampler = EpochShuffleSampler(len(train_set), seed=config["seed"])
    train_loader = DataLoader(train_set, batch_size=8, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, **loader_options)
    references = _validation_reference(manifest, root, val_ids, val_transform)
    optimizer = build_optimizer(model, training_config)
    records = _epoch_log(run_dir / "epochs.jsonl")
    if last_path.exists():
        payload = load_checkpoint(last_path, model=model, optimizer=optimizer,
            expected_architecture=architecture, expected_run_contract=contract,
            expected_training_config=training_config, restore_rng=True)
        start_epoch = int(payload["epoch"]) + 1
        if len(records) == start_epoch - 2:
            _append_epoch(run_dir / "epochs.jsonl", {"epoch": start_epoch - 1,
                "global_step": payload["global_step"], "train": payload["train_metrics"],
                "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last_path), "recovered_from_checkpoint": True})
            records = _epoch_log(run_dir / "epochs.jsonl")
        if len(records) != start_epoch - 1:
            raise ValueError("Epoch log does not match latest checkpoint")
    elif records or best_path.exists():
        raise ValueError("Training artifacts exist without latest checkpoint")
    else:
        start_epoch = 1
    if best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if best_payload.get("run_contract") != contract:
            raise ValueError("Best checkpoint contract mismatch")
        best_epoch = int(best_payload["epoch"])
        best_metrics = best_payload["validation_metrics"]
        best_key = selection_key(best_metrics, best_epoch)
        best_sha = sha256_file(best_path)
        del best_payload
    else:
        best_epoch, best_metrics, best_key, best_sha = 0, {}, None, None
    if last_path.exists():
        last_epoch = int(payload["epoch"])
        last_metrics = payload["validation_metrics"]
        last_key = selection_key(last_metrics, last_epoch)
        if best_key is None or last_key > best_key:
            # A prior process may have stopped after atomically writing last.pt
            # but before writing the improved best.pt. Restore that selection.
            best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
                epoch=last_epoch, global_step=int(payload["global_step"]),
                config=training_config, resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
                architecture=architecture, result_class="approximate reproduction",
                run_contract=contract, train_metrics=payload["train_metrics"],
                validation_metrics=last_metrics)
            best_epoch, best_metrics, best_key = last_epoch, last_metrics, last_key
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    started = time.monotonic()
    for epoch in range(start_epoch, 101):
        sampler.set_epoch(epoch - 1)
        train_transform.set_epoch(epoch - 1)
        trained = (_train_separator_epoch(model, train_loader, optimizer, device)
                   if separator_head else train_one_epoch(model, train_loader, optimizer,
                       device=device, separation_weight=3.0))
        evaluated = evaluate_word_geometry(model, val_loader, references, device=device,
                                           separator_head=separator_head)
        metrics = evaluated["aggregate"]
        step = (records[-1]["global_step"] if records else 0) + trained["steps"]
        last_sha = save_checkpoint(last_path, model=model, optimizer=optimizer,
            epoch=epoch, global_step=step, config=training_config,
            resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
            architecture=architecture, result_class="approximate reproduction",
            run_contract=contract, train_metrics=trained, validation_metrics=metrics)
        candidate = selection_key(metrics, epoch)
        if best_key is None or candidate > best_key:
            best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
                epoch=epoch, global_step=step, config=training_config,
                resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
                architecture=architecture, result_class="approximate reproduction",
                run_contract=contract, train_metrics=trained, validation_metrics=metrics)
            best_epoch, best_metrics, best_key = epoch, metrics, candidate
        row = {"epoch": epoch, "global_step": step, "train": trained,
               "validation": metrics, "last_checkpoint_sha256": last_sha}
        _append_epoch(run_dir / "epochs.jsonl", row)
        records.append(row)
        print(f"[{args.branch}] {epoch}/100 loss={trained['loss']:.5f} "
              f"word_F1={metrics['proposal_f1_iou_50']:.5f} "
              f"recall={metrics['word_recall_iou_50']:.5f} "
              f"fused_words={metrics['gt_words_touched_by_fused_components']} "
              f"best={best_epoch}", flush=True)
    if len(records) != 100 or best_epoch < 1 or best_sha is None:
        raise RuntimeError("Incomplete 100-epoch ablation")
    load_checkpoint(best_path, model=model, expected_architecture=architecture,
                    expected_run_contract=contract, expected_training_config=training_config)
    final_geometry = evaluate_word_geometry(model, val_loader, references, device=device,
                                            separator_head=separator_head, return_rows=True)
    if final_geometry["aggregate"] != best_metrics:
        raise RuntimeError("Selected checkpoint word geometry is not reproducible")
    _json(run_dir / "validation_word_geometry.json", final_geometry)
    report = {"schema_version": 1, "status": "SUCCESS",
              "classification": "PRACTICAL_APPROXIMATION", "result_class": "approximate reproduction",
              "run_contract": contract, "train_images": 1130, "validation_images": 125,
              "checkpoint": {"path": "best.pt", "epoch": best_epoch, "sha256": best_sha},
              "validation_word_geometry": final_geometry["aggregate"],
              "parameter_count": total_parameters, "trainable_parameter_count": trainable_parameters,
              "device": str(device), "elapsed_seconds": time.monotonic() - started,
              "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                           if device.type == "cuda" else None),
              "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                          if device.type == "cuda" else None)}
    _json(result_path, report)
    if sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest checkpoint changed before successful cleanup")
    last_path.unlink()
    print(f"[{args.branch}] RESULT {result_path} SHA256 {sha256_file(result_path)}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
