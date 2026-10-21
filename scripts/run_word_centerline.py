"""Resumable APPROXIMATE frozen-U-Net centerline/radius experiment."""

import argparse
from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from localize.word_centerline import METHODS, centerline_instances
from localize.word_multitask import parseq_word_proposals
from models.word_centerline import CenterlineRadiusAGUNet
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_word_multitask_crops import score_predictions, verify_geometry_exports, write_json
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_word_research_v2 import load_source_state
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_letterbox_word_f1_separator_head import _validation_reference, selection_key
from scripts.train_totaltext_v4_highres_ablation import _append_epoch, _epoch_log, _json, _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.train import EpochShuffleSampler
from segmentation.training import SegmentationTrainingConfig, load_checkpoint, save_checkpoint, seed_everything
from segmentation.word_centerline import (WordCenterlineDataset, centerline_targets,
    decoded_radius, train_centerline_epoch)
from segmentation.word_multitask_evaluation import aggregate_word_metrics
from segmentation.word_multitask_provenance import json_contract


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/word_centerline_v1.json"
SOURCE_FILES = (
    "configs/approximate/word_centerline_v1.json",
    "models/word_centerline.py", "segmentation/word_centerline.py", "localize/word_centerline.py",
    "scripts/run_word_centerline.py", "scripts/run_unet_word_research_v2.py",
    "models/totaltext_word_kernel.py", "models/unet_adv.py",
    "segmentation/data.py", "segmentation/transforms.py", "segmentation/training.py",
    "segmentation/train.py", "segmentation/evaluate.py", "segmentation/word_multitask.py",
    "segmentation/totaltext_word_kernel.py", "segmentation/totaltext_image_frequency_augment.py",
    "segmentation/word_multitask_evaluation.py", "segmentation/word_multitask_provenance.py",
    "localize/word_multitask.py", "localize/crop_recovery.py",
    "scripts/train_totaltext_v4_highres_ablation.py",
    "scripts/train_totaltext_v4_image_frequency_ablation.py",
    "scripts/train_totaltext_letterbox_word_f1_separator_head.py",
    "scripts/evaluate_word_multitask_crops.py",
    "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
    "scripts/evaluate_totaltext_highres_contour_padding.py",
    "recognition/scene_text_v1_evaluation.py",
)


@dataclass(frozen=True)
class CenterlineTrainingConfig(SegmentationTrainingConfig):
    loss: str = "centerline_BCE_Dice_and_log_radius_SmoothL1"

    def __post_init__(self):
        if (self.loss != "centerline_BCE_Dice_and_log_radius_SmoothL1"
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError("Invalid centerline training settings")
        values = asdict(self)
        values.update(loss="masked_BCEWithLogitsLoss", learning_rate=1e-4, batch_size=8)
        SegmentationTrainingConfig(**values)


def training_config(cfg):
    return CenterlineTrainingConfig(epochs=cfg["epochs"], seed=cfg["seed"],
        adam_betas=(.9, .999), adam_epsilon=1e-8, learning_rate_schedule="none",
        checkpoint_selection="validation_word_f1", deterministic_algorithms=True,
        learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"])


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or (REPO / "configs/approximate").resolve() not in path.parents:
        raise ValueError("Explicit APPROXIMATE config required; no namespace fallback")
    cfg = json.loads(path.read_text())
    if (cfg.get("schema_version") != 1 or cfg.get("experiment_kind") != "frozen_word_centerline_radius_v1"
            or cfg.get("reproduction_mode") != mode or cfg.get("claim_label") != "approximate reproduction"
            or cfg.get("methods") != list(METHODS) or cfg.get("checkpoint_selection_method") != "line_radius"
            or cfg.get("input_size") != [512, 512] or cfg.get("native_mask_size") != [256, 256]
            or cfg.get("validation_count") != 125 or cfg.get("expected_validation_words") != 983):
        raise ValueError("Unsupported centerline experiment contract")
    source_path = (REPO / cfg["source_config"]).resolve()
    if ((REPO / "configs/approximate").resolve() not in source_path.parents
            or sha256_file(source_path) != cfg["source_config_sha256"]):
        raise ValueError("Qualified approximate preprocessing profile changed")
    for key in ("source_checkpoint_sha256", "source_run_manifest_sha256", "word_manifest_sha256",
                "validation_ids_sha256"):
        if not isinstance(cfg[key], str) or len(cfg[key]) != 64 or any(c not in "0123456789abcdef" for c in cfg[key]):
            raise ValueError("Pinned checksum required: " + key)
    for key in ("epochs", "batch_size", "private_head_channels", "source_epoch"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if type(cfg["seed"]) is not int or type(cfg["num_workers"]) is not int or cfg["num_workers"] < 0:
        raise ValueError("Explicit seed and nonnegative worker count required")
    targets, loss, post = (cfg[k] for k in ("targets", "loss", "postprocessing"))
    if (not 0 <= cfg["skip_floor"] < 1 or not 0 < targets["distance_fraction"] < 1
            or any(not np.isfinite(v) or v < 0 for v in loss.values())
            or min(loss["smooth"], loss["radius_floor"], loss["radius_smooth_l1_beta"]) <= 0
            or loss["bce_weight"] + loss["dice_weight"] <= 0
            or not np.isfinite([cfg["initial_line_bias"], cfg["initial_radius"]]).all()
            or cfg["initial_radius"] <= loss["radius_floor"]):
        raise ValueError("Invalid target/loss/initialization settings")
    centerline_targets(np.zeros((8, 8), np.int32), np.ones((8, 8), bool), targets)
    dummy = {"words": np.zeros((2, 8, 8), np.float32), "line": np.zeros((8, 8), np.float32),
             "radius": np.ones((8, 8), np.float32)}
    for method in METHODS:
        centerline_instances(dummy, np.ones((8, 8), bool), post, method)
    if (post["method"] != "watershed" or post["padding"] != 0 or post["min_area"] != 0
            or post["max_instances"] is not None or post["min_line_area"] != 0
            or post["radius_floor"] != loss["radius_floor"]):
        raise ValueError("Matched diagnostic requires explicit unfiltered pad0 decoders")
    allowed = {"PAPER_SPECIFIED", "REPOSITORY_SPECIFIED", "ASSUMPTION", "HARDWARE_DEVIATION", "UNRESOLVED"}
    if not cfg["scientific_choices"] or any(c["classification"] not in allowed for c in cfg["scientific_choices"]):
        raise ValueError("Scientific choice classifications required")
    recognition = cfg["recognition"]
    if (recognition["decode_ar"] is not True or recognition["refine_iters"] != 1
            or recognition["casefold"] is not True or recognition["symbol_normalized_secondary_metric"] is not True
            or type(recognition["batch_size"]) is not int or recognition["batch_size"] != 8):
        raise ValueError("Qualified AR/refine1, casefold and batch8 recognition contract required")
    training_config(cfg)
    return cfg, json.loads(source_path.read_text())


def memory(device, cfg, *, oom=False):
    cuda = device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_initialized()
    return {"device": str(device), "dtype": "float32", "batch_size": cfg["batch_size"],
        "input_shape": [cfg["batch_size"], 3, *cfg["input_size"]], "oom": oom,
        "model_set": "one frozen Text+Kernel AG-U-Net plus private centerline/radius head; no recognizer resident",
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}


def source_state_hash(state):
    digest = sha256()
    for name in sorted(state):
        if name.startswith(CenterlineRadiusAGUNet.PRIVATE_PREFIX):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def qualified_source_state(directory, cfg):
    state = load_source_state(directory, cfg)
    report = json.loads((directory / "run_manifest.json").read_text())
    files = report["run_contract"]["source_sha256"]
    if not files:
        raise ValueError("Qualified control lacks its original scientific source hashes")
    for name, expected in files.items():
        path = (REPO / name).resolve()
        if REPO not in path.parents or sha256_file(path) != expected:
            raise ValueError("Qualified source-control code changed: " + name)
    return state


def _add(totals, values):
    for key, value in values.items():
        totals[key] = totals.get(key, 0) + value


def measured_geometry(labels, owners, expected, target, validity):
    """Entire predicted boxes; exclude only wholly invalid predictions after inference."""
    ignored = [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & validity)]
    care = labels.copy()
    care[np.isin(care, ignored)] = 0
    reference = owners.copy()
    reference[~target | ~validity] = 0
    native = instance_geometry(care, reference, expected)
    foreground = labels > 0
    tp, fp, fn = [int(np.count_nonzero(x & validity)) for x in
                  (foreground & target, foreground & ~target, ~foreground & target)]
    return native, ignored, (tp, fp, fn), reference


def _aggregate(totals, pixels):
    result = aggregate_word_metrics(totals)
    result["foreground_iou"] = float(np.mean([tp / (tp + fp + fn) if tp + fp + fn else 1.
                                               for tp, fp, fn in pixels]))
    result["foreground_f1_macro"] = float(np.mean([2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.
                                                    for tp, fp, fn in pixels]))
    tp, fp, fn = np.sum(pixels, axis=0)
    result.update(foreground_iou_micro=float(tp / (tp + fp + fn)) if tp + fp + fn else 1.,
                  foreground_f1_micro=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.)
    return result


def save_maps(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError("Existing maps differ: " + str(path))
    else:
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)


def render_preview(rgb, evidence, proposals, path):
    if path.exists():
        return
    first = rgb.copy()
    draw = ImageDraw.Draw(first)
    for row in proposals:
        x1, y1, x2, y2 = row["bbox"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="#00ee44", width=2)
    panels = [("Crop boxes", first), ("Word", evidence["words"][0]), ("Centerline", evidence["line"]),
              ("Radius / 32 (display only)", evidence["radius"] / 32)]
    sheet = Image.new("RGB", (256 * len(panels), 280), "white")
    for i, (title, item) in enumerate(panels):
        if isinstance(item, np.ndarray):
            item = Image.fromarray(np.rint(np.clip(item, 0, 1) * 255).astype(np.uint8))
        item = item.copy().convert("RGB")
        item.thumbnail((256, 256))
        sheet.paste(item, (256 * i, 24))
        ImageDraw.Draw(sheet).text((256 * i + 3, 4), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def evaluate(model, batches, references, cfg, device, *, methods=METHODS, output=None, samples=None, root=None):
    model.eval()
    parts = {m: {"totals": {}, "pixels": [], "per_image": []} for m in methods}
    seen, line_counts, radius_errors, log_radius_errors = [], np.zeros(3, np.int64), [], []
    with torch.inference_mode():
        for batch in batches:
            raw = model.forward_evidence(batch["image"].to(device))
            words = raw["words"].sigmoid().cpu().numpy()
            lines = raw["line"].sigmoid().cpu().numpy()[:, 0]
            radii = decoded_radius(raw["radius_raw"], cfg["loss"]["radius_floor"]).cpu().numpy()[:, 0]
            for i, sid in enumerate(batch["sample_id"]):
                seen.append(sid)
                owners, expected, canvas = references[sid]
                target, valid = [batch[k][i, 0].numpy().astype(bool) for k in ("mask", "validity")]
                evidence = {"words": words[i], "line": lines[i], "radius": radii[i]}
                # Freeze every prediction before reading GT for any metric.
                predictions = {m: centerline_instances(evidence, canvas, cfg["postprocessing"], m) for m in methods}
                line_target = batch["line"][i, 0].numpy() > 0
                line_prediction = lines[i] >= cfg["postprocessing"]["line_threshold"]
                line_counts += [int(np.count_nonzero(x & valid)) for x in
                    (line_prediction & line_target, line_prediction & ~line_target, ~line_prediction & line_target)]
                weight = batch["radius_weight"][i, 0].numpy()
                if weight.sum() > 0:
                    radius_target = batch["radius"][i, 0].numpy()
                    radius_errors.append(float((np.abs(radii[i] - radius_target) * weight).sum() / weight.sum()))
                    log_radius_errors.append(float((np.abs(np.log(radii[i]) - np.log(np.maximum(
                        radius_target, cfg["loss"]["radius_floor"]))) * weight).sum() / weight.sum()))
                maps_path = None
                if output is not None:
                    maps_path = output / "samples" / sid.replace(":", "__") / "maps.npz"
                    save_maps(maps_path, {**evidence, "owners": owners, "expected_owner_ids": np.asarray(expected, np.int32),
                        "canvas": canvas, "validity": valid, "text_target": target,
                        "line_target": batch["line"][i, 0].numpy(), "radius_target": batch["radius"][i, 0].numpy(),
                        "radius_weight": batch["radius_weight"][i, 0].numpy(),
                        "line_skeleton_target": batch["line_skeleton"][i, 0].numpy()})
                    image_path = Path(batch["image_path"][i])
                    if sha256_file(image_path) != samples[sid].image_sha256:
                        raise ValueError("Original RGB changed: " + sid)
                    with Image.open(image_path) as opened:
                        rgb = opened.convert("RGB")
                    rgb_array = np.asarray(rgb)
                for method, part in parts.items():
                    labels, audit, _ = predictions[method]
                    metrics, ignored, counts, _ = measured_geometry(labels, owners, expected, target, valid)
                    _add(part["totals"], metrics)
                    part["pixels"].append(counts)
                    row = {"sample_id": sid, "native_geometry": metrics, "audit": audit,
                           "ignored_prediction_ids": ignored}
                    if output is not None:
                        proposals = parseq_word_proposals(labels, rgb.size, padding=0, min_area=0, max_instances=None)
                        for proposal in proposals:
                            x1, y1, x2, y2 = proposal["bbox"]
                            proposal["crop_rgb_sha256"] = crop_array_hash(rgb_array[y1:y2, x1:x2])
                        base = output / "centerline_radius" / method / "samples" / sid.replace(":", "__")
                        save_maps(base / "instances.npz", {"labels": labels})
                        recipe = {"sample_id": sid, "result_class": "approximate reproduction",
                            "reproduction_mode": "approximate", "image_path": str(image_path),
                            "image_sha256": samples[sid].image_sha256, "proposals": proposals, "audit": audit,
                            "source_maps_sha256": sha256_file(maps_path)}
                        write_json(base / "proposals.json", recipe)
                        render_preview(rgb, evidence, proposals, base / "predictions.png")
                        gt, annotation_hash = _train_gt(samples[sid], root, rgb.size)
                        annotation_path = root / "datasets/total_text/train/annotations" / (sid.split(":", 1)[1] + ".txt")
                        row.update(recipe, ground_truth=gt, annotation_path=str(annotation_path),
                            annotation_sha256=annotation_hash,
                            original_geometry=score_predictions([p for p in proposals if p["instance_id"] not in ignored],
                                                               gt, text_available=False),
                            exports={str(p.relative_to(output)): sha256_file(p) for p in
                                     (maps_path, base / "instances.npz", base / "proposals.json", base / "predictions.png")})
                    part["per_image"].append(row)
    if not seen or len(set(seen)) != len(seen) or seen != list(references):
        raise ValueError("Incomplete, reordered or duplicate validation population")
    result = {m: {"native": _aggregate(p["totals"], p["pixels"]), "per_image": p["per_image"]}
              for m, p in parts.items()}
    tp, fp, fn = (int(v) for v in line_counts)
    evidence_metrics = {"line_tp_fp_fn": [tp, fp, fn],
        "line_precision": tp / (tp + fp) if tp + fp else 0.,
        "line_recall": tp / (tp + fn) if tp + fn else 0.,
        "line_f1_micro": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
        "radius_mae_native_pixels": float(np.mean(radius_errors)) if radius_errors else None,
        "radius_log_absolute_error": float(np.mean(log_radius_errors)) if log_radius_errors else None,
        "radius_metric_images_with_gt_line": len(radius_errors),
        "radius_metric_weighting": "equal owner weights within each image, then equal image mean; GT ribbon only"}
    for part in result.values():
        part["native"].update(evidence_metrics)
    if any(p["native"]["gt_words"] != cfg["expected_validation_words"] for p in result.values()):
        raise ValueError("Changed care-word denominator")
    return result


def preflight(dataset, indices, references, cfg, output):
    totals, pixels, rows = {}, [], []
    for position, index in enumerate(indices, 1):
        batch = dataset[index]
        sid = batch["sample_id"]
        owners, expected, canvas = references[sid]
        valid, target = [batch[k][0].numpy().astype(bool) for k in ("validity", "mask")]
        # Historical validation references retain the nearest-resized owner
        # map. The paired dataset additionally clips it to the transformed
        # binary target/validity, as every previous word-head trainer does.
        owners = owners.copy()
        owners[~target | ~valid] = 0
        if not np.array_equal(owners, batch["owners_native"].numpy()):
            raise ValueError("Paired owner/reference mapping changed: " + sid)
        targets = centerline_targets(owners, valid, cfg["targets"])
        evidence = {"words": np.stack((target, batch["kernel"][0].numpy())).astype(np.float32),
                    "line": targets["line"], "radius": targets["radius"]}
        labels, audit, _ = centerline_instances(evidence, canvas, cfg["postprocessing"], "line_radius")
        native, ignored, counts, _ = measured_geometry(labels, owners, expected, target, valid)
        _add(totals, native)
        pixels.append(counts)
        cc_count, components = cv2.connectedComponentsWithAlgorithm(
            (targets["line"] > 0).astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
        mixed = sum(len(np.unique(owners[(components == c) & (owners > 0)])) > 1 for c in range(1, cc_count))
        rows.append({"sample_id": sid, "expected_words": len(expected), "targets": targets["audit"],
                     "mixed_line_components": mixed, "oracle_native_geometry": native,
                     "oracle_decoder_audit": audit, "ignored_oracle_prediction_ids": ignored})
        if position <= 6:
            with Image.open(batch["image_path"]) as opened:
                rgb = opened.convert("RGB")
            proposals = parseq_word_proposals(labels, rgb.size, padding=0, min_area=0, max_instances=None)
            render_preview(rgb, evidence, proposals, output / "target_oracle_previews" / (sid.replace(":", "__") + ".png"))
        if position % 25 == 0:
            print(f"[target oracle] {position}/{len(indices)}", flush=True)
    aggregate = _aggregate(totals, pixels)
    if len(rows) != cfg["validation_count"] or aggregate["gt_words"] != cfg["expected_validation_words"]:
        raise ValueError("Incomplete centerline target qualification")
    return {"status": "PASS", "result_class": "smoke-test result", "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "images": len(rows), "resolved_config": cfg,
        "validation_ids_sha256": cfg["validation_ids_sha256"], "oracle_native_geometry": aggregate,
        "native_words": sum(r["targets"]["words"] for r in rows),
        "mixed_line_components": sum(r["mixed_line_components"] for r in rows), "per_image": rows,
        "limitations": "GT-target oracle only, not learned performance or OCR; native geometry not Table5 accuracy."}


def train_head(output, model, optimizer, train_loader, val_loader, references, transform, sampler,
               cfg, contract, device, stop_after_epoch=None):
    tc = training_config(cfg)
    last, best = output / "last.pt", output / "best.pt"
    history = _epoch_log(output / "epochs.jsonl")
    step, epoch, best_epoch, best_metric = 0, 0, None, None
    original_hash = source_state_hash(model.state_dict())

    def save(path, e, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=e, global_step=step, config=tc,
            resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract,
            train_metrics=trained, validation_metrics=metrics)

    def load(path, **kwargs):
        payload = load_checkpoint(path, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc, **kwargs)
        if source_state_hash(model.state_dict()) != original_hash:
            raise ValueError("Checkpoint changed frozen source tensors")
        return payload

    if best.exists():
        saved = load(best)
        best_epoch, best_metric = saved["epoch"], saved["validation_metrics"]
    if last.exists():
        saved = load(last, optimizer=optimizer, restore_rng=True)
        step, epoch = saved["global_step"], saved["epoch"]
        if epoch and len(history) == epoch - 1:
            recovered = {"epoch": epoch, "global_step": step, "train": saved["train_metrics"],
                "validation": saved["validation_metrics"], "last_checkpoint_sha256": sha256_file(last),
                "recovered_from_checkpoint": True}
            _append_epoch(output / "epochs.jsonl", recovered)
            history.append(recovered)
        if len(history) != epoch or (history and history[-1]["last_checkpoint_sha256"] != sha256_file(last)):
            raise ValueError("Checkpoint/history mismatch")
        if best_metric is None or selection_key(saved["validation_metrics"], epoch) > selection_key(best_metric, best_epoch):
            best_epoch, best_metric = epoch, saved["validation_metrics"]
            save(best, epoch, saved["train_metrics"], best_metric)
        del saved
    elif history or best.exists():
        raise ValueError("Training files lack a resumable latest checkpoint")
    else:
        initial = evaluate(model, val_loader, references, cfg, device, methods=("line_radius",))["line_radius"]["native"]
        best_epoch, best_metric = 0, initial
        save(last, 0, {}, initial)
        save(best, 0, {}, initial)
        _json(output / "epoch0.json", initial)
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Noncontiguous epoch history")
    budget = cfg["epochs"] if stop_after_epoch is None else stop_after_epoch
    if not 1 <= budget <= cfg["epochs"]:
        raise ValueError("Stop epoch must be within the configured training budget")
    for e in range(epoch + 1, budget + 1):
        sampler.set_epoch(e - 1)
        transform.set_epoch(e - 1)
        trained = train_centerline_epoch(model, train_loader, optimizer, device, cfg)
        metrics = evaluate(model, val_loader, references, cfg, device, methods=("line_radius",))["line_radius"]["native"]
        step += trained["steps"]
        digest = save(last, e, trained, metrics)
        if selection_key(metrics, e) > selection_key(best_metric, best_epoch):
            best_epoch, best_metric = e, metrics
            save(best, e, trained, metrics)
        _append_epoch(output / "epochs.jsonl", {"epoch": e, "global_step": step, "train": trained,
            "validation": metrics, "last_checkpoint_sha256": digest})
        history.append({"epoch": e})
        print(f"[centerline] {e}/{cfg['epochs']} loss={trained['loss']:.5f} line={trained['line']:.5f} "
              f"radius={trained['radius']:.5f} wordF1={metrics['proposal_f1_iou_50']:.5f} best={best_epoch}", flush=True)
    if source_state_hash(model.state_dict()) != original_hash:
        raise RuntimeError("Training modified the frozen source")
    complete = len(history) == cfg["epochs"]
    return {"status": "SUCCESS" if complete else "IN_PROGRESS", "training_complete": complete,
        "reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "completed_epochs": len(history), "run_contract": contract,
        "checkpoint": {"epoch": best_epoch, "path": "best.pt", "sha256": sha256_file(best)},
        "validation_word_geometry": best_metric, "memory": memory(device, cfg),
        "environment": experiment_environment(device)}


def qualify_complete(output, cfg, contract):
    report = json.loads((output / "run_manifest.json").read_text())
    if (report.get("status") != "SUCCESS" or report.get("training_complete") is not True
            or report.get("completed_epochs") != cfg["epochs"] or report.get("run_contract") != contract
            or report["checkpoint"]["path"] != "best.pt"
            or report["checkpoint"]["sha256"] != sha256_file(output / "best.pt")):
        raise ValueError("Evaluation requires unchanged, completed training; partial runs are not results")
    history = _epoch_log(output / "epochs.jsonl")
    if ([r["epoch"] for r in history] != list(range(1, cfg["epochs"] + 1))
            or history[-1]["last_checkpoint_sha256"] != sha256_file(output / "last.pt")):
        raise ValueError("Completed history/latest checkpoint changed")
    return report


def run(args):
    cfg, source = read_config(args.config, args.mode)
    if args.cpu_threads < 1:
        raise ValueError("Positive CPU thread count required")
    torch.set_num_threads(args.cpu_threads)
    for key in ("batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            if args.action != "smoke" or value < (1 if key == "batch_size" else 0):
                raise ValueError("Runtime batch/worker overrides are allowed only for isolated smokes")
            cfg[key] = value
            cfg["scientific_choices"].append({"component": key + "_smoke_override", "classification": "HARDWARE_DEVIATION",
                "rationale": "Explicit CLI smoke override", "expected_effect": "Engineering qualification only"})
    root, source_run, output = args.dataset_root.resolve(), args.source_run.resolve(), args.output_root.resolve()
    if output == REPO or any(output == p or p in output.parents for p in (root, source_run)):
        raise ValueError("Output must be isolated from source datasets/checkpoints")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Requested device unavailable; choose CPU smoke/preflight or a qualified CUDA host")
    manifest_path = root / cfg["train_manifest"]
    if sha256_file(manifest_path) != cfg["word_manifest_sha256"]:
        raise ValueError("Qualified word-target manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Qualified Total-Text TRAIN dataset required")
    train_ids, val_ids, fingerprint = _split_indices(manifest, cfg["seed"], cfg["validation_count"])
    if fingerprint != cfg["validation_ids_sha256"]:
        raise ValueError("Frozen validation split changed")
    state = qualified_source_state(source_run, cfg)
    original_hash = source_state_hash(state)
    transforms = [_transform(source, "letterbox", train=v) for v in (True, False)]
    datasets = [WordCenterlineDataset(SegmentationDataset(manifest, root, transform=t),
        distance_fraction=cfg["targets"]["distance_fraction"], target_config=cfg["targets"]) for t in transforms]
    references = _validation_reference(manifest, root, val_ids, transforms[1])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process holds the centerline experiment lock") from error
        try:
            if args.action == "preflight":
                result = preflight(datasets[1], val_ids, references, cfg, output)
                result["runtime_inputs"] = {"dataset_root": str(root), "source_run": str(source_run)}
                result["source_state_sha256"] = original_hash
                result["source_sha256"] = {p: sha256_file(REPO / p) for p in
                    dict.fromkeys((*SOURCE_FILES, cfg["source_config"]))}
                result["source_sha256"][str(Path(args.config).resolve().relative_to(REPO))] = sha256_file(args.config)
                result["environment"] = experiment_environment(device)
                write_json(output / "target_qualification.json", result)
                return result
            seed_everything(cfg["seed"], deterministic_algorithms=True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model = CenterlineRadiusAGUNet(cfg)
            initialization = model.initialize_control_state(state)
            del state
            source_files = dict.fromkeys((*SOURCE_FILES, cfg["source_config"]))
            source_hashes = {p: sha256_file(REPO / p) for p in source_files}
            source_hashes[str(Path(args.config).resolve().relative_to(REPO))] = sha256_file(args.config)
            contract = json_contract({"schema_version": 1, "reproduction_mode": "approximate",
                "result_class": "approximate reproduction", "resolved_config": cfg,
                "resolved_preprocessing": source, "architecture": model.architecture_identity,
                "validation_ids_sha256": fingerprint, "initialization": initialization,
                "source_state_sha256": original_hash, "source_sha256": source_hashes,
                "runtime_inputs": {"dataset_root": str(root), "source_run": str(source_run)},
                "training_config": asdict(training_config(cfg)), "execution_device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "backend": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "cpu_threads": torch.get_num_threads()}, "environment": experiment_environment(device)})
            write_json(output / "run_contract.json", contract)
            if device.type == "cuda":
                _prepare_cuda_memory_tracking(device)
            model.to(device)
            options = {"num_workers": cfg["num_workers"], "timeout": 60 if cfg["num_workers"] else 0,
                       "pin_memory": device.type == "cuda"}
            sampler = EpochShuffleSampler(len(train_ids), seed=cfg["seed"])
            train_loader = DataLoader(Subset(datasets[0], train_ids), batch_size=cfg["batch_size"], sampler=sampler, **options)
            val_loader = DataLoader(Subset(datasets[1], val_ids), batch_size=cfg["batch_size"], shuffle=False, **options)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], betas=(.9, .999), eps=1e-8)
            if args.action == "smoke":
                # Validation geometry supplies true canvas even for a shortened smoke;
                # training uses a separate augmented batch and never sees validation GT.
                batch = next(iter(train_loader))
                before = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
                trained = train_centerline_epoch(model, [batch], optimizer, device, cfg)
                updates = {k: float((p.detach() - before[k]).abs().max()) for k, p in model.named_parameters() if k in before}
                if not any(updates.values()) or source_state_hash(model.state_dict()) != original_hash:
                    raise RuntimeError("Smoke failed private update/frozen source invariance")
                vb = next(iter(val_loader))
                short_references = {sid: references[sid] for sid in vb["sample_id"]}
                short_cfg = {**cfg, "expected_validation_words": sum(len(r[1]) for r in short_references.values())}
                geometry = evaluate(model, [vb], short_references, short_cfg, device)
                result = {"status": "PASS", "result_class": "smoke-test result", "claim_label": "approximate reproduction",
                    "reproduction_mode": "approximate", "run_contract": contract, "train": trained,
                    "geometry": {m: p["native"] for m, p in geometry.items()}, "head_update_linf": updates,
                    "frozen_source_unchanged": True, "memory": memory(device, cfg), "environment": experiment_environment(device)}
                write_json(output / "smoke.json", result)
                return result
            if args.action == "train":
                receipt = output / "run_manifest.json"
                previous = json.loads(receipt.read_text()) if receipt.exists() else None
                if previous is not None and previous.get("status") == "SUCCESS":
                    return qualify_complete(output, cfg, contract)
                result = train_head(output, model, optimizer, train_loader, val_loader, references,
                    transforms[0], sampler, cfg, contract, device, args.stop_after_epoch)
                if previous is not None:
                    for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
                        old, now = previous.get("memory", {}).get(key), result["memory"][key]
                        if old is not None and now is not None:
                            result["memory"][key] = max(old, now)
                _json(receipt, result)
                return result
            report = qualify_complete(output, cfg, contract)
            protocol = {"run_contract": contract, "checkpoint_sha256": report["checkpoint"]["sha256"],
                "source_seed": cfg["seed"], "expected_images": cfg["validation_count"],
                "expected_care_words": cfg["expected_validation_words"],
                "metric_protocol": "Native filled-word bbox one-to-one IoU>=.5, entire predicted boxes; "
                    "all expected words retained. Original boxes use letterbox remap/pad0. Ignore only fully invalid predictions.",
                "checkpoint_selection": "line_radius native geometry F1, then fewer fused GT, recall, foreground IoU, earliest epoch; epoch0 included"}
            cached = output / "geometry_results.json"
            if cached.exists():
                result = json.loads(cached.read_text())
                if result.get("status") != "SUCCESS" or result.get("protocol") != protocol:
                    raise ValueError("Existing geometry provenance differs")
                verify_geometry_exports(result, output)
                return result
            load_checkpoint(output / "best.pt", model=model, expected_architecture=model.architecture_identity,
                expected_run_contract=contract, expected_training_config=training_config(cfg))
            if source_state_hash(model.state_dict()) != original_hash:
                raise ValueError("Selected checkpoint changed frozen source")
            parts = evaluate(model, val_loader, references, cfg, device, output=output,
                samples={s.sample_id: s for s in manifest.samples}, root=root)
            if parts["line_radius"]["native"] != report["validation_word_geometry"]:
                raise ValueError("Selected checkpoint geometry does not reproduce its training receipt")
            result = {"status": "SUCCESS", "reproduction_mode": "approximate", "result_class": "approximate reproduction",
                "protocol": protocol, "branches": {"centerline_radius": parts}, "memory": memory(device, cfg)}
            verify_geometry_exports(result, output)
            write_json(cached, result)
            return result
        except Exception as error:
            _json(output / "failure.json", {"status": "FAILED", "action": args.action,
                "error_type": type(error).__name__, "message": str(error),
                "reproduction_mode": "approximate", "result_class": "smoke-test result" if args.action in ("preflight", "smoke") else "approximate reproduction",
                "resolved_config": cfg, "memory": memory(device, cfg, oom=isinstance(error, torch.cuda.OutOfMemoryError))})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--action", choices=["preflight", "smoke", "train", "evaluate"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("dataset-root", "source-run", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args(argv)
    if args.stop_after_epoch is not None and args.action != "train":
        raise ValueError("Stop epoch is valid only for training")
    result = run(args)
    print(json.dumps({"status": result["status"], "action": args.action,
        "result_class": result["result_class"], "output_root": str(args.output_root.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
