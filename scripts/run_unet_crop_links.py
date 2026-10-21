"""Isolated learned second-pass experiment on frozen TRAIN predictions."""

import argparse
from dataclasses import asdict, replace
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from localize.unet_crop_links import (context_box, original_label_map, patches_from_labels,
    merge_duplicate_supports, reciprocal_components)
from localize.word_multitask import parseq_word_proposals, word_instances
from models.unet_crop_links import CropLinkAGUNet
from models.unet_word_research_v2 import WordResearchV2AGUNet
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _geometry_match, _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_word_research_v2 import (backend_state, load_source_state, memory,
    read_config as read_v2_config, training_config)
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _append_epoch, _epoch_log, _json, _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.train_totaltext_letterbox_word_f1_separator_head import selection_key
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.train import EpochShuffleSampler
from segmentation.training import load_checkpoint, save_checkpoint, seed_everything
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.unet_crop_links import PredictedCropDataset, train_crop_epoch
from segmentation.unet_word_research_v2 import build_research_v2_optimizer
from segmentation.word_multitask_evaluation import aggregate_word_metrics
from segmentation.word_multitask_provenance import json_contract


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/unet_crop_links_v1.json"
BRANCHES = ("control", "global_links", "crop_word", "crop_links")
FILES = ("models/unet_crop_links.py", "segmentation/unet_crop_links.py", "localize/unet_crop_links.py",
    "scripts/run_unet_crop_links.py", "models/unet_word_research_v2.py", "segmentation/unet_word_research_v2.py",
    "localize/unet_word_research_v2.py", "scripts/run_unet_word_research_v2.py", "models/unet_adv.py",
    "models/totaltext_word_kernel.py", "segmentation/data.py", "segmentation/transforms.py",
    "segmentation/training.py", "segmentation/totaltext_word_kernel.py", "segmentation/word_multitask.py",
    "segmentation/totaltext_image_frequency_augment.py", "localize/word_multitask.py", "localize/crop_recovery.py",
    "scripts/train_totaltext_v4_highres_ablation.py", "scripts/train_totaltext_v4_image_frequency_ablation.py",
    "scripts/train_totaltext_letterbox_word_f1_separator_head.py",
    "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py", "recognition/scene_text_v1_evaluation.py",
    "segmentation/word_multitask_evaluation.py", "segmentation/evaluate.py", "scripts/run_word_multitask.py")


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Learned crops require an explicit APPROXIMATE configuration")
    c = json.loads(path.read_text())
    if (c.get("experiment_kind") != "unet_crop_links_v1" or c.get("reproduction_mode") != mode
            or c.get("claim_label") != "approximate reproduction" or c.get("schema_version") != 1
            or c.get("branches") != list(BRANCHES) or c.get("embedding_arms") != []):
        raise ValueError("Unsupported learned-crop configuration")
    v2path = (REPO / c["v2_config"]).resolve()
    if sha256_file(v2path) != c["v2_config_sha256"]:
        raise ValueError("Explicit V2 source configuration changed")
    v2, source = read_v2_config(v2path, mode)
    for k in ("source_checkpoint_sha256", "source_run_manifest_sha256", "source_epoch", "source_architecture",
              "source_config", "source_config_sha256", "word_manifest_sha256", "train_manifest",
              "validation_ids_sha256", "validation_count", "expected_validation_words", "input_size", "native_mask_size",
              "seed", "skip_floor"):
        if c[k] != v2[k]:
            raise ValueError("Qualified source/split differs: " + k)
    for k in ("epochs", "batch_size", "private_head_channels"):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError("Invalid positive integer: " + k)
    if type(c["num_workers"]) is not int or c["num_workers"] < 0:
        raise ValueError("Invalid worker count")
    if type(c["cpu_threads"]) is not int or c["cpu_threads"] < 1 or not 0 <= c["crop_flip_probability"] <= 1:
        raise ValueError("Invalid CPU threads or crop augmentation")
    numeric = [c[k] for k in ("learning_rate", "head_learning_rate", "weight_decay")]
    if not np.isfinite(numeric).all() or min(numeric[:2]) <= 0 or numeric[2] < 0:
        raise ValueError("Invalid optimizer")
    crop, links = c["crop"], c["links"]
    if (not np.isfinite(list(crop.values())).all() or crop["context_fraction"] < 0
            or type(crop["minimum_context_pixels"]) is not int or crop["minimum_context_pixels"] < 0
            or not 0 < crop["minimum_visible_fraction"] <= 1 or not 0 < crop["duplicate_overlap"] <= 1
            or not 0 < links["threshold"] < 1 or not np.isfinite(links["loss_weight"]) or links["loss_weight"] <= 0
            or type(links["ramp_epochs"]) is not int or links["ramp_epochs"] < 1):
        raise ValueError("Invalid crop/link settings")
    if c["postprocessing"] != v2["postprocessing"] or c["targets"] != v2["targets"] or c["loss"] != v2["loss"]:
        raise ValueError("Frozen baseline targets/extraction/loss changed")
    if c["evaluation_iou_thresholds"] != [.25, .5, .75] or c["ocr_iou_threshold"] != .5:
        raise ValueError("This original-coordinate protocol requires explicit .25/.5/.75 geometry and .5 OCR")
    training_config(c, "control")
    return c, source


def qualify_global(directory, config):
    receipt_path = directory / "run_manifest.json"
    if not receipt_path.exists():
        raise ValueError("V2 links_isolated has not completed; wait for the V2 suite")
    receipt = json.loads(receipt_path.read_text()); contract = receipt["run_contract"]
    expected, _ = read_v2_config(REPO / config["v2_config"], "approximate")
    checkpoint = directory / receipt["checkpoint"]["path"]
    if (receipt.get("status") != "SUCCESS" or not receipt.get("training_complete")
            or receipt.get("completed_epochs") != expected["epochs"] or contract["branch"] != "links_isolated"
            or contract["resolved_config"] != expected or contract["validation_ids_sha256"] != config["validation_ids_sha256"]
            or sha256_file(checkpoint) != receipt["checkpoint"]["sha256"]):
        raise ValueError("V2 global-link checkpoint is incomplete or incompatible")
    report_path = directory / "validation.json"
    report = json.loads(report_path.read_text())
    if (report["run_contract"] != contract or report["checkpoint_sha256"] != receipt["checkpoint"]["sha256"]
            or report["aggregate"]["gt_words"] != config["expected_validation_words"]
            or len(report["per_image"]) != config["validation_count"]):
        raise ValueError("V2 global-link exports are incomplete")
    for name, digest in contract["source_sha256"].items():
        if sha256_file(REPO / name) != digest:
            raise ValueError("V2 scientific source changed: " + name)
    return receipt, report, {"manifest_sha256": sha256_file(receipt_path),
        "validation_sha256": sha256_file(report_path), "checkpoint_sha256": sha256_file(checkpoint)}


def crop_records(proposals, image_wh, config):
    if not proposals:
        return [{"seed_id": 0, "seed_bbox": [0, 0, *image_wh],
                 "context_bbox": [0, 0, *image_wh], "whole_image_fallback": True}]
    return [{"seed_id": p["instance_id"], "seed_bbox": p["bbox"],
             "context_bbox": context_box(p["bbox"], image_wh, config["crop"]),
             "whole_image_fallback": False} for p in proposals]


def prepare(output, raw, indices, config, source, state, device):
    path = output / "predicted_crop_manifest.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["indices"] != indices:
            raise ValueError("Prepared image population changed")
        for row in result["per_image"]:
            if sha256_file(output / row["maps_path"]) != row["maps_sha256"]:
                raise ValueError("Frozen first-pass maps changed")
        return result
    seed_everything(config["seed"], deterministic_algorithms=True)
    model = WordResearchV2AGUNet("reference_control", config)
    model.initialize_control_state(state)
    if device.type == "cuda": _prepare_cuda_memory_tracking(device)
    transform = _transform(source, "letterbox", train=False)
    rows = []
    try:
        model.to(device).eval()
        with torch.inference_mode():
            for count, index in enumerate(indices):
                sample = raw.manifest.samples[index]; batch = raw[index]
                w, h = batch["image"].size
                image, _ = transform(batch["image"], Image.new("L", (w, h)), index)
                p = model(image[None].to(device))[0].sigmoid().cpu().numpy()
                canvas = letterbox_canvas_mask((w, h), native_size=config["native_mask_size"][0]).numpy()
                labels, audit, _ = word_instances(p, canvas, config["postprocessing"])
                proposals = parseq_word_proposals(labels, (w, h), padding=0, min_area=0, max_instances=None)
                directory = output / "first_pass" / sample.sample_id.replace(":", "__")
                directory.mkdir(parents=True, exist_ok=True)
                arrays = {"probabilities": p, "labels": labels, "canvas": canvas}
                maps = directory / "maps.npz"
                if maps.exists():
                    with np.load(maps, allow_pickle=False) as saved:
                        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                            raise ValueError("Partial prepared maps differ from replay")
                else:
                    np.savez_compressed(maps, **arrays)
                rows.append({"sample_id": sample.sample_id, "index": index, "image_sha256": sample.image_sha256,
                    "image_wh": [w, h], "maps_path": str(maps.relative_to(output)), "maps_sha256": sha256_file(maps),
                    "proposals": proposals, "crops": crop_records(proposals, (w, h), config), "audit": audit})
                if (count + 1) % 50 == 0: print(f"[frozen first pass] {count + 1}/{len(indices)}", flush=True)
        result = {"status": "SUCCESS", "result_class": "approximate reproduction" if len(indices) == len(raw) else "smoke-test result",
            "claim_label": "approximate reproduction", "reproduction_mode": "approximate",
            "indices": indices, "per_image": rows, "crop_count": sum(len(r["crops"]) for r in rows),
            "geometry_uses_gt": False, "checkpoint_sha256": config["source_checkpoint_sha256"],
            "memory": memory(device, {**config, "batch_size": 1})}
        _json(path, result)
        return result
    except Exception as error:
        _json(output / "preparation_failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "message": str(error), "memory": memory(device, {**config, "batch_size": 1},
                oom=isinstance(error, torch.cuda.OutOfMemoryError))})
        raise
    finally:
        del model; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def refined_patches(model, rgb, row, labels, transform, config, device, *, evidence_dir=None):
    anchor_map = original_label_map(labels, rgb.size)
    patches, audit = [], {"crop_count": len(row["crops"]), "seed_fallbacks": 0,
                          "mapped_empty_instances": 0, "whole_image_fallbacks": 0}
    pending = row["crops"]
    for start in range(0, len(pending), config["batch_size"]):
        jobs = pending[start:start + config["batch_size"]]
        images = [rgb.crop(tuple(c["context_bbox"])) for c in jobs]
        inputs = [transform(im, Image.new("L", im.size), 0)[0] for im in images]
        evidence = model.forward_evidence(torch.stack(inputs).to(device))
        for i, (crop, image) in enumerate(zip(jobs, images)):
            p = evidence["words"][i].sigmoid().cpu().numpy()
            canvas = letterbox_canvas_mask(image.size, native_size=p.shape[-1]).numpy()
            if evidence["links"] is None:
                child, _, _ = word_instances(p, canvas, config["postprocessing"])
            else:
                child, _ = reciprocal_components(p, evidence["links"][i].sigmoid().cpu().numpy(), canvas,
                                                  config["postprocessing"], config["links"])
            x, y, r, b = crop["context_bbox"]
            anchor = None if crop["whole_image_fallback"] else anchor_map[y:b, x:r] == crop["seed_id"]
            if evidence_dir is not None:
                evidence_dir.mkdir(parents=True, exist_ok=True)
                path = evidence_dir / (str(crop["seed_id"]) + ".npz")
                arrays = {"probabilities": p, "labels": child, "canvas": canvas,
                          "context_bbox": np.asarray(crop["context_bbox"], np.int32)}
                if evidence["links"] is not None:
                    arrays["links_logits"] = evidence["links"][i].cpu().numpy()
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                            raise ValueError("Existing per-crop evidence changed")
                else:
                    np.savez_compressed(path, **arrays)
                audit.setdefault("crop_evidence", []).append({"seed_id": crop["seed_id"], "path": path.name,
                                                              "sha256": sha256_file(path)})
            local, skipped = patches_from_labels(child, image.size, offset=(x, y), anchor=anchor)
            audit["mapped_empty_instances"] += skipped
            audit["whole_image_fallbacks"] += int(crop["whole_image_fallback"])
            if not local and not crop["whole_image_fallback"]:
                sx, sy, sr, sb = crop["seed_bbox"]
                support = anchor_map[sy:sb, sx:sr] == crop["seed_id"]
                if support.any():
                    local = [{"bbox": crop["seed_bbox"], "mask": support}]
                    audit["seed_fallbacks"] += 1
            patches.extend(local)
    audit["proposals_before_duplicate_merge"] = len(patches)
    result = merge_duplicate_supports(patches, overlap_fraction=config["crop"]["duplicate_overlap"])
    audit["proposals_after_duplicate_merge"] = len(result)
    return result, audit


def evaluate(branch, model, raw, val_indices, records, output, source, config, device,
             *, export_dir=None, global_directory=None):
    if model is not None: model.eval()
    transform = _transform(source, "letterbox", train=False)
    total, rows, ious, f1s = {}, [], [], []
    global_rows = None
    if branch == "global_links":
        _, report, _ = qualify_global(global_directory, config)
        global_rows = {r["sample_id"]: r for r in report["per_image"]}
    with torch.inference_mode():
        for index in val_indices:
            sample = raw.manifest.samples[index]; batch = raw[index]; record = records[sample.sample_id]
            with np.load(output / record["maps_path"], allow_pickle=False) as saved: labels = saved["labels"]
            if branch == "global_links":
                old = global_rows[sample.sample_id]
                path = global_directory / "validation_exports" / sample.sample_id.replace(":", "__") / "maps.npz"
                if sha256_file(path) != old["maps_sha256"]:
                    raise ValueError("Global-link maps changed")
                with np.load(path, allow_pickle=False) as saved: labels = saved["labels"]
            if model is None:
                patches, skipped = patches_from_labels(labels, batch["image"].size)
                audit = {"mapped_empty_instances": skipped}
            else:
                evidence_dir = export_dir / sample.sample_id.replace(":", "__") / "crop_evidence" if export_dir is not None else None
                patches, audit = refined_patches(model, batch["image"], record, labels, transform, config, device,
                                                 evidence_dir=evidence_dir)
            # All proposals are now frozen; only here may scoring consult GT.
            valid = np.asarray(batch["validity"]) > 0
            with Image.open(raw.paths[index][4]) as opened: owners = np.asarray(opened, np.int32)
            reference = owners.copy(); reference[~valid] = 0
            expected = np.unique(owners[owners > 0])
            proposals, ignored, touched = [], [], {int(owner): set() for owner in expected}
            fused, foreground = set(), np.zeros(valid.shape, bool)
            for pi, patch in enumerate(patches, start=1):
                x, y, r, b = patch["bbox"]; support = patch["mask"]
                proposals.append({"instance_id": pi, "bbox": patch["bbox"], "bbox_original": patch["bbox"],
                                  "foreground_area": int(support.sum())})
                foreground[y:b, x:r] |= support
                if not np.any(support & valid[y:b, x:r]): ignored.append(pi); continue
                words = np.unique(reference[y:b, x:r][support]); words = words[words > 0]
                for word in words: touched[int(word)].add(pi)
                if len(words) > 1: fused.add(pi)
            gt, _ = _train_gt(sample, raw.data_root, batch["image"].size)
            if len(gt) != len(expected): raise ValueError("Original word GT/owner denominator differs")
            care = [p for p in proposals if p["instance_id"] not in ignored]
            fused_words = sum(bool(components & fused) for components in touched.values())
            metrics = {"gt_words": len(gt), "raw_proposals": len(care), "gt_words_lost_in_native_target": 0,
                "gt_words_split_across_predicted_components": sum(len(c) >= 2 for c in touched.values()),
                "gt_words_touched_by_fused_components": fused_words,
                "predicted_components_touching_multiple_gt_words": len(fused)}
            for threshold in config["evaluation_iou_thresholds"]:
                key = str(round(threshold * 100))
                metrics["matched_gt_iou_ge_" + key] = len(_geometry_match(care, gt, threshold))
            for key, value in metrics.items(): total[key] = total.get(key, 0) + value
            target = reference > 0
            tp, fp, fn = [int(np.count_nonzero(m & valid)) for m in
                          (foreground & target, foreground & ~target, ~foreground & target)]
            ious.append(tp / (tp + fp + fn) if tp + fp + fn else 1.)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.)
            row = {"sample_id": sample.sample_id, **metrics, "ignored_instance_ids": ignored, "audit": audit}
            if export_dir is not None:
                directory = export_dir / sample.sample_id.replace(":", "__"); directory.mkdir(parents=True, exist_ok=True)
                maps = directory / "maps.npz"
                arrays = {"owners": reference, "validity": valid}
                arrays.update({"support_" + str(i): p["mask"] for i, p in enumerate(patches, start=1)})
                arrays["boxes"] = np.asarray([p["bbox"] for p in patches], np.int32).reshape(-1, 4)
                if maps.exists():
                    with np.load(maps, allow_pickle=False) as saved:
                        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                            raise ValueError("Existing learned-crop export changed")
                else: np.savez_compressed(maps, **arrays)
                proposal_path = directory / "proposals.json"
                _json(proposal_path, {"sample_id": sample.sample_id, "image_path": batch["image_path"],
                    "image_sha256": sample.image_sha256, "maps_sha256": sha256_file(maps), "proposals": proposals,
                    "quality_scores": {}, "result_class": "approximate reproduction", "reproduction_mode": "approximate"})
                row.update(maps_sha256=sha256_file(maps), proposals_sha256=sha256_file(proposal_path))
            rows.append(row)
    if len({r["sample_id"] for r in rows}) != len(val_indices): raise ValueError("Incomplete validation")
    aggregate = aggregate_word_metrics(total)
    aggregate.update(foreground_iou=float(np.mean(ious)), foreground_f1_macro=float(np.mean(f1s)))
    if len(val_indices) == config["validation_count"] and aggregate["gt_words"] != config["expected_validation_words"]:
        raise ValueError("Changed validation denominator")
    return {"aggregate": aggregate, "per_image": rows, "coordinate_system": "original RGB; exclusive xyxy boxes",
            "matching": "maximum one-to-one bbox IoU; same original annotation boxes as PARSeq scoring"}


def train_branch(directory, model, optimizer, train_loader, dataset, sampler, evaluate_callback,
                 config, contract, device, stop_after=None):
    tc = training_config(config, "control")
    last, best = directory / "last.pt", directory / "best.pt"
    history = _epoch_log(directory / "epochs.jsonl")
    epoch, step, best_epoch, best_metrics = 0, 0, None, None
    def save(path, e, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=e, global_step=step, config=tc,
            resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract, train_metrics=trained, validation_metrics=metrics)
    if best.exists():
        saved = load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc)
        best_epoch, best_metrics = saved["epoch"], saved["validation_metrics"]
    if last.exists():
        saved = load_checkpoint(last, model=model, optimizer=optimizer, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc, restore_rng=True)
        epoch, step = saved["epoch"], saved["global_step"]
        if epoch and len(history) == epoch - 1:
            row = {"epoch": epoch, "global_step": step, "train": saved["train_metrics"],
                "validation": saved["validation_metrics"], "last_checkpoint_sha256": sha256_file(last),
                "recovered_from_checkpoint": True}
            _append_epoch(directory / "epochs.jsonl", row); history.append(row)
        if len(history) != epoch or (history and history[-1]["last_checkpoint_sha256"] != sha256_file(last)):
            raise ValueError("Learned-crop checkpoint/history mismatch")
        if best_metrics is None or selection_key(saved["validation_metrics"], epoch) > selection_key(best_metrics, best_epoch):
            best_epoch, best_metrics = epoch, saved["validation_metrics"]; save(best, epoch, saved["train_metrics"], best_metrics)
        del saved
    elif history or best.exists():
        raise ValueError("Training history lacks latest checkpoint")
    else:
        best_epoch, best_metrics = 0, evaluate_callback()["aggregate"]
        save(last, 0, {}, best_metrics); save(best, 0, {}, best_metrics)
        _json(directory / "epoch0.json", best_metrics)
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Noncontiguous learned-crop history")
    budget = config["epochs"] if stop_after is None else stop_after
    if not 1 <= budget <= config["epochs"]: raise ValueError("Invalid stop epoch")
    for e in range(epoch + 1, budget + 1):
        model.set_research_epoch(e); dataset.set_epoch(e - 1); sampler.set_epoch(e - 1)
        trained = train_crop_epoch(model, train_loader, optimizer, device, config, e)
        metrics = evaluate_callback()["aggregate"]
        step += trained["steps"]; digest = save(last, e, trained, metrics)
        if selection_key(metrics, e) > selection_key(best_metrics, best_epoch):
            best_epoch, best_metrics = e, metrics; save(best, e, trained, metrics)
        _append_epoch(directory / "epochs.jsonl", {"epoch": e, "global_step": step, "train": trained,
            "validation": metrics, "last_checkpoint_sha256": digest})
        history.append({"epoch": e})
        print(f"[{model.crop_variant}] {e}/{config['epochs']} loss={trained['loss']:.5f} wordF1={metrics['proposal_f1_iou_50']:.5f} best={best_epoch}", flush=True)
    return {"status": "SUCCESS" if len(history) == config["epochs"] else "IN_PROGRESS",
        "training_complete": len(history) == config["epochs"], "completed_epochs": len(history),
        "checkpoint": {"epoch": best_epoch, "path": "best.pt", "sha256": sha256_file(best)},
        "validation_word_geometry": best_metrics, "run_contract": contract, "result_class": "approximate reproduction",
        "memory": memory(device, config), "environment": experiment_environment(device)}


def bind_contract(path, contract):
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Run contract changed; use a new output directory")
    else:
        _json(path, contract)


def run(args):
    config, source = read_config(args.config, args.mode)
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"], deterministic_algorithms=True)
    for key in ("batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            if type(value) is not int or value < (1 if key == "batch_size" else 0): raise ValueError("Invalid runtime override")
            config[key] = value
            config["scientific_choices"].append({"component": key + "_override", "classification": "HARDWARE_DEVIATION",
                "rationale": "Explicit CLI override", "expected_effect": "May change optimization and BN"})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no fallback")
    root, output, v1 = args.word_root.resolve(), args.output_root.resolve(), args.v1_root.resolve()
    v2 = args.v2_root.resolve() if args.v2_root else None
    if any(output == p or p in output.parents for p in (root, v1, v2) if p is not None) or output == REPO:
        raise ValueError("Output must be isolated from source artifacts")
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["word_manifest_sha256"]: raise ValueError("Word manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train": raise ValueError("Only official TRAIN inputs accepted")
    train_ids, val_ids, split = _split_indices(manifest, config["seed"], config["validation_count"])
    if split != config["validation_ids_sha256"]: raise ValueError("Frozen split changed")
    raw = SegmentationDataset(manifest, root)
    full_population = args.limit_images is None
    if not full_population:
        if args.limit_images < 1 or args.action not in ("prepare", "smoke"):
            raise ValueError("Partial image limits are only allowed for prepare/smoke qualification")
        train_ids, val_ids = train_ids[:args.limit_images], val_ids[:args.limit_images]
    indices = sorted(train_ids + val_ids)
    context = json_contract({"schema_version": 1, "resolved_config": config, "resolved_preprocessing": source,
        "validation_ids_sha256": split, "indices": indices, "execution_device": str(device),
        "runtime_inputs": {"word_root": str(root), "v1_root": str(v1), "v2_root": str(v2) if v2 else None},
        "source_sha256": {name: sha256_file(REPO / name) for name in FILES},
        "backend": backend_state(), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": experiment_environment(device), "result_class": "approximate reproduction"})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_contract(output / "run_contract.json", context)
        state = load_source_state(v1 / "control", config)
        if args.action == "prepare":
            return prepare(output, raw, indices, config, source, state, device)
        prepared = json.loads((output / "predicted_crop_manifest.json").read_text())
        if prepared["indices"] != indices or prepared["checkpoint_sha256"] != config["source_checkpoint_sha256"]:
            raise ValueError("Prepared crop population/checkpoint changed")
        records = {r["sample_id"]: r for r in prepared["per_image"]}
        for r in records.values():
            if sha256_file(output / r["maps_path"]) != r["maps_sha256"]: raise ValueError("Frozen maps changed")
        transform = _transform(source, "letterbox", train=False)
        transform.config = replace(transform.config, horizontal_flip_probability=config["crop_flip_probability"],
                                   rotation_probability=0.)
        dataset = PredictedCropDataset(raw, records, train_ids, transform, config)
        sampler = EpochShuffleSampler(len(dataset), seed=config["seed"])
        loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler,
            num_workers=config["num_workers"], pin_memory=device.type == "cuda",
            timeout=60 if config["num_workers"] else 0)
        names = (("crop_word", "crop_links") if args.action in ("train", "smoke") else BRANCHES) if args.branch == "all" else (args.branch,)
        results = []
        for branch in names:
            directory = output / branch; directory.mkdir(exist_ok=True)
            contract = json_contract({**context, "branch": branch, "prepared_crop_manifest_sha256": sha256_file(output / "predicted_crop_manifest.json"),
                                     "training_config": asdict(training_config(config, "control"))})
            if branch == "global_links":
                if v2 is None: raise ValueError("Provide --v2-root with completed global links")
                old, _, dependency = qualify_global(v2 / "links_isolated", config)
                contract["global_links_provenance"] = dependency
            bind_contract(directory / "run_contract.json", contract)
            model = optimizer = None
            try:
                seed_everything(config["seed"], deterministic_algorithms=True)
                if device.type == "cuda": _prepare_cuda_memory_tracking(device)
                if branch.startswith("crop_"):
                    model = CropLinkAGUNet(branch, config); model.initialize_control_state(state); model.to(device)
                    optimizer = build_research_v2_optimizer(model, config)
                callback = lambda: evaluate(branch, model, raw, val_ids, records, output, source, config, device,
                                             global_directory=v2 / "links_isolated" if v2 else None)
                if args.action == "smoke":
                    if model is None: raise ValueError("Smoke selects crop_word or crop_links")
                    before = model.head.weight.detach().clone()
                    trained = train_crop_epoch(model, [next(iter(loader))], optimizer, device, config, 1)
                    result = {"status": "PASS", "result_class": "smoke-test result", "run_contract": contract,
                        "train": trained, "head_update_linf": float((model.head.weight.detach() - before).abs().max()),
                        "validation": callback(), "memory": memory(device, config), "environment": experiment_environment(device)}
                    _json(directory / "smoke.json", result)
                elif args.action == "train":
                    if model is None: raise ValueError("Only crop branches are trainable")
                    result = train_branch(directory, model, optimizer, loader, dataset, sampler, callback,
                                          config, contract, device, args.stop_after_epoch)
                    prior = directory / "run_manifest.json"
                    if prior.exists():
                        old_receipt = json.loads(prior.read_text())
                        for k in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
                            a, b = old_receipt["memory"][k], result["memory"][k]
                            if a is not None and b is not None: result["memory"][k] = max(a, b)
                    _json(prior, result)
                elif args.action == "evaluate":
                    if model is not None:
                        receipt = json.loads((directory / "run_manifest.json").read_text())
                        if receipt["run_contract"] != contract or not receipt["training_complete"] or receipt["completed_epochs"] != config["epochs"]:
                            raise ValueError("Evaluation requires completed crop training")
                        if sha256_file(directory / "best.pt") != receipt["checkpoint"]["sha256"]: raise ValueError("Selected checkpoint changed")
                        load_checkpoint(directory / "best.pt", model=model, expected_architecture=model.architecture_identity,
                            expected_run_contract=contract, expected_training_config=training_config(config, "control"))
                    else:
                        checkpoint = v1 / "control/best.pt" if branch == "control" else v2 / "links_isolated/best.pt"
                        receipt = {"status": "SUCCESS", "training_complete": True, "completed_epochs": 0,
                            "execution_kind": "frozen_reference", "run_contract": contract,
                            "checkpoint": {"epoch": config["source_epoch"] if branch == "control" else old["checkpoint"]["epoch"],
                                           "path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                            "result_class": "approximate reproduction", "memory": memory(device, config)}
                        receipt["memory"].update(model_set="cached first-pass labels; no neural model resident",
                                                 input_shape=None, batch_size=None, dtype="int32")
                        _json(directory / "run_manifest.json", receipt)
                    result = evaluate(branch, model, raw, val_ids, records, output, source, config, device,
                        export_dir=directory / "validation_exports", global_directory=v2 / "links_isolated" if v2 else None)
                    result.update(run_contract=contract, checkpoint_sha256=receipt["checkpoint"]["sha256"], result_class="approximate reproduction")
                    _json(directory / "validation.json", result)
                else: raise ValueError("Unsupported action")
                results.append(result)
            except Exception as error:
                _json(directory / "failure.json", {"status": "FAILED", "action": args.action,
                    "error_type": type(error).__name__, "message": str(error), "run_contract": contract,
                    "memory": memory(device, config, oom=isinstance(error, torch.cuda.OutOfMemoryError))})
                raise
            finally:
                del model, optimizer; gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
        return {"status": "IN_PROGRESS" if any(r.get("status") == "IN_PROGRESS" for r in results) else "PASS",
                "action": args.action, "branches": len(results)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--v1-root", type=Path, required=True)
    p.add_argument("--v2-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--action", choices=("prepare", "smoke", "train", "evaluate"), required=True)
    p.add_argument("--branch", choices=(*BRANCHES, "all"), default="all")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--limit-images", type=int)
    p.add_argument("--stop-after-epoch", type=int)
    return p


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2), flush=True)
