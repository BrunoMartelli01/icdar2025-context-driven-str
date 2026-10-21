"""Resumable, checksum-bound Text+Kernel+Gap and matched-control TRAIN study."""

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.adaspot_gap import gap_instances
from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import BRANCHES, AdaSpotGapAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_word_research import backend_state, load_source_state
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_letterbox_word_f1_separator_head import _validation_reference, selection_key
from scripts.train_totaltext_v4_highres_ablation import _append_epoch, _epoch_log, _json, _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config, _transform
from segmentation.adaspot_gap import AdaSpotGapDataset, AdaSpotGapTrainingConfig, train_gap_epoch
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.train import EpochShuffleSampler
from segmentation.training import load_checkpoint, save_checkpoint, seed_everything
from segmentation.word_multitask_evaluation import aggregate_word_metrics
from segmentation.word_multitask_provenance import json_contract


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/adaspot_gap_v1.json"
METRIC_PROTOCOL = {
    "geometry": "native_bbox_maximum_cardinality_one_to_one_IoU_0.50",
    "ignore": "exclude_wholly_invalid_instances_after_GT_blind_decoding;_retain_entire_partially_invalid_boxes",
    "checkpoint_selection": "word_F1_then_fewer_merged_words_then_recall_then_macro_foreground_IoU_then_earliest;_include_epoch0",
    "gap_ablation": "decode_identical_three_head_probabilities_with_gap_on_and_off",
    "test_usage": "none;_reused_TRAIN_1130_125_holdout;_exploratory",
    "ocr": "qualified_PARSeq_AR_refine1;_original_RGB_bbox_IoU_0.50_and_exact_Unicode_casefold;_punctuation_retained",
}
SCIENTIFIC_FILES = (
    "models/adaspot_gap.py", "segmentation/adaspot_gap.py", "localize/adaspot_gap.py",
    "scripts/run_adaspot_gap.py", "models/unet_adv.py", "models/totaltext_word_kernel.py",
    "models/unet_word_research.py", "segmentation/data.py", "segmentation/transforms.py",
    "segmentation/training.py", "segmentation/train.py", "segmentation/evaluate.py",
    "segmentation/word_multitask.py", "segmentation/totaltext_word_kernel.py",
    "segmentation/totaltext_image_frequency_augment.py", "localize/word_multitask.py",
    "localize/crop_recovery.py", "scripts/run_unet_word_research.py", "scripts/run_word_multitask.py",
    "scripts/train_totaltext_v4_image_frequency_ablation.py", "scripts/train_totaltext_v4_highres_ablation.py",
    "scripts/train_totaltext_letterbox_word_f1_separator_head.py",
    "segmentation/word_multitask_evaluation.py", "segmentation/word_multitask_provenance.py",
    "recognition/scene_text_v1_evaluation.py", "scripts/prune_segmentation_checkpoints.py",
)


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("AdaSpot gap requires explicit APPROXIMATE configuration without namespace fallback")
    config = json.loads(path.read_text())
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != mode
            or config.get("claim_label") != "approximate reproduction"
            or config.get("experiment_kind") != "adaspot_gap_controlled_ablation"
            or config.get("branches") != list(BRANCHES) or config.get("validation_count") != 125
            or config.get("expected_validation_words") != 983
            or config.get("input_size") != [512, 512] or config.get("native_mask_size") != [256, 256]
            or config.get("precision") != "float32" or config.get("allow_tf32") is not False
            or config.get("deterministic_algorithms") is not True):
        raise ValueError("Unsupported AdaSpot gap scientific configuration contract")
    source_path = (REPO / config["source_config"]).resolve()
    if REPO / "configs/approximate" not in source_path.parents:
        raise ValueError("Preprocessing may not fall back across configuration namespaces")
    if sha256_file(source_path) != config["source_config_sha256"]:
        raise ValueError("Qualified source preprocessing checksum changed")
    source = source_config()
    if config.get("metric_protocol") != METRIC_PROTOCOL:
        raise ValueError("Metric protocol differs from the implemented fixed diagnostic")
    for key, minimum in (("epochs", 1), ("batch_size", 1), ("num_workers", 0),
                         ("loader_timeout_seconds", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError("Invalid explicit training parameter: " + key)
    target = config["targets"]
    if (type(target["gap_radius"]) is not int or target["gap_radius"] < 1
            or not 0 < target["distance_fraction"] < 1 or not 0 <= config["skip_floor"] < 1
            or not np.isfinite(config["initial_gap_bias"]) or config["initial_gap_bias"] >= 0):
        raise ValueError("Invalid gap target or initialization")
    loss = config["loss"]
    if (any(not np.isfinite(v) or v < 0 for v in loss.values())
            or min(loss["lambda_text"], loss["lambda_kernel"], loss["lambda_gap"], loss["smooth"]) <= 0
            or loss["bce_weight"] + loss["dice_weight"] <= 0 or not 1 <= loss["separation_weight"] <= 20):
        raise ValueError("Invalid gap loss weights")
    post = config["postprocessing"]
    if post["padding"] != 0 or post["min_area"] != 0 or post["max_instances"] is not None:
        raise ValueError("This diagnostic requires declared uncapped, unpadded instances")
    gap_instances(np.zeros((2, 4, 4), np.float32), np.zeros((4, 4), np.float32),
                  np.ones((4, 4), bool), post)
    choices = config.get("scientific_choices", [])
    if not choices or any(c.get("classification") not in (
            "PAPER_SPECIFIED", "REPOSITORY_SPECIFIED", "ASSUMPTION", "HARDWARE_DEVIATION", "UNRESOLVED")
            or not c.get("rationale") or not c.get("expected_effect") for c in choices):
        raise ValueError("Every scientific choice requires a classification and rationale")
    training_config(config)
    if not np.isfinite(config["gap_learning_rate"]) or config["gap_learning_rate"] <= 0:
        raise ValueError("Gap-head learning rate must be finite and positive")
    return config, source


def training_config(config):
    return AdaSpotGapTrainingConfig(epochs=config["epochs"], seed=config["seed"],
        adam_betas=tuple(config["adam_betas"]), adam_epsilon=config["adam_epsilon"],
        learning_rate_schedule="none", checkpoint_selection="validation_word_f1",
        deterministic_algorithms=config["deterministic_algorithms"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        batch_size=config["batch_size"], separation_weight=config["loss"]["separation_weight"])


def build_optimizer(model, config):
    common = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("gap.")]
    groups = [{"params": common, "lr": config["learning_rate"], "name": "qualified_decoder_and_heads"}]
    gap = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("gap.")]
    if gap:
        groups.append({"params": gap, "lr": config["gap_learning_rate"], "name": "new_gap_head"})
    return torch.optim.Adam(groups, weight_decay=config["weight_decay"],
                            betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"])


def memory(device, config, *, oom=False):
    return {"device": str(device), "dtype": "float32", "batch_size": config["batch_size"],
        "input_shape": [config["batch_size"], 3, *config["input_size"]],
        "native_mask_shape": [config["batch_size"], 1, *config["native_mask_size"]],
        "model_set": "one ResNet50 attention U-Net with text/kernel and optional gap; no OCR resident",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "oom": oom,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def _add(totals, values):
    for key, value in values.items():
        totals[key] = totals.get(key, 0) + value


def evaluate(model, loader, references, config, device, *, export_dir=None):
    model.eval()
    totals, off_totals, rows, ious, f1s = {}, {}, [], [], []
    with torch.inference_mode():
        for batch in loader:
            evidence = model.forward_evidence(batch["image"].to(device))
            words = evidence["words"].sigmoid().cpu().numpy()
            gaps = evidence["gap"].sigmoid().cpu().numpy()[:, 0] if evidence["gap"] is not None else None
            for index, sample_id in enumerate(batch["sample_id"]):
                owners, expected, canvas = references[sample_id]
                labels, audit, foreground = gap_instances(words[index], None if gaps is None else gaps[index],
                    canvas, config["postprocessing"], use_gap=gaps is not None)
                target, valid = [batch[key][index, 0].numpy().astype(bool) for key in ("mask", "validity")]
                reference = owners.copy(); reference[~target | ~valid] = 0
                def care_geometry(predicted):
                    ignored = [int(v) for v in np.unique(predicted[predicted > 0])
                               if not np.any((predicted == v) & valid)]
                    care = predicted.copy(); care[np.isin(care, ignored)] = 0
                    return instance_geometry(care, reference, expected), ignored
                metrics, ignored = care_geometry(labels)
                _add(totals, metrics)
                off_labels = gap_instances(words[index], None if gaps is None else gaps[index],
                    canvas, config["postprocessing"], use_gap=False)[0]
                off_metrics, off_ignored = care_geometry(off_labels)
                _add(off_totals, off_metrics)
                tp, fp, fn = [int(np.count_nonzero(mask & valid)) for mask in
                              (foreground & target, foreground & ~target, ~foreground & target)]
                ious.append(tp / (tp + fp + fn) if tp + fp + fn else 1.)
                f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.)
                row = {"sample_id": sample_id, **metrics, "audit": audit, "ignored_instance_ids": ignored,
                       "gap_disabled_geometry": off_metrics,
                       "gap_disabled_ignored_instance_ids": off_ignored}
                if export_dir is not None:
                    directory = Path(export_dir) / sample_id.replace(":", "__")
                    directory.mkdir(parents=True, exist_ok=True)
                    arrays = {"probabilities": words[index], "labels": labels, "gap_disabled_labels": off_labels,
                        "gap_probability": gaps[index] if gaps is not None else np.zeros_like(target, dtype=np.float32),
                        "text_target": target, "kernel_target": batch["kernel"][index, 0].numpy(),
                        "gap_target": batch["gap_target"][index, 0].numpy(), "validity": valid,
                        "owners": reference, "canvas": canvas, "expected_owner_ids": np.asarray(expected, np.int32)}
                    path = directory / "maps.npz"
                    if path.exists():
                        with np.load(path, allow_pickle=False) as saved:
                            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                                raise ValueError("Existing frozen maps differ: " + sample_id)
                    else:
                        np.savez_compressed(path, **arrays)
                    with Image.open(batch["image_path"][index]) as image:
                        image_wh = image.size
                    proposal_path = directory / "proposals.json"
                    payload = {"sample_id": sample_id, "image_path": batch["image_path"][index],
                        "image_sha256": sha256_file(Path(batch["image_path"][index])), "maps_sha256": sha256_file(path),
                        "proposals": parseq_word_proposals(labels, image_wh, padding=0, min_area=0, max_instances=None),
                        "gap_disabled_proposals": parseq_word_proposals(off_labels, image_wh, padding=0, min_area=0, max_instances=None),
                        "result_class": "approximate reproduction", "reproduction_mode": "approximate"}
                    if proposal_path.exists() and json.loads(proposal_path.read_text()) != payload:
                        raise ValueError("Existing frozen proposal receipt differs: " + sample_id)
                    _json(proposal_path, payload)
                    row.update(maps_sha256=sha256_file(path), proposals_sha256=sha256_file(proposal_path))
                rows.append(row)
    expected_count = len(loader.dataset) if hasattr(loader, "dataset") else len(references)
    if not rows or len(rows) != expected_count or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete/duplicate AdaSpot validation population")
    aggregate = aggregate_word_metrics(totals)
    aggregate.update(foreground_iou=float(np.mean(ious)), foreground_f1_macro=float(np.mean(f1s)))
    return {"aggregate": aggregate, "per_image": rows,
            "gap_disabled_same_maps": aggregate_word_metrics(off_totals)}


def train_branch(directory, model, optimizer, train_loader, val_loader, references, transform, sampler,
                 config, contract, device, stop_after_epoch=None):
    """Atomic best/latest checkpoints, exact resume and epoch0 included in selection."""
    directory = Path(directory)
    tc = training_config(config)
    last, best = directory / "last.pt", directory / "best.pt"
    history = _epoch_log(directory / "epochs.jsonl")
    step, epoch, best_epoch, best_metric = 0, 0, None, None
    def save(path, e, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=e, global_step=step,
            config=tc, resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract,
            train_metrics=trained, validation_metrics=metrics)
    if best.exists():
        saved = load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc)
        best_epoch, best_metric = saved["epoch"], saved["validation_metrics"]
        del saved
    if last.exists():
        saved = load_checkpoint(last, model=model, optimizer=optimizer, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc, restore_rng=True)
        step, epoch = saved["global_step"], saved["epoch"]
        if epoch and len(history) == epoch - 1:
            recovered = {"epoch": epoch, "global_step": step, "train": saved["train_metrics"],
                "validation": saved["validation_metrics"], "last_checkpoint_sha256": sha256_file(last),
                "recovered_from_checkpoint": True}
            _append_epoch(directory / "epochs.jsonl", recovered); history.append(recovered)
        if len(history) != epoch or (history and history[-1]["last_checkpoint_sha256"] != sha256_file(last)):
            raise ValueError("AdaSpot checkpoint/history mismatch")
        if best_metric is None or selection_key(saved["validation_metrics"], epoch) > selection_key(best_metric, best_epoch):
            best_epoch, best_metric = epoch, saved["validation_metrics"]
            save(best, epoch, saved["train_metrics"], best_metric)
        del saved
    elif history or best.exists():
        raise ValueError("Training artifacts lack a resumable latest checkpoint")
    else:
        initial = evaluate(model, val_loader, references, config, device)["aggregate"]
        if initial["gt_words"] != config["expected_validation_words"]:
            raise ValueError("Initial validation denominator changed")
        best_epoch, best_metric = 0, initial
        save(last, 0, {}, initial); save(best, 0, {}, initial)
        _json(directory / "epoch0.json", initial)
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Noncontiguous AdaSpot epoch history")
    budget = config["epochs"] if stop_after_epoch is None else stop_after_epoch
    if type(budget) is not int or not 1 <= budget <= config["epochs"] or epoch > config["epochs"]:
        raise ValueError("Stop epoch must be inside the declared training budget")
    for e in range(epoch + 1, budget + 1):
        started = time.monotonic()
        sampler.set_epoch(e - 1); transform.set_epoch(e - 1)
        trained = train_gap_epoch(model, train_loader, optimizer, device, config)
        metrics = evaluate(model, val_loader, references, config, device)["aggregate"]
        if metrics["gt_words"] != config["expected_validation_words"]:
            raise ValueError("Validation denominator changed")
        step += trained["steps"]
        digest = save(last, e, trained, metrics)
        if selection_key(metrics, e) > selection_key(best_metric, best_epoch):
            best_epoch, best_metric = e, metrics
            save(best, e, trained, metrics)
        row = {"epoch": e, "global_step": step, "train": trained, "validation": metrics,
               "last_checkpoint_sha256": digest, "epoch_seconds": time.monotonic() - started,
               "memory": memory(device, config)}
        _append_epoch(directory / "epochs.jsonl", row); history.append(row)
        print("[{}] epoch={}/{} loss={:.5f} wordF1={:.5f} best={} seconds={:.1f}".format(
            model.branch, e, config["epochs"], trained["loss"], metrics["proposal_f1_iou_50"],
            best_epoch, row["epoch_seconds"]), flush=True)
        _json(directory / "progress.json", {"status": "RUNNING", "completed_epochs": e,
            "global_step": step, "last_epoch": row, "result_class": "approximate reproduction"})
    complete = len(history) == config["epochs"]
    _json(directory / "progress.json", {"status": "SUCCESS" if complete else "IN_PROGRESS",
        "completed_epochs": len(history), "global_step": step,
        "last_epoch": history[-1] if history else None, "result_class": "approximate reproduction"})
    return {"status": "SUCCESS" if complete else "IN_PROGRESS", "training_complete": complete,
        "completed_epochs": len(history), "run_contract": contract, "result_class": "approximate reproduction",
        "checkpoint": {"epoch": best_epoch, "path": "best.pt", "sha256": sha256_file(best)},
        "validation_word_geometry": best_metric, "memory": memory(device, config),
        "environment": experiment_environment(device)}


def run(args):
    config, source = read_config(args.config, args.mode)
    root, output = args.word_root.resolve(), args.output_root.resolve()
    if output == REPO or REPO in output.parents or root == output or root in output.parents:
        raise ValueError("Generated outputs must be outside the repository and input dataset")
    device = torch.device(args.device)
    if args.action == "train" and device.type != "cuda":
        raise RuntimeError("Full AdaSpot training requires CUDA; there is no CPU fallback")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; training cannot start on GPU")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["word_manifest_sha256"]:
        raise ValueError("Qualified word-target manifest checksum changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train" or len(manifest.samples) != 1255:
        raise ValueError("Only the qualified complete Total-Text TRAIN population is accepted")
    train_ids, val_ids, split_sha = _split_indices(manifest, config["seed"], config["validation_count"])
    if split_sha != config["validation_ids_sha256"]:
        raise ValueError("Frozen TRAIN holdout checksum changed")
    transforms = [_transform(source, "letterbox", train=value) for value in (True, False)]
    datasets = [AdaSpotGapDataset(SegmentationDataset(manifest, root, transform=t), **config["targets"])
                for t in transforms]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "preflight":
            rows = []
            for index in range(len(datasets[1])):
                batch = datasets[1][index]
                rows.append({"sample_id": batch["sample_id"], "expected_words": batch["expected_word_count"],
                    "native_words": int(len(np.unique(batch["owners_native"][batch["owners_native"] > 0]))),
                    "gap_pixels": int(batch["gap_target"].sum()),
                    "gap_over_text": int((batch["gap_target"] * batch["mask"]).sum())})
                if (index + 1) % 100 == 0:
                    print("[gap targets] {}/{}".format(index + 1, len(datasets[1])), flush=True)
            report = {"status": "PASS", "result_class": "smoke-test result", "reproduction_mode": "approximate",
                "images": len(rows), "resolved_config": config, "validation_ids_sha256": split_sha,
                "per_image": rows, "expected_words": sum(r["expected_words"] for r in rows),
                "native_words": sum(r["native_words"] for r in rows),
                "runtime_inputs": {"word_root": str(root)},
                "source_sha256": {f: sha256_file(REPO / f) for f in SCIENTIFIC_FILES}}
            validation_rows = [rows[index] for index in val_ids]
            if (sum(r["expected_words"] for r in validation_rows) != config["expected_validation_words"]
                    or any(r["native_words"] != r["expected_words"] for r in validation_rows)):
                raise ValueError("Validation target word population changed before GPU execution")
            report["validation_expected_words"] = config["expected_validation_words"]
            report["native_word_losses"] = report["expected_words"] - report["native_words"]
            existing = output / "target_qualification.json"
            if existing.exists() and json.loads(existing.read_text()) != report:
                raise ValueError("Target qualification changed; preserve outputs and select a new directory")
            _json(output / "target_qualification.json", report)
            return {"status": "PASS", "images": len(rows), "expected_words": report["expected_words"],
                    "native_words": report["native_words"]}
        if args.kernel_run is None:
            raise ValueError("A qualified --kernel-run is required")
        kernel_run = args.kernel_run.resolve()
        if kernel_run == output or kernel_run in output.parents:
            raise ValueError("Output may not overwrite qualified source checkpoints")
        state = load_source_state(kernel_run, config)
        sampler = EpochShuffleSampler(len(train_ids), seed=config["seed"])
        options = {"num_workers": config["num_workers"], "pin_memory": device.type == "cuda",
                   "timeout": config["loader_timeout_seconds"] if config["num_workers"] else 0}
        train_loader = DataLoader(Subset(datasets[0], train_ids), batch_size=config["batch_size"], sampler=sampler, **options)
        val_loader = DataLoader(Subset(datasets[1], val_ids), batch_size=config["batch_size"], shuffle=False, **options)
        references = _validation_reference(manifest, root, val_ids, transforms[1])
        results = []
        for branch in config["branches"] if args.branch == "all" else [args.branch]:
            seed_everything(config["seed"], deterministic_algorithms=True)
            model = AdaSpotGapAGUNet(branch, skip_floor=config["skip_floor"], initial_gap_bias=config["initial_gap_bias"])
            initialization = model.initialize_kernel_state(state)
            contract = json_contract({"schema_version": 1, "branch": branch, "resolved_config": config,
                "resolved_preprocessing": source, "architecture": model.architecture_identity,
                "validation_ids_sha256": split_sha, "initialization": initialization,
                "active_loss_components": ["text", "kernel"] + (["gap"] if branch == "gap" else []),
                "runtime_inputs": {"word_root": str(root), "kernel_run": str(kernel_run)},
                "execution_device": str(device), "backend": backend_state(),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "training_config": asdict(training_config(config)),
                "source_sha256": {f: sha256_file(REPO / f) for f in SCIENTIFIC_FILES}})
            directory = output / branch; directory.mkdir(exist_ok=True)
            path = directory / "run_contract.json"
            if path.exists():
                if json.loads(path.read_text()) != contract:
                    raise ValueError("Run contract changed; preserve outputs and use a new directory")
            elif any(directory.iterdir()):
                raise ValueError("Nonempty branch lacks its immutable run contract")
            else:
                _json(path, contract)
            optimizer = None
            try:
                seed_everything(config["seed"], deterministic_algorithms=True)
                if device.type == "cuda":
                    _prepare_cuda_memory_tracking(device)
                model.to(device)
                optimizer = build_optimizer(model, config)
                if args.action == "smoke":
                    batch = next(iter(train_loader))
                    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                              if n.startswith(("head.", "gap."))}
                    trained = train_gap_epoch(model, [batch], optimizer, device, config)
                    updates = {n: float((p.detach().cpu() - before[n]).abs().max())
                               for n, p in model.named_parameters() if n in before}
                    if trained["steps"] != 1 or (branch == "gap" and not updates["gap.weight"] > 0):
                        raise RuntimeError("Smoke did not optimize the requested gap head")
                    result = {"status": "PASS", "result_class": "smoke-test result", "run_contract": contract,
                        "train": trained, "head_update_linf": updates, "memory": memory(device, config),
                        "environment": experiment_environment(device)}
                    _json(directory / "smoke.json", result)
                elif args.action == "train":
                    receipt = directory / "run_manifest.json"
                    previous = json.loads(receipt.read_text()) if receipt.exists() else None
                    if previous is not None and previous.get("status") == "SUCCESS":
                        if (previous.get("run_contract") != contract or not previous.get("training_complete")
                                or previous.get("completed_epochs") != config["epochs"]
                                or previous["checkpoint"]["sha256"] != sha256_file(directory / "best.pt")):
                            raise ValueError("Completed run provenance changed")
                        result = previous
                    else:
                        result = train_branch(directory, model, optimizer, train_loader, val_loader, references,
                            transforms[0], sampler, config, contract, device, args.stop_after_epoch)
                        _json(receipt, result)
                else:
                    receipt = json.loads((directory / "run_manifest.json").read_text())
                    if (receipt.get("status") != "SUCCESS" or not receipt.get("training_complete")
                            or receipt.get("run_contract") != contract or receipt.get("completed_epochs") != config["epochs"]
                            or receipt["checkpoint"]["sha256"] != sha256_file(directory / "best.pt")):
                        raise ValueError("Evaluation requires a completed, unchanged training run")
                    load_checkpoint(directory / "best.pt", model=model, expected_architecture=model.architecture_identity,
                                    expected_run_contract=contract, expected_training_config=training_config(config))
                    result = evaluate(model, val_loader, references, config, device,
                                      export_dir=directory / "validation_exports")
                    result.update(run_contract=contract, checkpoint_sha256=receipt["checkpoint"]["sha256"],
                                  result_class="approximate reproduction", memory=memory(device, config))
                    _json(directory / "validation.json", result)
                results.append(result)
            except Exception as error:
                _json(directory / "failure.json", {"status": "FAILED", "action": args.action,
                    "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                    "run_contract": contract, "memory": memory(device, config, oom=isinstance(error, torch.cuda.OutOfMemoryError))})
                raise
            finally:
                del optimizer, model; gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print("[{}] {} completed".format(branch, args.action), flush=True)
        return {"status": "IN_PROGRESS" if any(r.get("status") == "IN_PROGRESS" for r in results) else "PASS",
                "action": args.action, "branches": len(results)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--kernel-run", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--action", choices=("preflight", "smoke", "train", "evaluate"), required=True)
    p.add_argument("--branch", choices=(*BRANCHES, "all"), default="all")
    p.add_argument("--stop-after-epoch", type=int)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2), flush=True)
