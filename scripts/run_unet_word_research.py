"""Matched, resumable APPROXIMATE glyph/word and auxiliary-head experiments."""

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.unet_word_research import limit_instances, research_instances
from localize.word_multitask import parseq_word_proposals, word_instances
from models.unet_word_research import BRANCHES, WordResearchAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _append_epoch, _epoch_log, _json, _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config, _transform
from scripts.train_totaltext_letterbox_word_f1_separator_head import _validation_reference, selection_key
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.train import EpochShuffleSampler
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything
from segmentation.unet_word_research import GlyphWordDataset, WordResearchTrainingConfig, mask_quality_targets, train_research_epoch
from segmentation.word_multitask_evaluation import aggregate_word_metrics
from segmentation.word_multitask_provenance import json_contract


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/unet_word_research_v1.json"


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("This experimental suite requires explicit APPROXIMATE configuration")
    config = json.loads(path.read_text())
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != mode
            or config.get("claim_label") != "approximate reproduction" or config.get("branches") != list(BRANCHES)
            or config.get("input_size") != [512, 512] or config.get("native_mask_size") != [256, 256]
            or config.get("validation_count") != 125 or config.get("expected_validation_words") != 983):
        raise ValueError("Unsupported word-research configuration contract")
    source = source_config()
    if sha256_file(REPO / config["source_config"]) != config["source_config_sha256"]:
        raise ValueError("Source preprocessing profile changed")
    if (type(config["epochs"]) is not int or config["epochs"] < 1 or type(config["batch_size"]) is not int
            or config["batch_size"] < 1 or type(config["num_workers"]) is not int or config["num_workers"] < 0
            or not 0 < config["targets"]["distance_fraction"] < 1
            or not 0 <= config["skip_floor"] < 1):
        raise ValueError("Invalid training or target parameters")
    loss = config["loss"]
    if (any(not np.isfinite(v) or v < 0 for v in loss.values())
            or min(loss["lambda_word"], loss["lambda_kernel"], loss["smooth"]) <= 0
            or loss["bce_weight"] + loss["dice_weight"] <= 0 or loss["separation_weight"] < 1):
        raise ValueError("Invalid auxiliary loss weights")
    embedding, adaptive, glyph = [config[k] for k in ("embedding", "adaptive", "glyph_postprocessing")]
    if (type(embedding["dimensions"]) is not int or embedding["dimensions"] < 1
            or not 0 < embedding["pull_margin"] < embedding["push_margin"]
            or not 0 < adaptive["threshold_range"][0] < adaptive["threshold_range"][1] < 1
            or type(adaptive["border_width"]) is not int or adaptive["border_width"] < 1
            or not np.isfinite(adaptive["sigmoid_steepness"]) or adaptive["sigmoid_steepness"] <= 0
            or not 0 < glyph["threshold"] < 1 or not 0.5 < glyph["minimum_overlap"] <= 1):
        raise ValueError("Invalid embedding/adaptive/glyph settings")
    post = config["postprocessing"]
    word_instances(np.zeros((2, 8, 8), np.float32), np.ones((8, 8), bool), post)
    if (post["padding"] != 0 or post["min_area"] != 0 or post["max_instances"] is not None
            or not np.isfinite(post["embedding_distance"]) or post["embedding_distance"] <= 0):
        raise ValueError("Diagnostic localization must use declared unfiltered word instances")
    if (set(config["quality"]) != {"context_pixels", "pool_size"}
            or type(config["quality"]["context_pixels"]) is not int or config["quality"]["context_pixels"] < 0
            or type(config["quality"]["pool_size"]) is not int or config["quality"]["pool_size"] < 1
            or type(config["quality_budget"]) is not int or config["quality_budget"] < 1):
        raise ValueError("Invalid quality crop/budget settings")
    if (not config["stability_thresholds"] or any(not np.isfinite(t) or not 0 < t < 1 for t in config["stability_thresholds"])):
        raise ValueError("Invalid stability thresholds")
    if not 0 < config["stability_bad_bbox_iou"] <= 1 or not 0 < config["stability_top_fraction"] <= 1:
        raise ValueError("Invalid stability diagnostic protocol")
    training_config(config, "control")
    return config, source


def training_config(config, branch):
    return WordResearchTrainingConfig(epochs=config["epochs"], seed=config["seed"],
        adam_betas=(0.9, 0.999), adam_epsilon=1e-8, learning_rate_schedule="none",
        checkpoint_selection="validation_quality_loss" if branch == "quality" else "validation_word_f1",
        deterministic_algorithms=True, learning_rate=config["learning_rate"], batch_size=config["batch_size"],
        separation_weight=config["loss"]["separation_weight"])


def memory(device, config, *, oom=False):
    return {"device": str(device), "dtype": "float32", "batch_size": config["batch_size"],
            "input_shape": [config["batch_size"], 3, *config["input_size"]],
            "model_set": "one shared-decoder AG-U-Net; sequential branches; no recognizer resident",
            "oom": oom, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def backend_state():
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cpu_threads": torch.get_num_threads()}


def load_source_state(directory, config):
    checkpoint, receipt = directory / "best.pt", directory / "run_manifest.json"
    if (sha256_file(checkpoint) != config["source_checkpoint_sha256"]
            or sha256_file(receipt) != config["source_run_manifest_sha256"]):
        raise ValueError("Qualified Text + Kernel checkpoint/report checksum mismatch")
    report = json.loads(receipt.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (report.get("status") != "SUCCESS" or payload.get("architecture") != config["source_architecture"]
            or payload.get("reproduction_mode") != "approximate" or payload.get("claim_label") != "approximate reproduction"
            or report["checkpoint"]["sha256"] != config["source_checkpoint_sha256"]
            or payload.get("epoch") != report["checkpoint"]["epoch"]
            or payload.get("run_contract") != report.get("run_contract")
            or json_contract(payload.get("training_config")) != report["run_contract"]["training_config"]):
        raise ValueError("Qualified source checkpoint scientific provenance mismatch")
    return payload["model_state_dict"]


def _add(totals, values):
    for key, value in values.items():
        totals[key] = totals.get(key, 0) + value


def evaluate(model, loader, references, config, device, *, export_dir=None):
    model.eval()
    totals, rows, pixel_iou, pixel_f1, quality_losses = {}, [], [], [], []
    budgets = {"area": {}, "quality": {}} if model.branch == "quality" else {}
    word_only = {} if model.branch == "glyph_word" else None
    expected_samples = 0
    with torch.inference_mode():
        for batch in loader:
            expected_samples += len(batch["image"])
            local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            evidence = model.forward_evidence(local["image"])
            predicted = [research_instances(evidence, references[s][2], config, i) for i, s in enumerate(batch["sample_id"])]
            for i, sample_id in enumerate(batch["sample_id"]):
                labels, audit = predicted[i]
                owners, expected, canvas = references[sample_id]
                valid = batch["validity"][i, 0].numpy().astype(bool)
                target = batch["mask"][i, 0].numpy().astype(bool)
                reference = owners.copy(); reference[~target | ~valid] = 0
                care = labels.copy()
                ignored = [int(v) for v in np.unique(labels[labels > 0]) if not np.any((labels == v) & valid)]
                care[np.isin(care, ignored)] = 0
                metrics = instance_geometry(care, reference, expected)
                _add(totals, metrics)
                foreground = labels > 0
                tp, fp, fn = [np.count_nonzero(mask & valid) for mask in
                    (foreground & target, foreground & ~target, ~foreground & target)]
                pixel_iou.append(tp / (tp + fp + fn) if tp + fp + fn else 1.)
                pixel_f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.)
                row = {"sample_id": sample_id, **metrics, "audit": audit, "ignored_instance_ids": ignored}
                if word_only is not None:
                    base = word_instances(evidence["words"][i].sigmoid().cpu().numpy(), canvas, config["postprocessing"])[0]
                    for instance in np.unique(base[base > 0]):
                        if not np.any((base == instance) & valid): base[base == instance] = 0
                    _add(word_only, instance_geometry(base, reference, expected))
                quality_scores = {}
                if budgets:
                    ids, scores = model.quality_scores(evidence["features"][i], labels, **config["quality"])
                    values = scores.sigmoid().cpu().numpy()
                    target_ids, target_values = mask_quality_targets(labels, reference, valid)
                    if target_ids:
                        positions = [ids.index(instance) for instance in target_ids]
                        quality_losses.append((float(np.mean((values[positions] - target_values) ** 2)), 1))
                    quality_scores = {instance: float(np.prod(value)) for instance, value in zip(ids, values)}
                    row["quality_predictions"] = [{"instance_id": instance, "bbox_iou_coverage_purity": value.tolist(),
                        "rank_score": quality_scores[instance]} for instance, value in zip(ids, values)]
                    for rank in budgets:
                        capped = limit_instances(labels, config["quality_budget"], quality_scores if rank == "quality" else None)
                        for instance in np.unique(capped[capped > 0]):
                            if not np.any((capped == instance) & valid): capped[capped == instance] = 0
                        metric = instance_geometry(capped, reference, expected)
                        _add(budgets[rank], metric)
                        row[rank + "_budget_geometry"] = metric
                if export_dir is not None:
                    directory = Path(export_dir) / sample_id.replace(":", "__")
                    directory.mkdir(parents=True, exist_ok=True)
                    arrays = {"probabilities": evidence["words"][i].sigmoid().cpu().numpy(), "labels": labels,
                        "owners": reference, "validity": valid, "canvas": canvas, "text_target": target,
                        "glyph_target": batch["glyph_mask"][i, 0].numpy(), "expected_owner_ids": np.asarray(expected, np.int32)}
                    for key in ("glyph", "embedding", "threshold"):
                        if evidence[key] is not None:
                            arrays[key] = evidence[key][i].cpu().numpy()
                    path = directory / "maps.npz"
                    if path.exists():
                        with np.load(path, allow_pickle=False) as saved:
                            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                                raise ValueError("Existing prediction maps differ: " + sample_id)
                    else:
                        np.savez_compressed(path, **arrays)
                    with Image.open(batch["image_path"][i]) as image:
                        image_wh = image.size
                    proposals = parseq_word_proposals(labels, image_wh, padding=0, min_area=0, max_instances=None)
                    _json(directory / "proposals.json", {"sample_id": sample_id, "image_path": batch["image_path"][i],
                        "image_sha256": sha256_file(Path(batch["image_path"][i])), "maps_sha256": sha256_file(path),
                        "proposals": proposals, "result_class": "approximate reproduction", "reproduction_mode": "approximate",
                        "quality_scores": quality_scores})
                    row["maps_sha256"] = sha256_file(path)
                    row["proposals_sha256"] = sha256_file(directory / "proposals.json")
                rows.append(row)
    expected = len(loader.dataset) if hasattr(loader, "dataset") else expected_samples
    if len(rows) != expected or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete/duplicate validation samples")
    aggregate = aggregate_word_metrics(totals)
    aggregate.update(foreground_iou=float(np.mean(pixel_iou)), foreground_f1_macro=float(np.mean(pixel_f1)))
    if quality_losses:
        aggregate["quality_loss"] = sum(v * n for v, n in quality_losses) / sum(n for _, n in quality_losses)
    elif model.branch == "quality":
        raise ValueError("Validation has no supervised quality proposals")
    return {"aggregate": aggregate, "per_image": rows,
            "glyph_word_head_only": aggregate_word_metrics(word_only) if word_only is not None else None,
            "budget_comparison": {k: aggregate_word_metrics(v) for k, v in budgets.items()}}


def checkpoint_key(metrics, epoch, branch):
    return (-metrics["quality_loss"], -epoch) if branch == "quality" else selection_key(metrics, epoch)


def train_branch(directory, model, optimizer, train_loader, val_loader, references, transform, sampler,
                 config, contract, device, stop_after_epoch=None):
    tc = training_config(config, model.branch)
    last, best = directory / "last.pt", directory / "best.pt"
    history = _epoch_log(directory / "epochs.jsonl")
    step, epoch, best_epoch, best_metric = 0, 0, None, None
    def save(path, e, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=e, global_step=step, config=tc,
            resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract, train_metrics=trained, validation_metrics=metrics)
    if best.exists():
        saved = load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc)
        best_epoch, best_metric = saved["epoch"], saved["validation_metrics"]
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
            raise ValueError("Checkpoint/history mismatch")
        if best_metric is None or checkpoint_key(saved["validation_metrics"], epoch, model.branch) > checkpoint_key(best_metric, best_epoch, model.branch):
            best_epoch, best_metric = epoch, saved["validation_metrics"]
            save(best, epoch, saved["train_metrics"], best_metric)
        del saved
    elif history or best.exists():
        raise ValueError("Training files lack a resumable latest checkpoint")
    else:
        initial = evaluate(model, val_loader, references, config, device)["aggregate"]
        best_epoch, best_metric = 0, initial
        save(last, 0, {}, initial); save(best, 0, {}, initial)
        _json(directory / "epoch0.json", initial)
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Noncontiguous epoch history")
    budget = config["epochs"] if stop_after_epoch is None else stop_after_epoch
    if not 1 <= budget <= config["epochs"]:
        raise ValueError("Stop epoch must be within configured training budget")
    for e in range(epoch + 1, budget + 1):
        sampler.set_epoch(e - 1); transform.set_epoch(e - 1)
        trained = train_research_epoch(model, train_loader, optimizer, device, config)
        metrics = evaluate(model, val_loader, references, config, device)["aggregate"]
        if metrics["gt_words"] != config["expected_validation_words"]:
            raise ValueError("Changed validation word denominator")
        step += trained["steps"]
        digest = save(last, e, trained, metrics)
        if checkpoint_key(metrics, e, model.branch) > checkpoint_key(best_metric, best_epoch, model.branch):
            best_epoch, best_metric = e, metrics; save(best, e, trained, metrics)
        row = {"epoch": e, "global_step": step, "train": trained, "validation": metrics, "last_checkpoint_sha256": digest}
        _append_epoch(directory / "epochs.jsonl", row); history.append(row)
        print(f"[{model.branch}] {e}/{config['epochs']} loss={trained['loss']:.5f} wordF1={metrics['proposal_f1_iou_50']:.5f} best={best_epoch}", flush=True)
    complete = len(history) == config["epochs"]
    return {"status": "SUCCESS" if complete else "IN_PROGRESS", "training_complete": complete,
        "run_contract": contract, "completed_epochs": len(history), "result_class": "approximate reproduction",
        "checkpoint": {"epoch": best_epoch, "path": "best.pt", "sha256": sha256_file(best)},
        "validation_word_geometry": best_metric, "memory": memory(device, config), "environment": experiment_environment(device)}


def run(args):
    config, source = read_config(args.config, args.mode)
    for key in ("batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            if type(value) is not int or value < (1 if key == "batch_size" else 0):
                raise ValueError("Invalid explicit runtime override: " + key)
            config[key] = value
            config["scientific_choices"].append({"component": key + "_override", "classification": "HARDWARE_DEVIATION",
                "rationale": "Explicit command-line execution override", "expected_effect": "Batch may change optimization/BN; workers preserve indexed augmentations"})
    root, glyph_root, output = args.word_root.resolve(), args.glyph_root.resolve(), args.output_root.resolve()
    if any(output == p or p in output.parents for p in (root, glyph_root)) or output == REPO:
        raise ValueError("Output must be isolated from source datasets")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use CPU preflight/smoke or the qualified GPU host")
    manifests = [r / config["train_manifest"] for r in (root, glyph_root)]
    for path, expected in zip(manifests, (config["word_manifest_sha256"], config["glyph_manifest_sha256"])):
        if sha256_file(path) != expected: raise ValueError("Qualified target manifest changed: " + str(path))
    word, glyph = [load_segmentation_manifest(p, require_checksums=True) for p in manifests]
    if any(m.dataset != "total_text" or m.split_role != "train" for m in (word, glyph)):
        raise ValueError("Only paired Total-Text TRAIN manifests are accepted")
    train_ids, val_ids, split_sha = _split_indices(word, config["seed"], config["validation_count"])
    if split_sha != config["validation_ids_sha256"]: raise ValueError("Frozen holdout changed")
    transforms = [_transform(source, "letterbox", train=value) for value in (True, False)]
    datasets = [GlyphWordDataset(SegmentationDataset(word, root, transform=t), glyph, glyph_root,
                 **config["targets"]) for t in transforms]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "preflight":
            rows = []
            for index in range(len(datasets[1])):
                batch = datasets[1][index]
                rows.append({"sample_id": batch["sample_id"], "word_pixels": int(batch["mask"].sum()),
                    "glyph_pixels": int(batch["glyph_mask"].sum()), "expected_words": batch["expected_word_count"],
                    "words": int(len(np.unique(batch["owners_native"][batch["owners_native"] > 0])))})
                if (index + 1) % 100 == 0: print(f"[paired targets] {index + 1}/{len(datasets[1])}", flush=True)
            report = {"status": "PASS", "result_class": "smoke-test result", "reproduction_mode": "approximate",
                "images": len(rows), "resolved_config": config, "validation_ids_sha256": split_sha, "per_image": rows,
                "expected_words": sum(r["expected_words"] for r in rows), "native_words": sum(r["words"] for r in rows),
                "words_lost_at_native_resolution": sum(r["expected_words"] - r["words"] for r in rows),
                "runtime_inputs": {"word_root": str(root), "glyph_root": str(glyph_root)}}
            _json(output / "paired_target_qualification.json", report)
            return report
        if args.kernel_run is None: raise ValueError("Provide a qualified --kernel-run for real model execution")
        state = load_source_state(args.kernel_run.resolve(), config)
        sampler = EpochShuffleSampler(len(train_ids), seed=config["seed"])
        options = {"num_workers": config["num_workers"], "timeout": 60 if config["num_workers"] else 0,
                   "pin_memory": device.type == "cuda"}
        train_loader = DataLoader(Subset(datasets[0], train_ids), batch_size=config["batch_size"], sampler=sampler, **options)
        val_loader = DataLoader(Subset(datasets[1], val_ids), batch_size=config["batch_size"], shuffle=False, **options)
        references = _validation_reference(word, root, val_ids, transforms[1])
        files = ("models/unet_word_research.py", "segmentation/unet_word_research.py", "localize/unet_word_research.py",
                 "scripts/run_unet_word_research.py", "models/unet_adv.py", "models/totaltext_word_kernel.py",
                 "segmentation/data.py", "segmentation/transforms.py", "segmentation/training.py",
                 "segmentation/evaluate.py",
                 "segmentation/word_multitask.py", "segmentation/totaltext_word_kernel.py",
                 "segmentation/totaltext_image_frequency_augment.py", "localize/word_multitask.py", "localize/crop_recovery.py",
                 "scripts/train_totaltext_v4_image_frequency_ablation.py", "scripts/train_totaltext_v4_highres_ablation.py",
                 "scripts/train_totaltext_letterbox_word_f1_separator_head.py", "segmentation/word_multitask_evaluation.py",
                 "recognition/scene_text_v1_evaluation.py")
        results = []
        for branch in config["branches"] if args.branch == "all" else [args.branch]:
            seed_everything(config["seed"], deterministic_algorithms=True)
            model = WordResearchAGUNet(branch, skip_floor=config["skip_floor"], embedding_dimensions=config["embedding"]["dimensions"])
            initialization = model.initialize_kernel_state(state)
            contract = json_contract({"schema_version": 1, "branch": branch, "resolved_config": config,
                "resolved_preprocessing": source, "architecture": model.architecture_identity,
                "validation_ids_sha256": split_sha, "initialization": initialization,
                "runtime_inputs": {"word_root": str(root), "glyph_root": str(glyph_root), "kernel_run": str(args.kernel_run.resolve())},
                "execution_device": str(device), "backend": backend_state(),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "training_config": asdict(training_config(config, branch)),
                "source_sha256": {f: sha256_file(REPO / f) for f in files}})
            directory = output / branch; directory.mkdir(exist_ok=True)
            contract_path = directory / "run_contract.json"
            if contract_path.exists():
                if json.loads(contract_path.read_text()) != contract: raise ValueError("Run contract changed; use a new output directory")
            elif any(directory.iterdir()):
                raise ValueError("Nonempty branch lacks its run contract")
            else: _json(contract_path, contract)
            optimizer = None
            try:
                seed_everything(config["seed"], deterministic_algorithms=True)
                if device.type == "cuda": _prepare_cuda_memory_tracking(device)
                model.to(device)
                optimizer = build_optimizer(model, training_config(config, branch))
                if args.action == "smoke":
                    batch = next(iter(train_loader))
                    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                              if name.startswith(("head.", "glyph.", "embedding.", "threshold.", "quality."))}
                    trained = train_research_epoch(model, [batch], optimizer, device, config)
                    updates = {name: float((parameter.detach() - before[name]).abs().max())
                               for name, parameter in model.named_parameters() if name in before}
                    sample = evaluate(model, [batch], {s: (batch["owners_native"][i].numpy(),
                        tuple(int(v) for v in np.unique(batch["owners_native"][i]) if v > 0), batch["canvas"][i].numpy())
                        for i, s in enumerate(batch["sample_id"])}, config, device)
                    result = {"status": "PASS", "result_class": "smoke-test result", "run_contract": contract,
                              "train": trained, "validation": sample, "memory": memory(device, config),
                              "environment": experiment_environment(device), "head_update_linf": updates}
                    _json(directory / "smoke.json", result)
                elif args.action == "train":
                    receipt = directory / "run_manifest.json"
                    previous = json.loads(receipt.read_text()) if receipt.exists() else None
                    if previous is not None and previous.get("status") == "SUCCESS":
                        if (previous.get("run_contract") != contract or not previous.get("training_complete")
                                or previous.get("completed_epochs") != config["epochs"]
                                or previous["checkpoint"]["sha256"] != sha256_file(directory / "best.pt")):
                            raise ValueError("Completed run provenance/checkpoint changed")
                        # Preserve the measured training peaks when no training
                        # runs in this invocation; a checkpoint-only reload is
                        # not a replacement GPU-memory qualification.
                        result = previous
                    else:
                        result = train_branch(directory, model, optimizer, train_loader, val_loader, references,
                            transforms[0], sampler, config, contract, device, args.stop_after_epoch)
                        if previous is not None:
                            for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
                                before = previous.get("memory", {}).get(key)
                                now = result["memory"].get(key)
                                if before is not None and now is not None: result["memory"][key] = max(before, now)
                        _json(receipt, result)
                else:
                    result = json.loads((directory / "run_manifest.json").read_text())
                    if (result["status"] != "SUCCESS" or not result["training_complete"]
                            or result.get("run_contract") != contract or result.get("completed_epochs") != config["epochs"]):
                        raise ValueError("Evaluation requires completed training; partial runs are not results")
                    if result["checkpoint"]["sha256"] != sha256_file(directory / "best.pt"):
                        raise ValueError("Selected checkpoint checksum changed")
                    load_checkpoint(directory / "best.pt", model=model, expected_architecture=model.architecture_identity,
                        expected_run_contract=contract, expected_training_config=training_config(config, branch))
                    result = evaluate(model, val_loader, references, config, device, export_dir=directory / "validation_exports")
                    result.update(run_contract=contract, checkpoint_sha256=sha256_file(directory / "best.pt"),
                                  result_class="approximate reproduction", memory=memory(device, config))
                    _json(directory / "validation.json", result)
                results.append(result)
            except Exception as error:
                _json(directory / "failure.json", {"status": "FAILED", "error_type": type(error).__name__, "message": str(error),
                    "action": args.action, "run_contract": contract,
                    "memory": memory(device, config, oom=isinstance(error, torch.cuda.OutOfMemoryError))})
                raise
            finally:
                del optimizer, model; gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            print("[" + branch + "] " + args.action + " completed", flush=True)
        return {"status": "IN_PROGRESS" if any(r.get("status") == "IN_PROGRESS" for r in results) else "PASS",
                "action": args.action, "branches": len(results)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--glyph-root", type=Path, required=True)
    p.add_argument("--kernel-run", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--action", choices=("preflight", "smoke", "train", "evaluate"), required=True)
    p.add_argument("--branch", choices=(*BRANCHES, "all"), default="all")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--stop-after-epoch", type=int)
    return p


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "per_image" and k != "resolved_config"}, indent=2), flush=True)
