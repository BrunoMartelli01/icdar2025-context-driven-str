"""Isolated, restartable matched word-head ablation; reuses existing training I/O.

Explicit --mode approximate is mandatory. Preflight qualifies new targets;
smoke exercises a real batch; train and evaluate preserve older run directories.
"""

import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from models.word_multitask import VARIANTS, build_word_model
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices, _epoch_log, _append_epoch
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config, _transform, CONFIG_PATH as SOURCE_PATH
from scripts.train_totaltext_letterbox_word_f1_separator_head import _validation_reference, selection_key
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.train import EpochShuffleSampler
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything, runtime_environment
from segmentation.word_multitask import WordMultitaskDataset, WordMultitaskTrainingConfig, train_multitask_epoch
from segmentation.word_multitask_evaluation import evaluate_multitask
from segmentation.word_multitask_provenance import json_contract, ensure_branch_contract, reusable_smoke
from localize.word_multitask import word_instances
from dataclasses import asdict


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/word_multitask_v1.json"
METRIC_PROTOCOL = {
    "geometry": "native_bbox_maximum_cardinality_one_to_one_IoU_0.50",
    "merge_rate": "care_words_touched_by_instances_touching_at_least_two_care_words_divided_by_all_expected_care_words",
    "split_rate": "care_words_touched_by_at_least_two_instances_divided_by_all_expected_care_words",
    "touch": "any_shared_native_pixel",
    "ignore_predictions": "ignore_only_instances_with_no_valid_pixel;_keep_entire_box_of_partial_overlap",
    "empty_word_denominator": "zero",
    "pixel_empty_policy": "perfect_if_both_empty",
    "checkpoint_selection": "word_F1_then_fewer_merged_words_then_recall_then_macro_pixel_IoU_then_earliest",
    "test_usage": "none; held_out_TRAIN_only; exploratory_due_to_prior_ablation_reuse",
}


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Word multi-task experiments require --mode approximate and configs/approximate/")
    cfg = json.loads(path.read_text())
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != mode
        or cfg.get("claim_label") != "approximate reproduction" or set(cfg.get("variants", [])) != set(VARIANTS)):
        raise ValueError("Invalid multi-task configuration/mode")
    if ((REPO / cfg["source_config"]).resolve() != SOURCE_PATH or sha256_file(SOURCE_PATH) != cfg["source_config_sha256"]):
        raise ValueError("Frozen source preprocessing configuration changed")
    source = source_config()
    if (cfg["input_size"] != [512, 512] or cfg["native_mask_size"] != [256, 256]
        or cfg["seed"] != source["training"]["seed"] or cfg["batch_size"] != 8
        or cfg["validation_count"] != 125 or cfg["encoder_weights"] != "IMAGENET1K_V2"):
        raise ValueError("Matched protocol requires the qualified 512/256, seed202503, batch8, validation125 and V2 encoder")
    if (not isinstance(cfg["epochs"], int) or cfg["epochs"] < 1 or not isinstance(cfg["num_workers"], int)
        or cfg["num_workers"] < 0 or not isinstance(cfg["loader_timeout_seconds"], int)
        or cfg["loader_timeout_seconds"] < 1):
        raise ValueError("Invalid epoch/worker count")
    if not 0 <= cfg["skip_floor"] < 1:
        raise ValueError("Invalid skip floor")
    target = cfg["targets"]
    if (not np.isfinite(list(target.values())).all() or not 0 < target["distance_fraction"] < 1
        or min(target["sigma_scale"], target["min_sigma"]) <= 0):
        raise ValueError("Invalid kernel/Gaussian target configuration")
    loss = cfg["loss"]
    weights = [loss[k] for k in ("lambda_text", "lambda_kernel", "lambda_center", "bce_weight", "dice_weight")]
    if (not np.isfinite(weights + [loss["smooth"], loss["separation_weight"]]).all()
        or min(weights) < 0 or weights[0] <= 0 or sum(weights[-2:]) <= 0
        or loss["smooth"] <= 0 or not 1 <= loss["separation_weight"] <= 20
        or loss["center_loss"] not in ("mse", "balanced_mse")):
        raise ValueError("Invalid multi-task loss configuration")
    post = cfg["postprocessing"]
    word_instances(np.zeros((3, 8, 8), np.float32), np.ones((8, 8), bool), post)
    if (post["padding"] != 0 or post["min_area"] != 0 or post["max_instances"] is not None):
        raise ValueError("Matched geometry comparison currently requires pad0, no area filter, no top10")
    from copy import deepcopy
    # Metrics are implemented by the shared existing native metric. Reject
    # unsupported metadata edits rather than silently report another protocol.
    if cfg["metric_protocol"] != METRIC_PROTOCOL:
        raise ValueError("Metric protocol differs from the implemented native diagnostic")
    landscape_weights = np.asarray(post["watershed_weights"])
    if (landscape_weights.shape != (3,) or not np.isfinite(landscape_weights).all()
        or np.any(landscape_weights < 0) or landscape_weights[0] <= 0):
        raise ValueError("Three nonnegative watershed weights with a positive text weight are required")
    if not isinstance(post["center_min_distance"], int) or post["center_min_distance"] < 1:
        raise ValueError("Positive integer center NMS radius required")
    return deepcopy(cfg), source


def training_config(cfg):
    return WordMultitaskTrainingConfig(epochs=cfg["epochs"], seed=cfg["seed"], adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8, learning_rate_schedule="none", checkpoint_selection="validation_word_f1",
        deterministic_algorithms=True, separation_weight=cfg["loss"]["separation_weight"])


def run_contract(cfg, source, variant, encoder_sha, split_sha):
    modules = ["models/word_multitask.py", "models/totaltext_word_kernel.py", "models/unet_adv.py",
        "segmentation/word_multitask.py", "segmentation/word_multitask_provenance.py",
        "localize/word_multitask.py", "localize/crop_recovery.py",
        "segmentation/word_multitask_evaluation.py", "segmentation/totaltext_word_kernel.py",
        "segmentation/totaltext_image_frequency_augment.py", "segmentation/data.py",
        "segmentation/transforms.py", "segmentation/training.py", "segmentation/metrics.py",
        "scripts/run_word_multitask.py", "scripts/train_totaltext_v4_highres_ablation.py",
        "scripts/train_totaltext_v4_image_frequency_ablation.py",
        "scripts/train_totaltext_letterbox_word_f1_separator_head.py", "recognition/scene_text_v1_evaluation.py"]
    return json_contract({"schema_version": 1, "reproduction_mode": "approximate", "result_class": "approximate reproduction",
            "variant": variant, "config": cfg, "resolved_preprocessing": source,
            "encoder_sha256": encoder_sha, "validation_ids_sha256": split_sha,
            "training_config": asdict(training_config(cfg)),
            "source_sha256": {name: sha256_file(REPO / name) for name in modules}})


def qualify_targets(dataset):
    rows, native_words = [], 0
    for index in range(len(dataset)):
        batch = dataset[index]
        owners = batch["owners_native"].numpy()
        ids = np.unique(owners[owners > 0])
        center = batch["center"][0].numpy()
        if not np.isfinite(center).all() or np.any((center > 0) & ((owners == 0) | (batch["validity"][0].numpy() == 0))):
            raise ValueError("Invalid Gaussian target: " + batch["sample_id"])
        for owner in ids:
            if np.max(center[owners == owner]) != 1:
                raise ValueError("Missing center peak: " + batch["sample_id"])
        sample = dataset.base.manifest.samples[index]
        with Image.open(dataset.base.data_root / sample.word_owner_annotation_path) as opened:
            expected = len(np.unique(np.asarray(opened)[np.asarray(opened) > 0]))
        native_words += len(ids)
        rows.append({"sample_id": batch["sample_id"], "expected_words": expected,
            "native_words": len(ids), "lost_before_auxiliary_targets": expected - len(ids),
            "center_sha256": sha256(center.astype("<f4").tobytes()).hexdigest()})
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print("[targets] {}/{}".format(index + 1, len(dataset)), flush=True)
    return {"status": "PASS", "result_class": "smoke-test result", "images": len(dataset),
            "native_words": native_words, "lost_before_auxiliary_targets": sum(r["lost_before_auxiliary_targets"] for r in rows),
            "augmentation": "disabled; spatial augmentation replay covered by unit tests", "per_image": rows}


def memory(device):
    return {"device": str(device), "dtype": "float32", "batch_size": 8, "input_shape": [8, 3, 512, 512],
            "model_set": "segmentation only; sequential branches", "oom": False,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def experiment_environment(device):
    import cv2
    import PIL
    import scipy
    import torchvision
    return {**runtime_environment(), "numpy": np.__version__, "opencv": cv2.__version__,
            "pillow": PIL.__version__, "scipy": scipy.__version__, "torchvision": torchvision.__version__,
            "execution_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def train_branch(directory, model, optimizer, train_loader, val_loader, references, transform, sampler, cfg, contract, device):
    tc = training_config(cfg)
    records = _epoch_log(directory / "epochs.jsonl")
    last, best = directory / "last.pt", directory / "best.pt"
    start, best_epoch, best_metrics, best_key, step = 1, 0, None, None, 0
    def save(path, epoch, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, global_step=step,
            config=tc, resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract, train_metrics=trained, validation_metrics=metrics)
    if best.exists():
        saved = load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
                                expected_run_contract=contract, expected_training_config=tc)
        best_epoch, best_metrics = saved["epoch"], saved["validation_metrics"]
        best_key = selection_key(best_metrics, best_epoch)
        del saved
    if last.exists():
        saved = load_checkpoint(last, model=model, optimizer=optimizer, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc, restore_rng=True)
        start, step = saved["epoch"] + 1, saved["global_step"]
        if len(records) == start - 2:
            _append_epoch(directory / "epochs.jsonl", {"epoch": start - 1, "global_step": step,
                "train": saved["train_metrics"], "validation": saved["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last), "recovered_from_checkpoint": True})
            records = _epoch_log(directory / "epochs.jsonl")
        if len(records) != start - 1 or sha256_file(last) != records[-1]["last_checkpoint_sha256"]:
            raise ValueError("Epoch log/latest checkpoint disagree")
        key = selection_key(saved["validation_metrics"], start - 1)
        if best_key is None or key > best_key:
            best_epoch, best_metrics, best_key = start - 1, saved["validation_metrics"], key
            save(best, best_epoch, saved["train_metrics"], best_metrics)
        del saved
    elif records or best.exists():
        raise ValueError("Training artifacts without a resumable latest checkpoint")
    if [r["epoch"] for r in records] != list(range(1, start)):
        raise ValueError("Non-contiguous training history")
    for epoch in range(start, cfg["epochs"] + 1):
        sampler.set_epoch(epoch - 1)
        transform.set_epoch(epoch - 1)
        trained = train_multitask_epoch(model, train_loader, optimizer, device,
            variant=contract["variant"], loss_config=cfg["loss"])
        metrics = evaluate_multitask(model, val_loader, references, device, cfg)["aggregate"]
        if metrics["gt_words"] != cfg["expected_validation_words"]:
            raise ValueError("Validation care-word denominator changed")
        step += trained["steps"]
        digest = save(last, epoch, trained, metrics)
        key = selection_key(metrics, epoch)
        if best_key is None or key > best_key:
            best_epoch, best_metrics, best_key = epoch, metrics, key
            save(best, epoch, trained, metrics)
        row = {"epoch": epoch, "global_step": step, "train": trained, "validation": metrics,
               "last_checkpoint_sha256": digest}
        _append_epoch(directory / "epochs.jsonl", row)
        records.append(row)
        print("[{}] {}/{} F1={:.5f} merge={:.5f} split={:.5f} best={}".format(
            contract["variant"], epoch, cfg["epochs"], metrics["proposal_f1_iou_50"],
            metrics["merge_rate"], metrics["split_rate"], best_epoch), flush=True)
    if len(records) != cfg["epochs"] or best_metrics is None:
        raise RuntimeError("Incomplete multi-task training")
    return {"status": "SUCCESS", "result_class": "approximate reproduction", "run_contract": contract,
            "checkpoint": {"epoch": best_epoch, "path": "best.pt", "sha256": sha256_file(best)},
            "validation_word_geometry": best_metrics, "memory": memory(device), "environment": experiment_environment(device)}


def run(args):
    cfg, source = read_config(args.config, args.mode)
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be nonnegative")
        cfg["num_workers"] = args.num_workers
        cfg["scientific_choices"].append({"component": "explicit_worker_override", "classification": "HARDWARE_DEVIATION",
            "rationale": "Explicit --num-workers override for execution environment compatibility; no silent fallback.",
            "expected_effect": "Per-sample augmentation seeds unchanged; different worker runtime recorded.",
            "reference": "docs/WORD_MULTITASK.md"})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run --action preflight --device cpu or use a qualified GPU host")
    root, output = args.target_root.resolve(), args.output_root.resolve()
    # Do not allow generated outputs to contaminate datasets or old experiments.
    if output == root or root in output.parents or output == REPO:
        raise ValueError("Use an isolated output directory, outside the dataset and repository root")
    manifest_path = root / cfg["train_manifest"]
    if sha256_file(manifest_path) != cfg["train_manifest_sha256"]:
        raise ValueError("Qualified manifest checksum differs")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This matched runner accepts qualified Total-Text TRAIN word-owner artifacts only")
    train_ids, val_ids, split_sha = _split_indices(manifest, cfg["seed"], cfg["validation_count"])
    if split_sha != cfg["validation_ids_sha256"]:
        raise ValueError("Frozen validation split differs")
    transforms = [_transform(source, "letterbox", train=value) for value in (True, False)]
    datasets = [WordMultitaskDataset(SegmentationDataset(manifest, root, transform=t), **cfg["targets"]) for t in transforms]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args._owns_output = True
        if args.action == "preflight":
            audit_contract = {"config": cfg, "target_source_sha256": sha256_file(REPO / "segmentation/word_multitask.py"),
                              "paired_sources_sha256": {name: sha256_file(REPO / name) for name in
                                  ("segmentation/totaltext_word_kernel.py", "segmentation/transforms.py",
                                   "segmentation/data.py", "segmentation/totaltext_image_frequency_augment.py")},
                              "manifest_sha256": sha256_file(manifest_path), "validation_ids_sha256": split_sha}
            path = output / "target_qualification.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if existing["contract"] != audit_contract or existing["status"] != "PASS":
                    raise ValueError("Existing target qualification has a different contract")
                return existing
            report = {**qualify_targets(datasets[1]), "contract": audit_contract}
            _json(path, report)
            return report
        if args.encoder_checkpoint is None or args.encoder_sha256 is None:
            raise ValueError("Provide --encoder-checkpoint and full --encoder-sha256 (local ImageNet V2, no download)")
        encoder_sha = sha256_file(args.encoder_checkpoint)
        if encoder_sha != args.encoder_sha256 or not encoder_sha.startswith(cfg["encoder_sha256_prefix"]):
            raise ValueError("Encoder checksum differs from the declared ImageNet V2 artifact")
        options = {"num_workers": cfg["num_workers"], "pin_memory": device.type == "cuda",
                   "timeout": cfg["loader_timeout_seconds"] if cfg["num_workers"] else 0}
        sampler = EpochShuffleSampler(len(train_ids), seed=cfg["seed"])
        train_loader = DataLoader(Subset(datasets[0], train_ids), batch_size=cfg["batch_size"], sampler=sampler, **options)
        val_loader = DataLoader(Subset(datasets[1], val_ids), batch_size=cfg["batch_size"], shuffle=False, **options)
        references = _validation_reference(manifest, root, val_ids, transforms[1])
        image_paths = {s.sample_id: root / s.image_path for s in manifest.samples}
        variants = cfg["variants"] if args.variant == "all" else [args.variant]
        summaries = []
        for variant in variants:
            contract = run_contract(cfg, source, variant, encoder_sha, split_sha)
            contract["runtime_inputs"] = {"target_root": str(root), "encoder_checkpoint": str(args.encoder_checkpoint.resolve())}
            directory = output / variant
            directory.mkdir(exist_ok=True)
            ensure_branch_contract(directory, contract)
            if args.action == "smoke":
                cached_smoke = reusable_smoke(directory, contract, device)
                if cached_smoke is not None:
                    summaries.append(cached_smoke)
                    print("[reuse smoke] " + str(directory / "smoke.json"), flush=True)
                    continue
            result_path = directory / "run_manifest.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result["run_contract"] != contract or result["status"] != "SUCCESS" or sha256_file(directory / "best.pt") != result["checkpoint"]["sha256"]:
                    raise ValueError("Completed branch artifacts differ")
                if args.action == "train":
                    summaries.append(result)
                    print("[reuse] " + str(result_path), flush=True)
                    continue
                if args.action == "evaluate" and (directory / "validation.json").exists():
                    cached = json.loads((directory / "validation.json").read_text())
                    if cached["run_contract"] != contract or cached["checkpoint_sha256"] != result["checkpoint"]["sha256"]:
                        raise ValueError("Completed evaluation provenance differs")
                    for row in cached["per_image"]:
                        exported = directory / "validation_exports" / row["sample_id"].replace(":", "__")
                        for name, digest in row["exports"].items():
                            if sha256_file(exported / name) != digest:
                                raise ValueError("Completed validation export checksum differs")
                    summaries.append(cached)
                    print("[reuse] " + str(directory / "validation.json"), flush=True)
                    continue
            elif args.action == "evaluate":
                raise ValueError("Evaluation requires a completed run_manifest.json")
            seed_everything(cfg["seed"], deterministic_algorithms=True)
            if device.type == "cuda":
                _prepare_cuda_memory_tracking(device)
            model = build_word_model(variant, skip_floor=cfg["skip_floor"], encoder_pretrained=False)
            state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
            model.encoder.load_state_dict(state, strict=True)
            del state
            model.to(device)
            # Model creation consumes different RNG counts across variants;
            # reseed afterward for identical sampler/worker stochastic streams.
            seed_everything(cfg["seed"], deterministic_algorithms=True)
            optimizer = build_optimizer(model, training_config(cfg))
            started = time.monotonic()
            if args.action == "smoke":
                transforms[0].set_epoch(0)
                sampler.set_epoch(0)
                batch = next(iter(train_loader))
                metrics = train_multitask_epoch(model, [batch], optimizer, device, variant=variant, loss_config=cfg["loss"])
                report = {"result_class": "smoke-test result", "run_contract": contract, "train": metrics,
                          "memory": memory(device), "environment": experiment_environment(device),
                          "parameters": sum(p.numel() for p in model.parameters())}
                _json(directory / "smoke.json", report)
                summaries.append(report)
            elif args.action == "train":
                result = train_branch(directory, model, optimizer, train_loader, val_loader, references,
                    transforms[0], sampler, cfg, contract, device)
                result["elapsed_seconds"] = time.monotonic() - started
                result["parameters"] = sum(p.numel() for p in model.parameters())
                _json(result_path, result)
                summaries.append(result)
            else:
                export = directory / "validation_exports"
                load_checkpoint(directory / "best.pt", model=model, expected_architecture=model.architecture_identity,
                    expected_run_contract=contract, expected_training_config=training_config(cfg))
                report = evaluate_multitask(model, val_loader, references, device, cfg, export_dir=export,
                    image_paths=image_paths, compare_postprocessing=True)
                if report["aggregate"] != result["validation_word_geometry"]:
                    raise ValueError("Selected checkpoint validation drift")
                report.update(result_class="approximate reproduction", reproduction_mode="approximate",
                              run_contract=contract, checkpoint_sha256=result["checkpoint"]["sha256"],
                              memory=memory(device), environment=experiment_environment(device))
                _json(directory / "validation.json", report)
                summaries.append(report)
            del model, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return {"action": args.action, "branches": len(summaries), "output_root": str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate", "strict"), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--target-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--encoder-checkpoint", type=Path)
    p.add_argument("--encoder-sha256")
    p.add_argument("--num-workers", type=int, help="Explicit, recorded worker override; use 0 in restricted sandboxes")
    p.add_argument("--variant", choices=("all", *VARIANTS), default="all")
    p.add_argument("--action", choices=("preflight", "smoke", "train", "evaluate"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    try:
        report = run(args)
        print(json.dumps({k: v for k, v in report.items() if k not in ("per_image", "contract")}, indent=2))
    except Exception as error:
        # Persist failure only in a directory already established by this run.
        if getattr(args, "_owns_output", False):
            _json(args.output_root / "failure.json", {"status": "FAILED", "result_class": "smoke-test result" if args.action == "smoke" else "approximate reproduction",
                "type": type(error).__name__, "message": str(error), "action": args.action, "device": args.device,
                "oom": isinstance(error, torch.cuda.OutOfMemoryError), "precision": "float32",
                "model_set": "segmentation only", "variant": args.variant, "input_shape": [8, 3, 512, 512],
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if args.device == "cuda:0" and torch.cuda.is_available() else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if args.device == "cuda:0" and torch.cuda.is_available() else None})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
