"""Matched APPROXIMATE coherent-center-vote loss training; unchanged U-Net architecture."""

import argparse
from dataclasses import asdict
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices, _epoch_log, _append_epoch
from scripts.train_totaltext_v4_image_frequency_ablation import _config, _transform, CONFIG_PATH as SOURCE_PATH
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.train import EpochShuffleSampler
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything, runtime_environment
from segmentation.word_center_offsets import WordCenterOffsetDataset
from segmentation.word_center_votes import (
    load_vote_config, params_from_config, CenterVoteTrainingConfig, center_vote_objective,
    train_center_vote_epoch as train_word_center_offset_epoch, evaluate_center_votes as evaluate_offsets,
)
from scripts.train_word_center_offsets import initialize_model, memory


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/word_center_votes_totaltext_train_v2.json"


def read_config(path, mode, variant):
    cfg = load_vote_config(path, mode=mode, variant=variant)
    if ((REPO / cfg["preprocessing_source"]).resolve() != SOURCE_PATH
        or sha256_file(SOURCE_PATH) != cfg["preprocessing_source_sha256"]):
        raise ValueError("Qualified preprocessing source changed; use an independently qualified approximate profile")
    source = _config()
    data, training = cfg["dataset"], cfg["training"]
    if (cfg["input_size"] != source["input_size"] or cfg["native_mask_size"] != source["native_mask_size"]
        or training["seed"] != source["training"]["seed"] or cfg["skip_floor"] != source["training"]["skip_floor"]
        or data["manifest"] != source["train_manifest"] or data["manifest_sha256"] != source["train_manifest_sha256"]
        or data["images"] != 1255 or data["validation_images"] != 125 or data["validation_expected_words"] != 983):
        raise ValueError("Profile must use the qualified Total-Text v4 TRAIN1255 and frozen validation125 protocol")
    if (type(training["num_workers"]) is not int or training["num_workers"] < 0
        or type(training["loader_timeout_seconds"]) is not int or training["loader_timeout_seconds"] < 1):
        raise ValueError("Invalid loader worker/timeout settings")
    if (training["checkpoint_selection"] != "validation_offset_loss" or training["precision"] != "float32"
        or training["deterministic_algorithms"] is not True):
        raise ValueError("Only explicit FP32 deterministic offset-loss validation selection is implemented")
    tolerance = cfg["qualification"]["perfect_vote_loss_max"]
    if type(tolerance) not in (int,float) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite numerical target-qualification tolerance required")
    training_config(cfg)
    return cfg, source


def training_config(cfg):
    t = cfg["training"]
    return CenterVoteTrainingConfig(epochs=t["epochs"], seed=t["seed"],
        optimizer=t["optimizer"], learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
        batch_size=t["batch_size"], adam_betas=tuple(t["adam_betas"]), adam_epsilon=t["adam_epsilon"],
        deterministic_algorithms=t["deterministic_algorithms"], learning_rate_schedule="none",
        checkpoint_selection="validation_offset_loss", offset_beta=cfg["loss"]["beta"],
        **{k:v for k,v in asdict(params_from_config(cfg)).items() if k != "beta"})


def train(directory, model, optimizer, train_loader, val_loader, transform, sampler, cfg, contract, device, *, stop_after_epoch=None):
    """Atomic last/best checkpoints with interruption recovery and earliest ties."""
    tc = training_config(cfg)
    last, best, log = directory / "last.pt", directory / "best.pt", directory / "epochs.jsonl"
    records = _epoch_log(log)
    start, step, best_loss, best_epoch = 1, 0, float("inf"), None
    def save(path, epoch, trained, validated):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, global_step=step,
            config=tc, resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract, train_metrics=trained, validation_metrics=validated)
    if best.exists():
        saved = load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc)
        best_loss, best_epoch = saved["validation_metrics"]["offset_loss"], saved["epoch"]
        del saved
    if last.exists():
        saved = load_checkpoint(last, model=model, optimizer=optimizer, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc, restore_rng=True)
        start, step = saved["epoch"]+1, saved["global_step"]
        if len(records) == start-2:
            _append_epoch(log, {"epoch": start-1, "global_step": step, "train": saved["train_metrics"],
                "validation": saved["validation_metrics"], "last_checkpoint_sha256": sha256_file(last), "recovered_from_checkpoint": True})
            records = _epoch_log(log)
        if len(records) != start-1 or sha256_file(last) != records[-1]["last_checkpoint_sha256"]:
            raise ValueError("Epoch log and resumable checkpoint disagree")
        if saved["validation_metrics"]["offset_loss"] < best_loss:
            best_loss, best_epoch = saved["validation_metrics"]["offset_loss"], saved["epoch"]
            save(best, best_epoch, saved["train_metrics"], saved["validation_metrics"])
        del saved
    elif records or best.exists():
        raise ValueError("Training artifacts lack last.pt; preserve this run and select a new output directory")
    if [r["epoch"] for r in records] != list(range(1, start)):
        raise ValueError("Epoch history is not contiguous")
    budget = tc.epochs if stop_after_epoch is None else stop_after_epoch
    if type(budget) is not int or not 1 <= budget <= tc.epochs:
        raise ValueError("--stop-after-epoch must be within the configured epoch budget")
    for epoch in range(start, budget+1):
        sampler.set_epoch(epoch-1); transform.set_epoch(epoch-1)
        trained = train_word_center_offset_epoch(model, train_loader, optimizer, device, params=params_from_config(cfg))
        validated = evaluate_offsets(model, val_loader, device, params=params_from_config(cfg))["aggregate"]
        if (validated["samples"] != cfg["dataset"]["validation_images"]
            or validated["expected_words"] != cfg["dataset"]["validation_expected_words"]):
            raise ValueError("Frozen validation sample/word denominator changed")
        step += trained["steps"]
        digest = save(last, epoch, trained, validated)
        if validated["offset_loss"] < best_loss:
            best_loss, best_epoch = validated["offset_loss"], epoch
            save(best, epoch, trained, validated)
        _append_epoch(log, {"epoch": epoch, "global_step": step, "train": trained, "validation": validated,
                            "last_checkpoint_sha256": digest})
        records = _epoch_log(log)
        print(f"[center votes {cfg['variant']}] {epoch}/{tc.epochs} train_total={trained['loss']:.6f} "
              f"val_regression={validated['offset_loss']:.6f} consensus={validated['consensus_loss']:.6f} "
              f"neighbor={validated['separation_loss']:.6f} "
              f"endpoint={validated['endpoint_error_native_pixels']:.3f}px "
              f"scatter={validated['endpoint_scatter_native_pixels']:.3f}px "
              f"wrong_neighbor={validated['wrong_neighbor_vote_rate']:.4f} best={best_epoch}", flush=True)
    if len(records) != tc.epochs:
        return {"status": "IN_PROGRESS", "completed_epochs": len(records), "configured_epochs": tc.epochs}
    load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
        expected_run_contract=contract, expected_training_config=tc)
    final = evaluate_offsets(model, val_loader, device, params=params_from_config(cfg), export_dir=directory / "validation_exports")
    if final["aggregate"] != records[best_epoch-1]["validation"]:
        raise RuntimeError("Selected auxiliary checkpoint validation drifted")
    _json(directory / "validation.json", {"result_class": "approximate reproduction", "run_contract": contract, **final})
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "run_contract": contract,
              "completed_epochs": tc.epochs, "checkpoint": {"path": "best.pt", "epoch": best_epoch, "sha256": sha256_file(best)},
              "validation": final["aggregate"], "memory": memory(device, cfg), "environment": runtime_environment()}
    _json(directory / "run_manifest.json", result)
    return result


def run(args):
    cfg, source = read_config(args.config, args.mode, args.variant)
    for argument, key, classification in ((args.epochs, "epochs", "ASSUMPTION"),
        (args.batch_size, "batch_size", "HARDWARE_DEVIATION"), (args.num_workers, "num_workers", "HARDWARE_DEVIATION")):
        if argument is not None:
            cfg["training"][key] = argument
            cfg["scientific_choices"].append({"component": "explicit_"+key+"_override", "classification": classification,
                "rationale": "Explicit launcher override recorded in resolved training configuration", "expected_effect": "Changes the recorded execution protocol",
                "reference": "docs/WORD_CENTER_VOTE_LOSS.md"})
    tc = training_config(cfg)
    if cfg["training"]["num_workers"] < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= tc.epochs:
        raise ValueError("--stop-after-epoch must be within configured epochs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this training command on the physical GPU host")
    seed_everything(tc.seed, deterministic_algorithms=tc.deterministic_algorithms)
    root, output = args.target_root.resolve(), args.output_root.resolve()
    if output == REPO or output in REPO.parents or output == root or root in output.parents or output in root.parents:
        raise ValueError("Use an isolated output directory outside the target dataset")
    if args.base_checkpoint is not None:
        checkpoint_root = args.base_checkpoint.resolve().parent
        if output == checkpoint_root or checkpoint_root in output.parents or output in checkpoint_root.parents:
            raise ValueError("Use a new output directory outside the source checkpoint run")
    manifest_path = root / cfg["dataset"]["manifest"]
    if sha256_file(manifest_path) != cfg["dataset"]["manifest_sha256"]:
        raise ValueError("Qualified Total-Text manifest checksum differs")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Only qualified Total-Text TRAIN owner artifacts are supported")
    train_ids, val_ids, split_sha = _split_indices(manifest, tc.seed, cfg["dataset"]["validation_images"])
    if split_sha != cfg["dataset"]["validation_ids_sha256"]:
        raise ValueError("Frozen Total-Text validation split changed")
    transforms = [_transform(source, "letterbox", train=value) for value in (True, False)]
    datasets = [WordCenterOffsetDataset(SegmentationDataset(manifest, root, transform=t)) for t in transforms]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_files = ("scripts/train_word_center_votes.py", "segmentation/word_center_votes.py", "scripts/train_word_center_offsets.py", "models/word_center_offsets.py", "models/unet_adv.py",
            "segmentation/word_center_offsets.py", "segmentation/data.py", "segmentation/training.py", "segmentation/transforms.py",
            "segmentation/totaltext_word_kernel.py", "segmentation/totaltext_image_frequency_augment.py", "segmentation/train.py",
            "scripts/train_totaltext_v4_highres_ablation.py", "scripts/train_totaltext_v4_image_frequency_ablation.py")
        data_contract = {"resolved_config": cfg, "resolved_preprocessing": source, "training_config": asdict(tc),
            "manifest_sha256": sha256_file(manifest_path), "validation_ids_sha256": split_sha,
            "target_root": str(root), "source_sha256": {name: sha256_file(REPO / name) for name in source_files},
            "environment": {**runtime_environment(), "numpy": np.__version__,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}
        if args.action == "preflight":
            rows = []
            for i in range(len(datasets[1])):
                batch = datasets[1][i]
                ids = np.unique(batch["owners_native"].numpy()); surviving = int((ids > 0).sum())
                stats = center_vote_objective(torch.atanh(batch["center_offsets"][None]), batch["center_offsets"][None],
                    batch["owners_native"][None], batch["validity"][None], params=params_from_config(cfg))
                if not torch.isfinite(stats["loss"]) or float(stats["loss"]) > cfg["qualification"]["perfect_vote_loss_max"]:
                    raise ValueError("Perfect target votes failed numerical qualification: " + batch["sample_id"])
                rows.append({"sample_id": batch["sample_id"], "expected_words": batch["expected_word_count"],
                    "surviving_words": surviving, "coincident_centroid_words": stats["coincident_centroid_words"],
                    "perfect_vote_loss": float(stats["loss"]), "offset_sha256": sha256(batch["center_offsets"].numpy().tobytes()).hexdigest()})
                if (i+1) % 100 == 0: print(f"[targets] {i+1}/{len(datasets[1])}", flush=True)
            report = {"status": "PASS", "result_class": "smoke-test result", "contract": data_contract,
                      "train_images": len(train_ids), "validation_images": len(val_ids), "per_image": rows}
            path = output / args.variant / "target_qualification.json"
            path.parent.mkdir(exist_ok=True)
            if path.exists() and json.loads(path.read_text()) != report:
                raise ValueError("Existing target qualification differs; choose a new output directory")
            _json(path, report)
            return report
        if args.base_checkpoint is None or args.base_sha256 is None:
            raise ValueError("Provide --base-checkpoint and its full --base-sha256; there is no random-weight training fallback")
        base_path = args.base_checkpoint.resolve()
        if sha256_file(base_path) != args.base_sha256:
            raise ValueError("Base U-Net checkpoint SHA256 differs")
        contract = json.loads(json.dumps({**data_contract, "base_checkpoint": str(base_path), "base_sha256": args.base_sha256,
            "execution_device": str(device), "claim_label": "approximate reproduction"}))
        directory = output / args.variant / ("smoke" if args.action == "smoke" else "train")
        directory.mkdir(parents=True, exist_ok=True)
        contract_path = directory / "run_contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError("Existing run provenance differs; preserve it and choose a new output directory")
        elif any(directory.iterdir()):
            raise ValueError("Nonempty run directory has no provenance")
        else:
            _json(contract_path, contract)
        args._owns_output = True
        args._resolved_cfg = cfg
        completed = directory / "run_manifest.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if (result.get("status") != "SUCCESS" or result["run_contract"] != contract
                or sha256_file(directory / "best.pt") != result["checkpoint"]["sha256"]):
                raise ValueError("Completed auxiliary artifacts differ")
            print("[reuse completed] " + str(completed), flush=True)
            return result
        model, receipt = initialize_model(cfg, base_path)
        _json(directory / "warm_start.json", receipt)
        seed_everything(tc.seed, deterministic_algorithms=tc.deterministic_algorithms)
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        optimizer = build_optimizer(model, tc)
        sampler = EpochShuffleSampler(len(train_ids), seed=tc.seed)
        loader_options = {"num_workers": cfg["training"]["num_workers"], "pin_memory": device.type == "cuda"}
        if loader_options["num_workers"]:
            loader_options["timeout"] = cfg["training"]["loader_timeout_seconds"]
        train_loader = DataLoader(Subset(datasets[0], train_ids), batch_size=tc.batch_size, sampler=sampler, **loader_options)
        val_loader = DataLoader(Subset(datasets[1], val_ids), batch_size=tc.batch_size, shuffle=False, **loader_options)
        _json(directory / "resolved_config.json", contract)
        if args.action == "smoke":
            metrics = train_word_center_offset_epoch(model, [next(iter(train_loader))], optimizer, device, params=params_from_config(cfg))
            if not torch.any(model.head.weight != 0): raise RuntimeError("Auxiliary vector head did not update on the real training batch")
            report = {"status": "PASS", "result_class": "smoke-test result", "run_contract": contract,
                      "train": metrics, "memory": memory(device, cfg), "environment": runtime_environment()}
            _json(directory / "smoke.json", report)
            return report
        return train(directory, model, optimizer, train_loader, val_loader, transforms[0], sampler, cfg, contract, device,
                     stop_after_epoch=args.stop_after_epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("approximate", "strict"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=("control", "coherent"), required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--base-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--action", choices=("preflight", "smoke", "train"), default="train")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({k: v for k, v in result.items() if k not in ("per_image", "contract", "run_contract")}, indent=2))
    except Exception as error:
        if getattr(args, "_owns_output", False):
            cfg = args._resolved_cfg
            _json(args.output_root / "failure.json", {"status": "FAILED", "action": args.action,
                "variant": args.variant, "loss_parameters": asdict(params_from_config(cfg)),
                "result_class": "smoke-test result" if args.action == "smoke" else "approximate reproduction",
                "type": type(error).__name__, "message": str(error), "oom": isinstance(error, torch.cuda.OutOfMemoryError),
                **memory(torch.device(args.device), cfg)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
