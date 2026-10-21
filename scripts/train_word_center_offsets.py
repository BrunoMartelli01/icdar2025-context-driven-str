"""Restartable APPROXIMATE auxiliary center-vector training on qualified Total-Text."""

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

from models.unet_adv import PracticalTotalTextHighResAGUNet
from models.word_center_offsets import WordCenterOffsetAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices, _epoch_log, _append_epoch
from scripts.train_totaltext_v4_image_frequency_ablation import _config, _transform, CONFIG_PATH as SOURCE_PATH
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.train import EpochShuffleSampler
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything, runtime_environment
from segmentation.word_center_offsets import (
    load_center_offset_config, WordCenterOffsetDataset, WordCenterOffsetTrainingConfig,
    word_center_offset_loss, train_word_center_offset_epoch,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/word_center_offsets_totaltext_train_v1.json"


def read_config(path, mode):
    cfg = load_center_offset_config(path, mode=mode)
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
    training_config(cfg)
    return cfg, source


def training_config(cfg):
    t = cfg["training"]
    return WordCenterOffsetTrainingConfig(epochs=t["epochs"], seed=t["seed"],
        optimizer=t["optimizer"], learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
        batch_size=t["batch_size"], adam_betas=tuple(t["adam_betas"]), adam_epsilon=t["adam_epsilon"],
        deterministic_algorithms=t["deterministic_algorithms"], learning_rate_schedule="none",
        checkpoint_selection="validation_offset_loss", offset_beta=cfg["loss"]["beta"])


def evaluate_offsets(model, batches, device, *, beta, export_dir=None):
    """Word means per image, then supervised-image means; keep every sample row.

    Also report endpoint error in native pixels using native W/H, not input
    dimensions. Empty images have null objectives and explicit counts. No base
    mask or recognition metric enters offset checkpoint selection.
    """
    model.eval()
    rows, losses, endpoints = [], [], []
    with torch.inference_mode():
        for batch in batches:
            local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            raw = model(local["image"])
            for index, sample_id in enumerate(batch["sample_id"]):
                loss = word_center_offset_loss(raw[index:index+1], local["center_offsets"][index:index+1],
                    local["owners_native"][index:index+1], local["validity"][index:index+1], beta=beta)
                owners, valid = local["owners_native"][index], local["validity"][index, 0].bool()
                ids = torch.unique(owners[valid & (owners > 0)])
                row = {"sample_id": sample_id, "expected_words": int(batch["expected_word_count"][index]),
                       "surviving_words": len(ids), "offset_loss": None, "endpoint_error_native_pixels": None}
                if len(ids):
                    h, w = raw.shape[-2:]
                    delta = (raw[index].tanh() - local["center_offsets"][index])
                    error = (delta * delta.new_tensor([w, h])[:, None, None]).square().sum(0).sqrt()
                    endpoint = float(torch.stack([error[(owners == owner) & valid].mean() for owner in ids]).mean())
                    row.update(offset_loss=float(loss), endpoint_error_native_pixels=endpoint)
                    losses.append(float(loss)); endpoints.append(endpoint)
                if export_dir is not None:
                    directory = Path(export_dir) / sample_id.replace(":", "__")
                    directory.mkdir(parents=True, exist_ok=True)
                    arrays = {"normalized_offsets": raw[index].tanh().cpu().numpy(),
                              "target_offsets": local["center_offsets"][index].cpu().numpy(),
                              "owners": owners.cpu().numpy(), "validity": valid.cpu().numpy()}
                    path = directory / "offsets.npz"
                    if path.exists():
                        with np.load(path, allow_pickle=False) as cached:
                            if set(cached.files) != set(arrays) or any(not np.array_equal(cached[k], v) for k, v in arrays.items()):
                                raise ValueError("Existing offset export differs: " + sample_id)
                    else:
                        np.savez_compressed(path, **arrays)
                    row["export_sha256"] = sha256_file(path)
                rows.append(row)
    if not rows or not losses:
        raise ValueError("Validation requires samples and at least one surviving supervised word")
    expected = sum(r["expected_words"] for r in rows)
    surviving = sum(r["surviving_words"] for r in rows)
    return {"aggregate": {"offset_loss": sum(losses)/len(losses), "endpoint_error_native_pixels": sum(endpoints)/len(endpoints),
            "samples": len(rows), "supervised_images": len(losses), "empty_images": len(rows)-len(losses),
            "expected_words": expected, "surviving_words": surviving, "lost_words": expected-surviving}, "per_image": rows}


def memory(device, cfg):
    return {"device": str(device), "precision": "float32", "batch_size": cfg["training"]["batch_size"],
            "input_shape": [cfg["training"]["batch_size"], 3, *cfg["input_size"]],
            "model_set": "auxiliary center-offset U-Net only; base warm start loaded on CPU then released",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def initialize_model(cfg, base_path):
    base = PracticalTotalTextHighResAGUNet(skip_floor=cfg["skip_floor"], encoder_pretrained=False)
    saved = load_checkpoint(base_path, model=base, expected_architecture=base.architecture_identity)
    source_floor = saved.get("run_contract", {}).get("config", {}).get("skip_floor")
    if source_floor != cfg["skip_floor"]:
        raise ValueError("Base checkpoint must record the same practical skip_floor in its run contract")
    model = WordCenterOffsetAGUNet(skip_floor=cfg["skip_floor"], encoder_pretrained=False)
    receipt = model.initialize_from_text_model(base)
    receipt.update(source_epoch=saved["epoch"], source_run_contract=saved["run_contract"])
    del saved, base
    return model, receipt


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
        trained = train_word_center_offset_epoch(model, train_loader, optimizer, device, beta=tc.offset_beta)
        validated = evaluate_offsets(model, val_loader, device, beta=tc.offset_beta)["aggregate"]
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
        print(f"[center offsets] {epoch}/{tc.epochs} train={trained['loss']:.6f} val={validated['offset_loss']:.6f} "
              f"endpoint={validated['endpoint_error_native_pixels']:.3f}px best={best_epoch}", flush=True)
    if len(records) != tc.epochs:
        return {"status": "IN_PROGRESS", "completed_epochs": len(records), "configured_epochs": tc.epochs}
    load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
        expected_run_contract=contract, expected_training_config=tc)
    final = evaluate_offsets(model, val_loader, device, beta=tc.offset_beta, export_dir=directory / "validation_exports")
    if final["aggregate"] != records[best_epoch-1]["validation"]:
        raise RuntimeError("Selected auxiliary checkpoint validation drifted")
    _json(directory / "validation.json", {"result_class": "approximate reproduction", "run_contract": contract, **final})
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "run_contract": contract,
              "completed_epochs": tc.epochs, "checkpoint": {"path": "best.pt", "epoch": best_epoch, "sha256": sha256_file(best)},
              "validation": final["aggregate"], "memory": memory(device, cfg), "environment": runtime_environment()}
    _json(directory / "run_manifest.json", result)
    return result


def run(args):
    cfg, source = read_config(args.config, args.mode)
    for argument, key, classification in ((args.epochs, "epochs", "ASSUMPTION"),
        (args.batch_size, "batch_size", "HARDWARE_DEVIATION"), (args.num_workers, "num_workers", "HARDWARE_DEVIATION")):
        if argument is not None:
            cfg["training"][key] = argument
            cfg["scientific_choices"].append({"component": "explicit_"+key+"_override", "classification": classification,
                "rationale": "Explicit launcher override recorded in resolved training configuration", "expected_effect": "Changes the recorded execution protocol",
                "reference": "docs/WORD_CENTER_OFFSETS.md"})
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
    if output == REPO or output == root or root in output.parents or output in root.parents:
        raise ValueError("Use an isolated output directory outside the target dataset")
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
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_files = ("scripts/train_word_center_offsets.py", "models/word_center_offsets.py", "models/unet_adv.py",
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
                rows.append({"sample_id": batch["sample_id"], "expected_words": batch["expected_word_count"],
                             "surviving_words": surviving, "offset_sha256": sha256(batch["center_offsets"].numpy().tobytes()).hexdigest()})
                if (i+1) % 100 == 0: print(f"[targets] {i+1}/{len(datasets[1])}", flush=True)
            report = {"status": "PASS", "result_class": "smoke-test result", "contract": data_contract,
                      "train_images": len(train_ids), "validation_images": len(val_ids), "per_image": rows}
            path = output / "target_qualification.json"
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
        directory = output / ("smoke" if args.action == "smoke" else "train")
        directory.mkdir(exist_ok=True)
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
            metrics = train_word_center_offset_epoch(model, [next(iter(train_loader))], optimizer, device, beta=tc.offset_beta)
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
                "result_class": "smoke-test result" if args.action == "smoke" else "approximate reproduction",
                "type": type(error).__name__, "message": str(error), "oom": isinstance(error, torch.cuda.OutOfMemoryError),
                **memory(torch.device(args.device), cfg)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
