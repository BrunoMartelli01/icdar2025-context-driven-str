"""Fork one frozen letterbox checkpoint into two isolated 500-epoch TRAIN runs.

Both branches start from the exact epoch-71 model, optimizer, and RNG state.
This is a practical ablation, not a paper-faithful training protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

from torch.utils.data import DataLoader, Subset
import torch

from models.unet_adv import PracticalTotalTextHighResAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import (
    _append_epoch, _epoch_log, _json, _split_indices, _training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    BASE_MODEL_SOURCE, CONFIG_PATH as SOURCE_CONFIG_PATH,
    TRANSFORM_SOURCE, WAVELET_SOURCE, _config as source_config,
    _transform, _validation_geometry,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    build_optimizer, load_checkpoint, measure_foreground_iou, save_checkpoint,
    seed_everything, train_one_epoch,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_letterbox_500_separator_pair_v1.json"
BRANCHES = {"letterbox_500_control": 3.0, "letterbox_500_sep5": 5.0}


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (value.get("reproduction_mode") != "approximate"
        or value.get("classification") != "PRACTICAL_APPROXIMATION"
        or value.get("split") != "frozen_TRAIN_1130_validation_125"
        or value.get("source_variant") != "letterbox"
        or value.get("source_completed_epochs") != 100
        or value.get("fork_checkpoint_epoch") != 71
        or value.get("total_epochs_including_fork_prefix") != 500
        or {name: item["separation_weight"] for name, item in value.get("branches", {}).items()}
            != BRANCHES):
        raise ValueError("Frozen concurrent letterbox ablation configuration changed")
    return value


def _source(args, config: dict, training: dict):
    root = args.target_root.resolve()
    manifest_path = root / training["train_manifest"]
    if sha256_file(manifest_path) != training["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text v4 TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("These two branches permit TRAIN only")
    train_ids, val_ids, val_sha = _split_indices(manifest, training["training"]["seed"], 125)
    if len(train_ids) != 1130 or len(val_ids) != 125:
        raise ValueError("Frozen TRAIN/validation split changed")
    source_dir = args.source_run.resolve()
    source_manifest = source_dir / "run_manifest.json"
    source_checkpoint = source_dir / "best.pt"
    if (sha256_file(source_manifest) != config["fork_run_manifest_sha256"]
        or sha256_file(source_checkpoint) != config["fork_checkpoint_sha256"]):
        raise ValueError("Frozen source run/checkpoint SHA256 mismatch")
    report = json.loads(source_manifest.read_text(encoding="utf-8"))
    if (report.get("status") != "SUCCESS" or not report.get("training_complete")
        or len(report.get("epochs", [])) != 100
        or report.get("checkpoint", {}).get("epoch") != 71
        or report.get("checkpoint", {}).get("sha256") != config["fork_checkpoint_sha256"]
        or report.get("run_contract", {}).get("variant") != "letterbox"
        or report.get("run_contract", {}).get("validation_ids_sha256") != val_sha
        or report.get("run_contract", {}).get("config_sha256") != sha256_file(SOURCE_CONFIG_PATH)):
        raise ValueError("Source run is not the qualified 100-epoch letterbox run")
    return manifest, train_ids, val_ids, val_sha, source_checkpoint, report


def _contract(config: dict, training: dict, branch: str, architecture: str,
              validation_sha: str, train_ids: list[int]) -> dict:
    return {"branch": branch, "classification": "PRACTICAL_APPROXIMATION",
            "config_sha256": sha256_file(CONFIG_PATH),
            "source_config_sha256": sha256_file(SOURCE_CONFIG_PATH),
            "source_run_manifest_sha256": config["fork_run_manifest_sha256"],
            "source_checkpoint_sha256": config["fork_checkpoint_sha256"],
            "fork_checkpoint_epoch": config["fork_checkpoint_epoch"],
            "transform_source_sha256": sha256_file(TRANSFORM_SOURCE),
            "wavelet_source_sha256": sha256_file(WAVELET_SOURCE),
            "base_model_source_sha256": sha256_file(BASE_MODEL_SOURCE),
            "manifest_sha256": training["train_manifest_sha256"],
            "validation_ids_sha256": validation_sha,
            "train_indices_sha256": sha256("".join(f"{index}\n" for index in train_ids).encode()).hexdigest(),
            "input_shape": [3, 512, 512], "logits_shape": [1, 256, 256],
            "architecture": architecture, "separation_weight": BRANCHES[branch],
            "skip_floor": 0.25, "letterbox": training["variants"]["letterbox"],
            "checkpoint_selection": "best_validation_foreground_iou_tie_earliest_epoch"}


def run(args) -> dict:
    config = _config()
    training = source_config()
    if args.branch not in BRANCHES:
        raise ValueError("Unknown frozen branch")
    manifest, train_ids, val_ids, val_sha, source_checkpoint, source_report = (
        _source(args, config, training))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training was not started")
    seed_everything(training["training"]["seed"], deterministic_algorithms=True)
    model = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False).to(device)
    architecture = model.architecture_identity
    count = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if (count, trainable) != (58155081, 32598049):
        raise RuntimeError("Frozen letterbox model parameter count changed")
    old_training_config = _training_config(training, {"separation_weight": 3.0})
    branch_training_config = _training_config(training, {"separation_weight": BRANCHES[args.branch]})
    if branch_training_config.epochs != 500:
        raise ValueError("Both branches must end after 500 total epochs")
    contract = _contract(config, training, args.branch, architecture, val_sha, train_ids)
    optimizer = build_optimizer(model, branch_training_config)
    if args.check:
        payload = load_checkpoint(source_checkpoint, model=model, optimizer=optimizer,
            expected_architecture=architecture, expected_run_contract=source_report["run_contract"],
            expected_training_config=old_training_config, restore_rng=False)
        if int(payload["epoch"]) != 71:
            raise ValueError("Source checkpoint is not epoch 71")
        summary = {"status": "CHECK_PASS", "branch": args.branch,
                   "source_epoch": 71, "target_epoch": 500,
                   "source_checkpoint_sha256": config["fork_checkpoint_sha256"],
                   "separation_weight": BRANCHES[args.branch],
                   "parameter_count": count, "trainable_parameter_count": trainable,
                   "validation_ids_sha256": val_sha}
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    output_root = args.output_root.resolve()
    if output_root == REPO or REPO in output_root.parents:
        raise ValueError("Training outputs must be outside the repository")
    run_dir = output_root / args.branch
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing branch belongs to a different frozen contract")
    else:
        if any(run_dir.iterdir()):
            raise ValueError("Nonempty branch directory without contract")
        _json(contract_path, contract)
    result_path = run_dir / "run_manifest.json"
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    log_path = run_dir / "epochs.jsonl"
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if (report.get("status") != "SUCCESS" or report.get("run_contract") != contract
            or report.get("total_epochs") != 500 or len(report.get("continuation_epochs", [])) != 429
            or sha256_file(best_path) != report["checkpoint"]["sha256"]):
            raise ValueError("Completed branch provenance mismatch")
        if last_path.exists():
            if sha256_file(last_path) != report["continuation_epochs"][-1]["last_checkpoint_sha256"]:
                raise ValueError("Completed latest checkpoint changed; refusing cleanup")
            last_path.unlink()
        print(f"[{args.branch}] verified completed 500-epoch branch", flush=True)
        return report
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    train_transform = _transform(training, "letterbox", train=True)
    val_transform = _transform(training, "letterbox", train=False)
    root = args.target_root.resolve()
    train_data = Subset(SegmentationDataset(manifest, root, transform=train_transform), train_ids)
    val_data = Subset(SegmentationDataset(manifest, root, transform=val_transform), val_ids)
    loader_args = {"num_workers": training["training"]["num_workers"],
                   "pin_memory": device.type == "cuda"}
    sampler = EpochShuffleSampler(len(train_data), seed=training["training"]["seed"])
    train_loader = DataLoader(train_data, batch_size=branch_training_config.batch_size,
                              sampler=sampler, **loader_args)
    val_loader = DataLoader(val_data, batch_size=branch_training_config.batch_size,
                            shuffle=False, **loader_args)
    records = _epoch_log(log_path)
    if last_path.exists():
        payload = load_checkpoint(last_path, model=model, optimizer=optimizer,
            expected_architecture=architecture, expected_run_contract=contract,
            expected_training_config=branch_training_config, restore_rng=True)
        start_epoch = int(payload["epoch"]) + 1
        if len(records) == start_epoch - 73:
            _append_epoch(log_path, {"epoch": payload["epoch"],
                "global_step": payload["global_step"], "train": payload["train_metrics"],
                "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last_path),
                "recovered_from_last_checkpoint": True})
            records = _epoch_log(log_path)
        if (len(records) != start_epoch - 72
            or records[-1]["epoch"] != start_epoch - 1):
            raise ValueError("Continuation epoch log does not match last checkpoint")
    elif records:
        raise ValueError("Continuation records exist without last checkpoint")
    else:
        # The epoch-100 state was not retained. Both branches fork identically
        # from the available best epoch-71 checkpoint, including optimizer/RNG.
        payload = load_checkpoint(source_checkpoint, model=model, optimizer=optimizer,
            expected_architecture=architecture, expected_run_contract=source_report["run_contract"],
            expected_training_config=old_training_config, restore_rng=True)
        if int(payload["epoch"]) != 71:
            raise ValueError("Source checkpoint is not epoch 71")
        start_epoch = 72
        if not best_path.exists():
            best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
                epoch=71, global_step=int(payload["global_step"]), config=branch_training_config,
                resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
                architecture=architecture, result_class="approximate reproduction",
                run_contract=contract, train_metrics=payload["train_metrics"],
                validation_metrics=payload["validation_metrics"])
            print(f"[{args.branch}] forked epoch 71; initial best SHA256 {best_sha}", flush=True)
    if start_epoch < 72 or start_epoch > 501:
        raise ValueError("Continuation epoch is outside the frozen 72-500 range")
    if not best_path.exists():
        raise ValueError("Missing initial or current best checkpoint")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if (best_payload.get("run_contract") != contract
        or best_payload.get("training_config") != asdict(branch_training_config)):
        raise ValueError("Best branch checkpoint provenance mismatch")
    best_iou = float(best_payload["validation_metrics"]["foreground_iou"])
    best_meta = {"epoch": int(best_payload["epoch"]), "sha256": sha256_file(best_path),
                 "validation": best_payload["validation_metrics"]}
    del best_payload
    started = time.monotonic()
    for epoch in range(start_epoch, 501):
        sampler.set_epoch(epoch - 1)
        train_transform.set_epoch(epoch - 1)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device,
            separation_weight=BRANCHES[args.branch])
        val_metrics = measure_foreground_iou(model, val_loader, device=device,
            probability_threshold=0.5, aggregation="per_image_macro")
        step = (records[-1]["global_step"] if records else int(payload["global_step"])) + train_metrics["steps"]
        last_sha = save_checkpoint(last_path, model=model, optimizer=optimizer,
            epoch=epoch, global_step=step, config=branch_training_config,
            resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
            architecture=architecture, result_class="approximate reproduction",
            run_contract=contract, train_metrics=train_metrics,
            validation_metrics=val_metrics)
        if float(val_metrics["foreground_iou"]) > best_iou:
            best_iou = float(val_metrics["foreground_iou"])
            best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
                epoch=epoch, global_step=step, config=branch_training_config,
                resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
                architecture=architecture, result_class="approximate reproduction",
                run_contract=contract, train_metrics=train_metrics,
                validation_metrics=val_metrics)
            best_meta = {"epoch": epoch, "sha256": best_sha, "validation": val_metrics}
        row = {"epoch": epoch, "global_step": step, "train": train_metrics,
               "validation": val_metrics, "last_checkpoint_sha256": last_sha}
        _append_epoch(log_path, row)
        records.append(row)
        print(f"[{args.branch}] epoch {epoch}/500 "
              f"loss={train_metrics['loss']:.5f} "
              f"val_fgIoU={val_metrics['foreground_iou']:.5f} "
              f"best_fgIoU={best_iou:.5f}", flush=True)
    if len(records) != 429 or best_meta is None:
        raise RuntimeError("Continuation did not reach epoch 500")
    load_checkpoint(best_path, model=model, expected_architecture=architecture,
        expected_run_contract=contract, expected_training_config=branch_training_config)
    geometry = _validation_geometry(model, val_loader, manifest, root, val_transform,
                                     device=device)
    _json(run_dir / "validation_word_geometry.json", geometry)
    report = {"schema_version": 1, "status": "SUCCESS", "training_complete": True,
        "classification": "PRACTICAL_APPROXIMATION", "result_class": "approximate reproduction",
        "run_contract": contract, "source_prefix_epochs": 71,
        "source_completed_epochs_not_used_as_fork": 100,
        "continuation_epoch_range": [72, 500], "total_epochs": 500,
        "train_images": 1130, "validation_images": 125,
        "checkpoint": {"path": "best.pt", "epoch": best_meta["epoch"],
                       "sha256": best_meta["sha256"],
                       "validation_foreground_iou": best_iou},
        "validation_word_geometry": geometry["aggregate"],
        "parameter_count": count, "trainable_parameter_count": trainable,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "continuation_epochs": records}
    _json(result_path, report)
    if sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest checkpoint changed; refusing cleanup")
    last_path.unlink()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error("--output-root is required unless --check is set")
    result = run(args)
    if not args.check:
        print(json.dumps({"status": result["status"], "branch": args.branch,
                          "checkpoint": result["checkpoint"],
                          "validation_word_geometry": result["validation_word_geometry"],
                          "gpu_peak_allocated_bytes": result["gpu_peak_allocated_bytes"],
                          "gpu_peak_reserved_bytes": result["gpu_peak_reserved_bytes"]},
                         indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
