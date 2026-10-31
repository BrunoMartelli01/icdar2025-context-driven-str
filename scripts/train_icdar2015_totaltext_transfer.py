"""Train the isolated approximate ICDAR2015 U-Net transfer; never load TEST/OCR."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import socket
import time

import torch
from torch.utils.data import DataLoader, Subset

from models.icdar2015_unet import PracticalICDAR2015AGUNet
from segmentation.data import file_sha256, load_segmentation_manifest, SegmentationDataset
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_transfer import (
    evaluate, load_config, selection_key, split_indices, transform, validation_references,
)
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    SegmentationTrainingConfig, build_optimizer, load_checkpoint, runtime_environment,
    save_checkpoint, seed_everything, train_one_epoch,
)


REPO = Path(__file__).resolve().parents[1]
# Pin the reused scientific implementations as well as the new entry points.
SOURCES = (
    "models/icdar2015_unet.py", "models/unet_adv.py", "segmentation/icdar2015_transfer.py",
    "segmentation/data.py", "segmentation/transforms.py", "segmentation/training.py",
    "segmentation/evaluate.py",
    "segmentation/train.py", "recognition/scene_text_datasets.py",
    "segmentation/totaltext_word_regions.py", "segmentation/totaltext_image_frequency_augment.py",
    "scripts/train_icdar2015_totaltext_transfer.py", "scripts/prepare_icdar2015_word_regions.py",
    "scripts/train_totaltext_v4_highres_ablation.py",
    "scripts/train_totaltext_v4_image_frequency_ablation.py",
    "recognition/scene_text_v1_evaluation.py",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def training_config(config):
    item = config["training"]
    if item["precision"] != "float32" or item["deterministic_algorithms"] is not True:
        raise ValueError("This transfer requires deterministic float32 execution")
    return SegmentationTrainingConfig(
        epochs=item["epochs"], seed=item["seed"], adam_betas=tuple(item["adam_betas"]),
        adam_epsilon=item["adam_epsilon"], learning_rate_schedule="none",
        checkpoint_selection="validation_word_f1", deterministic_algorithms=True,
        optimizer=item["optimizer"], learning_rate=item["learning_rate"],
        weight_decay=item["weight_decay"], batch_size=item["batch_size"],
        loss="masked_BCEWithLogitsLoss", separation_weight=item["separation_weight"],
    )


def run(args):
    config = load_config(args.config)
    training = training_config(config)
    if args.smoke_steps is not None and args.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive")
    result_class = "smoke-test result" if args.smoke_steps else "approximate reproduction"
    root = args.data_root.resolve()
    manifest_path = root / config["source_manifest"]
    qualification_path = root / "qualification.json"
    qualification = json.loads(qualification_path.read_text())
    if (qualification["source_manifest_sha256"] != config["source_manifest_sha256"]
            or qualification["manifest_sha256"] != file_sha256(manifest_path)):
        raise ValueError("Derived ICDAR2015 target qualification/manifest mismatch")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if (manifest.reproduction_mode != "approximate" or manifest.acquisition is None
            or manifest.acquisition.get("config_sha256") != file_sha256(args.config)):
        raise ValueError("Derived target configuration/provenance differs")
    train_ids, validation_ids = split_indices(manifest, config)
    if ([manifest.samples[i].sample_id for i in train_ids] != qualification["train_ids"]
            or [manifest.samples[i].sample_id for i in validation_ids] != qualification["validation_ids"]):
        raise ValueError("Qualified TRAIN/validation assignment changed")
    if file_sha256(args.encoder_checkpoint) != config["encoder_checkpoint_sha256"]:
        raise ValueError("Pinned ResNet-50 ImageNet V2 checkpoint checksum mismatch")
    contract = {"schema_version": 1, "config": config,
        "config_sha256": file_sha256(args.config), "manifest_sha256": file_sha256(manifest_path),
        "qualification_sha256": file_sha256(qualification_path),
        "source_sha256": {name: file_sha256(REPO / name) for name in SOURCES},
        "train_ids": qualification["train_ids"], "validation_ids": qualification["validation_ids"],
        "paths": {"data_root": str(root), "encoder_checkpoint": str(args.encoder_checkpoint.resolve())},
        "result_class": result_class, "smoke_steps": args.smoke_steps,
        "device": args.device, "precision": "float32", "tf32": False}
    run_dir = args.output_dir.resolve()
    contract_path = run_dir / "run_contract.json"
    if run_dir.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume to verify and resume this exact run")
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
            raise ValueError("Resume run contract differs")
        if (run_dir / "run_manifest.json").exists():
            report = json.loads((run_dir / "run_manifest.json").read_text())
            if (report.get("status") == "SUCCESS" and report.get("run_contract") == contract
                    and file_sha256(run_dir / "best.pt") == report["checkpoint"]["sha256"]):
                details = json.loads((run_dir / "validation_word_geometry.json").read_text())
                if details["aggregate"] != report["validation"]:
                    raise ValueError("Saved validation summary differs")
                for row in details["per_image"]:
                    if file_sha256(run_dir / "validation_masks" / row["intermediate_file"]) != row["intermediate_sha256"]:
                        raise ValueError("Saved validation intermediate checksum differs")
                return report
            raise ValueError("Existing final report is incompatible")
    else:
        if args.resume:
            raise ValueError("Cannot resume a missing run directory")
        run_dir.mkdir(parents=True)
        write_json(contract_path, contract)
        write_json(run_dir / "resolved_config.json", {**config, "execution": contract})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; no CPU substitution")
    seed_everything(training.seed, deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    model = PracticalICDAR2015AGUNet(skip_floor=config["skip_floor"], encoder_pretrained=False)
    encoder_state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(encoder_state, strict=True)
    del encoder_state
    model.to(device)
    optimizer = build_optimizer(model, training)
    train_transform, val_transform = transform(config, augment=True), transform(config, augment=False)
    train_data = SegmentationDataset(manifest, root, transform=train_transform)
    val_data = SegmentationDataset(manifest, root, transform=val_transform)
    # A smoke is an explicitly partial run, never a shortened full result.
    evaluated_ids = (validation_ids[:args.smoke_steps * training.batch_size]
                     if args.smoke_steps else validation_ids)
    train_subset = Subset(train_data, train_ids)
    sampler = EpochShuffleSampler(len(train_subset), seed=training.seed)
    loader_options = {"batch_size": training.batch_size,
        "num_workers": config["training"]["num_workers"], "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(training.seed)}
    train_loader = DataLoader(train_subset, sampler=sampler, **loader_options)
    val_loader = DataLoader(Subset(val_data, evaluated_ids), shuffle=False, **loader_options)
    references = validation_references(manifest, root, evaluated_ids, val_transform)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    records, best_metrics, best_epoch, global_step = [], None, 0, 0
    if args.resume:
        payload = load_checkpoint(last_path, model=model, optimizer=optimizer,
            expected_architecture=model.architecture_identity, expected_run_contract=contract,
            expected_training_config=training, restore_rng=True)
        records = json.loads(payload["training_selection_metrics"]["records_json"])
        best_epoch = int(payload["training_selection_metrics"]["best_epoch"])
        best_metrics = payload["training_selection_metrics"]["best_metrics"]
        global_step = payload["global_step"]
        if not records or records[-1]["epoch"] != payload["epoch"]:
            raise ValueError("Resume history and checkpoint epoch differ")
        if not best_path.is_file():
            raise ValueError("Resume requires the previously selected best checkpoint")
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if (best_payload.get("run_contract") != contract or best_payload["epoch"] != best_epoch
                or best_payload["validation_metrics"] != best_metrics):
            raise ValueError("Selected checkpoint disagrees with resume state")
        del best_payload, payload
    start_time = time.monotonic()
    final_epoch = 1 if args.smoke_steps else training.epochs
    for epoch in range(len(records) + 1, final_epoch + 1):
        sampler.set_epoch(epoch - 1)
        train_transform.set_epoch(epoch - 1)
        metrics = train_one_epoch(model, train_loader, optimizer, device=device,
            max_steps=args.smoke_steps, separation_weight=training.separation_weight)
        validation = evaluate(model, val_loader, references, device=device,
            probability_threshold=config["probability_threshold"])["aggregate"]
        global_step += metrics["steps"]
        records.append({"epoch": epoch, "global_step": global_step,
                        "train": metrics, "validation": validation})
        checkpoint_args = dict(model=model, optimizer=optimizer, epoch=epoch,
            global_step=global_step, config=training, resolved_config=json.dumps(contract, sort_keys=True),
            architecture=model.architecture_identity, result_class=result_class,
            run_contract=contract, train_metrics=metrics, validation_metrics=validation)
        if best_metrics is None or selection_key(validation, epoch) > selection_key(best_metrics, best_epoch):
            best_epoch, best_metrics = epoch, validation
            save_checkpoint(best_path, **checkpoint_args)
        save_checkpoint(last_path, **checkpoint_args, training_selection_metrics={
            "records_json": json.dumps(records, sort_keys=True),
            "best_epoch": best_epoch, "best_metrics": best_metrics})
        write_json(run_dir / "epochs.json", records)
        print(f"[ICDAR2015 transfer] epoch {epoch}/{final_epoch} loss={metrics['loss']:.6f} "
              f"val_word_F1={validation['proposal_f1_iou_50']:.6f} "
              f"val_IoU={validation['foreground_iou']:.6f}", flush=True)
    load_checkpoint(best_path, model=model, expected_architecture=model.architecture_identity,
        expected_run_contract=contract, expected_training_config=training)
    # Preserve selected per-sample probabilities, masks and metric denominators.
    detailed = evaluate(model, val_loader, references, device=device,
        probability_threshold=config["probability_threshold"], output_dir=run_dir / "validation_masks")
    if detailed["aggregate"] != best_metrics:
        raise RuntimeError("Selected checkpoint validation replay differs")
    write_json(run_dir / "validation_word_geometry.json", detailed)
    report = {"schema_version": 1, "status": "SUCCESS", "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "result_class": result_class,
        "training_complete": not bool(args.smoke_steps) and len(records) == training.epochs,
        "configured_epochs": training.epochs, "completed_epochs": len(records),
        "run_contract": contract, "epochs": records,
        "train_population": len(train_ids), "validation_population": len(validation_ids),
        "evaluated_images": len(evaluated_ids), "validation": detailed["aggregate"],
        "checkpoint": {"path": "best.pt", "epoch": best_epoch, "sha256": file_sha256(best_path)},
        "runtime": runtime_environment(), "hostname": socket.gethostname(), "device": str(device),
        "packages": {name: version(name) for name in ("torchvision", "numpy", "Pillow", "scipy", "PyYAML")},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "oom": False, "precision": "float32", "batch_size": training.batch_size,
        "input_shape": [training.batch_size, 3, 512, 512],
        "output_shape": [training.batch_size, 1, 256, 256],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds_this_invocation": time.monotonic() - start_time}
    write_json(run_dir / "run_manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args)
    except torch.cuda.OutOfMemoryError as error:
        write_json(args.output_dir / "failure.json", {"status": "FAILED", "oom": True,
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "result_class": "smoke-test result" if args.smoke_steps else "approximate reproduction",
            "architecture": PracticalICDAR2015AGUNet.PRACTICAL_IDENTITY,
            "error": str(error), "device": args.device, "precision": "float32", "batch_size": 8,
            "input_shape": [8, 3, 512, 512], "output_shape": [8, 1, 256, 256],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        raise
    print(json.dumps({key: report[key] for key in (
        "status", "result_class", "training_complete", "evaluated_images",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
