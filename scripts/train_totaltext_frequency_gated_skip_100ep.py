"""TRAIN-only 100-epoch feature-wavelet gated-skip Total-Text ablation.

This is a separate PRACTICAL_APPROXIMATION; it never changes the paper model,
the frozen letterbox checkpoint, PARSeq, or TEST data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from models.totaltext_frequency_gated_skip import PracticalTotalTextFrequencyGatedSkipAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import (
    _append_epoch, _epoch_log, _json, _split_indices, _training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    CONFIG_PATH as SOURCE_CONFIG_PATH, _config as source_config, _transform,
    _validation_geometry,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    build_optimizer, load_checkpoint, measure_foreground_iou, save_checkpoint,
    seed_everything, train_one_epoch,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_frequency_gated_skip_100ep_v1.json"


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (value.get("reproduction_mode") != "approximate"
        or value.get("classification") != "PRACTICAL_APPROXIMATION"
        or value.get("split") != "frozen_TRAIN_1130_validation_125"
        or value.get("epochs") != 100 or value.get("seed") != 202503
        or value.get("batch_size") != 8
        or value.get("input_size") != [512, 512]
        or value.get("native_mask_size") != [256, 256]
        or value.get("model") != "feature_haar_gated_shallow_skip"
        or value.get("training_probability_threshold") != 0.5
        or value.get("comparison_probability_threshold") != 0.65
        or value.get("checkpoint_selection") != "best_validation_macro_foreground_iou_tie_earliest_epoch"
        or value.get("postprocessing") != "native256_8_connected_no_area_no_top10_no_padding"
        or sha256_file(SOURCE_CONFIG_PATH) != value.get("source_training_config_sha256")):
        raise ValueError("Frozen practical frequency-skip configuration changed")
    return value


def _contract(config: dict, source: dict, validation_sha: str,
              model: PracticalTotalTextFrequencyGatedSkipAGUNet) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (total, trainable) != (58179722, 32622690):
        raise ValueError(f"Frequency-skip parameter count changed: {total}/{trainable}")
    return {"classification": "PRACTICAL_APPROXIMATION",
            "config_sha256": sha256_file(CONFIG_PATH),
            "source_config_sha256": sha256_file(SOURCE_CONFIG_PATH),
            "model_source_sha256": sha256_file(REPO / "models/totaltext_frequency_gated_skip.py"),
            "base_model_source_sha256": sha256_file(REPO / "models/unet_adv.py"),
            "trainer_source_sha256": sha256_file(Path(__file__)),
            "training_primitives_sha256": sha256_file(REPO / "segmentation/training.py"),
            "transform_source_sha256": sha256_file(
                REPO / "segmentation/totaltext_image_frequency_augment.py"),
            "manifest_sha256": source["train_manifest_sha256"],
            "validation_ids_sha256": validation_sha,
            "architecture": model.architecture_identity,
            "parameters": total, "trainable_parameters": trainable,
            "input_shape": [3, 512, 512], "logits_shape": [1, 256, 256],
            "epochs": config["epochs"], "seed": config["seed"],
            "checkpoint_selection": config["checkpoint_selection"],
            "intervention": config["model"],
            "comparison_probability_threshold": config["comparison_probability_threshold"]}


def run(args) -> dict:
    config = _config()
    source = source_config()
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if (sha256_file(manifest_path) != config["train_manifest_sha256"]
        or source["train_manifest_sha256"] != config["train_manifest_sha256"]):
        raise ValueError("Qualified Total-Text TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This ablation accepts official TRAIN only")
    train_ids, validation_ids, validation_sha = _split_indices(manifest, 202503, 125)
    if len(train_ids) != 1130 or len(validation_ids) != 125:
        raise ValueError("Frozen Total-Text TRAIN split changed")
    if args.check:
        seed_everything(202503, deterministic_algorithms=True)
        model = PracticalTotalTextFrequencyGatedSkipAGUNet(encoder_pretrained=False)
        contract = _contract(config, source, validation_sha, model)
        result = {"status": "PREFLIGHT_PASS", "images_train": 1130,
                  "images_validation": 125, "parameter_count": contract["parameters"],
                  "trainable_parameter_count": contract["trainable_parameters"],
                  "validation_ids_sha256": validation_sha}
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; training was not started")
    device = torch.device(args.device)
    output_root = args.output_root.resolve()
    if output_root == REPO or REPO in output_root.parents:
        raise ValueError("Training output must be outside repository")
    run_dir = output_root / "frequency_gated_skip_100ep"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(202503, deterministic_algorithms=True)
    model = PracticalTotalTextFrequencyGatedSkipAGUNet(skip_floor=0.25,
                                                       encoder_pretrained=True).to(device)
    contract = _contract(config, source, validation_sha, model)
    training_config = replace(_training_config(source, {"separation_weight": 3.0}),
                              epochs=100)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Existing run contract differs")
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
        print(f"RESULT {result_path} SHA256 {sha256_file(result_path)}", flush=True)
        return result
    train_transform = _transform(source, "letterbox", train=True)
    val_transform = _transform(source, "letterbox", train=False)
    train_data = Subset(SegmentationDataset(manifest, root, transform=train_transform), train_ids)
    val_data = Subset(SegmentationDataset(manifest, root, transform=val_transform), validation_ids)
    loader_options = {"num_workers": source["training"]["num_workers"],
                      "pin_memory": device.type == "cuda"}
    sampler = EpochShuffleSampler(len(train_data), seed=202503)
    train_loader = DataLoader(train_data, batch_size=8, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, **loader_options)
    optimizer = build_optimizer(model, training_config)
    records = _epoch_log(run_dir / "epochs.jsonl")
    if last_path.exists():
        payload = load_checkpoint(last_path, model=model, optimizer=optimizer,
            expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=training_config,
            restore_rng=True)
        start_epoch = int(payload["epoch"]) + 1
        if len(records) == start_epoch - 2:
            _append_epoch(run_dir / "epochs.jsonl", {"epoch": start_epoch - 1,
                "global_step": payload["global_step"], "train": payload["train_metrics"],
                "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last_path),
                "recovered_from_last_checkpoint": True})
            records = _epoch_log(run_dir / "epochs.jsonl")
        if len(records) != start_epoch - 1:
            raise ValueError("Epoch log and last checkpoint disagree")
    elif records or best_path.exists():
        raise ValueError("Training artifacts without latest checkpoint")
    else:
        start_epoch = 1
    if best_path.exists():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        if best_payload.get("run_contract") != contract:
            raise ValueError("Best checkpoint contract mismatch")
        best_epoch = int(best_payload["epoch"])
        best_metrics = best_payload["validation_metrics"]
        best_iou = float(best_metrics["foreground_iou"])
        best_sha = sha256_file(best_path)
        del best_payload
    else:
        best_epoch, best_metrics, best_iou, best_sha = 0, {}, -1.0, None
    if last_path.exists() and float(payload["validation_metrics"]["foreground_iou"]) > best_iou:
        best_epoch = int(payload["epoch"])
        best_metrics = payload["validation_metrics"]
        best_iou = float(best_metrics["foreground_iou"])
        best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
            epoch=best_epoch, global_step=int(payload["global_step"]), config=training_config,
            resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
            architecture=model.architecture_identity, result_class="approximate reproduction",
            run_contract=contract, train_metrics=payload["train_metrics"],
            validation_metrics=best_metrics)
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    started = time.monotonic()
    for epoch in range(start_epoch, 101):
        sampler.set_epoch(epoch - 1)
        train_transform.set_epoch(epoch - 1)
        trained = train_one_epoch(model, train_loader, optimizer, device=device,
                                  separation_weight=3.0)
        evaluated = measure_foreground_iou(model, val_loader, device=device,
            probability_threshold=0.5, aggregation="per_image_macro")
        step = (records[-1]["global_step"] if records else 0) + trained["steps"]
        last_sha = save_checkpoint(last_path, model=model, optimizer=optimizer,
            epoch=epoch, global_step=step, config=training_config,
            resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
            architecture=model.architecture_identity, result_class="approximate reproduction",
            run_contract=contract, train_metrics=trained, validation_metrics=evaluated)
        score = float(evaluated["foreground_iou"])
        if score > best_iou:
            best_sha = save_checkpoint(best_path, model=model, optimizer=optimizer,
                epoch=epoch, global_step=step, config=training_config,
                resolved_config=CONFIG_PATH.read_text(encoding="utf-8"),
                architecture=model.architecture_identity, result_class="approximate reproduction",
                run_contract=contract, train_metrics=trained, validation_metrics=evaluated)
            best_epoch, best_metrics, best_iou = epoch, evaluated, score
        row = {"epoch": epoch, "global_step": step, "train": trained,
               "validation": evaluated, "last_checkpoint_sha256": last_sha}
        _append_epoch(run_dir / "epochs.jsonl", row)
        records.append(row)
        print(f"[frequency_gated_skip] {epoch}/100 loss={trained['loss']:.5f} "
              f"val_fgIoU={score:.5f} best={best_epoch}", flush=True)
    if len(records) != 100 or best_epoch < 1 or best_sha is None:
        raise RuntimeError("Incomplete 100-epoch ablation")
    load_checkpoint(best_path, model=model,
        expected_architecture=model.architecture_identity,
        expected_run_contract=contract, expected_training_config=training_config)
    geometry = _validation_geometry(model, val_loader, manifest, root, val_transform,
                                    device=device)
    _json(run_dir / "validation_word_geometry.json", geometry)
    report = {"schema_version": 1, "status": "SUCCESS",
              "classification": "PRACTICAL_APPROXIMATION",
              "result_class": "approximate reproduction",
              "run_contract": contract, "train_images": 1130, "validation_images": 125,
              "checkpoint": {"path": "best.pt", "epoch": best_epoch,
                             "sha256": best_sha,
                             "validation_foreground_iou": best_iou},
              "validation_word_geometry": geometry["aggregate"],
              "parameter_count": contract["parameters"],
              "trainable_parameter_count": contract["trainable_parameters"],
              "device": str(device), "elapsed_seconds": time.monotonic() - started,
              "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                           if device.type == "cuda" else None),
              "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                          if device.type == "cuda" else None)}
    _json(result_path, report)
    if sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest checkpoint changed before cleanup")
    last_path.unlink()
    print(f"RESULT {result_path} SHA256 {sha256_file(result_path)}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--check", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
