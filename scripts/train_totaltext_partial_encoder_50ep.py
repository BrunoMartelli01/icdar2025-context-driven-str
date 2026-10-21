"""Matched 50-epoch TRAIN-only continuations; no TEST or OCR in selection."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.totaltext_partial_encoder import (
    PracticalTotalTextPartialEncoderAGUNet, build_partial_encoder_optimizer,
)
from models.unet_adv import PracticalTotalTextHighResAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import (
    _append_epoch, _epoch_log, _json, _split_indices, _training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    CONFIG_PATH as SOURCE_CONFIG_PATH, _config as source_config,
    _owner_map, _transform, proposal_metrics,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    SegmentationTrainingConfig, load_checkpoint, save_checkpoint, seed_everything, train_one_epoch,
)
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_partial_encoder_50ep_v1.json"
BRANCHES = ("frozen_control", "partial_encoder")


@dataclass(frozen=True)
class PracticalPartialEncoderTrainingConfig(SegmentationTrainingConfig):
    """Isolated practical LR contract; the paper-valued base validator is unchanged."""

    def __post_init__(self):
        if self.learning_rate != 1e-5 or self.epochs != 50:
            raise ValueError("Practical partial-encoder schedule must be LR1e-5/50 epochs")
        # Validate all remaining existing fields without relaxing the shared
        # author-protocol config. The effective LR stays1e-5 in saved metadata.
        paper_fields = asdict(self)
        paper_fields["learning_rate"] = 1e-4
        SegmentationTrainingConfig(**paper_fields)


def _config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (config.get("reproduction_mode") != "approximate"
        or config.get("classification") != "PRACTICAL_APPROXIMATION"
        or config.get("branches") != list(BRANCHES)
        or config.get("additional_epochs") != 50
        or config.get("fork_checkpoint_epoch") != 71
        or config.get("probability_threshold") != 0.65
        or config.get("decoder_learning_rate") != 1e-5
        or config.get("encoder_learning_rate") != 1e-6
        or config.get("seed") != 202503 or config.get("batch_size") != 8
        or config.get("input_size") != [512, 512]
        or config.get("native_mask_size") != [256, 256]
        or config.get("padding") != 0 or config.get("area_filter") is not False
        or config.get("top10") is not False or config.get("separation_weight") != 3.0
        or config.get("split") != "frozen_TRAIN_1130_validation_125"
        or config.get("checkpoint_selection") != "validation_native_word_F1_then_fewer_fused_words_then_recall_then_foreground_IoU_then_earliest"
        or config.get("optimizer_initialization") != "fresh_Adam_both_branches"
        or config.get("encoder_batchnorm") != "all_affine_and_running_statistics_frozen"
        or sha256_file(SOURCE_CONFIG_PATH) != config.get("source_training_config_sha256")):
        raise ValueError("Frozen partial-encoder configuration changed")
    return config


def training_config(config, source):
    fields = asdict(_training_config(source, {"separation_weight": config["separation_weight"]}))
    fields.update(epochs=config["additional_epochs"], learning_rate=config["decoder_learning_rate"],
                  checkpoint_selection="validation_word_f1")
    return PracticalPartialEncoderTrainingConfig(**fields)


def _contract(config, source, validation_sha, model):
    if validation_sha != config["validation_ids_sha256"]:
        raise ValueError("Frozen validation population changed")
    return {"classification": "PRACTICAL_APPROXIMATION", "branch": model.branch,
            "architecture": model.architecture_identity,
            "config_sha256": sha256_file(CONFIG_PATH),
            "source_config_sha256": sha256_file(SOURCE_CONFIG_PATH),
            "model_source_sha256": sha256_file(REPO / "models/totaltext_partial_encoder.py"),
            "base_model_source_sha256": sha256_file(REPO / "models/unet_adv.py"),
            "trainer_source_sha256": sha256_file(Path(__file__)),
            "training_primitives_sha256": sha256_file(REPO / "segmentation/training.py"),
            "transform_source_sha256": sha256_file(REPO / "segmentation/totaltext_image_frequency_augment.py"),
            "manifest_sha256": source["train_manifest_sha256"],
            "validation_ids_sha256": validation_sha,
            "source_checkpoint_sha256": config["fork_checkpoint_sha256"],
            "source_run_manifest_sha256": config["fork_run_manifest_sha256"],
            "source_epoch": config["fork_checkpoint_epoch"],
            "additional_epochs": config["additional_epochs"],
            "optimizer_initialization": config["optimizer_initialization"],
            "encoder_learning_rate": config["encoder_learning_rate"],
            "decoder_learning_rate": config["decoder_learning_rate"],
            "checkpoint_selection": config["checkpoint_selection"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "precision": "float32", "batch_size": config["batch_size"]}


def verify_source(source_run, config, source, validation_sha):
    report_path, checkpoint = source_run / "run_manifest.json", source_run / "best.pt"
    if (sha256_file(report_path) != config["fork_run_manifest_sha256"]
        or sha256_file(checkpoint) != config["fork_checkpoint_sha256"]):
        raise ValueError("Frozen epoch71 source run/checkpoint SHA mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "SUCCESS" or not report.get("training_complete")
        or len(report.get("epochs", [])) != 100
        or report.get("checkpoint", {}).get("epoch") != config["fork_checkpoint_epoch"]
        or report.get("checkpoint", {}).get("sha256") != config["fork_checkpoint_sha256"]
        or report.get("run_contract", {}).get("variant") != "letterbox"
        or report.get("run_contract", {}).get("validation_ids_sha256") != validation_sha
        or report.get("run_contract", {}).get("config_sha256") != sha256_file(SOURCE_CONFIG_PATH)):
        raise ValueError("Source report is not qualified letterbox epoch71")
    return report, checkpoint


def initialize_source(model, checkpoint, report, source, *, restore_rng):
    base = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False)
    payload = load_checkpoint(checkpoint, model=base, expected_architecture=base.architecture_identity,
        expected_run_contract=report["run_contract"],
        expected_training_config=_training_config(source, {"separation_weight": 3.0}),
        restore_rng=restore_rng)
    if payload["epoch"] != 71:
        raise ValueError("Unexpected source epoch")
    model.load_state_dict(base.state_dict(), strict=True)


def selection_key(metrics):
    return (float(metrics["proposal_f1_iou_50"]),
            -int(metrics["gt_words_touched_by_fused_components"]),
            float(metrics["word_recall_iou_50"]), float(metrics["foreground_iou"]))


def evaluate(model, loader, owner_cache, *, device, threshold):
    """One batch8 forward per sample; native GT/validity and micro geometry."""
    model.eval()
    totals, ious = {}, []
    with torch.inference_mode():
        for batch in loader:
            pred = (torch.sigmoid(model(batch["image"].to(device))) >= threshold).cpu().numpy()[:, 0]
            for offset, sample_id in enumerate(batch["sample_id"]):
                raw_owners, expected_ids, canvas = owner_cache[sample_id]
                prediction = pred[offset] & canvas
                target = batch["mask"][offset, 0].numpy().astype(bool)
                valid = batch["validity"][offset, 0].numpy().astype(bool)
                union = np.count_nonzero((prediction | target) & valid)
                ious.append(np.count_nonzero(prediction & target & valid) / union if union else 1.0)
                owners = raw_owners.copy()
                owners[~target | ~valid] = 0
                metrics = proposal_metrics(prediction, owners, expected_owner_ids=expected_ids)
                for key, value in metrics.items():
                    if isinstance(value, (int, np.integer)):
                        totals[key] = totals.get(key, 0) + int(value)
    if len(ious) != len(loader.dataset):
        raise ValueError("Incomplete validation")
    tp, words, proposals = totals["matched_gt_iou_ge_50"], totals["gt_words"], totals["raw_proposals"]
    totals.update(foreground_iou=float(np.mean(ious)),
        word_recall_iou_50=tp / words if words else 0.0,
        proposal_precision_iou_50=tp / proposals if proposals else 0.0,
        proposal_f1_iou_50=2 * tp / (words + proposals) if words + proposals else 0.0)
    return totals


def run(args):
    config, source = _config(), source_config()
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified TRAIN manifest SHA changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Official TRAIN only; TEST is prohibited")
    train_ids, val_ids, val_sha = _split_indices(manifest, config["seed"], 125)
    if len(train_ids) != 1130 or len(val_ids) != 125:
        raise ValueError("Frozen split changed")
    report, checkpoint = verify_source(args.source_run.resolve(), config, source, val_sha)
    seed_everything(config["seed"], deterministic_algorithms=True)
    model = PracticalTotalTextPartialEncoderAGUNet(branch=args.branch)
    contract = _contract(config, source, val_sha, model)
    tc = training_config(config, source)
    if args.check:
        initialize_source(model, checkpoint, report, source, restore_rng=False)
        result = {"status": "PREFLIGHT_PASS", "branch": args.branch, "contract": contract,
                  "train_images": 1130, "validation_images": 125}
        print(json.dumps(result, indent=2), flush=True)
        return result
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback to CPU")
    output = args.output_root.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Training output must be outside repository")
    run_dir = output / args.branch
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing contract differs; refusing overwrite")
    elif any(run_dir.iterdir()):
        raise ValueError("Nonempty run without contract")
    else:
        _json(contract_path, contract)
    result_path, best, latest = run_dir / "run_manifest.json", run_dir / "best.pt", run_dir / "last.pt"
    records = _epoch_log(run_dir / "epochs.jsonl")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result.get("status") != "SUCCESS" or result.get("run_contract") != contract
            or len(records) != config["additional_epochs"]
            or sha256_file(best) != result["checkpoint"]["sha256"]):
            raise ValueError("Completed run integrity mismatch")
        if latest.exists():
            if sha256_file(latest) != records[-1]["last_checkpoint_sha256"]:
                raise ValueError("Completed latest checkpoint changed; refusing cleanup")
            latest.unlink()
        print(f"[{args.branch}] reuse completed run {result_path}", flush=True)
        return result
    model.to(device)
    optimizer = build_partial_encoder_optimizer(model, tc, config["encoder_learning_rate"])
    tt, vt = _transform(source, "letterbox", train=True), _transform(source, "letterbox", train=False)
    train_data = Subset(SegmentationDataset(manifest, root, transform=tt), train_ids)
    val_data = Subset(SegmentationDataset(manifest, root, transform=vt), val_ids)
    sampler = EpochShuffleSampler(len(train_data), seed=config["seed"])
    options = {"batch_size": config["batch_size"], "num_workers": source["training"]["num_workers"],
               "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, sampler=sampler, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    owners = {}
    for index in val_ids:
        sample = manifest.samples[index]
        labels, expected = _owner_map(sample, root, vt)
        with Image.open(root / sample.image_path) as image:
            canvas = letterbox_canvas_mask(image.size).numpy().astype(bool)
        owners[sample.sample_id] = labels, expected, canvas
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    started = time.monotonic()
    def save(path, epoch, step, trained, evaluated):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, global_step=step,
            config=tc, resolved_config=CONFIG_PATH.read_text(), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract,
            train_metrics=trained, validation_metrics=evaluated)
    if latest.exists():
        payload = load_checkpoint(latest, model=model, optimizer=optimizer,
            expected_architecture=model.architecture_identity, expected_run_contract=contract,
            expected_training_config=tc, restore_rng=True)
        epoch = int(payload["epoch"])
        if not 1 <= epoch <= config["additional_epochs"]:
            raise ValueError("Resume epoch outside 1..50")
        if len(records) == epoch - 1:
            _append_epoch(run_dir / "epochs.jsonl", {"epoch": epoch, "global_step": payload["global_step"],
                "train": payload["train_metrics"], "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(latest), "recovered_from_last_checkpoint": True})
            records = _epoch_log(run_dir / "epochs.jsonl")
        if len(records) != epoch or records[-1]["epoch"] != epoch:
            raise ValueError("Resume checkpoint/log mismatch")
        start, step = epoch + 1, int(payload["global_step"])
    else:
        if records:
            raise ValueError("Epoch records without resume checkpoint")
        if best.exists():
            payload = load_checkpoint(best, model=model, optimizer=optimizer,
                expected_architecture=model.architecture_identity, expected_run_contract=contract,
                expected_training_config=tc, restore_rng=True)
            if payload["epoch"] != 0:
                raise ValueError("Missing latest checkpoint for noninitial best")
        else:
            initialize_source(model, checkpoint, report, source, restore_rng=True)
            metrics = evaluate(model, val_loader, owners, device=device, threshold=config["probability_threshold"])
            if metrics["gt_words"] != 983:
                raise ValueError("Validation GT denominator changed")
            save(best, 0, 0, {}, metrics)
        start, step = 1, 0
    best_payload = torch.load(best, map_location="cpu", weights_only=False)
    if (best_payload.get("run_contract") != contract
        or best_payload.get("architecture") != model.architecture_identity
        or best_payload.get("training_config") != asdict(tc)
        or not 0 <= int(best_payload["epoch"]) <= config["additional_epochs"]):
        raise ValueError("Best checkpoint contract mismatch")
    best_epoch, best_metrics = int(best_payload["epoch"]), best_payload["validation_metrics"]
    del best_payload
    if latest.exists() and selection_key(payload["validation_metrics"]) > selection_key(best_metrics):
        best_epoch, best_metrics = int(payload["epoch"]), payload["validation_metrics"]
        save(best, best_epoch, step, payload["train_metrics"], best_metrics)
    for epoch in range(start, config["additional_epochs"] + 1):
        augmentation_epoch = config["fork_checkpoint_epoch"] + epoch - 1
        sampler.set_epoch(augmentation_epoch)
        tt.set_epoch(augmentation_epoch)
        trained = train_one_epoch(model, train_loader, optimizer, device=device,
                                  separation_weight=config["separation_weight"])
        metrics = evaluate(model, val_loader, owners, device=device, threshold=config["probability_threshold"])
        if metrics["gt_words"] != 983:
            raise ValueError("Validation GT denominator changed")
        step += trained["steps"]
        latest_sha = save(latest, epoch, step, trained, metrics)
        if selection_key(metrics) > selection_key(best_metrics):
            save(best, epoch, step, trained, metrics)
            best_epoch, best_metrics = epoch, metrics
        record = {"epoch": epoch, "global_step": step, "train": trained, "validation": metrics,
                  "last_checkpoint_sha256": latest_sha}
        _append_epoch(run_dir / "epochs.jsonl", record)
        records.append(record)
        print(f"[{args.branch}] {epoch}/50 loss={trained['loss']:.5f} "
              f"word_F1={metrics['proposal_f1_iou_50']:.5f} recall={metrics['word_recall_iou_50']:.5f} "
              f"fused={metrics['gt_words_touched_by_fused_components']} best={best_epoch}", flush=True)
    if len(records) != config["additional_epochs"]:
        raise ValueError("Incomplete additional 50 epochs")
    result = {"status": "SUCCESS", "training_complete": True,
        "reproduction_mode": "approximate", "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "approximate reproduction", "run_contract": contract,
        "train_images": 1130, "validation_images": 125, "additional_epochs": 50,
        "checkpoint": {"path": "best.pt", "epoch": best_epoch,
                       "source_epoch": 71, "sha256": sha256_file(best)},
        "validation_word_geometry": best_metrics, "epochs": records, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "elapsed_seconds_this_invocation": time.monotonic() - started}
    _json(result_path, result)
    if sha256_file(latest) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest hash changed; refusing retention cleanup")
    latest.unlink()  # validated task-owned latest; best remains persistent.
    print(f"RESULT {result_path} SHA256 {sha256_file(result_path)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error("--output-root required for training")
    try:
        run(args)
    except Exception as error:
        if not args.check and args.output_root is not None:
            directory = args.output_root.resolve() / args.branch
            if directory.is_dir() and REPO not in directory.parents:
                _json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
