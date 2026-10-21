"""Four isolated practical TRAIN interventions and a matched50-epoch control."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from models.totaltext_partial_encoder import PracticalTotalTextPartialEncoderAGUNet
from models.totaltext_word_ablation import BRANCHES, PracticalTotalTextWordAGUNet
from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_partial_encoder_50ep import (
    _config as previous_config, _contract as previous_contract, training_config as previous_tc,
    selection_key,
)
from scripts.train_totaltext_v4_highres_ablation import _box_iou, _json, _split_indices, _training_config
from scripts.train_totaltext_v4_image_frequency_ablation import (
    _config as source_config, _owner_map, _transform,
)
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_ablation import (
    OwnedSegmentationDataset, WordTrainingTransform, ablation_loss, inference_labels, label_boxes,
)
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    SegmentationTrainingConfig, build_optimizer, load_checkpoint, save_checkpoint, seed_everything,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_word_ablation_50ep_v1.json"


@dataclass(frozen=True)
class PracticalWordTrainingConfig(SegmentationTrainingConfig):
    def __post_init__(self):
        if self.learning_rate != 1e-5 or self.epochs != 50 or self.loss not in BRANCHES:
            raise ValueError("Word ablation schedule/loss must match frozen protocol")
        fields = asdict(self)
        fields.update(learning_rate=1e-4, loss="masked_BCEWithLogitsLoss")
        SegmentationTrainingConfig(**fields)


def _config():
    c = json.loads(CONFIG_PATH.read_text())
    if (c["reproduction_mode"] != "approximate" or c["classification"] != "PRACTICAL_APPROXIMATION"
        or c["branches"] != list(BRANCHES) or c["split"] != "frozen_TRAIN_1130_validation_125"
        or c["additional_epochs"] != 50 or c["encoder"] != "frozen_parameters_and_batchnorm"
        or c["input_size"] != [512, 512] or c["native_mask_size"] != [256, 256]
        or c["probability_threshold"] != 0.65 or c["padding"] != 0
        or c["area_filter"] is not False or c["top10"] is not False or c["ocr_selection"] is not False
        or c["batch_size"] != 8 or c["learning_rate"] != 1e-5
        or c["source_branch"] != "frozen_control" or c["source_additional_epoch"] != 23):
        raise ValueError("Frozen word ablation protocol changed")
    if (not 0 < c["link_threshold"] < 1 or c["link_loss_weight"] < 0 or c["hard_negative_ratio"] <= 0
        or not 0 <= c["zoom_probability"] <= 1 or not c["zoom_fractions"]
        or any(not 0 < v <= 1 for v in c["zoom_fractions"]) or len(c["auxiliary_weights"]) != 2
        or any(v < 0 for v in c["auxiliary_weights"])):
        raise ValueError("Invalid declared ablation parameters")
    return c


def training_config(c, source, branch):
    fields = asdict(_training_config(source, {"separation_weight": c["separation_weight"]}))
    fields.update(epochs=c["additional_epochs"], learning_rate=c["learning_rate"],
                  loss=branch, checkpoint_selection="validation_word_f1")
    return PracticalWordTrainingConfig(**fields)


def _contract(c, source, validation_sha, model):
    if validation_sha != c["validation_ids_sha256"]:
        raise ValueError("Frozen validation split changed")
    files = ("models/totaltext_word_ablation.py", "models/unet_adv.py",
             "segmentation/totaltext_word_ablation.py", "segmentation/transforms.py",
             "segmentation/data.py", "segmentation/training.py",
             "segmentation/totaltext_image_frequency_augment.py", "scripts/train_totaltext_word_ablation_50ep.py")
    return {"classification": "PRACTICAL_APPROXIMATION", "branch": model.branch,
            "architecture": model.architecture_identity, "config_sha256": sha256_file(CONFIG_PATH),
            "source_sha256": {file: sha256_file(REPO / file) for file in files},
            "manifest_sha256": c["train_manifest_sha256"], "validation_ids_sha256": validation_sha,
            "source_checkpoint_sha256": c["source_checkpoint_sha256"],
            "source_run_manifest_sha256": c["source_run_manifest_sha256"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "additional_epochs": c["additional_epochs"], "precision": c["precision"],
            "batch_size": c["batch_size"], "optimizer_initialization": c["optimizer_initialization"]}


def verify_source(directory, c, source, validation_sha):
    checkpoint, report_path = directory / "best.pt", directory / "run_manifest.json"
    if sha256_file(checkpoint) != c["source_checkpoint_sha256"] or sha256_file(report_path) != c["source_run_manifest_sha256"]:
        raise ValueError("Frozen best control checkpoint/report SHA mismatch")
    report = json.loads(report_path.read_text())
    model = PracticalTotalTextPartialEncoderAGUNet(branch="frozen_control")
    expected = previous_contract(previous_config(), source, validation_sha, model)
    if (report.get("status") != "SUCCESS" or not report.get("training_complete")
        or report.get("run_contract") != expected or len(report.get("epochs", [])) != 50
        or report["checkpoint"]["epoch"] != c["source_additional_epoch"]):
        raise ValueError("Source control run provenance changed")
    payload = load_checkpoint(checkpoint, model=model, expected_architecture=model.architecture_identity,
        expected_run_contract=expected, expected_training_config=previous_tc(previous_config(), source))
    if payload["epoch"] != c["source_additional_epoch"]:
        raise ValueError("Source checkpoint epoch mismatch")
    state = model.state_dict()
    # CPU tensors remain alive through the state; do not retain source optimizer.
    return state


def labels_metrics(labels, owners, expected_ids):
    proposals = label_boxes(labels)
    boxes = [p["bbox_native"] for p in proposals]
    if not set(int(v) for v in np.unique(owners) if v > 0).issubset(expected_ids):
        raise ValueError("Unexpected validation word-owner IDs")
    gt_boxes, associations = [], {}
    split = good = lost = 0
    for owner in expected_ids:
        y, x = np.where(owners == owner)
        gt = [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)] if len(y) else None
        gt_boxes.append(gt)
        components = set(int(v) for v in np.unique(labels[owners == owner]) if v > 0)
        split += len(components) >= 2
        lost += gt is None
        good += gt is not None and max((_box_iou(box, gt) for box in boxes), default=0) >= 0.5
        for component in components:
            associations.setdefault(component, set()).add(owner)
    fused = [ownerset for ownerset in associations.values() if len(ownerset) >= 2]
    result = {"gt_words": len(expected_ids), "gt_words_lost_in_native_target": lost,
        "predicted_components": len(boxes), "raw_proposals": len(boxes),
        "gt_with_predicted_bbox_iou_ge_0_5": good,
        "gt_words_split_across_predicted_components": split,
        "predicted_components_touching_multiple_gt_words": len(fused),
        "gt_words_touched_by_fused_components": len(set().union(*fused)) if fused else 0}
    for threshold, name in ((0.25, "25"), (0.5, "50"), (0.75, "75")):
        edges = [(i, j) for i, box in enumerate(boxes) for j, gt in enumerate(gt_boxes)
                 if gt is not None and _box_iou(box, gt) >= threshold]
        result["matched_gt_iou_ge_" + name] = len(maximum_matching(len(boxes), len(gt_boxes), edges))
    return result


def evaluate(model, loader, owners, *, device, config):
    model.eval()
    totals, ious, f1s = {}, [], []
    tp = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            evidence = model.forward_evidence(batch["image"].to(device))
            for i, sample_id in enumerate(batch["sample_id"]):
                raw_owners, expected_ids, canvas = owners[sample_id]
                labels = inference_labels(evidence, canvas, config, i)
                target = batch["mask"][i, 0].numpy().astype(bool)
                valid = batch["validity"][i, 0].numpy().astype(bool)
                reference = raw_owners.copy(); reference[~target | ~valid] = 0
                for name, value in labels_metrics(labels, reference, expected_ids).items():
                    totals[name] = totals.get(name, 0) + int(value)
                foreground = labels > 0
                ti = np.count_nonzero(foreground & target & valid)
                fi = np.count_nonzero(foreground & ~target & valid)
                ni = np.count_nonzero(~foreground & target & valid)
                tp += ti; fp += fi; fn += ni
                ious.append(ti / (ti + fi + ni) if ti + fi + ni else 1.0)
                f1s.append(2 * ti / (2 * ti + fi + ni) if 2 * ti + fi + ni else 1.0)
    if len(ious) != len(loader.dataset):
        raise ValueError("Incomplete validation")
    matched, words, count = totals["matched_gt_iou_ge_50"], totals["gt_words"], totals["raw_proposals"]
    totals.update(foreground_iou=float(np.mean(ious)), foreground_f1_macro=float(np.mean(f1s)),
        foreground_iou_micro=float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        foreground_f1_micro=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        word_recall_iou_50=matched / words if words else 0.0,
        proposal_precision_iou_50=matched / count if count else 0.0,
        proposal_f1_iou_50=2 * matched / (words + count) if words + count else 0.0)
    return totals


def train_epoch(model, loader, optimizer, device, config, epoch):
    model.train()
    count = steps = 0; total = 0.0; ordered = []
    for step, batch in enumerate(loader, 1):
        tensors = {name: batch[name].to(device, non_blocking=device.type == "cuda")
                   for name in ("image", "mask", "validity", "separation", "owners", "word_weights")}
        optimizer.zero_grad(set_to_none=True)
        loss = ablation_loss(model.forward_evidence(tensors["image"]), tensors, model.branch, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite word ablation loss")
        loss.backward(); optimizer.step()
        count += len(tensors["image"]); steps += 1
        total += float(loss.detach()) * len(tensors["image"])
        ordered.extend(str(value) for value in batch["sample_id"])
        if step % 25 == 0 or step == len(loader):
            print(f"[{model.branch}] epoch{epoch}/50 batch{step}/{len(loader)} loss={total/count:.5f}", flush=True)
    if count != len(loader.dataset):
        raise RuntimeError("Incomplete training epoch")
    return {"loss": total/count, "samples": count, "steps": steps,
            "sample_order_sha256": sha256("".join(v + "\n" for v in ordered).encode()).hexdigest()}


def load_population(root, config):
    path = root / config["train_manifest"]
    if sha256_file(path) != config["train_manifest_sha256"]:
        raise ValueError("Frozen TRAIN manifest SHA changed")
    manifest = load_segmentation_manifest(path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Only official Total-Text TRAIN is allowed")
    train_ids, val_ids, val_sha = _split_indices(manifest, config["seed"], 125)
    if len(train_ids) != 1130 or len(val_ids) != 125 or val_sha != config["validation_ids_sha256"]:
        raise ValueError("Frozen TRAIN/validation split changed")
    return manifest, train_ids, val_ids, val_sha


def reconcile_epoch_records(records, payload, latest_sha, epochs):
    """Recover exactly one committed checkpoint; reject changed or non-prefix logs."""
    if any(row.get("epoch") != index + 1 for index, row in enumerate(records)):
        raise ValueError("Epoch records are not a consecutive prefix")
    epoch = int(payload["epoch"])
    if not 1 <= epoch <= epochs:
        raise ValueError("Resume epoch out of range")
    result = list(records)
    if len(result) == epoch - 1:
        result.append({"epoch": epoch, "global_step": payload["global_step"],
            "train": payload["train_metrics"], "validation": payload["validation_metrics"],
            "last_checkpoint_sha256": latest_sha, "recovered_from_checkpoint": True})
    if len(result) != epoch or result[-1].get("last_checkpoint_sha256") != latest_sha:
        raise ValueError("Checkpoint/epoch log mismatch")
    return result


def run(args):
    c, source = _config(), source_config()
    root = args.target_root.resolve()
    manifest, train_ids, val_ids, val_sha = load_population(root, c)
    source_state = verify_source(args.source_run.resolve(), c, source, val_sha)
    seed_everything(c["seed"], deterministic_algorithms=True)
    model = PracticalTotalTextWordAGUNet(args.branch, link_initial_bias=c["link_initial_bias"])
    model.initialize_base_state(source_state); del source_state
    contract, tc = _contract(c, source, val_sha, model), training_config(c, source, args.branch)
    if args.check:
        result = {"status": "PREFLIGHT_PASS", "branch": args.branch,
                  "parameters": contract["parameters"], "trainable_parameters": contract["trainable_parameters"],
                  "source_checkpoint_sha256": c["source_checkpoint_sha256"],
                  "config_sha256": contract["config_sha256"], "train_images": 1130, "validation_images": 125}
        print(json.dumps(result), flush=True)
        return result
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable")
    output = args.output_root.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Training output must be outside repository")
    directory = output / args.branch; directory.mkdir(parents=True, exist_ok=True)
    contract_path = directory / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing run contract differs")
    elif any(directory.iterdir()):
        raise ValueError("Nonempty output has no run contract")
    else:
        _json(contract_path, contract)
    best, latest, result_path = directory / "best.pt", directory / "last.pt", directory / "run_manifest.json"
    records_path = directory / "epochs.json"
    records = json.loads(records_path.read_text())["epochs"] if records_path.exists() else []
    if any(row.get("epoch") != index + 1 for index, row in enumerate(records)):
        raise ValueError("Epoch records are not consecutive")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result.get("status") != "SUCCESS" or result.get("run_contract") != contract
            or len(records) != c["additional_epochs"] or sha256_file(best) != result["checkpoint"]["sha256"]):
            raise ValueError("Completed run integrity mismatch")
        if latest.exists():
            if sha256_file(latest) != records[-1]["last_checkpoint_sha256"]:
                raise ValueError("Latest checkpoint changed")
            latest.unlink()
        print(f"[{args.branch}] reuse {result_path}", flush=True)
        return result
    model.to(device)
    optimizer = build_optimizer(model, tc)
    paths = [root / sample.word_owner_annotation_path for sample in manifest.samples]
    base_train, base_val = _transform(source, "letterbox", train=True), _transform(source, "letterbox", train=False)
    transform = WordTrainingTransform(base_train, owner_paths=paths, zoom=args.branch == "word_zoom",
        zoom_probability=c["zoom_probability"], zoom_fractions=c["zoom_fractions"])
    val_transform = WordTrainingTransform(base_val, owner_paths=paths, zoom=False,
        zoom_probability=c["zoom_probability"], zoom_fractions=c["zoom_fractions"])
    train_dataset = OwnedSegmentationDataset(manifest, root, transform=transform)
    val_dataset = OwnedSegmentationDataset(manifest, root, transform=val_transform, verify_checksums=False)
    train_data, val_data = Subset(train_dataset, train_ids), Subset(val_dataset, val_ids)
    sampler = EpochShuffleSampler(len(train_data), seed=c["seed"])
    options = dict(batch_size=c["batch_size"], num_workers=c["num_workers"], pin_memory=device.type == "cuda")
    train_loader, val_loader = DataLoader(train_data, sampler=sampler, **options), DataLoader(val_data, shuffle=False, **options)
    owners = {}
    for index in val_ids:
        sample = manifest.samples[index]
        labels, ids = _owner_map(sample, root, base_val)
        with Image.open(root / sample.image_path) as image:
            canvas = letterbox_canvas_mask(image.size).numpy()
        owners[sample.sample_id] = labels, ids, canvas
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    started = time.monotonic()
    def save(path, epoch, step, trained, evaluated):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, global_step=step,
            config=tc, resolved_config=CONFIG_PATH.read_text(), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract,
            train_metrics=trained, validation_metrics=evaluated)
    payload = None
    if latest.exists():
        payload = load_checkpoint(latest, model=model, optimizer=optimizer,
            expected_architecture=model.architecture_identity, expected_run_contract=contract,
            expected_training_config=tc, restore_rng=True)
        epoch = int(payload["epoch"])
        reconciled = reconcile_epoch_records(records, payload, sha256_file(latest), c["additional_epochs"])
        if reconciled != records:
            _json(records_path, {"epochs": reconciled})
        records = reconciled
        start, step = epoch + 1, int(payload["global_step"])
    else:
        if records:
            raise ValueError("Resume records exist without latest checkpoint")
        if best.exists():
            payload = load_checkpoint(best, model=model, optimizer=optimizer,
                expected_architecture=model.architecture_identity, expected_run_contract=contract,
                expected_training_config=tc, restore_rng=True)
            if payload["epoch"] != 0:
                raise ValueError("Noninitial best has no latest checkpoint")
        else:
            initial = evaluate(model, val_loader, owners, device=device, config=c)
            if initial["gt_words"] != 983:
                raise ValueError("Validation denominator changed")
            save(best, 0, 0, {}, initial)
        start, step = 1, 0
    baseline = torch.load(best, map_location="cpu", weights_only=False)
    if (baseline.get("run_contract") != contract or baseline.get("training_config") != asdict(tc)
        or baseline.get("architecture") != model.architecture_identity):
        raise ValueError("Best checkpoint provenance changed")
    best_epoch, best_metrics = baseline["epoch"], baseline["validation_metrics"]; del baseline
    if latest.exists() and selection_key(payload["validation_metrics"]) > selection_key(best_metrics):
        best_epoch, best_metrics = payload["epoch"], payload["validation_metrics"]
        save(best, best_epoch, step, payload["train_metrics"], best_metrics)
    del payload
    for epoch in range(start, c["additional_epochs"] + 1):
        sampler.set_epoch(epoch - 1); transform.set_epoch(epoch - 1)
        trained = train_epoch(model, train_loader, optimizer, device, c, epoch)
        metrics = evaluate(model, val_loader, owners, device=device, config=c)
        if metrics["gt_words"] != 983:
            raise ValueError("Validation denominator changed")
        step += trained["steps"]
        latest_sha = save(latest, epoch, step, trained, metrics)
        if selection_key(metrics) > selection_key(best_metrics):
            save(best, epoch, step, trained, metrics); best_epoch, best_metrics = epoch, metrics
        records.append({"epoch": epoch, "global_step": step, "train": trained,
                        "validation": metrics, "last_checkpoint_sha256": latest_sha})
        _json(records_path, {"epochs": records})
        print(f"[{args.branch}] {epoch}/50 F1={metrics['proposal_f1_iou_50']:.5f} "
              f"recall={metrics['word_recall_iou_50']:.5f} fused={metrics['gt_words_touched_by_fused_components']} best={best_epoch}", flush=True)
    result = {"status": "SUCCESS", "training_complete": True, "reproduction_mode": "approximate",
        "classification": "PRACTICAL_APPROXIMATION", "result_class": "approximate reproduction",
        "run_contract": contract, "train_images": 1130, "validation_images": 125, "additional_epochs": 50,
        "checkpoint": {"path": "best.pt", "epoch": best_epoch, "sha256": sha256_file(best)},
        "validation_word_geometry": best_metrics, "epochs": records, "device": str(device),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    _json(result_path, result)
    if len(records) != 50 or sha256_file(latest) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Final resume-checkpoint integrity changed")
    latest.unlink()
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
                failure = {"type": type(error).__name__, "message": str(error), "device": args.device}
                if args.device == "cuda:0" and torch.cuda.is_initialized():
                    failure.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                   gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                _json(directory / "failure.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
