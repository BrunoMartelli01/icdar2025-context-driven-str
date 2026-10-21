"""One isolated 100-epoch Total-Text word-kernel TRAIN/validation experiment."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_letterbox_word_f1_separator_head import (
    _validation_reference, balanced_separator_bce, selection_key,
)
from scripts.train_totaltext_v4_highres_ablation import (
    _append_epoch, _epoch_log, _json, _split_indices, _training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    CONFIG_PATH as SOURCE_CONFIG_PATH, _config as source_config, _transform,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_word_kernel import (
    WordKernelDataset, expand_word_kernels, instance_geometry, make_word_kernels,
)
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    build_optimizer, load_checkpoint, save_checkpoint, seed_everything,
    separation_weighted_bce_with_logits,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_word_kernel_100ep_v1.json"


def _config():
    config = json.loads(CONFIG_PATH.read_text())
    expected = {"reproduction_mode": "approximate", "classification": "PRACTICAL_APPROXIMATION",
                "split": "frozen_TRAIN_1130_validation_125", "epochs": 100,
                "seed": 202503, "batch_size": 8, "num_workers": 2,
                "input_size": [512, 512], "native_mask_size": [256, 256],
                "skip_floor": 0.25, "foreground_threshold": 0.65,
                "kernel_threshold": 0.5, "kernel_distance_fraction": 0.5,
                "kernel_loss_weight": 1.0, "foreground_separation_weight": 3.0,
                "checkpoint_selection": "validation_word_f1_then_fewer_fusions_then_recall_then_fg_iou_then_earliest",
                "postprocessing": "8_connected_nuclei_simultaneous_BFS_unseeded_CC_fallback_instance_boxes_pad0_no_area_no_top10"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen word-kernel protocol changed")
    if sha256_file(SOURCE_CONFIG_PATH) != config["source_training_config_sha256"]:
        raise ValueError("Source letterbox configuration changed")
    return config


def training_config(source):
    return replace(_training_config(source, {"separation_weight": 3.0}),
                   epochs=100, checkpoint_selection="validation_word_f1")


def _contract(config, source, validation_sha, model):
    if validation_sha != config["validation_ids_sha256"]:
        raise ValueError("Frozen validation IDs differ")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (total, trainable) != (58155146, 32598114):
        raise ValueError("Word-kernel architecture count changed")
    dependencies = ["models/totaltext_word_kernel.py", "models/unet_adv.py",
                    "segmentation/totaltext_word_kernel.py", "segmentation/data.py",
                    "segmentation/transforms.py", "segmentation/totaltext_image_frequency_augment.py",
                    "segmentation/training.py", "scripts/train_totaltext_word_kernel_100ep.py",
                    "scripts/train_totaltext_letterbox_word_f1_separator_head.py",
                    "scripts/train_totaltext_v4_highres_ablation.py",
                    "scripts/train_totaltext_v4_image_frequency_ablation.py",
                    "recognition/scene_text_v1_evaluation.py"]
    return {"classification": "PRACTICAL_APPROXIMATION", "config_sha256": sha256_file(CONFIG_PATH),
            "source_config_sha256": sha256_file(SOURCE_CONFIG_PATH),
            "sources_sha256": {name: sha256_file(REPO / name) for name in dependencies},
            "manifest_sha256": config["train_manifest_sha256"],
            "validation_ids_sha256": validation_sha, "architecture": model.architecture_identity,
            "parameters": total, "trainable_parameters": trainable,
            "epochs": 100, "seed": 202503, "input_shape": [3, 512, 512],
            "logits_shape": [2, 256, 256], "selection": config["checkpoint_selection"]}


def native_instances(logits, canvas, config):
    if logits.shape != (2, 256, 256) or canvas.shape != (256, 256):
        raise ValueError("Expected two native256 maps and letterbox canvas")
    probability = torch.sigmoid(logits).detach().cpu().numpy()
    foreground = (probability[0] >= config["foreground_threshold"]) & canvas
    kernel = (probability[1] >= config["kernel_threshold"]) & canvas
    labels, audit = expand_word_kernels(foreground, kernel)
    return labels, audit, probability


def train_epoch(model, batches, optimizer, device, config):
    model.train()
    total = foreground_total = kernel_total = count = steps = 0
    ids = []
    for batch in batches:
        images, target, validity, separator, kernel = [batch[k].to(device)
            for k in ("image", "mask", "validity", "separation", "kernel")]
        if torch.any(kernel > target) or torch.any(kernel > validity):
            raise ValueError("Kernel target outside valid word foreground")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != (len(images), 2, 256, 256):
            raise ValueError("Unexpected two-head native output")
        foreground_loss = separation_weighted_bce_with_logits(logits[:, :1], target,
            validity, separator, separation_weight=config["foreground_separation_weight"])
        kernel_loss = balanced_separator_bce(logits[:, 1:2], kernel, validity)
        loss = foreground_loss + config["kernel_loss_weight"] * kernel_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite word-kernel loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(images)
        foreground_total += float(foreground_loss.detach()) * len(images)
        kernel_total += float(kernel_loss.detach()) * len(images)
        count += len(images)
        steps += 1
        ids.extend(batch["sample_id"])
    if not count:
        raise ValueError("Empty training loader")
    return {"loss": total / count, "foreground_loss": foreground_total / count,
            "kernel_loss": kernel_total / count, "samples": count, "steps": steps,
            "sample_order_sha256": sha256("".join(f"{x}\n" for x in ids).encode()).hexdigest()}


def evaluate(model, batches, references, device, config):
    totals, ious, rows = {}, [], []
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch["image"].to(device))
            for i, sample_id in enumerate(batch["sample_id"]):
                owners, expected, canvas = references[sample_id]
                target = batch["mask"][i, 0].numpy().astype(bool)
                validity = batch["validity"][i, 0].numpy().astype(bool)
                owners = owners.copy()
                owners[~target | ~validity] = 0
                labels, audit, _ = native_instances(logits[i], canvas, config)
                metrics = {**instance_geometry(labels, owners, expected), **audit}
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + value
                prediction = labels > 0
                union = np.count_nonzero((target | prediction) & validity)
                intersection = np.count_nonzero(target & prediction & validity)
                ious.append(intersection / union if union else 1.0)
                rows.append({"sample_id": sample_id, **metrics})
    tp, words, proposals = totals["matched_gt_iou_ge_50"], totals["gt_words"], totals["raw_proposals"]
    precision = tp / proposals if proposals else 0.0
    recall = tp / words if words else 0.0
    totals.update(foreground_iou=float(np.mean(ious)), proposal_precision_iou_50=precision,
                  word_recall_iou_50=recall,
                  proposal_f1_iou_50=2 * tp / (proposals + words) if proposals + words else 0.0)
    return {"aggregate": totals, "per_image": rows, "images": len(rows)}


def qualify_targets(manifest, root, source, config):
    transform = _transform(source, "letterbox", train=False)
    dataset = WordKernelDataset(SegmentationDataset(manifest, root, transform=transform),
                               distance_fraction=config["kernel_distance_fraction"])
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        owners = item["owners_native"].numpy()
        kernels, details = make_word_kernels(owners,
            distance_fraction=config["kernel_distance_fraction"])
        with Image.open(root / manifest.samples[i].word_owner_annotation_path) as opened:
            original_words = int(np.count_nonzero(np.unique(np.asarray(opened)) > 0))
        rows.append({"sample_id": item["sample_id"], "original_words": original_words,
                     "native_words": len(details), "kernel_words": len(details),
                     "native_kernel_sha256": sha256(kernels.astype('<i4').tobytes()).hexdigest(),
                     "discarded_core_fragments": sum(x["core_fragments_before_selection"] - 1 for x in details)})
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"[kernel targets] {i + 1}/{len(dataset)}", flush=True)
    return {"status": "PASS", "split": "TRAIN_only", "images": len(rows),
            "config_sha256": sha256_file(CONFIG_PATH),
            "manifest_sha256": config["train_manifest_sha256"],
            "target_source_sha256": sha256_file(REPO / "segmentation/totaltext_word_kernel.py"),
            "native_words": sum(r["native_words"] for r in rows),
            "kernel_words": sum(r["kernel_words"] for r in rows),
            "words_lost_before_kernel_construction": sum(r["original_words"] - r["native_words"] for r in rows),
            "per_image": rows}


def run(args):
    config, source = _config(), source_config()
    root, output = args.target_root.resolve(), args.output_root.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Outputs must be outside repository")
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Only official Total-Text TRAIN is allowed")
    train_ids, val_ids, val_sha = _split_indices(manifest, config["seed"], 125)
    if val_sha != config["validation_ids_sha256"]:
        raise ValueError("Frozen split changed")
    audit_path = output / "target-qualification.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if (audit.get("status") != "PASS" or audit.get("images") != 1255
            or audit.get("config_sha256") != sha256_file(CONFIG_PATH)
            or audit.get("target_source_sha256") != sha256_file(REPO / "segmentation/totaltext_word_kernel.py")
            or audit.get("manifest_sha256") != config["train_manifest_sha256"]):
            raise ValueError("Cached target qualification differs")
        # Instantiate checksum-verifying dataset even when qualification is cached.
        SegmentationDataset(manifest, root, transform=_transform(source, "letterbox", train=False))
    else:
        _json(audit_path, qualify_targets(manifest, root, source, config))
    seed_everything(config["seed"], deterministic_algorithms=True)
    model = PracticalTotalTextWordKernelAGUNet(encoder_pretrained=False)
    contract = _contract(config, source, val_sha, model)
    contract["target_qualification_sha256"] = sha256_file(audit_path)
    if args.preflight_only:
        result = {"status": "PREFLIGHT_PASS", "train_images": len(train_ids),
                  "validation_images": len(val_ids), "parameters": contract["parameters"],
                  "extra_parameters": 65, "target_qualification": str(audit_path)}
        print(json.dumps(result, indent=2), flush=True)
        return result
    del model
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no training started")
    run_dir = output / "word_kernel_100ep"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Run contract differs; refusing overwrite")
    elif any(run_dir.iterdir()):
        raise ValueError("Nonempty run without matching contract")
    else:
        _json(contract_path, contract)
    result_path, last_path, best_path = [run_dir / name for name in ("run_manifest.json", "last.pt", "best.pt")]
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if (result.get("status") != "SUCCESS" or result.get("run_contract") != contract
            or sha256_file(best_path) != result["checkpoint"]["sha256"]):
            raise ValueError("Completed run differs")
        if last_path.exists():
            records = _epoch_log(run_dir / "epochs.jsonl")
            if len(records) != 100 or sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
                raise ValueError("Completed latest checkpoint is inconsistent")
            last_path.unlink()
        print(f"RESULT {result_path} SHA256 {sha256_file(result_path)}", flush=True)
        return result
    seed_everything(config["seed"], deterministic_algorithms=True)
    model = PracticalTotalTextWordKernelAGUNet(encoder_pretrained=True).to(device)
    transforms = [_transform(source, "letterbox", train=x) for x in (True, False)]
    datasets = [WordKernelDataset(SegmentationDataset(manifest, root, transform=t)) for t in transforms]
    sampler = EpochShuffleSampler(len(train_ids), seed=config["seed"])
    options = {"num_workers": config["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(datasets[0], train_ids), batch_size=8, sampler=sampler, **options)
    val_loader = DataLoader(Subset(datasets[1], val_ids), batch_size=8, shuffle=False, **options)
    references = _validation_reference(manifest, root, val_ids, transforms[1])
    tc = training_config(source)
    optimizer = build_optimizer(model, tc)
    records = _epoch_log(run_dir / "epochs.jsonl")
    start, payload = 1, None
    if last_path.exists():
        payload = load_checkpoint(last_path, model=model, optimizer=optimizer,
            expected_architecture=model.architecture_identity, expected_run_contract=contract,
            expected_training_config=tc, restore_rng=True)
        start = int(payload["epoch"]) + 1
        if len(records) == start - 2:
            _append_epoch(run_dir / "epochs.jsonl", {"epoch": start - 1,
                "global_step": payload["global_step"], "train": payload["train_metrics"],
                "validation": payload["validation_metrics"],
                "last_checkpoint_sha256": sha256_file(last_path), "recovered_from_checkpoint": True})
            records = _epoch_log(run_dir / "epochs.jsonl")
        if len(records) != start - 1:
            raise ValueError("Epoch log/latest checkpoint disagree")
    elif records or best_path.exists():
        raise ValueError("Training artifacts without resume checkpoint")
    best_epoch, best_metrics, best_key, best_sha = 0, {}, None, None
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best.get("run_contract") != contract:
            raise ValueError("Best checkpoint contract differs")
        best_epoch, best_metrics = int(best["epoch"]), best["validation_metrics"]
        best_key, best_sha = selection_key(best_metrics, best_epoch), sha256_file(best_path)
        del best
    def save(path, epoch, step, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch,
            global_step=step, config=tc, resolved_config=CONFIG_PATH.read_text(),
            architecture=model.architecture_identity, result_class="approximate reproduction",
            run_contract=contract, train_metrics=trained, validation_metrics=metrics)
    if payload is not None:
        key = selection_key(payload["validation_metrics"], int(payload["epoch"]))
        if best_key is None or key > best_key:
            best_epoch, best_metrics, best_key = int(payload["epoch"]), payload["validation_metrics"], key
            best_sha = save(best_path, best_epoch, payload["global_step"], payload["train_metrics"], best_metrics)
        del payload
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    started = time.monotonic()
    for epoch in range(start, config["epochs"] + 1):
        sampler.set_epoch(epoch - 1)
        transforms[0].set_epoch(epoch - 1)
        trained = train_epoch(model, train_loader, optimizer, device, config)
        metrics = evaluate(model, val_loader, references, device, config)["aggregate"]
        if metrics["gt_words"] != 983:
            raise RuntimeError("Validation GT denominator drift")
        step = (records[-1]["global_step"] if records else 0) + trained["steps"]
        last_sha = save(last_path, epoch, step, trained, metrics)
        key = selection_key(metrics, epoch)
        if best_key is None or key > best_key:
            best_epoch, best_metrics, best_key = epoch, metrics, key
            best_sha = save(best_path, epoch, step, trained, metrics)
        row = {"epoch": epoch, "global_step": step, "train": trained,
               "validation": metrics, "last_checkpoint_sha256": last_sha}
        _append_epoch(run_dir / "epochs.jsonl", row)
        records.append(row)
        print(f"[word kernel] {epoch}/100 loss={trained['loss']:.5f} "
              f"F1={metrics['proposal_f1_iou_50']:.5f} recall={metrics['word_recall_iou_50']:.5f} "
              f"fused_words={metrics['gt_words_touched_by_fused_components']} best={best_epoch}", flush=True)
    if len(records) != 100 or best_sha is None:
        raise RuntimeError("Incomplete experiment")
    load_checkpoint(best_path, model=model, expected_architecture=model.architecture_identity,
                    expected_run_contract=contract, expected_training_config=tc)
    final = evaluate(model, val_loader, references, device, config)
    if final["aggregate"] != best_metrics:
        raise RuntimeError("Selected checkpoint validation drift")
    _json(run_dir / "validation_word_geometry.json", final)
    result = {"status": "SUCCESS", "classification": "PRACTICAL_APPROXIMATION",
              "result_class": "approximate reproduction", "run_contract": contract,
              "train_images": 1130, "validation_images": 125,
              "checkpoint": {"epoch": best_epoch, "path": "best.pt", "sha256": best_sha},
              "validation_word_geometry": best_metrics, "elapsed_seconds": time.monotonic() - started,
              "device": str(device), "batch_size": 8, "dtype": "float32",
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    _json(result_path, result)
    if sha256_file(last_path) != records[-1]["last_checkpoint_sha256"]:
        raise ValueError("Latest checkpoint changed")
    last_path.unlink()
    print(f"RESULT {result_path} SHA256 {sha256_file(result_path)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        output = args.output_root.resolve()
        if output != REPO and REPO not in output.parents:
            _json(output / "failure.json", {"status": "FAILED",
                "classification": "PRACTICAL_APPROXIMATION", "device": args.device,
                "type": type(error).__name__, "message": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
