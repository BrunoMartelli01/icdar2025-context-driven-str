"""Restart compact-v2 with held-out word metrics and at most four checkpoint files.

APPROXIMATE only. Original v2 masks/model/loss remain unchanged. The new TRAIN
split excludes125 images for checkpoint selection. No TEST, crops or OCR job.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from models.unet_adv import AuthorProtocolAGUNet
from segmentation.config import load_segmentation_config
from segmentation.data import SegmentationDataset, file_sha256, load_segmentation_manifest
from segmentation.totaltext_word_kernel import native_owner_map
from segmentation.train import EpochShuffleSampler, _training_config, _transform
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything, train_one_epoch
from segmentation.word_checkpoint_metrics import bbox_metrics, counts, native_metrics, selection_records
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
from scripts.crop_clip_runtime import write_json
from scripts.train_totaltext_v4_highres_ablation import _split_indices

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/totaltext_compact_v2_word_selection_v1.json"
CODE = ("scripts/train_totaltext_compact_v2_word_selection.py", "segmentation/word_checkpoint_metrics.py",
    "models/unet_adv.py", "segmentation/config.py", "segmentation/data.py", "segmentation/transforms.py",
    "segmentation/train.py", "segmentation/training.py", "segmentation/totaltext_word_kernel.py",
    "localize/repository_contours.py", "localize/native_repository_contours.py",
    "recognition/scene_text_v1_evaluation.py", "scripts/evaluate_totaltext_v1_findcontours_parseq.py",
    "scripts/train_totaltext_v4_highres_ablation.py")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prune_to_winners(directory, records):
    winners = selection_records(records)
    keep = {r["checkpoint"]: r for r in winners.values()}
    known = {r["checkpoint"]: r for r in records}
    paths = list(directory.glob("epoch-*.pt"))
    if any(r["checkpoint"] != f"epoch-{r['epoch']:04d}.pt" for r in records):
        raise ValueError("Epoch/checkpoint basename mismatch")
    if any(p.is_symlink() or p.name not in known for p in paths):
        raise ValueError("Unknown/symlinked training weights; refusing retention cleanup")
    for name, row in keep.items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("A selected checkpoint is missing or changed")
    # Check every obsolete file before performing any unlink.
    for p in paths:
        if p.name not in keep and file_sha256(p) != known[p.name]["checkpoint_sha256"]:
            raise ValueError("An obsolete checkpoint changed; refusing deletion")
    removed = []
    for p in paths:
        if p.name not in keep:
            p.unlink()
            removed.append(p.name)
    if len(list(directory.glob("epoch-*.pt"))) > 4:
        raise RuntimeError("Four-checkpoint retention limit violated")
    return winners, removed


def references(manifest, root, indices, dataset, transform):
    result = {}
    for i in indices:
        sample, item = manifest.samples[i], dataset[i]
        with Image.open(root / sample.word_owner_annotation_path) as im:
            ids = [int(v) for v in np.unique(np.asarray(im)) if v > 0]
            owners = native_owner_map(im.convert("I"), transform, i, item["mask"], item["validity"])
        path = Path(item["image_path"])
        with Image.open(path) as rgb:
            w, h = rgb.size
        gt = _gt(path.parent.parent / "annotations" / (sample.sample_id.partition(":")[2] + ".txt"), (w, h))
        if len(ids) != len(gt):
            raise ValueError("Original care-owner/word denominator differs")
        result[sample.sample_id] = {"owners": owners, "expected_ids": ids, "gt": gt, "image_hw": (h, w)}
    return result


def evaluate(model, batches, refs, device, cfg):
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch["image"].to(device))
            if tuple(logits.shape[1:]) != (1, 128, 128) or not torch.isfinite(logits).all():
                raise ValueError("Invalid native128 validation logits")
            foreground = (logits.sigmoid() >= cfg["probability_threshold"]).cpu().numpy()[:, 0]
            for offset, sid in enumerate(batch["sample_id"]):
                ref = refs[sid]
                target = batch["mask"][offset, 0].numpy().astype(bool)
                valid = batch["validity"][offset, 0].numpy().astype(bool)
                values = native_metrics(foreground[offset], target, valid, ref["owners"], ref["expected_ids"],
                    connectivity=cfg["connectivity"], iou_threshold=cfg["mask_iou_threshold"])
                values.update(bbox_metrics(foreground[offset], ref["image_hw"], ref["gt"],
                    min_area=cfg["bbox_min_area"], padding=cfg["bbox_padding"], nmax=cfg["bbox_nmax"],
                    iou_threshold=cfg["bbox_iou_threshold"]))
                rows.append({"sample_id": sid, **values})
    if len(rows) != len(refs) or {r["sample_id"] for r in rows} != set(refs):
        raise ValueError("Incomplete/duplicate validation population")
    aggregate = {"images": len(rows), "foreground_iou": float(np.mean([r["pixel_iou"] for r in rows]))}
    for kind in ("native_instance", "bbox_detection"):
        aggregate[kind] = counts(*(sum(r[kind][name] for r in rows) for name in ("tp", "predicted", "gt")))
    aggregate.update(native_word_f1=aggregate["native_instance"]["f1"], bbox_word_f1=aggregate["bbox_detection"]["f1"])
    for name in ("missing_gt_at_native", "fused_components_touching_multiple_words",
                 "words_touched_by_fused_components", "words_split_across_components"):
        aggregate[name] = sum(r[name] for r in rows)
    return {"aggregate": aggregate, "per_image": rows}


def prepare(args):
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("Explicit isolated APPROXIMATE configuration required")
    cfg = json.loads(args.config.read_text())
    if (args.mode != "approximate" or cfg["reproduction_mode"] != "approximate"
            or cfg["claim_label"] != "approximate reproduction" or cfg["table_5_claim"] is not False
            or cfg["retained_criteria"] != ["native_word_f1", "bbox_word_f1", "foreground_iou", "last"]):
        raise ValueError("Unsupported declared compact-v2 experiment")
    source_path = REPO / cfg["source_config"]
    if file_sha256(source_path) != cfg["source_config_sha256"]:
        raise ValueError("Original compact-v2 scientific protocol changed")
    source = load_segmentation_config(source_path)
    path = REPO / cfg["train_manifest"]
    if file_sha256(path) != cfg["train_manifest_sha256"]:
        raise ValueError("Qualified compact-word TRAIN manifest changed")
    manifest = load_segmentation_manifest(path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Only official TRAIN may enter model selection")
    tc = replace(_training_config(source), checkpoint_selection="validation_word_f1")
    train, val, split_sha = _split_indices(manifest, tc.seed, cfg["validation_images"])
    if split_sha != cfg["validation_ids_sha256"]:
        raise ValueError("Frozen held-out split differs")
    if file_sha256(args.encoder_checkpoint) != cfg["encoder_checkpoint_sha256"]:
        raise ValueError("Pinned local ResNet50 IMAGENET1K_V2 bytes changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the GPU host; no training started")
    seed_everything(tc.seed, deterministic_algorithms=True)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tt = _transform(source.transforms, seed=tc.seed, augmentation_enabled=True)
    vt = _transform(source.transforms, seed=tc.seed, augmentation_enabled=False)
    training = SegmentationDataset(manifest, args.data_root, transform=tt, verify_checksums=True)
    validation = SegmentationDataset(manifest, args.data_root, transform=vt, verify_checksums=False)
    ref = references(manifest, args.data_root, val, validation, vt)
    model = AuthorProtocolAGUNet(encoder_pretrained=False)
    model.encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.float().to(device)
    contract = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "config": cfg, "config_sha256": file_sha256(args.config),
        "source_config": OmegaConf.to_container(source, resolve=True),
        "source_config_sha256": cfg["source_config_sha256"], "training_config": asdict(tc),
        "manifest_sha256": cfg["train_manifest_sha256"], "validation_ids_sha256": split_sha,
        "train_ids": [manifest.samples[i].sample_id for i in train],
        "validation_ids": [manifest.samples[i].sample_id for i in val],
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()), "data_root": str(args.data_root.resolve()),
        "code_sha256": {n: file_sha256(REPO / n) for n in CODE}, "device": str(device),
        "hardware_deviation": "CPU FP32 engineering smoke" if device.type == "cpu" else "single-GPU FP32; actual GPU recorded",
        "software": {"torch": torch.__version__, "numpy": np.__version__},
        "deterministic_algorithms": True, "tf32": False, "test_loaded": False}
    return cfg, tc, device, model, training, validation, tt, ref, train, val, contract


def run(args):
    cfg, tc, device, model, training, validation, transform, refs, train, val, contract = prepare(args)
    optimizer = build_optimizer(model, tc)
    if args.check:
        n = tc.batch_size if device.type == "cuda" else 1
        batch = next(iter(DataLoader(Subset(training, train[:n]), batch_size=n, num_workers=0)))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        trained = train_one_epoch(model, [batch], optimizer, device=device, separation_weight=tc.separation_weight)
        ids = val[:2]
        subset_refs = {training.manifest.samples[i].sample_id: refs[training.manifest.samples[i].sample_id] for i in ids}
        measured = evaluate(model, DataLoader(Subset(validation, ids), batch_size=1, num_workers=0), subset_refs, device, cfg)
        receipt = {"status": "CHECK_PASS", "claim_label": "approximate reproduction", "training_started": False,
            "train_images": len(train), "validation_images": len(val), "validation_ids_sha256": contract["validation_ids_sha256"],
            "engineering_training_steps": trained["steps"], "batch_size": n, "input_shape": [n, 3, 256, 256],
            "precision": "float32", "device": str(device), "validation_smoke": measured["aggregate"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        if args.check_report:
            write_json(args.check_report, receipt)
        print(json.dumps(receipt), flush=True)
        return
    if device.type != "cuda":
        raise ValueError("Full training requires explicitly selected CUDA; CPU is engineering check only")
    root = args.output_root.resolve()
    if root.is_relative_to(REPO) or root.is_relative_to(args.data_root.resolve() / cfg["gt_relative_root"]):
        raise ValueError("Weights require an isolated external run directory")
    directory = root / cfg["run_name"]
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    cp = directory / "run_contract.json"
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError("Existing run contract differs; choose a fresh output root")
    write_json(cp, contract)
    log = directory / "epochs.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    existing = sorted(checkpoints.glob("epoch-*.pt"))
    if existing:
        latest = existing[-1]
        payload = load_checkpoint(latest, model=model, optimizer=optimizer,
            expected_architecture=model.architecture_identity, expected_run_contract=contract,
            expected_training_config=tc, restore_rng=True)
        ep = int(payload["epoch"])
        if len(records) == ep - 1:
            row = {"epoch": ep, "global_step": payload["global_step"], "train": payload["train_metrics"],
                "validation": payload["validation_metrics"], "checkpoint": latest.name,
                "checkpoint_sha256": file_sha256(latest), "recovered_from_checkpoint": True}
            append(log, row)
            records.append(row)
        if len(records) != ep or records[-1]["checkpoint_sha256"] != file_sha256(latest):
            raise ValueError("Latest checkpoint/log provenance differs")
        prune_to_winners(checkpoints, records)
        del payload
        gc.collect()
    elif records:
        raise ValueError("Epoch records exist without resumable checkpoint")
    sampler = EpochShuffleSampler(len(train), seed=tc.seed)
    options = {"num_workers": cfg["num_workers"], "pin_memory": True}
    tl = DataLoader(Subset(training, train), batch_size=tc.batch_size, sampler=sampler, **options)
    vl = DataLoader(Subset(validation, val), batch_size=tc.batch_size, shuffle=False, **options)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for ep in range(len(records) + 1, tc.epochs + 1):
        if shutil.disk_usage(directory).free < cfg["minimum_free_bytes"]:
            raise RuntimeError("Insufficient disk for one atomic checkpoint; no epoch started")
        sampler.set_epoch(ep - 1)
        transform.set_epoch(ep - 1)
        trained = train_one_epoch(model, tl, optimizer, device=device, separation_weight=tc.separation_weight)
        evaluated = evaluate(model, vl, refs, device, cfg)
        measured = evaluated["aggregate"]
        write_json(directory / "validation" / f"epoch-{ep:04d}.json", evaluated)
        step = (records[-1]["global_step"] if records else 0) + trained["steps"]
        path = checkpoints / f"epoch-{ep:04d}.pt"
        digest = save_checkpoint(path, model=model, optimizer=optimizer, epoch=ep, global_step=step,
            config=tc, resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract,
            train_metrics=trained, validation_metrics=measured)
        row = {"epoch": ep, "global_step": step, "train": trained, "validation": measured,
               "checkpoint": path.name, "checkpoint_sha256": digest}
        append(log, row)
        records.append(row)
        winners = selection_records(records)
        write_json(directory / "checkpoint_selection.json", {name: r for name, r in winners.items()})
        _, removed = prune_to_winners(checkpoints, records)
        append(directory / "retention.jsonl", {"epoch": ep, "removed": removed,
            "kept": sorted({r["checkpoint"] for r in winners.values()})})
        receipt = {"status": "SUCCESS" if ep == tc.epochs else "IN_PROGRESS", "training_complete": ep == tc.epochs,
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction", "result_class": "approximate reproduction",
            "run_contract_sha256": file_sha256(cp), "completed_epochs": ep, "configured_epochs": tc.epochs,
            "train_images": len(train), "validation_images": len(val), "checkpoint_selection": winners,
            "device": str(device), "gpu": torch.cuda.get_device_name(device), "precision": "float32", "batch_size": tc.batch_size,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "elapsed_seconds_this_invocation": time.monotonic() - started, "test_loaded": False}
        write_json(directory / "run_manifest.json", receipt)
        print(f"[compact-v2] {ep}/{tc.epochs} loss={trained['loss']:.5f} nativeWordF1={measured['native_word_f1']:.5f} "
              f"bboxWordF1={measured['bbox_word_f1']:.5f} fgIoU={measured['foreground_iou']:.5f} kept={len(set(r['epoch'] for r in winners.values()))}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["approximate"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--encoder-checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--check-report", type=Path)
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.check_report:
            cuda_initialized = args.device == "cuda:0" and torch.cuda.is_initialized()
            write_json(args.check_report, {"status": "FAILED", "error_type": type(error).__name__,
                "error": str(error), "out_of_memory": isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)),
                "stage": "preflight" if args.check else "training", "device": args.device, "precision": "float32",
                "model_set": ["AuthorProtocolAGUNet/frozenResNet50"],
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0) if cuda_initialized else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0) if cuda_initialized else None})
        raise


if __name__ == "__main__":
    main()
