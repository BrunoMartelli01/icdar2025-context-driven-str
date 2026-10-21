"""APPROXIMATE V3 training with the V2 split/protocol and bounded checkpoint retention."""
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
from segmentation.guarded_word_dataset import make_audited_guarded_dataset
from collections import Counter
import gzip
from scripts.train_totaltext_compact_v2_word_selection import append, prune_to_winners, evaluate
from segmentation.train import EpochShuffleSampler, _training_config, _transform
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything, train_one_epoch
from segmentation.word_checkpoint_metrics import bbox_metrics, counts, native_metrics, selection_records
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
from scripts.crop_clip_runtime import write_json
from scripts.train_totaltext_v4_highres_ablation import _split_indices

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/guarded_v3_training_v1.json"
CODE = ("scripts/train_guarded_v3.py", "scripts/train_totaltext_compact_v2_word_selection.py",
    "segmentation/guarded_word_gt.py", "segmentation/guarded_word_dataset.py", "segmentation/word_checkpoint_metrics.py",
    "models/unet_adv.py", "segmentation/config.py", "segmentation/data.py", "segmentation/transforms.py",
    "segmentation/train.py", "segmentation/training.py", "segmentation/totaltext_word_kernel.py",
    "localize/repository_contours.py", "localize/native_repository_contours.py",
    "recognition/scene_text_v1_evaluation.py", "scripts/evaluate_totaltext_v1_findcontours_parseq.py",
    "scripts/train_totaltext_v4_highres_ablation.py")



def references(manifest, root, indices, dataset, transform):
    result={}
    for i in indices:
        sample,item=manifest.samples[i],dataset[i]
        audit=json.loads(item['gt_guard_audit_json'])
        ids=[w['owner'] for w in audit['per_word']]
        owners=item['native_owners'].numpy().copy()
        if not np.array_equal(owners>0,item['mask'][0].numpy()>0):
            raise ValueError('Validation references must use final V3 owners')
        path=Path(item['image_path'])
        with Image.open(path) as im:w,h=im.size
        gt=_gt(path.parent.parent/'annotations'/(sample.sample_id.partition(':')[2]+'.txt'),(w,h))
        if len(ids)!=len(gt):raise ValueError('Original care owner/word denominator differs')
        result[sample.sample_id]={'owners':owners,'expected_ids':ids,'gt':gt,'image_hw':(h,w)}
    return result


def guarded_batches(batches, path, totals):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for batch in batches:
            if any(v!='guarded-word-targets-v3' for v in batch['target_variant']):
                raise ValueError('Non-V3 targets entered V3 training')
            for raw in batch['gt_guard_audit_json']:
                row=json.loads(raw);totals.update(row['stats']);stream.write(raw+'\n')
            yield batch


def prepare(args):
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("Explicit isolated APPROXIMATE configuration required")
    cfg = json.loads(args.config.read_text())
    if (args.mode != "approximate" or cfg["reproduction_mode"] != "approximate"
            or cfg["claim_label"] != "approximate reproduction" or cfg["table_5_claim"] is not False
            or cfg["retained_criteria"] != ["native_word_f1", "bbox_word_f1", "foreground_iou", "last"]):
        raise ValueError("Unsupported declared V3 experiment")
    if cfg.get('target_variant')!='guarded-word-targets-v3':
        raise ValueError('Explicit V3 target variant required')
    recipe=REPO/cfg['guard_recipe']
    if file_sha256(recipe)!=cfg['guard_recipe_sha256']:
        raise ValueError('Qualified V3 recipe changed')
    # This qualification is source provenance, not an input target cache.
    qualification=REPO/cfg['gt_qualification']
    if file_sha256(qualification)!=cfg['gt_qualification_sha256']:
        raise ValueError('V3 target qualification changed')
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
    training=make_audited_guarded_dataset(recipe,augmentation_enabled=True,verify_checksums=True)
    validation=make_audited_guarded_dataset(recipe,augmentation_enabled=False,verify_checksums=False)
    tt,vt=training.transform,validation.transform
    if tt.base.base_seed!=tc.seed or training.source.data_root!=args.data_root.resolve():
        raise ValueError('V3 seed or data root differs from matched V2 protocol')
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
        "deterministic_algorithms": True, "tf32": False, "test_loaded": False, "target_variant":"guarded-word-targets-v3",
        "guard_recipe_sha256":cfg["guard_recipe_sha256"], "guard_recipe_resolved":json.loads(recipe.read_text())}
    return cfg, tc, device, model, training, validation, tt, ref, train, val, contract

def run(args):
    cfg, tc, device, model, training, validation, transform, refs, train, val, contract = prepare(args)
    optimizer = build_optimizer(model, tc)
    if args.check:
        n = tc.batch_size if device.type == "cuda" else 1
        batch = next(iter(DataLoader(Subset(training, train[:n]), batch_size=n, num_workers=cfg["num_workers"])))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        trained = train_one_epoch(model, [batch], optimizer, device=device, separation_weight=tc.separation_weight)
        ids = val[:2]
        subset_refs = {training.manifest.samples[i].sample_id: refs[training.manifest.samples[i].sample_id] for i in ids}
        measured = evaluate(model, DataLoader(Subset(validation, ids), batch_size=1, num_workers=cfg["num_workers"]), subset_refs, device, cfg)
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
        guard_totals=Counter()
        batches=guarded_batches(tl,directory/'gt_audits'/f'epoch-{ep:04d}.jsonl.gz',guard_totals)
        trained = train_one_epoch(model, batches, optimizer, device=device, separation_weight=tc.separation_weight)
        trained['gt_guard']=dict(guard_totals)
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
        print(f"[guarded-v3] {ep}/{tc.epochs} loss={trained['loss']:.5f} nativeWordF1={measured['native_word_f1']:.5f} "
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
