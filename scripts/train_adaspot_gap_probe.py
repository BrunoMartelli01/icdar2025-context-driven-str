"""Balanced frozen-feature Gap probe, fully isolated from qualified M/K weights."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.adaspot_gap import AdaSpotGapAGUNet
from localize.adaspot_gap import gap_instances
from localize.word_multitask import parseq_word_proposals
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.diagnose_adaspot_phase0 import (REPO, SOURCES, check_hash, paired_changes,
    summarize, write_json)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.training import seed_everything


def sample_pixels(target, validity, predicted_text, rng, floor):
    if target.shape != validity.shape or target.shape != predicted_text.shape:
        raise ValueError("Paired masks required")
    if np.any(target & ~validity): raise ValueError("Gap positive outside validity")
    positive = np.flatnonzero(target & validity)
    k = max(floor, len(positive))
    groups = [positive]
    for stratum in (validity & ~target & predicted_text, validity & ~target & ~predicted_text):
        indices = np.flatnonzero(stratum)
        groups.append(rng.choice(indices, min(k, len(indices)), replace=False))
    return groups


def class_weights(classes):
    counts = torch.bincount(classes, minlength=3)
    if torch.any(counts == 0): raise ValueError("All three training strata must be present")
    return len(classes) / (3 * counts.float()), counts


def head_probability(head, features):
    if features.ndim != 3 or features.shape[0] != 64: raise ValueError("Expected 64-channel native features")
    return F.conv2d(features[None], head.weight[:, :, None, None], head.bias).sigmoid()[0, 0]


def evaluate_head(head, rows, source, pp, device):
    totals, details = Counter(), []
    for row in rows:
        sid = row["sample_id"]
        features = torch.load(Path(row["feature_path"]), map_location="cpu", weights_only=True).to(device)
        with torch.inference_mode(): probability = head_probability(head, features).cpu().numpy()
        with np.load(source / "control/validation_exports" / sid.replace(":", "__") / "maps.npz") as data:
            labels, audit, _ = gap_instances(data["probabilities"], probability, data["canvas"], pp)
            ignored = [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & data["validity"])]
            care = labels.copy(); care[np.isin(care, ignored)] = 0
            metrics = instance_geometry(care, data["owners"], data["expected_owner_ids"].tolist())
            difficult = data["gap_target"].astype(bool) & (data["probabilities"][0] >= pp["text_threshold"]) & data["canvas"]
            audit.update(target_bridge_pixels=int(difficult.sum()), target_bridge_detected=int((difficult & (probability > pp["gap_threshold"])).sum()))
        totals.update(metrics)
        details.append({"sample_id": sid, "metrics": metrics, "audit": audit, "ignored_ids": ignored})
    tp, n, gt = totals["matched_gt_iou_ge_50"], totals["raw_proposals"], totals["gt_words"]
    return {"aggregate": {**totals, "word_f1": 2 * tp / (n + gt)}, "per_image": details}


def run(args):
    config_path = args.config.resolve(); c = json.loads(config_path.read_text())
    if (args.mode != "approximate" or REPO / "configs/approximate" not in config_path.parents
            or c["reproduction_mode"] != args.mode or c["epochs"] < 1 or c["pixel_batch_size"] < 1):
        raise ValueError("Explicit valid APPROXIMATE config required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("GPU required; no CPU fallback")
    source, phase0, root, out = [p.resolve() for p in (args.source_run, args.phase0_run, args.word_root, args.output_root)]
    if any(out == p or p in out.parents for p in (REPO, source, phase0, root)):
        raise ValueError("Output must be isolated outside inputs and repository")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    phase = json.loads((phase0 / "results.json").read_text())
    if phase["status"] != "SUCCESS": raise ValueError("Completed qualified Phase0 required")
    check_hash(phase0 / "results.json", json.loads((phase0 / "independent_audit.json").read_text())["result_sha256"])
    hashes = dict(phase["protocol"]["source_hashes"])
    for p, h in hashes.items(): check_hash(Path(p), h)
    contract = json.loads((source / "control/run_contract.json").read_text())
    config = contract["resolved_config"]; pp = config["postprocessing"]
    model = AdaSpotGapAGUNet("control", skip_floor=config["skip_floor"], initial_gap_bias=-2.)
    payload = torch.load(source / "control/best.pt", map_location="cpu", weights_only=False)
    if payload["run_contract"] != contract: raise ValueError("Source checkpoint contract differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval().to(device); del payload
    manifest_path = root / config["train_manifest"]
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    train_ids, val_ids, split_hash = _split_indices(manifest, c["seed"], config["validation_count"])
    if split_hash != config["validation_ids_sha256"]: raise ValueError("Split changed")
    transform = _transform(contract["resolved_preprocessing"], "letterbox", train=False)
    dataset = AdaSpotGapDataset(SegmentationDataset(manifest, root, transform=transform), **config["targets"])
    hashes[str(config_path)] = sha256_file(config_path)
    hashes[str(phase0 / "results.json")] = sha256_file(phase0 / "results.json")
    for f in ("scripts/train_adaspot_gap_probe.py", "models/adaspot_gap.py", "models/unet_word_research.py",
            "models/totaltext_word_kernel.py", "models/unet_adv.py", "segmentation/adaspot_gap.py",
            "segmentation/data.py", "segmentation/totaltext_image_frequency_augment.py"):
        hashes[str(REPO / f)] = sha256_file(REPO / f)
    protocol = {"mode": "approximate", "result_class": "approximate reproduction", "resolved_config": c,
        "source_hashes": hashes, "source_run": str(source), "phase0_run": str(phase0), "word_root": str(root),
        "gpu": torch.cuda.get_device_name(device), "device": str(device), "dtype": "float32",
        "torch": torch.__version__, "cuda": torch.version.cuda, "train_images": len(train_ids), "val_images": len(val_ids),
        "source_contract": contract, "selection": "native word F1, fewer fused words, earliest epoch; epoch0 eligible; no OCR selection",
        "cache": "fixed validation-style TRAIN features, no augmentation; sampled train pixels; full native validation features FP32"}
    write_json(out / "protocol.json", protocol)
    torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(c["seed"])
    vectors, strata, train_rows, val_rows = [], [], [], []
    start = time.monotonic()
    for phase_name, indices in (("train", train_ids), ("val", val_ids)):
        loader = DataLoader(Subset(dataset, indices), batch_size=c["image_batch_size"], shuffle=False,
            num_workers=c["num_workers"], timeout=60 if c["num_workers"] else 0)
        for batch_index, batch in enumerate(loader):
            with torch.no_grad(): evidence = model.forward_evidence(batch["image"].to(device))
            for j, sid in enumerate(batch["sample_id"]):
                feature = evidence["features"][j].detach().cpu().contiguous()
                if phase_name == "train":
                    target = batch["gap_target"][j, 0].numpy().astype(bool)
                    validity = batch["validity"][j, 0].numpy().astype(bool)
                    predicted_text = evidence["words"][j, 0].sigmoid().cpu().numpy() >= pp["text_threshold"]
                    groups = sample_pixels(target, validity, predicted_text, rng, c["negative_floor_per_image"])
                    flat = feature.permute(1, 2, 0).reshape(-1, 64)
                    for group, positions in enumerate(groups):
                        vectors.append(flat[torch.from_numpy(positions.astype(np.int64))])
                        strata.append(torch.full((len(positions),), group, dtype=torch.long))
                    train_rows.append({"sample_id": sid, "stratum_counts": [len(v) for v in groups]})
                else:
                    folder = source / "control/validation_exports" / sid.replace(":", "__")
                    with np.load(folder / "maps.npz") as saved:
                        current = evidence["words"][j].sigmoid().cpu().numpy()
                        if not np.allclose(current, saved["probabilities"], atol=1e-6, rtol=1e-5):
                            raise ValueError("Frozen source logits changed during feature extraction")
                    path = out / "validation_features" / (sid.replace(":", "__") + ".pt")
                    path.parent.mkdir(parents=True, exist_ok=True); torch.save(feature, path)
                    val_rows.append({"sample_id": sid, "feature_path": str(path), "feature_sha256": sha256_file(path)})
            if batch_index % 25 == 0: print(f"[features {phase_name}] {min((batch_index+1)*c['image_batch_size'],len(indices))}/{len(indices)}", flush=True)
    extraction_memory = {"allocated_bytes": torch.cuda.max_memory_allocated(device), "reserved_bytes": torch.cuda.max_memory_reserved(device),
        "seconds": time.monotonic() - start, "model_set": "frozen Text+Kernel AG-UNet", "input_shape": [c["image_batch_size"],3,512,512]}
    del model, evidence, dataset, loader; gc.collect(); torch.cuda.empty_cache()
    x = torch.cat(vectors); classes = torch.cat(strata); del vectors, strata
    weights, counts = class_weights(classes)
    torch.save({"features": x, "strata": classes}, out / "training_pixels.pt")
    write_json(out / "feature_manifest.json", {"train": train_rows, "validation": val_rows,
        "train_cache_sha256": sha256_file(out / "training_pixels.pt"), "stratum_counts": counts.tolist(), "class_weights": weights.tolist()})
    x, classes, weights = x.to(device), classes.to(device), weights.to(device)
    y = (classes == 0).float()
    head = torch.nn.Linear(64, 1).to(device)
    with torch.no_grad(): head.weight.zero_(); head.bias.fill_(c["initial_bias"])
    optimizer = torch.optim.Adam(head.parameters(), lr=c["learning_rate"])
    report = evaluate_head(head, val_rows, source, pp, device)
    original_validation = json.loads((source / "control/validation.json").read_text())
    for key in ("matched_gt_iou_ge_50", "raw_proposals", "gt_words_touched_by_fused_components"):
        if report["aggregate"][key] != sum(row[key] for row in original_validation["per_image"]):
            raise ValueError("Epoch0 did not reproduce frozen control native geometry")
    baseline_f1 = report["aggregate"]["word_f1"]
    best_key = (baseline_f1, -report["aggregate"]["gt_words_touched_by_fused_components"], 0)
    best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_epoch = 0
    history = [{"epoch": 0, "validation": report}]
    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu").manual_seed(c["seed"])
    for epoch in range(1, c["epochs"] + 1):
        order = torch.randperm(len(x), generator=generator).to(device)
        loss_sum, steps = 0., 0
        for position in range(0, len(x), c["pixel_batch_size"]):
            indices = order[position:position + c["pixel_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = head(x[indices])[:, 0]
            loss = (F.binary_cross_entropy_with_logits(logits, y[indices], reduction="none") * weights[classes[indices]]).mean()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite Gap probe loss")
            before = head.weight.detach().clone() if epoch == 1 and position == 0 else None
            loss.backward(); optimizer.step(); steps += 1; loss_sum += float(loss.detach()) * len(indices)
            if before is not None:
                change = float((head.weight.detach() - before).abs().max())
                if change <= 0 or logits.device.type != "cuda": raise RuntimeError("GPU head failed to update")
                write_json(out / "first_cuda_step.json", {"result_class":"smoke-test result","status":"PASS",
                    "device":str(logits.device),"weight_change_linf":change,"loss":float(loss.detach()),
                    "feature_shape":list(x[indices].shape),"trainable_parameters":sum(p.numel() for p in head.parameters())})
        report = evaluate_head(head, val_rows, source, pp, device)
        a = report["aggregate"]
        key = (a["word_f1"], -a["gt_words_touched_by_fused_components"], -epoch)
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append({"epoch": epoch, "loss": loss_sum / len(x), "cuda_steps": steps, "validation": report})
        write_json(out / "epochs" / f"{epoch:03d}.json", history[-1])
        torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "generator": generator.get_state(), "source_checkpoint_sha256": sha256_file(source / "control/best.pt")}, out / "last.pt")
        print(f"[probe] epoch={epoch}/{c['epochs']} loss={loss_sum/len(x):.5f} wordF1={a['word_f1']:.5f} best={best_epoch}", flush=True)
    head.load_state_dict(best_state, strict=True)
    checkpoint = {"architecture": "linear64_to_gap_equivalent_conv1x1", "state_dict": best_state,
        "best_epoch": best_epoch, "source_checkpoint_sha256": sha256_file(source / "control/best.pt"), "protocol": protocol}
    torch.save(checkpoint, out / "best.pt")
    head_memory = {"allocated_bytes": torch.cuda.max_memory_allocated(device), "reserved_bytes": torch.cuda.max_memory_reserved(device),
        "model_set": "65-parameter Gap head + cached FP32 points/features", "pixel_batch_shape": [c["pixel_batch_size"],64]}
    # Freeze chosen predictions before loading OCR; no OCR checkpoint selection.
    baseline = phase["arms"]["pred/axis/ar1"]
    base = {s["sample_id"]:s for s in baseline["per_image"]}
    readings = json.loads((phase0 / "ar1_readings.json").read_text())
    pending, samples = {}, []
    for row in val_rows:
        sid = row["sample_id"]
        features = torch.load(Path(row["feature_path"]), map_location="cpu", weights_only=True).to(device)
        with torch.inference_mode(): gap = head_probability(head, features).cpu().numpy()
        folder = source / "control/validation_exports" / sid.replace(":", "__")
        record = json.loads((folder / "proposals.json").read_text())
        with np.load(folder / "maps.npz") as maps:
            labels = gap_instances(maps["probabilities"], gap, maps["canvas"], pp)[0]
            ignored = [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & maps["validity"])]
        with Image.open(record["image_path"]) as image: rgb = np.asarray(image.convert("RGB"))
        predictions = []
        for p in parseq_word_proposals(labels, (rgb.shape[1],rgb.shape[0]), padding=0,min_area=0,max_instances=None):
            x0,y0,x1,y1 = p["bbox"]; crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]); digest=crop_array_hash(crop)
            path=out/"crops"/(digest+".png");path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): Image.fromarray(crop).save(path)
            if p["instance_id"] not in ignored:
                predictions.append({**p,"crop_rgb_sha256":digest})
                if digest not in readings: pending[digest]=path
        samples.append({"sample_id":sid,"ground_truth":base[sid]["ground_truth"],"predictions":predictions})
        path=out/"predicted_maps"/(sid.replace(':','__')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,gap_probability=gap,labels=labels)
    write_json(out / "frozen_ocr_inputs.json", samples)
    del head,x,y,classes,weights,features;gc.collect();torch.cuda.empty_cache()
    r=None
    try:
        r=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_archive,device=str(device)))
        torch.cuda.reset_peak_memory_stats(device)
        jobs=sorted(pending.items())
        for start in range(0,len(jobs),c['image_batch_size']):
            batch=jobs[start:start+c['image_batch_size']]
            crops=[np.asarray(Image.open(path).convert('RGB')) for _,path in batch]
            for (digest,_),p in zip(batch,r.recognize_candidate_batch(crops)): readings[digest]=p.to_dict()
        ocr_memory={"allocated_bytes":torch.cuda.max_memory_allocated(device),"reserved_bytes":torch.cuda.max_memory_reserved(device),
            "new_crops":len(pending),"model_set":"qualified PARSeq AR1 only","input_shape":[c['image_batch_size'],3,32,128]}
    finally:
        if r is not None:r.release()
    for s in samples:
        s['predictions']=[{**p,"text":readings[p['crop_rgb_sha256']]['raw_text']} for p in s['predictions']]
    scored=summarize(samples)
    order={s['sample_id']:i for i,s in enumerate(baseline['per_image'])}
    scored['per_image'].sort(key=lambda s:order[s['sample_id']])
    for p,h in hashes.items():check_hash(Path(p),h)
    for row in val_rows:check_hash(Path(row['feature_path']),row['feature_sha256'])
    check_hash(out/'training_pixels.pt',json.loads((out/'feature_manifest.json').read_text())['train_cache_sha256'])
    result={"status":"SUCCESS","result_class":"approximate reproduction","protocol":protocol,
        "completed_epochs":c['epochs'],"best_epoch":best_epoch,"history":history,"ocr":scored,
        "changes_vs_baseline":paired_changes(baseline,scored),"checkpoint_sha256":sha256_file(out/'best.pt'),
        "memory":{"feature_extraction":extraction_memory,"head_training":head_memory,"ocr":ocr_memory,"oom":False}}
    write_json(out/'results.json',result)
    a=scored['aggregate'];changes=result['changes_vs_baseline']
    (out/'report.md').write_text(f"# Frozen-feature Gap probe — approximate reproduction\n\nCompleted {c['epochs']} CUDA head-training epochs; selected epoch {best_epoch}, including epoch0. Frozen M/K, no augmentation, balanced sampled-pixel BCE; exploratory holdout.\n\nExact OCR: {a['exact_casefold_iou_ge_50']}/983; localized {a['geometry_iou_ge_50']}; proposals {a['proposals']}; Hmean {a['hmean']:.4%}. Recovered {changes['recovered']}, damaged {changes['damaged']} versus Text+Kernel baseline524.\n")
    print(json.dumps({"best_epoch":best_epoch,"ocr":a,"changes":{k:v for k,v in changes.items() if not k.endswith('_ids')}}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['approximate'],required=True)
    for name in ('config','source-run','phase0-run','word-root','output-root','parseq-source','parseq-checkpoint','parseq-archive'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        if args.output_root.exists():
            failure={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failure.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/'failure.json',failure)
        raise
