"""Matched 1x1/3x3 Gap probes with TRAIN-only hard-positive weighting."""

import argparse
from collections import Counter
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from localize.adaspot_gap import gap_instances
from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_context import GapContextHead, bridge_strata, sampled_patches, stratum_weights
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, summarize, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.training import seed_everything


def read_config(path, mode):
    path = path.resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE namespace required; no cross-mode fallback")
    c = json.loads(path.read_text())
    if (c.get("schema_version") != 1 or c.get("reproduction_mode") != mode
            or c.get("claim_label") != "approximate reproduction"
            or c.get("arms") != {"pointwise": 1, "spatial": 3}):
        raise ValueError("Unsupported matched spatial-head configuration")
    for key in ("epochs", "pixel_batch_size", "image_batch_size", "negative_floor_per_image", "cpu_threads"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"Positive integer {key} required")
    if (type(c["num_workers"]) is not int or c["num_workers"] < 0
            or not math.isfinite(c["learning_rate"]) or c["learning_rate"] <= 0
            or c["weight_decay"] != 0 or not math.isfinite(c["initial_bias"]) or c["initial_bias"] >= 0):
        raise ValueError("Invalid declared optimizer/initialization")
    choices = c.get("scientific_choices", [])
    if not choices or any(v.get("classification") not in ("PAPER_SPECIFIED", "REPOSITORY_SPECIFIED",
            "ASSUMPTION", "HARDWARE_DEVIATION", "UNRESOLVED") or not v.get("rationale")
            or not v.get("expected_effect") for v in choices):
        raise ValueError("Scientific choices require classification, rationale and expected effect")
    return c


def memory(device, model_set, shape):
    return {"device": str(device), "gpu": torch.cuda.get_device_name(device), "dtype": "float32",
        "model_set": model_set, "input_shape": shape, "oom": False,
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device)}


def native_result(head, item, pp, device):
    with torch.inference_mode():
        gap = head.native_logits(item["features"].to(device)).sigmoid().cpu().numpy()
    maps = item["maps"]
    labels, audit, _ = gap_instances(maps["probabilities"], gap, maps["canvas"], pp)
    ignored = [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & maps["validity"])]
    care = labels.copy(); care[np.isin(care, ignored)] = 0
    counts = instance_geometry(care, maps["owners"], maps["expected_owner_ids"].tolist())
    raw = maps["canvas"] & (maps["probabilities"][0] >= pp["text_threshold"])
    hard = maps["gap_target"].astype(bool) & raw & maps["validity"]
    cut = raw & (gap > pp["gap_threshold"]) & maps["validity"]
    audit.update(hard_gap_pixels=int(hard.sum()), hard_gap_detected=int((hard & cut).sum()),
        valid_foreground_cut=int(cut.sum()), true_text_cut=int((cut & maps["text_target"]).sum()))
    row = {"sample_id": item["sample_id"], "metrics": counts, "audit": audit, "ignored_ids": ignored}
    return row, gap, labels


def evaluate(head, validation, pp, device):
    totals, rows = Counter(), []
    for item in validation:
        row, _, _ = native_result(head, item, pp, device)
        totals.update(row["metrics"]); rows.append(row)
    tp, n, gt = totals["matched_gt_iou_ge_50"], totals["raw_proposals"], totals["gt_words"]
    return {"aggregate": {**totals, "word_f1": 2 * tp / (n + gt) if n + gt else 0.}, "per_image": rows}


def selection_key(report, epoch):
    a = report["aggregate"]
    return a["word_f1"], -a["gt_words_touched_by_fused_components"], -epoch


def load_validation(rows, source):
    result = []
    for row in rows:
        check_hash(Path(row["feature_path"]), row["feature_sha256"])
        features = torch.load(row["feature_path"], map_location="cpu", weights_only=True)
        if features.dtype != torch.float32 or tuple(features.shape) != (64, 256, 256):
            raise ValueError("Qualified FP32 native validation feature shape changed")
        folder = source / "control/validation_exports" / row["sample_id"].replace(":", "__")
        with np.load(folder / "maps.npz", allow_pickle=False) as saved:
            maps = {k: saved[k].copy() for k in saved.files}
        proposal = json.loads((folder / "proposals.json").read_text())
        result.append({**row, "features": features, "maps": maps, "proposal": proposal})
    return result


def extract_training(args, c, contract, train_ids, manifest, device):
    root, out = args.word_root.resolve(), args.output_root.resolve()
    source = args.source_run.resolve()
    model = AdaSpotGapAGUNet("control", skip_floor=contract["resolved_config"]["skip_floor"], initial_gap_bias=c["initial_bias"])
    payload = torch.load(source / "control/best.pt", map_location="cpu", weights_only=False)
    if payload["run_contract"] != contract:
        raise ValueError("Source checkpoint scientific contract changed")
    model.load_state_dict(payload["model_state_dict"], strict=True); del payload
    model.eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    transform = _transform(contract["resolved_preprocessing"], "letterbox", train=False)
    dataset = AdaSpotGapDataset(SegmentationDataset(manifest, root, transform=transform), **contract["resolved_config"]["targets"])
    loader = DataLoader(Subset(dataset, train_ids), batch_size=c["image_batch_size"], shuffle=False,
        num_workers=c["num_workers"], timeout=60 if c["num_workers"] else 0)
    rng = np.random.default_rng(c["seed"])
    vectors, strata, rows = [], [], []
    torch.cuda.reset_peak_memory_stats(device); started = time.monotonic()
    for batch_index, batch in enumerate(loader):
        with torch.no_grad(): evidence = model.forward_evidence(batch["image"].to(device))
        for i, sid in enumerate(batch["sample_id"]):
            features = evidence["features"][i].detach().cpu().contiguous()
            target = batch["gap_target"][i, 0].numpy().astype(bool)
            valid = batch["validity"][i, 0].numpy().astype(bool)
            text = evidence["words"][i, 0].sigmoid().cpu().numpy() >= contract["resolved_config"]["postprocessing"]["text_threshold"]
            groups = bridge_strata(target, valid, text, rng, c["negative_floor_per_image"])
            positions = np.concatenate(groups).astype(np.int64)
            labels = torch.cat([torch.full((len(g),), j, dtype=torch.long) for j, g in enumerate(groups)])
            vectors.append(sampled_patches(features, positions)); strata.append(labels)
            path = out / "pixel_positions" / (sid.replace(":", "__") + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, positions=positions, strata=labels.numpy())
            rows.append({"sample_id": sid, "stratum_counts": [len(g) for g in groups],
                "positions_path": str(path), "positions_sha256": sha256_file(path)})
        if batch_index % 25 == 0:
            print(f"[TRAIN features] {min((batch_index+1)*c['image_batch_size'],len(train_ids))}/{len(train_ids)}", flush=True)
    measured = memory(device, "frozen Text+Kernel AG-UNet; all parameters/BN frozen", [c["image_batch_size"],3,512,512])
    measured["seconds"] = time.monotonic() - started
    del model, evidence, dataset, loader; gc.collect(); torch.cuda.empty_cache()
    patches, classes = torch.cat(vectors), torch.cat(strata)
    weights, counts = stratum_weights(classes)
    path = out / "training_patches.pt"
    torch.save({"patches": patches, "strata": classes}, path)
    write_json(out / "training_manifest.json", {"per_image": rows, "stratum_counts": counts.tolist(),
        "stratum_weights": weights.tolist(), "cache_path": str(path), "cache_sha256": sha256_file(path),
        "patch_order": "channel-major (C,dy,dx); zero padding; center 4::9", "patch_shape": list(patches.shape)})
    return patches, classes, weights, measured


def train_arm(name, kernel, patches, classes, weights, validation, pp, original, c, device, out):
    directory = out / name; directory.mkdir()
    head = GapContextHead(kernel, initial_bias=c["initial_bias"]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    labels = (classes < 2).float()
    initial = evaluate(head, validation, pp, device)
    for key in ("matched_gt_iou_ge_50", "raw_proposals", "gt_words_touched_by_fused_components"):
        if initial["aggregate"][key] != sum(row[key] for row in original["per_image"]):
            raise ValueError("Epoch0 must replay the qualified Text+Kernel geometry")
    best, best_epoch = initial, 0
    best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    history = [{"epoch": 0, "validation": initial}]
    generator = torch.Generator(device="cpu").manual_seed(c["seed"])
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, c["epochs"] + 1):
        cpu_order = torch.randperm(len(patches), generator=generator)
        order_sha = sha256(cpu_order.numpy().tobytes()).hexdigest()
        order = cpu_order.to(device)
        total_loss, steps = 0., 0
        for start in range(0, len(patches), c["pixel_batch_size"]):
            indices = order[start:start + c["pixel_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = head(patches[indices])
            loss = (F.binary_cross_entropy_with_logits(logits, labels[indices], reduction="none") * weights[classes[indices]]).mean()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite weighted Gap loss")
            before = head.linear.weight.detach().clone() if epoch == 1 and start == 0 else None
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(indices); steps += 1
            if before is not None:
                change = float((head.linear.weight.detach() - before).abs().max())
                if logits.device.type != "cuda" or change <= 0: raise RuntimeError("Head did not update on CUDA")
                positions = torch.tensor([0, 255, 256, 32896, 65535])
                with torch.inference_mode():
                    point = head(sampled_patches(validation[0]["features"], positions).to(device))
                    native = head.native_logits(validation[0]["features"].to(device)).flatten()[positions.to(device)]
                error = float((point-native).abs().max())
                if error > 2e-5: raise RuntimeError("Patch training and native convolution differ")
                write_json(directory / "first_cuda_step.json", {"status":"PASS", "result_class":"smoke-test result",
                    "device":str(logits.device), "weight_change_linf":change, "point_native_error":error,
                    "parameters":sum(p.numel() for p in head.parameters()), "input_shape":[len(indices),576]})
        report = evaluate(head, validation, pp, device)
        if selection_key(report, epoch) > selection_key(best, best_epoch):
            best, best_epoch = report, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        row = {"epoch": epoch, "cuda_steps": steps, "loss": total_loss / len(patches),
            "sample_order_sha256": order_sha, "validation": report}
        history.append(row); write_json(directory / "epochs" / f"{epoch:03d}.json", row)
        torch.save({"head":head.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch,
            "generator_state":generator.get_state()}, directory / "last.pt")
        print(f"[{name}] epoch={epoch}/{c['epochs']} loss={row['loss']:.5f} wordF1={report['aggregate']['word_f1']:.5f} best={best_epoch}", flush=True)
    torch.save({"kernel_size":kernel, "head":best_state, "epoch":best_epoch}, directory / "best.pt")
    result = {"completed_epochs":c["epochs"], "kernel_size":kernel, "parameters":sum(p.numel() for p in head.parameters()),
        "best_epoch":best_epoch, "history":history, "best_validation":best,
        "best_checkpoint_sha256":sha256_file(directory / "best.pt"),
        "memory":memory(device, f"Gap {kernel}x{kernel} head + matched cached patches", [c["pixel_batch_size"],576])}
    write_json(directory / "training_complete.json", result)
    return result


def freeze_predictions(name, trained, validation, pp, baseline, c, device, out):
    directory = out / name
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    head = GapContextHead(state["kernel_size"], initial_bias=c["initial_bias"]).to(device)
    head.load_state_dict(state["head"], strict=True)
    gt = {row["sample_id"]:row["ground_truth"] for row in baseline["per_image"]}
    samples, unique, map_hashes = [], {}, {}
    for item in validation:
        sid = item["sample_id"]
        row, gap, labels = native_result(head, item, pp, device)
        with Image.open(item["proposal"]["image_path"]) as image: rgb = np.asarray(image.convert("RGB"))
        predictions = []
        for p in parseq_word_proposals(labels, (rgb.shape[1],rgb.shape[0]), padding=0, min_area=0, max_instances=None):
            x0,y0,x1,y1 = p["bbox"]
            crop = np.ascontiguousarray(rgb[y0:y1,x0:x1]); digest = crop_array_hash(crop)
            path = out / "crops" / (digest + ".png"); path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with Image.open(path) as image:
                    if crop_array_hash(np.asarray(image.convert("RGB"))) != digest: raise ValueError("Frozen crop changed")
            else: Image.fromarray(crop).save(path)
            if p["instance_id"] not in row["ignored_ids"]:
                predictions.append({**p,"crop_rgb_sha256":digest}); unique[digest] = path
        samples.append({"sample_id":sid,"ground_truth":gt[sid],"predictions":predictions})
        path = directory / "predicted_maps" / (sid.replace(":","__") + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path,gap_probability=gap,labels=labels)
        map_hashes[str(path)] = sha256_file(path)
    write_json(directory / "frozen_ocr_inputs.json", samples)
    return samples, unique, map_hashes


def run(args):
    c = read_config(args.config, args.mode)
    source, previous, phase0, root, out = [p.resolve() for p in
        (args.source_run,args.previous_probe,args.phase0_run,args.word_root,args.output_root)]
    if any(out == p or p in out.parents for p in (REPO,source,previous,phase0,root)):
        raise ValueError("New output must be isolated from repository and all input roots")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("GPU required; no CPU fallback")
    out.mkdir(parents=True,exist_ok=False)
    args._output_created = True
    torch.set_num_threads(c["cpu_threads"]); cv2.setNumThreads(1)
    seed_everything(c["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    old = json.loads((previous / "results.json").read_text())
    old_audit = json.loads((previous / "independent_audit.json").read_text())
    check_hash(previous / "results.json",old_audit["result_sha256"])
    if old["status"] != "SUCCESS" or old_audit["status"] != "PASS" or Path(old["protocol"]["source_run"]) != source:
        raise ValueError("Qualified completed probe and matching source required")
    hashes = dict(old["protocol"]["source_hashes"])
    for p,h in hashes.items(): check_hash(Path(p),h)
    for p in (previous/"results.json",previous/"feature_manifest.json",phase0/"results.json",phase0/"ar1_readings.json",args.config.resolve()):
        hashes[str(p)] = sha256_file(p)
    files = ("models/adaspot_gap_context.py","scripts/run_adaspot_gap_context.py",
        "scripts/train_totaltext_v4_image_frequency_ablation.py","segmentation/transforms.py")
    hashes.update({str(REPO/f):sha256_file(REPO/f) for f in files})
    contract = json.loads((source/"control/run_contract.json").read_text())
    sc = contract["resolved_config"]; pp=sc["postprocessing"]
    manifest_path=root/sc["train_manifest"]; check_hash(manifest_path,sc["word_manifest_sha256"])
    manifest=load_segmentation_manifest(manifest_path,require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train": raise ValueError("TRAIN-only experiment")
    train_ids,val_ids,split_hash=_split_indices(manifest,c["seed"],sc["validation_count"])
    cached=json.loads((previous/"feature_manifest.json").read_text())
    expected_train={manifest.samples[i].sample_id for i in train_ids}
    expected_val={manifest.samples[i].sample_id for i in val_ids}
    if (split_hash != sc["validation_ids_sha256"] or len(train_ids)!=c["expected_train_images"]
            or len(val_ids)!=c["expected_validation_images"] or {r["sample_id"] for r in cached["validation"]}!=expected_val
            or len(cached["validation"])!=len(val_ids) or expected_train & expected_val):
        raise ValueError("Split/cache membership changed")
    phase=json.loads((phase0/"results.json").read_text()); baseline=phase["arms"]["pred/axis/ar1"]
    if baseline["aggregate"]["gt_words"]!=c["expected_words"]: raise ValueError("Care denominator changed")
    protocol={"mode":"approximate","result_class":"approximate reproduction","resolved_config":c,
        "source_hashes":hashes,"source_run":str(source),"previous_probe":str(previous),"phase0_run":str(phase0),
        "word_root":str(root),"device":str(device),"gpu":torch.cuda.get_device_name(device),"dtype":"float32",
        "torch":str(torch.__version__),"cuda":torch.version.cuda,"source_contract":contract,
        "train_ids":sorted(expected_train),"validation_ids":sorted(expected_val),"validation_ids_sha256":split_hash,
        "backend":{"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
            "matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32},
        "parseq_runtime_source":str(args.parseq_source.resolve()),"parseq_checkpoint":str(args.parseq_checkpoint.resolve()),
        "parseq_runtime_archive":str(args.parseq_archive.resolve()),
        "metrics":"native max one-to-one bbox IoU>=.5 for selection; original RGB bbox IoU>=.5 + exact Unicode casefold retaining punctuation for OCR; all983care words; ignored predictions only if wholly invalid after GT-free decoding"}
    write_json(out/"protocol.json",protocol)
    patches,classes,weights,extraction=extract_training(args,c,contract,train_ids,manifest,device)
    validation=load_validation(cached["validation"],source)
    patches,classes,weights=patches.to(device),classes.to(device),weights.to(device)
    original=json.loads((source/"control/validation.json").read_text())
    trained={}
    for name,kernel in c["arms"].items():
        trained[name]=train_arm(name,kernel,patches,classes,weights,validation,pp,original,c,device,out)
    if [r.get("sample_order_sha256") for r in trained["pointwise"]["history"]] != [r.get("sample_order_sha256") for r in trained["spatial"]["history"]]:
        raise ValueError("Matched arm training permutations differ")
    frozen,unique,map_hashes={},{},{}
    for name,training in trained.items():
        frozen[name],new,maps=freeze_predictions(name,training,validation,pp,baseline,c,device,out)
        unique.update(new);map_hashes.update(maps)
    del patches,classes,weights,validation;gc.collect();torch.cuda.empty_cache()
    # Fresh OCR for every care crop, with full confidences and mode provenance.
    recognizer=None;readings={}
    try:
        recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_archive,device=str(device)))
        torch.cuda.reset_peak_memory_stats(device)
        jobs=sorted(unique.items())
        for start in range(0,len(jobs),c["image_batch_size"]):
            batch=jobs[start:start+c["image_batch_size"]]
            crops=[]
            for digest,path in batch:
                with Image.open(path) as image:crop=np.asarray(image.convert("RGB"))
                if crop_array_hash(crop)!=digest:raise ValueError("Frozen OCR input changed")
                crops.append(crop)
            for (digest,_),prediction in zip(batch,recognizer.recognize_candidate_batch(crops)):
                readings[digest]={**prediction.to_dict(),"crop_rgb_sha256":digest,"decode_ar":True,"refine_iters":1}
        ocr_memory=memory(device,"qualified PARSeq AR1 only",[c["image_batch_size"],3,32,128])
    finally:
        if recognizer is not None:recognizer.release()
    write_json(out/"ocr_readings.json",readings)
    results={}
    baseline_order={r["sample_id"]:i for i,r in enumerate(baseline["per_image"])}
    for name,samples in frozen.items():
        for sample in samples:
            sample["predictions"]=[{**p,"text":readings[p["crop_rgb_sha256"]]["raw_text"]} for p in sample["predictions"]]
        samples.sort(key=lambda s:baseline_order[s["sample_id"]])
        scored=summarize(samples)
        if scored["aggregate"]["gt_words"]!=c["expected_words"]:raise ValueError("Care word denominator changed")
        results[name]={**trained[name],"ocr":scored,"changes_vs_baseline":paired_changes(baseline,scored)}
    results["spatial"]["changes_vs_pointwise"]=paired_changes(results["pointwise"]["ocr"],results["spatial"]["ocr"])
    for p,h in hashes.items():check_hash(Path(p),h)
    for row in cached["validation"]:check_hash(Path(row["feature_path"]),row["feature_sha256"])
    train_manifest=json.loads((out/"training_manifest.json").read_text())
    check_hash(Path(train_manifest["cache_path"]),train_manifest["cache_sha256"])
    for row in train_manifest["per_image"]:check_hash(Path(row["positions_path"]),row["positions_sha256"])
    write_json(out/"results.json",{"status":"SUCCESS","result_class":"approximate reproduction","protocol":protocol,
        "arms":results,"frozen_map_sha256":map_hashes,"extraction_memory":extraction,"ocr_memory":ocr_memory,
        "validation_cache_manifest_sha256":sha256_file(previous/"feature_manifest.json")})
    lines=["# Gap 1x1 vs 3x3 — approximate reproduction","",
        "Frozen Text+Kernel; same hard-positive weighting and pixel order; reused TRAIN holdout125/all983carewords; local bbox metric. Both20epoch trainings finished before OCR. No official or independent-generalization claim.","",
        "| Arm | Best epoch | Exact /983 | Localized RGB | Proposals | Hmean | Recovered / damaged vs baseline524 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,r in results.items():
        a,ch=r["ocr"]["aggregate"],r["changes_vs_baseline"]
        lines.append(f"| {name} | {r['best_epoch']} | {a['exact_casefold_iou_ge_50']} | {a['geometry_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} | {ch['recovered']} / {ch['damaged']} |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({name:{"epoch":r["best_epoch"],"ocr":r["ocr"]["aggregate"]} for name,r in results.items()},indent=2),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","source-run","previous-probe","phase0-run","word-root","output-root","parseq-source","parseq-checkpoint","parseq-archive"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True)
    return p


if __name__=="__main__":
    args=parser().parse_args()
    try:run(args)
    except Exception as error:
        if getattr(args,"_output_created",False):
            failure={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failure.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/"failure.json",failure)
        raise
