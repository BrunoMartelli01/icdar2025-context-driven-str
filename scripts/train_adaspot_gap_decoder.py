"""Train a genuine Gap-specific attention decoder, keeping Text/Kernel immutable."""

import argparse
from collections import Counter
import gc
from hashlib import sha256
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.adaspot_gap import gap_instances
from localize.adaspot_gap_preserve import decode_gap, preserve_support
from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_decoder import TrainableGapDecoder
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, summarize, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import load_validation, memory
from scripts.train_adaspot_gap_normalized import counts_for, aggregate, candidate_key
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.adaspot_gap_decoder import decoder_loss
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.training import seed_everything


def state_digest(module):
    digest = sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def predict(model, dataset, indices, c, device):
    model.eval(); output = {}
    loader = DataLoader(Subset(dataset, indices), batch_size=c["image_batch_size"],
                        num_workers=c["num_workers"], shuffle=False, timeout=60 if c["num_workers"] else 0)
    with torch.inference_mode():
        for batch in loader:
            gap = model(batch["image"].to(device)).sigmoid().cpu().numpy()
            output.update({sid: gap[i, 0] for i, sid in enumerate(batch["sample_id"])})
    return output


def calibrate_maps(gaps, items, pp, c, epoch, out):
    grid = {(t, d): [] for t in c["thresholds"] for d in c["decoders"]}
    for item in items:
        sid, maps = item["sample_id"], item["maps"]
        gap = gaps[sid]
        path = out/"calibration_predictions"/str(epoch)/(sid.replace(":", "__")+".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, gap_probability=gap)
        raw = maps["canvas"] & (maps["probabilities"][0] >= pp["text_threshold"])
        for threshold in c["thresholds"]:
            cut = decode_gap(maps["probabilities"], gap, maps["canvas"], pp, threshold, "cut") if np.any(raw & (gap > threshold)) else item["baseline"]
            for decoder in c["decoders"]:
                labels = preserve_support(cut, raw) if decoder == "preserve" and not np.array_equal(cut > 0, raw) else cut
                counts, _ = counts_for(labels, item)
                grid[threshold, decoder].append({"sample_id": sid, "counts": counts})
    return [{"epoch": epoch, "threshold": t, "decoder": d, "aggregate": aggregate(rows), "per_image": rows}
            for (t, d), rows in grid.items()]


def run(args):
    cp = args.config.resolve()
    if args.mode != "approximate" or REPO/"configs/approximate" not in cp.parents:
        raise ValueError("Explicit approximate configuration required")
    c = json.loads(cp.read_text())
    if (c["reproduction_mode"] != args.mode or c["epochs"] < 1 or c["learning_rate"] <= 0
            or any(not 0 < t < 1 for t in c["thresholds"]) or c["decoders"] != ["cut", "preserve"]
            or any(not 1 <= e <= c["epochs"] for e in c["checkpoint_epochs"])
            or any(c["loss"][k] < 0 for k in ("balanced_bce_weight", "dice_weight", "text_protection_weight"))):
        raise ValueError("Unsupported explicit decoder configuration")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.init(); torch.set_num_threads(c["cpu_threads"]); cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    normalized = json.loads((args.normalized_run/"results.json").read_text())
    if normalized["status"] != "SUCCESS": raise ValueError("Completed normalized source required")
    old = normalized["protocol"]
    for key in ("source_run", "context_run", "previous_probe", "phase0_run", "word_root", "parseq_source", "parseq_checkpoint", "parseq_archive"):
        if getattr(args, key).resolve() != Path(old[key]).resolve(): raise ValueError("Pinned path differs: "+key)
    hashes = dict(old["source_hashes"])
    paths = [cp, args.normalized_run/"results.json", args.normalized_run/"calibration_manifest.json"]
    paths += [REPO/p for p in ("models/adaspot_gap_decoder.py", "segmentation/adaspot_gap_decoder.py", "scripts/train_adaspot_gap_decoder.py")]
    for path in paths: hashes[str(path.resolve())] = sha256_file(path)
    for path, h in hashes.items(): check_hash(Path(path), h)
    contract = json.loads((args.source_run/"control/run_contract.json").read_text())
    sc, pp = contract["resolved_config"], contract["resolved_config"]["postprocessing"]
    if c["loss"]["teacher_text_threshold"] != pp["text_threshold"]: raise ValueError("Teacher threshold differs from qualified source")
    manifest = load_segmentation_manifest(args.word_root/sc["train_manifest"], require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train": raise ValueError("TRAIN only")
    by_id = {s.sample_id: i for i, s in enumerate(manifest.samples)}
    fit, calib, evaluation = [old[k] for k in ("fit_ids", "calibration_ids", "evaluation_ids")]
    if (len(fit), len(calib), len(evaluation)) != (1030, 100, 125) or any(set(a) & set(b) for a, b in ((fit, calib), (fit, evaluation), (calib, evaluation))):
        raise ValueError("Disjoint fit/calibration/evaluation populations required")
    out = args.output_root.resolve()
    if any(out == p.resolve() or p.resolve() in out.parents for p in (REPO, args.source_run, args.context_run, args.previous_probe, args.phase0_run, args.word_root, args.normalized_run)):
        raise ValueError("New external output root required")
    out.mkdir(parents=True, exist_ok=False); args._output_created = True
    protocol = {**old, "source_hashes": hashes, "resolved_config": c, "normalized_run": str(args.normalized_run.resolve()),
                "gpu": torch.cuda.get_device_name(device), "device": str(device), "dtype": "float32", "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "deterministic_algorithms": True, "tf32": False,
                "architecture": "frozen-qualified-Text-Kernel-and-encoder-plus-copied-trainable-Gap-attention-decoder"}
    write_json(out/"protocol.json", protocol)
    train_transform = _transform(contract["resolved_preprocessing"], "letterbox", train=True)
    val_transform = _transform(contract["resolved_preprocessing"], "letterbox", train=False)
    dataset = AdaSpotGapDataset(SegmentationDataset(manifest, args.word_root, transform=train_transform), **sc["targets"])
    val_dataset = AdaSpotGapDataset(SegmentationDataset(manifest, args.word_root, transform=val_transform), **sc["targets"])
    items = []
    for row in json.loads((args.normalized_run/"calibration_manifest.json").read_text()):
        check_hash(Path(row["maps_path"]), row["maps_sha256"])
        with np.load(row["maps_path"], allow_pickle=False) as saved: maps = {k: saved[k] for k in saved.files}
        baseline = gap_instances(maps["probabilities"], None, maps["canvas"], pp, use_gap=False)[0]
        items.append({"sample_id": row["sample_id"], "maps": maps, "baseline": baseline})
    if {i["sample_id"] for i in items} != set(calib): raise ValueError("Calibration cache IDs differ")
    source = AdaSpotGapAGUNet("control", skip_floor=sc["skip_floor"], initial_gap_bias=c["initial_bias"])
    payload = torch.load(args.source_run/"control/best.pt", map_location="cpu", weights_only=False)
    if payload["run_contract"] != contract: raise ValueError("Source checkpoint contract differs")
    source.load_state_dict(payload["model_state_dict"], strict=True); del payload
    model = TrainableGapDecoder(source, initial_bias=c["initial_bias"]).to(device)
    source_digest = state_digest(model.source)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["learning_rate"], weight_decay=c["weight_decay"])
    directory = out/"decoder"; directory.mkdir()
    rows = [{"sample_id": item["sample_id"], "counts": counts_for(item["baseline"], item)[0]} for item in items]
    best = {"epoch": 0, "threshold": None, "decoder": "cut", "aggregate": aggregate(rows), "per_image": rows}
    torch.save({"branch": model.branch_state(), "selection": best}, directory/"best.pt")
    candidates, history = [best], []
    generator = torch.Generator().manual_seed(c["seed"])
    global_step = 0; torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, c["epochs"]+1):
        started = time.monotonic(); train_transform.set_epoch(epoch-1); model.train()
        loader = DataLoader(Subset(dataset, [by_id[s] for s in fit]), batch_size=c["image_batch_size"], shuffle=True,
                            generator=generator, num_workers=c["num_workers"], timeout=60 if c["num_workers"] else 0)
        totals, seen, steps = Counter(), [], 0
        for batch in loader:
            local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            evidence = model.forward_evidence(local["image"])
            if evidence["gap"].device != device: raise RuntimeError("Training left requested CUDA device")
            loss, terms = decoder_loss(evidence["gap"], local["gap_target"], local["mask"], local["validity"], evidence["words"][:, :1].sigmoid(), c["loss"])
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite decoder loss")
            if global_step == 0:
                with torch.no_grad(): reference = model.source.forward_evidence(local["image"])["words"]
                teacher_error = float((reference-evidence["words"]).abs().max())
                if teacher_error != 0: raise RuntimeError("Qualified Text/Kernel forward changed")
            if local["validity"].any():
                before = model.gap.weight.detach().clone() if global_step == 1 else None
                loss.backward()
                if any(p.grad is not None for p in model.source.parameters()): raise RuntimeError("Frozen source received gradients")
                gradient = sum(float(p.grad.detach().abs().sum()) for p in model.decoder.parameters() if p.grad is not None) if global_step == 1 else None
                optimizer.step(); global_step += 1; steps += 1
                if before is not None:
                    change = float((model.gap.weight.detach()-before).abs().max())
                    if not gradient > 0 or not change > 0: raise RuntimeError("Decoder/head did not learn on CUDA")
                    write_json(directory/"first_cuda_step.json", {"status": "PASS", "result_class": "smoke-test result", "device": str(device),
                        "decoder_gradient_l1": gradient, "head_weight_update_linf": change, "teacher_max_error": teacher_error})
            for key, value in {"loss": loss, **terms}.items(): totals[key] += float(value.detach())*len(batch["sample_id"])
            seen.extend(batch["sample_id"])
            if steps == 1 or steps % 50 == 0: print(f"[Gap decoder] epoch={epoch} batch={steps}/{len(loader)} loss={float(loss.detach()):.5f}", flush=True)
        if len(seen) != len(fit) or set(seen) != set(fit): raise ValueError("Fit epoch omitted/duplicated images")
        record = {"epoch": epoch, "samples": len(seen), "cuda_steps": steps, "skipped_all_invalid_batches": len(loader)-steps,
                  "sample_order_sha256": sha256("".join(s+"\n" for s in seen).encode()).hexdigest(),
                  **{k: v/len(seen) for k, v in totals.items()}}
        if epoch in c["checkpoint_epochs"]:
            gaps = predict(model, val_dataset, [by_id[s] for s in calib], c, device)
            current = calibrate_maps(gaps, items, pp, c, epoch, directory); candidates.extend(current)
            write_json(directory/"calibration"/f"{epoch:03d}.json", current)
            candidate = max(current, key=candidate_key)
            if candidate_key(candidate) > candidate_key(best):
                best = candidate; torch.save({"branch": model.branch_state(), "selection": best}, directory/"best.pt")
            del gaps
        record["seconds"] = time.monotonic()-started
        record["memory"] = memory(device, "frozen ResNet50/Text/Kernel + trainable Gap attention decoder/AdamW", [c["image_batch_size"], 3, 512, 512])
        history.append(record); write_json(directory/"epochs"/f"{epoch:03d}.json", record)
        torch.save({"branch": model.branch_state(), "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
                    "generator_state": generator.get_state(), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)}, directory/"last.pt")
        print(f"[Gap decoder] epoch={epoch}/{c['epochs']} loss={record['loss']:.5f} selected={best['epoch']} calibrationF1={best['aggregate']['word_f1']:.5f} threshold={best['threshold']} decoder={best['decoder']} seconds={record['seconds']:.1f}", flush=True)
    if state_digest(model.source) != source_digest: raise RuntimeError("Qualified source weights or BN statistics changed")
    trained = {"decoder": {"completed_epochs": c["epochs"], "history": history, "candidates": candidates, "selection": best,
        "checkpoint_sha256": sha256_file(directory/"best.pt"), "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "source_parameters": sum(p.numel() for p in model.source.parameters()), "source_state_sha256_before_after": source_digest,
        "memory": record["memory"]}}
    write_json(directory/"training_complete.json", trained["decoder"])
    saved = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
    model.load_branch(saved["branch"]); del saved
    # Evaluation images and labels are consulted only after selection is final.
    gaps = predict(model, val_dataset, [by_id[s] for s in evaluation], c, device)
    del optimizer, model, source, dataset, val_dataset, items, evidence, local, reference
    gc.collect(); torch.cuda.empty_cache()
    cached=json.loads((args.previous_probe/"feature_manifest.json").read_text())
    validation=load_validation(cached["validation"],args.source_run)
    baseline=json.loads((args.phase0_run/"results.json").read_text())["arms"]["pred/axis/ar1"]
    by_gt={r["sample_id"]:r["ground_truth"] for r in baseline["per_image"]}
    frozen={"baseline":[]};native={};unique={}
    for kind in ["baseline"]+["decoder"]:
        samples=[];native_rows=[]
        if kind!="baseline":selection=best
        for item in validation:
            sid=item["sample_id"];maps=item["maps"]
            if kind=="baseline":gap=None;labels=decode_gap(maps["probabilities"],None,maps["canvas"],pp,None,"cut")
            else:
                gap=gaps[sid]
                labels=decode_gap(maps["probabilities"],gap,maps["canvas"],pp,selection["threshold"],selection["decoder"])
            counts,ignored=counts_for(labels,item);native_rows.append({"sample_id":sid,"counts":counts})
            with Image.open(item["proposal"]["image_path"]) as image:rgb=np.asarray(image.convert("RGB"))
            predictions=[]
            for p in parseq_word_proposals(labels,(rgb.shape[1],rgb.shape[0]),padding=0,min_area=0,max_instances=None):
                if p["instance_id"] in ignored:continue
                x0,y0,x1,y1=p["bbox"];crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]);digest=crop_array_hash(crop)
                path=out/"crops"/(digest+".png");path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():Image.fromarray(crop).save(path)
                unique[digest]=path;predictions.append({**p,"crop_rgb_sha256":digest})
            samples.append({"sample_id":sid,"ground_truth":by_gt[sid],"predictions":predictions})
            path=out/kind/"evaluation_maps"/(sid.replace(":","__")+".npz");path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,labels=labels,gap_probability=gap if gap is not None else np.zeros_like(labels,dtype=np.float32))
        frozen[kind]=samples;native[kind]={"aggregate":aggregate(native_rows),"per_image":native_rows}
        write_json(out/kind/"frozen_ocr_inputs.json",samples)
    del gaps,validation;gc.collect();torch.cuda.empty_cache()
    recognizer=None;readings={}
    try:
        recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_archive,device=str(device)))
        torch.cuda.reset_peak_memory_stats(device);jobs=sorted(unique.items())
        for start in range(0,len(jobs),c["image_batch_size"]):
            batch=jobs[start:start+c["image_batch_size"]];crops=[]
            for digest,path in batch:
                with Image.open(path) as image:crop=np.asarray(image.convert("RGB"))
                if crop_array_hash(crop)!=digest:raise ValueError("Frozen OCR crop changed")
                crops.append(crop)
            for (digest,_),p in zip(batch,recognizer.recognize_candidate_batch(crops)):readings[digest]=p.to_dict()
        ocr_memory=memory(device,"qualified PARSeq AR1 only",[c["image_batch_size"],3,32,128])
    finally:
        if recognizer is not None:recognizer.release()
    write_json(out/"ocr_readings.json",readings)
    results={};order={s["sample_id"]:i for i,s in enumerate(baseline["per_image"])}
    for kind,samples in frozen.items():
        for sample in samples:sample["predictions"]=[{**p,"text":readings[p["crop_rgb_sha256"]]["raw_text"]} for p in sample["predictions"]]
        samples.sort(key=lambda s:order[s["sample_id"]]);scored=summarize(samples)
        if scored["aggregate"]["gt_words"]!=baseline["aggregate"]["gt_words"]:raise ValueError("Evaluation denominator changed")
        results[kind]={"ocr":scored,"native":native[kind],"changes_vs_baseline":paired_changes(baseline,scored),**(trained[kind] if kind in trained else {})}
    if results["baseline"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"]!=baseline["aggregate"]["exact_casefold_iou_ge_50"]:
        raise ValueError("Fresh baseline OCR failed to replay")
    for p,h in hashes.items():check_hash(Path(p),h)
    for row in cached["validation"]:check_hash(Path(row["feature_path"]),row["feature_sha256"])
    for row in json.loads((args.normalized_run/"calibration_manifest.json").read_text()):
        check_hash(Path(row["maps_path"]),row["maps_sha256"])
    write_json(out/"results.json",{"status":"SUCCESS","result_class":"approximate reproduction","protocol":protocol,"arms":results,
        "ocr_memory":ocr_memory})
    lines=["# Trainable Gap decoder — approximate reproduction","","Decoder fit1030 / calibration100 / evaluation125; calibration-only choices, exploratory reused evaluation holdout. No official or independent-transfer claim.","",
        "| Arm | Chosen epoch | Threshold / decoder | Exact /983 | Localized RGB | Proposals | Hmean | Recovered / damaged |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind,r in results.items():
        a,ch=r["ocr"]["aggregate"],r["changes_vs_baseline"];s=r.get("selection",{"epoch":0,"threshold":None,"decoder":"off"})
        lines.append(f"| {kind} | {s['epoch']} | {s['threshold']} / {s['decoder']} | {a['exact_casefold_iou_ge_50']} | {a['geometry_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} | {ch['recovered']} / {ch['damaged']} |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:{"selected_epoch":v.get("selection",{}).get("epoch"),"ocr":v["ocr"]["aggregate"]} for k,v in results.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","normalized-run","context-run","previous-probe","source-run","phase0-run","word-root","output-root","parseq-source","parseq-checkpoint","parseq-archive"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if getattr(args,"_output_created",False):
            failed={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failed.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/"failure.json",failed)
        raise
