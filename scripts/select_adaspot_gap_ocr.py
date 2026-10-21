"""Calibrate deployable Gap checkpoints against actual word OCR, TRAIN-only."""

import argparse
import gc
import json
from pathlib import Path
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from localize.adaspot_gap_preserve import decode_gap
from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_decoder import TrainableGapDecoder
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, summarize, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import load_validation, memory
from scripts.train_adaspot_gap_decoder import predict
from scripts.train_adaspot_gap_normalized import counts_for, aggregate
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.training import seed_everything


def selection_key(row):
    a = row["ocr"]["aggregate"]
    return (a["exact_casefold_iou_ge_50"], a["hmean"], -row["epoch"],
            row["threshold"] if row["threshold"] is not None else 1., row["decoder"] == "cut")


def freeze(sid, maps, gap, choice, pp, rgb, out, jobs):
    labels = decode_gap(maps["probabilities"], gap, maps["canvas"], pp, choice["threshold"], choice["decoder"])
    counts, ignored = counts_for(labels, {"maps": maps})
    predictions = []
    for p in parseq_word_proposals(labels, (rgb.shape[1], rgb.shape[0]), padding=0, min_area=0, max_instances=None):
        if p["instance_id"] in ignored: continue
        x0, y0, x1, y1 = p["bbox"]; crop = np.ascontiguousarray(rgb[y0:y1, x0:x1]); h = crop_array_hash(crop)
        path = out/"crops"/(h+".png"); path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): Image.fromarray(crop).save(path)
        jobs[h] = path; predictions.append({**p, "crop_rgb_sha256": h})
    return {"sample_id": sid, "predictions": predictions}, labels, counts


def recognize(jobs, readings, protocol, c, device, out, stage):
    model = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(protocol["parseq_source"]), checkpoint=Path(protocol["parseq_checkpoint"]),
        runtime_source_type=PARSEQ_ARCHIVE_SOURCE, runtime_source_archive=Path(protocol["parseq_archive"]), device=str(device)))
    missing = sorted((h, p) for h, p in jobs.items() if h not in readings)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for start in range(0, len(missing), c["image_batch_size"]):
            batch = missing[start:start+c["image_batch_size"]]; crops = []
            for h, p in batch:
                with Image.open(p) as image: rgb = np.asarray(image.convert("RGB"))
                if crop_array_hash(rgb) != h: raise ValueError("Frozen crop changed")
                crops.append(rgb)
            for (h, _), prediction in zip(batch, model.recognize_candidate_batch(crops)): readings[h] = prediction.to_dict()
        measured = memory(device, "qualified PARSeq AR1", [c["image_batch_size"], 3, 32, 128])
    finally: model.release()
    write_json(out/(stage+"_readings.json"), {h: readings[h] for h in jobs})
    print(f"[{stage}] unique crops={len(jobs)}, new CUDA OCR={len(missing)}", flush=True)
    return measured


def score(samples, readings):
    return summarize([{**s, "predictions": [{**p, "text": readings[p["crop_rgb_sha256"]]["raw_text"]} for p in s["predictions"]]} for s in samples])


def run(args):
    cp = args.config.resolve(); c = json.loads(cp.read_text()); device = torch.device(args.device)
    if (args.mode != "approximate" or c["reproduction_mode"] != args.mode or REPO/"configs/approximate" not in cp.parents
            or c["checkpoint_sources"] != ["baseline", "best", "last"] or any(not 0<t<1 for t in c["thresholds"])):
        raise ValueError("Explicit approximate OCR-selection configuration required")
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.init(); torch.set_num_threads(c["cpu_threads"]); cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source_run = args.decoder_run.resolve(); protocol = json.loads((source_run/"protocol.json").read_text())
    audit = json.loads((source_run/"independent_audit.json").read_text())
    if audit["status"] != "PASS": raise ValueError("Audited source required")
    check_hash(source_run/"results.json", audit["result_sha256"])
    trained = json.loads((source_run/"decoder/training_complete.json").read_text())
    if trained["completed_epochs"] != protocol["resolved_config"]["epochs"]: raise ValueError("Complete source training required")
    hashes = dict(protocol["source_hashes"])
    for p in (cp, Path(__file__), source_run/"protocol.json", source_run/"decoder/training_complete.json", source_run/"independent_audit.json",
              source_run/"decoder/best.pt", source_run/"decoder/last.pt", source_run/"ocr_readings.json"):
        hashes[str(p.resolve())] = sha256_file(p)
    for p, h in hashes.items(): check_hash(Path(p), h)
    out = args.output_root.resolve()
    if any(out == p or p in out.parents for p in (REPO, source_run, Path(protocol["word_root"]), Path(protocol["source_run"]), Path(protocol["normalized_run"]))):
        raise ValueError("New external output root required")
    out.mkdir(parents=True, exist_ok=False); args._output_created = True
    base = Path(protocol["source_run"]); normalized = Path(protocol["normalized_run"]); root = Path(protocol["word_root"])
    contract = json.loads((base/"control/run_contract.json").read_text()); sc = contract["resolved_config"]; pp = sc["postprocessing"]
    manifest = load_segmentation_manifest(root/sc["train_manifest"], require_checksums=True)
    by_id = {s.sample_id: s for s in manifest.samples}
    choices = [{"name": "off", "epoch": 0, "checkpoint_source": "baseline", "threshold": None, "decoder": "cut"}]
    for source, epoch in (("best", trained["selection"]["epoch"]), ("last", trained["completed_epochs"])):
        for threshold in c["thresholds"]:
            for decoder in c["decoders"]:
                choices.append({"name": f"{source}-{threshold}-{decoder}", "epoch": epoch, "checkpoint_source": source,
                                "threshold": threshold, "decoder": decoder})
    resolved = {**protocol, "resolved_config": c, "source_hashes": hashes, "decoder_run": str(source_run),
                "candidates": choices, "selection_uses_evaluation": False, "gpu": torch.cuda.get_device_name(device)}
    write_json(out/"protocol.json", resolved)
    # Only calibration imagery/GT are read until selection.json has been saved.
    frozen = {ch["name"]: [] for ch in choices}; jobs = {}
    calibration = json.loads((normalized/"calibration_manifest.json").read_text())
    if {r["sample_id"] for r in calibration} != set(protocol["calibration_ids"]): raise ValueError("Calibration population differs")
    for row in calibration:
        sid = row["sample_id"]; check_hash(Path(row["maps_path"]), row["maps_sha256"])
        with np.load(row["maps_path"]) as z: maps = {k: z[k] for k in z.files}
        sample = by_id[sid]; image_path = root/sample.image_path; check_hash(image_path, sample.image_sha256)
        with Image.open(image_path) as image: rgb = np.asarray(image.convert("RGB"))
        gaps = {}
        for source, epoch in (("best", trained["selection"]["epoch"]), ("last", trained["completed_epochs"])):
            path = source_run/"decoder/calibration_predictions"/str(epoch)/(sid.replace(":", "__")+".npz")
            hashes[str(path)] = sha256_file(path)
            with np.load(path) as z: gaps[source] = z["gap_probability"]
        current = []
        for ch in choices:
            result, _, _ = freeze(sid, maps, gaps.get(ch["checkpoint_source"]), ch, pp, rgb, out, jobs)
            current.append((ch["name"], result))
        gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
        annotation = root/"datasets/total_text/train/annotations"/(sid.split(":", 1)[1]+".txt")
        hashes[str(annotation)] = annotation_sha
        for name, result in current: frozen[name].append({**result, "ground_truth": gt})
    write_json(out/"calibration_frozen_inputs.json", frozen)
    readings = json.loads((source_run/"ocr_readings.json").read_text())
    calibration_memory = recognize(jobs, readings, protocol, c, device, out, "calibration")
    candidates = [{**ch, "ocr": score(frozen[ch["name"]], readings)} for ch in choices]
    if any(len(row["ocr"]["per_image"]) != len(calibration) or row["ocr"]["aggregate"]["gt_words"] != candidates[0]["ocr"]["aggregate"]["gt_words"] for row in candidates):
        raise ValueError("Calibration denominator/population changed")
    selected = max(candidates, key=selection_key)
    write_json(out/"calibration_candidates.json", candidates); write_json(out/"selection.json", selected)
    print("[selection] "+json.dumps({k:v for k,v in selected.items() if k!="ocr"})+f" exact={selected['ocr']['aggregate']['exact_casefold_iou_ge_50']}", flush=True)
    del frozen, jobs; gc.collect(); torch.cuda.empty_cache()
    # All choices are now frozen; only now load evaluation images/labels.
    cached = json.loads((Path(protocol["previous_probe"])/"feature_manifest.json").read_text())
    validation = load_validation(cached["validation"], base)
    evaluation = protocol["evaluation_ids"]; gaps = {}
    segmentation_memory = None
    if selected["threshold"] is not None:
        source = AdaSpotGapAGUNet("control", skip_floor=sc["skip_floor"], initial_gap_bias=protocol["resolved_config"]["initial_bias"])
        payload = torch.load(base/"control/best.pt", map_location="cpu", weights_only=False)
        if payload["run_contract"] != contract: raise ValueError("Source contract differs")
        source.load_state_dict(payload["model_state_dict"], strict=True); del payload
        model = TrainableGapDecoder(source, initial_bias=protocol["resolved_config"]["initial_bias"]).to(device)
        payload = torch.load(source_run/"decoder"/(selected["checkpoint_source"]+".pt"), map_location="cpu", weights_only=True)
        model.load_branch(payload["branch"]); del payload
        dataset = AdaSpotGapDataset(SegmentationDataset(manifest, root, transform=_transform(contract["resolved_preprocessing"], "letterbox", train=False)), **sc["targets"])
        by_index = {s.sample_id:i for i,s in enumerate(manifest.samples)}
        torch.cuda.reset_peak_memory_stats(device)
        gaps = predict(model, dataset, [by_index[s] for s in evaluation], c, device)
        segmentation_memory = memory(device, "frozen Text/Kernel source plus selected Gap decoder inference", [c["image_batch_size"],3,512,512])
        del source, model, dataset; gc.collect(); torch.cuda.empty_cache()
    baseline = json.loads((Path(protocol["phase0_run"])/"results.json").read_text())["arms"]["pred/axis/ar1"]
    gt_by_id = {s["sample_id"]:s["ground_truth"] for s in baseline["per_image"]}
    evaluation_order = {s["sample_id"]:i for i,s in enumerate(baseline["per_image"])}
    arms, native, jobs = {}, {}, {}
    for name, choice in (("baseline", choices[0]), ("selected", selected)):
        samples, rows = [], []
        for item in validation:
            sid = item["sample_id"]
            with Image.open(item["proposal"]["image_path"]) as image: rgb = np.asarray(image.convert("RGB"))
            sample, labels, counts = freeze(sid, item["maps"], gaps.get(sid), choice, pp, rgb, out, jobs)
            samples.append({**sample, "ground_truth":gt_by_id[sid]}); rows.append({"sample_id":sid, "counts":counts})
            path = out/name/"evaluation_maps"/(sid.replace(":", "__")+".npz"); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, labels=labels, gap_probability=gaps.get(sid, np.zeros_like(labels,dtype=np.float32)))
        samples.sort(key=lambda s:evaluation_order[s["sample_id"]])
        arms[name]=samples; native[name]={"aggregate":aggregate(rows), "per_image":rows}
    del gaps, validation; gc.collect(); torch.cuda.empty_cache()
    evaluation_memory = recognize(jobs, readings, protocol, c, device, out, "evaluation")
    results = {name:{"ocr":score(samples,readings),"native":native[name]} for name,samples in arms.items()}
    for name, arm in results.items():
        if arm["ocr"]["aggregate"]["gt_words"] != 983: raise ValueError("Evaluation denominator differs")
        arm["changes_vs_baseline"]=paired_changes(baseline,arm["ocr"])
    if results["baseline"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"] != 524: raise ValueError("Baseline failed to replay")
    for p,h in hashes.items(): check_hash(Path(p),h)
    resolved["source_hashes"]=hashes
    write_json(out/"resolved_provenance.json",resolved)
    write_json(out/"results.json",{"status":"SUCCESS","result_class":"approximate reproduction","protocol":resolved,"selection":selected,
        "arms":results,"memory":{"calibration_OCR":calibration_memory,"selected_segmentation":segmentation_memory,"evaluation_OCR":evaluation_memory}})
    lines=["# OCR-selected Gap — approximate reproduction","","Post-hoc method development; selection only on100TRAIN-calibration images, reused125evaluation images. Source training completed10epochs on GPU. No independent-transfer claim.","",
           f"Selected epoch{selected['epoch']}, threshold{selected['threshold']}, decoder{selected['decoder']}; calibration exact{selected['ocr']['aggregate']['exact_casefold_iou_ge_50']}.","",
           "| Arm | Localized RGB | Exact /983 | Proposals | OCR Hmean | Recovered / damaged |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,arm in results.items():
        a,ch=arm["ocr"]["aggregate"],arm["changes_vs_baseline"]
        lines.append(f"| {name} | {a['geometry_iou_ge_50']} | {a['exact_casefold_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} | {ch['recovered']} / {ch['damaged']} |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v['ocr']['aggregate'] for k,v in results.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","decoder-run","output-root"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if getattr(args,"_output_created",False):
            failure={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failure.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/"failure.json",failure)
        raise
