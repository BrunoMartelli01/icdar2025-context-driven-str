"""Filter frozen high-recall PARSeq crops; score without labels, then evaluate.

All outputs are APPROXIMATE TRAIN development. No new model downloads or OCR.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch

from recognition.clip_crop_admission import (FEATURES, JOINT, CropAdmissionScorer,
    admission_scores, choose_operating_point, filter_rows, fit_calibration, validate_rows)
from recognition.clip_innovative_verifier import font_identity
from recognition.group_word_selection import reading_confidence
from scripts import run_clip_word_verification as base
from scripts.crop_clip_runtime import _cuda_peak, read_image, sha, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/clip_high_recall_admission_train_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/clip_high_recall_admission_local_paths_v1.json"
CODE = ("recognition/clip_crop_admission.py", "scripts/run_clip_high_recall_admission.py",
        "recognition/clip_innovative_verifier.py", *base.CODE,
        "recognition/group_word_selection.py", "scripts/run_crop_group_word_selection.py",
        "scripts/evaluate_totaltext_highres_contour_padding.py")


def read(path):
    return json.loads(path.read_text())


def seal(path: Path, value: dict):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"immutable artifact differs: {path}; choose a new output root")
    else:
        write_json(path, value)


def payload_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k != "payload_sha256"},
        sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def validate_config(config: dict):
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("claim_label") != "approximate reproduction"
        or config.get("table_5_claim") is not False
        or config.get("experiment") != "clip_high_recall_crop_admission"
        or config.get("source_method") != "guarded_consolidated"
        or config["execution"]["precision"] != "float32"
        or config["evaluation"]["matching_iou"] != .5
        or config["evaluation"]["all_care_gt_in_denominator"] is not True):
        raise ValueError("invalid isolated approximate admission configuration")
    for key in ("expected_images", "expected_outputs", "expected_baseline_correct", "expected_care_words"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("invalid source population")
    for key in ("native_prefilter", "minimum_baseline_correct_retention"):
        if not np.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError("invalid explicit admission/retention policy")
    thresholds = config["thresholds"]
    if (not thresholds or thresholds != sorted(set(thresholds)) or thresholds[0] != 0
        or any(not np.isfinite(v) or not 0 <= v <= 1 for v in thresholds)
        or len({f"{v:.2f}" for v in thresholds}) != len(thresholds)):
        raise ValueError("invalid admission threshold grid")
    if any(t.count("{text}") != 1 for t in config["word_templates"]) or not config["word_templates"]:
        raise ValueError("invalid word templates")
    if any(not config["textness_prompts"][key] for key in ("positive", "negative")):
        raise ValueError("missing text/background contrast prompts")
    for field in ("batch_size", "cpu_threads"):
        if type(config["execution"][field]) is not int or config["execution"][field] < 1:
            raise ValueError("invalid CLIP execution bounds")
    glyph = config["glyph"]
    if (any(type(glyph[k]) is not int or glyph[k] < 1 for k in ("font_size", "padding"))
        or not glyph["font_sha256"] or not glyph["polarities"] or not glyph["blank_colors"]):
        raise ValueError("invalid glyph probe configuration")
    colors = list(glyph["blank_colors"])
    for polarity in glyph["polarities"]:
        colors.extend((polarity["foreground"], polarity["background"]))
        if polarity["foreground"] == polarity["background"]:
            raise ValueError("invisible glyph polarity")
    if any(len(c) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in c) for c in colors):
        raise ValueError("invalid RGB probe color")


def checked(path: Path, expected: str) -> dict:
    if sha(path) != expected:
        raise RuntimeError(f"source artifact checksum differs: {path}")
    return read(path)


def inputs(config: dict, paths: dict, limit: int | None):
    run = paths["observable_run"]
    prepared = read(run / "prepared.json")
    if (prepared["protocol_sha256"] != config["source_protocol_sha256"]
        or prepared["inputs_sha256"] != config["source_inputs_sha256"]):
        raise RuntimeError("source observable protocol differs from declared high-recall pool")
    protocol = checked(run / "protocol.json", prepared["protocol_sha256"])
    observed = checked(run / "inputs.json", prepared["inputs_sha256"])
    if (protocol["result_class"] != "APPROXIMATE_TRAIN_DEVELOPMENT"
        or protocol["reproduction_mode"] != "approximate" or protocol["table_5_claim"] is not False
        or observed["protocol_sha256"] != prepared["protocol_sha256"]):
        raise ValueError("expected sealed approximate TRAIN observations")
    ids = protocol["sample_ids"]
    if (len(ids) != config["expected_images"] or len(set(ids)) != len(ids)
        or [s["sample_id"] for s in observed["samples"]] != ids
        or sum(len(s["groups"]) for s in observed["samples"]) != config["expected_outputs"]):
        raise RuntimeError("high-recall observation population changed")
    if limit is not None and not 1 <= limit < len(ids):
        raise ValueError("smoke limit must be below full population; omit --limit for full analysis")
    refinement_root = Path(protocol["source_dir"])
    refinement = checked(refinement_root / "resolved_configuration.json", protocol["source_plan_sha256"])
    generator = checked(Path(refinement["source"]) / "resolved_configuration.json", protocol["generator_plan_sha256"])
    raw_root = Path(refinement["raw_selector"])
    checked(raw_root / "resolved_configuration.json", protocol["raw_plan_sha256"])
    manifest = Path(refinement["manifest"])
    if sha(manifest) != protocol["manifest_sha256"]:
        raise RuntimeError("source dataset manifest differs")
    samples = {s.sample_id: s for s in load_segmentation_manifest(manifest, require_checksums=True).samples}
    selected = observed["samples"][:limit] if limit else observed["samples"]
    result = []
    for sample in selected:
        sid, wh = sample["sample_id"], tuple(sample["image_wh"])
        raw_path = raw_root / "inputs" / (sid.replace(":", "__") + ".json")
        raw = checked(raw_path, protocol["source_files_sha256"][str(raw_path)])
        by_id = {r["candidate_id"]: r for r in raw["rows"]}
        selection_path = refinement_root / "selection" / raw_path.name
        original_selection = checked(selection_path, protocol["source_files_sha256"][str(selection_path)])
        baseline = original_selection["methods"][config["source_method"]]
        if [g["baseline"] for g in sample["groups"]] != baseline:
            raise RuntimeError(f"source baseline changed: {sid}")
        rows = []
        for group in sample["groups"]:
            item, candidate = group["baseline"], group["candidates"][0]
            source = by_id[item["candidate_id"]]
            if (any(item[k] != candidate[k] for k in ("candidate_id", "bbox", "raw_text", "word"))
                or source["bbox"] != item["bbox"] or source["ocr"]["raw_text"] != item["raw_text"]
                or source["crop_rgb_sha256"] != candidate["crop_rgb_sha256"]
                or reading_confidence(source["ocr"])["confidence_balanced"] != candidate["physical"]["confidence"]):
                raise RuntimeError(f"observable/native PARSeq evidence differs: {sid}")
            rows.append({**item, "crop_rgb_sha256": candidate["crop_rgb_sha256"],
                         "confidence": candidate["physical"]["confidence"],
                         "physical": candidate["physical"]["admission_visual"]})
        validate_rows(rows, wh)
        if raw["image_sha256"] != sample["image_sha256"]:
            raise RuntimeError("source image identity differs")
        result.append({"sample_id": sid, "image_sha256": sample["image_sha256"],
                       "image_wh": list(wh), "rows": rows, "ground_truth_use": "none"})
    return result, samples, Path(generator["paths"]["data_root"]), prepared, protocol


def score_path(root, sid):
    return root / "scores" / (sid.replace(":", "__") + ".json")


def run(args):
    if args.config.resolve().parent != REPO / "configs/approximate" or args.paths.resolve().parent != REPO / "configs/approximate":
        raise ValueError("admission configs must belong to configs/approximate")
    config, profile = read(args.config), read(args.paths)
    validate_config(config)
    if profile["schema_version"] != 1 or profile["path_base"] != "repository_root":
        raise ValueError("invalid explicit local path profile")
    resolve = lambda p: (Path(p) if Path(p).is_absolute() else REPO / p).resolve()
    paths = {k: resolve(profile[k]) for k in ("observable_run", "output_root", "model_paths_profile")}
    fonts = {k: resolve(p) for k, p in profile["fonts"].items()}
    observed, samples, data_root, prepared, source_protocol = inputs(config, paths, args.limit)
    model_cfg = base.load_config(resolve(config["model_config"]))
    model_paths = base.paths_profile(paths["model_paths_profile"])
    spec = {**model_cfg["models"][args.model], "weight_sha256": config["model_weight_sha256"][args.model]}
    identity = base.model_identity(model_paths["models"][args.model], spec)
    font_meta = font_identity(config["glyph"], fonts)
    # Labels are neither opened nor hashed here. Their prepared SHA is just provenance.
    plan = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "table_5_claim": False, "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TRAIN_DEVELOPMENT",
            "resolved_config": config, "config_sha256": sha(args.config), "paths_sha256": sha(args.paths),
            "model": identity, "fonts": font_meta, "device": args.device, "limit": args.limit,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda:0" and torch.cuda.is_available() else None,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "tf32": False,
            "source_observable_run": str(paths["observable_run"]), "prepared": prepared,
            "data_root": str(data_root), "observations": observed,
            "code_sha256": {name: sha(REPO / name) for name in CODE},
            "software": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")},
            "ground_truth_use": "only after every score and label-free admission decision is frozen"}
    token = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    root = (args.output_root.resolve() if args.output_root else paths["output_root"]) / args.model / token
    if root == REPO or REPO in root.parents:
        raise ValueError("generated artifacts must stay outside the repository")
    if args.check:
        print(json.dumps({"status": "INPUTS_VERIFIED", "images": len(observed),
                          "outputs": sum(len(s["rows"]) for s in observed),
                          "output": str(root), "cuda_available": torch.cuda.is_available()}))
        return root
    stage, model, scorer = "preflight", None, None
    try:
        if args.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run the host launcher or explicitly select --device cpu")
        seal(root / "resolved_configuration.json", plan)
        plan_sha = sha(root / "resolved_configuration.json")
        seed = config["execution"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(config["execution"]["cpu_threads"])
        torch.use_deterministic_algorithms(config["execution"]["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        shards = []
        for index, observation in enumerate(observed, 1):
            sid, rows = observation["sample_id"], observation["rows"]
            path = score_path(root, sid)
            contract = {"sample_id": sid, "plan_sha256": plan_sha}
            if path.exists():
                shard = read(path)
                if (shard.get("payload_sha256") != payload_digest(shard)
                    or shard["contract"] != contract or len(shard["rows"]) != len(rows)
                    or any({k:r[k] for k in expected} != expected
                           for r,expected in zip(shard["rows"], rows))):
                    raise RuntimeError(f"cached CLIP score/input contract differs: {sid}")
            else:
                stage = "image/crop_validation"
                image, _ = read_image(data_root, samples[sid], observation["image_sha256"])
                if list(image.shape[1::-1]) != observation["image_wh"]:
                    raise RuntimeError("original-image coordinates differ")
                active, crops = [], []
                for row in rows:
                    x1, y1, x2, y2 = row["bbox"]
                    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
                    if crop_array_hash(crop) != row["crop_rgb_sha256"]:
                        raise RuntimeError(f"original crop pixels changed: {row['candidate_id']}")
                    if row["confidence"] >= config["native_prefilter"]:
                        active.append(row)
                        crops.append(crop)
                if active and scorer is None:
                    stage = "model_loading"
                    from transformers import CLIPModel, CLIPProcessor
                    if args.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(args.device)
                    start = time.monotonic()
                    model, info = CLIPModel.from_pretrained(identity["local_path"], local_files_only=True,
                        use_safetensors=spec["weight_file"].endswith("safetensors"), output_loading_info=True)
                    if any(info.values()):
                        raise RuntimeError(f"CLIP loading-key mismatch: {info}")
                    model = model.eval().float().to(args.device)
                    processor = CLIPProcessor.from_pretrained(identity["local_path"], local_files_only=True)
                    scorer = CropAdmissionScorer(model, processor, config=config, fonts=fonts, device=args.device)
                    seal(root / "model_loading" / f"{time.time_ns()}.json", {"seconds":time.monotonic()-start, "model":identity,
                        "gpu":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                        "device":args.device, "precision":"float32", "cuda_peak":_cuda_peak(args.device)})
                stage = "CLIP_scoring"
                start = time.monotonic()
                values = scorer.score(active, crops) if active else []
                scores = {r["candidate_id"]:v for r,v in zip(active, values)}
                scored = [{**row, "clip_status":"SCORED" if row["candidate_id"] in scores else "NATIVE_PREFILTER_REJECTED",
                           "clip":scores.get(row["candidate_id"])} for row in rows]
                shard = {"contract":contract, "rows":scored, "ground_truth_use":"none",
                         "render_metadata":{word:scorer.metadata[word] for word in sorted({r["raw_text"] for r in active})} if scorer else {},
                         "runtime":{"seconds":time.monotonic()-start, "device":args.device, "precision":"float32",
                                    "batch_size":config["execution"]["batch_size"], "processed_shape":[3,224,224],
                                    "model_set":[spec["model_id"]] if active else [], "cuda_peak":_cuda_peak(args.device)}}
                shard["payload_sha256"] = payload_digest(shard)
                seal(path, shard)
            shards.append(shard)
            print(f"[CLIP admission {args.model}] {index}/{len(observed)} {sid}", flush=True)
        del scorer, model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        stage = "label-free_rank_calibration_and_selection"
        calibration = fit_calibration([r for s in shards for r in s["rows"] if r["clip_status"] == "SCORED"])
        seal(root / "calibration.json", {"plan_sha256":plan_sha,"method":"empirical mid-CDF; no labels", "reference":calibration})
        selected = []
        for shard in shards:
            values = [admission_scores(r, calibration, config["native_prefilter"]) for r in shard["rows"]]
            selection = {"sample_id":shard["contract"]["sample_id"], "rows":shard["rows"], "scores":values,
                         "score_shard_sha256":sha(score_path(root, shard["contract"]["sample_id"])), "ground_truth_use":"none"}
            seal(root / "selection" / (selection["sample_id"].replace(":","__")+".json"), selection)
            selected.append(selection)
        stage = "evaluation"
        labels = checked(paths["observable_run"] / "evaluation_labels.json", prepared["evaluation_labels_sha256"])
        if labels["protocol_sha256"] != prepared["protocol_sha256"] or [s["sample_id"] for s in labels["samples"]] != source_protocol["sample_ids"]:
            raise RuntimeError("label/source population differs")
        by_id = {s["sample_id"]:s["ground_truth"] for s in labels["samples"]}
        totals, denominator = {}, 0
        families = ("confidence", "physical", *(f"rank_{f}" for f in FEATURES), *JOINT)
        for sample in selected:
            sid, rows, values = sample["sample_id"], sample["rows"], sample["scores"]
            gt = by_id[sid]
            baseline = evaluate_predictions(rows, gt, config["evaluation"]["matching_iou"])
            methods = {"baseline":baseline}
            base_ids = set(baseline["correct_gt_indices"])
            for family in families:
                for threshold in config["thresholds"]:
                    kept = filter_rows(rows, values, family, threshold)
                    methods[f"{family}@{threshold:.2f}"] = evaluate_predictions(kept, gt, config["evaluation"]["matching_iou"])
            for name, score in methods.items():
                matched = set(score["correct_gt_indices"])
                totals.setdefault(name, Counter()).update(predictions=score["predictions"],correct=score["correct"],
                    retained_baseline_correct=len(matched & base_ids),lost=len(base_ids-matched),recovered=len(matched-base_ids))
            denominator += len(gt)
            seal(root / "evaluation" / (sid.replace(":","__")+".json"), {"sample_id":sid,"gt_words":len(gt),"methods":methods})
        summary = {}
        for name, counts in totals.items():
            p = counts["correct"] / counts["predictions"] if counts["predictions"] else 0.
            r = counts["correct"] / denominator if denominator else 0.
            summary[name] = {**dict(counts),"precision_proxy":p,"recall_proxy":r,"hmean_proxy":2*p*r/(p+r) if p+r else 0.}
        if not args.limit and (summary["baseline"]["correct"] != config["expected_baseline_correct"]
            or summary["baseline"]["predictions"] != config["expected_outputs"] or denominator != config["expected_care_words"]):
            raise RuntimeError("original high-recall baseline or GT denominator changed")
        points = {family:choose_operating_point(summary,family,config["minimum_baseline_correct_retention"],summary["baseline"]["correct"]) for family in families}
        eligible = [name for name in points.values() if name]
        best = min(eligible,key=lambda n:(-summary[n]["precision_proxy"],-summary[n]["retained_baseline_correct"],n)) if eligible else None
        result = {"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
            "result_class":plan["result_class"],"table_5_claim":False,"test_loaded":False,"images":len(observed),
            "gt_words":denominator,"plan_sha256":plan_sha,"summary":summary,"operating_points":points,"best_operating_point":best,
            "threshold_selection_uses_train_labels":True,"inference_uses_ground_truth":False,
            "clip_scored_outputs":sum(r["clip_status"] == "SCORED" for s in shards for r in s["rows"]),
            "execution_seconds":sum(s["runtime"]["seconds"] for s in shards),
            "cuda_peak":{k:max((s["runtime"]["cuda_peak"][k] for s in shards if s["runtime"]["cuda_peak"][k] is not None),default=None) for k in ("peak_allocated_bytes","peak_reserved_bytes")},
            "limitation":"TRAIN development operating-point selection; local bbox/exact-text proxies, not official polygon Hmean"}
        seal(root / "results.json", result)
        exports = {name:[] for name in ("confidence@0.95", *(n for n in points.values() if n))}
        for sample in selected:
            for name in exports:
                family, threshold = name.rsplit("@",1)
                exports[name].append({"sample_id":sample["sample_id"],"predictions":filter_rows(sample["rows"],sample["scores"],family,float(threshold))})
        seal(root / "selected_outputs.json", {"plan_sha256":plan_sha,"selection_uses_ground_truth":False,
             "operating_points_selected_on_train":True,"choices":exports})
        print(json.dumps({"report":str(root/"results.json"),"best":best,
                         "result":summary.get(best),"confidence_control":summary["confidence@0.95"]}),flush=True)
        return root
    except Exception as error:
        write_json(root / "failure.json", {"status":"FAILED","stage":stage,"error_type":type(error).__name__,
            "error":str(error),"traceback":traceback.format_exc(),"reproduction_mode":"approximate","device":args.device,
            "precision":"float32","batch_size":config["execution"]["batch_size"],
            "out_of_memory":isinstance(error,(torch.cuda.OutOfMemoryError,MemoryError)),
            "cuda_peak":_cuda_peak(args.device) if args.device == "cpu" or torch.cuda.is_available() else None})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--paths",type=Path,default=DEFAULT_PATHS)
    parser.add_argument("--model",choices=("vit_b32","vit_b16","vit_l14"),default="vit_l14")
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--output-root",type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
