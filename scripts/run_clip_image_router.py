"""Score every Text+Kernel/PARSeq output, calibrate and audit whole-image routing."""
from __future__ import annotations

import argparse
import hashlib
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

from recognition.clip_crop_admission import CropAdmissionScorer, fit_calibration
from recognition.clip_image_router import accepted, choose_threshold, image_scores, partition, summarize
from recognition.group_word_selection import reading_confidence
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.crop_clip_runtime import _cuda_peak, sha, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.run_clip_word_verification import model_identity, paths_profile

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/clip_image_router_train_v1.json"
PATHS = REPO / "configs/approximate/clip_image_router_local_paths_v1.json"
CODE = ("scripts/run_clip_image_router.py", "recognition/clip_image_router.py",
        "recognition/clip_crop_admission.py", "recognition/clip_innovative_verifier.py",
        "recognition/clip_word_verifier.py", "recognition/group_word_selection.py",
        "recognition/group_word_selection_runtime.py", "scripts/crop_clip_runtime.py",
        "scripts/run_clip_word_verification.py", "scripts/run_clip_high_recall_admission.py",
        "scripts/evaluate_totaltext_highres_contour_padding.py",
        "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
        "recognition/scene_text_v1_evaluation.py")


def checked(path, digest):
    if sha(path) != digest:
        raise RuntimeError("Frozen artifact changed: " + str(path))
    return json.loads(path.read_text())


def load_inputs(args):
    if args.config.resolve().parent != REPO/"configs/approximate" or args.paths.resolve().parent != REPO/"configs/approximate":
        raise ValueError("Explicit isolated APPROXIMATE config/paths required")
    cfg = json.loads(args.config.read_text())
    if (args.mode != "approximate" or cfg["reproduction_mode"] != "approximate"
        or cfg["claim_label"] != "approximate reproduction" or cfg["table_5_claim"] is not False
        or cfg["source_variant"] != "text_kernel" or cfg["source_method"] != "seeded"
        or cfg["execution"]["precision"] != "float32" or cfg["matching_iou"] != .5
        or not 0 < cfg["target_accuracy"] <= 1):
        raise ValueError("invalid image router scientific configuration")
    for name in ("batch_size", "cpu_threads", "seed"):
        if type(cfg["execution"][name]) is not int or cfg["execution"][name] < (0 if name == "seed" else 1):
            raise ValueError("invalid router execution parameter")
    raw = json.loads(args.paths.read_text())
    if raw.get("schema_version") != 1 or raw.get("path_base") != "repository_root":
        raise ValueError("invalid router path profile")
    resolve = lambda p: (Path(p) if Path(p).is_absolute() else REPO/p).resolve()
    paths = {k: resolve(raw[k]) for k in ("source_run", "segmentation_run", "model_paths_profile", "output_root")}
    paths["fonts"] = {k: resolve(v) for k,v in raw["fonts"].items()}
    source = paths["source_run"]
    report = checked(source/"recognition_results.json", cfg["source_recognition_sha256"])
    geometry = checked(source/"geometry_results.json", cfg["source_geometry_sha256"])
    validation = checked(paths["segmentation_run"]/"validation.json", cfg["source_validation_sha256"])
    if (report["status"] != "SUCCESS" or geometry["status"] != "SUCCESS"
        or validation["checkpoint_sha256"] != cfg["source_checkpoint_sha256"]
        or report["protocol"]["geometry_results_sha256"] != cfg["source_geometry_sha256"]
        or report["protocol"]["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256
        or report["protocol"]["parseq_runtime_archive_sha256"] != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("source model/PARSeq provenance differs")
    branch = report["branches"][cfg["source_variant"]][cfg["source_method"]]
    geo = geometry["branches"][cfg["source_variant"]][cfg["source_method"]]
    if [r["sample_id"] for r in branch["per_image"]] != [r["sample_id"] for r in geo["per_image"]]:
        raise ValueError("source population/order differs")
    observed = []
    for row, gr in zip(branch["per_image"], geo["per_image"]):
        image_path = Path(gr["image_path"])
        if sha(image_path) != gr["image_sha256"]:
            raise RuntimeError("source original RGB changed")
        words = []
        if len(row["predictions_all"]) != len(gr["proposals"]):
            raise RuntimeError("frozen source proposal population differs")
        for pred, proposal in zip(row["predictions_all"], gr["proposals"]):
            if {k:v for k,v in pred.items() if k != "text"} != proposal:
                raise RuntimeError("frozen recognition crop differs from geometry")
            key = pred["crop_rgb_sha256"]
            ocr_path = source/"ocr"/(key+".json")
            ocr = checked(ocr_path, report["ocr_records_sha256"][key])
            if (ocr["raw_text"] != pred["text"] or ocr["crop_rgb_sha256"] != key
                or ocr["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256
                or ocr["parseq_source_archive_sha256"] != PARSEQ_RUNTIME_ARCHIVE_SHA256
                or ocr.get("decode_ar", True) is not True or ocr.get("refine_iters", 1) != 1):
                raise ValueError("source frozen PARSeq reading differs")
            native = reading_confidence(ocr)
            words.append({"instance_id":pred["instance_id"], "bbox":pred["bbox"],
                          "raw_text":pred["text"], "crop_rgb_sha256":key, "native":native,
                          "confidence":native["confidence_balanced"], "ocr_sha256":sha(ocr_path)})
        if len({r["instance_id"] for r in words}) != len(words):
            raise ValueError("duplicate word identity")
        # Deliberately discard source GT/ignore fields before any CLIP/routing operation.
        observed.append({"sample_id":row["sample_id"], "image_path":str(image_path),
                         "image_sha256":gr["image_sha256"], "rows":words})
    if len(observed) != cfg["expected_images"] or sum(len(r["rows"]) for r in observed) != cfg["expected_all_outputs"]:
        raise ValueError("requested Text+Kernel complete source population differs")
    if args.limit is not None:
        if not 0 < args.limit <= len(observed):
            raise ValueError("invalid smoke limit")
        observed = observed[:args.limit]
    return cfg, paths, observed


def run(args):
    cfg, paths, observed = load_inputs(args)
    if args.output_root:
        paths["output_root"] = args.output_root.resolve()
    if paths["output_root"] == REPO or any(paths["output_root"].is_relative_to(paths[k]) for k in ("source_run","segmentation_run")):
        raise ValueError("use an isolated output root")
    model_cfg = json.loads((REPO/cfg["model_config"]).read_text())
    spec = {**model_cfg["models"][args.model], "weight_sha256":cfg["model_weight_sha256"][args.model]}
    identity = model_identity(paths_profile(paths["model_paths_profile"])["models"][args.model], spec)
    if args.check:
        print(json.dumps({"status":"PREFLIGHT_PASS", "images":len(observed),
                          "all_words":sum(len(r["rows"]) for r in observed), "cuda_available":torch.cuda.is_available(),
                          "model":args.model}), flush=True)
        return None
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable here: run on GPU host or explicitly select --device cpu")
    seed = cfg["execution"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats(args.device)
    ids = [r["sample_id"] for r in observed]
    split = partition(ids, cfg["calibration_images"], seed) if not args.limit else {"calibration":ids,"validation":[]}
    plan = {"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
            "result_class":"smoke-test result" if args.limit else "partial-dataset result",
            "table_5_claim":False,"config":cfg,"paths":{k:str(v) for k,v in paths.items() if k != "fonts"},
            "fonts":{k:str(v) for k,v in paths["fonts"].items()}, "model":identity,"device":args.device,
            "split":split,"code_sha256":{n:sha(REPO/n) for n in CODE},
            "input_digest":hashlib.sha256(json.dumps(observed,sort_keys=True).encode()).hexdigest(),
            "software":{"torch":torch.__version__,"numpy":np.__version__,"pillow":Image.__version__,
                        "transformers":__import__("transformers").__version__},
            "cuda_available":torch.cuda.is_available(),
            "gpu":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "cuda_build":torch.version.cuda,"deterministic_algorithms":True,"tf32":False}
    fingerprint = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    root = paths["output_root"]/args.model/fingerprint
    seal(root/"resolved_configuration.json", plan)
    plan_sha = sha(root/"resolved_configuration.json")
    shards, model, scorer = [], None, None
    stage = "scoring"
    try:
        for i, sample in enumerate(observed,1):
            path = root/"scores"/(sample["sample_id"].replace(":","__")+".json")
            if path.exists():
                shard = json.loads(path.read_text())
                if shard["plan_sha256"] != plan_sha or shard["payload_sha256"] != payload_digest(shard):
                    raise RuntimeError("cached scoring shard contract/payload differs")
            else:
                start = time.monotonic()
                if scorer is None:
                    from transformers import CLIPModel, CLIPProcessor
                    model, info = CLIPModel.from_pretrained(identity["local_path"],local_files_only=True,
                        use_safetensors=spec["weight_file"].endswith("safetensors"),output_loading_info=True)
                    if any(info.values()):
                        raise RuntimeError(f"CLIP checkpoint key mismatch: {info}")
                    model = model.eval().float().to(args.device)
                    scorer = CropAdmissionScorer(model, CLIPProcessor.from_pretrained(identity["local_path"],local_files_only=True),
                                                config=cfg,fonts=paths["fonts"],device=args.device)
                if sha(Path(sample["image_path"])) != sample["image_sha256"]:
                    raise RuntimeError("source RGB changed during inference")
                with Image.open(sample["image_path"]) as image:
                    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
                crops = []
                for row in sample["rows"]:
                    x1,y1,x2,y2 = row["bbox"]
                    if not (all(type(v) is int for v in row["bbox"]) and 0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
                        raise ValueError("crop outside stored original coordinate frame")
                    crop = np.ascontiguousarray(rgb[y1:y2,x1:x2])
                    if crop_array_hash(crop) != row["crop_rgb_sha256"]:
                        raise RuntimeError("source exact crop RGB differs")
                    crops.append(crop)
                nonempty = [i for i,r in enumerate(sample["rows"]) if r["raw_text"].strip()]
                clip = scorer.score([sample["rows"][i] for i in nonempty],[crops[i] for i in nonempty])
                scene = scorer.encode_images([rgb])[0]
                evidence = {}
                for j,v in zip(nonempty,clip):
                    v["scene_ensemble"] = float(scene @ scorer.texts[sample["rows"][j]["raw_text"]]["ensemble"])
                    evidence[j] = v
                rows = [{**r,"clip":evidence.get(j)} for j,r in enumerate(sample["rows"])]
                shard = {**sample,"rows":rows,"plan_sha256":plan_sha,"ground_truth_in_scoring":False,
                         "runtime":{"seconds":time.monotonic()-start,"model_set":[spec["model_id"]],
                                    "precision":"float32","device":args.device,"batch_size":cfg["execution"]["batch_size"],
                                    "processed_shape":[3,224,224],"cuda_peak":_cuda_peak(args.device)}}
                shard["payload_sha256"] = payload_digest(shard)
                seal(path,shard)
            shards.append(shard)
            if i % 10 == 0 or i == len(observed):
                print(f"[CLIP image router {args.model}] {i}/{len(observed)}",flush=True)
        del scorer,model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        stage = "calibration_and_evaluation"
        calibration_rows = [r for s in shards if s["sample_id"] in split["calibration"] for r in s["rows"] if r["clip"] is not None]
        reference = fit_calibration(calibration_rows)
        reference["scene_ensemble"] = sorted(r["clip"]["scene_ensemble"] for r in calibration_rows)
        scores = {s["sample_id"]:image_scores(s["rows"],reference) for s in shards}
        seal(root/"scores_by_image.json", {"scores":scores,"rank_reference":reference,"fit_sample_ids":split["calibration"],
                                          "ground_truth_in_scores":False,"aggregation":"minimum over ALL original predictions"})
        # Labels enter threshold calibration only after all features/aggregates are sealed.
        report = checked(paths["source_run"]/"recognition_results.json", cfg["source_recognition_sha256"])
        geometry = checked(paths["source_run"]/"geometry_results.json", cfg["source_geometry_sha256"])
        gr = {r["sample_id"]:r for r in geometry["branches"][cfg["source_variant"]][cfg["source_method"]]["per_image"]}
        evaluated = []
        for r in report["branches"][cfg["source_variant"]][cfg["source_method"]]["per_image"]:
            if r["sample_id"] not in ids:
                continue
            if sha(Path(gr[r["sample_id"]]["annotation_path"])) != gr[r["sample_id"]]["annotation_sha256"]:
                raise RuntimeError("source evaluation annotation changed")
            ignored = set(r["ignored_prediction_ids"])
            care = [p for p in r["predictions_all"] if p["instance_id"] not in ignored]
            matched = _correct(care,r["ground_truth"],normalize=False)
            if len(matched) != r["counts"]["exact_casefold_iou_ge_50"]:
                raise RuntimeError("independent frozen baseline replay differs")
            evaluated.append({"sample_id":r["sample_id"],"tp":len(matched),"words":len(care),
                              "gt_words":len(r["ground_truth"]),"outputs_all":len(r["predictions_all"])})
        baseline = summarize(evaluated,scores,None,None)
        if not args.limit and (baseline["correct_words"] != cfg["expected_correct_words"]
            or baseline["detected_words"] != cfg["expected_care_outputs"] or baseline["accepted_gt_words"] != cfg["expected_gt_words"]):
            raise RuntimeError("source baseline/denominator replay differs")
        cal = [r for r in evaluated if r["sample_id"] in split["calibration"]]
        val = [r for r in evaluated if r["sample_id"] in split["validation"]]
        families = list(next(iter(scores.values())))
        choices = {}
        for family in families:
            chosen = choose_threshold(cal,scores,family,cfg["target_accuracy"])
            threshold = chosen["threshold"]
            choices[family] = {**chosen,"validation":summarize(val,scores,family,threshold),
                               "all_images":summarize(evaluated,scores,family,threshold)}
        selected = min(choices,key=lambda f:(-choices[f]["calibration"]["accepted_images"],
                        -choices[f]["calibration"]["detected_words"],f))
        # Global all-125 scans are explicitly post-hoc diagnostics, never deployment selection.
        exploratory = {f:choose_threshold(evaluated,scores,f,cfg["target_accuracy"]) for f in families}
        result = {"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
                  "result_class":plan["result_class"],"table_5_claim":False,"test_loaded":False,
                  "plan_sha256":plan_sha,"model":args.model,"baseline":baseline,"split":split,
                  "selected_family":selected,"selection_uses_validation_labels":False,
                  "choices":choices,"posthoc_all_images":exploratory,"per_image":evaluated,
                  "elapsed_scoring_seconds":sum(s["runtime"]["seconds"] for s in shards),
                  "cuda_peak":{k:max((s["runtime"]["cuda_peak"][k] for s in shards if s["runtime"]["cuda_peak"][k] is not None),default=None)
                               for k in ("peak_allocated_bytes","peak_reserved_bytes")},
                  "limitation":"Repeatedly used TRAIN holdout; small accepted image samples; no safety probability guarantee; DeepSolo not executed here"}
        seal(root/"results.json",result)
        route_families = set(choices) if args.export_all_families else {selected}
        routes = {}
        for family in sorted(route_families):
            threshold = choices[family]["threshold"]
            routes[family] = [{**s,"action":"accept_parseq" if accepted(scores[s["sample_id"]][family],threshold) else "deepsolo",
                              "score":scores[s["sample_id"]][family]} for s in observed]
        seal(root/"routing_requests.json", {"plan_sha256":plan_sha,"result_sha256":sha(root/"results.json"),
              "selected_family":selected,"families":routes,"thresholds":{f:choices[f]["threshold"] for f in routes},
              "fallback_status":"NOT_EXECUTED","selection_uses_validation_labels":False})
        print(json.dumps({"report":str(root/"results.json"),"family":selected,"threshold":choices[selected]["threshold"],
                          "calibration":choices[selected]["calibration"],"validation":choices[selected]["validation"],
                          "all_images":choices[selected]["all_images"]}),flush=True)
        return root
    except Exception as error:
        write_json(root/"failure.json",{"status":"FAILED","stage":stage,"error_type":type(error).__name__,
                   "error":str(error),"traceback":traceback.format_exc(),"out_of_memory":isinstance(error,(torch.cuda.OutOfMemoryError,MemoryError)),
                   "device":args.device,"precision":"float32","batch_size":cfg["execution"]["batch_size"],"cuda_peak":_cuda_peak(args.device)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=["approximate"])
    p.add_argument("--config",type=Path,default=CONFIG); p.add_argument("--paths",type=Path,default=PATHS)
    p.add_argument("--model",choices=["vit_b32","vit_b16","vit_l14"],default="vit_l14")
    p.add_argument("--device",choices=["cpu","cuda:0"],default="cuda:0")
    p.add_argument("--limit",type=int); p.add_argument("--check",action="store_true")
    p.add_argument("--output-root",type=Path); p.add_argument("--export-all-families",action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
