"""Frozen Total-Text TEST comparison: CRAFT/DBNet/FAST regions -> genuine PARSeq."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import torch

from localize.external_text_detectors import DetectorRuntime, crop_region
from localize.independent_word_slots import overlap_iou
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer, verify_parseq_runtime_source)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import maximum_matching
from recognition.yolo_word_refinement import arm_predictions, ocr_quality
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import _cuda_peak, array_sha, cached, read_image, sha, write_json
from scripts.evaluate_totaltext_yolo_direct_parseq import metrics
from scripts.prepare_totaltext_text_detectors import DEFAULT_CONFIG, DEFAULT_PATHS, artifact_source, verify_weight
from scripts.run_crop_group_word_selection import evaluate_predictions, normalize
from segmentation.data import load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent

REPO = Path(__file__).resolve().parents[1]
CODE = ("localize/external_text_detectors.py", "scripts/evaluate_totaltext_text_detectors.py",
    "scripts/prepare_totaltext_text_detectors.py", "recognition/parseq.py",
    "recognition/yolo_word_refinement.py", "recognition/group_word_selection.py",
    "recognition/scene_text_datasets.py", "recognition/scene_text_v1_evaluation.py",
    "scripts/run_crop_group_word_selection.py", "scripts/crop_clip_runtime.py",
    "segmentation/scene_text_preparation.py", "segmentation/data.py",
    "scripts/evaluate_totaltext_yolo_direct_parseq.py", "localize/independent_word_slots.py",
    "scripts/test_totaltext_text_detectors.sh", "environments/text-detectors/requirements.txt",
    "models/unet_adv.py")


def installed_package_inventory(distributions=None) -> dict:
    """Keep malformed metadata explicit without sorting None alongside strings.

    Required runtime versions are checked separately and remain fail-closed.
    Incomplete, unrelated egg-info directories must not block provenance capture.
    """
    packages, anomalies = set(), []
    for distribution in importlib.metadata.distributions() if distributions is None else distributions:
        name, version = distribution.metadata.get("Name"), distribution.version
        if isinstance(name,str) and name.strip() and isinstance(version,str) and version.strip():
            packages.add((name,version))
        else:
            metadata_path = getattr(distribution,"_path",None)
            anomalies.append({"name":name,"version":version,
                "metadata_path":str(metadata_path) if metadata_path is not None else None,
                "issue":"missing or empty package Name/Version metadata"})
    # Use JSON-native lists so immutable configuration comparison also works
    # after serialization and a later cached rerun (tuples would differ).
    return {"packages":[list(package) for package in sorted(packages)],
            "metadata_anomalies":sorted(anomalies,key=lambda row:
                (row["metadata_path"] or "",str(row["name"]),str(row["version"])))}


def validate_config(config: dict):
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("dataset") != "total_text" or config.get("split") != "TEST"
        or config.get("table_5_claim") is not False
        or config.get("crop_views") != ["axis_aligned", "rectified"]
        or config["execution"]["precision"] != "float32"
        or config["execution"]["detector_batch_size"] != 1
        or set(config["models"]) != {"craft","db_resnet50","fast_tiny"}
        or type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32):
        raise ValueError("invalid isolated approximate detector configuration")
    for key in ("cpu_threads", "parseq_batch_size"):
        if type(config["execution"][key]) is not int or config["execution"][key] < 1:
            raise ValueError("batch size and threads must be positive integers")
    for model in config["models"].values():
        floor = model.get("text_threshold",model.get("box_thresh"))
        controls = model["score_controls"]
        if (not controls or len(set(controls)) != len(controls)
            or any(type(x) not in (float,int) or not floor <= x <= 1 for x in controls)
            or model["reference_score"] not in controls):
            raise ValueError("detector score controls must respect native floor")
        for key in ("canvas_size", "input_size"):
            if key in model and (type(model[key]) is not int or model[key] < 32):
                raise ValueError("detector input dimensions must be positive integers >=32")
    craft = config["models"]["craft"]
    if (not 0 < craft["low_text"] <= craft["text_threshold"] <= 1
        or not 0 < craft["link_threshold"] < 1 or not 0 < craft["mag_ratio"] <= 4):
        raise ValueError("invalid CRAFT heatmap thresholds or magnification")
    for name in ("db_resnet50","fast_tiny"):
        if not 0 < config["models"][name]["bin_thresh"] < 1:
            raise ValueError("invalid docTR heatmap threshold")
    for value in [config["evaluation"]["matching_iou"],config["evaluation"]["same_text_nms_iou"],
                  *config["evaluation"]["joint_controls"]]:
        if not 0 < value < 1:
            raise ValueError("evaluation and admission thresholds must be in (0,1)")


def annotation_path(data_root: Path, sample) -> Path:
    return (data_root / sample.image_path).parent.parent / "annotations" / f"{sample.sample_id.split(':')[1]}.txt"


def care_ground_truth(path: Path, image_wh: tuple[int,int]) -> tuple[list[dict],dict]:
    # Official legacy records include edge points just beyond the stored raster.
    # Match the existing qualified preparer: parse, then explicitly clip pixels.
    instances = parse_totaltext_annotation(path)
    instances, clipped = clip_instances_to_pixel_extent(instances, image_wh)
    care = []
    for word in instances:
        if not word.ignore:
            x1,y1,x2,y2 = word.bbox_xyxy
            care.append({"bbox": [x1,y1,x2+1,y2+1], "text": word.transcription})
    return care, {"ignored_words": sum(word.ignore for word in instances), "clipped_polygons": clipped}


def select_arms(rows: list[dict], config: dict, model: dict) -> dict[str,list[dict]]:
    cfg = {"score_controls": model["score_controls"], "reference_score": model["reference_score"],
           "joint_controls": config["evaluation"]["joint_controls"],
           "merge_iou": config["evaluation"]["same_text_nms_iou"]}
    by_view = {view: [row for row in rows if row["view"] == view] for view in config["crop_views"]}
    by_region = {}
    for row in rows:
        by_region.setdefault(row["source_index"], []).append(row)
    # One GT-blind prediction per region. Tie order favors unchanged pixels.
    by_view["best_ocr"] = [max(group,key=ocr_quality) for group in by_region.values()]
    return {f"{view}/{arm}": predictions for view,group in by_view.items()
            for arm,predictions in arm_predictions(group,cfg).items()}


def run_model(name: str, config: dict, paths: dict, samples: list, device: str,
              output: Path, config_path: Path, paths_path: Path) -> dict:
    model_cfg = config["models"][name]
    artifacts = Path(paths["artifacts_root"])
    artifact = json.loads((artifacts / f"{name}.json").read_text())
    checkpoint = artifacts / model_cfg["filename"]
    if (artifact["name"] != name or artifact["source"] != artifact_source(model_cfg)
        or artifact["sha256"] != verify_weight(checkpoint, model_cfg)):
        raise RuntimeError("frozen detector artifact identity differs")
    archive, runtime, parseq_weight = (Path(paths[k]) for k in
        ("parseq_archive", "parseq_source", "parseq_checkpoint"))
    if sha(archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256 or sha(parseq_weight) != PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError("qualified PARSeq artifact differs")
    identity = verify_parseq_runtime_source(runtime, source_type=PARSEQ_ARCHIVE_SOURCE, source_archive=archive)
    data_root = Path(paths["data_root"])
    ann_hashes = {s.sample_id: sha(annotation_path(data_root,s)) for s in samples}
    packages = {package: importlib.metadata.version(package) for package in
                ("torch","torchvision","numpy","Pillow",model_cfg["package"])}
    if packages[model_cfg["package"]] != model_cfg["version"]:
        raise RuntimeError("detector package version differs")
    inventory = installed_package_inventory()
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction; exploratory external checkpoint comparison",
        "result_class": "APPROXIMATE_TEST_DIAGNOSTIC" if len(samples) == 300 else
                        "SMOKE_TEST" if len(samples) == 1 else "PARTIAL_DATASET",
        "table_5_claim": False, "config": config, "config_sha256": sha(config_path),
        "paths": paths, "paths_sha256": sha(paths_path), "model": name, "detector_artifact": artifact,
        "parseq_runtime": identity, "parseq_checkpoint_sha256": sha(parseq_weight),
        "manifest_sha256": sha(Path(paths["test_manifest"])), "annotations_sha256": ann_hashes,
        "sample_ids": [s.sample_id for s in samples], "device": device,
        "precision": "float32", "detector_batch_size": 1,
        "parseq_batch_size": config["execution"]["parseq_batch_size"],
        "torch_threads": torch.get_num_threads(), "packages": packages,
        "installed_packages":inventory["packages"],
        "package_metadata_anomalies":inventory["metadata_anomalies"],
        "software_tree_sha256": {}, "python": sys.version, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "hardware_deviation": "sequential model loading; single device; FP32; recorded batches and inputs; CPU when explicitly selected",
        "ground_truth_use": "only raw annotation checksums before inference; parsed after detector and OCR freeze",
        "code_sha256": {file: sha(REPO / file) for file in CODE}}
    # Record the exact installed implementation, not only the version label.
    package_name = "easyocr" if name == "craft" else "doctr"
    package_root = Path(importlib.util.find_spec(package_name).origin).parent
    plan["software_tree_sha256"] = {str(p.relative_to(package_root)): sha(p)
                                   for p in sorted(package_root.rglob("*.py"))}
    freeze(output / "resolved_configuration.json", plan)
    plan_hash = sha(output / "resolved_configuration.json")
    detector = recognizer = None
    stage = "detection"
    shape = None
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for index,sample in enumerate(samples,1):
            sid = sample.sample_id
            path = output / "detector" / f"{sid.replace(':','__')}.json"
            contract = {"plan_sha256": plan_hash, "image_sha256": sample.image_sha256}
            shard = cached(path,contract)
            if shard is None:
                image,image_sha = read_image(data_root,sample,sample.image_sha256)
                started = time.perf_counter()
                if detector is None:
                    detector = DetectorRuntime(name,model_cfg,checkpoint,device)
                regions = detector.infer(image)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                shape = detector.last_input_shape
                shard = {"contract": contract, "sample_id": sid, "image_sha256": image_sha,
                    "image_wh": [image.shape[1],image.shape[0]], "regions": regions,
                    "seconds": time.perf_counter()-started, "input_shape": shape,
                    "parameters": detector.parameters,
                    "parameters_before_reparameterize": detector.parameters_before_reparameterize,
                    "model_peak": _cuda_peak(device), "ground_truth_use": "none"}
                write_json(path,shard)
            print(f"[{name}] detector {index}/{len(samples)} {sid} regions={len(shard['regions'])}",flush=True)
        if detector is not None:
            detector.release()
            detector = None
        stage = "parseq"
        batch = config["execution"]["parseq_batch_size"]
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for index,sample in enumerate(samples,1):
            key = sample.sample_id.replace(':','__')
            det_path = output / "detector" / f"{key}.json"
            shard_path = output / "ocr" / f"{key}.json"
            contract = {"plan_sha256": plan_hash, "detector_sha256": sha(det_path)}
            shard = cached(shard_path,contract)
            if shard is None:
                image,image_sha = read_image(data_root,sample,sample.image_sha256)
                regions = json.loads(det_path.read_text())["regions"]
                requests = [(row,view) for row in regions if row["valid"] for view in config["crop_views"]]
                rows = []
                started = time.perf_counter()
                for offset in range(0,len(requests),batch):
                    group = requests[offset:offset+batch]
                    crops = [crop_region(image,row,view) for row,view in group]
                    if recognizer is None:
                        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                            runtime_source=runtime, runtime_source_archive=archive,
                            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,checkpoint=parseq_weight,device=device))
                    shape = [len(group),3,32,128]
                    values = recognizer.recognize_candidate_batch(crops)
                    if len(values) != len(group):
                        raise RuntimeError("incomplete genuine PARSeq batch")
                    for (row,view),crop,value in zip(group,crops,values):
                        rows.append({"candidate_id": f"{sample.sample_id}:{name}:{row['source_index']}:{view}",
                            "source_index": row["source_index"], "bbox": row["bbox"],
                            "polygon": row["polygon"], "view": view, "detector_score": row["score"],
                            "crop_shape": list(crop.shape), "crop_rgb_sha256": array_sha(crop),**value.to_dict()})
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                shard = {"contract": contract, "sample_id": sample.sample_id, "image_sha256": image_sha,
                    "rows": rows, "seconds": time.perf_counter()-started, "model_peak": _cuda_peak(device),
                    "recognizer_input_shape": shape if requests else None, "ground_truth_use": "none"}
                write_json(shard_path,shard)
            print(f"[{name}] PARSeq {index}/{len(samples)} {sample.sample_id} crops={len(shard['rows'])}",flush=True)
    except Exception as error:
        if stage == "detection" and detector is not None:
            shape = detector.last_input_shape
        write_json(output / "failure.json", {"status": "FAILED", "stage": stage,
            "reproduction_mode": "approximate", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "out_of_memory": isinstance(error,torch.cuda.OutOfMemoryError),
            "device": device, "precision": "float32", "last_input_shape": shape,
            "model_set": [name] if stage == "detection" else ["qualified PARSeq"],
            "batch_size": 1 if stage == "detection" else config["execution"]["parseq_batch_size"],
            "model_peak": _cuda_peak(device)})
        raise
    finally:
        if detector is not None:
            detector.release()
        if recognizer is not None:
            recognizer.release()
            recognizer = None
        import gc
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    totals,diagnostics = {},Counter()
    for sample in samples:
        key = sample.sample_id.replace(':','__')
        det_path,ocr_path = output / "detector" / f"{key}.json",output / "ocr" / f"{key}.json"
        det,ocr = json.loads(det_path.read_text()),json.loads(ocr_path.read_text())
        annotation = annotation_path(data_root,sample)
        if sha(annotation) != ann_hashes[sample.sample_id]:
            raise RuntimeError("annotation changed after inference")
        gt,ann_diag = care_ground_truth(annotation,tuple(det["image_wh"]))
        diagnostics.update(gt_words=len(gt),gt_nonempty=sum(bool(normalize(g["text"])) for g in gt),
            generated_regions=len(det["regions"]),rejected_regions=sum(not r["valid"] for r in det["regions"]),
            ocr_crops=len(ocr["rows"]),detector_seconds=det["seconds"],ocr_seconds=ocr["seconds"],**ann_diag)
        outcomes = {}
        arms = select_arms(ocr["rows"],config,model_cfg)
        for arm,predictions in arms.items():
            score = evaluate_predictions(predictions,gt,config["evaluation"]["matching_iou"])
            edges = [(pi,gi) for pi,p in enumerate(predictions) for gi,g in enumerate(gt)
                     if overlap_iou(p["bbox"],g["bbox"]) >= config["evaluation"]["matching_iou"]]
            empty = sum(not normalize(p["raw_text"]) for p in predictions)
            geometry = len(maximum_matching(len(predictions),len(gt),edges))
            totals.setdefault(arm,Counter()).update(correct=score["correct"],outputs=len(predictions),
                normalized_empty_outputs=empty,geometry_matches=geometry)
            outcomes[arm] = {**score,"geometry_matches":geometry,"normalized_empty_outputs":empty,
                             "selected_ids":[p["candidate_id"] for p in predictions]}
        freeze(output / "evaluation" / f"{key}.json", {"sample_id":sample.sample_id,
            "ground_truth":gt,"annotation_sha256":ann_hashes[sample.sample_id],
            "detector_sha256":sha(det_path),"ocr_sha256":sha(ocr_path),"arms":outcomes})
    peaks = {}
    for stage_name in ("detector","ocr"):
        memory = [json.loads((output / stage_name / f"{s.sample_id.replace(':','__')}.json").read_text())["model_peak"]
                  for s in samples]
        peaks[stage_name] = {k:max((r[k] for r in memory if r[k] is not None),default=None) for k in memory[0]}
    example = json.loads((output / "detector" / f"{samples[0].sample_id.replace(':','__')}.json").read_text())
    result = {"schema_version":1,"reproduction_mode":"approximate","result_class":plan["result_class"],
        "claim_label":plan["claim_label"],"table_5_claim":False,"model":name,"plan_sha256":plan_hash,
        "images":len(samples),**diagnostics,"parameters":example["parameters"],
        "parameters_before_reparameterize":example["parameters_before_reparameterize"],
        "practical_unet_reference_parameters":58155081,"checkpoint_bytes":checkpoint.stat().st_size,
        "stage_peak":peaks,"device":device,
        "summary":{arm:{**metrics(v["correct"],v["outputs"],diagnostics["gt_words"]),
                        "normalized_empty_outputs":v["normalized_empty_outputs"],
                        "geometry_matches":v["geometry_matches"]} for arm,v in totals.items()},
        "metric":{"normalization":"Unicode casefold; punctuation/symbols removed; whitespace preserved",
            "matching":"maximum-cardinality one-to-one normalized word and original envelope IoU",
            "matching_iou":config["evaluation"]["matching_iou"],
            "aggregation":"micro counts, every selected image and care GT retained",
            "ignore":"ignored GT excluded; their overlapping predictions count as FP to match earlier runs",
            "empty":"raw arms retain empty predictions as FP; nonempty/joint arms explicitly remove them",
            "protocol":"local bbox spotting proxy; not official Total-Text polygon Hmean"},
        "limitation":"scores differ between detectors; defaults and predeclared controls are exploratory, not TEST-tuned validation"}
    freeze(output / "results.json",result)
    return result


def write_comparison(root: Path, results: dict):
    """All controls, no winner fitted to TEST; English Markdown + machine CSV."""
    fields = ["model","arm","parameters","correct","outputs","gt_words","precision","recall","hmean"]
    with (root / "comparison.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for name,result in results.items():
            for arm,score in result["summary"].items():
                writer.writerow({"model":name,"arm":arm,"parameters":result["parameters"],
                                 **{field:score[field] for field in fields[3:]}})
    lines = ["# Total-Text detector + PARSeq comparison", "",
        "Approximate reproduction; exploratory external checkpoint comparison.", "",
        "Casefold and punctuation/symbol removal. One-to-one word + bbox IoU matching.",
        "Local Hmean proxy, not the official polygon spotting evaluation.", "",
        "| Model | Crop / selector | Correct | Outputs | Precision | Recall | Hmean |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name,result in results.items():
        for arm,score in result["summary"].items():
            lines.append(f"| {name} | {arm} | {score['correct']} | {score['outputs']} | "
                f"{score['precision']:.2%} | {score['recall']:.2%} | {score['hmean']:.2%} |")
    lines += ["", "## Model size and execution", "",
        "| Model | Inference parameters | Checkpoint MiB | Detector seconds | PARSeq seconds |",
        "|---|---:|---:|---:|---:|"]
    for name,result in results.items():
        lines.append(f"| {name} | {result['parameters']:,} | {result['checkpoint_bytes']/2**20:.1f} | "
            f"{result['detector_seconds']:.1f} | {result['ocr_seconds']:.1f} |")
    lines += ["", "Project practical U-Net reference: 58,155,081 parameters.",
        "Times include first model loading, preprocessing and postprocessing; not warmed latency.",
        "Native input sizes and confidence definitions differ between detectors.",
        "CUDA allocated/reserved peaks, input shapes and complete readings are in JSON shards.",
        "Pretrained docTR training overlap with Total-Text remains unqualified.", ""]
    (root / "comparison.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--paths",type=Path,default=DEFAULT_PATHS)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    parser.add_argument("--models",nargs="+",choices=("craft","db_resnet50","fast_tiny"))
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("scientific configuration must be in configs/approximate")
    cfg,paths = json.loads(args.config.read_text()),json.loads(args.paths.read_text())
    validate_config(cfg)
    if paths.get("schema_version") != 1 or paths.get("reproduction_mode") != "approximate":
        raise ValueError("invalid approximate paths profile")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select --device cpu for CPU experiment")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    manifest = load_segmentation_manifest(Path(paths["test_manifest"]),require_checksums=True)
    if len(manifest.samples) != 300 or len({s.sample_id for s in manifest.samples}) != 300:
        raise RuntimeError("expected complete official Total-Text TEST-300 manifest")
    if any(not s.sample_id.startswith("total_text:") for s in manifest.samples):
        raise ValueError("manifest is not Total-Text")
    samples = list(manifest.samples)[:args.limit] if args.limit else list(manifest.samples)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    root = Path(paths["output_root"]) / args.device.replace(':','-') / (f"smoke-{len(samples)}" if args.limit else "full")
    if root.resolve() == REPO or REPO in root.resolve().parents:
        raise ValueError("generated outputs must stay outside the repository")
    results = {}
    for name in args.models or list(cfg["models"]):
        results[name] = run_model(name,cfg,paths,samples,args.device,root / name,args.config,args.paths)
        print(json.dumps({"model":name,"report":str(root/name/"results.json"),"summary":results[name]["summary"]}),flush=True)
    # Deterministic aggregation permits a completed model to resume after another failed.
    combined = {"reproduction_mode":"approximate","result_class":next(iter(results.values()))["result_class"],
               "images":len(samples),"device":args.device,"models":results}
    filename = "results.json" if set(results) == set(cfg["models"]) else "results-" + "-".join(sorted(results)) + ".json"
    freeze(root / filename,combined)
    write_comparison(root,results)
    print(f"Text detector reports: {root / filename}",flush=True)


if __name__ == "__main__":
    main()
