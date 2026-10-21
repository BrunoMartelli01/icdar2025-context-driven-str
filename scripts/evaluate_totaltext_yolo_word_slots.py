#!/usr/bin/env python3
"""Evaluate a pinned YOLO word-object checkpoint as slots for frozen PARSeq crops."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from localize.independent_word_slots import assign_to_slots, overlap_iou
from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import sha
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_yolo_word_slots_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json"
CODE = ("scripts/evaluate_totaltext_yolo_word_slots.py",
        "localize/independent_word_slots.py", "recognition/scene_text_v1_evaluation.py",
        "scripts/run_crop_group_word_selection.py", "segmentation/data.py")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pil_to_yolo_bgr(image: Image.Image) -> np.ndarray:
    """Ultralytics 8.1.0 expects HWC BGR arrays and converts them to RGB."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])


def git_blob_sha1(path: Path) -> str:
    digest = sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path, source: dict) -> dict:
    if not path.is_file():
        url = (f"https://raw.githubusercontent.com/{source['repository']}/"
               f"{source['commit']}/{source['path']}")
        raise FileNotFoundError(f"YOLO word checkpoint missing: {path}; download {url}")
    size = path.stat().st_size
    if size != source["size_bytes"]:
        raise ValueError(f"YOLO checkpoint size differs: {size} != {source['size_bytes']}")
    blob = git_blob_sha1(path)
    if blob != source["git_blob_sha1"]:
        raise ValueError(f"YOLO checkpoint Git blob SHA1 differs: {blob}")
    return {"path": str(path.resolve()), "size_bytes": size, "git_blob_sha1": blob,
            "sha256": sha(path), "source": source}


def validate_config(config: dict) -> None:
    execution = config.get("inference", {})
    source = config.get("checkpoint_source", {})
    scores = config.get("score_controls")
    ious = config.get("crop_slot_iou_controls")
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("table_5_claim") is not False
        or config.get("ultralytics_version") != "8.1.0"
        or not isinstance(config.get("seed"), int) or isinstance(config.get("seed"), bool)
        or config["seed"] < 0
        or config.get("source_selection") != "guarded_consolidated"
        or not isinstance(config.get("evaluation_matching_iou"), (int, float))
        or not 0 < config["evaluation_matching_iou"] <= 1
        or set(execution) != {"device", "precision", "batch_size", "image_size",
                              "confidence_floor", "nms_iou", "max_detections"}
        or execution["device"] != "cuda:0" or execution["precision"] != "float32"
        or execution["batch_size"] != 1
        or not isinstance(execution["image_size"], int) or execution["image_size"] <= 0
        or not isinstance(execution["max_detections"], int)
        or execution["max_detections"] <= 0
        or not 0 < execution["confidence_floor"] < 1
        or not 0 < execution["nms_iou"] < 1
        or not isinstance(scores, list) or not scores or len(scores) != len(set(scores))
        or any(not isinstance(x, (int, float)) or not execution["confidence_floor"] <= x <= 1
               for x in scores)
        or not isinstance(ious, list) or not ious or len(ious) != len(set(ious))
        or any(not isinstance(x, (int, float)) or not 0 < x <= 1 for x in ious)
        or set(source) != {"repository", "commit", "path", "git_blob_sha1", "size_bytes"}
        or not isinstance(source["commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is None
        or not isinstance(source["git_blob_sha1"], str)
        or re.fullmatch(r"[0-9a-f]{40}", source["git_blob_sha1"]) is None
        or not isinstance(source["size_bytes"], int) or source["size_bytes"] <= 0):
        raise ValueError("invalid isolated approximate YOLO word-slot config")


def boxes_to_slots(xyxy: np.ndarray, confidence: np.ndarray, classes: np.ndarray,
                   image_wh: tuple[int, int], confidence_floor: float
                   ) -> tuple[list[dict], list[dict]]:
    """Convert one-class object boxes to original-raster xyxy slots."""
    xyxy = np.asarray(xyxy, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.float64)
    if (xyxy.ndim != 2 or xyxy.shape[1] != 4 or confidence.shape != (len(xyxy),)
        or classes.shape != (len(xyxy),) or not np.isfinite(xyxy).all()
        or not np.isfinite(confidence).all() or not np.isfinite(classes).all()
        or np.any(xyxy[:, 2] <= xyxy[:, 0]) or np.any(xyxy[:, 3] <= xyxy[:, 1])
        or not np.all(classes == 0) or not np.all((confidence >= confidence_floor) &
                                                  (confidence <= 1))):
        raise ValueError("invalid YOLO one-class detection output")
    width, height = image_wh
    if width <= 0 or height <= 0:
        raise ValueError("invalid image size")
    slots, rejected = [], []
    for index, (box, score) in enumerate(zip(xyxy, confidence)):
        clipped = [max(0, min(width, math.floor(box[0]))),
                   max(0, min(height, math.floor(box[1]))),
                   max(0, min(width, math.ceil(box[2]))),
                   max(0, min(height, math.ceil(box[3])))]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            rejected.append({"source_index": index, "score": float(score),
                             "reason": "no_positive_in_image_bbox",
                             "unclipped_bbox": box.tolist(), "clipped_bbox": clipped})
            continue
        slots.append({"source_index": index, "score": float(score), "bbox": clipped})
    return slots, rejected


def label(score: float, iou: float) -> str:
    return f"score{round(score * 1000):03d}_iou{round(iou * 1000):03d}"


def record_failure(output: Path, stage: str, error: Exception, execution: dict,
                   image_wh: tuple[int, int] | None = None) -> None:
    device = execution["device"]
    freeze(output / "failure.json", {
        "schema_version": 1, "reproduction_mode": "approximate",
        "status": "FAILED", "stage": stage, "error_type": type(error).__name__,
        "error": str(error), "traceback": traceback.format_exc(),
        "out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
        "device": device, "precision": execution["precision"],
        "batch_size": execution["batch_size"], "original_image_wh": image_wh,
        "input_image_size": execution["image_size"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    })


def run(args: argparse.Namespace) -> Path:
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("YOLO scientific config must be in configs/approximate")
    config = read_json(args.config)
    validate_config(config)
    paths = read_json(args.paths)
    if paths.get("reproduction_mode") != "approximate":
        raise ValueError("source paths profile is not approximate")
    frozen = Path(paths["frozen_refinement_run"]).resolve()
    frozen_plan = read_json(frozen / "resolved_configuration.json")
    all_ids = frozen_plan["sample_ids"]
    if (len(all_ids) != 300 or len(set(all_ids)) != len(all_ids)
        or sha(Path(paths["test_manifest"])) != frozen_plan["manifest_sha256"]):
        raise RuntimeError("frozen 300-image Total-Text TEST manifest differs")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    ids = all_ids[:args.limit] if args.limit else all_ids
    checkpoint = validate_checkpoint(args.checkpoint, config["checkpoint_source"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; YOLO inference requires the GPU host")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is absent; use an explicitly authorized isolated runtime") from exc
    runtime_version = importlib.metadata.version("ultralytics")
    if runtime_version != config["ultralytics_version"]:
        raise RuntimeError(f"Ultralytics version differs: {runtime_version} != "
                           f"{config['ultralytics_version']}")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    source = Path(frozen_plan["source"]).resolve()
    source_plan = read_json(source / "resolved_configuration.json")
    if source_plan["manifest_sha256"] != frozen_plan["manifest_sha256"]:
        raise RuntimeError("source GT manifest differs")
    data_root = Path(source_plan["paths"]["data_root"]).resolve()
    samples = {sample.sample_id: sample for sample in load_segmentation_manifest(
        Path(paths["test_manifest"]), require_checksums=True).samples}
    images = {}
    for sid in ids:
        sample = samples[sid]
        path = (data_root / sample.image_path).resolve()
        if data_root not in path.parents or sha(path) != sample.image_sha256:
            raise RuntimeError(f"Total-Text input image identity differs: {sid}")
        with Image.open(path) as image:
            images[sid] = (path, image.size)

    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("generated detector results must stay outside the repository")
    execution = config["inference"]
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TEST_DIAGNOSTIC",
            "table_5_claim": False, "config": config, "config_sha256": sha(args.config),
            "local_paths_sha256": sha(args.paths), "frozen_run": str(frozen),
            "frozen_plan_sha256": sha(frozen / "resolved_configuration.json"),
            "source_plan_sha256": sha(source / "resolved_configuration.json"),
            "manifest_sha256": frozen_plan["manifest_sha256"], "sample_ids": ids,
            "checkpoint": checkpoint,
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "seed": config["seed"],
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "ultralytics": runtime_version,
            "gpu": torch.cuda.get_device_name(0),
            "code_sha256": {name: sha(REPO / name) for name in CODE},
            "coordinate_frame": "stored PIL raster, BGR numpy input, original xyxy output",
            "ground_truth_use": "after detector and assignment shards are frozen"}
    freeze(output / "resolved_configuration.json", plan)
    plan_hash = sha(output / "resolved_configuration.json")
    torch.cuda.reset_peak_memory_stats(execution["device"])
    try:
        model = YOLO(str(args.checkpoint))
    except Exception as exc:
        record_failure(output, "model_load", exc, execution)
        raise
    if len(model.names) != 1 or str(model.names[0]).lower() != "text":
        raise RuntimeError(f"checkpoint is not a one-class word/text detector: {model.names}")
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        image_path, image_wh = images[sid]
        shard_path = output / "detector" / f"{key}.json"
        if shard_path.is_file():
            shard = read_json(shard_path)
            if (shard.get("plan_sha256") != plan_hash or shard.get("sample_id") != sid
                or shard.get("image_sha256") != sha(image_path)):
                raise RuntimeError(f"cached YOLO detection differs: {sid}")
        else:
            with Image.open(image_path) as image:
                bgr = pil_to_yolo_bgr(image)
            start = time.perf_counter()
            try:
                result = model.predict(bgr, imgsz=execution["image_size"],
                                       conf=execution["confidence_floor"],
                                       iou=execution["nms_iou"],
                                       max_det=execution["max_detections"],
                                       device=execution["device"], half=False,
                                       verbose=False)[0]
                torch.cuda.synchronize()
            except Exception as exc:
                record_failure(output, f"inference:{sid}", exc, execution, image_wh)
                raise
            latency = time.perf_counter() - start
            if tuple(result.orig_shape) != (image_wh[1], image_wh[0]):
                raise RuntimeError(f"YOLO output image shape differs: {sid}")
            boxes = result.boxes
            slots, rejected = boxes_to_slots(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy(), image_wh, execution["confidence_floor"])
            shard = {"sample_id": sid, "plan_sha256": plan_hash,
                     "image_sha256": sha(image_path), "image_wh": list(image_wh),
                     "slots": slots, "rejected": rejected,
                     "raw_detections": len(boxes), "latency_seconds": latency,
                     "ground_truth_use": "none"}
            freeze(shard_path, shard)
        if position % 25 == 0 or position == len(ids):
            print(f"[YOLO detections] {position}/{len(ids)}", flush=True)
    memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(execution["device"]),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(execution["device"])}
    del model
    torch.cuda.empty_cache()

    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        detector = read_json(output / "detector" / f"{key}.json")
        frozen_selection_path = frozen / "selection" / f"{key}.json"
        predictions = read_json(frozen_selection_path)["methods"][config["source_selection"]]
        arms = {}
        for score in config["score_controls"]:
            slots = [slot for slot in detector["slots"] if slot["score"] >= score]
            for iou in config["crop_slot_iou_controls"]:
                selected, assignments, memberships = assign_to_slots(
                    predictions, slots, min_iou=iou)
                covered_ids = {row["candidate_id"] for group in memberships
                               for row in group["eligible_candidates"]}
                covered = [row for row in predictions if row["candidate_id"] in covered_ids]
                arms[label(score, iou)] = {"slots": slots, "selected": selected,
                                           "covered": covered, "assignments": assignments,
                                           "memberships": memberships}
        freeze(output / "selection" / f"{key}.json", {
            "sample_id": sid, "plan_sha256": plan_hash,
            "detector_sha256": sha(output / "detector" / f"{key}.json"),
            "frozen_selection_sha256": sha(frozen_selection_path),
            "original_predictions": predictions, "arms": arms,
            "ground_truth_use": "none"})
        if position % 25 == 0 or position == len(ids):
            print(f"[YOLO slots] {position}/{len(ids)}", flush=True)

    denominator = 0
    totals = {label(score, iou): Counter() for score in config["score_controls"]
              for iou in config["crop_slot_iou_controls"]}
    baseline = Counter()
    geometry = Counter()
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        selection_path = output / "selection" / f"{key}.json"
        selection = read_json(selection_path)
        detector = read_json(output / "detector" / f"{key}.json")
        gt = read_json(source / "evaluation" / f"{key}.json")["ground_truth"]
        denominator += len(gt)
        original_score = evaluate_predictions(
            selection["original_predictions"], gt, config["evaluation_matching_iou"])
        baseline.update(outputs=original_score["predictions"], correct=original_score["correct"])
        geometry.update(raw_detections=detector["raw_detections"],
                        valid_slots=len(detector["slots"]),
                        rejected_detections=len(detector["rejected"]))
        arm_scores = {}
        for name, arm in selection["arms"].items():
            selected = evaluate_predictions(arm["selected"], gt,
                                            config["evaluation_matching_iou"])
            covered = evaluate_predictions(arm["covered"], gt,
                                           config["evaluation_matching_iou"])
            edges = [(si, gi) for si, slot in enumerate(arm["slots"])
                     for gi, word in enumerate(gt)
                     if overlap_iou(slot["bbox"], word["bbox"]) >=
                     config["evaluation_matching_iou"]]
            box_matches = len(maximum_matching(len(arm["slots"]), len(gt), edges))
            totals[name].update(slots=len(arm["slots"]), outputs=selected["predictions"],
                                correct=selected["correct"],
                                covered_outputs=covered["predictions"],
                                covered_correct=covered["correct"],
                                detector_gt_box_matches=box_matches)
            arm_scores[name] = {"selected": selected, "covered": covered,
                                "detector_gt_box_matches": box_matches}
        freeze(output / "evaluation" / f"{key}.json", {
            "sample_id": sid, "selection_sha256": sha(selection_path),
            "gt_words": len(gt), "original": original_score, "arms": arm_scores})
        if position % 25 == 0 or position == len(ids):
            print(f"[YOLO evaluation] {position}/{len(ids)}", flush=True)
    summary = {}
    for name, row in totals.items():
        precision = row["correct"] / row["outputs"] if row["outputs"] else 0.0
        recall = row["correct"] / denominator if denominator else 0.0
        summary[name] = {**dict(row), "precision_proxy": precision,
                         "recall_proxy": recall,
                         "hmean_proxy": 2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0}
    report = {"schema_version": 1, "reproduction_mode": "approximate",
              "result_class": plan["result_class"], "table_5_claim": False,
              "plan_sha256": plan_hash, "images": len(ids), "gt_words": denominator,
              "baseline": dict(baseline), "detector_geometry": dict(geometry),
              "gpu_peak": memory, "summary": summary,
              "metric": "Unicode casefold/P-S stripped exact reading; bbox IoU>=0.5; one-to-one",
              "limitation": "local word/bbox Hmean proxy, not official Total-Text polygon Hmean"}
    freeze(output / "results.json", report)
    print(json.dumps({"report": str(output / "results.json"), "summary": summary,
                      "gpu_peak": memory}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
