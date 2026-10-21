"""Recognize frozen YOLO word boxes directly with genuine PARSeq; no group selection."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import torch

from localize.independent_word_slots import overlap_iou
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
    verify_parseq_runtime_source)
from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import _cuda_peak, array_sha, cached, read_image, sha, write_json
from scripts.evaluate_totaltext_yolo_word_slots import DEFAULT_PATHS, read_json
from scripts.run_crop_group_word_selection import evaluate_predictions, normalize
from segmentation.data import load_segmentation_manifest

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_yolo_direct_parseq_v1.json"
CODE = ("scripts/evaluate_totaltext_yolo_direct_parseq.py", "recognition/parseq.py",
        "scripts/crop_clip_runtime.py", "scripts/run_crop_group_word_selection.py",
        "recognition/scene_text_v1_evaluation.py", "segmentation/data.py")


def validate_config(config: dict) -> None:
    execution = config.get("execution", {})
    scores = config.get("score_controls")
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("dataset") != "total_text" or config.get("split") != "TEST"
        or config.get("table_5_claim") is not False or config.get("padding_pixels") != 0
        or set(execution) != {"precision", "batch_size", "cpu_threads"}
        or execution["precision"] != "float32"
        or any(type(execution[k]) is not int or execution[k] < 1
               for k in ("batch_size", "cpu_threads"))
        or type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32
        or not isinstance(scores, list) or not scores or len(scores) != len(set(scores))
        or any(type(x) not in (int, float) or not 0 < x <= 1 for x in scores)
        or type(config.get("evaluation_matching_iou")) not in (int, float)
        or not 0 < config["evaluation_matching_iou"] <= 1):
        raise ValueError("invalid isolated approximate YOLO direct PARSeq configuration")


def direct_crop(image: np.ndarray, slot: dict) -> np.ndarray:
    """Exactly the stored exclusive xyxy rectangle, without extra padding."""
    box = slot["bbox"]
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
        or len(box) != 4 or any(type(v) is not int for v in box)):
        raise ValueError("invalid direct YOLO crop input")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError("YOLO crop is empty or outside original image")
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def metrics(correct: int, outputs: int, gt_words: int) -> dict:
    return {"correct": correct, "outputs": outputs, "gt_words": gt_words,
            "precision": correct / outputs if outputs else 0.0,
            "recall": correct / gt_words if gt_words else 0.0,
            "hmean": 2 * correct / (outputs + gt_words) if outputs + gt_words else 0.0}


def run(args: argparse.Namespace) -> Path:
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("scientific config must be in configs/approximate")
    config = read_json(args.config)
    validate_config(config)
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("select cpu or cuda:0 explicitly")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly use --device cpu for a CPU experiment")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("generated results must stay outside the repository")

    detector = args.detector_run.resolve()
    detector_plan = read_json(detector / "resolved_configuration.json")
    detector_report = read_json(detector / "results.json")
    detector_hash = sha(detector / "resolved_configuration.json")
    paths = read_json(args.paths)
    if (detector_report["plan_sha256"] != detector_hash
        or detector_plan["reproduction_mode"] != "approximate"
        or detector_report["images"] != 300 or len(detector_plan["sample_ids"]) != 300
        or len(set(detector_plan["sample_ids"])) != 300
        or paths["reproduction_mode"] != "approximate"
        or paths["frozen_refinement_run"] != detector_plan["frozen_run"]
        or sha(Path(paths["test_manifest"])) != detector_plan["manifest_sha256"]
        or min(config["score_controls"]) < detector_plan["config"]["inference"]["confidence_floor"]):
        raise RuntimeError("frozen YOLO / Total-Text source contract differs")
    frozen = Path(detector_plan["frozen_run"])
    if sha(frozen / "resolved_configuration.json") != detector_plan["frozen_plan_sha256"]:
        raise RuntimeError("frozen source plan changed")
    source = Path(read_json(frozen / "resolved_configuration.json")["source"])
    if sha(source / "resolved_configuration.json") != detector_plan["source_plan_sha256"]:
        raise RuntimeError("source plan changed")
    source_plan = read_json(source / "resolved_configuration.json")
    source_paths = source_plan["paths"]
    checkpoint, archive = Path(source_paths["parseq_checkpoint"]), Path(source_paths["parseq_archive"])
    runtime = Path(source_paths["parseq_source"])
    if sha(checkpoint) != PARSEQ_CHECKPOINT_SHA256 or sha(archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("qualified PARSeq artifact differs")
    parseq_identity = verify_parseq_runtime_source(runtime, source_type=PARSEQ_ARCHIVE_SOURCE,
                                                   source_archive=archive)
    manifest = load_segmentation_manifest(Path(paths["test_manifest"]), require_checksums=True)
    samples = {sample.sample_id: sample for sample in manifest.samples}
    ids = detector_plan["sample_ids"][:args.limit] if args.limit else detector_plan["sample_ids"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["execution"]["cpu_threads"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats(args.device)

    # Validate and identify every detector shard before any new inference.
    shard_hashes = {}
    for sid in ids:
        path = detector / "detector" / f"{sid.replace(':', '__')}.json"
        shard = read_json(path)
        if (shard["plan_sha256"] != detector_hash or shard["sample_id"] != sid
            or shard["image_sha256"] != samples[sid].image_sha256
            or len({s["source_index"] for s in shard["slots"]}) != len(shard["slots"])):
            raise RuntimeError(f"detector shard contract differs: {sid}")
        shard_hashes[sid] = sha(path)

    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "result_class": "APPROXIMATE_TEST_DIAGNOSTIC" if len(ids) == 300 else
                            "SMOKE_TEST" if len(ids) == 1 else "PARTIAL_DATASET",
            "table_5_claim": False, "config": config, "config_sha256": sha(args.config),
            "paths_profile_sha256": sha(args.paths),
            "device": args.device, "precision": "float32", "batch_size": config["execution"]["batch_size"],
            "hardware_deviation": "CPU FP32 recognition" if args.device == "cpu" else
                                  "GPU recognition; hardware and batch differ from paper",
            "sample_ids": ids, "detector_run": str(detector), "detector_plan_sha256": detector_hash,
            "detector_shards_sha256": shard_hashes, "manifest_sha256": detector_plan["manifest_sha256"],
            "gt_source": str(source), "source_plan_sha256": sha(source / "resolved_configuration.json"),
            "parseq_checkpoint_sha256": sha(checkpoint), "parseq_runtime": parseq_identity,
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "timm": importlib.metadata.version("timm"),
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "torch_threads": torch.get_num_threads(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "coordinate_frame": "stored PIL RGB raster; stored clipped exclusive integer YOLO xyxy",
            "ground_truth_use": "opened only after all direct PARSeq OCR shards are frozen",
            "code_sha256": {name: sha(REPO / name) for name in CODE}}
    freeze(output / "resolved_configuration.json", plan)
    plan_hash = sha(output / "resolved_configuration.json")
    recognizer = None
    started = time.perf_counter()
    try:
        for position, sid in enumerate(ids, 1):
            key = sid.replace(":", "__")
            shard_path = output / "ocr" / f"{key}.json"
            contract = {"plan_sha256": plan_hash, "detector_sha256": shard_hashes[sid]}
            previous = cached(shard_path, contract)
            if previous is None:
                slots = read_json(detector / "detector" / f"{key}.json")["slots"]
                image, image_sha = read_image(Path(source_paths["data_root"]), samples[sid],
                                             samples[sid].image_sha256)
                rows = []
                infer_start = time.perf_counter()
                for offset in range(0, len(slots), config["execution"]["batch_size"]):
                    group = slots[offset:offset + config["execution"]["batch_size"]]
                    crops = [direct_crop(image, slot) for slot in group]
                    if recognizer is None:
                        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                            runtime_source=runtime, runtime_source_archive=archive,
                            runtime_source_type=PARSEQ_ARCHIVE_SOURCE, checkpoint=checkpoint,
                            device=args.device))
                    values = recognizer.recognize_candidate_batch(crops)
                    if len(values) != len(group):
                        raise RuntimeError("PARSeq returned incomplete direct crop batch")
                    for slot, crop, value in zip(group, crops, values):
                        rows.append({"candidate_id": f"{sid}:yolo:{slot['source_index']}",
                                     "bbox": slot["bbox"], "detector_score": slot["score"],
                                     "crop_rgb_sha256": array_sha(crop), "crop_shape": list(crop.shape),
                                     **value.to_dict()})
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                previous = {"contract": contract, "sample_id": sid, "image_sha256": image_sha,
                            "rows": rows, "seconds": time.perf_counter() - infer_start,
                            "model_peak": _cuda_peak(args.device), "ground_truth_use": "none"}
                write_json(shard_path, previous)
            print(f"[YOLO direct PARSeq] {position}/{len(ids)} {sid} crops={len(previous['rows'])}", flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED", "reproduction_mode": "approximate",
            "stage": "direct_parseq", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "out_of_memory": isinstance(exc, torch.cuda.OutOfMemoryError),
            "device": args.device, "precision": "float32", "batch_size": config["execution"]["batch_size"],
            "model_input_shape": [config["execution"]["batch_size"], 3, 32, 128],
            "model_peak": _cuda_peak(args.device)})
        raise
    finally:
        if recognizer is not None:
            recognizer.release()
    write_json(output / "last_execution.json", {"device": args.device,
        "current_ocr_pass_seconds": time.perf_counter() - started,
        "model_peak": _cuda_peak(args.device)})

    # New readings are now frozen. No annotations influenced crops or recognition.
    totals = {str(score): Counter() for score in config["score_controls"]}
    gt_words, gt_nonempty, recorded_seconds = 0, 0, 0.0
    for sid in ids:
        key = sid.replace(":", "__")
        ocr = read_json(output / "ocr" / f"{key}.json")
        rows = ocr["rows"]
        recorded_seconds += ocr["seconds"]
        gt = read_json(source / "evaluation" / f"{key}.json")["ground_truth"]
        gt_words += len(gt)
        gt_nonempty += sum(bool(normalize(word["text"])) for word in gt)
        arms = {}
        for score in config["score_controls"]:
            chosen = [row for row in rows if row["detector_score"] >= score]
            outcome = evaluate_predictions(chosen, gt, config["evaluation_matching_iou"])
            edges = [(pi, gi) for pi, row in enumerate(chosen) for gi, word in enumerate(gt)
                     if overlap_iou(row["bbox"], word["bbox"]) >= config["evaluation_matching_iou"]]
            box_matches = len(maximum_matching(len(chosen), len(gt), edges))
            empty = sum(not normalize(row["raw_text"]) for row in chosen)
            totals[str(score)].update(outputs=len(chosen), correct=outcome["correct"],
                                     normalized_empty_outputs=empty, detector_gt_box_matches=box_matches)
            arms[str(score)] = {**outcome, "detector_gt_box_matches": box_matches,
                               "normalized_empty_outputs": empty}
        freeze(output / "evaluation" / f"{key}.json", {"sample_id": sid, "gt_words": len(gt),
            "ocr_sha256": sha(output / "ocr" / f"{key}.json"),
            "gt_sha256": sha(source / "evaluation" / f"{key}.json"), "arms": arms})
    summary = {score: {**metrics(row["correct"], row["outputs"], gt_words),
                      "normalized_empty_outputs": row["normalized_empty_outputs"],
                      "detector_gt_box_matches": row["detector_gt_box_matches"]}
               for score, row in totals.items()}
    result = {"schema_version": 1, "reproduction_mode": "approximate",
              "result_class": plan["result_class"], "table_5_claim": False,
              "plan_sha256": plan_hash, "images": len(ids), "gt_words": gt_words,
              "gt_nonempty": gt_nonempty, "summary": summary,
              "device": args.device, "model_peak": _cuda_peak(args.device),
              "recorded_ocr_seconds": recorded_seconds,
              "metric": {"normalization": "Unicode casefold, strip punctuation/symbols; whitespace preserved",
                         "matching": "one-to-one exact normalized text and original bbox IoU",
                         "iou_threshold": config["evaluation_matching_iou"],
                         "aggregation": "micro counts over all selected images",
                         "empty_predictions": "retained as false positives",
                         "ground_truth": "frozen care GT; ignored words already excluded by source protocol"},
              "limitation": "local word/bbox Hmean proxy; not official Total-Text polygon Hmean"}
    freeze(output / "results.json", result)
    print(json.dumps({"report": str(output / "results.json"), "summary": summary}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--detector-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
