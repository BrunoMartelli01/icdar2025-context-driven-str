"""Transfer frozen YOLO -> direct PARSeq -> fixed confidence filtering to TEST splits."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from localize.independent_word_slots import overlap_iou
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
    verify_parseq_runtime_source)
from recognition.scene_text_manifest import build_scene_text_manifest, load_scene_text_manifest
from recognition.scene_text_v1_evaluation import maximum_matching
from recognition.yolo_word_refinement import arm_predictions
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import _cuda_peak, array_sha, cached, contained, sha, write_json
from scripts.evaluate_totaltext_text_detectors import installed_package_inventory
from scripts.evaluate_totaltext_yolo_direct_parseq import direct_crop
from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots, read_json, validate_checkpoint
from scripts.run_crop_group_word_selection import evaluate_predictions, normalize
from segmentation.data import load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/scene_text_yolo_parseq_transfer_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/scene_text_yolo_parseq_transfer_local_paths_v1.json"
CODE = ("scripts/evaluate_scene_text_yolo_parseq.py", "scripts/test_scene_text_yolo.sh",
    "recognition/icdar13_annotations.py", "recognition/scene_text_manifest.py",
    "recognition/scene_text_datasets.py", "recognition/parseq.py",
    "recognition/yolo_word_refinement.py", "recognition/group_word_selection.py",
    "scripts/evaluate_totaltext_yolo_direct_parseq.py", "scripts/evaluate_totaltext_yolo_word_slots.py",
    "scripts/evaluate_totaltext_text_detectors.py", "scripts/run_crop_group_word_selection.py",
    "scripts/analyze_totaltext_crop_word_groups.py", "scripts/crop_clip_runtime.py",
    "recognition/scene_text_v1_evaluation.py", "localize/independent_word_slots.py",
    "segmentation/data.py", "segmentation/scene_text_preparation.py")


def validate_config(cfg: dict) -> None:
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("table_5_claim") is not False or cfg.get("split") != "TEST"
        or cfg.get("datasets") != {"icdar13": {"images": 233, "granularity": "word"},
            "icdar2015": {"images": 500, "granularity": "word"},
            "ctw1500": {"images": 500, "granularity": "text_line"}}
        or cfg.get("ultralytics_version") != "8.1.0" or cfg.get("padding_pixels") != 0
        or cfg.get("inference") != {"image_size": 640, "confidence_floor": 0.05,
                                    "nms_iou": 0.7, "max_detections": 300}
        or cfg.get("score_controls") != [0.05, 0.25, 0.4] or cfg.get("reference_score") != 0.4
        or cfg.get("joint_controls") != [0.6] or cfg.get("merge_iou") != 0.5
        or cfg.get("matching_iou") != 0.5 or cfg.get("primary_method") != "joint_text_nms_0.6"
        or type(cfg.get("seed")) is not int or not 0 <= cfg["seed"] < 2**32):
        raise ValueError("invalid frozen approximate transfer configuration; no TEST-specific settings")
    execution = cfg.get("execution", {})
    if (set(execution) != {"precision", "detector_batch_size", "parseq_batch_size", "cpu_threads"}
        or execution["precision"] != "float32" or execution["detector_batch_size"] != 1
        or any(type(execution[k]) is not int or execution[k] < 1
               for k in ("parseq_batch_size", "cpu_threads"))):
        raise ValueError("invalid transfer execution configuration")


def absolute(path: str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else REPO / value).resolve()


def exclusive_bbox(bbox, image_wh) -> list[int]:
    """Source maxima are inclusive pixel indices; evaluation uses exclusive xyxy."""
    x1, y1, x2, y2 = bbox
    width, height = image_wh
    result = [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("care annotation has no positive in-image bounding box")
    return result


def qualify_dataset(name: str, profile: dict, spec: dict, qualification_root: Path):
    """GT is read for identity/evaluation; inference receives image records only."""
    root = absolute(profile["data_root"])
    truth, records, clipped_total = {}, [], 0
    if name == "icdar13":
        samples, qualification = load_icdar13_recognition_samples(absolute(profile["manifest"]), root)
        for sample in samples:
            sid = f"{name}:{sample.sample_id}"
            records.append({"sample_id": sid, "image_path": str(sample.image_path),
                "image_sha256": sample.image_sha256, "annotation_path": str(sample.annotation_path),
                "annotation_sha256": sample.annotation_sha256, "image_size_wh": list(sample.image_size_wh)})
            truth[sid] = {"care": [{"text": word.transcription,
                "bbox": exclusive_bbox(word.bbox_xyxy, sample.image_size_wh)} for word in sample.words],
                "ignored": [], "clipped_polygons": 0}
    else:
        if name == "icdar2015":
            manifest_path = qualification_root / "icdar2015_recognition_test.json"
            if not manifest_path.exists():
                build_scene_text_manifest(dataset=name, split="test", data_root=root,
                    image_directory=contained(root, profile["image_directory"]),
                    annotation_directory=contained(root, profile["annotation_directory"]),
                    output=manifest_path, enforce_official_count=True)
        else:
            manifest_path = absolute(profile["manifest"])
        raw_manifest = read_json(manifest_path)
        if (raw_manifest["dataset"] != name or raw_manifest["split"] != "test"
            or raw_manifest["annotation_granularity"] != spec["granularity"]
            or raw_manifest.get("reproduction_mode") != "approximate"):
            raise ValueError("recognition manifest dataset/split/granularity mismatch")
        samples, qualification = load_scene_text_manifest(manifest_path, root)
        if name == "icdar2015":
            prepared_path = absolute(profile["prepared_manifest"])
            prepared = load_segmentation_manifest(prepared_path, require_checksums=True)
            if prepared.dataset != name or prepared.split_role != "test":
                raise ValueError("prepared ICDAR2015 manifest is not the TEST split")
            expected = {s.sample_id: (contained(root, s.image_path), s.image_sha256) for s in prepared.samples}
            actual = {f"{name}:{s.sample_id}": (s.image_path, sha(s.image_path)) for s in samples}
            if expected != actual:
                raise ValueError("ICDAR2015 recognition/prepared raster identities differ")
            qualification["prepared_manifest_sha256"] = sha(prepared_path)
        for sample in samples:
            sid = f"{name}:{sample.sample_id}"
            records.append({"sample_id": sid, "image_path": str(sample.image_path),
                "image_sha256": sha(sample.image_path), "annotation_path": str(sample.annotation_path),
                "annotation_sha256": sha(sample.annotation_path), "image_size_wh": list(sample.image_size_wh)})
            instances, clipped = clip_instances_to_pixel_extent(sample.instances, sample.image_size_wh)
            clipped_total += clipped
            truth[sid] = {"care": [], "ignored": [], "clipped_polygons": clipped}
            for instance in instances:
                if instance.granularity != spec["granularity"]:
                    raise ValueError("annotation granularity changed within dataset")
                if instance.ignore:
                    truth[sid]["ignored"].append({"text": instance.transcription,
                        "polygon_xy": [list(point) for point in instance.polygon_xy]})
                else:
                    truth[sid]["care"].append({"text": instance.transcription,
                        "bbox": exclusive_bbox(instance.bbox_xyxy, sample.image_size_wh),
                        "polygon_xy": [list(point) for point in instance.polygon_xy]})
        if sum(len(t["care"]) for t in truth.values()) != raw_manifest["care_instance_count"]:
            raise ValueError("care-instance count differs from qualified manifest")
    if len(records) != spec["images"] or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError(f"incomplete or duplicate {name} TEST samples")
    qualification.update(care_gt_units=sum(len(t["care"]) for t in truth.values()),
        ignored_gt_units=sum(len(t["ignored"]) for t in truth.values()),
        clipped_polygons=clipped_total, granularity=spec["granularity"])
    return records, truth, qualification


def read_pixels(record: dict) -> np.ndarray:
    if sha(Path(record["image_path"])) != record["image_sha256"]:
        raise RuntimeError(f"image bytes changed: {record['sample_id']}")
    with Image.open(record["image_path"]) as opened:
        pixels = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
    if [pixels.shape[1], pixels.shape[0]] != record["image_size_wh"]:
        raise RuntimeError("decoded image dimensions changed")
    return pixels


def peak_from_shards(shards: list[dict]) -> dict:
    return {key: max((s["model_peak"][key] or 0 for s in shards), default=0) or None
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")}


def unit_metrics(correct: int, outputs: int, gt: int, granularity: str) -> dict:
    return {"correct": correct, "outputs": outputs, "gt_units": gt, "gt_granularity": granularity,
        "precision": correct / outputs if outputs else 0.0,
        "recall": correct / gt if gt else 0.0,
        "hmean": 2 * correct / (outputs + gt) if outputs + gt else 0.0}


def run_dataset(name, records, truth, qualification, cfg, paths, args, common, output):
    if args.limit:
        records = records[:args.limit]
    full = len(records) == cfg["datasets"][name]["images"]
    granularity = cfg["datasets"][name]["granularity"]
    plan = json.loads(json.dumps({**common, "config": cfg, "dataset": name, "split": "TEST",
        "qualification": qualification, "samples": records, "annotation_granularity": granularity,
        "result_class": "APPROXIMATE_REPRODUCTION" if full else "SMOKE_TEST" if len(records) == 1 else "PARTIAL_DATASET",
        "ground_truth_use": "qualification and evaluation only; none in detection/crops/OCR/filtering"}))
    freeze(output / "resolved_configuration.json", plan)
    ph = sha(output / "resolved_configuration.json")
    yolo = recognizer = None
    stage, sid = "detector", None
    current_detector = current_ocr = 0
    started = time.perf_counter()
    try:
        for position, record in enumerate(records, 1):
            sid = record["sample_id"]
            key = sid.replace(":", "__")
            contract = {"plan_sha256": ph, "sample_id": sid, "image_sha256": record["image_sha256"]}
            previous = cached(output / "detector" / f"{key}.json", contract)
            if previous is None:
                if yolo is None:
                    from ultralytics import YOLO
                    if args.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(args.device)
                    yolo = YOLO(paths["yolo_checkpoint"])
                    if yolo.model.names != {0: "text"}:
                        raise RuntimeError("checkpoint is not the qualified one-class text detector")
                    input_shape = []
                    yolo.model.register_forward_pre_hook(lambda module, inputs:
                        input_shape.__setitem__(slice(None), list(inputs[0].shape)))
                pixels = read_pixels(record)
                infer_start = time.perf_counter()
                result = yolo.predict(np.ascontiguousarray(pixels[:, :, ::-1]),
                    imgsz=cfg["inference"]["image_size"], conf=cfg["inference"]["confidence_floor"],
                    iou=cfg["inference"]["nms_iou"], max_det=cfg["inference"]["max_detections"],
                    device=args.device, half=False, verbose=False)[0]
                slots, rejected = boxes_to_slots(result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                    tuple(record["image_size_wh"]), cfg["inference"]["confidence_floor"])
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                previous = {"contract": contract, "sample_id": sid, "slots": slots, "rejected": rejected,
                    "seconds": time.perf_counter() - infer_start, "model_input_shape": list(input_shape),
                    "model_peak": _cuda_peak(args.device), "model_parameters": sum(p.numel() for p in yolo.model.parameters())}
                write_json(output / "detector" / f"{key}.json", previous)
                current_detector += 1
            print(f"[YOLO] {name} {position}/{len(records)} boxes={len(previous['slots'])}", flush=True)
        del yolo
        yolo = None
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        stage = "parseq"
        for position, record in enumerate(records, 1):
            sid = record["sample_id"]
            key = sid.replace(":", "__")
            detector_path = output / "detector" / f"{key}.json"
            contract = {"plan_sha256": ph, "sample_id": sid, "detector_sha256": sha(detector_path)}
            previous = cached(output / "ocr" / f"{key}.json", contract)
            if previous is None:
                pixels, rows = read_pixels(record), []
                slots = read_json(detector_path)["slots"]
                infer_start = time.perf_counter()
                batch_size = cfg["execution"]["parseq_batch_size"]
                for offset in range(0, len(slots), batch_size):
                    group = slots[offset:offset + batch_size]
                    crops = [direct_crop(pixels, slot) for slot in group]
                    if recognizer is None:
                        if args.device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats(args.device)
                        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                            runtime_source=Path(paths["parseq_source"]),
                            runtime_source_archive=Path(paths["parseq_archive"]),
                            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                            checkpoint=Path(paths["parseq_checkpoint"]), device=args.device))
                    readings = recognizer.recognize_candidate_batch(crops)
                    if len(readings) != len(group):
                        raise RuntimeError("PARSeq returned incomplete crop batch")
                    for slot, crop, reading in zip(group, crops, readings):
                        rows.append({"candidate_id": f"{sid}:yolo:{slot['source_index']}",
                            "bbox": slot["bbox"], "detector_score": slot["score"],
                            "crop_rgb_sha256": array_sha(crop), "crop_shape": list(crop.shape), **reading.to_dict()})
                    current_ocr += len(group)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                previous = {"contract": contract, "sample_id": sid, "rows": rows,
                    "seconds": time.perf_counter() - infer_start, "model_peak": _cuda_peak(args.device),
                    "model_input_shape": [min(batch_size, len(slots)), 3, 32, 128] if slots else None}
                write_json(output / "ocr" / f"{key}.json", previous)
            print(f"[PARSeq] {name} {position}/{len(records)} crops={len(previous['rows'])}", flush=True)
        stage = "evaluation"
        totals, count, nonempty, ignored, clipped = {}, 0, 0, 0, 0
        detector_shards, ocr_shards = [], []
        for record in records:
            sid, key = record["sample_id"], record["sample_id"].replace(":", "__")
            if sha(Path(record["annotation_path"])) != record["annotation_sha256"]:
                raise RuntimeError("annotation bytes changed after qualification")
            gt = truth[sid]["care"]
            count += len(gt)
            nonempty += sum(bool(normalize(w["text"])) for w in gt)
            ignored += len(truth[sid]["ignored"])
            clipped += truth[sid]["clipped_polygons"]
            detector_shards.append(read_json(output / "detector" / f"{key}.json"))
            ocr_shards.append(read_json(output / "ocr" / f"{key}.json"))
            scored = {}
            for method, rows in arm_predictions(ocr_shards[-1]["rows"], cfg).items():
                outcome = evaluate_predictions(rows, gt, cfg["matching_iou"])
                edges = [(pi, gi) for pi, row in enumerate(rows) for gi, word in enumerate(gt)
                         if overlap_iou(row["bbox"], word["bbox"]) >= cfg["matching_iou"]]
                spatial = len(maximum_matching(len(rows), len(gt), edges))
                empty = sum(not normalize(row["raw_text"]) for row in rows)
                totals.setdefault(method, Counter()).update(outputs=len(rows), correct=outcome["correct"],
                    detector_gt_box_matches=spatial, normalized_empty_outputs=empty)
                scored[method] = {**outcome, "selected_candidate_ids": [r["candidate_id"] for r in rows],
                    "detector_gt_box_matches": spatial, "normalized_empty_outputs": empty}
            freeze(output / "evaluation" / f"{key}.json", {"sample_id": sid, "ground_truth": truth[sid],
                "ocr_sha256": sha(output / "ocr" / f"{key}.json"), "arms": scored})
        report = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "table_5_claim": False, "result_class": plan["result_class"], "dataset": name,
            "plan_sha256": ph, "images": len(records), "gt_units": count,
            "gt_nonempty": nonempty, "ignored_gt_units": ignored, "clipped_polygons": clipped,
            "annotation_granularity": granularity, "primary_method": cfg["primary_method"],
            "summary": {method: {**unit_metrics(t["correct"], t["outputs"], count, granularity),
                "detector_gt_box_matches": t["detector_gt_box_matches"],
                "normalized_empty_outputs": t["normalized_empty_outputs"]} for method, t in totals.items()},
            "generated_crops": sum(len(s["rows"]) for s in ocr_shards),
            "rejected_detector_boxes": sum(len(s["rejected"]) for s in detector_shards),
            "device": args.device, "precision": "float32",
            "detector_peak": peak_from_shards(detector_shards), "parseq_peak": peak_from_shards(ocr_shards),
            "recorded_detector_seconds": sum(s["seconds"] for s in detector_shards),
            "recorded_parseq_seconds": sum(s["seconds"] for s in ocr_shards),
            "metric": {"aggregation": "micro", "normalization": "Unicode casefold; remove punctuation/symbols; preserve whitespace",
                "matching": "maximum one-to-one exact normalized text plus original bbox IoU >=0.5",
                "ignore": "exclude ignored GT; retain predictions on ignored regions as FP",
                "empty": "normalized-empty GT retained in denominator; empty predictions are FP",
                "limitation": "local bbox proxy, not official polygon evaluation",
                "ctw1500": "exact full-line diagnostic, not word accuracy; no GT-derived word boxes/grouping"}}
        freeze(output / "results.json", report)
        write_json(output / "last_execution.json", {"status": "COMPLETE", "device": args.device,
            "seconds": time.perf_counter() - started, "new_detector_images": current_detector,
            "new_ocr_crops": current_ocr})
        print(json.dumps({"report": str(output / "results.json"),
            "primary": report["summary"][cfg["primary_method"]]}), flush=True)
        return report
    except Exception as exc:
        write_json(output / "failures" / f"{time.time_ns()}.json", {"status": "FAILED", "stage": stage,
            "sample_id": sid, "device": args.device, "reproduction_mode": "approximate",
            "precision": "float32", "execution": cfg["execution"], "inference": cfg["inference"],
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "out_of_memory": isinstance(exc, torch.cuda.OutOfMemoryError), "model_peak": _cuda_peak(args.device)})
        raise
    finally:
        if recognizer is not None:
            recognizer.release()
        del yolo
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=("icdar13", "icdar2015", "ctw1500"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("scientific configuration must be isolated in configs/approximate")
    cfg, paths = read_json(args.config), read_json(args.paths)
    validate_config(cfg)
    if paths.get("reproduction_mode") != "approximate" or set(paths["datasets"]) != set(cfg["datasets"]):
        raise ValueError("invalid approximate local path profile")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run this command on the GPU host, or explicitly select --device cpu")
    if importlib.metadata.version("ultralytics") != cfg["ultralytics_version"]:
        raise RuntimeError("qualified Ultralytics version differs; use the isolated YOLO environment")
    root = (args.output_root or absolute(paths["output_root"])).resolve()
    if root == REPO or root.is_relative_to(REPO):
        raise ValueError("generated outputs must stay outside the repository")
    names = args.datasets or list(cfg["datasets"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate requested datasets")
    yolo_identity = validate_checkpoint(absolute(paths["yolo_checkpoint"]), cfg["checkpoint_source"])
    if yolo_identity["sha256"] != cfg["checkpoint_sha256"]:
        raise RuntimeError("YOLO checkpoint SHA256 differs")
    if (sha(absolute(paths["parseq_checkpoint"])) != PARSEQ_CHECKPOINT_SHA256
        or sha(absolute(paths["parseq_archive"])) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise RuntimeError("qualified PARSeq artifacts differ")
    parseq_identity = verify_parseq_runtime_source(absolute(paths["parseq_source"]),
        source_type=PARSEQ_ARCHIVE_SOURCE, source_archive=absolute(paths["parseq_archive"]))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    common = {"reproduction_mode": "approximate", "device": args.device,
        "paths": paths, "config_sha256": sha(args.config), "paths_sha256": sha(args.paths),
        "yolo_identity": yolo_identity, "parseq_identity": parseq_identity,
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "code_sha256": {name: sha(REPO / name) for name in CODE},
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "packages": installed_package_inventory(), "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "determinism": {"seed": cfg["seed"], "cudnn_benchmark": False, "cudnn_deterministic": True,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}
    # Qualify every requested split before starting any detector inference.
    qualified = {name: qualify_dataset(name, paths["datasets"][name], cfg["datasets"][name],
                                      root / "qualified-datasets") for name in names}
    suite = root / args.device.replace(":", "-") / (f"smoke-{args.limit}" if args.limit else "full")
    reports = {}
    for name, (records, truth, qualification) in qualified.items():
        reports[name] = run_dataset(name, records, truth, qualification, cfg, paths, args, common, suite / name)
    aggregate = {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "table_5_claim": False, "primary_method": cfg["primary_method"], "datasets": reports}
    freeze(suite / "results.json", aggregate)
    with (suite / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "images", "granularity", "method", "gt_units", "correct", "outputs", "precision", "recall", "hmean"])
        for name, report in reports.items():
            for method, result in report["summary"].items():
                writer.writerow([name, report["images"], report["annotation_granularity"], method,
                    result["gt_units"], result["correct"], result["outputs"],
                    result["precision"], result["recall"], result["hmean"]])
    print(f"Suite report: {suite / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
