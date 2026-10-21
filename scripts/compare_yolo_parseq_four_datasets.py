"""Add frozen Total-Text controls and profile YOLO/PARSeq dominant arithmetic."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from localize.independent_word_slots import overlap_iou
from recognition.group_word_selection import normalize
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
    verify_parseq_runtime_source)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.yolo_word_refinement import joint_score, nms
from scripts.analyze_scene_text_yolo_thresholds import (
    best_control, score_subset, shared_control, threshold_grid, threshold_key,
)
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import array_sha, contained, sha
from scripts.evaluate_scene_text_yolo_parseq import exclusive_bbox, unit_metrics
from scripts.evaluate_totaltext_yolo_direct_parseq import direct_crop
from scripts.evaluate_totaltext_yolo_word_slots import read_json
from scripts.neural_macs import DenseMacCounter, letterbox_shape
from segmentation.data import load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/yolo_parseq_four_dataset_comparison_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/yolo_parseq_four_dataset_comparison_local_paths_v1.json"
CODE = ("scripts/compare_yolo_parseq_four_datasets.py", "scripts/neural_macs.py",
    "scripts/test_yolo_four_datasets.sh", "scripts/analyze_scene_text_yolo_thresholds.py",
    "recognition/yolo_word_refinement.py", "recognition/group_word_selection.py",
    "recognition/scene_text_v1_evaluation.py", "recognition/parseq.py",
    "scripts/evaluate_scene_text_yolo_parseq.py", "scripts/evaluate_totaltext_yolo_direct_parseq.py",
    "scripts/crop_clip_runtime.py", "recognition/scene_text_datasets.py", "segmentation/data.py")


def validate_config(cfg):
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("table_5_claim") is not False
        or cfg.get("datasets") != {"icdar13": 233, "icdar2015": 500, "ctw1500": 500, "total_text": 300}
        or cfg.get("joint_threshold") != 0.6 or cfg.get("nms_iou") != 0.5
        or cfg.get("matching_iou") != 0.5 or 0.6 not in threshold_grid(cfg)):
        raise ValueError("invalid isolated four-dataset approximate comparison")
    p = cfg.get("profiling", {})
    if (set(p) != {"precision", "device", "sample_images_per_dataset", "parseq_batch_size", "cpu_threads", "flops_per_mac"}
        or p["precision"] != "float32" or p["device"] != "cpu" or p["flops_per_mac"] != 2
        or p["parseq_batch_size"] != 32
        or any(type(p[key]) is not int or p[key] < 1 for key in ("sample_images_per_dataset", "cpu_threads"))):
        raise ValueError("invalid explicit CPU FP32 compute profiling configuration")


def verify_code(plan):
    for filename, expected in plan["code_sha256"].items():
        if sha(REPO / filename) != expected:
            raise RuntimeError(f"frozen source implementation differs: {filename}")


def load_sources(paths, cfg):
    transfer, thresholds = Path(paths["transfer_source"]), Path(paths["threshold_source"])
    aggregate = read_json(transfer / "results.json")
    previous = read_json(thresholds / "results.json")
    threshold_plan = read_json(thresholds / "resolved_configuration.json")
    if (previous["plan_sha256"] != sha(thresholds / "resolved_configuration.json")
        or threshold_plan["source_results_sha256"] != sha(transfer / "results.json")
        or threshold_plan["config"]["threshold_grid"] != cfg["threshold_grid"]
        or read_json(thresholds / "independent_audit.json")["status"] != "PASS"):
        raise RuntimeError("three-dataset cached threshold source differs")
    verify_code(threshold_plan)
    sources = {}
    for name, report in aggregate["datasets"].items():
        root = transfer / name
        plan = read_json(root / "resolved_configuration.json")
        if (report != read_json(root / "results.json") or report["plan_sha256"] != sha(root / "resolved_configuration.json")
            or report["images"] != cfg["datasets"][name] or len(plan["samples"]) != report["images"]):
            raise RuntimeError("incomplete or changed three-dataset source")
        verify_code(plan)
        sources[name] = {"root": str(root), "plan_sha256": report["plan_sha256"],
            "records": plan["samples"], "source_device": report["device"],
            "gpu": plan["gpu"], "accuracy": previous["datasets"][name]}
    reference = read_json(transfer / "icdar13/resolved_configuration.json")
    direct, selection = Path(paths["total_text_direct"]), Path(paths["total_text_selection"])
    dp, sp = read_json(direct / "resolved_configuration.json"), read_json(selection / "resolved_configuration.json")
    dr, sr = read_json(direct / "results.json"), read_json(selection / "results.json")
    detector = Path(dp["detector_run"])
    detector_plan = read_json(detector / "resolved_configuration.json")
    gt_root = Path(dp["gt_source"])
    gt_plan = read_json(gt_root / "resolved_configuration.json")
    if (dr["plan_sha256"] != sha(direct / "resolved_configuration.json")
        or sr["plan_sha256"] != sha(selection / "resolved_configuration.json")
        or sp["baseline_run"] != str(direct)
        or sp["baseline_plan_sha256"] != dr["plan_sha256"]
        or dp["detector_plan_sha256"] != sha(detector / "resolved_configuration.json")
        or dp["source_plan_sha256"] != sha(gt_root / "resolved_configuration.json")
        or dr["images"] != cfg["datasets"]["total_text"] or len(dp["sample_ids"]) != dr["images"]
        or dp["sample_ids"] != sp["sample_ids"] or dp["sample_ids"] != detector_plan["sample_ids"]
        or dp["parseq_checkpoint_sha256"] != reference["parseq_checkpoint_sha256"]
        or dp["parseq_runtime"] != reference["parseq_identity"]
        or detector_plan["checkpoint"]["sha256"] != reference["yolo_identity"]["sha256"]
        or detector_plan["config"]["inference"]["image_size"] != 640
        or dp["config"]["padding_pixels"] != 0):
        raise RuntimeError("Total-Text direct640 artifact/model contract differs")
    for plan in (dp, sp, detector_plan):
        verify_code(plan)
    manifest_path = Path(paths["total_text_manifest"])
    if sha(manifest_path) != dp["manifest_sha256"]:
        raise RuntimeError("Total-Text TEST manifest differs")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    by_id = {s.sample_id: s for s in manifest.samples}
    data_root = Path(gt_plan["paths"]["data_root"])
    records = []
    for sid in dp["sample_ids"]:
        sample, key = by_id[sid], sid.replace(":", "__")
        detector_path, ocr_path = detector / "detector" / f"{key}.json", direct / "ocr" / f"{key}.json"
        anno_path = contained(data_root, Path(paths["total_text_annotation_directory"]) / f"{sid.split(':')[1]}.txt")
        gt = read_json(gt_root / "evaluation" / f"{key}.json")
        if (sha(detector_path) != dp["detector_shards_sha256"][sid]
            or sha(ocr_path) != sp["baseline_ocr_sha256"][sid]
            or sha(detector_path) != sp["baseline_detector_sha256"][sid]
            or gt["annotation_sha256"] != sha(anno_path)):
            raise RuntimeError("Total-Text frozen shard or annotation differs")
        image_path = contained(data_root, sample.image_path)
        if sha(image_path) != sample.image_sha256:
            raise RuntimeError("Total-Text source image differs")
        shard = read_json(detector_path)
        records.append({"sample_id": sid, "image_path": str(image_path),
            "image_sha256": sample.image_sha256, "image_size_wh": shard["image_wh"],
            "annotation_path": str(anno_path), "annotation_sha256": sha(anno_path)})
    if len(set(r["sample_id"] for r in records)) != cfg["datasets"]["total_text"]:
        raise RuntimeError("Total-Text sample IDs are incomplete or duplicated")
    sources["total_text"] = {"root": str(direct), "plan_sha256": dr["plan_sha256"],
        "records": records, "source_device": dp["device"], "gpu": dp["gpu"],
        "detector_root": str(detector), "gt_root": str(gt_root), "selection_root": str(selection),
        "accuracy": None, "baseline": sr["summary"]["base640/joint_text_nms_0.6"]}
    return sources, reference


def frozen_rows(source, record):
    key = record["sample_id"].replace(":", "__")
    op = Path(source["root"]) / "ocr" / f"{key}.json"
    rows = read_json(op)
    detector_path = Path(source.get("detector_root", source["root"])) / "detector" / f"{key}.json"
    expected = {"plan_sha256": source["plan_sha256"], "detector_sha256": sha(detector_path)}
    if "detector_root" not in source:
        expected["sample_id"] = record["sample_id"]
    if rows["contract"] != expected:
        raise RuntimeError("frozen crop contract differs")
    return rows["rows"]


def total_text_controls(source, cfg, output, ph):
    grid = threshold_grid(cfg)
    totals = {family: {t: Counter() for t in grid} for family in ("detector_score", "joint_text_nms")}
    gt_count = candidates = ignored = clipped_total = 0
    for index, record in enumerate(source["records"], 1):
        key = record["sample_id"].replace(":", "__")
        rows = frozen_rows(source, record)
        gtpath = Path(source["gt_root"]) / "evaluation" / f"{key}.json"
        gt = read_json(gtpath)["ground_truth"]
        legacy = read_json(Path(source["selection_root"]) / "evaluation" / f"{key}.json")["arms"]["base640/joint_text_nms_0.6"]
        ids = {r["candidate_id"]: i for i, r in enumerate(rows)}
        normalized = [normalize(r["raw_text"]) for r in rows]
        scores = [joint_score(r) for r in rows]
        survivors = nms([r for r, t in zip(rows, normalized) if t], cfg["nms_iou"], same_text=True, joint=True)
        survivor_ids = [ids[r["candidate_id"]] for r in survivors]
        baseline_ids = [r["candidate_id"] for r in survivors if scores[ids[r["candidate_id"]]] >= cfg["joint_threshold"]]
        if baseline_ids != legacy["selected_candidate_ids"]:
            raise RuntimeError("Total-Text baseline selection differs")
        spatial = [(pi, gi) for pi, r in enumerate(rows) for gi, g in enumerate(gt)
                   if overlap_iou(r["bbox"], g["bbox"]) >= cfg["matching_iou"]]
        text = [(pi, gi) for pi, gi in spatial if normalize(gt[gi]["text"]) and normalized[pi] == normalize(gt[gi]["text"])]
        cells = {}
        for family, controls in totals.items():
            cells[family] = {}
            for threshold, counter in controls.items():
                chosen = ([pi for pi, r in enumerate(rows) if r["detector_score"] >= threshold]
                          if family == "detector_score" else [pi for pi in survivor_ids if scores[pi] >= threshold])
                cell = score_subset(chosen, len(gt), text, spatial, normalized)
                cells[family][threshold_key(threshold)] = cell
                counter.update({k: cell[k] for k in ("correct", "outputs", "normalized_empty_outputs", "detector_gt_box_matches")})
                if family == "joint_text_nms":
                    counter.update(crops_necessary_before_parseq=sum(math.sqrt(r["detector_score"]) >= threshold for r in rows))
        baseline = cells["joint_text_nms"][threshold_key(cfg["joint_threshold"])]
        if baseline["correct"] != legacy["correct"] or baseline["correct_gt_indices"] != legacy["correct_gt_indices"]:
            raise RuntimeError("Total-Text baseline normalized matching differs")
        annotation, clipped = clip_instances_to_pixel_extent(
            parse_totaltext_annotation(Path(record["annotation_path"])), tuple(record["image_size_wh"]))
        clipped_total += clipped
        physical_gt = [{"text": instance.transcription,
            "bbox": exclusive_bbox(instance.bbox_xyxy, record["image_size_wh"])}
            for instance in annotation if not instance.ignore]
        if physical_gt != gt:
            raise RuntimeError("Total-Text care labels/bboxes differ from frozen GT after canonical clipping")
        if sum(not instance.ignore for instance in annotation) != len(gt):
            raise RuntimeError("Total-Text care count differs from physical annotations")
        ignored += sum(instance.ignore for instance in annotation)
        candidates += len(rows)
        gt_count += len(gt)
        freeze(output / "total_text_controls" / f"{key}.json", {"plan_sha256": ph,
            "sample_id": record["sample_id"], "source_ocr_sha256": sha(Path(source["root"]) / "ocr" / f"{key}.json"),
            "source_gt_sha256": sha(gtpath), "controls": cells})
        if index == 1 or index % 100 == 0:
            print(f"[Total-Text cached controls] {index}/{len(source['records'])}", flush=True)
    controls = {family: [{"threshold": t, **unit_metrics(c["correct"], c["outputs"], gt_count, "word"),
                         **{k: c[k] for k in c if k not in ("correct", "outputs")}} for t, c in rows.items()]
                for family, rows in totals.items()}
    baseline = next(r for r in controls["joint_text_nms"] if r["threshold"] == cfg["joint_threshold"])
    if any(baseline[k] != source["baseline"][k] for k in ("correct", "outputs", "precision", "recall", "hmean")):
        raise RuntimeError("Total-Text aggregate baseline differs")
    return {"images": len(source["records"]), "gt_units": gt_count, "annotation_granularity": "word",
        "generated_crops": candidates, "ignored_gt_units": ignored, "clipped_polygons": clipped_total, "controls": controls,
        "best_exploratory": {family: best_control(rows) for family, rows in controls.items()}, "baseline": baseline}


def sample_indices(count, requested):
    if count < 1 or requested < 1:
        raise ValueError("sample count must be positive")
    return np.linspace(0, count - 1, min(count, requested), dtype=int).tolist()


def model_forward_profile(model, tensor, **kwargs):
    with DenseMacCounter(model) as counter, torch.inference_mode():
        result = model(tensor, **kwargs)
    return result, counter.snapshot()


def compute_profile(sources, reference, cfg, output):
    p = cfg["profiling"]
    torch.set_num_threads(p["cpu_threads"])
    torch.manual_seed(0)
    checkpoint = Path(reference["paths"]["yolo_checkpoint"])
    if sha(checkpoint) != reference["yolo_identity"]["sha256"]:
        raise RuntimeError("YOLO profiling weights differ")
    if importlib.metadata.version("ultralytics") != reference["config"]["ultralytics_version"]:
        raise RuntimeError("qualified YOLO runtime differs")
    shapes, per_image = {}, {}
    for name, source in sources.items():
        per_image[name] = []
        for record in source["records"]:
            key = record["sample_id"].replace(":", "__")
            dp = Path(source.get("detector_root", source["root"])) / "detector" / f"{key}.json"
            detector = read_json(dp)
            shape = letterbox_shape(record["image_size_wh"], reference["config"]["inference"]["image_size"], 32)
            if "model_input_shape" in detector and shape != detector["model_input_shape"]:
                raise RuntimeError("derived LetterBox shape differs from recorded genuine inference")
            rows = frozen_rows(source, record)
            per_image[name].append({"sample_id": record["sample_id"], "detector_input_shape": shape,
                "shape_source": "recorded" if "model_input_shape" in detector else "derived_qualified_letterbox",
                "original_crops": len(rows),
                "eligible_crops_at_joint_threshold": sum(math.sqrt(r["detector_score"]) >= cfg["joint_threshold"] for r in rows)})
            shapes[tuple(shape)] = None
    from ultralytics import YOLO
    yolo = YOLO(str(checkpoint)).model.float().eval().fuse(verbose=False)
    try:
        for shape in shapes:
            _, measured = model_forward_profile(yolo, torch.zeros(shape, dtype=torch.float32))
            shapes[shape] = measured
        _, nominal = model_forward_profile(yolo, torch.zeros((1, 3, 640, 640), dtype=torch.float32))
    finally:
        del yolo
        gc.collect()
    paths = reference["paths"]
    if sha(Path(paths["parseq_checkpoint"])) != PARSEQ_CHECKPOINT_SHA256 or sha(Path(paths["parseq_archive"])) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("qualified PARSeq profiling artifacts differ")
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths["parseq_source"]),
        runtime_source_archive=Path(paths["parseq_archive"]), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        checkpoint=Path(paths["parseq_checkpoint"]), device=p["device"]))
    sampled = {}
    try:
        for name, source in sources.items():
            image_profiles = []
            for index in sample_indices(len(source["records"]), p["sample_images_per_dataset"]):
                record = source["records"][index]
                if sha(Path(record["image_path"])) != record["image_sha256"]:
                    raise RuntimeError("profile image checksum differs")
                with Image.open(record["image_path"]) as opened:
                    pixels = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
                rows = frozen_rows(source, record)
                batches = []
                for offset in range(0, len(rows), p["parseq_batch_size"]):
                    group = rows[offset:offset + p["parseq_batch_size"]]
                    crops = [direct_crop(pixels, r) for r in group]
                    for crop, row in zip(crops, group):
                        if array_sha(crop) != row["crop_rgb_sha256"]:
                            raise RuntimeError("profile crop pixels differ from recognized frozen crop")
                    tensor = torch.stack([recognizer.transform(Image.fromarray(crop)) for crop in crops])
                    logits, measured = model_forward_profile(recognizer.model, tensor)
                    readings = recognizer.model.tokenizer.decode(logits.softmax(-1))[0]
                    head = [e for e in measured["events"] if e["module"] == "model.head"]
                    if not head or measured["macs_by_kind"].get("multihead_attention", 0) == 0:
                        raise RuntimeError("PARSeq decoder/projection compute was not counted")
                    measured.update(input_shape=list(tensor.shape), crops=len(group),
                        autoregressive_steps=len(head) - recognizer.model.model.refine_iters,
                        cpu_reading_differences=sum(a != b["raw_text"] for a, b in zip(readings, group)),
                        candidate_ids=[r["candidate_id"] for r in group])
                    batches.append(measured)
                image_profiles.append({"sample_id": record["sample_id"], "crops": len(rows),
                    "macs": sum(b["macs"] for b in batches), "batches": batches})
                print(f"[PARSeq MAC profile] {name} {record['sample_id']} crops={len(rows)}", flush=True)
            sampled[name] = image_profiles
        # Fixed maximum length disables early-EOS for this separate compute budget only.
        _, full_length = model_forward_profile(recognizer.model, torch.zeros((1, 3, 32, 128)),
                                               max_length=int(recognizer.model.hparams.max_label_length))
    finally:
        recognizer.release()
    summary = {}
    for name, image_profiles in sampled.items():
        crop_count = sum(s["crops"] for s in image_profiles)
        if not crop_count:
            raise RuntimeError("all chosen profiling images have no recognized crops")
        recognition_per_crop = 2 * sum(s["macs"] for s in image_profiles) / crop_count / 1e9
        rows = per_image[name]
        for row in rows:
            yolo_flops = shapes[tuple(row["detector_input_shape"])]["flops"] / 1e9
            row.update(yolo_gflops=yolo_flops,
                pipeline_gflops_estimated=yolo_flops + row["original_crops"] * recognition_per_crop,
                pipeline_gflops_safe_prefilter_estimated=yolo_flops + row["eligible_crops_at_joint_threshold"] * recognition_per_crop,
                pipeline_gflops_full_length_budget=yolo_flops + row["original_crops"] * full_length["flops"] / 1e9)
        mean = lambda key: sum(r[key] for r in rows) / len(rows)
        summary[name] = {"images": len(rows), "source_device": sources[name]["source_device"],
            "mean_crops_before_filter": mean("original_crops"),
            "mean_crops_eligible_before_parseq": mean("eligible_crops_at_joint_threshold"),
            "mean_yolo_gflops": mean("yolo_gflops"), "parseq_gflops_per_crop_sampled": recognition_per_crop,
            "mean_pipeline_gflops_estimated": mean("pipeline_gflops_estimated"),
            "mean_pipeline_gflops_safe_prefilter_estimated": mean("pipeline_gflops_safe_prefilter_estimated"),
            "mean_pipeline_gflops_full_length_budget": mean("pipeline_gflops_full_length_budget"),
            "sample_images": len(image_profiles), "sample_crops": crop_count,
            "sample_cpu_reading_differences": sum(b["cpu_reading_differences"] for s in image_profiles for b in s["batches"]),
            "sample_crop_gflops_range": [min(2*b["macs"]/b["crops"]/1e9 for s in image_profiles for b in s["batches"]),
                                        max(2*b["macs"]/b["crops"]/1e9 for s in image_profiles for b in s["batches"])]}
    result = {"reproduction_mode": "approximate", "result_class": "COMPUTE_ESTIMATE",
        "convention": "2 FLOPs/MAC; dense conv/linear, MHA projections and QK/AV products",
        "excluded": ["bias additions", "activation/normalization/softmax", "embedding gathers",
                     "preprocessing", "NMS and confidence filtering", "tokenizer operations"],
        "profile_device": "cpu", "precision": "float32", "detector_batch": 1,
        "parseq_batch": p["parseq_batch_size"], "new_accuracy_claim": False,
        "yolo_nominal640_gflops": nominal["flops"] / 1e9,
        "parseq_full_length_gflops_per_crop": full_length["flops"] / 1e9,
        "summary": summary, "per_image_estimates": per_image,
        "detector_shape_profiles": [{"shape": list(shape), **profile} for shape, profile in shapes.items()],
        "nominal_yolo_profile": nominal, "sampled_parseq_profiles": sampled,
        "full_length_parseq_profile": full_length,
        "limitation": "Sampled CPU conditional decoding scaled by all-image crop density; not exact GPU FLOPs for every input, not paper Table4 reproduction. Prefilter cost is hypothetical."}
    freeze(output / "compute_profile.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("thresholds", "all"), default="all")
    args = parser.parse_args()
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("scientific configuration must stay isolated in configs/approximate")
    cfg, paths = read_json(args.config), read_json(args.paths)
    validate_config(cfg)
    if paths.get("reproduction_mode") != "approximate":
        raise ValueError("approximate path profile required")
    output = (args.output or Path(paths["output"])).resolve()
    if output.is_relative_to(REPO) or any(output == Path(paths[k]).resolve() or output.is_relative_to(Path(paths[k]).resolve())
        for k in ("transfer_source", "threshold_source", "total_text_direct", "total_text_selection")):
        raise ValueError("comparison outputs must be external and separate from frozen source outputs")
    sources, reference = load_sources(paths, cfg)
    identity = verify_parseq_runtime_source(Path(reference["paths"]["parseq_source"]),
        source_type=PARSEQ_ARCHIVE_SOURCE, source_archive=Path(reference["paths"]["parseq_archive"]))
    if json.loads(json.dumps(identity)) != reference["parseq_identity"]:
        raise RuntimeError("qualified PARSeq runtime differs")
    plan = {"reproduction_mode": "approximate", "config": cfg, "paths": paths, "output": str(output),
        "source_plans_sha256": {name: source["plan_sha256"] for name, source in sources.items()},
        "source_devices": {name: source["source_device"] for name, source in sources.items()},
        "code_sha256": {name: sha(REPO / name) for name in CODE}, "python": sys.version,
        "config_sha256": sha(args.config), "paths_sha256": sha(args.paths),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "timm", "ultralytics")},
        "parseq_identity": identity, "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "yolo_identity": reference["yolo_identity"]}
    freeze(output / "resolved_configuration.json", plan)
    ph = sha(output / "resolved_configuration.json")
    accuracy = {name: source["accuracy"] for name, source in sources.items() if source["accuracy"] is not None}
    accuracy["total_text"] = total_text_controls(sources["total_text"], cfg, output, ph)
    sources["total_text"]["accuracy"] = accuracy["total_text"]
    for name, data in accuracy.items():
        data["source_device"] = sources[name]["source_device"]
    shared = shared_control(accuracy, {name: data["baseline"] for name, data in accuracy.items()})
    report = {"reproduction_mode": "approximate", "claim_label": cfg["claim_label"], "table_5_claim": False,
        "result_class": "EXPLORATORY_TEST_ANALYSIS", "plan_sha256": ph, "datasets": accuracy,
        "shared_joint_threshold_improving_all": shared, "test_selection_bias": True,
        "metric": "local normalized exact text and original-axis bbox IoU0.5 one-to-one micro; CTW is full-line GT"}
    freeze(output / "results.json", report)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "source_device", "granularity", "family", "threshold", "correct", "outputs", "precision", "recall", "hmean"])
        for name, data in accuracy.items():
            for family, controls in data["controls"].items():
                for row in controls:
                    writer.writerow([name, data["source_device"], data["annotation_granularity"], family,
                        row["threshold"], row["correct"], row["outputs"], row["precision"], row["recall"], row["hmean"]])
    compute = compute_profile(sources, reference, cfg, output) if args.stage == "all" else None
    print(json.dumps({"report": str(output / "results.json"), "shared_threshold": shared,
        "total_text_best": accuracy["total_text"]["best_exploratory"],
        "compute_summary": compute["summary"] if compute else None}), flush=True)


if __name__ == "__main__":
    main()
