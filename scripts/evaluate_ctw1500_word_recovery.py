"""Rescore frozen CTW TEST outputs as line-supported word occurrences on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
import sys
import traceback
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from recognition.group_word_selection import normalize, reading_confidence
from recognition.line_token_evaluation import evaluate_line_tokens, word_metrics
from recognition.scene_text_datasets import parse_ctw1500_test_annotation
from recognition.yolo_word_refinement import joint_score, nms
from scripts.analyze_scene_text_yolo_thresholds import threshold_grid
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import sha, write_json
from scripts.evaluate_scene_text_yolo_parseq import exclusive_bbox
from scripts.evaluate_totaltext_text_detectors import installed_package_inventory
from scripts.evaluate_totaltext_yolo_word_slots import read_json
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/ctw1500_word_recovery_audit_v2.json"
DEFAULT_PATHS = REPO / "configs/approximate/ctw1500_word_recovery_audit_local_paths_v2.json"
CODE = ("scripts/evaluate_ctw1500_word_recovery.py", "recognition/line_token_evaluation.py",
    "scripts/test_ctw_words.sh", "recognition/yolo_word_refinement.py",
    "recognition/group_word_selection.py", "recognition/scene_text_v1_evaluation.py",
    "scripts/crop_clip_runtime.py", "scripts/analyze_totaltext_crop_word_groups.py",
    "scripts/analyze_scene_text_yolo_thresholds.py", "scripts/evaluate_scene_text_yolo_parseq.py",
    "recognition/scene_text_datasets.py", "segmentation/scene_text_preparation.py",
    "scripts/evaluate_totaltext_text_detectors.py", "scripts/evaluate_totaltext_yolo_word_slots.py")


def validate_config(cfg):
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("claim_label") != "approximate reproduction"
        or cfg.get("dataset") != "ctw1500" or cfg.get("split") != "TEST"
        or cfg.get("table_5_claim") is not False):
        raise ValueError("requires an explicit approximate CTW TEST configuration")
    evaluation, selection = cfg.get("evaluation", {}), cfg.get("selection", {})
    if (evaluation.get("metric_id") != "line_supported_word_occurrences_bbox_v1"
        or evaluation.get("normalization") != "Unicode casefold; remove punctuation/symbols; split on whitespace"):
        raise ValueError("unsupported word recovery protocol or normalization")
    semantics = {
        "matching": "maximum one-to-one exact token and line bbox support; repeated tokens retain separate occurrences",
        "ignore": "exclude ignored GT; predictions on ignored regions remain FP",
        "empty": "normalized-empty GT lines contribute zero words and are reported; empty predictions count as one FP",
        "limitation": "line-supported recovery proxy; no GT word boxes and no word-position validation; not official polygon Hmean",
    }
    if any(evaluation.get(key) != value for key, value in semantics.items()):
        raise ValueError("declared evaluation semantics differ from implemented protocol")
    for value in (evaluation.get("minimum_prediction_coverage"), selection.get("joint_threshold"),
                  selection.get("same_text_nms_iou")):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("metric and selection thresholds must be finite in (0,1]")
    if selection["same_text_nms_iou"] == 1:
        raise ValueError("NMS IoU must be less than one")
    floors = selection.get("prefilter_thresholds")
    if (not isinstance(floors, list) or not floors
        or any(type(value) not in (int, float) or not math.isfinite(value)
               or not 0 < value <= 1 for value in floors) or len(set(floors)) != len(floors)
        or selection.get("reference_prefilter") not in floors):
        raise ValueError("invalid prefilter controls or reference")
    if "prefilter_grid" in selection:
        threshold_grid({"threshold_grid": selection["prefilter_grid"]})
    if selection.get("primary_prefilter_policy", "analytic_joint_upper_bound") != "analytic_joint_upper_bound":
        raise ValueError("primary policy must be the analytic joint-score bound")


def prefilter_controls(selection):
    values = set(selection["prefilter_thresholds"])
    if "prefilter_grid" in selection:
        values.update(threshold_grid({"threshold_grid": selection["prefilter_grid"]}))
    return {**{str(value): value for value in sorted(values)}, "safe_upper_bound": None}


def safe_prefilter_eligible(detector_score, joint_threshold):
    """Use the upper bound directly; squaring can reject a floating-point tie."""
    if (not math.isfinite(detector_score) or not 0 <= detector_score <= 1
        or not math.isfinite(joint_threshold) or not 0 < joint_threshold <= 1):
        raise ValueError("invalid detector confidence or joint threshold")
    return math.sqrt(detector_score) >= joint_threshold


def select_predictions(rows, selection, detector_threshold):
    if detector_threshold is None:
        eligible = [row for row in rows if safe_prefilter_eligible(
            row["detector_score"], selection["joint_threshold"])]
    else:
        eligible = [row for row in rows if row["detector_score"] >= detector_threshold]
    admitted = [row for row in eligible if normalize(row["raw_text"])
                and joint_score(row) >= selection["joint_threshold"]]
    return eligible, nms(admitted, selection["same_text_nms_iou"], same_text=True, joint=True)


def verify_physical_gt(record, cached_gt):
    size = tuple(record["image_size_wh"])
    annotation = Path(record["annotation_path"])
    # Match the source loader's parser-time clipping before its post-load count.
    instances, clipped = clip_instances_to_pixel_extent(
        parse_ctw1500_test_annotation(annotation, image_size_wh=size), size)
    _, raw_clipped = clip_instances_to_pixel_extent(parse_ctw1500_test_annotation(annotation), size)
    physical = {"care": [], "ignored": [], "clipped_polygons": clipped}
    for instance in instances:
        item = {"text": instance.transcription,
                "polygon_xy": [list(point) for point in instance.polygon_xy]}
        if instance.ignore:
            physical["ignored"].append(item)
        else:
            item["bbox"] = exclusive_bbox(instance.bbox_xyxy, record["image_size_wh"])
            physical["care"].append(item)
    if physical != cached_gt:
        raise RuntimeError(f"cached GT differs from physical annotation: {record['sample_id']}")
    return raw_clipped


def verify_detector_rows(rows, detector, record):
    expected = {f"{record['sample_id']}:yolo:{slot['source_index']}": slot
                for slot in detector["slots"]}
    if len({row["candidate_id"] for row in rows}) != len(rows) or len(rows) != len(expected):
        raise RuntimeError("incomplete or duplicated frozen OCR proposals")
    width, height = record["image_size_wh"]
    for row in rows:
        slot = expected.get(row["candidate_id"])
        if slot is None or row["bbox"] != slot["bbox"] or row["detector_score"] != slot["score"]:
            raise RuntimeError("OCR proposal differs from frozen detector slot")
        x1, y1, x2, y2 = row["bbox"]
        if (any(type(value) is not int for value in row["bbox"])
            or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height
            or row["crop_shape"] != [y2 - y1, x2 - x1, 3]
            or not math.isfinite(row["detector_score"]) or not 0 <= row["detector_score"] <= 1):
            raise ValueError("invalid frozen proposal geometry or confidence")
        reading_confidence(row)


def independent_word_counts(predictions, ground_truth, coverage_threshold):
    """Independent normalization/rectangle support and SciPy maximum matching."""
    def split_text(text):
        return "".join(c for c in text.casefold()
                       if unicodedata.category(c)[0] not in ("P", "S")).split()
    truth = [(token, line["bbox"]) for line in ground_truth for token in split_text(line["text"])]
    output = [(token, row["bbox"]) for row in predictions
              for token in split_text(row["raw_text"]) or [""]]
    left, right = [], []
    for pi, (text, a) in enumerate(output):
        for gi, (expected, b) in enumerate(truth):
            if not text or text != expected:
                continue
            width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if width * height >= coverage_threshold * (a[2] - a[0]) * (a[3] - a[1]):
                left.append(pi)
                right.append(gi)
    graph = csr_matrix((np.ones(len(left), dtype=np.int8), (left, right)),
                       shape=(len(output), len(truth)))
    correct = int(np.sum(maximum_bipartite_matching(graph, perm_type="column") >= 0))
    return correct, len(output), len(truth)


def run(args):
    if (args.config.resolve().parent != REPO / "configs/approximate"
        or args.paths.resolve().parent != REPO / "configs/approximate"):
        raise ValueError("configuration namespaces must be configs/approximate")
    cfg, paths = read_json(args.config), read_json(args.paths)
    validate_config(cfg)
    if paths.get("reproduction_mode") != "approximate":
        raise ValueError("local path profile must explicitly select approximate mode")
    source = Path(paths["source"]).resolve()
    output = (args.output or Path(paths["output"])).resolve()
    if output == REPO or REPO in output.parents or output == source or source in output.parents:
        raise ValueError("new evaluation output must stay outside repository and frozen source")
    plan_path, report_path = source / "resolved_configuration.json", source / "results.json"
    source_plan, source_report = read_json(plan_path), read_json(report_path)
    expected_images = source_plan["config"]["datasets"]["ctw1500"]["images"]
    if (source_report["plan_sha256"] != sha(plan_path) or source_plan["dataset"] != "ctw1500"
        or source_plan["split"] != "TEST" or source_plan["reproduction_mode"] != "approximate"
        or source_report["annotation_granularity"] != "text_line"
        or source_report["images"] != expected_images or len(source_plan["samples"]) != expected_images):
        raise RuntimeError("source is not the complete qualified CTW TEST line experiment")
    for filename, expected in source_plan["code_sha256"].items():
        if sha(REPO / filename) != expected:
            raise RuntimeError(f"frozen source implementation changed: {filename}")
    controls = prefilter_controls(cfg["selection"])
    if min(value for value in controls.values() if value is not None) < source_plan["config"]["inference"]["confidence_floor"]:
        raise ValueError("prefilter below the frozen detector floor needs new inference")
    if args.limit is not None and (args.limit < 1 or args.limit > expected_images):
        raise ValueError("limit must be within the complete source population")
    records = source_plan["samples"][:args.limit]
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate source sample IDs")
    result_class = ("SMOKE_TEST" if len(records) == 1 else "PARTIAL_DATASET"
                    if len(records) < expected_images else "APPROXIMATE_REPRODUCTION")
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": cfg["claim_label"], "result_class": result_class,
        "config": cfg, "paths": paths, "config_sha256": sha(args.config),
        "paths_sha256": sha(args.paths), "source_plan_sha256": sha(plan_path),
        "source_report_sha256": sha(report_path), "source_samples": records,
        "source_software_hardware": {key: source_plan[key] for key in
            ("python", "packages", "torch", "cuda_build", "platform", "device", "gpu",
             "determinism", "yolo_identity", "parseq_identity", "parseq_checkpoint_sha256")},
        "execution": {"device": "cpu", "models_loaded": False, "new_crops": 0,
            "new_inference": False, "prediction_selection_uses_gt": False},
        "code_sha256": {filename: sha(REPO / filename) for filename in CODE},
        "python": sys.version, "platform": platform.platform(),
        "packages": installed_package_inventory(), "output": str(output)}
    freeze(output / "resolved_configuration.json", plan)
    ph = sha(output / "resolved_configuration.json")
    totals = {name: Counter() for name in controls}
    preserved = {name: True for name in controls}
    counts, source_base_correct, source_base_outputs, checked = Counter(), 0, 0, 0
    selection = cfg["selection"]
    stage, sid = "source_validation", None
    try:
        for position, record in enumerate(records, 1):
            sid, key = record["sample_id"], record["sample_id"].replace(":", "__")
            op, ep, dp = (source / directory / f"{key}.json"
                          for directory in ("ocr", "evaluation", "detector"))
            ocr, evaluation, detector = read_json(op), read_json(ep), read_json(dp)
            if (ocr["contract"] != {"plan_sha256": sha(plan_path), "sample_id": sid,
                                    "detector_sha256": sha(dp)}
                or evaluation["sample_id"] != sid or evaluation["ocr_sha256"] != sha(op)
                or sha(Path(record["annotation_path"])) != record["annotation_sha256"]
                or sha(Path(record["image_path"])) != record["image_sha256"]):
                raise RuntimeError(f"frozen CTW source identity changed: {sid}")
            if detector["contract"] != {"plan_sha256": sha(plan_path), "sample_id": sid,
                                        "image_sha256": record["image_sha256"]}:
                raise RuntimeError("detector image identity changed")
            raw_clipped = verify_physical_gt(record, evaluation["ground_truth"])
            rows, gt = ocr["rows"], evaluation["ground_truth"]["care"]
            verify_detector_rows(rows, detector, record)
            counts.update(candidates=len(rows), gt_lines=len(gt),
                raw_annotation_polygons_clipped=raw_clipped,
                ignored_gt_lines=len(evaluation["ground_truth"]["ignored"]),
                normalized_empty_gt_lines=sum(not normalize(g["text"]).split() for g in gt))
            primary = evaluation["arms"][source_report["primary_method"]]
            source_base_correct += primary["correct"]
            source_base_outputs += primary["predictions"]
            cells = {}
            stage = "word_evaluation"
            _, reference = select_predictions(rows, selection, selection["reference_prefilter"])
            reference_ids = [row["candidate_id"] for row in reference]
            for name, threshold in controls.items():
                eligible, chosen = select_predictions(rows, selection, threshold)
                ids = [row["candidate_id"] for row in chosen]
                preserved[name] &= ids == reference_ids
                if threshold is None:
                    unfiltered = nms([r for r in rows if normalize(r["raw_text"])
                        and joint_score(r) >= selection["joint_threshold"]],
                        selection["same_text_nms_iou"], same_text=True, joint=True)
                    if ids != [r["candidate_id"] for r in unfiltered]:
                        raise RuntimeError("analytic upper-bound prefilter changed unfiltered final outputs")
                if (selection["joint_threshold"] in source_plan["config"]["joint_controls"]
                    and selection["same_text_nms_iou"] == source_plan["config"]["merge_iou"]
                    and threshold is None):
                    legacy = evaluation["arms"][f"joint_text_nms_{selection['joint_threshold']}"]
                    if [r["candidate_id"] for r in chosen] != legacy["selected_candidate_ids"]:
                        raise RuntimeError("safe prefilter changed frozen baseline selected IDs")
                result = evaluate_line_tokens(chosen, gt, cfg["evaluation"]["minimum_prediction_coverage"])
                independent = independent_word_counts(chosen, gt, cfg["evaluation"]["minimum_prediction_coverage"])
                if independent != (result["correct_words"], result["predicted_words"], result["gt_words"]):
                    raise RuntimeError(f"independent word matching differs: {sid}, {threshold}")
                totals[name].update(correct_words=result["correct_words"],
                    predicted_words=result["predicted_words"], gt_words=result["gt_words"],
                    crops_eligible_before_parseq=len(eligible),
                    normalized_empty_predictions=result["normalized_empty_predictions"],
                    image_lexical_multiset_correct_without_geometry=result["image_lexical_multiset_correct_without_geometry"])
                cells[name] = {**{k: v for k, v in result.items() if k != "gt_word_tokens"},
                    "crops_eligible_before_parseq": len(eligible), "selected_candidate_ids": ids}
                checked += 1
            freeze(output / "evaluation" / f"{key}.json", {"plan_sha256": ph, "sample_id": sid,
                "source_ocr_sha256": sha(op), "source_evaluation_sha256": sha(ep),
                "ground_truth": evaluation["ground_truth"], "gt_word_tokens": result["gt_word_tokens"],
                "controls": cells})
            if position == 1 or position % 100 == 0 or position == len(records):
                print(f"[CTW words] {position}/{len(records)}", flush=True)
        stage = "aggregate"
        summaries = {threshold: {**word_metrics(c["correct_words"], c["predicted_words"], c["gt_words"]),
            **{key: c[key] for key in ("crops_eligible_before_parseq", "normalized_empty_predictions",
                "image_lexical_multiset_correct_without_geometry")},
            "same_final_outputs_as_reference": preserved[threshold]} for threshold, c in totals.items()}
        if len(records) == expected_images and (counts["gt_lines"] != source_report["gt_units"]
            or counts["candidates"] != source_report["generated_crops"]
            or counts["ignored_gt_lines"] != source_report["ignored_gt_units"]):
            raise RuntimeError("complete source denominator differs")
        best_key = max((name for name, value in controls.items() if value is not None),
            key=lambda name: (summaries[name]["hmean"], summaries[name]["correct_words"], -controls[name]))
        preserving_key = max((name for name, value in controls.items()
            if value is not None and preserved[name]), key=lambda name: controls[name])
        report = {"schema_version": 2, "reproduction_mode": "approximate",
            "claim_label": cfg["claim_label"], "result_class": result_class,
            "dataset": cfg["dataset"], "split": cfg["split"], "table_5_claim": False,
            "plan_sha256": ph, "images": len(records), **dict(counts),
            "source_line_metric_reference": {"correct_lines": source_base_correct,
                "outputs": source_base_outputs, "gt_lines": counts["gt_lines"],
                "hmean": 2 * source_base_correct / (source_base_outputs + counts["gt_lines"])},
            "metric": cfg["evaluation"], "selection": selection, "summary": summaries,
            "primary_method": "safe_upper_bound",
            "safe_prefilter": {"predicate": "sqrt(detector_score) >= joint_threshold",
                "nominal_detector_threshold": selection["joint_threshold"] ** 2,
                "reason": "OCR quality <=1; therefore joint score <=sqrt(detector_score)",
                "scope": "same frozen OCR and same final filter; new batch execution remains unqualified"},
            "largest_tested_output_preserving_threshold": controls[preserving_key],
            "best_observed_test_f1": {"threshold": controls[best_key], **summaries[best_key],
                "exploratory_test_selection": True, "deployment_default_changed": False},
            "new_inference": False, "failures": 0}
        freeze(output / "results.json", report)
        freeze(output / "independent_audit.json", {"status": "PASS", "plan_sha256": ph,
            "images": len(records), "word_controls": checked,
            "normalization_and_geometry": "independent Unicode and rectangle coverage implementation",
            "matching": "SciPy maximum_bipartite_matching",
            "physical_ground_truth": "all care/ignore texts and canonically clipped polygons/bboxes verified",
            "frozen_detector_ocr": "every candidate ID, bbox, detector score, crop shape and native character scores verified",
            "safe_prefilter": "upper-bound predicate checked against unfiltered final outputs and original baseline IDs"})
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["prefilter_threshold", *next(iter(summaries.values()))])
            writer.writeheader()
            writer.writerows({"prefilter_threshold": t, **summary} for t, summary in summaries.items())
        print(json.dumps({"report": str(output / "results.json"),
            "primary": summaries["safe_upper_bound"],
            "largest_tested_output_preserving_threshold": controls[preserving_key],
            "best_observed_test_f1": report["best_observed_test_f1"]}), flush=True)
        return report
    except Exception as error:
        write_json(output / "failure.json", {"stage": stage, "sample_id": sid,
            "error": str(error), "error_type": type(error).__name__, "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
