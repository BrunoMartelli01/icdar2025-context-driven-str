"""Rescore frozen four-dataset YOLO/PARSeq outputs; ignore regions affect metrics only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from recognition.dont_care_evaluation import apply_neutralization
from recognition.icdar13_annotations import parse_task1_annotation
from recognition.line_token_evaluation import evaluate_line_tokens, word_metrics
from recognition.scene_text_datasets import (parse_icdar2015_annotation, parse_totaltext_annotation,
    parse_ctw1500_test_annotation)
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.compare_yolo_parseq_four_datasets import load_sources, frozen_rows
from scripts.crop_clip_runtime import sha, write_json
from scripts.evaluate_ctw1500_word_recovery import (prefilter_controls, select_predictions,
    verify_detector_rows, verify_physical_gt, independent_word_counts)
from scripts.evaluate_scene_text_yolo_parseq import exclusive_bbox
from scripts.evaluate_totaltext_text_detectors import installed_package_inventory
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/yolo_dont_care_word_evaluation_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/yolo_dont_care_word_evaluation_local_paths_v1.json"
CODE = ("scripts/evaluate_yolo_dont_care.py", "scripts/dont_care_polygon_worker.py",
    "recognition/dont_care_evaluation.py", "scripts/test_yolo_evaluation.sh",
    "scripts/compare_yolo_parseq_four_datasets.py", "scripts/evaluate_ctw1500_word_recovery.py",
    "recognition/line_token_evaluation.py", "recognition/group_word_selection.py",
    "recognition/yolo_word_refinement.py", "recognition/scene_text_v1_evaluation.py",
    "scripts/run_crop_group_word_selection.py", "recognition/icdar13_annotations.py",
    "recognition/scene_text_datasets.py", "segmentation/scene_text_preparation.py",
    "scripts/evaluate_scene_text_yolo_parseq.py", "scripts/crop_clip_runtime.py",
    "scripts/analyze_totaltext_crop_word_groups.py", "scripts/analyze_scene_text_yolo_thresholds.py",
    "scripts/evaluate_totaltext_text_detectors.py", "scripts/evaluate_totaltext_yolo_word_slots.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def validate_config(cfg):
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("claim_label") != "approximate reproduction" or cfg.get("table_5_claim") is not False
        or cfg.get("datasets") != {"icdar13": 233, "icdar2015": 500, "ctw1500": 500, "total_text": 300}
        or cfg.get("selection", {}).get("primary_method") != "safe_upper_bound"):
        raise ValueError("requires isolated approximate four-dataset evaluation")
    s, e, g = cfg["selection"], cfg["evaluation"], cfg["geometry_runtime"]
    for value in (s["joint_threshold"], s["same_text_nms_iou"],
                  e["ignored_prediction_coverage_threshold"], e["word_bbox_matching_iou"],
                  e["ctw_word_minimum_prediction_coverage"]):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("thresholds must be finite in (0,1)")
    if s["joint_threshold"] != .6 or s["same_text_nms_iou"] != .5:
        raise ValueError("this update must preserve qualified baseline confidence and NMS")
    if e["ignored_prediction_coverage_threshold"] != .5:
        raise ValueError("the benchmark-style ignore comparison requires coverage >0.5")
    if (e["word_bbox_matching_iou"] != .5 or e["ctw_word_minimum_prediction_coverage"] != .5):
        raise ValueError("care matching must preserve the prior word protocols")
    if (g["expected_versions"] != {"Polygon3": "3.0.9.1", "Shapely": "2.0.0"}
        or type(g["timeout_seconds"]) is not int or g["timeout_seconds"] < 1):
        raise ValueError("requires qualified isolated polygon runtime")
    controls = prefilter_controls(s)
    if any(value is not None and (type(value) not in (int, float)
           or not math.isfinite(value) or not .05 <= value <= 1)
           for value in controls.values()):
        raise ValueError("detector cutoff must lie within the frozen proposal population")


def qualify_truth(name, source, record):
    sid, key = record["sample_id"], record["sample_id"].replace(":", "__")
    annotation, size = Path(record["annotation_path"]), tuple(record["image_size_wh"])
    if (sha(annotation) != record["annotation_sha256"]
        or sha(Path(record["image_path"])) != record["image_sha256"]):
        raise RuntimeError(f"physical source bytes changed: {sid}")
    root = Path(source.get("gt_root", source["root"]))
    ep = root / "evaluation" / f"{key}.json"
    cached = read_json(ep)
    op = Path(source["root"]) / "ocr" / f"{key}.json"
    if cached["sample_id"] != sid:
        raise RuntimeError("GT sample identity differs")
    if name != "total_text" and cached["ocr_sha256"] != sha(op):
        raise RuntimeError("frozen evaluation/OCR identity differs")
    raw_clipped = 0
    if name == "icdar13":
        physical = {"care": [{"text": w.transcription, "bbox": exclusive_bbox(w.bbox_xyxy, size)}
            for w in parse_task1_annotation(annotation, image_size_wh=size)],
            "ignored": [], "clipped_polygons": 0}
    elif name == "ctw1500":
        raw_clipped = verify_physical_gt(record, cached["ground_truth"])
        physical = cached["ground_truth"]
    else:
        parser = parse_totaltext_annotation if name == "total_text" else parse_icdar2015_annotation
        instances, clipped = clip_instances_to_pixel_extent(parser(annotation), size)
        physical = {"care": [], "ignored": [], "clipped_polygons": clipped}
        for instance in instances:
            item = {"text": instance.transcription,
                "polygon_xy": [list(p) for p in instance.polygon_xy]}
            if instance.ignore:
                physical["ignored"].append(item)
            else:
                item["bbox"] = exclusive_bbox(instance.bbox_xyxy, size)
                physical["care"].append(item)
    if name == "total_text":
        if ([{"text": g["text"], "bbox": g["bbox"]} for g in physical["care"]]
            != cached["ground_truth"] or cached["annotation_sha256"] != sha(annotation)):
            raise RuntimeError("Total-Text physical care GT differs from frozen source")
        legacy = read_json(Path(source["selection_root"]) / "evaluation" / f"{key}.json")[
            "arms"]["base640/joint_text_nms_0.6"]
    else:
        if physical != cached["ground_truth"]:
            raise RuntimeError(f"physical GT differs from frozen source: {sid}")
        legacy = cached["arms"]["joint_text_nms_0.6"]
    # Benchmark ignore geometry uses original annotation vertices. Clamping each
    # vertex is not polygon intersection and could alter the in-image shape.
    parser = {"icdar2015": parse_icdar2015_annotation, "total_text": parse_totaltext_annotation,
        "ctw1500": parse_ctw1500_test_annotation}.get(name)
    ignored_raw = [] if parser is None else [instance for instance in parser(annotation) if instance.ignore]
    if len(ignored_raw) != len(physical["ignored"]):
        raise RuntimeError("original and cached ignore-region populations differ")
    physical = {**physical, "ignored_evaluation_polygons": [{"text": instance.transcription,
        "polygon_xy": [list(point) for point in instance.polygon_xy]} for instance in ignored_raw],
        "ignored_raw_polygons_outside_pixel_extent": sum(any(
            x < 0 or x > size[0] - 1 or y < 0 or y > size[1] - 1 for x, y in instance.polygon_xy)
            for instance in ignored_raw)}
    return physical, legacy, ep, raw_clipped


def independent_bbox_counts(predictions, ground_truth, minimum_iou):
    def text(value):
        return "".join(c for c in value.casefold() if unicodedata.category(c)[0] not in ("P", "S"))
    left, right = [], []
    for pi, pred in enumerate(predictions):
        a, reading = pred["bbox"], text(pred["raw_text"])
        for gi, gt in enumerate(ground_truth):
            if not reading or reading != text(gt["text"]):
                continue
            b = gt["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
            if intersection >= minimum_iou * union:
                left.append(pi)
                right.append(gi)
    graph = csr_matrix((np.ones(len(left), dtype=np.int8), (left, right)),
        shape=(len(predictions), len(ground_truth)))
    correct = int(np.sum(maximum_bipartite_matching(graph, perm_type="column") >= 0))
    return correct, len(predictions), len(ground_truth)


def score_words(name, predictions, truth, evaluation):
    if name == "ctw1500":
        threshold = evaluation["ctw_word_minimum_prediction_coverage"]
        result = evaluate_line_tokens(predictions, truth, threshold)
        independent = independent_word_counts(predictions, truth, threshold)
    else:
        threshold = evaluation["word_bbox_matching_iou"]
        prior = evaluate_predictions(predictions, truth, threshold)
        result = {**word_metrics(prior["correct"], prior["predictions"], len(truth)),
            "matched_gt_word_indices": prior["correct_gt_indices"]}
        independent = independent_bbox_counts(predictions, truth, threshold)
    if independent != (result["correct_words"], result["predicted_words"], result["gt_words"]):
        raise RuntimeError("independent normalized word matching differs")
    return result


def run_geometry(request, output, python, timeout):
    request_path, response_path = output / "polygon_request.json", output / "polygon_response.json"
    freeze(request_path, request)
    if response_path.exists():
        result = read_json(response_path)
    else:
        process = subprocess.run([str(python), str(REPO / "scripts/dont_care_polygon_worker.py"),
            "--request", str(request_path), "--response", str(response_path)],
            capture_output=True, text=True, timeout=timeout, check=False)
        if process.returncode:
            raise RuntimeError(f"polygon worker failed ({process.returncode}): {process.stderr}")
        result = read_json(response_path)
    if result.get("status") != "PASS" or result.get("request_sha256") != sha(request_path):
        raise RuntimeError("failed or stale isolated polygon response")
    if result["runtime"]["versions"] != request["expected_versions"]:
        raise RuntimeError("cached polygon runtime versions differ")
    expected = {im["sample_id"]: [r["candidate_id"] for r in im["rows"]] for im in request["images"]}
    actual = {im["sample_id"]: [r["candidate_id"] for r in im["rows"]] for im in result["images"]}
    if len(actual) != len(result["images"]) or actual != expected:
        raise RuntimeError("polygon response ordering or candidate population differs")
    return result


def run(args):
    for path in (args.config, args.paths):
        if path.resolve().parent != REPO / "configs/approximate":
            raise ValueError("configuration lookup is limited to configs/approximate")
    cfg, paths = read_json(args.config), read_json(args.paths)
    validate_config(cfg)
    if paths.get("reproduction_mode") != "approximate":
        raise ValueError("path profile must explicitly select approximate mode")
    source_cfg_path, source_paths_path = local_path(paths["source_config"]), local_path(paths["source_paths"])
    if any(p.resolve().parent != REPO / "configs/approximate" for p in (source_cfg_path, source_paths_path)):
        raise ValueError("source configuration namespace must be approximate")
    sources, reference = load_sources(read_json(source_paths_path), read_json(source_cfg_path))
    if args.limit is not None and (type(args.limit) is not int or not 1 <= args.limit <= min(cfg["datasets"].values())):
        raise ValueError("limit must be within every dataset population")
    output = (args.output or local_path(paths["output"])).resolve()
    protected = [REPO, *(Path(s[key]).resolve() for s in sources.values() for key in
        ("root", "detector_root", "gt_root", "selection_root") if key in s),
        local_path(paths["ctw_reference"]).resolve()]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError("new output must be outside repository and frozen source roots")
    controls = prefilter_controls(cfg["selection"])
    result_class = "APPROXIMATE_REPRODUCTION" if args.limit is None else (
        "SMOKE_TEST" if args.limit == 1 else "PARTIAL_DATASET")
    # Keep the virtualenv entry point: resolving its Python symlink loses isolation.
    geometry_python = local_path(paths["geometry_python"])
    ctw_root = local_path(paths["ctw_reference"])
    ctw_report = read_json(ctw_root / "results.json")
    ctw_plan = read_json(ctw_root / "resolved_configuration.json")
    if (ctw_report["plan_sha256"] != sha(ctw_root / "resolved_configuration.json")
        or read_json(ctw_root / "independent_audit.json")["status"] != "PASS"
        or ctw_report["images"] != cfg["datasets"]["ctw1500"]
        or ctw_plan["source_plan_sha256"] != sources["ctw1500"]["plan_sha256"]):
        raise RuntimeError("prior CTW word audit is not qualified or has different frozen inputs")
    plan = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": cfg["claim_label"],
        "result_class": result_class, "config": cfg, "paths": paths,
        "config_sha256": sha(args.config), "paths_sha256": sha(args.paths),
        "source_config_sha256": sha(source_cfg_path), "source_paths_sha256": sha(source_paths_path),
        "ctw_reference_report_sha256": sha(ctw_root / "results.json"),
        "ctw_reference_plan_sha256": sha(ctw_root / "resolved_configuration.json"),
        "sources": sources, "samples_used": {n: s["records"][:args.limit] for n, s in sources.items()},
        "source_model_identities": {k: reference[k] for k in
            ("yolo_identity", "parseq_identity", "parseq_checkpoint_sha256")},
        "code_sha256": {p: sha(REPO / p) for p in CODE}, "python": sys.version,
        "platform": platform.platform(), "packages": installed_package_inventory(), "output": str(output),
        "execution": {"device": "cpu", "geometry_python": str(geometry_python), "models_loaded": False,
            "new_crops": 0, "new_inference": False, "prediction_selection_uses_gt": False,
            "ignored_gt_use": "metrics only; after final selection and NMS"}}
    freeze(output / "resolved_configuration.json", plan)
    ph = sha(output / "resolved_configuration.json")
    prepared, geometry_images, stage = {}, [], "source_validation"
    try:
        for name, source in sources.items():
            prepared[name] = []
            for record in source["records"][:args.limit]:
                rows = frozen_rows(source, record)
                gt, legacy, ep, raw_clipped = qualify_truth(name, source, record)
                dp = Path(source.get("detector_root", source["root"])) / "detector" / (
                    record["sample_id"].replace(":", "__") + ".json")
                verify_detector_rows(rows, read_json(dp), record)
                prepared[name].append({"record": record, "rows": rows, "truth": gt,
                    "legacy": legacy, "evaluation_path": str(ep), "raw_clipped": raw_clipped})
                geometry_images.append({"sample_id": record["sample_id"],
                    "ignored_regions": [{"polygon_xy": g["polygon_xy"]}
                        for g in gt["ignored_evaluation_polygons"]],
                    "rows": [{"candidate_id": r["candidate_id"], "bbox": r["bbox"]} for r in rows]})
            print(f"[validate] {name}: {len(prepared[name])} images", flush=True)
        stage = "polygon_geometry"
        request = {"schema_version": 1, "reproduction_mode": "approximate", "phase": "evaluation_only",
            "images_modified": False, "ignored_prediction_coverage_threshold": cfg["evaluation"][
                "ignored_prediction_coverage_threshold"], "expected_versions": cfg["geometry_runtime"]["expected_versions"],
            "worker_sha256": sha(REPO / "scripts/dont_care_polygon_worker.py"), "plan_sha256": ph,
            "images": geometry_images}
        geometry = run_geometry(request, output, geometry_python, cfg["geometry_runtime"]["timeout_seconds"])
        geometry_by_id = {im["sample_id"]: im for im in geometry["images"]}
        print(f"[geometry] PASS: {geometry['intersection_comparisons']} independent comparisons", flush=True)
        stage, dataset_reports, checked, distinct_checked = "word_evaluation", {}, 0, 0
        for name, images in prepared.items():
            totals = {control: Counter() for control in controls}
            before_totals = {control: Counter() for control in controls}
            counts, preserved = Counter(), {control: True for control in controls}
            for position, item in enumerate(images, 1):
                record, rows, gt = item["record"], item["rows"], item["truth"]
                sid, key = record["sample_id"], record["sample_id"].replace(":", "__")
                geo = geometry_by_id[sid]
                counts.update(candidates=len(rows), care_gt_regions=len(gt["care"]),
                    ignored_gt_regions=len(gt["ignored"]), clipped_polygons=gt["clipped_polygons"],
                    ignored_raw_polygons_outside_pixel_extent=gt["ignored_raw_polygons_outside_pixel_extent"],
                    raw_annotation_polygons_clipped=item["raw_clipped"],
                    geometry_annotations=len(geo["geometry_annotations"]))
                _, baseline = select_predictions(rows, cfg["selection"], .05)
                baseline_ids = [r["candidate_id"] for r in baseline]
                if baseline_ids != item["legacy"]["selected_candidate_ids"]:
                    raise RuntimeError("final baseline selection differs from original output IDs")
                old_ctw = read_json(ctw_root / "evaluation" / f"{key}.json") if name == "ctw1500" else None
                cache, cells = {}, {}
                for control, threshold in controls.items():
                    eligible, chosen = select_predictions(rows, cfg["selection"], threshold)
                    ids = [r["candidate_id"] for r in chosen]
                    preserved[control] &= ids == baseline_ids
                    if threshold is None and ids != baseline_ids:
                        raise RuntimeError("analytic prefilter changed final outputs")
                    evaluated, neutralized = apply_neutralization(chosen, geo["rows"])
                    signature = tuple(ids)
                    if signature not in cache:
                        before = score_words(name, chosen, gt["care"], cfg["evaluation"])
                        after = score_words(name, evaluated, gt["care"], cfg["evaluation"])
                        cache[signature] = before, after
                        distinct_checked += 2
                    before, after = cache[signature]
                    if old_ctw is not None:
                        prior = old_ctw["controls"][control]
                        if (ids != prior["selected_candidate_ids"] or any(before[k] != prior[k] for k in
                            ("correct_words", "predicted_words", "gt_words", "matched_gt_word_indices"))):
                            raise RuntimeError("baseline CTW word result changed before ignore neutralization")
                    elif threshold is None and (before["correct_words"] != item["legacy"]["correct"]
                        or before["predicted_words"] != item["legacy"]["predictions"]
                        or before["matched_gt_word_indices"] != item["legacy"]["correct_gt_indices"]):
                        raise RuntimeError("baseline word metric differs from frozen report")
                    if after["correct_words"] > before["correct_words"]:
                        raise RuntimeError("removing predictions increased maximum matching count")
                    counts_keys = ("correct_words", "predicted_words", "gt_words")
                    before_totals[control].update({k: before[k] for k in counts_keys})
                    totals[control].update({k: after[k] for k in counts_keys})
                    totals[control].update(produced_outputs=len(chosen), evaluated_outputs=len(evaluated),
                        neutralized_outputs=len(neutralized), crops_eligible_before_parseq=len(eligible),
                        lost_correct_words=before["correct_words"] - after["correct_words"])
                    cells[control] = {"before": before, "after": after,
                        "crops_eligible_before_parseq": len(eligible), "selected_candidate_ids": ids,
                        "evaluated_candidate_ids": [r["candidate_id"] for r in evaluated],
                        "neutralized_candidate_ids": [r["candidate_id"] for r in neutralized]}
                    checked += 1
                freeze(output / "evaluation" / f"{key}.json", {"plan_sha256": ph, "sample_id": sid,
                    "source_ocr_sha256": sha(Path(sources[name]["root"]) / "ocr" / f"{key}.json"),
                    "source_evaluation_sha256": sha(Path(item["evaluation_path"])),
                    "ground_truth": gt, "polygon_audit": geo, "controls": cells})
                if position == 1 or position % 100 == 0 or position == len(images):
                    print(f"[evaluate] {name}: {position}/{len(images)}", flush=True)
            summary = {}
            for control, t in totals.items():
                b = before_totals[control]
                summary[control] = {"before": word_metrics(b["correct_words"], b["predicted_words"], b["gt_words"]),
                    "after": word_metrics(t["correct_words"], t["predicted_words"], t["gt_words"]),
                    **{k: t[k] for k in ("produced_outputs", "evaluated_outputs", "neutralized_outputs",
                        "crops_eligible_before_parseq", "lost_correct_words")},
                    "same_final_outputs_as_reference": preserved[control]}
            primary = summary[cfg["selection"]["primary_method"]]
            best = max((c for c in controls if controls[c] is not None),
                key=lambda c: (summary[c]["after"]["hmean"], summary[c]["after"]["correct_words"], -controls[c]))
            dataset_reports[name] = {"images": len(images), **dict(counts), "summary": summary,
                "primary": primary, "exploratory_best_test_cutoff": best,
                "largest_tested_cutoff_preserving_all_outputs": max(
                    (controls[c] for c in controls if controls[c] is not None and preserved[c]), default=None)}
        stage = "aggregate"
        report = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": cfg["claim_label"],
            "result_class": result_class, "table_5_claim": False, "split": "TEST", "plan_sha256": ph,
            "selection": cfg["selection"], "evaluation": cfg["evaluation"], "datasets": dataset_reports,
            "execution": plan["execution"], "geometry_response_sha256": sha(output / "polygon_response.json"),
            "official_end_to_end_hmean": False}
        freeze(output / "results.json", report)
        freeze(output / "independent_audit.json", {"status": "PASS", "plan_sha256": ph,
            "per_image_controls_checked": checked, "distinct_selected_sets_scored_before_and_after": distinct_checked,
            "polygon_intersections_checked": geometry["intersection_comparisons"],
            "geometry_runtime": geometry["runtime"], "final_output_ids_unchanged": True,
            "frozen_ctw_before_controls_unchanged": True, "images_modified": False, "new_inference": False})
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset", "detector_cutoff", "produced_outputs", "neutralized_outputs", "evaluated_outputs",
                "gt_words", "correct_words", "precision", "recall", "f1_hmean", "old_hmean", "lost_correct_words"])
            for name, dataset in dataset_reports.items():
                for control, cell in dataset["summary"].items():
                    a = cell["after"]
                    writer.writerow([name, control, cell["produced_outputs"], cell["neutralized_outputs"],
                        cell["evaluated_outputs"], a["gt_words"], a["correct_words"], a["precision"],
                        a["recall"], a["hmean"], cell["before"]["hmean"], cell["lost_correct_words"]])
        print(json.dumps({"report": str(output / "results.json"), "primary": {
            n: r["primary"] for n, r in dataset_reports.items()}}, sort_keys=True), flush=True)
        return report
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "stage": stage,
            "error": str(error), "error_type": type(error).__name__, "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Images per dataset; smoke/partial only")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
