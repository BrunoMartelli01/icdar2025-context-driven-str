"""Cached TEST threshold controls and CTW full-line failure diagnostics; no inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import yaml

from localize.independent_word_slots import overlap_iou
from recognition.group_word_selection import normalize
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, verify_parseq_runtime_source
from recognition.scene_text_v1_evaluation import maximum_matching
from recognition.yolo_word_refinement import joint_score, nms, ocr_quality
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import sha
from scripts.evaluate_scene_text_yolo_parseq import unit_metrics
from scripts.evaluate_totaltext_yolo_word_slots import read_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/scene_text_yolo_threshold_sweep_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/scene_text_yolo_threshold_sweep_local_paths_v1.json"
CODE = ("scripts/analyze_scene_text_yolo_thresholds.py", "scripts/test_scene_text_yolo_thresholds.sh",
    "recognition/yolo_word_refinement.py", "recognition/group_word_selection.py",
    "recognition/scene_text_v1_evaluation.py", "localize/independent_word_slots.py",
    "scripts/evaluate_scene_text_yolo_parseq.py", "scripts/crop_clip_runtime.py",
    "scripts/analyze_totaltext_crop_word_groups.py")


def threshold_grid(cfg: dict) -> list[float]:
    grid = cfg.get("threshold_grid", {})
    if set(grid) != {"minimum", "maximum", "step"}:
        raise ValueError("threshold grid must specify minimum, maximum and step")
    low, high, step = (grid[k] for k in ("minimum", "maximum", "step"))
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (low, high, step)):
        raise ValueError("threshold grid must be finite")
    if not 0 < low <= high <= 1 or step <= 0:
        raise ValueError("threshold grid is outside confidence range")
    count = (high - low) / step
    if not math.isclose(count, round(count), abs_tol=1e-9) or count > 1000:
        raise ValueError("threshold grid must be finite, evenly spaced and bounded")
    return [round(low + index * step, 10) for index in range(round(count) + 1)]


def validate_config(cfg: dict) -> list[float]:
    grid = threshold_grid(cfg)
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("table_5_claim") is not False or cfg.get("exploratory_test_analysis") is not True
        or cfg.get("datasets") != ["icdar13", "icdar2015", "ctw1500"]
        or cfg.get("families") != ["detector_score", "joint_text_nms"]
        or cfg.get("baseline_joint_threshold") != 0.6 or 0.6 not in grid
        or cfg.get("text_nms_iou") != 0.5 or cfg.get("matching_iou") != 0.5
        or cfg.get("diagnostic_minimum_prediction_overlap") != 0.5):
        raise ValueError("invalid isolated approximate TEST threshold analysis")
    return grid


def threshold_key(value: float) -> str:
    return format(value, ".10f").rstrip("0").rstrip(".")


def prediction_overlap(box, reference) -> float:
    common = max(0, min(box[2], reference[2]) - max(box[0], reference[0])) * max(
        0, min(box[3], reference[3]) - max(box[1], reference[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        raise ValueError("prediction has no positive area")
    return common / area


def line_diagnostics(rows, gt, correct_indices, overlap_minimum) -> dict:
    correct_indices = set(correct_indices)
    counts, failures = {}, []
    for gi, word in enumerate(gt):
        tokens = normalize(word["text"]).split()
        kind = "multi_token" if len(tokens) > 1 else "single_token" if tokens else "normalized_empty"
        counts.setdefault(kind, Counter()).update(gt=1, correct=int(gi in correct_indices))
        if gi in correct_indices:
            continue
        fragments = []
        for row in rows:
            reading = normalize(row["raw_text"]).split()
            if not reading or len(reading) >= len(tokens):
                continue
            if (any(tokens[start:start + len(reading)] == reading
                    for start in range(len(tokens) - len(reading) + 1))
                and prediction_overlap(row["bbox"], word["bbox"]) >= overlap_minimum):
                fragments.append({"candidate_id": row["candidate_id"], "raw_text": row["raw_text"],
                    "bbox": row["bbox"], "joint_score": joint_score(row)})
        distinct = {normalize(f["raw_text"]) for f in fragments}
        counts[kind].update(missed_with_fragment=int(bool(fragments)),
            missed_with_two_distinct_fragments=int(len(distinct) >= 2))
        failures.append({"gt_index": gi, "text": word["text"], "bbox": word["bbox"],
                         "token_count": len(tokens), "fragments": fragments})
    return {"counts": {name: dict(count) for name, count in counts.items()}, "failures": failures}


def score_subset(indices, gt_count, text_edges, spatial_edges, normalized_rows) -> dict:
    remap = {old: new for new, old in enumerate(indices)}
    text = [(remap[pi], gi) for pi, gi in text_edges if pi in remap]
    geometry = [(remap[pi], gi) for pi, gi in spatial_edges if pi in remap]
    correct = maximum_matching(len(indices), gt_count, text)
    coverage = maximum_matching(len(indices), gt_count, geometry)
    return {"outputs": len(indices), "correct": len(correct), "correct_gt_indices": sorted(correct),
        "detector_gt_box_matches": len(coverage),
        "normalized_empty_outputs": sum(not normalized_rows[pi] for pi in indices)}


def best_control(controls: list[dict]) -> dict:
    return max(controls, key=lambda row: (row["hmean"], row["correct"], -row["threshold"]))


def shared_control(datasets: dict, baseline: dict) -> dict | None:
    by_name = {name: {r["threshold"]: r for r in data["controls"]["joint_text_nms"]}
               for name, data in datasets.items()}
    common = set.intersection(*(set(rows) for rows in by_name.values()))
    candidates = []
    for threshold in sorted(common):
        gains = {name: rows[threshold]["hmean"] - baseline[name]["hmean"]
                 for name, rows in by_name.items()}
        if all(gain > 0 for gain in gains.values()):
            candidates.append({"threshold": threshold, "gains": gains,
                "datasets": {name: rows[threshold] for name, rows in by_name.items()}})
    return max(candidates, key=lambda row: (min(row["gains"].values()),
        sum(row["gains"].values()) / len(row["gains"]), -row["threshold"])) if candidates else None


def analyze_dataset(name, source, output, source_plan, report, cfg, grid, plan_hash):
    totals = {family: {threshold: Counter() for threshold in grid} for family in cfg["families"]}
    diag_totals, diag_examples = {}, []
    records = source_plan["samples"]
    granularity = report["annotation_granularity"]
    gt_count, candidates, ignored = 0, 0, 0
    candidate_floor_counts = {threshold: 0 for threshold in grid}
    for position, record in enumerate(records, 1):
        sid, key = record["sample_id"], record["sample_id"].replace(":", "__")
        op, ep = source / "ocr" / f"{key}.json", source / "evaluation" / f"{key}.json"
        ocr, evaluation = read_json(op), read_json(ep)
        dp = source / "detector" / f"{key}.json"
        if (ocr["contract"] != {"plan_sha256": report["plan_sha256"], "sample_id": sid,
                                "detector_sha256": sha(dp)}
            or evaluation["ocr_sha256"] != sha(op)
            or sha(Path(record["annotation_path"])) != record["annotation_sha256"]):
            raise RuntimeError(f"frozen OCR/evaluation identity differs for {sid}")
        rows, gt = ocr["rows"], evaluation["ground_truth"]["care"]
        gt_count += len(gt)
        ignored += len(evaluation["ground_truth"]["ignored"])
        candidates += len(rows)
        ids = {row["candidate_id"]: pi for pi, row in enumerate(rows)}
        if len(ids) != len(rows):
            raise RuntimeError("duplicate frozen candidate IDs")
        normalized = [normalize(row["raw_text"]) for row in rows]
        detector_scores = [row["detector_score"] for row in rows]
        qualities = [ocr_quality(row) for row in rows]
        joint_scores = [joint_score(row) for row in rows]
        # A rejected weaker box never suppresses a stronger box in descending NMS.
        survivors = nms([r for r, text in zip(rows, normalized) if text],
                        cfg["text_nms_iou"], same_text=True, joint=True)
        survivor_indices = [ids[row["candidate_id"]] for row in survivors]
        primary = [row for row in survivors if joint_scores[ids[row["candidate_id"]]] >= cfg["baseline_joint_threshold"]]
        legacy = evaluation["arms"][report["primary_method"]]
        if [r["candidate_id"] for r in primary] != legacy["selected_candidate_ids"]:
            raise RuntimeError("nested-threshold NMS differs from frozen baseline IDs")
        spatial_edges = [(pi, gi) for pi, row in enumerate(rows) for gi, word in enumerate(gt)
                         if overlap_iou(row["bbox"], word["bbox"]) >= cfg["matching_iou"]]
        text_edges = [(pi, gi) for pi, gi in spatial_edges
                      if normalize(gt[gi]["text"]) and normalized[pi] == normalize(gt[gi]["text"])]
        cells = {}
        for family in cfg["families"]:
            cells[family] = {}
            for threshold in grid:
                selected = ([pi for pi, score in enumerate(detector_scores) if score >= threshold]
                            if family == "detector_score" else
                            [pi for pi in survivor_indices if joint_scores[pi] >= threshold])
                outcome = score_subset(selected, len(gt), text_edges, spatial_edges, normalized)
                totals[family][threshold].update({k: outcome[k] for k in
                    ("outputs", "correct", "detector_gt_box_matches", "normalized_empty_outputs")})
                cells[family][threshold_key(threshold)] = outcome
                if family == "joint_text_nms":
                    # Necessary bound Qocr<=1: prune before PARSeq without losing an admitted crop.
                    candidate_floor_counts[threshold] += sum(math.sqrt(s) >= threshold for s in detector_scores)
        baseline = cells["joint_text_nms"][threshold_key(cfg["baseline_joint_threshold"])]
        if (baseline["correct"] != legacy["correct"]
            or baseline["correct_gt_indices"] != legacy["correct_gt_indices"]):
            raise RuntimeError("cached threshold rescore differs from baseline matching")
        diag = line_diagnostics(primary, gt, baseline["correct_gt_indices"],
                               cfg["diagnostic_minimum_prediction_overlap"]) if granularity == "text_line" else None
        if diag:
            for category, count in diag["counts"].items():
                diag_totals.setdefault(category, Counter()).update(count)
            for failure in diag["failures"]:
                if len(diag_examples) < 10 and len({normalize(f["raw_text"]) for f in failure["fragments"]}) >= 2:
                    diag_examples.append({"sample_id": sid, **failure})
        freeze(output / "evaluation" / name / f"{key}.json", {"plan_sha256": plan_hash,
            "sample_id": sid, "source_ocr_sha256": sha(op), "source_evaluation_sha256": sha(ep),
            "gt_units": len(gt), "controls": cells, "line_diagnostics": diag,
            "candidate_scores": [{"candidate_id": row["candidate_id"],
                "detector_score": detector_scores[pi], "ocr_quality": qualities[pi],
                "joint_score": joint_scores[pi], "joint_nms_survivor": pi in survivor_indices}
                for pi, row in enumerate(rows)]})
        if position == 1 or position % 100 == 0 or position == len(records):
            print(f"[cached thresholds] {name} {position}/{len(records)}", flush=True)
    if (gt_count != report["gt_units"] or candidates != report["generated_crops"]
        or ignored != report["ignored_gt_units"]):
        raise RuntimeError("frozen TEST denominators or crop counts differ")
    controls = {family: [{"threshold": threshold, **unit_metrics(t["correct"], t["outputs"], gt_count, granularity),
        "detector_gt_box_matches": t["detector_gt_box_matches"],
        "normalized_empty_outputs": t["normalized_empty_outputs"],
        **({"crops_necessary_before_parseq": candidate_floor_counts[threshold]} if family == "joint_text_nms" else {})}
        for threshold, t in total.items()] for family, total in totals.items()}
    return {"images": len(records), "gt_units": gt_count, "annotation_granularity": granularity,
        "generated_crops": candidates, "ignored_gt_units": ignored, "controls": controls,
        "best_exploratory": {family: best_control(rows) for family, rows in controls.items()},
        "baseline": report["summary"][report["primary_method"]],
        "line_diagnostics": {"counts": {k: dict(v) for k, v in diag_totals.items()}, "examples": diag_examples}
                             if granularity == "text_line" else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("scientific configuration must stay in configs/approximate")
    cfg, paths = read_json(args.config), read_json(args.paths)
    grid = validate_config(cfg)
    if paths.get("reproduction_mode") != "approximate":
        raise ValueError("local paths must declare approximate mode")
    source, output = Path(paths["source"]).resolve(), (args.output or Path(paths["output"])).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output) or output.is_relative_to(REPO):
        raise ValueError("choose a separate external analysis output directory")
    aggregate = read_json(source / "results.json")
    if aggregate.get("reproduction_mode") != "approximate" or set(aggregate["datasets"]) != set(cfg["datasets"]):
        raise RuntimeError("source is not the complete approximate three-dataset transfer run")
    source_plans = {}
    for name in cfg["datasets"]:
        report = read_json(source / name / "results.json")
        plan_path = source / name / "resolved_configuration.json"
        plan = read_json(plan_path)
        if (report != aggregate["datasets"][name] or report["plan_sha256"] != sha(plan_path)
            or plan["reproduction_mode"] != "approximate" or plan["split"] != "TEST"
            or report["images"] != plan["config"]["datasets"][name]["images"]
            or plan["config"]["matching_iou"] != cfg["matching_iou"]
            or plan["config"]["merge_iou"] != cfg["text_nms_iou"]
            or report["primary_method"] != f"joint_text_nms_{cfg['baseline_joint_threshold']}"
            or min(grid) < plan["config"]["inference"]["confidence_floor"]):
            raise RuntimeError("source split/metric/threshold contract differs")
        for filename, expected in plan["code_sha256"].items():
            if sha(REPO / filename) != expected:
                raise RuntimeError(f"frozen inference code differs: {filename}")
        source_plans[name] = plan
    runtime = Path(source_plans["ctw1500"]["paths"]["parseq_source"])
    identity = verify_parseq_runtime_source(runtime, source_type=PARSEQ_ARCHIVE_SOURCE,
        source_archive=Path(source_plans["ctw1500"]["paths"]["parseq_archive"]))
    if json.loads(json.dumps(identity)) != source_plans["ctw1500"]["parseq_identity"]:
        raise RuntimeError("qualified PARSeq runtime identity changed since the frozen GPU run")
    charset_path, main_path = runtime / "configs/charset/94_full.yaml", runtime / "configs/main.yaml"
    charset = yaml.safe_load(charset_path.read_text())["model"]["charset_train"]
    if any(character.isspace() for character in charset):
        raise RuntimeError("qualified PARSeq charset now includes whitespace; update failure diagnosis")
    plan = {"reproduction_mode": "approximate", "config": cfg, "paths": paths, "output": str(output),
        "source_results_sha256": sha(source / "results.json"),
        "source_plan_sha256": {name: sha(source / name / "resolved_configuration.json") for name in cfg["datasets"]},
        "code_sha256": {name: sha(REPO / name) for name in CODE}, "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML")},
        "inference": "none; CPU cached rescore only", "test_selection_bias": True,
        "parseq_charset": {"characters": len(charset), "contains_whitespace": False,
            "max_label_length": yaml.safe_load(main_path.read_text())["model"]["max_label_length"],
            "charset_config_sha256": sha(charset_path), "main_config_sha256": sha(main_path)}}
    freeze(output / "resolved_configuration.json", plan)
    ph = sha(output / "resolved_configuration.json")
    datasets = {name: analyze_dataset(name, source / name, output, source_plans[name],
        aggregate["datasets"][name], cfg, grid, ph) for name in cfg["datasets"]}
    common = shared_control(datasets, {name: data["baseline"] for name, data in datasets.items()})
    result = {"reproduction_mode": "approximate", "result_class": "EXPLORATORY_TEST_ANALYSIS",
        "table_5_claim": False, "plan_sha256": ph, "test_selection_bias": True,
        "datasets": datasets, "shared_joint_threshold_improving_all": common,
        "limitation": "TEST-selected maxima; local bbox metric. CTW full-line diagnostic is not word accuracy."}
    freeze(output / "results.json", result)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "granularity", "family", "threshold", "correct", "outputs", "precision", "recall", "hmean", "crops_necessary_before_parseq"])
        writer.writeheader()
        for name, data in datasets.items():
            for family, controls in data["controls"].items():
                for row in controls:
                    writer.writerow({"dataset": name, "granularity": data["annotation_granularity"],
                        "family": family, **{k: row.get(k, "") for k in writer.fieldnames[3:]}})
    print(json.dumps({"report": str(output / "results.json"), "best": {name: d["best_exploratory"] for name, d in datasets.items()},
                      "shared": common}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
