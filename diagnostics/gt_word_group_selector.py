#!/usr/bin/env python3
"""Measure crop choice with one oracle-localized group per annotated word.

Ground truth is used to assign crops to groups. Within each group, the frozen
selector receives a reference bbox chosen only from member crop geometry.
This is an oracle diagnostic, never an inference or benchmark result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from recognition.crop_word_group_metric import grouped_oracle
from recognition.group_word_selection import group_evidence, normalize, select_groups
from recognition.scene_text_v1_evaluation import box_iou
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import sha
from scripts.run_crop_group_postfilters import discover_selector, load_mass, DEFAULT_PATHS
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/gt_word_group_selector_v1.json"
CODE = ("diagnostics/gt_word_group_selector.py", "recognition/group_word_selection.py",
        "recognition/crop_word_group_metric.py", "recognition/scene_text_v1_evaluation.py",
        "scripts/run_crop_group_postfilters.py", "scripts/run_crop_group_word_selection.py",
        "scripts/analyze_totaltext_crop_word_groups.py", "scripts/crop_clip_runtime.py",
        "scripts/evaluate_totaltext_letterbox_threshold_sweep_train.py",
        "scripts/run_totaltext_crop_recovery_suite.py", "segmentation/data.py")


def load_config(path: Path) -> dict:
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("GT grouping oracle config must belong to configs/approximate")
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1 or value.get("reproduction_mode") != "approximate"
        or value.get("result_class") != "ORACLE_TEST_DIAGNOSTIC"
        or value.get("table_5_claim") is not False or value.get("matching_iou") != .5
        or value.get("primary_selector") != "mask_completeness_confidence"
        or value.get("control_selector") != "confidence_balanced"
        or value.get("normalization") != "casefold_remove_unicode_P_S"):
        raise ValueError("invalid GT grouping oracle configuration")
    return value


def build_gt_groups(rows: list[dict], gt: list[dict], *, iou_min: float) -> list[dict]:
    """GT supplies membership only; reference boxes are member-crop medoids."""
    if not 0 < iou_min <= 1:
        raise ValueError("invalid oracle membership IoU")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate original crop IDs")
    groups = []
    for gi, word in enumerate(gt):
        members = [row for row in rows if box_iou(row["bbox"], word["bbox"]) >= iou_min]
        if members:
            median = np.median(np.asarray([row["bbox"] for row in members], dtype=np.float64), axis=0)
            reference = min(members, key=lambda row: (
                float(np.abs(np.asarray(row["bbox"], dtype=np.float64) - median).sum()),
                row["candidate_id"]))
            reference_box, reference_id = reference["bbox"], reference["candidate_id"]
        else:
            reference_box, reference_id = None, None
        groups.append({"group_id": f"gt:{gi}", "gt_index": gi,
                       "candidate_ids": sorted(row["candidate_id"] for row in members),
                       "reference_candidate_id": reference_id,
                       "reference_bbox": reference_box})
    return groups


def run(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    selector = (args.selector_dir or discover_selector(args.paths)).resolve()
    selector_plan_path, selector_result_path = selector / "resolved_configuration.json", selector / "results.json"
    selector_plan = json.loads(selector_plan_path.read_text())
    selector_result = json.loads(selector_result_path.read_text())
    if (selector_plan["reproduction_mode"] != "approximate"
        or selector_result["result_class"] != "APPROXIMATE_TEST_DIAGNOSTIC"
        or selector_result["images"] != len(selector_plan["sample_ids"])):
        raise ValueError("expected complete frozen TEST selectors")
    source = Path(selector_plan["source"])
    source_plan = json.loads((source / "resolved_configuration.json").read_text())
    grouping = Path(selector_plan["grouping"])
    grouping_plan = json.loads((grouping / "resolved_configuration.json").read_text())
    allowed = set(grouping_plan["config"]["source_methods"])
    if (sha(source / "resolved_configuration.json") != selector_plan["source_plan_sha256"]
        or sha(grouping / "resolved_configuration.json") != selector_plan["grouping_plan_sha256"]):
        raise RuntimeError("frozen source or grouping plan changed")
    manifest_path = Path(source_plan["paths_profile"]["test_manifest"])
    if sha(manifest_path) != selector_plan["manifest_sha256"]:
        raise RuntimeError("frozen manifest changed")
    samples = {sample.sample_id: sample for sample in
               load_segmentation_manifest(manifest_path, require_checksums=True).samples}
    ids = selector_plan["sample_ids"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        ids = ids[:args.limit]
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "result_class": "SMOKE_TEST" if args.limit else cfg["result_class"],
            "table_5_claim": False, "oracle_grouping": True,
            "selector_dir": str(selector), "selector_plan_sha256": sha(selector_plan_path),
            "selector_result_sha256": sha(selector_result_path), "source": str(source),
            "source_plan_sha256": sha(source / "resolved_configuration.json"),
            "manifest_sha256": sha(manifest_path), "config": cfg, "config_sha256": sha(args.config),
            "sample_ids": ids, "code_sha256": {name: sha(REPO / name) for name in CODE},
            "execution": {"device": "cpu", "new_model_inference": False,
                          "new_ocr_crops": 0, "numpy": np.__version__},
            "ground_truth_use": "GT boxes define crop-group membership; GT text is used only after selections are frozen"}
    identity = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() if args.output else selector.parent / f"gt-word-groups-{identity}"
    if output == REPO or REPO in output.parents:
        raise ValueError("oracle artifacts must be outside the repository")
    freeze(output / "resolved_configuration.json", plan)
    method_names = (cfg["primary_selector"], cfg["control_selector"])
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        input_path = selector / "inputs" / f"{key}.json"
        source_eval_path = source / "evaluation" / f"{key}.json"
        context, source_eval = json.loads(input_path.read_text()), json.loads(source_eval_path.read_text())
        if context["sample_id"] != sid or source_eval["sample_id"] != sid:
            raise RuntimeError(f"sample identity mismatch: {sid}")
        gt, rows = source_eval["ground_truth"], context["rows"]
        groups = build_gt_groups(rows, gt, iou_min=cfg["matching_iou"])
        nonempty = [group for group in groups if group["candidate_ids"]]
        mass = load_mass(sid, selector_input=context, source=source,
                         source_plan=source_plan, sample=samples[sid])
        evidence = group_evidence(rows, nonempty, allowed_methods=allowed, mass=mass)
        selected = {method: select_groups(evidence, method) for method in method_names}
        freeze(output / "selection" / f"{key}.json", {
            "sample_id": sid, "selector_input_sha256": sha(input_path),
            "source_evaluation_sha256": sha(source_eval_path),
            "groups": groups, "methods": selected,
            "ground_truth_use": plan["ground_truth_use"]})
        print(f"[GT groups] {position}/{len(ids)} {sid}", flush=True)
    totals = {method: Counter() for method in method_names}
    counts = Counter()
    denominator = 0
    for sid in ids:
        key = sid.replace(":", "__")
        source_eval_path = source / "evaluation" / f"{key}.json"
        source_eval = json.loads(source_eval_path.read_text())
        gt = source_eval["ground_truth"]
        shard_path = output / "selection" / f"{key}.json"
        shard = json.loads(shard_path.read_text())
        if shard["source_evaluation_sha256"] != sha(source_eval_path):
            raise RuntimeError(f"oracle GT source changed: {sid}")
        groups = shard["groups"]
        by_group = {group["group_id"]: group for group in groups}
        witnesses = {int(gi): cids for gi, cids in
                     source_eval["arms"]["all_methods"]["witnesses"].items()}
        oracle_correct = len(grouped_oracle(groups, witnesses, len(gt)))
        counts.update(groups=len(groups), groups_with_crops=sum(bool(g["candidate_ids"]) for g in groups),
                      groups_with_exact_crop=sum(bool(set(g["candidate_ids"]) & set(witnesses.get(g["gt_index"], [])))
                                                 for g in groups), oracle_correct=oracle_correct)
        per_method = {}
        for method, predictions in shard["methods"].items():
            direct = sum(normalize(prediction["raw_text"]) == normalize(gt[by_group[prediction["group_id"]]["gt_index"]]["text"])
                         and bool(normalize(gt[by_group[prediction["group_id"]]["gt_index"]]["text"]))
                         for prediction in predictions)
            evaluated = evaluate_predictions(predictions, gt, cfg["matching_iou"])
            totals[method].update(predictions=len(predictions), correct=direct,
                                  globally_matched=evaluated["correct"])
            per_method[method] = {"predictions": len(predictions), "correct": direct,
                                  "globally_matched": evaluated["correct"],
                                  "selected_group_indices": [by_group[p["group_id"]]["gt_index"] for p in predictions],
                                  "correct_gt_indices": evaluated["correct_gt_indices"]}
        denominator += len(gt)
        freeze(output / "evaluation" / f"{key}.json", {
            "sample_id": sid, "selection_sha256": sha(shard_path),
            "source_evaluation_sha256": sha(source_eval_path), "gt_words": len(gt),
            "oracle_correct": oracle_correct, "methods": per_method})
    if not args.limit and (denominator != selector_result["gt_words"]
                           or counts["oracle_correct"] != selector_result["group_oracle_correct"]):
        raise RuntimeError("GT-group oracle ceiling differs from frozen crop-group oracle")
    summary = {}
    for method, values in totals.items():
        correct, predictions = values["correct"], values["predictions"]
        precision = correct / predictions if predictions else 0.0
        recall = correct / denominator if denominator else 0.0
        summary[method] = {"predictions": predictions, "correct": correct,
                           "globally_matched": values["globally_matched"],
                           "wrong_or_unmatched": predictions - correct,
                           "precision_proxy": precision, "recall_proxy": recall,
                           "hmean_proxy": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    result = {"schema_version": 1, "reproduction_mode": "approximate",
              "result_class": plan["result_class"], "table_5_claim": False,
              "oracle_grouping": True, "images": len(ids), "gt_words": denominator,
              "groups": counts["groups"], "groups_with_crops": counts["groups_with_crops"],
              "groups_with_exact_crop": counts["groups_with_exact_crop"],
              "group_oracle_correct": counts["oracle_correct"], "summary": summary,
              "per_image": [f"evaluation/{sid.replace(':', '__')}.json" for sid in ids],
              "plan_sha256": sha(output / "resolved_configuration.json")}
    freeze(output / "results.json", result)
    print(json.dumps({"report": str(output / "results.json"), "summary": result}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--selector-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
