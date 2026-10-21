#!/usr/bin/env python3
"""Evaluate GT-blind group/reference refinements on frozen crop artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

from recognition.crop_group_refinements import (confidence_guarded_selection,
                                                consolidate_near_identical_outputs,
                                                robust_group_evidence)
from recognition.crop_word_group_metric import grouped_oracle
from recognition.group_word_selection import select_groups
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import sha
from scripts.run_crop_group_postfilters import load_mass
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/crop_group_refinements_v1.json"
CODE = ("recognition/crop_group_refinements.py", "recognition/group_word_selection.py",
        "recognition/crop_word_group_metric.py", "recognition/scene_text_v1_evaluation.py",
        "scripts/evaluate_crop_group_refinements.py", "scripts/run_crop_group_postfilters.py")
ARMS = ("baseline", "robust_reference", "guarded_reference",
        "baseline_consolidated", "robust_consolidated", "guarded_consolidated")


def validate_config(cfg: dict) -> None:
    q = cfg.get("reference_quantile")
    consolidation = cfg.get("consolidation", {})
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("table_5_claim") is not False or not isinstance(q, (float, int))
        or not 0 <= q < .5 or not isinstance(consolidation.get("box_iou_min"), (float, int))
        or not 0 < consolidation["box_iou_min"] <= 1
        or not isinstance(consolidation.get("one_edit_min_word_length"), int)
        or consolidation["one_edit_min_word_length"] < 1
        or cfg.get("evaluation", {}).get("matching_iou") != .5):
        raise ValueError("invalid isolated approximate crop-group refinement configuration")


def _context(run: Path, sid: str) -> tuple[Path, Path]:
    key = sid.replace(":", "__")
    if (run / "inference").is_dir():
        path = run / "inference" / f"{key}.json"
        return path, path
    return run / "inputs" / f"{key}.json", run / "selection" / f"{key}.json"


def _read_run_context(run: Path, sid: str) -> tuple[dict, list[dict]]:
    context_path, selected_path = _context(run, sid)
    context = json.loads(context_path.read_text())
    selected = context if context_path == selected_path else json.loads(selected_path.read_text())
    if context["sample_id"] != sid or selected["sample_id"] != sid:
        raise RuntimeError("frozen context/sample identity differs")
    return context, selected["methods"]["mask_completeness_confidence"]


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    if config_path.parent != REPO / "configs/approximate":
        raise ValueError("refinement config must remain in configs/approximate")
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    source_run = args.run.resolve()
    run_plan_path = source_run / "resolved_configuration.json"
    run_plan = json.loads(run_plan_path.read_text())
    source = Path(run_plan["source"])
    source_plan_path = source / "resolved_configuration.json"
    source_plan = json.loads(source_plan_path.read_text())
    if (run_plan.get("reproduction_mode") != "approximate"
        or source_plan.get("reproduction_mode") != "approximate"
        or run_plan.get("table_5_claim") is not False
        or source_plan.get("manifest_sha256") != run_plan.get("manifest_sha256")):
        raise RuntimeError("frozen approximate run/source protocol differs")
    manifest_path = args.manifest.resolve()
    if sha(manifest_path) != run_plan["manifest_sha256"]:
        raise RuntimeError("manifest differs from frozen run")
    raw_selector = Path(run_plan.get("selector_run", source_run))
    if not (raw_selector / "inputs").is_dir():
        raise FileNotFoundError("frozen source selector inputs are required for probability provenance")
    samples = {sample.sample_id: sample for sample in
               load_segmentation_manifest(manifest_path, require_checksums=True).samples}
    sample_ids = run_plan["sample_ids"][:args.limit] if args.limit else run_plan["sample_ids"]
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if any(sid not in samples for sid in sample_ids):
        raise RuntimeError("sample absent from frozen manifest")
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("generated experiment artifacts must stay outside the repository")
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "claim_label": cfg["claim_label"], "result_class": "SMOKE_TEST" if args.limit else
            ("APPROXIMATE_TRAIN_DIAGNOSTIC" if source_plan["config"]["split"] == "TRAIN_heldout_125"
             else "APPROXIMATE_TEST_DIAGNOSTIC"),
            "table_5_claim": False, "config": cfg, "config_sha256": sha(config_path),
            "source_run": str(source_run), "source_run_plan_sha256": sha(run_plan_path),
            "source": str(source), "source_plan_sha256": sha(source_plan_path),
            "raw_selector": str(raw_selector), "manifest": str(manifest_path),
            "manifest_sha256": sha(manifest_path), "sample_ids": sample_ids,
            "code_sha256": {path: sha(REPO / path) for path in CODE},
            "execution": {"device": "cpu", "new_model_inference": False,
                          "new_ocr_crops": 0, "gpu_peak_allocated_bytes": None,
                          "gpu_peak_reserved_bytes": None, "python": sys.version,
                          "platform": platform.platform()},
            "ground_truth_use": "all selection shards frozen before opening any evaluation GT"}
    freeze(output / "resolved_configuration.json", plan)
    for position, sid in enumerate(sample_ids, 1):
        key = sid.replace(":", "__")
        context_path, selected_path = _context(source_run, sid)
        context, baseline = _read_run_context(source_run, sid)
        rows = context.get("filtered_rows", context.get("rows"))
        groups = context["groups"]["groups"] if isinstance(context["groups"], dict) else context["groups"]
        original_path = raw_selector / "inputs" / f"{key}.json"
        original = json.loads(original_path.read_text())
        mass = load_mass(sid, selector_input=original, source=source,
                         source_plan=source_plan, sample=samples[sid])
        robust, references = robust_group_evidence(context["evidence"], groups, rows,
            mass, quantile=cfg["reference_quantile"])
        chosen = select_groups(robust, "robust_mask_confidence")
        base_again = select_groups(context["evidence"], "mask_completeness_confidence")
        if baseline != base_again:
            raise RuntimeError(f"frozen baseline selection differs: {sid}")
        guarded, guard_audit = confidence_guarded_selection(baseline, chosen, robust)
        duplicate_cfg = cfg["consolidation"]
        baseline_filtered, baseline_rejected = consolidate_near_identical_outputs(
            baseline, box_iou_min=duplicate_cfg["box_iou_min"],
            min_edit_length=duplicate_cfg["one_edit_min_word_length"])
        robust_filtered, robust_rejected = consolidate_near_identical_outputs(
            chosen, box_iou_min=duplicate_cfg["box_iou_min"],
            min_edit_length=duplicate_cfg["one_edit_min_word_length"])
        guarded_filtered, guarded_rejected = consolidate_near_identical_outputs(
            guarded, box_iou_min=duplicate_cfg["box_iou_min"],
            min_edit_length=duplicate_cfg["one_edit_min_word_length"])
        selected_components = {item["group_id"]: {
            "robust_reference_bbox": references[item["group_id"]],
            "ocr_confidence": robust_by_group["candidates"][item["candidate_id"]]["scores"]["confidence_balanced"],
            "robust_mask_support": robust_by_group["candidates"][item["candidate_id"]]["scores"]["robust_mask_support"]}
            for robust_by_group in robust for item in chosen
            if item["group_id"] == robust_by_group["group_id"]}
        freeze(output / "selection" / f"{key}.json", {
            "sample_id": sid, "context_sha256": sha(context_path),
            "frozen_selection_sha256": sha(selected_path),
            "raw_selector_input_sha256": sha(original_path),
            "aligned_probability_sha256": original["contract"]["aligned_probability_sha256"],
            "groups": len(groups), "source_candidates": len(rows),
            "methods": {"baseline": baseline, "robust_reference": chosen,
                        "guarded_reference": guarded,
                        "baseline_consolidated": baseline_filtered,
                        "robust_consolidated": robust_filtered,
                        "guarded_consolidated": guarded_filtered},
            "robust_score_components": selected_components,
            "guard_audit": guard_audit,
            "rejections": {"baseline_consolidated": baseline_rejected,
                           "robust_consolidated": robust_rejected,
                           "guarded_consolidated": guarded_rejected},
            "ground_truth_use": "none"})
        if position % 25 == 0 or position == len(sample_ids):
            print(f"[selection] {position}/{len(sample_ids)}", flush=True)
    totals = {name: Counter() for name in ARMS}
    all_oracle = denominator = groups_total = crops_total = 0
    for position, sid in enumerate(sample_ids, 1):
        key = sid.replace(":", "__")
        context, _ = _read_run_context(source_run, sid)
        rows = context.get("filtered_rows", context.get("rows"))
        groups = context["groups"]["groups"] if isinstance(context["groups"], dict) else context["groups"]
        frozen = json.loads((output / "selection" / f"{key}.json").read_text())
        upstream = json.loads((source / "evaluation" / f"{key}.json").read_text())
        gt = upstream["ground_truth"]
        evaluated = {name: evaluate_predictions(frozen["methods"][name], gt,
                                                cfg["evaluation"]["matching_iou"])
                     for name in ARMS}
        base = set(evaluated["baseline"]["correct_gt_indices"])
        for name, item in evaluated.items():
            item["recovered_vs_baseline"] = len(set(item["correct_gt_indices"]) - base)
            item["lost_vs_baseline"] = len(base - set(item["correct_gt_indices"]))
            totals[name].update({field: item[field] for field in
                                ("predictions", "correct", "recovered_vs_baseline", "lost_vs_baseline")})
        witnesses = {int(gi): cids for gi, cids in
                     upstream["arms"]["all_methods"]["witnesses"].items()}
        oracle = grouped_oracle(groups, witnesses, len(gt))
        if any(item["correct"] > len(oracle) for item in evaluated.values()):
            raise RuntimeError("selection exceeds one-per-group oracle")
        all_oracle += len(oracle)
        denominator += len(gt)
        groups_total += len(groups)
        crops_total += len(rows)
        freeze(output / "evaluation" / f"{key}.json", {
            "sample_id": sid, "selection_sha256": sha(output / "selection" / f"{key}.json"),
            "source_evaluation_sha256": sha(source / "evaluation" / f"{key}.json"),
            "gt_words": len(gt), "group_oracle_correct_gt_indices": sorted(oracle),
            "methods": evaluated})
        if position % 25 == 0 or position == len(sample_ids):
            print(f"[evaluation] {position}/{len(sample_ids)}", flush=True)
    result = {"schema_version": 1, "result_class": plan["result_class"],
              "reproduction_mode": "approximate", "table_5_claim": False,
              "plan_sha256": sha(output / "resolved_configuration.json"),
              "images": len(sample_ids), "gt_words": denominator,
              "source_candidates": crops_total, "source_groups": groups_total,
              "group_oracle_correct": all_oracle,
              "summary": {name: {**dict(total),
                         "correct_rate": total["correct"] / denominator if denominator else 0,
                         "precision_proxy": total["correct"] / total["predictions"] if total["predictions"] else 0}
                          for name, total in totals.items()},
              "metric": "exact normalized reading; axis-aligned bbox IoU>=0.5; one-to-one matching; all GT counted",
              "limitation": "Approximate local diagnostic, not official Total-Text polygon spotting Hmean"}
    freeze(output / "results.json", result)
    print(json.dumps({"report": str(output / "results.json"), "summary": result["summary"]}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="frozen selector run or filtered grouping/selection run")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, help="smoke test only")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
