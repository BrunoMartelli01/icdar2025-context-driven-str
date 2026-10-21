#!/usr/bin/env python3
"""Filter frozen crop-group outputs without rerunning segmentation, PARSeq or CLIP."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from recognition.group_output_filter import abstain_uncorroborated, suppress_duplicate_regions
from recognition.group_word_selection import ProbabilityMass, aligned_probability
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import read_image, sha
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _load_probability
from scripts.run_crop_group_word_selection import evaluate_predictions
from scripts.run_totaltext_crop_recovery_suite import read_tile
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/crop_group_postfilters_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/totaltext_crop_group_test_local_paths_v1.json"
METHODS = ("duplicate_suppression", "evidence_abstention", "abstention_then_suppression")
CODE = ("recognition/group_output_filter.py", "scripts/run_crop_group_postfilters.py",
        "recognition/group_word_selection.py", "scripts/run_crop_group_word_selection.py",
        "recognition/scene_text_v1_evaluation.py")


def validate_config(value: dict) -> None:
    duplicate = value.get("duplicate_suppression", {})
    if (value.get("schema_version") != 1 or value.get("reproduction_mode") != "approximate"
        or value.get("table_5_claim") is not False or tuple(value.get("methods", [])) != METHODS
        or value.get("input_method") != "mask_completeness_confidence"
        or value.get("evaluation", {}).get("matching_iou") != .5
        or value.get("evidence_abstention", {}).get("confidence_cutoff", False) is not None
        or any(not isinstance(duplicate.get(field), (float, int)) or not 0 < duplicate[field] <= 1
               for field in ("box_iou_min", "soft_mask_iou_min"))
        or "dataset" in value or "weights" in value):
        raise ValueError("invalid isolated approximate postfilter configuration")


def load_config(path: Path) -> dict:
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("postfilter configuration must belong to configs/approximate")
    value = json.loads(path.read_text())
    validate_config(value)
    return value


def discover_selector(paths: Path) -> Path:
    profile = json.loads(paths.read_text())
    if profile.get("path_base") != "repository_root":
        raise ValueError("invalid local path profile")
    root = Path(profile["output_root"])
    root = root if root.is_absolute() else REPO / root
    eligible = []
    for path in root.glob("*/selectors/results.json"):
        result = json.loads(path.read_text())
        if (result.get("images") == 300 and result.get("cached_only") is False
            and result.get("result_class") == "APPROXIMATE_TEST_DIAGNOSTIC"
            and "mask_completeness_confidence" in result.get("summary", {})):
            eligible.append(path)
    if not eligible:
        raise FileNotFoundError("no completed 300-image TEST selector run; pass --selector-dir")
    return max(eligible, key=lambda path: (path.stat().st_mtime_ns, str(path))).parent


def load_mass(sid: str, *, selector_input: dict, source: Path, source_plan: dict,
              sample) -> ProbabilityMass:
    key = sid.replace(":", "__")
    prepared_path = source / "prepared" / f"{key}.json"
    prepared = json.loads(prepared_path.read_text())
    if sha(prepared_path) != selector_input["contract"]["prepared_sha256"]:
        raise RuntimeError(f"source prepared crop contract changed: {sid}")
    image, image_sha = read_image(Path(source_plan["paths"]["data_root"]), sample,
                                  prepared["contract"]["image_sha256"])
    if image_sha != selector_input["image_sha256"]:
        raise RuntimeError(f"selector image contract changed: {sid}")
    probability_root = Path(source_plan["paths"]["probability_cache"]) / "probabilities"
    p_path = probability_root / f"{key}.npy"
    sidecar = probability_root / f"{key}.json"
    if (sha(sidecar) != prepared["contract"]["probability_sidecar_sha256"]
        or sha(p_path) != selector_input["contract"]["whole_probability_sha256"]):
        raise RuntimeError(f"source probability contract changed: {sid}")
    whole = _load_probability(p_path, json.loads(sidecar.read_text()), sample=sample,
                              protocol_sha=source_plan["probability_sidecar_contract_sha256"])
    tiles = []
    if prepared["contract"]["tile_shard_sha256"] is not None:
        tile_path = source / "tiles" / f"{key}.json"
        if sha(tile_path) != prepared["contract"]["tile_shard_sha256"]:
            raise RuntimeError(f"source tile contract changed: {sid}")
        tile_rows = json.loads(tile_path.read_text())["tiles"]
        if [{"path": row["path"], "sha256": row["file_sha256"]} for row in tile_rows] != selector_input["contract"]["tiles"]:
            raise RuntimeError(f"selector tile list changed: {sid}")
        tiles = [(read_tile(source, row), row["window"]) for row in tile_rows]
    probability = aligned_probability(whole, tiles, (image.shape[1], image.shape[0]))
    if sha256(probability.tobytes()).hexdigest() != selector_input["contract"]["aligned_probability_sha256"]:
        raise RuntimeError(f"aligned probability changed: {sid}")
    return ProbabilityMass(probability, (image.shape[1], image.shape[0]))


def _rates(result: Counter, denominator: int) -> dict:
    predictions, correct = result["predictions"], result["correct"]
    precision = correct / predictions if predictions else 0.0
    recall = correct / denominator if denominator else 0.0
    return {"predictions": predictions, "correct": correct,
            "unmatched_predictions": predictions - correct,
            "missed_gt": denominator - correct,
            "precision_proxy": precision, "recall_proxy": recall,
            "hmean_proxy": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def run(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    selector = (args.selector_dir or discover_selector(args.paths)).resolve()
    selector_plan_path, selector_result_path = selector / "resolved_configuration.json", selector / "results.json"
    selector_plan = json.loads(selector_plan_path.read_text())
    selector_result = json.loads(selector_result_path.read_text())
    if (selector_plan["reproduction_mode"] != "approximate"
        or selector_plan["table_5_claim"] is not False
        or selector_result["result_class"] != "APPROXIMATE_TEST_DIAGNOSTIC"
        or selector_result["images"] != len(selector_plan["sample_ids"])
        or selector_plan["cached_only"]):
        raise ValueError("expected a complete approximate TEST selector artifact")
    source = Path(selector_plan["source"])
    source_plan = json.loads((source / "resolved_configuration.json").read_text())
    if sha(source / "resolved_configuration.json") != selector_plan["source_plan_sha256"]:
        raise RuntimeError("selector source plan changed")
    manifest_path = Path(source_plan["paths_profile"]["test_manifest"])
    if sha(manifest_path) != selector_plan["manifest_sha256"]:
        raise RuntimeError("selector image manifest changed")
    samples = {sample.sample_id: sample for sample in
               load_segmentation_manifest(manifest_path, require_checksums=True).samples}
    ids = selector_plan["sample_ids"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        ids = ids[:args.limit]
    if any(sid not in samples for sid in ids):
        raise RuntimeError("selector sample absent from manifest")
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "claim_label": cfg["claim_label"], "table_5_claim": False,
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TEST_DIAGNOSTIC",
            "selector_dir": str(selector), "selector_plan_sha256": sha(selector_plan_path),
            "selector_result_sha256": sha(selector_result_path), "source": str(source),
            "source_plan_sha256": sha(source / "resolved_configuration.json"),
            "manifest_sha256": sha(manifest_path), "config": cfg,
            "config_sha256": sha(args.config), "sample_ids": ids,
            "code_sha256": {name: sha(REPO / name) for name in CODE},
            "execution": {"device": "cpu", "model_set": [], "precision": "float64 mask integrals",
                          "new_ocr_crops": 0, "gpu_peak_allocated_bytes": None,
                          "gpu_peak_reserved_bytes": None, "python": sys.version},
            "ground_truth_use": "only after every postfiltered selection shard is frozen"}
    identity = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() if args.output else selector.parent / f"postfilters-{identity}"
    if output == REPO or REPO in output.parents:
        raise ValueError("experiment artifacts must be outside the repository")
    freeze(output / "resolved_configuration.json", plan)
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        input_path, selection_path = selector / "inputs" / f"{key}.json", selector / "selection" / f"{key}.json"
        context, selected = json.loads(input_path.read_text()), json.loads(selection_path.read_text())
        if context["sample_id"] != sid or selected["sample_id"] != sid or selected["input_sha256"] != sha(input_path):
            raise RuntimeError(f"selector shard contract changed: {sid}")
        base = selected["methods"][cfg["input_method"]]
        mass = load_mass(sid, selector_input=context, source=source,
                         source_plan=source_plan, sample=samples[sid])
        duplicate, duplicate_rejected = suppress_duplicate_regions(
            base, context["groups"], context["rows"], mass,
            box_iou_min=cfg["duplicate_suppression"]["box_iou_min"],
            soft_iou_min=cfg["duplicate_suppression"]["soft_mask_iou_min"])
        accepted, abstained = abstain_uncorroborated(base, context["groups"], context["rows"])
        combined, combined_rejected = suppress_duplicate_regions(
            accepted, context["groups"], context["rows"], mass,
            box_iou_min=cfg["duplicate_suppression"]["box_iou_min"],
            soft_iou_min=cfg["duplicate_suppression"]["soft_mask_iou_min"])
        freeze(output / "selection" / f"{key}.json", {
            "sample_id": sid, "selector_input_sha256": sha(input_path),
            "selector_selection_sha256": sha(selection_path),
            "source_prediction_count": len(base),
            "methods": {"duplicate_suppression": duplicate,
                        "evidence_abstention": accepted,
                        "abstention_then_suppression": combined},
            "rejections": {"duplicate_suppression": duplicate_rejected,
                           "evidence_abstention": abstained,
                           "abstention_then_suppression": abstained + combined_rejected},
            "ground_truth_use": "none"})
        print(f"[postfilter] {position}/{len(ids)} {sid}", flush=True)
    totals = {name: Counter() for name in ("source", *METHODS)}
    denominator = 0
    for sid in ids:
        key = sid.replace(":", "__")
        selected_path = selector / "selection" / f"{key}.json"
        source_eval_path = source / "evaluation" / f"{key}.json"
        old_eval = json.loads((selector / "evaluation" / f"{key}.json").read_text())
        if (old_eval["selection_sha256"] != sha(selected_path)
            or old_eval["source_evaluation_sha256"] != sha(source_eval_path)):
            raise RuntimeError(f"selector evaluation contract changed: {sid}")
        gt = json.loads(source_eval_path.read_text())["ground_truth"]
        source_predictions = json.loads(selected_path.read_text())["methods"][cfg["input_method"]]
        filtered_path = output / "selection" / f"{key}.json"
        filtered = json.loads(filtered_path.read_text())["methods"]
        results = {"source": evaluate_predictions(source_predictions, gt, cfg["evaluation"]["matching_iou"])}
        results.update({name: evaluate_predictions(predictions, gt, cfg["evaluation"]["matching_iou"])
                        for name, predictions in filtered.items()})
        for name, result in results.items():
            totals[name].update({field: result[field] for field in
                                 ("predictions", "correct", "unmatched_predictions")})
        denominator += len(gt)
        freeze(output / "evaluation" / f"{key}.json", {
            "sample_id": sid, "selection_sha256": sha(filtered_path),
            "source_evaluation_sha256": sha(source_eval_path),
            "gt_words": len(gt), "methods": results})
    if not args.limit:
        reference = selector_result["summary"][cfg["input_method"]]
        if (totals["source"]["predictions"] != reference["predictions"]
            or totals["source"]["correct"] != reference["correct"]
            or denominator != selector_result["gt_words"]):
            raise RuntimeError("postfilter source evaluation does not reproduce frozen selector")
    summary = {name: _rates(result, denominator) for name, result in totals.items()}
    report = {"schema_version": 1, "reproduction_mode": "approximate",
              "claim_label": cfg["claim_label"], "result_class": plan["result_class"],
              "table_5_claim": False, "images": len(ids), "gt_words": denominator,
              "summary": summary, "plan_sha256": sha(output / "resolved_configuration.json"),
              "per_image": [f"evaluation/{sid.replace(':', '__')}.json" for sid in ids],
              "ground_truth_use": plan["ground_truth_use"]}
    freeze(output / "results.json", report)
    lines = ["# Crop-group output postfilters", "", cfg["claim_label"], "",
             f"Images: {len(ids)}; GT words: {denominator}; no new model inference.", "",
             "| Method | Correct | Predictions | Unmatched | Precision proxy | Recall proxy | Hmean proxy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in summary.items():
        lines.append(f"| {name} | {result['correct']} | {result['predictions']} | "
                     f"{result['unmatched_predictions']} | {result['precision_proxy']:.4f} | "
                     f"{result['recall_proxy']:.4f} | {result['hmean_proxy']:.4f} |")
    lines.extend(["", "Local axis-aligned-box metric; not official Total-Text polygon Hmean."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output / "results.json"), "summary": summary}), flush=True)
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
