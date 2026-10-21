#!/usr/bin/env python3
"""CPU-only grouping battery reusing frozen Total-Text crop/OCR shards."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from localize.crop_word_groups import ALLOWED_METHODS, geometry, group_crops
from recognition.crop_word_group_metric import grouped_oracle
from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.rescore_totaltext_highres_padding_symbols import normalize_word


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_crop_word_groups_train_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/totaltext_crop_recovery_local_paths_v1.json"
CODE = ("localize/crop_word_groups.py", "recognition/crop_word_group_metric.py",
        "scripts/analyze_totaltext_crop_word_groups.py",
        "recognition/scene_text_v1_evaluation.py",
        "scripts/rescore_totaltext_highres_padding_symbols.py")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"existing artifact differs: {path}; choose a new --output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def format_optional(value: float | None, precision: int) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}"


def validate_config(config: dict) -> None:
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("dataset") != "total_text" or config.get("split") not in ("TRAIN_heldout_125", "official_TEST_300")
        or config.get("table_5_claim") is not False or config.get("final_variations") is not False
        or set(config.get("source_methods", [])) != ALLOWED_METHODS
        or config.get("evaluation", {}).get("matching_iou") != .5
        or config.get("evaluation", {}).get("group_capacity") != 1
        or config.get("evaluation", {}).get("crop_capacity") != 1):
        raise ValueError("expected isolated approximate threshold/window grouping config")
    if not config["strategies"] or len(set(config["strategies"])) != len(config["strategies"]):
        raise ValueError("strategies must be nonempty and unique")
    if not config["iou_min_values"] or len(set(config["iou_min_values"])) != len(config["iou_min_values"]):
        raise ValueError("IoU values must be nonempty and unique")
    for strategy in config["strategies"]:
        for cutoff in config["iou_min_values"]:
            group_crops([], strategy=strategy, iou_min=cutoff,
                        center_limit=config["center_limit"], ambiguity=config["ambiguity"])


def discover_source(paths_path: Path) -> Path:
    paths = json.loads(paths_path.read_text())
    if paths.get("path_base") != "repository_root":
        raise ValueError("expected repository_root local path profile")
    root = Path(paths["output_root"])
    root = root if root.is_absolute() else REPO / root
    valid = []
    for result_path in root.glob("*/results.json"):
        result = json.loads(result_path.read_text())
        if (result.get("images") == 125 and result.get("reproduction_mode") == "approximate"
            and result.get("summary", {}).get("baseline", {}).get("gt_words") == 983
            and not (result_path.parent / "failure.json").exists()):
            valid.append(result_path)
    if not valid:
        raise FileNotFoundError("no completed TRAIN-125 crop suite found; pass --source RUN_DIRECTORY")
    return max(valid, key=lambda path: (path.stat().st_mtime_ns, str(path))).parent.resolve()


def audit_geometry(rows: list[dict], grouping: dict, gt: list[dict], cutoff: float) -> dict:
    # Diagnostics only: geometry determines an audit label after groups are frozen.
    boxes = np.array([row["bbox"] for row in rows] + [word["bbox"] for word in gt])
    overlap = geometry(boxes)[0][:len(rows), len(rows):]
    labels = {}
    ties = 0
    for index, row in enumerate(rows):
        eligible = np.flatnonzero(overlap[index] >= cutoff)
        if not len(eligible):
            continue
        best = overlap[index, eligible].max()
        winners = eligible[np.isclose(overlap[index, eligible], best, rtol=0, atol=1e-12)]
        if len(winners) != 1:
            ties += 1
            continue
        labels[row["candidate_id"]] = int(winners[0])
    pair_total = pair_same = mixed = labelled_groups = 0
    word_groups = {index: set() for index in range(len(gt))}
    for group in grouping["groups"]:
        # Measure core coherence separately from deliberately shared parents.
        counts = Counter(labels[cid] for cid in group["core_candidate_ids"] if cid in labels)
        pair_total += sum(counts.values()) * (sum(counts.values()) - 1) // 2
        pair_same += sum(count * (count - 1) // 2 for count in counts.values())
        mixed += len(counts) > 1
        labelled_groups += bool(counts)
        for gi in counts:
            word_groups[gi].add(group["group_id"])
    return {"labelled_candidates": len(labels), "label_ties": ties,
            "labelled_core_groups": labelled_groups, "mixed_core_groups": mixed,
            "labelled_pairs": pair_total, "same_word_pairs": pair_same,
            "geometry_word_group_counts": {str(gi): len(groups)
                                           for gi, groups in word_groups.items() if groups}}


def run(args) -> Path:
    config_path = args.config.resolve()
    if config_path.parent != REPO / "configs/approximate":
        raise ValueError("grouping config must belong to configs/approximate")
    config = json.loads(config_path.read_text())
    validate_config(config)
    source = args.source.resolve() if args.source else discover_source(args.paths)
    source_plan_path = source / "resolved_configuration.json"
    source_plan = json.loads(source_plan_path.read_text())
    if (source_plan.get("reproduction_mode") != "approximate"
        or source_plan["config"].get("dataset") != "total_text"
        or source_plan["config"].get("split") != config["split"]
        or source_plan["config"].get("matching_iou") != config["evaluation"]["matching_iou"]
        or source_plan["config"].get("normalization") != config["evaluation"]["normalization"]):
        raise ValueError("source must match the frozen approximate Total-Text split/protocol")
    for path in ("recognition/scene_text_v1_evaluation.py",
                 "scripts/rescore_totaltext_highres_padding_symbols.py"):
        if source_plan["code_sha256"].get(path) != digest(REPO / path):
            raise RuntimeError(f"source metric implementation differs: {path}")
    source_result = json.loads((source / "results.json").read_text())
    if source_result["plan_sha256"] != digest(source_plan_path):
        raise RuntimeError("source result plan hash differs")
    sample_ids = source_plan["sample_ids"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        sample_ids = sample_ids[:args.limit]
    full_count = 125 if config["split"] == "TRAIN_heldout_125" else 300
    split_label = "TRAIN" if config["split"] == "TRAIN_heldout_125" else "TEST"
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "claim_label": f"approximate reproduction; {split_label} grouping diagnostic",
            "result_class": "PARTIAL_DATASET" if len(sample_ids) != full_count else "APPROXIMATE_REPRODUCTION",
            "table_5_claim": False, "config": config, "config_sha256": digest(config_path),
            "source": str(source), "source_plan_sha256": digest(source_plan_path),
            "source_results_sha256": digest(source / "results.json"), "sample_ids": sample_ids,
            "code_sha256": {path: digest(REPO / path) for path in CODE},
            "execution": {"device": "cpu", "new_model_inference": False,
                          "precision": "float64 geometry; upstream OCR unchanged",
                          "randomness": "none; candidate-ID tie breaking", "python": sys.version,
                          "numpy": np.__version__, "platform": platform.platform()}}
    identity = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() if args.output else source / "word_groups" / identity
    freeze(output / "resolved_configuration.json", plan)
    totals = {}
    per_image = []
    reference_total = Counter()
    for position, sid in enumerate(sample_ids, 1):
        key = sid.replace(":", "__")
        prepared_path = source / "prepared" / f"{key}.json"
        prepared = json.loads(prepared_path.read_text())
        rows = [row for row in prepared["candidates"] if set(row["methods"]) & ALLOWED_METHODS]
        # Inference boundary: copy only geometry and generator provenance.
        inputs = [{field: row[field] for field in ("candidate_id", "bbox", "origins", "methods",
                                                   "quad", "neutral_fraction")} for row in rows]
        variants = {}
        for strategy in config["strategies"]:
            for cutoff in config["iou_min_values"]:
                label = f"{strategy}_iou{cutoff:g}"
                variants[label] = group_crops(inputs, strategy=strategy, iou_min=cutoff,
                                             center_limit=config["center_limit"], ambiguity=config["ambiguity"])
        grouping_path = output / "groups" / f"{key}.json"
        freeze(grouping_path, {"sample_id": sid, "prepared_sha256": digest(prepared_path),
                               "ground_truth_use": "none", "variants": variants})
        # GT and frozen OCR witness edges enter ONLY after grouping has been saved.
        evaluation_path = source / "evaluation" / f"{key}.json"
        evaluation = json.loads(evaluation_path.read_text())
        if (evaluation["sample_id"] != sid or evaluation["prepared_sha256"] != digest(prepared_path)
            or evaluation["ocr_sha256"] != digest(source / "ocr" / f"{key}.json")):
            raise RuntimeError(f"source prepared/OCR/evaluation contract differs for {sid}")
        gt = evaluation["ground_truth"]
        witnesses = {int(gi): ids for gi, ids in evaluation["arms"]["all_methods"]["witnesses"].items()}
        indices = {row["candidate_id"]: index for index, row in enumerate(rows)}
        edges = [(indices[cid], gi) for gi, ids in witnesses.items() for cid in ids if cid in indices]
        unconstrained = maximum_matching(len(rows), len(gt), edges)
        reference_total.update(candidates=len(rows), correct=len(unconstrained), gt_words=len(gt),
                               gt_nonempty=sum(bool(normalize_word(word["text"])) for word in gt))
        scored = {}
        for label, grouping in variants.items():
            matched = grouped_oracle(grouping["groups"], witnesses, len(gt))
            if len(matched) > len(unconstrained):
                raise RuntimeError("group oracle exceeds distinct-crop oracle")
            audit = audit_geometry(rows, grouping, gt, config["evaluation"]["matching_iou"])
            witness_groups = {str(gi): sum(bool(set(group["candidate_ids"]) & set(ids))
                                          for group in grouping["groups"])
                              for gi, ids in witnesses.items() if set(ids) & set(indices)}
            scored[label] = {"groups": len(grouping["groups"]),
                "parent_only_groups": sum(group["parent_only"] for group in grouping["groups"]),
                "memberships": sum(len(group["candidate_ids"]) for group in grouping["groups"]),
                "shared_parent_crops": len(grouping["ambiguous_parents"]),
                "correct": len(matched), "correct_gt_indices": sorted(matched),
                "correct_witness_group_counts": witness_groups, **audit}
            total = totals.setdefault(label, Counter())
            total.update({field: value for field, value in scored[label].items() if isinstance(value, int)})
            total["correct_words_in_multiple_groups"] += sum(count > 1 for count in witness_groups.values())
            total["total_correct_word_groups"] += sum(witness_groups.values())
        image_path = output / "evaluation" / f"{key}.json"
        freeze(image_path, {"sample_id": sid, "source_evaluation_sha256": digest(evaluation_path),
                            "groups_sha256": digest(grouping_path), "gt_words": len(gt),
                            "unconstrained_correct_gt_indices": sorted(unconstrained), "variants": scored})
        per_image.append(str(image_path.relative_to(output)))
        print(f"[groups] {position}/{len(sample_ids)} {sid} crops={len(rows)}", flush=True)
    for total in totals.values():
        total["pair_precision_core"] = (total["same_word_pairs"] / total["labelled_pairs"]
                                        if total["labelled_pairs"] else None)
        total["oracle_lost_count"] = reference_total["correct"] - total["correct"]
        total["mean_groups_per_correct_word"] = (total["total_correct_word_groups"] / reference_total["correct"]
                                                  if reference_total["correct"] else None)
    expected_words = 983 if config["split"] == "TRAIN_heldout_125" else 2204
    if len(sample_ids) == full_count and reference_total["gt_words"] != expected_words:
        raise RuntimeError(f"full frozen {split_label} denominator differs from {expected_words}")
    result = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": plan["claim_label"],
              "result_class": plan["result_class"], "table_5_claim": False,
              "plan_sha256": digest(output / "resolved_configuration.json"), "images": len(sample_ids),
              "reference": dict(reference_total), "summary": {k: dict(v) for k, v in totals.items()},
              "per_image": per_image, "metric": "one group and one original crop per normalized exact GT word; frozen IoU>=0.5 witnesses",
              "limitation": f"{split_label} diagnostic; geometry grouping is a hypothesis, not proof of word identity; OCR selection deferred"}
    freeze(output / "results.json", result)
    lines = ["# Threshold/window crop grouping", "", plan["claim_label"], "",
             f"{len(sample_ids)} images; {reference_total['candidates']} crops; unrestricted oracle {reference_total['correct']}/{reference_total['gt_words']}.", "",
             "Shared parents retain one ID and one OCR output. Oracle capacities: group=1, original crop=1, GT=1.", "",
             "| Strategy | Groups | Shared parent crops | Oracle correct | Lost | Core pair precision | Mean groups per recoverable word |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, total in totals.items():
        precision = total['pair_precision_core']
        lines.append(f"| {label} | {total['groups']} | {total['shared_parent_crops']} | {total['correct']} | {total['oracle_lost_count']} | {format_optional(precision, 4)} | {format_optional(total['mean_groups_per_correct_word'], 2)} |")
    lines += ["", "Pair precision audits only uniquely geometry-labelled core members; shared parents are intentionally excluded.",
              f"These comparisons use the same {split_label} data and do not by themselves establish selector accuracy."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output / "results.json"), "reference": dict(reference_total)}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="completed crop recovery suite directory; default discovers latest full run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        # Preserve failure details if the user supplied an isolated destination.
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "failure.json").write_text(json.dumps({"error": str(error),
                "error_type": type(error).__name__, "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
