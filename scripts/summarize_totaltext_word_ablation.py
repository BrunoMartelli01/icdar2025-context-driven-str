"""Paired TRAIN-holdout comparisons; never reads TEST or selects thresholds."""

import argparse
import json
from pathlib import Path

from models.totaltext_word_ablation import BRANCHES
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json


def paired_changes(control, variant, field):
    before = {r["sample_id"]: r for r in control}
    after = {r["sample_id"]: r for r in variant}
    if len(before) != 125 or set(before) != set(after):
        raise ValueError("Validation image IDs differ")
    gained = lost = 0
    for sample_id, row in before.items():
        if row["ground_truth"] != after[sample_id]["ground_truth"]:
            raise ValueError("Per-image GT differs")
        a, b = set(row["matched_gt_indices"][field]), set(after[sample_id]["matched_gt_indices"][field])
        gained += len(b - a); lost += len(a - b)
    return {"gained": gained, "lost": lost, "net": gained - lost}


def run(output):
    reports, hashes = {}, {}
    for branch in BRANCHES:
        path = output / (branch + "-validation-parseq") / "results.json"
        reports[branch] = json.loads(path.read_text())
        hashes[branch] = sha256_file(path)
    control = reports["control"]
    common = ("config_sha256", "validation_ids_sha256", "baseline_report_sha256", "parseq_checkpoint_sha256",
              "parseq_runtime_archive_sha256", "segmentation", "matching", "crop", "device", "parseq_batch_size")
    changes = {}
    for branch, result in reports.items():
        if result["aggregate"]["gt_words"] != 983 or result["protocol"]["branch"] != branch:
            raise ValueError("Summary denominator/branch differs")
        if any(result["protocol"][field] != control["protocol"][field] for field in common):
            raise ValueError("Branches have incompatible OCR validation protocols")
        changes[branch] = {field: paired_changes(control["per_image"], result["per_image"], field)
            for field in ("geometry_iou_ge_50", "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}
    result = {"classification": "PRACTICAL_APPROXIMATION", "split": "TRAIN_heldout_125",
              "images": 125, "gt_words": 983, "branches": {b: r["aggregate"] for b, r in reports.items()},
              "changes_vs_matched_control": changes, "report_sha256": hashes,
              "prior_source_control_reference": {"exact_casefold": 446, "exact_no_symbols": 478, "geometry_iou_50": 567},
              "selection_note": "comparison only; no automatic promotion or TEST evaluation"}
    destination = output / "comparison.json"
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise ValueError("Existing comparison differs; refusing overwrite")
    if not destination.exists():
        _json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"JSON {output / 'comparison.json'} SHA256 {sha256_file(output / 'comparison.json')}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args().output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
