"""Read-only comparison of the two completed held-out TRAIN diagnostics."""

import argparse
import json
from pathlib import Path

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    reports, hashes = {}, {}
    for branch in ("frozen_control", "partial_encoder"):
        path = root / (branch + "-validation-parseq") / "results.json"
        report = json.loads(path.read_text())
        if (report["protocol"]["branch"] != branch or report["aggregate"]["gt_words"] != 983
            or len(report["per_image"]) != 125):
            raise ValueError("Incomplete or wrong-branch held-out report")
        reports[branch], hashes[branch] = report, sha256_file(path)
    a, b = reports["frozen_control"], reports["partial_encoder"]
    keys = ("validation_ids_sha256", "baseline_report_sha256", "parseq_checkpoint_sha256",
            "segmentation", "localization", "matching", "device", "parseq_batch_size")
    if any(a["protocol"][key] != b["protocol"][key] for key in keys):
        raise ValueError("Controlled comparison protocols differ")
    by_id = {row["sample_id"]: row for row in a["per_image"]}
    transitions = {}
    for metric in ("geometry_iou_ge_50", "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50"):
        gained = lost = 0
        for row in b["per_image"]:
            other = by_id[row["sample_id"]]
            if row["ground_truth"] != other["ground_truth"]:
                raise ValueError("GT changed between branches")
            old, new = set(other["matched_gt_indices"][metric]), set(row["matched_gt_indices"][metric])
            gained += len(new - old)
            lost += len(old - new)
        transitions[metric] = {"gained": gained, "lost": lost, "net": gained - lost}
    summary = {"classification": "PRACTICAL_APPROXIMATION", "split": "TRAIN_heldout_125",
               "images": 125, "gt_words": 983, "report_sha256": hashes,
               "frozen_letterbox_065": a["frozen_letterbox_065"],
               "branches": {branch: report["aggregate"] for branch, report in reports.items()},
               "partial_vs_control": transitions}
    destination = root / "comparison.json"
    if destination.exists() and json.loads(destination.read_text()) != summary:
        raise ValueError("Existing comparison differs; refusing overwrite")
    if not destination.exists():
        _json(destination, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"JSON {destination} SHA256 {sha256_file(destination)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
