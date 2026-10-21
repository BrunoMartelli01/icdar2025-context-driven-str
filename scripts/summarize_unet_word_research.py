"""Compare completed heads and per-word OCR gains/losses without promotion."""

import argparse
import json
from pathlib import Path

from models.unet_word_research import BRANCHES
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json


def summarize(root):
    rows, configs = {}, []
    for branch in BRANCHES:
        report = json.loads((root / branch / "run_manifest.json").read_text())
        validation = json.loads((root / branch / "validation.json").read_text())
        if (report.get("status") != "SUCCESS" or not report.get("training_complete")
                or report.get("completed_epochs") != report["run_contract"]["resolved_config"]["epochs"]
                or validation.get("run_contract") != report["run_contract"]
                or report["checkpoint"]["sha256"] != sha256_file(root / branch / "best.pt")
                or validation["checkpoint_sha256"] != report["checkpoint"]["sha256"]):
            raise ValueError("Only completed compatible experiments may enter comparison: " + branch)
        configs.append(report["run_contract"]["resolved_config"])
        rows[branch] = {"epoch": report["checkpoint"]["epoch"], "word_geometry": validation["aggregate"],
            "glyph_word_head_only": validation["glyph_word_head_only"], "budget_comparison": validation["budget_comparison"],
            "checkpoint_sha256": report["checkpoint"]["sha256"], "validation_sha256": sha256_file(root / branch / "validation.json")}
    if any(config != configs[0] for config in configs):
        raise ValueError("Branches did not run the same resolved configuration")
    ocr = json.loads((root / "ocr_comparison.json").read_text())
    if ocr.get("status") != "SUCCESS" or ocr["protocol"]["result_class"] != "approximate reproduction":
        raise ValueError("Completed qualified OCR comparison is required")
    for branch, row in rows.items():
        if ocr["protocol"]["validation_report_sha256"][branch] != row["validation_sha256"]:
            raise ValueError("OCR source exports differ from segmentation comparison")
    result = {"reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "selection": "comparison only; no automatic promotion or TEST evaluation", "branches": rows,
        "ocr": {arm: {k: v for k, v in data.items() if k != "per_image"} for arm, data in ocr["arms"].items()},
        "ocr_comparison_sha256": sha256_file(root / "ocr_comparison.json"), "resolved_config": configs[0]}
    _json(root / "comparison.json", result)
    lines = ["# U-Net word experiments — approximate reproduction", "", "Held-out TRAIN, single seed, exploratory reused holdout. No automatic promotion.", "",
             "| Branch | Epoch | Native word F1 | Merge | Split | Exact OCR / all983 | OCR gained/lost vs control |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for branch, row in rows.items():
        geometry = row["word_geometry"]; text = result["ocr"][branch]
        change = text["exact_casefold_changes_vs_control"]
        lines.append(f"| {branch} | {row['epoch']} | {geometry['proposal_f1_iou_50']:.2%} | {geometry['merge_rate']:.2%} | {geometry['split_rate']:.2%} | {text['aggregate']['exact_casefold_iou_ge_50']} | +{change['gained']} / -{change['lost']} |")
    lines += ["", "Quality branch freezes the segmentation. Its unconstrained geometry/OCR should equal the source;",
              "the separate area10/quality_rank10 comparison measures budget ranking.", "", "| Ranking | Exact OCR / all983 |",
              "| --- | ---: |"]
    for arm in ("quality_area10", "quality_rank10"):
        lines.append(f"| {arm} | {result['ocr'][arm]['aggregate']['exact_casefold_iou_ge_50']} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print("RESULT " + str(root / "comparison.md"), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    summarize(parser.parse_args().run_root.resolve())
