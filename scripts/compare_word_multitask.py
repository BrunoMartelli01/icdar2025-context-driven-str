"""Compare completed matched branches and reuse historical reports, without training."""

import argparse
import json
from pathlib import Path

from models.word_multitask import VARIANTS
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json
from segmentation.word_multitask_evaluation import aggregate_word_metrics


def compare(run_root, historical_reports):
    matched, contracts, histories = {}, [], []
    for variant in VARIANTS:
        path = Path(run_root) / variant / "run_manifest.json"
        if not path.exists():
            raise ValueError("Missing completed matched branch: " + str(path))
        report = json.loads(path.read_text())
        contract = report["run_contract"]
        if report["status"] != "SUCCESS" or contract["variant"] != variant:
            raise ValueError("Incomplete/inconsistent branch " + variant)
        if sha256_file(path.parent / report["checkpoint"]["path"]) != report["checkpoint"]["sha256"]:
            raise ValueError("Branch checkpoint changed: " + variant)
        contracts.append({k: v for k, v in contract.items() if k != "variant"})
        matched[variant] = {"metrics": report["validation_word_geometry"], "checkpoint": report["checkpoint"],
                            "report_sha256": sha256_file(path), "parameters": report["parameters"]}
        evaluation = path.parent / "validation.json"
        if evaluation.exists():
            ev = json.loads(evaluation.read_text())
            if ev["run_contract"] != contract or ev["checkpoint_sha256"] != report["checkpoint"]["sha256"]:
                raise ValueError("Validation report provenance differs")
            matched[variant]["postprocessing_ablation"] = ev["postprocessing_ablation"]
    if any(c != contracts[0] for c in contracts[1:]):
        raise ValueError("Branches differ in dataset/split/loss/training/encoder/code; cannot claim a controlled comparison")
    for path in historical_reports:
        path = Path(path)
        report = json.loads(path.read_text())
        if report.get("status") != "SUCCESS" or "validation_word_geometry" not in report:
            raise ValueError("Historical reference must be a SUCCESS training run manifest: " + str(path))
        metrics = aggregate_word_metrics(report["validation_word_geometry"])
        histories.append({"path": str(path.resolve()), "sha256": sha256_file(path), "metrics": metrics,
                          "checkpoint": report.get("checkpoint"), "comparison_role": "historical_reference_only",
                          "limitation": "Loss, thresholds, ignore policy or training budget may differ; no causal head comparison."})
    baseline = matched["text"]["metrics"]
    deltas = {variant: {key: row["metrics"][key] - baseline[key] for key in
                       ("proposal_f1_iou_50", "proposal_precision_iou_50", "word_recall_iou_50", "merge_rate", "split_rate")}
              for variant, row in matched.items() if variant != "text"}
    return {"result_class": "approximate reproduction", "reproduction_mode": "approximate",
            "interpretation": "Exploratory held-out TRAIN geometry; strict paper claims and TEST performance are unresolved.",
            "matched_branches": matched, "deltas_vs_text_BCE_Dice": deltas, "historical_references": histories}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--historical-report", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare(args.run_root, args.historical_report)
    if args.output.exists():
        if json.loads(args.output.read_text()) != report:
            raise ValueError("Existing comparison has different inputs; select a new filename")
        print("[reuse] " + str(args.output))
    else:
        _json(args.output, report)
    for name, value in report["matched_branches"].items():
        metrics = value["metrics"]
        print("{}: P={:.4f} R={:.4f} F1={:.4f} merge={:.4f} split={:.4f}".format(name,
              metrics["proposal_precision_iou_50"], metrics["word_recall_iou_50"],
              metrics["proposal_f1_iou_50"], metrics["merge_rate"], metrics["split_rate"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
