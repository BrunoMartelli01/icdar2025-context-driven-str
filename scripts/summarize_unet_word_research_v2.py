"""Compare corrective experiments; preserve frozen and continued baselines."""

import argparse
import json
from pathlib import Path

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json


def summarize(root, subset):
    ocr_path = root / ("ocr_comparison.json" if subset == "all" else subset + "_ocr_comparison.json")
    ocr = json.loads(ocr_path.read_text())
    if ocr.get("status") != "SUCCESS" or ocr["protocol"].get("result_class") != "approximate reproduction":
        raise ValueError("A successful qualified approximate OCR report is required")
    branches = {}
    for name in ocr["protocol"]["branches"]:
        directory = root / name
        receipt = json.loads((directory / "run_manifest.json").read_text())
        path = directory / "validation.json"; validation = json.loads(path.read_text())
        checkpoint = Path(receipt["checkpoint"]["path"])
        if not checkpoint.is_absolute():
            checkpoint = directory / checkpoint
        expected_epochs = 0 if name == "reference_control" or name.startswith("embedding50_") else receipt["run_contract"]["resolved_config"]["epochs"]
        if (receipt.get("status") != "SUCCESS" or not receipt.get("training_complete")
                or receipt["completed_epochs"] != expected_epochs
                or sha256_file(checkpoint) != receipt["checkpoint"]["sha256"]
                or validation["checkpoint_sha256"] != receipt["checkpoint"]["sha256"]
                or validation["run_contract"] != receipt["run_contract"]
                or sha256_file(path) != ocr["protocol"]["validation_report_sha256"][name]):
            raise ValueError("Completed output provenance mismatch: " + name)
        branches[name] = {"epoch": receipt["checkpoint"]["epoch"],
            "execution_kind": receipt.get("execution_kind", "training"),
            "checkpoint_sha256": receipt["checkpoint"]["sha256"], "geometry": validation["aggregate"],
            "glyph_word_head_only": validation.get("glyph_word_head_only"),
            "validation_sha256": sha256_file(path), "memory": receipt["memory"]}
    result = {"reproduction_mode": "approximate", "result_class": "approximate reproduction", "subset": subset,
        "branches": branches, "ocr": {name: {k: v for k, v in row.items() if k != "per_image"}
            for name, row in ocr["arms"].items()}, "ocr_sha256": sha256_file(ocr_path),
        "interpretation": "Single-seed reused TRAIN holdout. Compare extraction on identical epoch50 maps; frozen head branches against reference_control; staged against both reference and continuation. No promotion."}
    lines = ["# Corrective U-Net experiments — approximate reproduction", "", result["interpretation"], "",
        "| Arm | Epoch | Word F1 | Merge | Split | Glyph pixel F1 | Exact OCR/all GT | Baseline | Gained/lost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for name, b in branches.items():
        m = b["geometry"]; text = result["ocr"][name]
        if name.startswith("embedding50_"):
            baseline = "embedding50_hard"; delta = text["exact_casefold_changes_vs_embedding50_hard"]
        elif name == "glyph_staged" and "control" in result["ocr"]:
            baseline = "control"; delta = text["exact_casefold_changes_vs_control"]
        else:
            baseline = "reference_control"; delta = text["exact_casefold_changes_vs_reference"]
        glyph = f"{m['glyph_pixel_f1']:.2%}" if "glyph_pixel_f1" in m else "—"
        lines.append(f"| {name} | {b['epoch']} | {m['proposal_f1_iou_50']:.2%} | {m['merge_rate']:.2%} | {m['split_rate']:.2%} | {glyph} | {text['aggregate']['exact_casefold_iou_ge_50']}/{m['gt_words']} | {baseline} | +{delta['gained']}/-{delta['lost']} |")
    lines += ["", "OCR baseline: " + ocr["protocol"]["comparison_baseline"], "",
              "Frozen heads protect the source word map; glyph association and thresholding can still change localized instances.",
              "DB core is a raster/core-expansion adaptation, not a full DBNet reproduction. All geometry uses native bbox IoU>=.50; OCR uses original-coordinate bbox IoU>=.50 and exact casefold."]
    basename = "comparison" if subset == "all" else subset + "_comparison"
    _json(root / (basename + ".json"), result)
    (root / (basename + ".md")).write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("postprocess", "training", "all"), required=True)
    args = parser.parse_args()
    summarize(args.run_root.resolve(), args.subset)
