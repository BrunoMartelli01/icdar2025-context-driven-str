"""Audit four completed arms and compare exact PARSeq word identities."""

import argparse
import json
from pathlib import Path
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json


def paired_changes(rows, reference):
    old = {r["sample_id"]: r for r in reference}
    if set(old) != {r["sample_id"] for r in rows} or len(old) != len(rows):
        raise ValueError("Paired OCR population differs")
    gained, lost = [], []
    for row in rows:
        before = set(old[row["sample_id"]]["matched_gt_indices"]["exact_casefold_iou_ge_50"])
        after = set(row["matched_gt_indices"]["exact_casefold_iou_ge_50"])
        if row["ground_truth"] != old[row["sample_id"]]["ground_truth"]:
            raise ValueError("Paired OCR GT differs")
        for collection, indices in ((gained, after - before), (lost, before - after)):
            collection.extend({"sample_id": row["sample_id"], "gt_index": i,
                               "text": row["ground_truth"][i]["text"]} for i in sorted(indices))
    return {"gained": len(gained), "lost": len(lost), "net": len(gained) - len(lost),
            "gained_words": gained, "lost_words": lost}


def summarize(root):
    ocrpath = root / "ocr_comparison.json"; ocr = json.loads(ocrpath.read_text())
    if ocr["status"] != "SUCCESS" or ocr["protocol"]["result_class"] != "approximate reproduction":
        raise ValueError("Successful qualified PARSeq report required")
    branches = {}; lines = ["# Learned crop and pixel-link comparison — approximate reproduction", "",
        "Original RGB bbox IoU>=.5; exact casefold PARSeq; all983 care words in reused125-image TRAIN holdout.", "",
        "| Variant | Selected epoch | Word bbox F1 | Merge words | Split words | Exact PARSeq | Gained/lost vs source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, arm in ocr["arms"].items():
        directory = root / name
        receipt = json.loads((directory / "run_manifest.json").read_text())
        path = directory / "validation.json"; report = json.loads(path.read_text())
        checkpoint = Path(receipt["checkpoint"]["path"])
        if not checkpoint.is_absolute(): checkpoint = directory / checkpoint
        expected = receipt["run_contract"]["resolved_config"]["epochs"] if name.startswith("crop_") else 0
        if (receipt["status"] != "SUCCESS" or not receipt["training_complete"] or receipt["completed_epochs"] != expected
                or receipt["run_contract"] != report["run_contract"]
                or sha256_file(checkpoint) != receipt["checkpoint"]["sha256"]
                or report["checkpoint_sha256"] != receipt["checkpoint"]["sha256"]
                or sha256_file(path) != ocr["protocol"]["validation_report_sha256"][name]):
            raise ValueError("Completed result provenance mismatch: " + name)
        paired = paired_changes(arm["per_image"], ocr["arms"]["control"]["per_image"])
        geometry = report["aggregate"]
        branches[name] = {"epoch": receipt["checkpoint"]["epoch"], "geometry": geometry,
            "ocr": arm["aggregate"], "exact_changes_vs_source": paired, "memory": receipt["memory"],
            "checkpoint_sha256": receipt["checkpoint"]["sha256"], "validation_sha256": sha256_file(path)}
        if name == "crop_links":
            branches[name]["exact_changes_vs_crop_word"] = paired_changes(arm["per_image"], ocr["arms"]["crop_word"]["per_image"])
        lines.append(f"| {name} | {branches[name]['epoch']} | {geometry['proposal_f1_iou_50']:.2%} | "
            f"{geometry['gt_words_touched_by_fused_components']} | {geometry['gt_words_split_across_predicted_components']} | "
            f"{arm['aggregate']['exact_casefold_iou_ge_50']}/{geometry['gt_words']} | +{paired['gained']}/-{paired['lost']} |")
    result = {"reproduction_mode": "approximate", "result_class": "approximate reproduction", "branches": branches,
              "ocr_sha256": sha256_file(ocrpath), "protocol": ocr["protocol"],
              "limitations": "One seed, repeatedly reused TRAIN validation; final official TEST untouched. User-reported93% GT OCR not remeasured."}
    _json(root / "comparison.json", result)
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    summarize(p.parse_args().run_root.resolve())
