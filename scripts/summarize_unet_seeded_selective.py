"""Report geometry, recognition and matched ablations without model promotion."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_seeded_selective import ARMS
from scripts.summarize_unet_crop_links import paired_changes
from scripts.train_totaltext_v4_highres_ablation import _json


def summarize(root):
    ocr_path = root / "ocr_comparison.json"; ocr = json.loads(ocr_path.read_text())
    if ocr["status"] != "SUCCESS" or set(ocr["arms"]) != set(ARMS): raise ValueError("Complete six-arm PARSeq result required")
    result_class = ocr["protocol"]["result_class"]
    contract = json.loads((root / "run_contract.json").read_text())
    if sha256_file(root / "run_contract.json") != ocr["protocol"]["replay_contract_sha256"]:
        raise ValueError("Replay contract changed")
    branches = {}
    reference = {"crop_links_seeded_all": "crop_links_raw", "crop_word_selective": "crop_word_all",
                 "crop_links_seeded_selective": "crop_links_seeded_all"}
    lines = [f"# Seeded links and selective refinement — {result_class}", "",
             "Frozen checkpoints/logits; original RGB bbox IoU>=.5; exact casefold PARSeq; no new training or GT-informed selection.", "",
             "| Arm | Word F1 | Proposals | Merge words | Split words | Exact PARSeq | Gained/lost vs control |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name in ARMS:
        path = root / name / "validation.json"; report = json.loads(path.read_text())
        receipt = json.loads((root / name / "run_manifest.json").read_text())
        if (receipt["status"] != "SUCCESS" or receipt["run_contract"] != contract or report["run_contract"] != contract
                or sha256_file(path) != receipt["validation_sha256"]
                or sha256_file(path) != ocr["protocol"]["validation_report_sha256"][name]):
            raise ValueError("Summary provenance mismatch: " + name)
        arm = ocr["arms"][name]; geometry = report["aggregate"]
        if arm["aggregate"]["gt_words"] != geometry["gt_words"]: raise ValueError("Geometry/OCR denominator differs")
        paired = paired_changes(arm["per_image"], ocr["arms"]["control"]["per_image"])
        actions, gate_reasons, guard_reasons = Counter(), Counter(), Counter()
        for row in report["per_image"]:
            for d in row["audit"].get("decisions", []):
                actions[d["action"]] += 1
                gate_reasons.update(d.get("gate", {}).get("reasons", []))
                guard_reasons.update(d.get("guard", {}).get("reasons", []))
        branches[name] = {"geometry": geometry, "ocr": arm["aggregate"], "exact_changes_vs_control": paired,
                          "decisions": dict(actions), "gate_reasons": dict(gate_reasons), "guard_reasons": dict(guard_reasons),
                          "validation_sha256": sha256_file(path), "source_checkpoint": receipt["source_checkpoint"]}
        if name in reference:
            branches[name]["matched_ablation"] = {"reference": reference[name],
                **paired_changes(arm["per_image"], ocr["arms"][reference[name]]["per_image"])}
        lines.append(f"| {name} | {geometry['proposal_f1_iou_50']:.2%} | {geometry['raw_proposals']} | "
                     f"{geometry['gt_words_touched_by_fused_components']} | {geometry['gt_words_split_across_predicted_components']} | "
                     f"{arm['aggregate']['exact_casefold_iou_ge_50']}/{geometry['gt_words']} | +{paired['gained']}/-{paired['lost']} |")
    lines.extend(["", "Selective decisions:", ""])
    for arm in ("crop_word_selective", "crop_links_seeded_selective"):
        lines.append(f"- {arm}: {branches[arm]['decisions']}; guards: {branches[arm]['guard_reasons']}")
    lines.extend(["", "One seed and repeatedly reused TRAIN holdout; official TEST untouched. "
                  "Word and link checkpoints differ, so causal comparisons use the matched ablations. "
                  "No automatic promotion. Reported93% GT OCR is not remeasured."])
    result = {"reproduction_mode": "approximate", "result_class": result_class, "branches": branches,
              "ocr_sha256": sha256_file(ocr_path), "protocol": ocr["protocol"]}
    _json(root / "comparison.json", result)
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    summarize(p.parse_args().run_root.resolve())
