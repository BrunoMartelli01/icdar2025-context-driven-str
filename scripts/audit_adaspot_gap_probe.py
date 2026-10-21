"""CPU count/cache/selection audit for the completed frozen-feature Gap probe."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_adaspot_phase0 import count, digest


def audit(root):
    result = json.loads((root / "results.json").read_text())
    assert result["status"] == "SUCCESS"
    protocol = result["protocol"]; config = protocol["resolved_config"]
    assert result["completed_epochs"] == config["epochs"]
    assert [row["epoch"] for row in result["history"]] == list(range(config["epochs"] + 1))
    assert all(row["cuda_steps"] > 0 for row in result["history"][1:])
    selected = max(result["history"], key=lambda r: (r["validation"]["aggregate"]["word_f1"],
        -r["validation"]["aggregate"]["gt_words_touched_by_fused_components"], -r["epoch"]))
    assert selected["epoch"] == result["best_epoch"]
    for path, h in protocol["source_hashes"].items(): assert digest(Path(path)) == h, path
    assert digest(root / "best.pt") == result["checkpoint_sha256"]
    features = json.loads((root / "feature_manifest.json").read_text())
    assert len(features["train"]) == protocol["train_images"] == 1130
    assert len(features["validation"]) == protocol["val_images"] == 125
    assert not ({r["sample_id"] for r in features["train"]} & {r["sample_id"] for r in features["validation"]})
    assert digest(root / "training_pixels.pt") == features["train_cache_sha256"]
    cuda = json.loads((root / "first_cuda_step.json").read_text())
    assert cuda["device"].startswith("cuda") and cuda["weight_change_linf"] > 0 and cuda["trainable_parameters"] == 65
    source = Path(protocol["source_run"])
    geometry = {row["sample_id"]:row for row in selected["validation"]["per_image"]}
    for row in features["validation"]:
        sid = row["sample_id"]
        assert digest(Path(row["feature_path"])) == row["feature_sha256"]
        with np.load(root / "predicted_maps" / (sid.replace(":", "__") + ".npz")) as predicted:
            labels = predicted["labels"]
        with np.load(source / "control/validation_exports" / sid.replace(":", "__") / "maps.npz") as data:
            valid, owners, ids = data["validity"], data["owners"], data["expected_owner_ids"]
        boxes = []
        for i in np.unique(labels[labels > 0]):
            if not np.any((labels == i) & valid): continue
            y, x = np.where(labels == i)
            boxes.append({"bbox":[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]})
        gt = []
        for i in ids:
            y, x = np.where(owners == i)
            if len(x): gt.append({"bbox":[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]})
        assert count(boxes, gt, False) == geometry[sid]["metrics"]["matched_gt_iou_ge_50"]
        assert len(boxes) == geometry[sid]["metrics"]["raw_proposals"]
    exact = localized = proposals = words = 0
    for row in result["ocr"]["per_image"]:
        pred, gt = row["predictions"], row["ground_truth"]
        e, g = count(pred, gt, True), count(pred, gt, False)
        assert e == row["counts"]["exact_casefold_iou_ge_50"]
        assert g == row["counts"]["geometry_iou_ge_50"]
        exact += e; localized += g; proposals += len(pred); words += len(gt)
    totals = result["ocr"]["aggregate"]
    assert words == totals["gt_words"] == 983
    assert exact == totals["exact_casefold_iou_ge_50"]
    assert localized == totals["geometry_iou_ge_50"] and proposals == totals["proposals"]
    assert exact - 524 == result["changes_vs_baseline"]["net"]
    report = {"status":"PASS", "method":"independent SciPy Hungarian native/RGB bbox count audit; complete epochs, CUDA update, split, cache and selection audit",
        "completed_epochs":config["epochs"], "best_epoch":result["best_epoch"], "native_images":125,"ocr_images":125,
        "exact":exact,"words":words,"result_sha256":digest(root/"results.json"),"auditor_sha256":digest(Path(__file__))}
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    path=root/'independent_audit.json'
    if path.exists(): assert path.read_text()==text
    else:path.write_text(text)
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    audit(p.parse_args().run_root)
