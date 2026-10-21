"""GT-blind uncertainty ranking from saved TRAIN-holdout word probability maps."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from localize.unet_word_research import mask_stability
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_word_research import DEFAULT_CONFIG, read_config
from scripts.train_totaltext_v4_highres_ablation import _json
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.unet_word_research import mask_quality_targets
from segmentation.word_multitask_evaluation import aggregate_word_metrics


def binary_auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("AUC expects paired finite score/label vectors")
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def run(args):
    config, _ = read_config(args.config, args.mode)
    directory = args.kernel_run.resolve()
    report = json.loads((directory / "validation.json").read_text())
    training = json.loads((directory / "run_manifest.json").read_text())
    if (sha256_file(directory / "run_manifest.json") != config["source_run_manifest_sha256"]
            or training["checkpoint"]["sha256"] != config["source_checkpoint_sha256"]
            or report.get("checkpoint_sha256") != config["source_checkpoint_sha256"]
            or len(report["per_image"]) != config["validation_count"]):
        raise ValueError("Stability diagnostic requires the qualified Text + Kernel validation exports")
    rows, images, totals = [], [], {}
    for item in report["per_image"]:
        key = item["sample_id"].replace(":", "__")
        path = directory / "validation_exports" / key / "maps.npz"
        if sha256_file(path) != item["exports"]["maps.npz"]:
            raise ValueError("Saved probability map checksum changed: " + key)
        with np.load(path, allow_pickle=False) as saved:
            maps = {k: saved[k] for k in saved.files}
        labels, diagnostics = mask_stability(maps["probabilities"], maps["canvas"], config["postprocessing"], config["stability_thresholds"])
        if not np.array_equal(labels, maps["labels"]):
            raise ValueError("Declared baseline does not reproduce saved labels: " + key)
        care = labels.copy()
        for instance in np.unique(labels[labels > 0]):
            if not np.any((labels == instance) & maps["validity"]): care[care == instance] = 0
        metrics = instance_geometry(care, maps["owners"], tuple(maps["expected_owner_ids"]))
        for k, v in metrics.items(): totals[k] = totals.get(k, 0) + v
        # Targets are computed only after frozen localization and risk ranking.
        ids, target = mask_quality_targets(labels, maps["owners"], maps["validity"])
        by_id = dict(zip(ids, target))
        for row in diagnostics:
            if row["instance_id"] not in by_id:
                continue
            quality = by_id[row["instance_id"]]
            rows.append({"sample_id": item["sample_id"], **row,
                         "posthoc_bbox_iou_coverage_purity": quality.tolist(),
                         "posthoc_bad_bbox": bool(quality[0] < config["stability_bad_bbox_iou"]),
                         "mean_pixel_confidence": float(maps["probabilities"][0][labels == row["instance_id"]].mean())})
        images.append({"sample_id": item["sample_id"], "maps_sha256": sha256_file(path), **metrics})
        if len(images) % 25 == 0: print(f"[stability] {len(images)}/{len(report['per_image'])}", flush=True)
    rows.sort(key=lambda r: (-r["risk"], r["sample_id"], r["instance_id"]))
    count = max(1, math.ceil(config["stability_top_fraction"] * len(rows)))
    bad = [r["posthoc_bad_bbox"] for r in rows]
    confidence_rank = sorted(rows, key=lambda r: (r["mean_pixel_confidence"], r["sample_id"], r["instance_id"]))
    result = {"status": "SUCCESS", "reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "kind": "diagnostic ranking only; masks unchanged", "images": len(images), "proposals_scored": len(rows),
        "posthoc_bad_bbox_count": sum(bad), "risk_auc": binary_auc([r["risk"] for r in rows], bad),
        "low_confidence_auc": binary_auc([1 - r["mean_pixel_confidence"] for r in rows], bad),
        "bad_bbox_prevalence": float(np.mean(bad)) if bad else None,
        "top_risk_count": min(count, len(rows)),
        "top_risk_bad_bbox_precision": float(np.mean(bad[:count])) if bad else None,
        "top_low_confidence_bad_bbox_precision": float(np.mean([r["posthoc_bad_bbox"] for r in confidence_rank[:count]])) if bad else None,
        "baseline_geometry": aggregate_word_metrics(totals), "resolved_config": config,
        "metric_protocol": "Native bbox one-to-one IoU>=.50; ignore only wholly invalid predictions; all983 care words retained; risk AUC uses post-hoc per-proposal best bbox IoU, not one-to-one correctness",
        "source_validation_sha256": sha256_file(directory / "validation.json"), "per_image": images, "ranked_masks": rows}
    args.output_root.mkdir(parents=True, exist_ok=True)
    _json(args.output_root / "stability.json", result)
    print(json.dumps({k: result[k] for k in ("images", "proposals_scored", "risk_auc", "low_confidence_auc", "top_risk_bad_bbox_precision")}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--kernel-run", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    run(p.parse_args())
