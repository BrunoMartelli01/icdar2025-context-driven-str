"""CPU TRAIN-only post-bilinear binary-mask threshold diagnostic, no models."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from diagnostics.gt_mask_localization_oracle import (
    checked_native, compare_success, original_proposals, save_json, sha, summarize, verify_file,
)
from localize.directional_component_neighbors import directional_neighbors
from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.repository_contours import RepositoryContourParams
from localize.v1_local_rescue import LocalRescueParams
from segmentation.data import load_segmentation_manifest


def interpolate_binary(mask, output_hw):
    if mask.shape != (128, 128) or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("binary uint8 native128 only, not probabilities")
    if len(output_hw) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in output_hw):
        raise ValueError("positive integer output H/W required")
    return F.interpolate(torch.from_numpy(mask).to(device="cpu", dtype=torch.float32)[None, None],
                         size=tuple(output_hw), mode="bilinear", align_corners=False,
                         antialias=False)[0, 0].numpy()


def threshold_binary(values, threshold, *, minimum=.5):
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("interpolated values must be finite and in [0,1]")
    if minimum not in (.3, .5) or not np.isfinite(threshold) or not minimum <= threshold <= 1:
        raise ValueError("threshold outside declared diagnostic range")
    return (values >= threshold).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "train-manifest", "data-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    config = json.loads(a.config.read_text())
    if (config["reproduction_mode"], config["split"], config["images"], config["gt_words"]) != ("approximate", "TRAIN", 229, 849):
        raise ValueError("TRAIN229/849 only")
    thresholds = config["thresholds"]
    grids = {"bilinear-threshold-train-v1": [.5, .55, .6, .65, .7],
             "bilinear-lower-threshold-train-v1": [.5, .45, .4, .35, .3]}
    if config["experiment"] not in grids or thresholds != grids[config["experiment"]] or config["resize"] != {
        "input": "cached_binary_native128", "mode": "bilinear", "align_corners": False,
        "antialias": False, "dtype": "float32", "device": "cpu"
    } or config["comparison"] != "greater_than_or_equal":
        raise ValueError("declared protocol drift")
    verify_file(a.reference, config["reference_sha256"])
    reference = json.loads(a.reference.read_text())
    nearest_path = Path(reference["protocol"]["nearest_report_path"])
    verify_file(nearest_path, reference["protocol"]["nearest_report_sha256"])
    nearest = json.loads(nearest_path.read_text())
    oracle_path = Path(nearest["protocol"]["source_oracle_path"])
    verify_file(oracle_path, nearest["protocol"]["oracle_report_sha256"])
    oracle = json.loads(oracle_path.read_text())
    source_path = Path(oracle["protocol"]["source_paths"]["source_report"])
    verify_file(source_path, oracle["protocol"]["source_report_sha256"])
    source = json.loads(source_path.read_text())
    for report in (reference, nearest, oracle):
        for path, expected in report["protocol"]["code_sha256"].items():
            verify_file(path, expected)
    verify_file(a.train_manifest, oracle["protocol"]["train_manifest_sha256"])
    manifest = load_segmentation_manifest(a.train_manifest, require_checksums=False)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("TRAIN229 required")
    params = reference["protocol"]["parameters"]
    v1 = GroupedWordBlockParams(**params["v1"])
    contours = RepositoryContourParams(**params["contours"])
    rescue = LocalRescueParams(**params["rescue"])
    words = defaultdict(list)
    for w in source["per_word"]:
        words[w["sample_id"]].append({"text": w["text"], "bbox": w["bbox"]})
    if sum(map(len, words.values())) != 849:
        raise ValueError("GT population drift")
    refs = {s["sample_id"]: s for s in reference["samples"]}
    mask_refs = {s["sample_id"]: s for s in reference["mask_comparison"] if s["condition"] == "PRED128"}
    oracle_refs = {s["sample_id"]: s for s in oracle["samples"]}
    methods = [m for m in reference["summary"] if m != "REPOSITORY_NATIVE128"]
    protocol = {**config, "config_sha256": sha(a.config), "reference": str(a.reference.resolve()),
        "parameters": params, "source_report": str(source_path),
        "source_report_sha256": sha(source_path), "checkpoint_sha256": oracle["protocol"]["checkpoint_sha256"],
        "train_manifest_sha256": sha(a.train_manifest), "models_loaded": False,
        "code_sha256": {p: sha(p) for p in [__file__, "diagnostics/gt_mask_localization_oracle.py", "localize/directional_component_neighbors.py"]},
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__}}
    a.output_dir.mkdir(parents=True)
    save_json(a.output_dir / "protocol.json", protocol)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    records, mask_rows = [], []
    for i, sample in enumerate(manifest.samples, 1):
        if i == 1 or i % 10 == 0 or i == 229:
            print(f"[{config['experiment']} CPU] {i}/229 sample={sample.sample_id}", flush=True)
        for rel in (sample.image_path, sample.mask_path, sample.validity_annotation_path):
            verify_file(a.data_root / rel, source["protocol"]["dataset_files"][str(rel)]["sha256"])
        with Image.open(a.data_root / sample.image_path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        if list(rgb.shape[:2]) != oracle_refs[sample.sample_id]["image_hw"]:
            raise RuntimeError("image dimensions changed")
        native, _ = checked_native(source_path.parent / "masks" / f"{sample.sample_id}.json",
            source["contract_sha256"], sample.sample_id,
            source["protocol"]["dataset_files"][str(sample.image_path)]["sha256"], protocol["checkpoint_sha256"])
        values = interpolate_binary(native, rgb.shape[:2])
        control = threshold_binary(values, .5)
        if hashlib.sha256(control.tobytes()).hexdigest() != mask_refs[sample.sample_id]["bilinear_binary_sha256"]:
            raise RuntimeError(".50 bilinear control mask mismatch")
        conditions = {"0.50": {m: refs[sample.sample_id]["conditions"]["PRED128"][m] for m in methods}}
        for t in thresholds:
            key = f"{t:.2f}"
            binary = control if t == .5 else threshold_binary(values, t, minimum=min(thresholds))
            if t != .5:
                results = original_proposals(binary, rgb, contours, v1, rescue)
                results.update(directional_neighbors(binary, v1)["methods"])
                conditions[key] = results
            mask_rows.append({"sample_id": sample.sample_id, "threshold": t,
                "mask_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
                "foreground_pixels": int(binary.sum()),
                "raw_components": conditions[key]["CURRENT_ALG1"]["raw_count"]})
        records.append({"sample_id": sample.sample_id, "conditions": conditions})
    summary, rows = summarize(records, words, methods, [f"{t:.2f}" for t in thresholds])
    for m in methods:
        if summary[m]["0.50"] != reference["summary"][m]["PRED128"]:
            raise RuntimeError("reused .50 baseline metrics mismatch")
    transitions = {m: {f"{t:.2f}": compare_success(
        [r["best_iou"] for r in rows if r["method"] == m and r["condition"] == "0.50"],
        [r["best_iou"] for r in rows if r["method"] == m and r["condition"] == f"{t:.2f}"])
        for t in thresholds} for m in methods}
    with (a.output_dir / "per-word.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {f"{t:.2f}": {k: sum(r[k] for r in mask_rows if r["threshold"] == t)
        for k in ("foreground_pixels", "raw_components")} for t in thresholds}
    report = {"protocol": protocol, "summary": summary, "transitions": transitions,
        "mask_totals": totals, "masks": mask_rows, "samples": records,
        "csv_sha256": sha(a.output_dir / "per-word.csv"),
        "validation": {"images": 229, "gt_words": 849, "cached_native_hashes_verified": True,
            "all_229_bilinear_0_50_control_masks_verified": True, "baseline_metrics_identical": True}}
    save_json(a.output_dir / "results.json", report)
    for t in thresholds:
        key = f"{t:.2f}"
        print(key, {m: summary[m][key]["geometry"]["gt_with_proposal_iou_ge_0_50"] for m in methods}, flush=True)
    print(f"JSON {a.output_dir / 'results.json'} SHA256 {sha(a.output_dir / 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
