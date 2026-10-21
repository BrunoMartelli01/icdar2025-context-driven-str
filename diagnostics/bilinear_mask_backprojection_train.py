"""Only change binary native-mask resizing; CPU TRAIN geometry, frozen methods."""
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

from diagnostics.directional_neighbors_train import orientation_summary
from diagnostics.gt_mask_localization_oracle import (
    checked_native, compare_success, original_proposals, save_json, sha, summarize, verify_file,
)
from localize.directional_component_neighbors import directional_neighbors
from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.repository_contours import RepositoryContourParams
from localize.v1_local_rescue import LocalRescueParams
from pipeline.frozen_segmentation import backproject_binary_mask
from segmentation.data import load_segmentation_manifest, icdar13_task22_rgb_to_target_and_validity
from segmentation.evaluate import resize_binary_target_to_native_logits


def bilinear_binary(mask, output_hw, *, threshold, align_corners, antialias):
    if mask.shape != (128, 128) or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("requires cached binary uint8 native128, not probabilities")
    if len(output_hw) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in output_hw):
        raise ValueError("requires positive original image H/W")
    if threshold != .5 or align_corners is not False or antialias is not False:
        raise ValueError("this diagnostic fixes >=.5, align_corners=False, antialias=False")
    tensor = torch.from_numpy(mask).to(dtype=torch.float32, device="cpu")[None, None]
    resized = F.interpolate(tensor, size=tuple(output_hw), mode="bilinear",
                            align_corners=align_corners, antialias=antialias)[0, 0]
    return (resized.numpy() >= threshold).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "nearest-report", "train-manifest", "data-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to replace {a.output_dir}")
    config = json.loads(a.config.read_text())
    if config["reproduction_mode"] != "approximate" or config["scope"] != {"split": "TRAIN", "images": 229, "gt_words": 849}:
        raise ValueError("APPROXIMATE TRAIN229/849 diagnostic only")
    policy = config["resize"]
    if policy != {"input": "cached_binary_uint8_native128_not_probabilities",
                  "implementation": "torch.nn.functional.interpolate", "dtype": "float32", "mode": "bilinear",
                  "align_corners": False, "antialias": False, "threshold": .5, "comparison": "greater_than_or_equal"}:
        raise ValueError("resize protocol drift")
    verify_file(a.nearest_report, config["nearest_report_sha256"])
    nearest = json.loads(a.nearest_report.read_text())
    oracle_path = Path(nearest["protocol"]["source_oracle_path"])
    verify_file(oracle_path, nearest["protocol"]["oracle_report_sha256"])
    oracle = json.loads(oracle_path.read_text())
    source_path = Path(oracle["protocol"]["source_paths"]["source_report"])
    verify_file(source_path, oracle["protocol"]["source_report_sha256"])
    source = json.loads(source_path.read_text())
    verify_file(a.train_manifest, oracle["protocol"]["train_manifest_sha256"])
    manifest = load_segmentation_manifest(a.train_manifest, require_checksums=False)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("official TRAIN229 only")
    for report in (nearest, oracle):
        for path, expected in report["protocol"]["code_sha256"].items():
            verify_file(path, expected)
    parameters = oracle["protocol"]["parameters"]
    v1_params = GroupedWordBlockParams(**parameters["v1"])
    contour_params = RepositoryContourParams(**parameters["contours"])
    rescue_params = LocalRescueParams(**parameters["rescue"])
    words = defaultdict(list)
    for word in source["per_word"]:
        words[word["sample_id"]].append({"text": word["text"], "bbox": word["bbox"]})
    nearest_samples = {s["sample_id"]: s for s in nearest["samples"]}
    oracle_samples = {s["sample_id"]: s for s in oracle["samples"]}
    protocol = {**config, "config_sha256": sha(a.config), "parameters": parameters,
                "nearest_report_path": str(a.nearest_report.resolve()), "device": "cpu", "models_loaded": False,
                "code_sha256": {path: sha(path) for path in [__file__,
                    "diagnostics/gt_mask_localization_oracle.py", "localize/directional_component_neighbors.py"]},
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
                "orientation_warning": "Reused PCA proxy only; prior visual audit rejected5 of7 putative diagonals and marked1 ambiguous. No official orientation labels."}
    a.output_dir.mkdir(parents=True)
    save_json(a.output_dir / "protocol.json", protocol)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    records, mask_comparison = [], []
    for i, sample in enumerate(manifest.samples, 1):
        if i == 1 or i % 10 == 0:
            print(f"[bilinear mask CPU] {i}/229 sample={sample.sample_id}", flush=True)
        old = nearest_samples[sample.sample_id]
        reference = oracle_samples[sample.sample_id]
        for rel in (sample.image_path, sample.mask_path, sample.validity_annotation_path):
            verify_file(a.data_root / rel, source["protocol"]["dataset_files"][str(rel)]["sha256"])
        with Image.open(a.data_root / sample.image_path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        with Image.open(a.data_root / sample.mask_path) as opened:
            target, _ = icdar13_task22_rgb_to_target_and_validity(
                opened, (a.data_root / sample.validity_annotation_path).read_text(encoding="utf-8-sig"))
        native_gt = resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
        pred, cache = checked_native(source_path.parent / "masks" / f"{sample.sample_id}.json",
            source["contract_sha256"], sample.sample_id,
            source["protocol"]["dataset_files"][str(sample.image_path)]["sha256"],
            oracle["protocol"]["checkpoint_sha256"])
        if list(rgb.shape[:2]) != reference["image_hw"]:
            raise RuntimeError("Original image dimensions changed")
        conditions = {"GT_FULLRES": old["conditions"]["GT_FULLRES"]}
        for condition, native, expected in (("PRED128", pred, reference["prediction_native_sha256"]),
                                            ("GT128", native_gt, reference["gt_native_sha256"])):
            if hashlib.sha256(native.tobytes()).hexdigest() != expected:
                raise RuntimeError("Native input differs from frozen experiment")
            nearest_mask = backproject_binary_mask(native, rgb.shape[:2])
            if condition == "PRED128" and hashlib.sha256(nearest_mask.tobytes()).hexdigest() != cache["segmentation_metadata"]["output_mask_sha256"]:
                raise RuntimeError("Nearest control differs from actual pipeline")
            binary = bilinear_binary(native, rgb.shape[:2], threshold=policy["threshold"],
                                     align_corners=policy["align_corners"], antialias=policy["antialias"])
            methods = original_proposals(binary, rgb, contour_params, v1_params, rescue_params)
            methods.update(directional_neighbors(binary, v1_params)["methods"])
            methods["REPOSITORY_NATIVE128"] = old["conditions"][condition]["REPOSITORY_NATIVE128"]
            conditions[condition] = methods
            mask_comparison.append({"sample_id": sample.sample_id, "condition": condition,
                "native_mask_sha256": expected, "bilinear_binary_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
                "nearest_binary_sha256": hashlib.sha256(nearest_mask.tobytes()).hexdigest(),
                "pixels_changed": int(np.count_nonzero(binary != nearest_mask)),
                "nearest_foreground_pixels": int(nearest_mask.sum()), "bilinear_foreground_pixels": int(binary.sum()),
                "nearest_raw_components": old["conditions"][condition]["CURRENT_ALG1"]["raw_count"],
                "bilinear_raw_components": methods["CURRENT_ALG1"]["raw_count"]})
        records.append({"sample_id": sample.sample_id, "conditions": conditions})
    methods = list(nearest["summary"])
    conditions = ["PRED128", "GT128", "GT_FULLRES"]
    summary, rows = summarize(records, words, methods, conditions)
    _, old_rows = summarize(nearest["samples"], words, methods, conditions)
    for method, values in summary.items():
        for condition, metrics in values.items():
            if condition == "GT_FULLRES" or method == "REPOSITORY_NATIVE128":
                if metrics != nearest["summary"][method][condition]:
                    raise RuntimeError("No-resize control changed")
    comparisons = {}
    for method, values in summary.items():
        comparisons[method] = {}
        for condition in values:
            new = [r["best_iou"] for r in rows if r["method"] == method and r["condition"] == condition]
            old = [r["best_iou"] for r in old_rows if r["method"] == method and r["condition"] == condition]
            comparisons[method][condition] = compare_success(old, new)
    for row in rows:
        label = nearest["orientations"][row["sample_id"]][row["gt_index"]]
        row.update(orientation=label["orientation"], orientation_angle_degrees=label["angle_degrees"])
    with (a.output_dir / "per-word.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mask_totals = {c: {k: sum(row[k] for row in mask_comparison if row["condition"] == c)
                       for k in ("pixels_changed", "nearest_foreground_pixels", "bilinear_foreground_pixels",
                                 "nearest_raw_components", "bilinear_raw_components")}
                   for c in ("PRED128", "GT128")}
    result = {"protocol": protocol, "summary": summary, "nearest_summary": nearest["summary"],
              "nearest_to_bilinear": comparisons, "mask_totals": mask_totals,
              "mask_comparison": mask_comparison, "by_orientation": orientation_summary(rows),
              "samples": records, "csv_sha256": sha(a.output_dir / "per-word.csv"),
              "validation": {"images": len(records), "gt_words": sum(map(len, words.values())),
                             "all_native_input_hashes_match": True, "unaffected_controls_unchanged": True}}
    if result["validation"]["images"] != 229 or result["validation"]["gt_words"] != 849:
        raise RuntimeError("Dataset population changed")
    save_json(a.output_dir / "results.json", result)
    for method in methods:
        print(method, "nearest", nearest["summary"][method]["PRED128"]["geometry"]["gt_with_proposal_iou_ge_0_50"],
              "bilinear", summary[method]["PRED128"]["geometry"]["gt_with_proposal_iou_ge_0_50"])
    print(f"JSON {a.output_dir / 'results.json'} SHA256 {sha(a.output_dir / 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
