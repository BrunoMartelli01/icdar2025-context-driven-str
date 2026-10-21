"""CPU-only TRAIN audit: word orientation and frozen directional neighbours."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from diagnostics.gt_mask_localization_oracle import (
    checked_native, compare_success, save_json, sha, summarize, verify_file,
)
from localize.directional_component_neighbors import directional_neighbors
from localize.practical_remap_protocol import _TRAIN_LINE
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.export_train_v1_parseq_inputs import _params
from segmentation.data import load_segmentation_manifest, icdar13_task22_rgb_to_target_and_validity
from segmentation.evaluate import resize_binary_target_to_native_logits


CATEGORIES = ("HORIZONTAL", "VERTICAL", "DIAGONAL", "UNKNOWN")


def estimate_orientation(points, *, minimum_glyphs, minimum_pca_eigenvalue_ratio, axis_tolerance_degrees):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    base = {"glyph_count": len(points), "orientation": "UNKNOWN", "angle_degrees": None,
            "pca_eigenvalue_ratio": None}
    if len(points) < minimum_glyphs:
        return {**base, "reason": "too_few_glyphs"}
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    if eigenvalues[-1] <= 1.e-12:
        return {**base, "reason": "degenerate_centroids"}
    ratio = float(eigenvalues[-1] / max(eigenvalues[0], 1.e-12))
    axis = eigenvectors[:, -1]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180)
    base.update(angle_degrees=angle, pca_eigenvalue_ratio=ratio)
    if ratio < minimum_pca_eigenvalue_ratio:
        return {**base, "reason": "no_dominant_axis"}
    horizontal = min(angle, 180 - angle)
    vertical = abs(angle - 90)
    category = ("HORIZONTAL" if horizontal <= axis_tolerance_degrees else
                "VERTICAL" if vertical <= axis_tolerance_degrees else "DIAGONAL")
    return {**base, "orientation": category, "reason": "source_gt_glyph_centroid_pca"}


def annotation_word_colors(text):
    groups, current = [], []
    for line in text.splitlines():
        if not line.strip():
            if current:
                groups.append(current)
                current = []
            continue
        if line.lstrip().startswith("#"):
            continue
        matched = _TRAIN_LINE.match(line)
        if matched is None:
            raise ValueError("Malformed Task2.2 glyph record")
        current.append(tuple(int(matched.group(i)) for i in (1, 2, 3)))
    if current:
        groups.append(current)
    return groups


def word_orientations(rgb_target, annotation, policy):
    """Only post-hoc GT annotation: these results never enter the localizer."""
    groups = annotation_word_colors(annotation)
    centers = {}
    for color in {c for group in groups for c in group}:
        yy, xx = np.where(np.all(rgb_target == np.asarray(color, dtype=np.uint8), axis=2))
        if not len(xx):
            raise ValueError(f"GT glyph color absent from mask: {color}")
        centers[color] = (float(xx.mean()), float(yy.mean()))
    return [estimate_orientation([centers[c] for c in group], **policy) for group in groups]


def orientation_summary(rows):
    result = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["condition"], row["orientation"])].append(row)
    for (method, condition, category), values in grouped.items():
        value = {"words": len(values)}
        for threshold, suffix in ((.25, "25"), (.5, "50"), (.75, "75")):
            value[f"iou_ge_0_{suffix}"] = sum(r["best_iou"] >= threshold for r in values)
        value["missed_at_0_50"] = len(values) - value["iou_ge_0_50"]
        value["coverage_at_0_50"] = value["iou_ge_0_50"] / len(values)
        value["top10_losses"] = sum(r["best_iou"] < .5 <= r["best_iou_before_cap"] for r in values)
        result.setdefault(method, {}).setdefault(condition, {})[category] = value
    return result


def directional_transitions(rows, conditions):
    indexed = {(r["sample_id"], r["gt_index"], r["condition"], r["method"]): r for r in rows}
    words = sorted({(r["sample_id"], r["gt_index"]) for r in rows})
    results = {}
    for condition in conditions:
        results[condition] = {}
        for before, after in (("PRACTICAL_REMAP_V1", "DIRECTIONAL_H"),
                              ("DIRECTIONAL_H", "DIRECTIONAL_HV"),
                              ("DIRECTIONAL_HV", "DIRECTIONAL_HVD"),
                              ("PRACTICAL_REMAP_V1", "DIRECTIONAL_HVD")):
            pair = f"{before}_to_{after}"
            results[condition][pair] = {}
            for category in ("ALL", *CATEGORIES):
                selected = [key for key in words if category == "ALL" or
                            indexed[(*key, condition, before)]["orientation"] == category]
                results[condition][pair][category] = compare_success(
                    [indexed[(*k, condition, before)]["best_iou"] for k in selected],
                    [indexed[(*k, condition, after)]["best_iou"] for k in selected])
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "oracle-report", "train-manifest", "data-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    config = json.loads(a.config.read_text())
    verify_file(a.oracle_report, config["oracle_report_sha256"])
    oracle = json.loads(a.oracle_report.read_text())
    verify_file(a.train_manifest, oracle["protocol"]["train_manifest_sha256"])
    manifest = load_segmentation_manifest(a.train_manifest, require_checksums=False)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("Official TRAIN229 required")
    if config["scope"] != oracle["protocol"]["scope"] or config["scope"]["gt_words"] != 849:
        raise RuntimeError("Scope mismatch")
    source_path = Path(oracle["protocol"]["source_paths"]["source_report"])
    verify_file(source_path, oracle["protocol"]["source_report_sha256"])
    source = json.loads(source_path.read_text())
    for path, expected in oracle["protocol"]["code_sha256"].items():
        verify_file(path, expected)
    params = _params(Path(config["v1_config"]))
    samples_by_id = {s["sample_id"]: s for s in oracle["samples"]}
    words = defaultdict(list)
    for row in source["per_word"]:
        words[row["sample_id"]].append({"text": row["text"], "bbox": row["bbox"]})
    protocol = {**config, "config_sha256": sha(a.config),
                "source_oracle_path": str(a.oracle_report.resolve()),
                "transferred_v1_parameters": oracle["protocol"]["parameters"]["v1"],
                "device": "cpu", "models_loaded": False,
                "code_sha256": {path: sha(path) for path in
                                [__file__, "localize/directional_component_neighbors.py"]},
                "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "torch": torch.__version__}}
    a.output_dir.mkdir(parents=True)
    save_json(a.output_dir / "protocol.json", protocol)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    records, orientations, all_stages = [], {}, {}
    for i, sample in enumerate(manifest.samples, 1):
        if i == 1 or i % 10 == 0:
            print(f"[directional grouping CPU] {i}/229 sample={sample.sample_id}", flush=True)
        old = samples_by_id[sample.sample_id]
        for rel in (sample.mask_path, sample.validity_annotation_path):
            verify_file(a.data_root / rel, source["protocol"]["dataset_files"][str(rel)]["sha256"])
        annotation = (a.data_root / sample.validity_annotation_path).read_text(encoding="utf-8-sig")
        with Image.open(a.data_root / sample.mask_path) as opened:
            target, _ = icdar13_task22_rgb_to_target_and_validity(opened, annotation)
            rgb_target = np.asarray(opened.convert("RGB"))
        native = resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
        full = (np.asarray(target) > 0).astype(np.uint8)
        pred, value = checked_native(source_path.parent / "masks" / f"{sample.sample_id}.json",
            source["contract_sha256"], sample.sample_id,
            source["protocol"]["dataset_files"][str(sample.image_path)]["sha256"],
            oracle["protocol"]["checkpoint_sha256"])
        for array, expected in ((pred, old["prediction_native_sha256"]),
                                (native, old["gt_native_sha256"]), (full, old["gt_fullres_sha256"])):
            if hashlib.sha256(array.tobytes()).hexdigest() != expected:
                raise RuntimeError("Mask differs from frozen oracle comparison")
        conditions, stages = {}, {}
        for name, binary in (("PRED128", pred), ("GT128", native), ("GT_FULLRES", full)):
            mask = full if name == "GT_FULLRES" else backproject_binary_mask(binary, full.shape)
            new = directional_neighbors(mask, params)
            conditions[name] = {**old["conditions"][name], **new["methods"]}
            stages[name] = new["stages"]
        # Post-hoc orientation labels, never passed to directional_neighbors.
        labels = word_orientations(rgb_target, annotation, config["orientation_diagnostic"])
        if len(labels) != len(words[sample.sample_id]):
            raise RuntimeError("GT glyph-group/word population mismatch")
        orientations[sample.sample_id] = labels
        records.append({"sample_id": sample.sample_id, "conditions": conditions})
        all_stages[sample.sample_id] = stages
    methods = list(oracle["summary"]) + config["variants"]
    summary, rows = summarize(records, words, methods, config["conditions"])
    for method in oracle["summary"]:
        if summary[method] != oracle["summary"][method]:
            raise RuntimeError(f"Frozen control aggregate changed: {method}")
    for row in rows:
        label = orientations[row["sample_id"]][row["gt_index"]]
        row.update(orientation=label["orientation"], orientation_angle_degrees=label["angle_degrees"])
    orientation_counts = Counter(v["orientation"] for labels in orientations.values() for v in labels)
    if sum(orientation_counts.values()) != 849:
        raise RuntimeError("Expected 849 oriented-or-unknown words")
    with (a.output_dir / "per-word.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Existing OCR correctness is not recomputed or claimed for new boxes.
    report = {"protocol": protocol, "summary": summary,
              "orientation_population": {k: orientation_counts[k] for k in CATEGORIES},
              "by_orientation": orientation_summary(rows),
              "transitions": directional_transitions(rows, config["conditions"]),
              "orientations": orientations, "stages": all_stages, "samples": records,
              "validation": {"images": 229, "gt_words": 849, "mask_hashes_match_oracle": True,
                             "all_frozen_method_aggregates_unchanged": True},
              "csv_sha256": sha(a.output_dir / "per-word.csv")}
    save_json(a.output_dir / "results.json", report)
    print(json.dumps({"orientation_population": report["orientation_population"],
                      "prediction_summary": {k: v["PRED128"] for k, v in summary.items()}}, indent=2))
    print(f"JSON {a.output_dir / 'results.json'} SHA256 {sha(a.output_dir / 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
