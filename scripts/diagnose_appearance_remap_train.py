#!/usr/bin/env python3
"""One configuration, CPU-only TRAIN diagnostics from verified native-mask cache.

No segmentation/recognizer is instantiated, no fit or grid is performed, and no
TEST image is opened. The three methods consume the same cached prediction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from localize.appearance_remap_config import load_appearance_config
from localize.appearance_word_remap import MODE, appearance_guided_local_remap
from localize.practical_remap_protocol import _box_iou, proposal_geometry, train_task22_word_boxes
from segmentation.data import load_segmentation_manifest


METHODS = ("v1_frozen", "v1_matched_postprocessing", "appearance_remap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact paths must be relative to the supplied root")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes supplied root")
    return result


def verify_train_contract(cache: dict, manifest: Any, calibration: dict, config: Any) -> tuple[dict, dict]:
    """Metadata guard before reading any source RGB/GT; never infer a split."""
    expected = config.resolved["qualification"]
    guard = calibration["train_only_guard"]
    if (manifest.split_role != "train" or manifest.split != expected["split"]
            or len(manifest.samples) != expected["sample_count"]
            or cache["split"] != expected["split"] or cache["sample_count"] != expected["sample_count"]
            or cache["test_data_loaded"] is not False or guard["test_data_loaded"] is not False
            or guard["split_role"] != "train" or guard["sample_count"] != expected["sample_count"]):
        raise ValueError("requires only official TRAIN-229, with explicit uncontaminated provenance")
    inference = cache["inference"]
    if (inference["native_shape"] != [128, 128] or inference["threshold"] != 0.5
            or inference["activation"] != "sigmoid" or inference["comparison"] != "greater_than_or_equal"
            or inference["backprojection_applied"] is not False):
        raise ValueError("cache is not the qualified native128 sigmoid>=0.5 prediction")
    if cache["checkpoint"]["sha256"] != expected["checkpoint_sha256"]:
        raise ValueError("cached segmentation checkpoint identity mismatch")
    if calibration["frozen_segmentation"]["checkpoint"]["sha256"] != expected["checkpoint_sha256"]:
        raise ValueError("v1 calibration uses a different segmentation checkpoint")
    if calibration["calibration"]["selected"]["parameters"] != asdict(config.v1):
        raise ValueError("v1 parameters differ from the frozen calibration")
    if (calibration["scientific_boundary"]["selection_used_ocr"] is not False
            or calibration["scientific_boundary"]["selection_used_test"] is not False):
        raise ValueError("v1 calibration is contaminated")
    ids = {sample.sample_id for sample in manifest.samples}
    cached = {str(row["sample_id"]): row for row in cache["samples"]}
    sources = {str(row["sample_id"]): row for row in calibration["train_source"]["samples"]}
    if (len(ids) != 229 or len(cache["samples"]) != 229
            or len(calibration["train_source"]["samples"]) != 229
            or set(cached) != ids or set(sources) != ids):
        raise ValueError("cache, manifest and v1 sample identities/counts differ")
    return cached, sources


def read_native_mask(root: Path, entry: dict) -> np.ndarray:
    path = _contained(root, entry["native_mask_path"])
    if sha256_file(path) != entry["native_mask_file_sha256"]:
        raise ValueError(f"native PNG SHA mismatch: {entry['sample_id']}")
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError("cached native PNG must be L mode")
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (128, 128) or not np.isin(array, [0, 255]).all():
        raise ValueError("cached native PNG must encode binary128 as 0/255")
    native = (array > 0).astype(np.uint8)
    if (hashlib.sha256(native.tobytes()).hexdigest() != entry["native_mask_array_sha256"]
            or int(native.sum()) != entry["foreground_pixels"]):
        raise ValueError("decoded native mask SHA/count mismatch")
    return native


def crop_geometry(blocks: tuple, gt: tuple, significant_coverage: float) -> tuple[list, dict, list]:
    """Crop counts and per-GT geometry are distinct; no OCR-based association."""
    counts = {key: 0 for key in ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")}
    crops = []
    for block in blocks:
        box = block.padded_bbox
        overlaps, ious = [], []
        for word in gt:
            intersection = max(0, min(box[2], word[2])-max(box[0], word[0])) * max(0, min(box[3], word[3])-max(box[1], word[1]))
            overlaps.append(intersection / ((word[2]-word[0])*(word[3]-word[1])))
            ious.append(_box_iou(box, word))
        significant = sum(value >= significant_coverage for value in overlaps)
        best = max(range(len(gt)), key=lambda k: (ious[k], -k)) if gt else None
        best_iou = ious[best] if best is not None else 0.0
        category = ("MULTIWORD_MERGE" if significant >= 2 else "WORD_GOOD" if best_iou >= .5
                    else "WORD_PARTIAL" if significant == 1 else "UNMATCHED")
        counts[category] += 1
        crops.append({"rank": block.rank, "raw_component_ids": block.raw_component_ids,
                      "bbox": block.bbox, "padded_bbox": box, "foreground_area": block.foreground_area,
                      "best_gt_index": best+1 if best is not None and best_iou > 0 else None,
                      "best_gt_iou": best_iou, "significantly_intersected_words": significant,
                      "wordness": category})
    per_word = []
    for index, word in enumerate(gt, 1):
        best = max(blocks, key=lambda block: (_box_iou(block.padded_bbox, word), -block.rank), default=None)
        iou = _box_iou(best.padded_bbox, word) if best is not None else 0.0
        per_word.append({"gt_word_index": index, "gt_bbox": word, "best_iou": iou,
                         "best_crop_rank": best.rank if best is not None and iou > 0 else None})
    return crops, counts, per_word


def _geometry_input(groups: tuple, blocks: tuple, gt: tuple) -> tuple:
    return ([block.padded_bbox for block in blocks], gt, len(groups),
            sum(len(g["raw_component_ids"])-1 for g in groups),
            sum(len(g["raw_component_ids"]) > 1 for g in groups))


def _write_exclusive(path: Path, content: str) -> None:
    """Complete-file publish without overwriting an existing result, even in a race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink()
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2.setNumThreads(1)
    config = load_appearance_config(args.config)
    expected = config.resolved["qualification"]
    output, csv_output = args.output.resolve(), args.csv_output.resolve()
    if output == csv_output or output.exists() or csv_output.exists():
        raise FileExistsError("JSON/CSV must be distinct new paths; existing reports are never overwritten")
    cache_path = args.cache_manifest.resolve()
    if sha256_file(cache_path) != expected["cache_manifest_sha256"]:
        raise ValueError("cache manifest SHA mismatch")
    if sha256_file(args.v1_calibration) != expected["v1_calibration_sha256"]:
        raise ValueError("v1 calibration SHA mismatch")
    cache = json.loads(cache_path.read_text())
    calibration = json.loads(args.v1_calibration.read_text())
    if sha256_file(args.train_manifest) != cache["train_manifest"]["sha256"]:
        raise ValueError("training manifest SHA mismatch")
    manifest = load_segmentation_manifest(args.train_manifest)
    cached, sources = verify_train_contract(cache, manifest, calibration, config)
    # Verify all input identities first, before producing any experimental output.
    paths = {}
    for index, sample in enumerate(manifest.samples, 1):
        image = _contained(args.data_root, str(sample.image_path))
        target = _contained(args.data_root, str(sample.mask_path))
        annotation = _contained(args.data_root, str(sample.validity_annotation_path))
        source = sources[sample.sample_id]
        if (sha256_file(image) != cached[sample.sample_id]["image_sha256"]
                or cached[sample.sample_id]["image_sha256"] != source["image_sha256"]
                or source["image_sha256"] != sample.image_sha256
                or sha256_file(target) != source["mask_sha256"] or source["mask_sha256"] != sample.mask_sha256
                or sha256_file(annotation) != source["annotation_sha256"]):
            raise ValueError(f"source RGB/GT identity mismatch: sample {sample.sample_id}")
        read_native_mask(cache_path.parent, cached[sample.sample_id])
        paths[sample.sample_id] = image, target, annotation
        if index % 25 == 0 or index == 229:
            print(f"[verify CPU] {index}/229 RGB/GT/native-mask hashes PASS", flush=True)

    started = time.perf_counter()
    per_image, csv_rows = [], []
    geometry_rows = {key: [] for key in METHODS}
    wordness = {key: {name: 0 for name in ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")} for key in METHODS}
    for index, sample in enumerate(manifest.samples, 1):
        image_path, target_path, annotation_path = paths[sample.sample_id]
        native = read_native_mask(cache_path.parent, cached[sample.sample_id])
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        image_started = time.perf_counter()
        result = appearance_guided_local_remap(native, rgb, v1_params=config.v1, params=config.remap)
        # GT enters only after all proposal decisions have been finalized.
        gt = tuple(word.bbox for word in train_task22_word_boxes(target_path, annotation_path))
        if len(gt) != sources[sample.sample_id]["word_box_count"]:
            raise ValueError(f"GT word population drift: {sample.sample_id}")
        methods = {
            "v1_frozen": (result.base.groups_before_cap, result.base.retained),
            "v1_matched_postprocessing": (result.control_groups_before_cap, result.control_retained),
            "appearance_remap": (result.groups_before_cap, result.retained),
        }
        row = {"sample_id": sample.sample_id, "image_shape": list(rgb.shape),
               "source": sources[sample.sample_id], "cache": cached[sample.sample_id],
               "diagnostic_trace": result.as_dict(), "methods": {}}
        for method, (groups, blocks) in methods.items():
            geometry_rows[method].append(_geometry_input(groups, blocks, gt))
            crops, counts, per_word = crop_geometry(blocks, gt, expected["significant_gt_coverage"])
            for key, count in counts.items():
                wordness[method][key] += count
            row["methods"][method] = {"crops": crops, "per_word": per_word, "wordness": counts}
            csv_rows.extend({"sample_id": sample.sample_id, "method": method, **word} for word in per_word)
        row["localization_and_geometry_seconds"] = time.perf_counter()-image_started
        per_image.append(row)
        print(f"[appearance CPU] {index}/229 sample={sample.sample_id} "
              f"v1={len(result.base.retained)} control={len(result.control_retained)} "
              f"appearance={len(result.retained)} accepted_merges={sum(r['accepted'] for r in result.merge_decisions)}",
              flush=True)
    metrics = {method: {"geometry": proposal_geometry(geometry_rows[method]).as_dict(), "wordness": wordness[method]} for method in METHODS}
    if any(item["geometry"]["gt_words"] != 849 for item in metrics.values()):
        raise RuntimeError("full TRAIN diagnostic must account for all 849 GT words")
    if metrics["v1_frozen"]["geometry"] != calibration["calibration"]["selected"]["geometry"]:
        raise RuntimeError("recomputed v1 differs from frozen baseline; no report accepted")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["sample_id", "method", "gt_word_index", "gt_bbox", "best_iou", "best_crop_rank"])
    writer.writeheader()
    for row in csv_rows:
        writer.writerow({**row, "gt_bbox": json.dumps(row["gt_bbox"])})
    csv_content = buffer.getvalue()
    report = {
        "schema_version": 1, "mode": MODE, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "result_class": "PRACTICAL_APPROXIMATION",
        "status": "single_configuration_train_diagnostic_not_promoted", "table_5_accuracy": None,
        "configuration": config.resolved, "resolved_config_sha256": config.resolved_sha256,
        "configuration_file_sha256": sha256_file(config.source),
        "cache_manifest_sha256": sha256_file(cache_path),
        "v1_calibration_sha256": sha256_file(args.v1_calibration),
        "checkpoint_sha256_evidence": {"sha256": expected["checkpoint_sha256"], "method": "pinned_cache_manifest; no checkpoint/model loaded"},
        "execution": {"device": "cpu", "samples": len(per_image), "gt_words": 849,
                      "segmentation_inference": False, "training": False, "ocr": False,
                      "test_data_loaded": False, "grid_or_selection": False,
                      "seconds_after_preflight": time.perf_counter()-started},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads()},
        "metric_contract": {
            "proposal_matching": "frozen diagnostic greedy one-to-one descending IoU; threshold .5",
            "gt_coverage_counts": "per-GT best retained padded bbox at .25/.50/.75",
            "wordness": "crop-level MULTIWORD priority when >=2 GT boxes each covered >= configured fraction; then IoU>=.5 GOOD; one significant overlap PARTIAL; else UNMATCHED",
            "failure_causality": "no exclusive under/overmerge causal classification inferred",
            "comparisons": "frozen v1 kept intact; matched-v1 uses exactly the same original-pixel area filter, padding, ordering and cap as appearance remap",
        },
        "metrics": metrics, "per_image": per_image,
        "csv": {"path": str(csv_output), "sha256": hashlib.sha256(csv_content.encode()).hexdigest()},
        "source_sha256": {str(path): sha256_file(path) for path in (
            Path(__file__).resolve(), config.source,
            Path(__file__).resolve().parents[1]/"localize/appearance_word_remap.py",
            Path(__file__).resolve().parents[1]/"localize/appearance_remap_config.py",
            Path(__file__).resolve().parents[1]/"localize/grouped_word_blocks.py",
            Path(__file__).resolve().parents[1]/"localize/practical_remap_protocol.py",
        )},
    }
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"
    _write_exclusive(csv_output, csv_content)
    _write_exclusive(output, content)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {sha256_file(output)}\nCSV {csv_output}\nSHA256 {sha256_file(csv_output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
