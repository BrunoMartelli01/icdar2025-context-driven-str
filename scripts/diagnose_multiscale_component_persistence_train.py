#!/usr/bin/env python3
"""CPU-only TRAIN diagnostic for MULTISCALE_COMPONENT_PERSISTENCE_V1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
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

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.multiscale_component_persistence import (
    LINE_MAX_CENTER_Y_RATIO, LINE_MAX_HEIGHT_RATIO, LINE_VERTICAL_OVERLAP_MIN,
    MODE, PAD_ORIGINAL, SCALES, group_multiscale_component_persistence,
    groups_in_original_coordinates,
)
from localize.practical_remap_protocol import _box_iou, train_task22_word_boxes
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.evaluate_supervised_word_boundary_train import (
    EXPECTED_CACHE_SHA256, WordEvidence, _evaluate_groups, _raw_boxes_from_labels,
    _sha256_file, _summary, _word_evidence,
)
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest


EXPECTED_V1_CALIBRATION_SHA256 = "6c0e4cb36fe49f2c1d2306fc7226a5a7d7938208b7b223b4490a6dae377a659f"
EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"


@dataclass(frozen=True)
class TrainRecord:
    sample_id: str
    width: int
    height: int
    word_boxes: tuple[tuple[int, int, int, int], ...]
    supervised_word_evidence: tuple[WordEvidence, ...]
    v1_word_evidence: tuple[WordEvidence, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _contained(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("dataset artifact path must be relative")
    value = (root / relative).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("dataset artifact escapes data root")
    return value


def _read_mask(cache_root: Path, entry: dict[str, Any]) -> np.ndarray:
    path = _contained(cache_root, Path(entry["native_mask_path"]))
    if _sha256_file(path) != entry["native_mask_file_sha256"]:
        raise RuntimeError(f"cached PNG hash mismatch: {entry['sample_id']}")
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("L"))
    if array.dtype != np.uint8 or array.shape != (128, 128) or not np.isin(array, (0, 255)).all():
        raise RuntimeError("cached prediction must be binary L-mode 128x128")
    mask = (array > 0).astype(np.uint8)
    if (hashlib.sha256(mask.tobytes()).hexdigest() != entry["native_mask_array_sha256"]
            or int(mask.sum()) != entry["foreground_pixels"]):
        raise RuntimeError(f"cached array hash/count mismatch: {entry['sample_id']}")
    return mask


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p25": None, "median": None, "p75": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "p25": float(np.quantile(array, .25)),
            "median": float(np.quantile(array, .5)), "p75": float(np.quantile(array, .75))}


def _publish(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main() -> int:
    args = _parse_args()
    cv2.setNumThreads(1)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cache_path, calibration_path = args.cache_manifest.resolve(), args.v1_calibration.resolve()
    if _sha256_file(cache_path) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("TRAIN native-mask cache hash mismatch")
    if _sha256_file(calibration_path) != EXPECTED_V1_CALIBRATION_SHA256:
        raise RuntimeError("frozen v1 calibration hash mismatch")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if (_sha256_file(args.train_manifest.resolve()) != cache["train_manifest"]["sha256"]
            or manifest.split_role != "train"
            or manifest.split != "official-training-all-229-author-protocol"
            or len(manifest.samples) != 229 or cache["sample_count"] != 229
            or cache["test_data_loaded"] is not False
            or cache["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("only uncontaminated official TRAIN-229 cache is permitted")
    v1_params = GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])
    baseline_expected = calibration["calibration"]["selected"]["geometry"]
    source_by_id = {str(value["sample_id"]): value for value in calibration["train_source"]["samples"]}
    cache_by_id = {str(value["sample_id"]): value for value in cache["samples"]}
    expected_ids = {sample.sample_id for sample in manifest.samples}
    if set(source_by_id) != expected_ids or set(cache_by_id) != expected_ids:
        raise RuntimeError("manifest/cache/calibration sample populations differ")

    # Phase 1: finalize every proposal from cached predictions before loading any GT.
    masks, dimensions, persistence_results, persistence_groups = [], [], [], []
    for index, sample in enumerate(manifest.samples, 1):
        entry = cache_by_id[sample.sample_id]
        image_path = _contained(args.data_root, sample.image_path)
        if _sha256_file(image_path) != entry["image_sha256"] or entry["image_sha256"] != sample.image_sha256:
            raise RuntimeError(f"source image identity mismatch: {sample.sample_id}")
        with Image.open(image_path) as opened:
            width, height = opened.size
        mask = _read_mask(cache_path.parent, entry)
        result = group_multiscale_component_persistence(mask)
        masks.append(mask); dimensions.append((width, height)); persistence_results.append(result)
        persistence_groups.append(groups_in_original_coordinates(result, width, height))
        if index % 10 == 0 or index == 229:
            print(f"[persistence inference CPU/no-GT] {index}/229", flush=True)

    # Phase 2: GT enters only for frozen post-hoc evaluation and diagnostics.
    records, v1_groups = [], []
    for index, (sample, mask, (width, height)) in enumerate(zip(manifest.samples, masks, dimensions), 1):
        source = source_by_id[sample.sample_id]
        target_path = _contained(args.data_root, sample.mask_path)
        annotation_path = _contained(args.data_root, sample.validity_annotation_path)
        if (_sha256_file(target_path) != source["mask_sha256"]
                or _sha256_file(annotation_path) != source["annotation_sha256"]):
            raise RuntimeError(f"GT source identity mismatch: {sample.sample_id}")
        words = tuple(word.bbox for word in train_task22_word_boxes(target_path, annotation_path))
        if len(words) != source["word_box_count"]:
            raise RuntimeError(f"GT word count mismatch: {sample.sample_id}")
        with Image.open(target_path) as opened:
            target = (np.asarray(icdar13_task22_rgb_to_binary_mask(opened)) > 0).astype(np.uint8)
        full = backproject_binary_mask(mask, (height, width))
        count, native_labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
        full_labels = cv2.resize(native_labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        raw_boxes = _raw_boxes_from_labels(full_labels)
        if len(raw_boxes) != count - 1:
            raise RuntimeError(f"native component loss in evaluation mapping: {sample.sample_id}")
        evidence = _word_evidence(full_labels, raw_boxes, full, target, words)
        v1 = group_word_blocks(full, full, v1_params)
        v1_labels = cv2.connectedComponentsWithStats(full, 8)[1]
        v1_evidence = _word_evidence(v1_labels, {v.component_id: v.bbox for v in v1.raw_components},
                                     full, target, words)
        records.append(TrainRecord(sample.sample_id, width, height, words, evidence, v1_evidence))
        groups = tuple(dict(value) for value in v1.groups_before_cap)
        v1_groups.append((groups, frozenset(tuple(value.raw_component_ids) for value in v1.retained)))
        if index % 25 == 0 or index == 229:
            print(f"[post-hoc GT evaluation prepared] {index}/229", flush=True)

    baseline_eval = _evaluate_groups(records, v1_groups, evidence_attribute="v1_word_evidence")
    persistence_eval = _evaluate_groups(records, persistence_groups,
                                         evidence_attribute="supervised_word_evidence")
    if baseline_eval["geometry"] != baseline_expected:
        raise RuntimeError("recomputed frozen v1 geometry differs from calibration")
    if baseline_eval["failure_breakdown"]["categories"]["GROUPING_UNDERMERGE"]["count"] != 92:
        raise RuntimeError("frozen v1 failure categories differ from qualified baseline")

    before = {(row["sample_id"], row["gt_word_index"]): row for row in baseline_eval["per_word"]}
    after = {(row["sample_id"], row["gt_word_index"]): row for row in persistence_eval["per_word"]}
    recovered = sum(before[key]["failure_category"] != "SUCCESS" and after[key]["failure_category"] == "SUCCESS" for key in before)
    lost = sum(before[key]["failure_category"] == "SUCCESS" and after[key]["failure_category"] != "SUCCESS" for key in before)

    selected_birth, selected_death, selected_persistence = [], [], []
    category_persistence: dict[str, list[float]] = {key: [] for key in ("SUCCESS", "GROUPING_UNDERMERGE", "GROUPING_OVERMERGE")}
    per_image_diagnostics = []
    for record, result, (groups, retained_ids) in zip(records, persistence_results, persistence_groups):
        retained = [row for row in groups if tuple(row["raw_component_ids"]) in retained_ids]
        for row in retained:
            selected_birth.append(row["birth_scale"]); selected_death.append(row["death_scale"])
            selected_persistence.append(row["persistence"])
        word_rows = [row for row in persistence_eval["per_word"] if row["sample_id"] == record.sample_id]
        for word, status in zip(record.word_boxes, word_rows):
            best = max(retained, key=lambda row: _box_iou(row["padded_bbox"], word), default=None)
            category = status["failure_category"]
            if best is not None and category in category_persistence:
                category_persistence[category].append(float(best["persistence"]))
        per_image_diagnostics.append({
            "sample_id": record.sample_id, "raw_components": len(result.raw_components),
            "lines": len(result.lines), "candidate_spans": len(result.candidates),
            "selected_groups_before_cap": len(result.selected), "retained_groups": len(retained),
        })

    summary_v1, summary_persistence = _summary(baseline_eval), _summary(persistence_eval)
    go = (persistence_eval["wordness"]["WORD_GOOD"] >= 552
          and summary_persistence["recall_iou_ge_0_50"] >= .669
          and summary_persistence["overmerge"] <= 45)
    report = {
        "schema_version": 1, "mode": MODE, "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "TRAIN_ONLY_DIAGNOSTIC", "table_5_accuracy": None,
        "scientific_boundary": {"device": "cpu", "cuda": False, "test_loaded": False,
                                "ocr_parseq": False, "training": False, "grid_or_tuning": False,
                                "gt_used_in_inference": False},
        "provenance": {"cache_manifest": str(cache_path), "cache_manifest_sha256": _sha256_file(cache_path),
                       "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                       "v1_calibration": str(calibration_path),
                       "v1_calibration_sha256": _sha256_file(calibration_path)},
        "inference_protocol": {
            "scales_r_over_H": list(SCALES), "morphology": "horizontal symmetric dilation",
            "pixel_radius": "floor(r/H * median_line_raw_component_height + 0.5)",
            "line_geometry_reused": {"vertical_overlap_min": LINE_VERTICAL_OVERLAP_MIN,
                                     "max_center_y_ratio": LINE_MAX_CENTER_Y_RATIO,
                                     "max_height_ratio": LINE_MAX_HEIGHT_RATIO},
            "selection": "parameter-free weighted interval DP",
            "weight": "(death_scale-birth_scale)/0.60 * raw_component_count",
            "bbox": "union of original native raw-component boxes; morphology excluded",
            "padding_original_pixels": PAD_ORIGINAL, "top10": True,
        },
        "v1": {"summary": summary_v1, "wordness": baseline_eval["wordness"],
               "failure_breakdown": baseline_eval["failure_breakdown"]},
        "persistence": {"summary": summary_persistence, "wordness": persistence_eval["wordness"],
                        "failure_breakdown": persistence_eval["failure_breakdown"]},
        "transition_from_v1": {"v1_failures_recovered": recovered, "v1_successes_lost": lost},
        "persistence_diagnostics": {
            "candidate_spans_per_image": _quantiles([row["candidate_spans"] for row in per_image_diagnostics]),
            "selected_birth_scale": _quantiles(selected_birth),
            "selected_death_scale": _quantiles(selected_death),
            "selected_persistence": _quantiles(selected_persistence),
            "selected_persistence_by_posthoc_gt_outcome": {
                key: _quantiles(values) for key, values in category_persistence.items()
            },
        },
        "per_image_diagnostics": per_image_diagnostics,
        "go_rule": {"word_good_min": 552, "recall_min": .669, "overmerge_max": 45},
        "go": go,
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__,
                    "opencv_threads": cv2.getNumThreads()},
        "source_sha256": {
            str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
            str(Path(__file__).resolve().parents[1] / "localize/multiscale_component_persistence.py"):
                _sha256_file(Path(__file__).resolve().parents[1] / "localize/multiscale_component_persistence.py"),
        },
    }
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _publish(output, content)
    print(json.dumps({"v1": report["v1"], "persistence": report["persistence"],
                      "transition_from_v1": report["transition_from_v1"],
                      "persistence_diagnostics": report["persistence_diagnostics"], "go": go}, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {_sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
