#!/usr/bin/env python3
"""CPU-only TRAIN feasibility oracle for inference-safe contiguous CC spans."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from localize.contiguous_span_candidates import (
    MAX_COMPONENTS_PER_SPAN, MAX_WIDTH_OVER_MEDIAN_HEIGHT, MODE, PAD_ORIGINAL,
    enumerate_contiguous_span_candidates, frozen_line_params,
    remap_and_deduplicate_candidates,
)
from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.practical_remap_protocol import _box_iou, train_task22_word_boxes
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.evaluate_supervised_word_boundary_train import EXPECTED_CACHE_SHA256, _sha256_file
from segmentation.data import load_segmentation_manifest


EXPECTED_V1_SHA256 = "6c0e4cb36fe49f2c1d2306fc7226a5a7d7938208b7b223b4490a6dae377a659f"
EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _contained(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be relative")
    value = (root / relative).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes root")
    return value


def _native(cache_root: Path, entry: dict[str, Any]) -> np.ndarray:
    path = _contained(cache_root, Path(entry["native_mask_path"]))
    if _sha256_file(path) != entry["native_mask_file_sha256"]:
        raise RuntimeError(f"native PNG SHA mismatch: {entry['sample_id']}")
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("L"))
    if array.dtype != np.uint8 or array.shape != (128, 128) or not np.isin(array, (0, 255)).all():
        raise RuntimeError("native cache must be binary L-mode 128x128")
    mask = (array > 0).astype(np.uint8)
    if hashlib.sha256(mask.tobytes()).hexdigest() != entry["native_mask_array_sha256"]:
        raise RuntimeError(f"native array SHA mismatch: {entry['sample_id']}")
    return mask


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()),
            "median": float(np.quantile(array, .5)), "p95": float(np.quantile(array, .95))}


def _oracle(boxes: Sequence[Sequence[int]], gt: Sequence[Sequence[int]]) -> tuple[dict[str, Any], list[float]]:
    values = [max((_box_iou(box, word) for box in boxes), default=0.0) for word in gt]
    return {
        "gt_words": len(gt),
        "iou_ge_0_25": sum(value >= .25 for value in values),
        "iou_ge_0_50": sum(value >= .50 for value in values),
        "iou_ge_0_75": sum(value >= .75 for value in values),
        "iou_ge_0_90": sum(value >= .90 for value in values),
        "recall_iou_ge_0_25": sum(value >= .25 for value in values) / len(gt),
        "recall_iou_ge_0_50": sum(value >= .50 for value in values) / len(gt),
        "recall_iou_ge_0_75": sum(value >= .75 for value in values) / len(gt),
        "recall_iou_ge_0_90": sum(value >= .90 for value in values) / len(gt),
        "median_best_iou": float(np.median(values)),
        "gt_words_without_candidate_iou_ge_0_50": sum(value < .5 for value in values),
    }, values


def _native_box_to_padded(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    mapped = (math.floor(box[0] * width / 128), math.floor(box[1] * height / 128),
              math.ceil(box[2] * width / 128), math.ceil(box[3] * height / 128))
    return (max(0, mapped[0] - PAD_ORIGINAL), max(0, mapped[1] - PAD_ORIGINAL),
            min(width, mapped[2] + PAD_ORIGINAL), min(height, mapped[3] + PAD_ORIGINAL))


def _union(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    return (min(v[0] for v in boxes), min(v[1] for v in boxes),
            max(v[2] for v in boxes), max(v[3] for v in boxes))


def _classify_failure(
    *, word: Sequence[int], associated: tuple[int, ...], lines: tuple[dict[str, Any], ...],
    raw_boxes: dict[int, tuple[int, int, int, int]], width: int, height: int,
) -> tuple[str, dict[str, Any]]:
    if not associated:
        return "NO_FOREGROUND_COMPONENTS", {"associated_raw_components": 0}
    line_by_raw = {int(value): int(line["line_id"]) for line in lines for value in line["raw_component_ids"]}
    line_ids = sorted({line_by_raw[value] for value in associated})
    associated_box = _native_box_to_padded(_union([raw_boxes[value] for value in associated]), width, height)
    associated_iou = _box_iou(associated_box, word)
    details = {"associated_raw_components": len(associated), "associated_line_ids": line_ids,
               "associated_union_iou": associated_iou}
    if len(line_ids) > 1:
        return "GT_WORD_REQUIRES_COMPONENTS_FROM_MULTIPLE_GENERATED_LINES", details
    line = next(value for value in lines if int(value["line_id"]) == line_ids[0])
    ids = tuple(int(value) for value in line["raw_component_ids"])
    positions = [ids.index(value) for value in associated]
    start, end = min(positions), max(positions) + 1
    hull_ids = ids[start:end]
    details["contiguous_hull_component_count"] = len(hull_ids)
    hull_iou = _box_iou(_native_box_to_padded(
        _union([raw_boxes[value] for value in hull_ids]), width, height), word)
    details["contiguous_hull_iou"] = hull_iou
    if len(hull_ids) > MAX_COMPONENTS_PER_SPAN:
        return "SPAN_GT_12_COMPONENTS", details
    if associated_iou >= .5 and hull_iou < .5:
        return "WRONG_LINE_ASSIGNMENT", details
    if associated_iou < .5:
        return "GEOMETRY_REMAP_ISSUE", details
    return "OTHER", details


def _publish(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name); stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main() -> int:
    args = _args(); cv2.setNumThreads(1)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cache_path, v1_path = args.cache_manifest.resolve(), args.v1_calibration.resolve()
    if _sha256_file(cache_path) != EXPECTED_CACHE_SHA256 or _sha256_file(v1_path) != EXPECTED_V1_SHA256:
        raise RuntimeError("cache or frozen-v1 provenance hash mismatch")
    cache = json.loads(cache_path.read_text()); v1_report = json.loads(v1_path.read_text())
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if (_sha256_file(args.train_manifest.resolve()) != cache["train_manifest"]["sha256"]
            or manifest.split_role != "train" or len(manifest.samples) != 229
            or cache["sample_count"] != 229 or cache["test_data_loaded"] is not False
            or cache["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("requires uncontaminated frozen TRAIN-229 cache")
    cache_by_id = {str(row["sample_id"]): row for row in cache["samples"]}
    source_by_id = {str(row["sample_id"]): row for row in v1_report["train_source"]["samples"]}
    if set(cache_by_id) != {s.sample_id for s in manifest.samples} or set(source_by_id) != set(cache_by_id):
        raise RuntimeError("source populations differ")
    v1_params = GroupedWordBlockParams(**v1_report["calibration"]["selected"]["parameters"])

    # Candidate generation is completed for every image before any GT is opened.
    prepared = []
    for index, sample in enumerate(manifest.samples, 1):
        entry = cache_by_id[sample.sample_id]
        image_path = _contained(args.data_root, sample.image_path)
        if _sha256_file(image_path) != entry["image_sha256"]:
            raise RuntimeError(f"RGB identity mismatch: {sample.sample_id}")
        with Image.open(image_path) as opened:
            width, height = opened.size
        mask = _native(cache_path.parent, entry)
        spans = enumerate_contiguous_span_candidates(mask)
        candidates, dedup = remap_and_deduplicate_candidates(spans, width, height)
        full = backproject_binary_mask(mask, (height, width))
        v1 = group_word_blocks(full, full, v1_params)
        prepared.append((sample, mask, width, height, spans, candidates, dedup,
                         tuple(block.padded_bbox for block in v1.retained)))
        if index % 10 == 0 or index == 229:
            print(f"[candidate generation CPU/no-GT] {index}/229", flush=True)

    all_gt, all_span_best, all_v1_best, failures = [], [], [], []
    image_counts, line_counts, dedup_totals = [], [], Counter()
    for index, (sample, mask, width, height, spans, candidates, dedup, v1_boxes) in enumerate(prepared, 1):
        source = source_by_id[sample.sample_id]
        target_path = _contained(args.data_root, sample.mask_path)
        annotation_path = _contained(args.data_root, sample.validity_annotation_path)
        if (_sha256_file(target_path) != source["mask_sha256"]
                or _sha256_file(annotation_path) != source["annotation_sha256"]):
            raise RuntimeError(f"GT identity mismatch: {sample.sample_id}")
        words = tuple(word.bbox for word in train_task22_word_boxes(target_path, annotation_path))
        if len(words) != source["word_box_count"]:
            raise RuntimeError(f"GT population mismatch: {sample.sample_id}")
        span_boxes = tuple(row["padded_bbox_original"] for row in candidates)
        _, span_best = _oracle(span_boxes, words); _, v1_best = _oracle(v1_boxes, words)
        count, native_labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        full_labels = cv2.resize(native_labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        raw_boxes = {component_id: (int(stats[component_id, 0]), int(stats[component_id, 1]),
                                    int(stats[component_id, 0] + stats[component_id, 2]),
                                    int(stats[component_id, 1] + stats[component_id, 3]))
                     for component_id in range(1, count)}
        for word_index, (word, best) in enumerate(zip(words, span_best), 1):
            if best >= .5:
                continue
            x1, y1, x2, y2 = word
            associated = tuple(sorted(int(value) for value in np.unique(full_labels[y1:y2, x1:x2]) if value))
            category, details = _classify_failure(word=word, associated=associated, lines=spans.lines,
                                                  raw_boxes=raw_boxes, width=width, height=height)
            failures.append({"sample_id": sample.sample_id, "gt_word_index": word_index,
                             "best_candidate_iou": best, "category": category, **details})
        all_gt.extend(words); all_span_best.extend(span_best); all_v1_best.extend(v1_best)
        image_counts.append(len(candidates)); line_counts.extend(line["candidate_span_count"] for line in spans.lines)
        dedup_totals.update(dedup)
        if index % 25 == 0 or index == 229:
            print(f"[oracle post-hoc GT] {index}/229", flush=True)

    def aggregate(best: list[float]) -> dict[str, Any]:
        return {"gt_words": len(best),
                **{f"iou_ge_{str(threshold).replace('.', '_')}": sum(v >= threshold for v in best)
                   for threshold in (.25, .5, .75, .9)},
                **{f"recall_iou_ge_{str(threshold).replace('.', '_')}": sum(v >= threshold for v in best)/len(best)
                   for threshold in (.25, .5, .75, .9)},
                "median_best_iou": float(np.median(best)),
                "gt_words_without_candidate_iou_ge_0_5": sum(v < .5 for v in best)}
    span_metrics, v1_metrics = aggregate(all_span_best), aggregate(all_v1_best)
    # Frozen v1 must reproduce known oracle counts where already available.
    if any(v1_metrics[key] != expected for key, expected in
           (("iou_ge_0_25", 679), ("iou_ge_0_5", 568), ("iou_ge_0_75", 324))):
        raise RuntimeError("frozen practical-remap-v1 geometry mismatch")
    breakdown = Counter(row["category"] for row in failures)
    if sum(breakdown.values()) != span_metrics["gt_words_without_candidate_iou_ge_0_5"]:
        raise RuntimeError("failure accounting mismatch")
    conclusion = ("CANDIDATE_RERANKING_FEASIBLE"
                  if span_metrics["recall_iou_ge_0_5"] >= .85
                  else "CANDIDATE_GENERATION_INSUFFICIENT")
    report = {
        "schema_version": 1, "mode": MODE, "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "table_5_accuracy": None,
        "scientific_boundary": {"device": "cpu", "cuda": False, "test_loaded": False,
                                "ocr_parseq": False, "training": False, "gt_used_in_candidate_generation": False,
                                "threshold_or_parameter_selection": False},
        "provenance": {"cache_manifest": str(cache_path), "cache_manifest_sha256": _sha256_file(cache_path),
                       "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                       "v1_calibration_sha256": _sha256_file(v1_path)},
        "candidate_protocol": {
            "line_geometry": vars(frozen_line_params()),
            "max_raw_components_per_span": MAX_COMPONENTS_PER_SPAN,
            "max_span_width_over_median_component_height": MAX_WIDTH_OVER_MEDIAN_HEIGHT,
            "contiguous_only": True, "padding_original_pixels": PAD_ORIGINAL,
            "bbox_source": "union of unmorphed original native raw-component boxes",
            "deduplication": "exact equality of final padded original-coordinate bbox; no extra near-IoU threshold",
            "top10_applied_to_oracle_pool": False,
        },
        "oracle": {"raw_contiguous_spans": span_metrics, "practical_remap_v1_retained": v1_metrics},
        "candidate_counts": {"per_image_after_deduplication": _quantiles(image_counts),
                             "per_line_before_image_deduplication": _quantiles(line_counts),
                             "deduplication_totals": dict(dedup_totals)},
        "failure_breakdown": {key: {"count": breakdown[key], "fraction": breakdown[key]/len(failures)}
                              for key in ("NO_FOREGROUND_COMPONENTS", "WRONG_LINE_ASSIGNMENT",
                                          "GT_WORD_REQUIRES_COMPONENTS_FROM_MULTIPLE_GENERATED_LINES",
                                          "SPAN_GT_12_COMPONENTS", "GEOMETRY_REMAP_ISSUE", "OTHER")},
        "failure_records": failures, "conclusion": conclusion,
        "source_sha256": {str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
                          str(Path(__file__).resolve().parents[1]/"localize/contiguous_span_candidates.py"):
                              _sha256_file(Path(__file__).resolve().parents[1]/"localize/contiguous_span_candidates.py")},
    }
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _publish(output, content)
    print(json.dumps({"oracle": report["oracle"], "candidate_counts": report["candidate_counts"],
                      "failure_breakdown": report["failure_breakdown"], "conclusion": conclusion}, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {_sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
