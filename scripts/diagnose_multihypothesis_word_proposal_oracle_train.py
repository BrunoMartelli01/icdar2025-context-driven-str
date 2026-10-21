#!/usr/bin/env python3
"""CPU-only TRAIN oracle for multi-hypothesis mask-driven word proposals."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from localize.contiguous_span_candidates import (
    MAX_WIDTH_OVER_MEDIAN_HEIGHT,
    enumerate_contiguous_span_candidates,
    frozen_line_params,
    remap_and_deduplicate_candidates,
)
from localize.multihypothesis_word_proposals import (
    BBOX_VARIANTS,
    DEDUPLICATION_IOU,
    MAX_COMPONENTS_FULL,
    MODE,
    THRESHOLDS,
    box_iou,
    component_union_bbox_variants,
    deduplicate_candidates,
    enumerate_native_hypotheses,
    frozen_cross_line_params,
    materialize_bbox_variants,
)
from localize.practical_remap_protocol import train_task22_word_boxes
from scripts.evaluate_supervised_word_boundary_train import _sha256_file
from segmentation.data import load_segmentation_manifest


EXPECTED_PROBABILITY_MANIFEST_SHA256 = "9eb8f44ad939fdc72d4a4d97f18c56a674b1525b843cac44752f9a0f69d6f7ba"
EXPECTED_MASK_MANIFEST_SHA256 = "e9a7ce5badfbc3d99885fef5e655b20f7f5f167f29defa2d3f3c7e91c6d5851a"
EXPECTED_V1_SHA256 = "6c0e4cb36fe49f2c1d2306fc7226a5a7d7938208b7b223b4490a6dae377a659f"
EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"

# Deduplication keeps the first hypothesis.  The frozen .50/standard/same-line
# path therefore has priority, followed by progressively lower thresholds.
CANONICAL_THRESHOLD_ORDER = (0.50, 0.40, 0.30, 0.20)

ABLATIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("1_old_contiguous_spans_t050", {
        "thresholds": (0.50,), "max_span": 12, "line_robust": False,
        "bbox_variants": ("standard_pad6",),
    }),
    ("2_threshold_ensemble", {
        "thresholds": THRESHOLDS, "max_span": 12, "line_robust": False,
        "bbox_variants": ("standard_pad6",),
    }),
    ("3_max_span_20", {
        "thresholds": THRESHOLDS, "max_span": 20, "line_robust": False,
        "bbox_variants": ("standard_pad6",),
    }),
    ("4_line_robust", {
        "thresholds": THRESHOLDS, "max_span": 20, "line_robust": True,
        "bbox_variants": ("standard_pad6",),
    }),
    ("5_bbox_variants", {
        "thresholds": THRESHOLDS, "max_span": 20, "line_robust": True,
        "bbox_variants": BBOX_VARIANTS,
    }),
    ("6_full", {
        "thresholds": THRESHOLDS, "max_span": 20, "line_robust": True,
        "bbox_variants": BBOX_VARIANTS,
    }),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _contained(root: Path, relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("artifact path must be relative")
    value = (root / candidate).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes its declared root")
    return value


def _load_probability(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _contained(root, record["native_probability_path"])
    if _sha256_file(path) != record["native_probability_file_sha256"]:
        raise RuntimeError(f"probability file SHA mismatch: {record['sample_id']}")
    value = np.load(path, allow_pickle=False)
    if (
        value.shape != (128, 128)
        or value.dtype.str != "<f4"
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        or np.any(value > 1.0)
    ):
        raise RuntimeError(f"invalid probability map: {record['sample_id']}")
    if hashlib.sha256(value.tobytes(order="C")).hexdigest() != record["native_probability_array_sha256"]:
        raise RuntimeError(f"probability array SHA mismatch: {record['sample_id']}")
    thresholded = np.ascontiguousarray((value >= 0.5).astype(np.uint8))
    if hashlib.sha256(thresholded.tobytes()).hexdigest() != record["reference_native_mask_array_sha256"]:
        raise RuntimeError(f"threshold-.50 native-mask mismatch: {record['sample_id']}")
    return value


def _quantiles(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _bbox_digest(candidates: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(
        [list(value["bbox_original"]) for value in candidates], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_stage_candidates(
    master: Sequence[dict[str, Any]], config: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    thresholds = set(config["thresholds"])
    variants = set(config["bbox_variants"])
    selected = [
        value for value in master
        if value["threshold"] in thresholds
        and len(value["raw_component_ids"]) <= config["max_span"]
        and (config["line_robust"] or value["source_kind"] == "same_line")
        and value["bbox_variant"] in variants
    ]
    return deduplicate_candidates(selected, threshold=DEDUPLICATION_IOU)


def _aggregate(best_ious: Sequence[float]) -> dict[str, Any]:
    if len(best_ious) != 849:
        raise RuntimeError(f"expected 849 GT words, got {len(best_ious)}")
    return {
        "gt_words": len(best_ious),
        **{
            f"iou_ge_{str(threshold).replace('.', '_')}": sum(value >= threshold for value in best_ious)
            for threshold in (0.25, 0.50, 0.75, 0.90)
        },
        **{
            f"recall_iou_ge_{str(threshold).replace('.', '_')}":
                sum(value >= threshold for value in best_ious) / len(best_ious)
            for threshold in (0.25, 0.50, 0.75, 0.90)
        },
        "median_best_iou": float(np.median(best_ious)),
        "failure_count_iou_0_50": sum(value < 0.50 for value in best_ious),
    }


def _union(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    return (
        min(value[0] for value in boxes), min(value[1] for value in boxes),
        max(value[2] for value in boxes), max(value[3] for value in boxes),
    )


def _full_topology_orders(hypotheses: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(line["raw_component_ids"] for line in hypotheses.base_lines) + tuple(
        line["raw_component_ids"] for line in hypotheses.robust_lines
    )


def _classify_full_failure(
    *, word: Sequence[int], mask: np.ndarray, hypotheses: Any,
    width: int, height: int,
) -> tuple[str, dict[str, Any]]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    expected = {value.component_id: value.bbox for value in hypotheses.raw_components}
    actual = {
        component_id: (
            int(stats[component_id, cv2.CC_STAT_LEFT]),
            int(stats[component_id, cv2.CC_STAT_TOP]),
            int(stats[component_id, cv2.CC_STAT_LEFT] + stats[component_id, cv2.CC_STAT_WIDTH]),
            int(stats[component_id, cv2.CC_STAT_TOP] + stats[component_id, cv2.CC_STAT_HEIGHT]),
        )
        for component_id in range(1, component_count)
    }
    if actual != expected:
        raise RuntimeError("threshold-.20 raw-component identity mismatch")
    original_labels = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = map(int, word)
    associated = tuple(sorted(
        int(value) for value in np.unique(original_labels[y1:y2, x1:x2]) if value
    ))
    if not associated:
        return "NO_FOREGROUND_EVEN_AT_0_20", {"associated_raw_components": 0}

    raw_by_id = {value.component_id: value for value in hypotheses.raw_components}
    union_variants, _ = component_union_bbox_variants(
        raw_by_id, associated, width=width, height=height
    )
    union_best = max(box_iou(box, word) for box in union_variants.values())
    details: dict[str, Any] = {
        "associated_raw_components": len(associated),
        "associated_union_best_iou": union_best,
    }
    if union_best < 0.50:
        return "GEOMETRY_REMAP", details

    orders = [order for order in _full_topology_orders(hypotheses) if set(associated) <= set(order)]
    if not orders:
        return "LINE_TOPOLOGY", details
    order = min(orders, key=lambda value: (len(value), value))
    positions = [order.index(value) for value in associated]
    hull_ids = order[min(positions):max(positions) + 1]
    details["contiguous_hull_component_count"] = len(hull_ids)
    reference_height = float(median(
        raw_by_id[value].bbox[3] - raw_by_id[value].bbox[1] for value in order
    ))
    hull_native = _union([raw_by_id[value].bbox for value in hull_ids])
    width_ratio = (hull_native[2] - hull_native[0]) / reference_height
    details["contiguous_hull_width_over_median_height"] = width_ratio
    if len(hull_ids) > MAX_COMPONENTS_FULL or width_ratio > MAX_WIDTH_OVER_MEDIAN_HEIGHT:
        return "CANDIDATE_PRUNING_SPAN_LIMIT", details
    hull_variants, _ = component_union_bbox_variants(
        raw_by_id, hull_ids, width=width, height=height
    )
    details["contiguous_hull_best_iou"] = max(box_iou(box, word) for box in hull_variants.values())
    if details["contiguous_hull_best_iou"] < 0.50:
        return "LINE_TOPOLOGY", details
    return "OTHER", details


def _publish(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main() -> int:
    args = _args(); cv2.setNumThreads(1)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    probability_manifest = args.probability_manifest.resolve()
    train_manifest = args.train_manifest.resolve()
    v1_path = args.v1_calibration.resolve()
    if _sha256_file(probability_manifest) != EXPECTED_PROBABILITY_MANIFEST_SHA256:
        raise RuntimeError("TRAIN probability manifest SHA256 mismatch")
    if _sha256_file(v1_path) != EXPECTED_V1_SHA256:
        raise RuntimeError("frozen v1 calibration SHA256 mismatch")
    probability_report = json.loads(probability_manifest.read_text(encoding="utf-8"))
    v1_report = json.loads(v1_path.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(train_manifest)
    if (
        probability_report.get("sample_count") != 229
        or probability_report.get("test_data_loaded") is not False
        or probability_report.get("gt_loaded") is not False
        or probability_report.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or probability_report.get("reference_mask_cache", {}).get("sha256") != EXPECTED_MASK_MANIFEST_SHA256
        or probability_report.get("reference_mask_cache", {}).get("all_thresholded_maps_match") is not True
        or probability_report.get("train_manifest", {}).get("sha256") != _sha256_file(train_manifest)
        or manifest.split_role != "train"
        or len(manifest.samples) != 229
    ):
        raise RuntimeError("requires uncontaminated frozen TRAIN-229 probability cache")
    probability_by_id = {str(row["sample_id"]): row for row in probability_report["samples"]}
    source_by_id = {str(row["sample_id"]): row for row in v1_report["train_source"]["samples"]}
    expected_ids = {sample.sample_id for sample in manifest.samples}
    if set(probability_by_id) != expected_ids or set(source_by_id) != expected_ids:
        raise RuntimeError("TRAIN source populations differ")

    # Complete all inference-safe candidate generation before opening any GT.
    prepared = []
    for image_index, sample in enumerate(manifest.samples, 1):
        record = probability_by_id[sample.sample_id]
        image_path = _contained(args.data_root.resolve(), sample.image_path)
        if _sha256_file(image_path) != record["image_sha256"]:
            raise RuntimeError(f"TRAIN RGB SHA mismatch: {sample.sample_id}")
        with Image.open(image_path) as opened:
            width, height = opened.size
        probability = _load_probability(probability_manifest.parent, record)
        hypotheses_by_threshold, master = {}, []
        for threshold in CANONICAL_THRESHOLD_ORDER:
            mask = np.ascontiguousarray((probability >= threshold).astype(np.uint8))
            hypotheses = enumerate_native_hypotheses(mask, max_components=MAX_COMPONENTS_FULL)
            hypotheses_by_threshold[threshold] = hypotheses
            master.extend(materialize_bbox_variants(
                hypotheses, threshold=threshold, width=width, height=height
            ))

        # Fail closed: the stage-1 pre-dedup candidate geometry is exactly the
        # earlier contiguous-span generator at .50/max12/standard padding.
        old = enumerate_contiguous_span_candidates((probability >= 0.50).astype(np.uint8))
        old_candidates, _ = remap_and_deduplicate_candidates(old, width, height)
        old_boxes = {tuple(value["padded_bbox_original"]) for value in old_candidates}
        new_boxes = {
            tuple(value["bbox_original"]) for value in master
            if value["threshold"] == 0.50
            and value["source_kind"] == "same_line"
            and len(value["raw_component_ids"]) <= 12
            and value["bbox_variant"] == "standard_pad6"
        }
        if old_boxes != new_boxes:
            raise RuntimeError(f"old contiguous-span geometry drift: {sample.sample_id}")

        stage_candidates = {
            name: _select_stage_candidates(master, config) for name, config in ABLATIONS
        }
        if stage_candidates["5_bbox_variants"] != stage_candidates["6_full"]:
            raise RuntimeError("FULL must be the exact cumulative bbox-variant stage")
        prepared.append({
            "sample": sample, "width": width, "height": height,
            "probability": probability,
            "hypotheses_by_threshold": hypotheses_by_threshold,
            "stage_candidates": stage_candidates,
        })
        if image_index % 10 == 0 or image_index == 229:
            print(f"[multi-hypothesis CPU/no-GT] {image_index}/229", flush=True)

    aggregate_best = {name: [] for name, _ in ABLATIONS}
    candidate_counts = {name: [] for name, _ in ABLATIONS}
    samples_output = []
    failures = []
    for image_index, prepared_sample in enumerate(prepared, 1):
        sample = prepared_sample["sample"]
        source = source_by_id[sample.sample_id]
        target_path = _contained(args.data_root.resolve(), sample.mask_path)
        annotation_path = _contained(args.data_root.resolve(), sample.validity_annotation_path)
        if (
            _sha256_file(target_path) != source["mask_sha256"]
            or _sha256_file(annotation_path) != source["annotation_sha256"]
        ):
            raise RuntimeError(f"TRAIN GT provenance mismatch: {sample.sample_id}")
        words = tuple(word.bbox for word in train_task22_word_boxes(target_path, annotation_path))
        if len(words) != source["word_box_count"]:
            raise RuntimeError(f"TRAIN GT word count mismatch: {sample.sample_id}")

        stage_output = {}
        full_best: list[float] = []
        for name, _ in ABLATIONS:
            candidates = prepared_sample["stage_candidates"][name]
            boxes = tuple(value["bbox_original"] for value in candidates)
            best = [max((box_iou(box, word) for box in boxes), default=0.0) for word in words]
            aggregate_best[name].extend(best)
            candidate_counts[name].append(len(candidates))
            stage_output[name] = {
                "candidate_count": len(candidates),
                "candidate_bbox_sha256": _bbox_digest(candidates),
                "best_iou_by_gt_word": best,
            }
            if name == "6_full":
                full_best = best

        mask_020 = np.ascontiguousarray((prepared_sample["probability"] >= 0.20).astype(np.uint8))
        for word_index, (word, best) in enumerate(zip(words, full_best), 1):
            if best >= 0.50:
                continue
            category, details = _classify_full_failure(
                word=word, mask=mask_020,
                hypotheses=prepared_sample["hypotheses_by_threshold"][0.20],
                width=prepared_sample["width"], height=prepared_sample["height"],
            )
            failures.append({
                "sample_id": sample.sample_id, "gt_word_index": word_index,
                "best_full_iou": best, "category": category, **details,
            })
        samples_output.append({
            "sample_id": sample.sample_id, "gt_word_count": len(words), "stages": stage_output,
        })
        if image_index % 25 == 0 or image_index == 229:
            print(f"[oracle post-hoc GT] {image_index}/229", flush=True)

    ablation = {}
    for name, config in ABLATIONS:
        ablation[name] = {
            "configuration": {
                **config,
                "thresholds": list(config["thresholds"]),
                "bbox_variants": list(config["bbox_variants"]),
            },
            "oracle": _aggregate(aggregate_best[name]),
            "candidates_per_image": _quantiles(candidate_counts[name]),
        }
    if ablation["5_bbox_variants"] != ablation["6_full"]:
        raise RuntimeError("FULL cumulative identity check failed")
    full_metrics = ablation["6_full"]["oracle"]
    failure_counts = Counter(value["category"] for value in failures)
    if sum(failure_counts.values()) != full_metrics["failure_count_iou_0_50"]:
        raise RuntimeError("FULL failure accounting mismatch")
    failure_order = (
        "NO_FOREGROUND_EVEN_AT_0_20", "GEOMETRY_REMAP", "LINE_TOPOLOGY",
        "CANDIDATE_PRUNING_SPAN_LIMIT", "OTHER",
    )
    failure_breakdown = {
        name: {
            "count": failure_counts[name],
            "fraction_of_full_failures": failure_counts[name] / len(failures) if failures else 0.0,
        }
        for name in failure_order
    }
    recall = full_metrics["recall_iou_ge_0_5"]
    decision = (
        "PROPOSAL_RERANKING_GO" if recall >= 0.90
        else "BORDERLINE" if recall >= 0.85
        else "NO-GO"
    )
    report = {
        "schema_version": 1,
        "mode": MODE,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "table_5_accuracy": None,
        "scientific_boundary": {
            "execution_device": "cpu", "cuda_used": False,
            "segmentation_rerun": False, "test_loaded": False,
            "ocr_parseq": False, "training": False,
            "gt_used_in_candidate_generation": False,
            "threshold_or_parameter_selection": False,
        },
        "provenance": {
            "probability_manifest": str(probability_manifest),
            "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
            "reference_mask_manifest_sha256": EXPECTED_MASK_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": _sha256_file(train_manifest),
            "v1_calibration_sha256": EXPECTED_V1_SHA256,
        },
        "proposal_protocol": {
            "thresholds": list(THRESHOLDS),
            "canonical_deduplication_precedence": list(CANONICAL_THRESHOLD_ORDER),
            "max_contiguous_components_full": MAX_COMPONENTS_FULL,
            "max_span_width_over_median_component_height": MAX_WIDTH_OVER_MEDIAN_HEIGHT,
            "line_geometry": vars(frozen_line_params()),
            "cross_line_geometry": vars(frozen_cross_line_params()),
            "cross_line_rule": "union line IDs linked by frozen-v1 compatible raw-component pairs",
            "bbox_variants": list(BBOX_VARIANTS),
            "adaptive_padding": "max(6 original pixels, round(0.15 * median remapped raw-component height))",
            "deduplication_iou_greater_than_or_equal": DEDUPLICATION_IOU,
            "deduplication_tie_rule": "keep first under frozen .50,.40,.30,.20 precedence; no GT score",
            "top10_applied_to_oracle_pool": False,
            "stage_5_and_full_are_identical_by_cumulative_definition": True,
        },
        "ablation": ablation,
        "full_oracle": full_metrics,
        "full_candidate_counts": ablation["6_full"]["candidates_per_image"],
        "full_failure_breakdown": failure_breakdown,
        "failure_records": failures,
        "samples": samples_output,
        "decision": decision,
        "source_sha256": {
            str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
            str(Path(__file__).resolve().parents[1] / "localize/multihypothesis_word_proposals.py"):
                _sha256_file(Path(__file__).resolve().parents[1] / "localize/multihypothesis_word_proposals.py"),
        },
    }
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _publish(output, content)
    print(json.dumps({
        "ablation": ablation, "full_failure_breakdown": failure_breakdown,
        "decision": decision,
    }, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {_sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
