#!/usr/bin/env python3
"""Evaluate train-frozen practical-remap-v1 on test geometry and first-ten OCR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.practical_remap_protocol import _box_iou, evaluate_test_masks
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_SOURCE_TYPES,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = np.asarray(opened.convert("L"))
    if not set(np.unique(image).tolist()).issubset({0, 255}):
        raise ValueError("Trace mask is not binary")
    return (image > 0).astype(np.uint8)


def _metrics(text: str | None, gt: str) -> dict[str, Any] | None:
    if text is None:
        return None
    from rapidfuzz.fuzz import ratio
    return {
        "casefold_exact": text.casefold() == gt.casefold(),
        "rapidfuzz_ratio_casefold": ratio(text.casefold(), gt.casefold()) / 100.0,
    }


def _best_group(groups: list[dict[str, Any]], gt_box: tuple[int, int, int, int]) -> dict[str, Any] | None:
    if not groups:
        return None
    return max(
        groups,
        key=lambda item: (
            _box_iou(item["padded_bbox"], gt_box),
            -item["rank"],
        ),
    )


def _ocr_aggregate(per_word: list[dict[str, Any]], method: str) -> dict[str, int]:
    values = [word["ocr_comparison"][method] for word in per_word]
    metrics = [value["text_metrics"] for value in values if value["text_metrics"] is not None]
    return {
        "denominator_gt_words": len(values),
        "available_predictions": len(metrics),
        "casefold_exact": sum(value["casefold_exact"] for value in metrics),
        "rapidfuzz_casefold_ge_0_8": sum(value["rapidfuzz_ratio_casefold"] >= 0.8 for value in metrics),
    }


def _write_csv(per_word: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for word in per_word:
        row = {
            "sample_id": word["sample_id"],
            "gt_index": word["gt_index"],
            "gt_text": word["gt_text"],
        }
        for method in ("ALG1", "PRACTICAL_REMAP", "GT_BOX", "PRED_COMPONENT_UNION"):
            value = word["ocr_comparison"][method]
            row["{}_text".format(method)] = value["text"]
            row["{}_iou".format(method)] = value["bbox_iou"]
            metrics = value["text_metrics"]
            row["{}_casefold_exact".format(method)] = None if metrics is None else metrics["casefold_exact"]
            row["{}_rapidfuzz_casefold".format(method)] = None if metrics is None else metrics["rapidfuzz_ratio_casefold"]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


def selected_train_parameters(calibration: dict[str, Any]) -> GroupedWordBlockParams:
    """Fail closed unless selection provenance is exclusively train-229."""

    guard = calibration.get("train_only_guard", {})
    boundary = calibration.get("scientific_boundary", {})
    if (
        guard.get("split_role") != "train"
        or guard.get("sample_count") != 229
        or guard.get("test_data_loaded") is not False
        or boundary.get("selection_used_test") is not False
        or boundary.get("selection_used_ocr") is not False
    ):
        raise ValueError("Calibration is contaminated by non-train selection data")
    return GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trace-index", type=Path, required=True)
    parser.add_argument("--train-calibration", type=Path, required=True)
    parser.add_argument("--prior-topology-report", type=Path, required=True)
    parser.add_argument("--segmentation-evaluation-config", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-type", choices=PARSEQ_SOURCE_TYPES, default=PARSEQ_ARCHIVE_SOURCE)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output, csv_output = args.output.resolve(), args.csv_output.resolve()
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite frozen-test practical report")
    calibration = json.loads(args.train_calibration.resolve().read_text(encoding="utf-8"))
    selected = calibration["calibration"]["selected"]["parameters"]
    params = selected_train_parameters(calibration)
    samples, dataset = load_icdar13_recognition_samples(args.test_manifest, args.data_root)
    if len(samples) != 233:
        raise ValueError("Frozen test requires all 233 Task 1 samples")
    index = json.loads(args.trace_index.resolve().read_text(encoding="utf-8"))
    index_by_id = {str(row["sample_id"]): row for row in index["samples"]}
    if not index_by_id:
        raise ValueError("Trace index must contain at least one frozen test output")
    masks = []
    trace_hash_checks = 0
    trace_mask_mismatches: list[dict[str, Any]] = []
    segmenter = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            evaluation_config=args.segmentation_evaluation_config.resolve(),
            checkpoint=args.segmentation_checkpoint.resolve(),
            device=args.device,
        )
    )
    try:
        for sample in samples:
            with Image.open(sample.image_path) as opened:
                image = np.asarray(opened.convert("RGB"))
            segmentation = segmenter(image)
            mask = segmentation["mask"]
            if sample.sample_id in index_by_id:
                trace = json.loads(Path(index_by_id[sample.sample_id]["result"]).read_text(encoding="utf-8"))
                if segmentation["metadata"]["output_mask_sha256"] != trace["segmentation"]["inference"]["output_mask_sha256"]:
                    recorded = _load_binary(Path(trace["segmentation"]["mask_artifact"]["path"]))
                    trace_mask_mismatches.append(
                        {
                            "sample_id": sample.sample_id,
                            "different_projected_pixels": int((mask != recorded).sum()),
                            "fresh_foreground_pixels": int(mask.sum()),
                            "recorded_foreground_pixels": int(recorded.sum()),
                        }
                    )
                trace_hash_checks += 1
            masks.append(mask)
    finally:
        segmentation_release = segmenter.release()
    geometry = evaluate_test_masks(samples, masks, params)

    prior = json.loads(args.prior_topology_report.resolve().read_text(encoding="utf-8"))
    prior_by_key = {
        (str(word["sample_id"]), int(word["gt_index"])): word
        for word in prior["phase_3_ocr"]["per_word"]
    }
    if len(prior_by_key) != 66:
        raise ValueError("Prior frozen first-ten topology report must contain 66 words")
    recognizer = ParseqApproxRecognizer(
        ParseqApproxConfig(
            runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint,
            device=args.device,
            runtime_source_type=args.parseq_source_type,
            runtime_source_archive=args.parseq_source_archive,
        )
    )
    per_word: list[dict[str, Any]] = []
    sample_graphs: dict[str, Any] = {}
    try:
        for sample, mask in zip(samples[:10], masks[:10]):
            with Image.open(sample.image_path) as opened:
                image = np.asarray(opened.convert("RGB"))
            practical = group_word_blocks(mask, image, params)
            practical_groups = []
            for block in practical.retained:
                parsed = recognizer.recognize(block.crop, input_role="T3_localized_crop").to_dict()
                practical_groups.append(
                    {
                        "rank": block.rank,
                        "raw_component_ids": list(block.raw_component_ids),
                        "foreground_area": block.foreground_area,
                        "bbox": list(block.bbox),
                        "padded_bbox": list(block.padded_bbox),
                        "parseq": {key: parsed.get(key) for key in ("raw_text", "token_ids", "confidence")},
                    }
                )
            sample_graphs[sample.sample_id] = {
                "raw_component_count": len(practical.raw_components),
                "graph_edges": [dict(edge) for edge in practical.graph_edges],
                "groups_before_cap": [
                    {
                        **dict(group),
                        "raw_component_ids": list(group["raw_component_ids"]),
                        "bbox": list(group["bbox"]),
                        "padded_bbox": list(group["padded_bbox"]),
                    }
                    for group in practical.groups_before_cap
                ],
                "retained_groups_with_parseq": practical_groups,
            }
            for word_index, word in enumerate(sample.words, start=1):
                prior_word = prior_by_key[(sample.sample_id, word_index)]
                gt_box = tuple(prior_word["gt_bbox_original_xyxy_exclusive"])
                chosen_group = _best_group(practical_groups, gt_box)
                practical_text = None if chosen_group is None else chosen_group["parseq"]["raw_text"]
                practical_iou = None if chosen_group is None else _box_iou(chosen_group["padded_bbox"], gt_box)
                alg1 = prior_word["ocr_diagnostic"]["ALG1"]["best_spatial_candidate"]
                gt_box_ocr = prior_word["ocr_diagnostic"]["GT_BOX"]
                union_ocr = prior_word["ocr_diagnostic"]["PRED_COMPONENT_UNION"]
                alg1_text = None if alg1 is None else alg1["parseq"]["raw_text"]
                union_text = None if union_ocr["parseq"] is None else union_ocr["parseq"]["raw_text"]
                gt_text = None if gt_box_ocr["parseq"] is None else gt_box_ocr["parseq"]["raw_text"]
                per_word.append(
                    {
                        "sample_id": sample.sample_id,
                        "gt_index": word_index,
                        "gt_text": word.transcription,
                        "ocr_comparison": {
                            "ALG1": {
                                "text": alg1_text,
                                "bbox_iou": None if alg1 is None else alg1["geometry"]["bbox_iou"],
                                "text_metrics": _metrics(alg1_text, word.transcription),
                            },
                            "PRACTICAL_REMAP": {
                                "text": practical_text,
                                "bbox_iou": practical_iou,
                                "group_rank": None if chosen_group is None else chosen_group["rank"],
                                "raw_component_ids": None if chosen_group is None else chosen_group["raw_component_ids"],
                                "text_metrics": _metrics(practical_text, word.transcription),
                            },
                            "GT_BOX": {
                                "text": gt_text,
                                "bbox_iou": 1.0,
                                "text_metrics": _metrics(gt_text, word.transcription),
                            },
                            "PRED_COMPONENT_UNION": {
                                "text": union_text,
                                "bbox_iou": None if union_ocr["geometry"] is None else union_ocr["geometry"]["bbox_iou"],
                                "text_metrics": _metrics(union_text, word.transcription),
                            },
                        },
                    }
                )
    finally:
        release = recognizer.release()
    examples = {}
    for name in ("Tiredness", "GREEK", "intelligent", "Harwich", "Ipswich"):
        word = next(item for item in per_word if item["gt_text"].casefold() == name.casefold())
        examples[name] = {**word, "graph": sample_graphs[word["sample_id"]]}
    report = {
        "schema_version": 1,
        "mode": "practical-remap-v1",
        "classification": "PRACTICAL_APPROXIMATION",
        "selection": {
            "train_calibration": str(args.train_calibration.resolve()),
            "train_calibration_sha256": _sha256(args.train_calibration.resolve()),
            "parameters": selected,
            "test_not_used_for_selection": True,
        },
        "frozen_test": {
            "manifest": dataset,
            "trace_index": str(args.trace_index.resolve()),
            "trace_index_sha256": _sha256(args.trace_index.resolve()),
            "sample_count": len(samples),
            "fresh_frozen_segmentation": {
                "checkpoint": dict(segmenter.checkpoint_record),
                "device": args.device,
                "hardware_deviation": "CPU test inference" if args.device == "cpu" else None,
                "recorded_trace_hash_checks": trace_hash_checks,
                "recorded_trace_hash_match_count": trace_hash_checks - len(trace_mask_mismatches),
                "recorded_trace_mask_mismatches": trace_mask_mismatches,
                "interpretation": "CPU float32 re-execution is a HARDWARE_DEVIATION; recorded GPU masks are retained as the paper-faithful baseline and are never overwritten.",
                "release": segmentation_release,
            },
            "geometry": geometry,
        },
        "first_ten_parseq": {
            "classification": "PRACTICAL_APPROXIMATION; CPU PARSeq is a hardware deviation",
            "prior_gt_box_and_union_report": str(args.prior_topology_report.resolve()),
            "prior_gt_box_and_union_report_sha256": _sha256(args.prior_topology_report.resolve()),
            "new_practical_parseq_only": True,
            "per_word_practical_group_assignment": "DIAGNOSTIC_ONLY: highest IoU fixed group for the GT word; GT never creates, merges, ranks, or chooses groups during practical inference.",
            "parseq_release": release,
            "aggregate": {
                method: _ocr_aggregate(per_word, method)
                for method in ("ALG1", "PRACTICAL_REMAP", "GT_BOX", "PRED_COMPONENT_UNION")
            },
            "per_word": per_word,
            "per_sample_graphs": sample_graphs,
            "required_examples": examples,
        },
        "scientific_boundary": {
            "accuracy": None,
            "table_5_claim": False,
            "prohibition": "No test geometry or OCR output feeds parameter selection; this is not paper-faithful Algorithm 1.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(report, output)
    _write_csv(per_word, csv_output)
    print(output)
    print(csv_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
