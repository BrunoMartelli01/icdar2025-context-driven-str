#!/usr/bin/env python3
"""Frozen test geometry and new-v2-only PARSeq crops for practical-remap-v2."""

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

from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.line_word_blocks import V2WordBlockParams, group_word_blocks_v2
from localize.practical_remap_protocol import _box_iou
from localize.practical_remap_v2_protocol import evaluate_v1_v2_test_geometry
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


def _v1_params(value: dict[str, Any]) -> GroupedWordBlockParams:
    guard, boundary = value.get("train_only_guard", {}), value.get("scientific_boundary", {})
    if (
        guard.get("split_role") != "train" or guard.get("sample_count") != 229
        or guard.get("test_data_loaded") is not False
        or boundary.get("selection_used_test") is not False
    ):
        raise ValueError("v1 parameters are not train-only frozen")
    return GroupedWordBlockParams(**value["calibration"]["selected"]["parameters"])


def _v2_params(value: dict[str, Any]) -> V2WordBlockParams:
    guard, boundary = value.get("train_only_guard", {}), value.get("scientific_boundary", {})
    if (
        guard.get("split_role") != "train" or guard.get("sample_count") != 229
        or guard.get("test_data_loaded") is not False
        or boundary.get("selection_used_test") is not False
        or boundary.get("selection_used_ocr") is not False
    ):
        raise ValueError("v2 parameters are not train-only frozen")
    return V2WordBlockParams(**value["calibration"]["selected"]["parameters"])


def _validate_v1_ocr_cache(
    report: dict[str, Any], *, expected_calibration_sha256: str
) -> dict[tuple[str, int], dict[str, Any]]:
    if (
        report.get("mode") != "practical-remap-v1"
        or report.get("classification") != "PRACTICAL_APPROXIMATION"
        or report.get("selection", {}).get("train_calibration_sha256")
        != expected_calibration_sha256
        or report.get("scientific_boundary", {}).get("table_5_claim") is not False
        or len(report.get("frozen_test", {}).get("geometry", {}).get("per_sample", [])) != 233
        or report.get("first_ten_parseq", {}).get("new_practical_parseq_only") is not True
    ):
        raise ValueError("v1 OCR cache provenance is incompatible with this v2 comparison")
    rows = report["first_ten_parseq"]["per_word"]
    by_key = {(str(value["sample_id"]), int(value["gt_index"])): value for value in rows}
    if len(rows) != 66 or len(by_key) != 66:
        raise ValueError("v1 OCR cache must contain 66 unique first-ten word records")
    required_methods = {"ALG1", "PRACTICAL_REMAP", "PRED_COMPONENT_UNION", "GT_BOX"}
    if any(set(value.get("ocr_comparison", {})) != required_methods for value in rows):
        raise ValueError("v1 OCR cache has an unexpected method schema")
    return by_key


def _metrics(text: str | None, gt: str) -> dict[str, Any] | None:
    if text is None:
        return None
    from rapidfuzz.fuzz import ratio
    return {
        "casefold_exact": text.casefold() == gt.casefold(),
        "rapidfuzz_ratio_casefold": ratio(text.casefold(), gt.casefold()) / 100.0,
    }


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, int]:
    values = [value["ocr_comparison"][method]["text_metrics"] for value in rows]
    present = [value for value in values if value is not None]
    return {
        "denominator_gt_words": len(values),
        "available_predictions": len(present),
        "casefold_exact": sum(value["casefold_exact"] for value in present),
        "rapidfuzz_casefold_ge_0_8": sum(value["rapidfuzz_ratio_casefold"] >= 0.8 for value in present),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["sample_id", "gt_index", "gt_text"]
    for method in ("ALG1", "PRACTICAL_REMAP_V1", "PRACTICAL_REMAP_V2", "PRED_COMPONENT_UNION", "GT_BOX"):
        fields.extend(("{}_text".format(method), "{}_bbox_iou".format(method), "{}_casefold_exact".format(method), "{}_rapidfuzz_casefold".format(method)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in rows:
            row = {"sample_id": value["sample_id"], "gt_index": value["gt_index"], "gt_text": value["gt_text"]}
            for method in ("ALG1", "PRACTICAL_REMAP_V1", "PRACTICAL_REMAP_V2", "PRED_COMPONENT_UNION", "GT_BOX"):
                result = value["ocr_comparison"][method]
                metrics = result["text_metrics"]
                row["{}_text".format(method)] = result["text"]
                row["{}_bbox_iou".format(method)] = result["bbox_iou"]
                row["{}_casefold_exact".format(method)] = None if metrics is None else metrics["casefold_exact"]
                row["{}_rapidfuzz_casefold".format(method)] = None if metrics is None else metrics["rapidfuzz_ratio_casefold"]
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--v2-calibration", type=Path, required=True)
    parser.add_argument("--v1-test-report", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
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
        raise FileExistsError("Refusing to overwrite frozen practical-remap-v2 test results")
    v1_calibration_path, v2_calibration_path = args.v1_calibration.resolve(), args.v2_calibration.resolve()
    v1_calibration_sha256 = _sha256(v1_calibration_path)
    v1_params = _v1_params(json.loads(v1_calibration_path.read_text(encoding="utf-8")))
    v2_params = _v2_params(json.loads(v2_calibration_path.read_text(encoding="utf-8")))
    v1_test_path = args.v1_test_report.resolve()
    v1_test = json.loads(v1_test_path.read_text(encoding="utf-8"))
    cached_by_key = _validate_v1_ocr_cache(
        v1_test, expected_calibration_sha256=v1_calibration_sha256
    )
    samples, dataset = load_icdar13_recognition_samples(args.test_manifest.resolve(), args.data_root.resolve())
    if len(samples) != 233:
        raise ValueError("test evaluation requires all 233 ICDAR13 Task 1 samples")
    print(
        "[setup] frozen-test samples=233 device={} checkpoint={}".format(
            args.device, args.segmentation_checkpoint.resolve()
        ),
        flush=True,
    )
    segmenter = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            evaluation_config=args.evaluation_config.resolve(),
            checkpoint=args.segmentation_checkpoint.resolve(), device=args.device,
        )
    )
    checkpoint = dict(segmenter.checkpoint_record)
    masks = []
    native_hashes = []
    try:
        for sample_index, sample in enumerate(samples, start=1):
            with Image.open(sample.image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            segmentation = segmenter(rgb)
            masks.append(segmentation["mask"])
            native_hashes.append({
                "sample_id": sample.sample_id,
                "native_mask_sha256": segmentation["metadata"]["native_mask_sha256"],
                "projected_mask_sha256": segmentation["metadata"]["output_mask_sha256"],
            })
            print(
                "[test masks] {}/{} sample_id={} projected_sha256={}".format(
                    sample_index, len(samples), sample.sample_id,
                    segmentation["metadata"]["output_mask_sha256"],
                ),
                flush=True,
            )
    finally:
        segmentation_release = segmenter.release()
    geometry = evaluate_v1_v2_test_geometry(samples, masks, v1_params, v2_params)
    print("[test geometry] {}".format(json.dumps(geometry["practical_remap_v2"], sort_keys=True)), flush=True)
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
        runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
        device=args.device, runtime_source_type=args.parseq_source_type,
        runtime_source_archive=args.parseq_source_archive.resolve(),
    ))
    rows: list[dict[str, Any]] = []
    graphs: dict[str, Any] = {}
    try:
        for sample_index, (sample, mask) in enumerate(zip(samples[:10], masks[:10]), start=1):
            with Image.open(sample.image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            v2 = group_word_blocks_v2(mask, rgb, v2_params)
            groups = []
            for block in v2.retained:
                parsed = recognizer.recognize(block.crop, input_role="T3_localized_crop").to_dict()
                groups.append({
                    "rank": block.rank, "raw_component_ids": list(block.raw_component_ids),
                    "atom_ids": list(block.atom_ids), "line_id": block.line_id,
                    "foreground_area": block.foreground_area, "bbox": list(block.bbox),
                    "padded_bbox": list(block.padded_bbox),
                    "parseq": {key: parsed.get(key) for key in ("raw_text", "token_ids", "confidence")},
                })
            print(
                "[PARSeq v2] {}/10 sample_id={} crops={}".format(
                    sample_index, sample.sample_id, len(groups)
                ),
                flush=True,
            )
            graphs[sample.sample_id] = {
                "raw_to_fragment_atoms": [
                    {"atom_id": value.atom_id, "raw_component_ids": list(value.raw_component_ids), "bbox": list(value.bbox)}
                    for value in v2.atoms
                ],
                "fragment_edges": [dict(value) for value in v2.fragment_edges],
                "line_formation": [
                    {**dict(value), "atom_ids": list(value["atom_ids"]), "raw_component_ids": list(value["raw_component_ids"]), "bbox": list(value["bbox"]), "gaps": [dict(gap) for gap in value["gaps"]]}
                    for value in v2.line_candidates
                ],
                "word_groups_before_cap": [
                    {**dict(value), "atom_ids": list(value["atom_ids"]), "raw_component_ids": list(value["raw_component_ids"]), "bbox": list(value["bbox"]), "padded_bbox": list(value["padded_bbox"])}
                    for value in v2.groups_before_cap
                ],
                "retained_v2_groups_with_new_parseq": groups,
            }
            for word_index, word in enumerate(sample.words, start=1):
                cached = cached_by_key[(sample.sample_id, word_index)]
                gt_box = tuple(cached.get("gt_bbox", ()))
                if len(gt_box) != 4:
                    # Earlier v1 output stores GT geometry in its diagnostic sources.
                    gt_box = tuple(word.bbox_xyxy[index] + (1 if index in (2, 3) else 0) for index in range(4))
                selected = max(groups, key=lambda value: _box_iou(value["padded_bbox"], gt_box), default=None)
                v2_text = None if selected is None else selected["parseq"]["raw_text"]
                v2_iou = None if selected is None else _box_iou(selected["padded_bbox"], gt_box)
                old = cached["ocr_comparison"]
                rows.append({
                    "sample_id": sample.sample_id, "gt_index": word_index, "gt_text": word.transcription,
                    "ocr_comparison": {
                        "ALG1": old["ALG1"],
                        "PRACTICAL_REMAP_V1": old["PRACTICAL_REMAP"],
                        "PRACTICAL_REMAP_V2": {
                            "text": v2_text, "bbox_iou": v2_iou,
                            "group_rank": None if selected is None else selected["rank"],
                            "raw_component_ids": None if selected is None else selected["raw_component_ids"],
                            "text_metrics": _metrics(v2_text, word.transcription),
                        },
                        "PRED_COMPONENT_UNION": old["PRED_COMPONENT_UNION"],
                        "GT_BOX": old["GT_BOX"],
                    },
                })
    finally:
        parseq_release = recognizer.release()
    examples = {}
    for name in ("Tiredness", "GREEK", "intelligent", "Harwich", "Ipswich"):
        value = next(row for row in rows if row["gt_text"].casefold() == name.casefold())
        examples[name] = {**value, "v2_raw_to_line_to_word": graphs[value["sample_id"]]}
    report = {
        "schema_version": 1,
        "mode": "practical-remap-v2",
        "classification": "PRACTICAL_APPROXIMATION",
        "selection": {
            "v1_calibration": str(v1_calibration_path), "v1_calibration_sha256": v1_calibration_sha256,
            "v2_calibration": str(v2_calibration_path), "v2_calibration_sha256": _sha256(v2_calibration_path),
            "v1_parameters": v1_params.__dict__.copy(), "v2_parameters": v2_params.__dict__.copy(),
            "test_not_used_for_selection": True,
        },
        "frozen_test": {
            "manifest": dataset, "sample_count": len(samples),
            "frozen_segmentation": {
                "checkpoint": checkpoint, "device": args.device,
                "hardware_deviation": "CPU test inference" if args.device == "cpu" else None,
                "release": segmentation_release, "mask_hashes": native_hashes,
            },
            "geometry": geometry,
        },
        "first_ten_parseq": {
            "classification": "PRACTICAL_APPROXIMATION; PARSeq is invoked only for new v2 crops",
            "v1_cached_report": str(v1_test_path), "v1_cached_report_sha256": _sha256(v1_test_path),
            "new_v2_parseq_only": True,
            "per_word_v2_group_assignment": "DIAGNOSTIC_ONLY: GT chooses the highest-IoU fixed v2 group only for comparison; it never affects v2 inference.",
            "parseq_release": parseq_release,
            "aggregate": {method: _aggregate(rows, method) for method in ("ALG1", "PRACTICAL_REMAP_V1", "PRACTICAL_REMAP_V2", "PRED_COMPONENT_UNION", "GT_BOX")},
            "per_word": rows, "per_sample_v2_graphs": graphs, "required_examples": examples,
        },
        "scientific_boundary": {
            "accuracy": None, "table_5_claim": False,
            "prohibition": "Test geometry and OCR are frozen after train-only configuration selection. This is not Algorithm 1 or a paper Table 5 result.",
        },
    }
    _write_atomic(report, output)
    _write_csv(rows, csv_output)
    print("[PARSeq summary] {}".format(json.dumps(report["first_ten_parseq"]["aggregate"], sort_keys=True)), flush=True)
    print(output, flush=True)
    print(csv_output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
