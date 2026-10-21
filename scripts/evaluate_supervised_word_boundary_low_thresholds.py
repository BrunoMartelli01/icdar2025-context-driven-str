#!/usr/bin/env python3
"""TRAIN-only low-threshold sweep from reconstructed OOF boundary probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold

from localize.grouped_word_blocks import GroupedWordBlockParams
from segmentation.data import load_segmentation_manifest
from scripts.evaluate_supervised_word_boundary_train import (
    EXPECTED_CACHE_SHA256,
    _evaluate_groups,
    _model_factories,
    _prepare_record,
    _sha256_file,
    _summary,
    _supervised_groups,
)


THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--supervised-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    cache_manifest = args.cache_manifest.resolve()
    if _sha256_file(cache_manifest) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache manifest SHA256 mismatch")
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if (
        manifest.split != "official-training-all-229-author-protocol"
        or manifest.split_role != "train"
        or len(manifest.samples) != 229
        or cache.get("sample_count") != 229
        or cache.get("test_data_loaded") is not False
    ):
        raise ValueError("low-threshold sweep permits only official TRAIN-229")

    supervised_report = json.loads(args.supervised_report.read_text(encoding="utf-8"))
    if (
        supervised_report["train_only"]["test_loaded"] is not False
        or supervised_report["train_only"]["ocr_called"] is not False
        or supervised_report["cache"]["manifest_sha256"] != EXPECTED_CACHE_SHA256
        or supervised_report["validation"]["method"] != "5-fold GroupKFold by image"
    ):
        raise ValueError("supervised report does not have the required TRAIN-only OOF provenance")
    model_name = supervised_report["selected"]["model"]
    if model_name not in _model_factories():
        raise ValueError(f"unsupported frozen model: {model_name}")

    v1_calibration = json.loads(args.v1_calibration.read_text(encoding="utf-8"))
    v1_params = GroupedWordBlockParams(
        **v1_calibration["calibration"]["selected"]["parameters"]
    )
    cache_by_id = {str(value["sample_id"]): value for value in cache["samples"]}
    if set(cache_by_id) != {sample.sample_id for sample in manifest.samples}:
        raise RuntimeError("cache/manifest sample identities differ")

    records = []
    for sample_index, sample in enumerate(manifest.samples, start=1):
        cached = cache_by_id[sample.sample_id]
        mask_path = (cache_manifest.parent / cached["native_mask_path"]).resolve()
        if _sha256_file(mask_path) != cached["native_mask_file_sha256"]:
            raise RuntimeError(f"cached mask file hash mismatch: sample {sample.sample_id}")
        with Image.open(mask_path) as opened:
            native_mask = (np.asarray(opened.convert("L")) > 0).astype(np.uint8)
        if (
            native_mask.shape != (128, 128)
            or hashlib.sha256(native_mask.tobytes()).hexdigest()
            != cached["native_mask_array_sha256"]
        ):
            raise RuntimeError(f"cached mask array mismatch: sample {sample.sample_id}")
        image_path = (args.data_root / sample.image_path).resolve()
        if _sha256_file(image_path) != cached["image_sha256"]:
            raise RuntimeError(f"source image hash mismatch: sample {sample.sample_id}")
        records.append(_prepare_record(
            sample.sample_id,
            native_mask,
            image_path,
            (args.data_root / sample.mask_path).resolve(),
            (args.data_root / sample.validity_annotation_path).resolve(),
            v1_params,
        ))
        if sample_index % 10 == 0 or sample_index == 229:
            print(f"[verified cache + prepared] {sample_index}/229", flush=True)

    features, labels, image_groups = [], [], []
    for image_index, record in enumerate(records):
        for pair in record.pairs:
            if pair["label"] is not None:
                features.append(pair["features"])
                labels.append(pair["label"])
                image_groups.append(image_index)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    image_groups_array = np.asarray(image_groups, dtype=np.int64)

    factory = _model_factories()[model_name][1]
    oof_probabilities = [[0.5] * len(record.pairs) for record in records]
    splitter = GroupKFold(n_splits=5)
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(np.arange(len(records)), groups=np.arange(len(records))), start=1
    ):
        train_images = set(train_indices.tolist())
        train_pair_indices = np.asarray([
            index for index, image_index in enumerate(image_groups_array)
            if image_index in train_images
        ])
        estimator = factory()
        estimator.fit(x[train_pair_indices], y[train_pair_indices])
        for image_index in validation_indices.tolist():
            matrix = np.asarray(
                [pair["features"] for pair in records[image_index].pairs],
                dtype=np.float64,
            )
            oof_probabilities[image_index] = (
                estimator.predict_proba(matrix)[:, 1].tolist() if len(matrix) else []
            )
        print(f"[OOF reconstruction] model={model_name} fold={fold_index}/5", flush=True)

    rows = []
    for threshold in THRESHOLDS:
        evaluation = _evaluate_groups(
            records,
            [
                _supervised_groups(record, probabilities, threshold)
                for record, probabilities in zip(records, oof_probabilities)
            ],
            evidence_attribute="supervised_word_evidence",
        )
        summary = _summary(evaluation)
        rows.append({
            "threshold": threshold,
            **summary,
            "wordness": evaluation["wordness"],
        })
        print(
            "[threshold] {:.2f} F1={:.6f} recall={:.6f} precision={:.6f} under={} over={}".format(
                threshold,
                summary["f1_iou_ge_0_50"],
                summary["recall_iou_ge_0_50"],
                summary["precision_iou_ge_0_50"],
                summary["undermerge"],
                summary["overmerge"],
            ),
            flush=True,
        )

    baseline = {
        "iou_ge_0_50": supervised_report["oof_v1"]["geometry"]["gt_with_proposal_iou_ge_0_50"],
        "f1_iou_ge_0_50": supervised_report["oof_v1"]["geometry"]["f1_proposal_level_iou_ge_0_50"],
        "undermerge": supervised_report["oof_v1"]["failure_breakdown"]["categories"]["GROUPING_UNDERMERGE"]["count"],
        "overmerge": supervised_report["oof_v1"]["failure_breakdown"]["categories"]["GROUPING_OVERMERGE"]["count"],
    }
    if baseline != {
        "iou_ge_0_50": 568,
        "f1_iou_ge_0_50": 0.553606237816764,
        "undermerge": 92,
        "overmerge": 36,
    }:
        raise RuntimeError("frozen v1 baseline mismatch")

    candidates = [
        row for row in rows
        if row["f1_iou_ge_0_50"] > baseline["f1_iou_ge_0_50"]
        and row["overmerge"] <= baseline["overmerge"]
    ]
    report = {
        "schema_version": 1,
        "mode": "supervised-word-boundary-low-threshold-sweep-v1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "train_only": {
            "images": 229,
            "gt_words": 849,
            "test_loaded": False,
            "ocr_called": False,
            "segmentation_executed": False,
            "device": "cpu",
        },
        "cache": {
            "manifest": str(cache_manifest),
            "manifest_sha256": EXPECTED_CACHE_SHA256,
            "verified_samples": 229,
        },
        "oof_reconstruction": {
            "model": model_name,
            "method": "5-fold GroupKFold by image",
            "source_report": str(args.supervised_report.resolve()),
            "source_report_sha256": _sha256_file(args.supervised_report.resolve()),
            "probabilities_persisted_before_this_run": False,
        },
        "thresholds_declared_before_execution": THRESHOLDS,
        "baseline_v1": baseline,
        "results": rows,
        "decision": {
            "label": "GO" if candidates else "NO-GO",
            "criterion": "F1 strictly above v1 without GROUPING_OVERMERGE above v1",
            "qualifying_thresholds": [row["threshold"] for row in candidates],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    print(output, flush=True)
    print(_sha256_file(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
