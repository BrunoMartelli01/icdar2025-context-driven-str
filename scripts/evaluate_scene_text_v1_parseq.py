#!/usr/bin/env python3
"""Run frozen practical-remap-v1 and qualified PARSeq on one official TEST split.

This is an APPROXIMATE diagnostic, not a paper Table-5 result.  Ground truth is
loaded only after segmentation, localization and OCR outputs have been frozen in
restartable per-sample/per-proposal shards.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from pipeline.frozen_segmentation import backproject_binary_mask
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from recognition.scene_text_datasets import (
    SceneTextInstance,
    parse_icdar2015_annotation,
    parse_totaltext_annotation,
)
from recognition.scene_text_v1_evaluation import evaluate_images
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint


EXPECTED_COUNTS = {"ctw1500": 500, "icdar2015": 500, "total_text": 300}
ARCHITECTURE = "author-clarified-ag-unet-2048x1024-v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED_COUNTS), required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--segmentation-run-manifest", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--ctw-source-archive", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_atomic(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        stream.write(_canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _sha(path: Path) -> str:
    return file_sha256(path.resolve())


def _sample_key(sample_id: str) -> str:
    return sample_id.replace(":", "__").replace("/", "_")


def _selected_v1_parameters(calibration: dict[str, Any]) -> GroupedWordBlockParams:
    """Load the frozen TRAIN-only V1 selection and reject contaminated calibration."""

    guard = calibration.get("train_only_guard", {})
    boundary = calibration.get("scientific_boundary", {})
    if (
        guard.get("split_role") != "train"
        or guard.get("sample_count") != 229
        or guard.get("test_data_loaded") is not False
        or boundary.get("selection_used_test") is not False
        or boundary.get("selection_used_ocr") is not False
    ):
        raise ValueError("V1 calibration is not the frozen TRAIN-only selection")
    return GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])


def _verify_training_run(
    run_manifest_path: Path, checkpoint: Path, *, expected_epochs: int = 100
) -> dict[str, Any]:
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run.get("status") != "SUCCESS" or run.get("success") is not True:
        raise ValueError("segmentation run is not successful")
    if len(run.get("epochs", [])) != expected_epochs:
        raise ValueError(
            f"segmentation run does not contain all {expected_epochs} epochs"
        )
    recorded = Path(run["checkpoint_output_location"]).resolve()
    if recorded != checkpoint.resolve():
        raise ValueError(f"selected checkpoint mismatch: {recorded} != {checkpoint.resolve()}")
    if _sha(checkpoint) != run["checkpoint"]["sha256"]:
        raise ValueError("selected checkpoint SHA256 mismatch")
    return run


def _ctw_ground_truth(archive_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not archive_path.is_file():
        raise FileNotFoundError("CTW1500 source archive is required for geometry evaluation")
    with ZipFile(archive_path) as archive:
        raw = json.loads(archive.read("ctw1500/textdet_test.json"))
    output: dict[str, list[dict[str, Any]]] = {}
    for item in raw["data_list"]:
        sample_id = str(int(Path(str(item["img_path"])).stem))
        values = []
        for instance in item["instances"]:
            polygon = instance["polygon"]
            xs = [int(polygon[index]) for index in range(0, len(polygon), 2)]
            ys = [int(polygon[index]) for index in range(1, len(polygon), 2)]
            values.append({
                "bbox": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
                "text": None,
                "ignore": bool(instance.get("ignore", False)),
                "granularity": "text_line",
            })
        output[sample_id] = values
    return output


def _instance_record(instance: SceneTextInstance, width: int, height: int) -> dict[str, Any]:
    x1, y1, x2, y2 = instance.bbox_xyxy
    return {
        "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
        "text": instance.transcription,
        "ignore": instance.ignore,
        "granularity": instance.granularity,
        "orientation": instance.orientation,
    }


def _ground_truth(
    dataset: str,
    sample_id: str,
    image_path: Path,
    width: int,
    height: int,
    ctw: dict[str, list[dict[str, Any]]] | None,
) -> tuple[list[dict[str, Any]], int]:
    identifier = sample_id.partition(":")[2]
    if dataset == "ctw1500":
        if ctw is None or identifier not in ctw:
            raise ValueError(f"missing CTW1500 geometry for {sample_id}")
        records = []
        for value in ctw[identifier]:
            x1, y1, x2, y2 = value["bbox"]
            records.append({
                **value,
                "bbox": [
                    min(max(0, x1), width),
                    min(max(0, y1), height),
                    min(max(0, x2), width),
                    min(max(0, y2), height),
                ],
            })
    else:
        annotation = image_path.parent.parent / "annotations" / f"{identifier}.txt"
        if dataset == "icdar2015":
            instances = parse_icdar2015_annotation(annotation, image_size_wh=(width, height))
        else:
            instances = parse_totaltext_annotation(annotation)
        records = [_instance_record(value, width, height) for value in instances]
    return [value for value in records if not value["ignore"]], sum(
        bool(value["ignore"]) for value in records
    )


class _Segmenter:
    def __init__(
        self, config_path: Path, checkpoint: Path, checkpoint_sha256: str, device: str
    ) -> None:
        config = load_segmentation_config(config_path)
        checks = {
            "architecture.identity": ARCHITECTURE,
            "transforms.image_size": [256, 256],
            "transforms.mask_size": [128, 128],
            "transforms.normalization": "imagenet",
            "checkpoint_selection.probability_threshold": 0.5,
        }
        for dotted, expected in checks.items():
            current: Any = config
            for part in dotted.split("."):
                current = getattr(current, part)
            actual = list(current) if isinstance(expected, list) else current
            if actual != expected:
                raise ValueError(f"segmentation config mismatch at {dotted}: {actual!r}")
        self.device = torch.device(device)
        self.model = _model_from_config(config)
        self.checkpoint = validate_checkpoint(
            checkpoint, self.model, checkpoint_sha256, ARCHITECTURE
        )
        self.model = self.model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, image_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        inputs = _preprocess(Image.fromarray(image_rgb, mode="RGB")).unsqueeze(0).to(self.device)
        logits = self.model(inputs)
        if tuple(logits.shape) != (1, 1, 128, 128):
            raise RuntimeError(f"unexpected logits shape: {list(logits.shape)}")
        native = (logits.sigmoid()[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
        native = np.ascontiguousarray(native)
        projected = backproject_binary_mask(native, image_rgb.shape[:2])
        return projected, {
            "native_mask_sha256": sha256(native.tobytes()).hexdigest(),
            "projected_mask_sha256": sha256(projected.tobytes()).hexdigest(),
            "native_foreground_pixels": int(native.sum()),
            "projected_foreground_pixels": int(projected.sum()),
        }

    def release(self) -> None:
        self.model = None


def main() -> int:
    args = _args()
    if args.parseq_batch_size < 1:
        raise ValueError("--parseq-batch-size must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(report["summary"], indent=2), flush=True)
        print(f"[reuse] {result_path} SHA256 {_sha(result_path)}", flush=True)
        return 0

    manifest_path = args.test_manifest.resolve()
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if (
        manifest.dataset != args.dataset
        or manifest.split_role != "test"
        or len(manifest.samples) != EXPECTED_COUNTS[args.dataset]
    ):
        raise ValueError("test manifest does not match the declared dataset/split/count")
    calibration_path = args.v1_calibration.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    params = _selected_v1_parameters(calibration)
    run = _verify_training_run(
        args.segmentation_run_manifest.resolve(), args.segmentation_checkpoint.resolve()
    )
    protocol = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "table_5_claim": False,
        "dataset": args.dataset,
        "split": manifest.split,
        "expected_samples": EXPECTED_COUNTS[args.dataset],
        "segmentation_config_sha256": _sha(args.segmentation_config),
        "segmentation_checkpoint_sha256": run["checkpoint"]["sha256"],
        "segmentation_checkpoint_epoch": run["checkpoint"]["epoch"],
        "test_manifest_sha256": _sha(manifest_path),
        "v1_calibration_sha256": _sha(calibration_path),
        "v1_parameters": calibration["calibration"]["selected"]["parameters"],
        "parseq_source_archive_sha256": _sha(args.parseq_source_archive),
        "parseq_checkpoint_sha256": _sha(args.parseq_checkpoint),
        "device": args.device,
        "ground_truth_boundary": "post-hoc evaluation only",
        "recognition_normalization": "Unicode casefold only; punctuation retained",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing output protocol differs; refusing mixed resume")
    else:
        _write_atomic(protocol_path, protocol)
    protocol_sha = _sha(protocol_path)
    ctw = (
        _ctw_ground_truth(args.ctw_source_archive.resolve())
        if args.dataset == "ctw1500" and args.ctw_source_archive is not None
        else None
    )

    proposal_dir = output / "proposals"
    segmenter = _Segmenter(
        args.segmentation_config.resolve(),
        args.segmentation_checkpoint.resolve(),
        run["checkpoint"]["sha256"],
        args.device,
    )
    try:
        for index, sample in enumerate(manifest.samples, start=1):
            shard = proposal_dir / f"{_sample_key(sample.sample_id)}.json"
            if shard.exists():
                continue
            image_path = args.data_root.resolve() / sample.image_path
            if sample.image_sha256 is not None and _sha(image_path) != sample.image_sha256:
                raise ValueError(f"image SHA256 mismatch for {sample.sample_id}")
            with Image.open(image_path) as opened:
                image = np.asarray(opened.convert("RGB"))
            mask, segmentation = segmenter(image)
            grouped = group_word_blocks(mask, image, params)
            ground_truth, ignored = _ground_truth(
                args.dataset,
                sample.sample_id,
                image_path,
                image.shape[1],
                image.shape[0],
                ctw,
            )
            _write_atomic(shard, {
                "schema_version": 1,
                "protocol_sha256": protocol_sha,
                "sample_id": sample.sample_id,
                "image": sample.image_path.as_posix(),
                "image_sha256": sample.image_sha256,
                "image_shape_hwc": list(image.shape),
                "segmentation": segmentation,
                "raw_component_count": len(grouped.raw_components),
                "groups_before_cap": len(grouped.groups_before_cap),
                "ignored_gt_instances": ignored,
                "ground_truth": ground_truth,
                "proposals": [
                    {
                        "rank": block.rank,
                        "bbox": list(block.padded_bbox),
                        "raw_component_ids": list(block.raw_component_ids),
                        "foreground_area": block.foreground_area,
                    }
                    for block in grouped.retained
                ],
            })
            if index % 10 == 0 or index == len(manifest.samples):
                print(f"[{args.dataset} segmentation+V1] {index}/{len(manifest.samples)}", flush=True)
    finally:
        segmenter.release()
        del segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shards = []
    missing: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    ocr_dir = output / "ocr"
    for sample in manifest.samples:
        shard_path = proposal_dir / f"{_sample_key(sample.sample_id)}.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if shard.get("protocol_sha256") != protocol_sha:
            raise ValueError("proposal shard protocol mismatch")
        shards.append(shard)
        for proposal in shard["proposals"]:
            ocr_path = ocr_dir / f"{_sample_key(sample.sample_id)}__{proposal['rank']:02d}.json"
            if not ocr_path.exists():
                missing.append((shard, proposal, ocr_path))

    if missing:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source.resolve(),
            checkpoint=args.parseq_checkpoint.resolve(),
            device=args.device,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive.resolve(),
        ))
        try:
            for start in range(0, len(missing), args.parseq_batch_size):
                batch_rows = missing[start : start + args.parseq_batch_size]
                crops = []
                for shard, proposal, _ in batch_rows:
                    image_path = args.data_root.resolve() / Path(shard["image"])
                    with Image.open(image_path) as opened:
                        image = np.asarray(opened.convert("RGB"))
                    x1, y1, x2, y2 = proposal["bbox"]
                    crops.append(np.ascontiguousarray(image[y1:y2, x1:x2]))
                results = recognizer.recognize_candidate_batch(crops)
                for (shard, proposal, ocr_path), result in zip(batch_rows, results):
                    _write_atomic(ocr_path, {
                        "schema_version": 1,
                        "protocol_sha256": protocol_sha,
                        "sample_id": shard["sample_id"],
                        "rank": proposal["rank"],
                        **result.to_dict(),
                    })
                completed = min(start + len(batch_rows), len(missing))
                print(f"[{args.dataset} PARSeq] {completed}/{len(missing)}", flush=True)
        finally:
            recognizer.release()

    rows = []
    per_sample = []
    for shard in shards:
        predictions = []
        for proposal in shard["proposals"]:
            ocr_path = ocr_dir / f"{_sample_key(shard['sample_id'])}__{proposal['rank']:02d}.json"
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            predictions.append({
                "rank": proposal["rank"],
                "bbox": proposal["bbox"],
                "text": ocr["raw_text"],
                "confidence": ocr["sequence_confidence_native"],
            })
        row = {
            "sample_id": shard["sample_id"],
            "ground_truth": shard["ground_truth"],
            "predictions": predictions,
        }
        rows.append(row)
        per_sample.append({
            "sample_id": shard["sample_id"],
            "gt_instances": len(shard["ground_truth"]),
            "ignored_gt_instances": shard["ignored_gt_instances"],
            "raw_components": shard["raw_component_count"],
            "groups_before_cap": shard["groups_before_cap"],
            "retained_proposals": len(predictions),
            "predictions": predictions,
        })
    summary = evaluate_images(rows, text_available=args.dataset != "ctw1500")
    report = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "official-test V1+PARSeq diagnostic; not paper Table 5",
        "protocol": protocol,
        "summary": {
            "images": len(rows),
            "dataset": args.dataset,
            "segmentation_model_run": args.segmentation_run_manifest.parent.name,
            **summary,
        },
        "per_sample": per_sample,
    }
    _write_atomic(result_path, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"JSON {result_path} SHA256 {_sha(result_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
