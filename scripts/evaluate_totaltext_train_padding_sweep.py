"""TRAIN-only native128 padding diagnostic with one frozen mask per image.

Phase ``masks`` exports checkpoint predictions; phase ``evaluate`` reuses
those masks, deduplicates identical crops across padding variants and runs
qualified PARSeq. Ground truth is loaded only after all OCR for an image.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from localize.native_repository_contours import remap_native_box
from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_scene_text_v1_parseq import ARCHITECTURE, _verify_training_run
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("masks", "evaluate"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, data: dict) -> None:
    _atomic_bytes(path, (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _key(sample_id: str) -> str:
    return sample_id.replace(":", "__").replace("/", "_")


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("experiment") != "totaltext-train-native-padding-sweep-v1"
        or config.get("classification") != "PRACTICAL_APPROXIMATION_DIAGNOSTIC"
        or config.get("split") != "train"
        or config.get("pads_native128") != [0, 2, 4, 6, 8, 10, 12]
        or config.get("mask_probability_threshold") != 0.5
        or config.get("min_bbox_area_native") != 0
        or config.get("maximum_crops_per_image") != 10
    ):
        raise ValueError("Padding sweep config differs from the predeclared TRAIN diagnostic")
    return config


def _context(args: argparse.Namespace) -> tuple[dict, tuple, dict, Path]:
    manifest_path = args.train_manifest.resolve()
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train" or len(manifest.samples) != 1255:
        raise ValueError("Only the qualified Total-Text TRAIN-1255 manifest is permitted")
    sweep_path = args.sweep_config.resolve()
    sweep = _read_config(sweep_path)
    config_path = args.segmentation_config.resolve()
    config = load_segmentation_config(config_path)
    if (
        config.architecture.identity != ARCHITECTURE
        or config.run.variant_class != "PRACTICAL_APPROXIMATION"
        or config.data.train.expected_sample_count != 1255
        or config.data.train.manifest_sha256 != file_sha256(manifest_path)
    ):
        raise ValueError("Segmentation config/manifest does not match trained Total-Text v1")
    checkpoint = args.checkpoint.resolve()
    run_path = args.training_run_manifest.resolve()
    run = _verify_training_run(run_path, checkpoint)
    trained = run.get("datasets", {}).get("train", {})
    if trained.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Selected checkpoint was trained on a different TRAIN manifest")
    protocol = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "split": "Total-Text TRAIN",
        "images": 1255,
        "train_manifest_sha256": file_sha256(manifest_path),
        "segmentation_config_sha256": file_sha256(config_path),
        "sweep_config": sweep,
        "sweep_config_sha256": file_sha256(sweep_path),
        "training_run_manifest_sha256": file_sha256(run_path),
        "segmentation_checkpoint": str(checkpoint),
        "segmentation_checkpoint_sha256": run["checkpoint"]["sha256"],
        "data_root": str(args.data_root.resolve()),
        "parseq_source": str(args.parseq_source.resolve()),
        "parseq_source_archive": str(args.parseq_source_archive.resolve()),
        "parseq_checkpoint": str(args.parseq_checkpoint.resolve()),
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "segmentation_and_parseq_device": args.device,
        "ground_truth_use": "post-hoc evaluation only, after per-image proposals and OCR are frozen",
        "table_5_claim": False,
    }
    output = args.output_root.resolve()
    repository = Path.cwd().resolve()
    data_root = args.data_root.resolve()
    if output in (repository, data_root) or repository in output.parents or data_root in output.parents:
        raise ValueError("Output root must not overwrite dataset or repository")
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    _atomic_json(protocol_path, protocol)
    return protocol, manifest.samples, run, output


def proposals_for_mask(mask: np.ndarray, image_hw: tuple[int, int], pads: list[int]) -> dict[int, dict]:
    """Apply the exact same native mask to every pad, with no area filter."""
    if mask.shape != (128, 128) or not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("Expected a binary native128 mask")
    output = {}
    for pad in pads:
        audit = audit_repository_contours(mask, RepositoryContourParams(0, pad, 10))
        output[pad] = {
            "raw_contours": audit["raw_external_contours"],
            "candidates_before_cap": audit["candidates_before_cap"],
            "proposals": [
                {
                    "rank": item["rank"],
                    "contour_index": item["contour_index"],
                    "bbox_native": item["bbox"],
                    "padded_bbox_native": item["padded_bbox"],
                    "bbox_original": remap_native_box(item["padded_bbox"], image_hw),
                }
                for item in audit["retained"]
            ],
        }
    return output


def _read_mask(output: Path, record: dict) -> np.ndarray:
    path = output / record["file"]
    if file_sha256(path) != record["file_sha256"]:
        raise ValueError(f"Cached native mask checksum mismatch: {path}")
    with Image.open(path) as opened:
        mask = np.asarray(opened.convert("L"))
    if mask.shape != (128, 128) or not np.isin(mask, (0, 255)).all():
        raise ValueError(f"Cached native mask is not binary 128x128: {path}")
    native = np.ascontiguousarray((mask > 0).astype(np.uint8))
    if sha256(native.tobytes()).hexdigest() != record["array_sha256"]:
        raise ValueError(f"Cached native mask array differs: {path}")
    return native


def _export_masks(args: argparse.Namespace, protocol: dict, samples: tuple, run: dict, output: Path) -> None:
    config = load_segmentation_config(args.segmentation_config.resolve())
    checkpoint = args.checkpoint.resolve()
    model = _model_from_config(config)
    validate_checkpoint(checkpoint, model, run["checkpoint"]["sha256"], ARCHITECTURE)
    device = torch.device(args.device)
    model.eval().to(device)
    records = []
    try:
        for index, sample in enumerate(samples, 1):
            key = _key(sample.sample_id)
            image_path = args.data_root.resolve() / sample.image_path
            if file_sha256(image_path) != sample.image_sha256:
                raise ValueError(f"TRAIN image checksum mismatch: {sample.sample_id}")
            record_path = output / "mask_shards" / f"{key}.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record["sample_id"] != sample.sample_id or record["checkpoint_sha256"] != run["checkpoint"]["sha256"]:
                    raise ValueError(f"Cached mask provenance mismatch: {sample.sample_id}")
                _read_mask(output, record)
            else:
                with Image.open(image_path) as opened:
                    rgb = opened.convert("RGB")
                inputs = _preprocess(rgb).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits = model(inputs)
                if tuple(logits.shape) != (1, 1, 128, 128):
                    raise RuntimeError(f"Unexpected segmentation logits shape: {tuple(logits.shape)}")
                native = np.ascontiguousarray((logits.sigmoid()[0, 0] >= 0.5).byte().cpu().numpy())
                stream = BytesIO()
                Image.fromarray(native * 255, mode="L").save(stream, format="PNG", optimize=True)
                relative = f"native_masks/{key}.png"
                mask_path = output / relative
                _atomic_bytes(mask_path, stream.getvalue())
                record = {
                    "sample_id": sample.sample_id,
                    "image_sha256": sample.image_sha256,
                    "checkpoint_sha256": run["checkpoint"]["sha256"],
                    "file": relative,
                    "file_sha256": file_sha256(mask_path),
                    "array_sha256": sha256(native.tobytes()).hexdigest(),
                }
                _atomic_json(record_path, record)
            records.append(record)
            if index % 25 == 0 or index == len(samples):
                print(f"[TRAIN native masks] {index}/{len(samples)}", flush=True)
    finally:
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _atomic_json(output / "mask_manifest.json", {
        "schema_version": 1,
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "images": len(records),
        "samples": records,
    })
    print("MASK_MANIFEST", output / "mask_manifest.json", file_sha256(output / "mask_manifest.json"), flush=True)


def _gt(image_path: Path, image_wh: tuple[int, int]) -> list[dict]:
    width, height = image_wh
    annotation = image_path.parent.parent / "annotations" / f"{image_path.stem}.txt"
    records = []
    for item in parse_totaltext_annotation(annotation):
        if item.ignore:
            continue
        x1, y1, x2, y2 = item.bbox_xyxy
        records.append({
            "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
            "text": item.transcription,
            "granularity": "word",
        })
    return records


def _evaluate(args: argparse.Namespace, protocol: dict, samples: tuple, output: Path) -> None:
    manifest_path = output / "mask_manifest.json"
    cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cached["images"] != 1255 or cached["protocol_sha256"] != file_sha256(output / "protocol.json"):
        raise ValueError("Native mask cache is incomplete or belongs to a different protocol")
    by_id = {record["sample_id"]: record for record in cached["samples"]}
    if len(by_id) != 1255:
        raise ValueError("Native mask cache contains duplicate/missing sample IDs")
    if file_sha256(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint checksum mismatch")
    if file_sha256(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq source archive checksum mismatch")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    pads = protocol["sweep_config"]["pads_native128"]
    recognizer = None
    rows_by_pad = {pad: [] for pad in pads}
    counts = {pad: {"raw_contours": 0, "candidates_before_cap": 0, "images_over_top10": 0} for pad in pads}
    try:
        for index, sample in enumerate(samples, 1):
            record = by_id[sample.sample_id]
            native = _read_mask(output, record)
            image_path = args.data_root.resolve() / sample.image_path
            if file_sha256(image_path) != sample.image_sha256:
                raise ValueError(f"TRAIN image checksum mismatch: {sample.sample_id}")
            shard_path = output / "ocr_shards" / f"{_key(sample.sample_id)}.json"
            if shard_path.exists():
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
                if (shard["sample_id"] != sample.sample_id or shard["native_mask_array_sha256"] != record["array_sha256"]
                    or shard["protocol_sha256"] != file_sha256(output / "protocol.json")
                    or set(shard["variants"]) != {str(pad) for pad in pads}):
                    raise ValueError(f"OCR shard provenance mismatch: {sample.sample_id}")
            else:
                with Image.open(image_path) as opened:
                    rgb = np.asarray(opened.convert("RGB"))
                height, width = rgb.shape[:2]
                generated = proposals_for_mask(native, (height, width), pads)
                boxes = dict.fromkeys(
                    tuple(proposal["bbox_original"])
                    for pad in pads for proposal in generated[pad]["proposals"]
                )
                if recognizer is None and boxes:
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=args.parseq_source.resolve(),
                        checkpoint=args.parseq_checkpoint.resolve(),
                        device=args.device,
                        runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=args.parseq_source_archive.resolve(),
                    ))
                unique = list(boxes)
                for start in range(0, len(unique), args.batch_size):
                    selected = unique[start:start + args.batch_size]
                    crops = [np.ascontiguousarray(rgb[y1:y2, x1:x2]) for x1, y1, x2, y2 in selected]
                    predictions = recognizer.recognize_candidate_batch(crops)
                    if len(predictions) != len(selected):
                        raise RuntimeError("PARSeq returned the wrong batch length")
                    for box, prediction in zip(selected, predictions):
                        boxes[box] = prediction.to_dict()
                for pad in pads:
                    for proposal in generated[pad]["proposals"]:
                        prediction = boxes[tuple(proposal["bbox_original"])]
                        proposal["text"] = prediction["raw_text"]
                        proposal["confidence"] = prediction["sequence_confidence_native"]
                shard = {
                    "schema_version": 1,
                    "sample_id": sample.sample_id,
                    "protocol_sha256": file_sha256(output / "protocol.json"),
                    "image_sha256": sample.image_sha256,
                    "native_mask_array_sha256": record["array_sha256"],
                    "unique_ocr_crops": len(unique),
                    "variants": {str(pad): generated[pad] for pad in pads},
                }
                _atomic_json(shard_path, shard)
            # GT enters only after the complete OCR shard exists.
            with Image.open(image_path) as opened:
                image_wh = opened.size
            gt = _gt(image_path, image_wh)
            for pad in pads:
                variant = shard["variants"][str(pad)]
                rows_by_pad[pad].append({
                    "sample_id": sample.sample_id,
                    "ground_truth": gt,
                    "predictions": [
                        {"rank": p["rank"], "bbox": p["bbox_original"], "text": p["text"], "confidence": p["confidence"]}
                        for p in variant["proposals"]
                    ],
                })
                counts[pad]["raw_contours"] += variant["raw_contours"]
                counts[pad]["candidates_before_cap"] += variant["candidates_before_cap"]
                counts[pad]["images_over_top10"] += int(variant["candidates_before_cap"] > 10)
            if index % 25 == 0 or index == len(samples):
                print(f"[TRAIN padding OCR] {index}/{len(samples)}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    summaries = {}
    for pad in pads:
        metrics = evaluate_images(rows_by_pad[pad], text_available=True)
        if metrics["geometry"]["gt_instances"] != 9285:
            raise ValueError("Total-Text TRAIN GT population differs from qualified 9285 words")
        summaries[str(pad)] = {**counts[pad], **metrics}
    report = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "protocol": protocol,
        "mask_manifest_sha256": file_sha256(manifest_path),
        "sample_shards": [f"ocr_shards/{_key(sample.sample_id)}.json" for sample in samples],
        "summaries_by_pad_native128": summaries,
    }
    result_path = output / "results.json"
    _atomic_json(result_path, report)
    for pad in pads:
        geometry = summaries[str(pad)]["geometry"]
        ocr = summaries[str(pad)]["recognition"]
        print(
            f"pad={pad:2d} crops={geometry['retained_proposals']:5d} "
            f"IoU.5={geometry['iou_ge_0_50_one_to_one']:4d}/9285 "
            f"exact={ocr['exact_casefold_spatial_one_to_one']:4d}/9285 "
            f"fuzzy={ocr['rapidfuzz_casefold_ge_0_8_spatial_one_to_one']:4d}/9285",
            flush=True,
        )
    print("JSON", result_path, "SHA256", file_sha256(result_path), flush=True)


def main() -> int:
    args = _args()
    protocol, samples, run, output = _context(args)
    if args.phase == "masks":
        _export_masks(args, protocol, samples, run, output)
    else:
        _evaluate(args, protocol, samples, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
