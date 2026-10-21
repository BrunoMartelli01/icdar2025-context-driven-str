"""Evaluate a trained Total-Text practical target with native-128 contours + PARSeq.

PRACTICAL_APPROXIMATION diagnostic on official TEST. The predicted binary mask
is contoured directly at 128x128; contour boxes are remapped to RGB and cropped.
Ground truth is loaded only after proposals have been generated and frozen.
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

import numpy as np
from PIL import Image
import torch

from localize.native_repository_contours import remap_native_box
from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_scene_text_v1_parseq import (
    ARCHITECTURE, EXPECTED_COUNTS, _preprocess, _verify_training_run,
)
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, validate_checkpoint


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--localization-config", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    return file_sha256(path.resolve())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sample_key(sample_id: str) -> str:
    return sample_id.replace(":", "__").replace("/", "_")


def _localization_params(path: Path) -> RepositoryContourParams:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("variant") not in {
        "repository-contours-native128-v1",
        "repository-contours-native128-no-area-v1",
    } or cfg.get("reproduction_mode") != "approximate":
        raise ValueError("expected repository-contours native128 diagnostic config")
    expected_filter = cfg.get("filter")
    if expected_filter not in {
        "unpadded_bbox_area_native_pixels_squared",
        "none",
    }:
        raise ValueError("unsupported contour area-filter semantics")
    loc = cfg.get("parameters", {})
    params = RepositoryContourParams(
        min_area=int(loc["min_area"]), pad=int(loc["pad"]), nmax=int(loc["nmax"])
    )
    expected_min_area = 0 if expected_filter == "none" else 150
    if (params.min_area, params.pad, params.nmax) != (expected_min_area, 6, 10):
        raise ValueError("This diagnostic requires the configured area policy, pad=6, top10")
    return params


def _gt(annotation_path: Path, image_wh: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_wh
    instances = parse_totaltext_annotation(annotation_path)
    records = []
    for item in instances:
        if item.ignore:
            continue
        x1, y1, x2, y2 = item.bbox_xyxy
        records.append({
            "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
            "text": item.transcription,
            "granularity": "word",
        })
    return records


def _verify_shard(output: Path, path: Path, protocol_sha: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"shard protocol mismatch: {path}")
    for record in [value["native_mask"], *value["proposals"]]:
        artifact = output / record["file"]
        if not artifact.is_file() or _sha(artifact) != record["sha256"]:
            raise ValueError(f"missing/corrupt artifact: {artifact}")
    return value


def main() -> int:
    args = _arguments()
    if args.parseq_batch_size < 1:
        raise ValueError("--parseq-batch-size must be positive")
    if _sha(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint SHA256 mismatch")
    if _sha(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq runtime archive SHA256 mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"

    test_manifest_path = args.test_manifest.resolve()
    test_manifest = load_segmentation_manifest(test_manifest_path, require_checksums=True)
    if test_manifest.dataset != "total_text" or test_manifest.split_role != "test" or len(test_manifest.samples) != EXPECTED_COUNTS["total_text"]:
        raise ValueError("Expected the qualified official Total-Text TEST-300 manifest")
    config_path = args.segmentation_config.resolve()
    config = load_segmentation_config(config_path)
    if (config.architecture.identity != ARCHITECTURE
        or config.run.variant_class != "PRACTICAL_APPROXIMATION"
        or config.data.train.role != "train"
        or config.data.train.expected_sample_count != 1255):
        raise ValueError("Training config is not a verified Total-Text practical target config")
    checkpoint = args.checkpoint.resolve()
    run_path = args.training_run_manifest.resolve()
    run = _verify_training_run(
        run_path, checkpoint, expected_epochs=int(config.training.epochs)
    )
    train_record = run.get("datasets", {}).get("train", {})
    resolved_config = run_path.parent / run.get("resolved_config", {}).get("path", "resolved_config.yaml")
    if (train_record.get("dataset") != "total_text"
        or train_record.get("split_role") != "train"
        or train_record.get("sample_count") != 1255
        or train_record.get("manifest_sha256") != config.data.train.manifest_sha256
        or not resolved_config.is_file()
        or _sha(resolved_config) != run.get("resolved_config", {}).get("sha256")):
        raise ValueError("Training run provenance does not match the frozen Total-Text v1 config")
    localizer_config = args.localization_config.resolve()
    params = _localization_params(localizer_config)

    protocol = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "Total-Text TEST word recognition diagnostic; not a paper Table 5 result",
        "dataset": "total_text",
        "split": test_manifest.split,
        "images": EXPECTED_COUNTS["total_text"],
        "ground_truth_granularity": "word",
        "segmentation_training_run_manifest_sha256": _sha(run_path),
        "segmentation_checkpoint": str(checkpoint),
        "segmentation_checkpoint_sha256": run["checkpoint"]["sha256"],
        "segmentation_config": str(config_path),
        "segmentation_config_sha256": _sha(config_path),
        "training_manifest_sha256": train_record["manifest_sha256"],
        "test_manifest_sha256": _sha(test_manifest_path),
        "localization_config": str(localizer_config),
        "localization_config_sha256": _sha(localizer_config),
        "localization": {
            "method": "cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)",
            "mask": "sigmoid >= 0.5 at native 128x128; no resize",
            "min_bbox_area_native": None if params.min_area == 0 else params.min_area,
            "area_filter": "disabled" if params.min_area == 0 else "unpadded_bbox_area_native_pixels_squared",
            "padding_native": params.pad,
            "sort": "clipped padded bbox area descending, stable contour order",
            "max_proposals_per_image": params.nmax,
            "crop": "remapped padded bbox from original RGB",
        },
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "device": args.device,
        "ground_truth_use": "loaded after mask localization and crops are frozen; post-hoc evaluation only",
        "table_5_claim": False,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing output directory belongs to a different protocol")
    else:
        _atomic_json(protocol_path, protocol)
    protocol_sha = _sha(protocol_path)
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        if report.get("protocol") != protocol:
            raise ValueError("Existing completed results belong to a different protocol")
        print(json.dumps(report["summary"], indent=2), flush=True)
        print("[reuse]", result_path, _sha(result_path), flush=True)
        return 0

    params_model = config
    model = _model_from_config(params_model)
    validate_checkpoint(checkpoint, model, run["checkpoint"]["sha256"], ARCHITECTURE)
    device = torch.device(args.device)
    model.eval().to(device)
    shards = []
    data_root = args.data_root.resolve()
    try:
        for index, sample in enumerate(test_manifest.samples, 1):
            key = _sample_key(sample.sample_id)
            shard_path = output / "samples" / f"{key}.json"
            if shard_path.exists():
                shards.append(_verify_shard(output, shard_path, protocol_sha))
                continue
            image_path = data_root / sample.image_path
            if sample.image_sha256 and _sha(image_path) != sample.image_sha256:
                raise ValueError(f"image checksum mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                rgb = opened.convert("RGB")
            image = np.asarray(rgb)
            inputs = _preprocess(rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(inputs)
            if tuple(logits.shape) != (1, 1, 128, 128):
                raise RuntimeError(f"unexpected logits shape: {list(logits.shape)}")
            native = np.ascontiguousarray((logits.sigmoid()[0, 0] >= 0.5).byte().cpu().numpy())
            contour_report = audit_repository_contours(native.astype(np.uint8), params)
            mask_relative = Path("native_masks") / f"{key}.png"
            mask_file = output / mask_relative
            mask_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(native * 255, mode="L").save(mask_file, optimize=True)
            proposals = []
            crop_dir = output / "crops" / key
            crop_dir.mkdir(parents=True, exist_ok=True)
            height, width = image.shape[:2]
            for item in contour_report["retained"]:
                bbox_native = item["bbox"]
                padded_native = item["padded_bbox"]
                bbox = remap_native_box(bbox_native, (height, width))
                padded = remap_native_box(padded_native, (height, width))
                rank = int(item["rank"])
                crop_relative = Path("crops") / key / f"{rank:02d}.png"
                crop_file = output / crop_relative
                x1, y1, x2, y2 = padded
                Image.fromarray(image[y1:y2, x1:x2]).save(crop_file, optimize=True)
                proposals.append({
                    "rank": rank,
                    "contour_index": int(item["contour_index"]),
                    "bbox_area_native": int(item["bbox_area"]),
                    "bbox_native": bbox_native,
                    "padded_bbox_native": padded_native,
                    "bbox": bbox,
                    "padded_bbox_original": padded,
                    "file": crop_relative.as_posix(),
                    "sha256": _sha(crop_file),
                })
            native_sha = sha256(native.tobytes()).hexdigest()
            gt = _gt(
                image_path.parent.parent / "annotations" / f"{sample.sample_id.partition(':')[2]}.txt",
                (width, height),
            )
            shard = {
                "schema_version": 1,
                "protocol_sha256": protocol_sha,
                "sample_id": sample.sample_id,
                "image_sha256": sample.image_sha256,
                "native_mask_array_sha256": native_sha,
                "native_mask": {"file": mask_relative.as_posix(), "sha256": _sha(mask_file)},
                "raw_contours": contour_report["raw_external_contours"],
                "candidates_before_cap": contour_report["candidates_before_cap"],
                "proposals": proposals,
                "rejected": contour_report["rejected"],
                "ground_truth": gt,
            }
            _atomic_json(shard_path, shard)
            shards.append(shard)
            if index % 10 == 0 or index == len(test_manifest.samples):
                print(f"[Total-Text FindContours] {index}/{len(test_manifest.samples)}", flush=True)
    finally:
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(shards) != EXPECTED_COUNTS["total_text"]:
        raise RuntimeError("Incomplete Total-Text TEST segmentation/localization")
    missing = []
    for shard in shards:
        key = _sample_key(shard["sample_id"])
        for proposal in shard["proposals"]:
            ocr_path = output / "ocr" / f"{key}__{proposal['rank']:02d}.json"
            crop_path = output / proposal["file"]
            if _sha(crop_path) != proposal["sha256"]:
                raise ValueError(f"crop checksum mismatch: {crop_path}")
            if ocr_path.exists():
                cached = json.loads(ocr_path.read_text(encoding="utf-8"))
                if cached["protocol_sha256"] != protocol_sha or cached["crop_sha256"] != proposal["sha256"]:
                    raise ValueError(f"OCR cache provenance mismatch: {ocr_path}")
            else:
                missing.append((shard, proposal, crop_path, ocr_path))
    print(f"[PARSeq] proposals={sum(len(s['proposals']) for s in shards)} pending={len(missing)}", flush=True)
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
                batch = missing[start:start + args.parseq_batch_size]
                crops = []
                for _, _, path, _ in batch:
                    with Image.open(path) as opened:
                        crops.append(np.ascontiguousarray(opened.convert("RGB")))
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(batch):
                    raise RuntimeError("PARSeq batch length mismatch")
                for (shard, proposal, _, path), prediction in zip(batch, predictions):
                    _atomic_json(path, {
                        "schema_version": 1,
                        "protocol_sha256": protocol_sha,
                        "sample_id": shard["sample_id"],
                        "rank": proposal["rank"],
                        "crop_sha256": proposal["sha256"],
                        **prediction.to_dict(),
                    })
                print(f"[PARSeq] {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)
        finally:
            recognizer.release()

    evaluated = []
    raw_contours = retained = candidates_before_cap = 0
    for shard in shards:
        key = _sample_key(shard["sample_id"])
        predictions = []
        for proposal in shard["proposals"]:
            ocr = json.loads((output / "ocr" / f"{key}__{proposal['rank']:02d}.json").read_text(encoding="utf-8"))
            predictions.append({
                "rank": proposal["rank"],
                "bbox": proposal["padded_bbox_original"],
                "text": ocr["raw_text"],
                "confidence": ocr["sequence_confidence_native"],
            })
        evaluated.append({"sample_id": shard["sample_id"], "ground_truth": shard["ground_truth"], "predictions": predictions})
        raw_contours += int(shard["raw_contours"])
        retained += len(shard["proposals"])
        candidates_before_cap += int(shard["candidates_before_cap"])
    metrics = evaluate_images(evaluated, text_available=True)
    summary = {
        "dataset": "total_text",
        "images": len(shards),
        "localizer": "cv2.findContours on native 128x128 binary mask",
        "raw_contours": raw_contours,
        "candidates_before_cap": candidates_before_cap,
        "retained_crops": retained,
        "images_over_top10_before_cap": sum(s["candidates_before_cap"] > 10 for s in shards),
        **metrics,
    }
    report = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "protocol": protocol,
        "summary": summary,
        "sample_shards": [f"samples/{_sample_key(s['sample_id'])}.json" for s in shards],
        "outputs": evaluated,
    }
    _atomic_json(result_path, report)
    print(json.dumps(summary, indent=2), flush=True)
    print("JSON", result_path, "SHA256", _sha(result_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
