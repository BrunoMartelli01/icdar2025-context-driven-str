#!/usr/bin/env python3
"""Export paper-Algorithm-1 approximate crops for one scene-text TEST split.

This command performs U-Net inference and localization only; it does not run
PARSeq, context scoring, routing, or DeepSolo.  Ground truth is loaded only
after proposals have been generated, for a post-hoc geometry summary.
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
from omegaconf import OmegaConf
from PIL import Image
import torch

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam
from localize.native_algorithm1 import localize_native_algorithm1
from recognition.scene_text_datasets import (
    parse_icdar2015_annotation,
    parse_totaltext_annotation,
)
from recognition.scene_text_manifest import load_scene_text_manifest
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_scene_text_v1_parseq import (
    EXPECTED_COUNTS,
    _Segmenter,
    _preprocess,
    _verify_training_run,
)
from segmentation.data import load_segmentation_manifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED_COUNTS), required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--localization-config", type=Path, required=True)
    parser.add_argument("--segmentation-run-manifest", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-v1-run", type=Path, required=True)
    parser.add_argument("--ctw-recognition-manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sample_key(sample_id: str) -> str:
    return sample_id.replace(":", "__")


def _params(config_path: Path) -> BlockParam:
    config = OmegaConf.load(config_path)
    if (
        config.get("experiment_kind") != "localization_diagnostic"
        or config.get("reproduction_mode") != "approximate"
        or config.get("classification") != "APPROXIMATE_PAPER_ALGORITHM1"
    ):
        raise ValueError("invalid isolated approximate localization configuration")
    values = OmegaConf.select(config, "localization")
    if values is None:
        raise ValueError("segmentation config has no localization section")
    params = BlockParam(
        min_area=int(values.min_area),
        pad=int(values.pad),
        connectivity=int(values.connectivity),
        tie_break=APPROXIMATE_TIE_BREAK,
        nmax=int(values.nmax),
    )
    if (params.min_area, params.pad, params.connectivity, params.nmax) != (150, 6, 8, 10):
        raise ValueError(
            "new-dataset Algorithm1 protocol requires frozen approximate parameters "
            "Amin=150, pad=6, connectivity=8, nmax=10"
        )
    return params


def _ctw_gt(manifest_path: Path, data_root: Path) -> tuple[dict[str, list[dict]], dict]:
    samples, qualification = load_scene_text_manifest(manifest_path, data_root)
    if qualification["dataset"] != "ctw1500" or qualification["split"] != "test" or len(samples) != 500:
        raise ValueError("CTW recognition manifest is not qualified TEST-500")
    output = {}
    for sample in samples:
        width, height = sample.image_size_wh
        records = []
        for instance in sample.instances:
            if instance.ignore:
                continue
            x1, y1, x2, y2 = instance.bbox_xyxy
            records.append({
                "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
                "text": instance.transcription,
                "granularity": "text_line",
            })
        output[f"ctw1500:{int(sample.sample_id)}"] = records
    return output, qualification


def _other_gt(dataset: str, image_path: Path, sample_id: str, image_wh: tuple[int, int]) -> list[dict]:
    identifier = sample_id.partition(":")[2]
    annotation = image_path.parent.parent / "annotations" / f"{identifier}.txt"
    instances = (
        parse_icdar2015_annotation(annotation, image_size_wh=image_wh)
        if dataset == "icdar2015"
        # Qualified Total-Text contains a small number of polygon vertices just
        # outside the raster.  Preserve parser provenance and clamp only the
        # post-hoc bbox below, matching the frozen V1 evaluation path.
        else parse_totaltext_annotation(annotation)
    )
    output = []
    width, height = image_wh
    for instance in instances:
        if instance.ignore:
            continue
        x1, y1, x2, y2 = instance.bbox_xyxy
        output.append({
            "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
            "text": instance.transcription,
            "granularity": instance.granularity,
        })
    return output


def _validate_existing_shard(shard_path: Path, output: Path, protocol_sha: str) -> dict:
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if shard.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"mixed protocol in existing shard: {shard_path}")
    paths = [(shard["native_mask_file"], shard["native_mask_file_sha256"])]
    paths.extend((row["crop_file"], row["crop_file_sha256"]) for row in shard["proposals"])
    for relative, expected in paths:
        path = output / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"missing or corrupted frozen artifact: {path}")
    return shard


def main() -> int:
    args = _args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps(report["summary"], indent=2), flush=True)
        print(f"[reuse] {result_path} SHA256 {_sha(result_path)}", flush=True)
        return 0

    data_root = args.data_root.resolve()
    manifest_path = args.test_manifest.resolve()
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if (
        manifest.dataset != args.dataset
        or manifest.split_role != "test"
        or len(manifest.samples) != EXPECTED_COUNTS[args.dataset]
    ):
        raise ValueError("test manifest does not match dataset/split/official count")
    config_path = args.segmentation_config.resolve()
    localization_config_path = args.localization_config.resolve()
    params = _params(localization_config_path)
    checkpoint = args.segmentation_checkpoint.resolve()
    run = _verify_training_run(args.segmentation_run_manifest.resolve(), checkpoint)
    frozen_v1 = args.frozen_v1_run.resolve()
    frozen_results = frozen_v1 / "results.json"
    frozen_protocol = frozen_v1 / "protocol.json"
    if not frozen_results.is_file() or not frozen_protocol.is_file():
        raise FileNotFoundError("frozen V1 run is incomplete")
    old = json.loads(frozen_results.read_text(encoding="utf-8"))
    if old["summary"]["dataset"] != args.dataset:
        raise ValueError("frozen V1 run dataset mismatch")

    ctw_gt = None
    ctw_qualification = None
    if args.dataset == "ctw1500":
        if args.ctw_recognition_manifest is None:
            raise ValueError("--ctw-recognition-manifest is required for CTW1500")
        ctw_gt, ctw_qualification = _ctw_gt(args.ctw_recognition_manifest.resolve(), data_root)

    protocol = {
        "schema_version": 1,
        "classification": "APPROXIMATE_PAPER_ALGORITHM1",
        "result_class": "paper Algorithm 1 crop diagnostic on additional datasets; not Table 5",
        "table_5_claim": False,
        "dataset": args.dataset,
        "split": manifest.split,
        "expected_samples": EXPECTED_COUNTS[args.dataset],
        "segmentation_architecture": "author-clarified-ag-unet-2048x1024-v1",
        "segmentation_config": str(config_path),
        "segmentation_config_sha256": _sha(config_path),
        "localization_config": str(localization_config_path),
        "localization_config_sha256": _sha(localization_config_path),
        "segmentation_checkpoint": str(checkpoint),
        "segmentation_checkpoint_sha256": run["checkpoint"]["sha256"],
        "segmentation_checkpoint_epoch": run["checkpoint"]["epoch"],
        "test_manifest": str(manifest_path),
        "test_manifest_sha256": _sha(manifest_path),
        "frozen_v1_results_sha256": _sha(frozen_results),
        "frozen_v1_protocol_sha256": _sha(frozen_protocol),
        "device": args.device,
        "mask_protocol": "sigmoid >= 0.5 at native 128x128 logits",
        "algorithm1": {
            "operation_order": [
                "connected_components_native128",
                "filter_true_foreground_component_area",
                "compute_bbox_native128",
                "pad_bbox_native128",
                "sort_true_component_area_descending",
                "top10",
                "map_padded_bbox_to_original",
                "crop_original_rgb",
            ],
            "parameters": {
                "min_area_native_foreground_pixels": params.min_area,
                "padding_native_pixels": params.pad,
                "connectivity": params.connectivity,
                "tie_break": params.tie_break,
                "nmax": params.nmax,
            },
            "paper_specified": [
                "connected regions", "component-area filtering", "bounding boxes",
                "padding", "descending component-area order", "top10", "original-image crops",
            ],
            "assumptions": [
                "native 128x128 localization coordinate frame", "Amin=150 native foreground pixels",
                "padding=6 native pixels", "8-connectivity", "spatial equal-area tie break",
                "floor-lower/ceil-upper bbox remap",
            ],
        },
        "ground_truth_boundary": "loaded only after crop generation for post-hoc geometry",
        "ctw_annotation_granularity": "text_line" if args.dataset == "ctw1500" else None,
        "ctw_recognition_qualification": ctw_qualification,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing protocol differs; refusing mixed resume")
    else:
        _write_atomic(protocol_path, protocol)
    protocol_sha = _sha(protocol_path)

    segmenter = _Segmenter(config_path, checkpoint, run["checkpoint"]["sha256"], args.device)
    shards = []
    try:
        for index, sample in enumerate(manifest.samples, start=1):
            key = _sample_key(sample.sample_id)
            shard_path = output / "samples" / f"{key}.json"
            if shard_path.exists():
                shards.append(_validate_existing_shard(shard_path, output, protocol_sha))
                continue
            image_path = data_root / sample.image_path
            if sample.image_sha256 is not None and _sha(image_path) != sample.image_sha256:
                raise ValueError(f"image SHA mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                rgb = opened.convert("RGB")
                image = np.asarray(rgb)
            inputs = _preprocess(rgb).unsqueeze(0).to(segmenter.device)
            with torch.inference_mode():
                logits = segmenter.model(inputs)
            if tuple(logits.shape) != (1, 1, 128, 128):
                raise RuntimeError(f"unexpected logits shape: {list(logits.shape)}")
            native = np.ascontiguousarray(
                (logits.sigmoid()[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
            )
            native_sha = sha256(native.tobytes()).hexdigest()
            frozen_shard_path = frozen_v1 / "proposals" / f"{key}.json"
            frozen_shard = json.loads(frozen_shard_path.read_text(encoding="utf-8"))
            if native_sha != frozen_shard["segmentation"]["native_mask_sha256"]:
                raise RuntimeError(f"native mask differs from frozen V1 run: {sample.sample_id}")
            localized = localize_native_algorithm1(native, image, params)

            mask_relative = Path("native_masks") / f"{key}.png"
            mask_path = output / mask_relative
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(native * 255, mode="L").save(mask_path, optimize=True)
            crop_rows = []
            crop_dir = output / "crops" / key
            crop_dir.mkdir(parents=True, exist_ok=True)
            for block in localized.retained:
                crop_relative = Path("crops") / key / f"{block.rank:02d}.png"
                crop_path = output / crop_relative
                Image.fromarray(block.crop).save(crop_path, optimize=True)
                crop_rows.append({
                    "rank": block.rank,
                    "component_id": block.component_id,
                    "component_area_native": block.component_area_native,
                    "bbox_native": list(block.bbox_native),
                    "padded_bbox_native": list(block.padded_bbox_native),
                    "bbox_original": list(block.bbox_original),
                    "padded_bbox_original": list(block.padded_bbox_original),
                    "crop_shape": list(block.crop.shape),
                    "crop_file": crop_relative.as_posix(),
                    "crop_file_sha256": _sha(crop_path),
                })

            # Evaluation annotations enter only after masks, boxes and crops are frozen.
            ground_truth = (
                ctw_gt[sample.sample_id]
                if ctw_gt is not None
                else _other_gt(
                    args.dataset, image_path, sample.sample_id, (image.shape[1], image.shape[0])
                )
            )
            rejected = localized.audit["rejected"]
            shard = {
                "schema_version": 1,
                "protocol_sha256": protocol_sha,
                "sample_id": sample.sample_id,
                "image": sample.image_path.as_posix(),
                "image_sha256": sample.image_sha256,
                "image_shape_hwc": list(image.shape),
                "native_mask_array_sha256": native_sha,
                "native_foreground_pixels": int(native.sum()),
                "native_mask_file": mask_relative.as_posix(),
                "native_mask_file_sha256": _sha(mask_path),
                "raw_component_count": len(localized.retained) + len(rejected),
                "candidates_before_cap": len(localized.retained) + sum(
                    value["reason"] == "top_10_limit" for value in rejected
                ),
                "below_area_rejections": sum(
                    value["reason"] == "below_min_area" for value in rejected
                ),
                "top10_rejections": sum(
                    value["reason"] == "top_10_limit" for value in rejected
                ),
                "ground_truth": ground_truth,
                "proposals": crop_rows,
            }
            _write_atomic(shard_path, shard)
            shards.append(shard)
            if index % 10 == 0 or index == len(manifest.samples):
                print(f"[{args.dataset} paper-Algorithm1 crops] {index}/{len(manifest.samples)}", flush=True)
    finally:
        segmenter.release()
        del segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(shards) != EXPECTED_COUNTS[args.dataset]:
        raise RuntimeError("incomplete official TEST population")
    rows = [{
        "sample_id": shard["sample_id"],
        "ground_truth": shard["ground_truth"],
        "predictions": [
            {"rank": value["rank"], "bbox": value["padded_bbox_original"], "text": ""}
            for value in shard["proposals"]
        ],
    } for shard in shards]
    geometry = evaluate_images(rows, text_available=False)["geometry"]
    summary = {
        "dataset": args.dataset,
        "images": len(shards),
        "annotation_granularity": "text_line" if args.dataset == "ctw1500" else "word",
        "raw_components": sum(value["raw_component_count"] for value in shards),
        "candidates_before_cap": sum(value["candidates_before_cap"] for value in shards),
        "retained_crops": sum(len(value["proposals"]) for value in shards),
        "below_area_rejections": sum(value["below_area_rejections"] for value in shards),
        "top10_rejections": sum(value["top10_rejections"] for value in shards),
        "images_over_top10_before_cap": sum(value["candidates_before_cap"] > 10 for value in shards),
        "geometry": geometry,
    }
    report = {
        "schema_version": 1,
        "classification": "APPROXIMATE_PAPER_ALGORITHM1",
        "result_class": "crop-generation and geometry diagnostic; no OCR",
        "protocol": protocol,
        "summary": summary,
        "sample_shards": [f"samples/{_sample_key(value['sample_id'])}.json" for value in shards],
    }
    _write_atomic(result_path, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"JSON {result_path} SHA256 {_sha(result_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
