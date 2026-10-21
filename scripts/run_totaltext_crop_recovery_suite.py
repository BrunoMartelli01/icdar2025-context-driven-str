#!/usr/bin/env python3
"""Run all GT-blind crop generators, PARSeq and post-hoc normalized oracle recall.

The local path profile selects files, while scientific parameters belong only
to the approximate experiment configuration. No TEST annotations are opened.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from localize.crop_recovery import (METHODS, add_border_variants, add_tile_candidates,
    mask_candidates, render_crop, tile_windows)
from recognition.crop_recovery_metric import evaluate_pool
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.crop_clip_runtime import _cuda_peak, cached, sha, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _load_probability
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import (
    _image, _model_and_run, _tensor, _train_gt)
from scripts.run_totaltext_crop_clip_battery import _ocr_record
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _config as training_config
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_crop_recovery_suite_train_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/totaltext_crop_recovery_local_paths_v1.json"
CODE_FILES = (
    "localize/crop_recovery.py", "recognition/crop_recovery_metric.py",
    "scripts/run_totaltext_crop_recovery_suite.py", "recognition/parseq.py",
    "scripts/crop_clip_runtime.py", "scripts/run_totaltext_crop_clip_battery.py",
    "scripts/evaluate_totaltext_highres_contour_padding.py",
    "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
    "scripts/evaluate_totaltext_letterbox_threshold_sweep_train.py",
    "scripts/train_totaltext_v4_image_frequency_ablation.py",
    "scripts/train_totaltext_v4_highres_ablation.py",
    "scripts/rescore_totaltext_highres_padding_symbols.py",
    "recognition/scene_text_v1_evaluation.py", "recognition/scene_text_datasets.py",
    "segmentation/data.py", "segmentation/scene_text_preparation.py",
    "segmentation/totaltext_image_frequency_augment.py", "segmentation/training.py",
    "segmentation/transforms.py", "models/unet_adv.py")
PATH_FIELDS = ("data_root", "probability_cache", "training_runs", "baseline_ocr_cache",
               "parseq_source", "parseq_archive", "parseq_checkpoint", "output_root")


def key(sample_id: str) -> str:
    return sample_id.replace(":", "__")


def validate_config(config: dict) -> None:
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("dataset") != "total_text" or config.get("split") != "TRAIN_heldout_125"
        or config.get("mask_size") != 256 or config.get("baseline_threshold") != .5
        or config.get("baseline_padding") != 0 or config.get("table_5_claim") is not False
        or config.get("execution", {}).get("precision") != "float32"):
        raise ValueError("Expected isolated approximate Total-Text TRAIN native-256 protocol")
    if not 0 < config["matching_iou"] <= 1:
        raise ValueError("invalid matching IoU")
    for name in ("threshold_tree", "tiled_segmentation"):
        values = config[name]["thresholds"]
        if not values or len(set(values)) != len(values) or any(not 0 < v < 1 for v in values):
            raise ValueError(f"invalid {name} thresholds")
    if .5 not in config["threshold_tree"]["thresholds"]:
        raise ValueError("threshold hypotheses must include baseline")
    for low, high in config["threshold_tree"]["seed_pairs"]:
        if not 0 < low < high < 1:
            raise ValueError("invalid seed thresholds")
    for name in ("threshold_tree", "erosion_reconstruction", "image_seams"):
        if not isinstance(config[name]["maximum_adjacent_union"], int) or config[name]["maximum_adjacent_union"] < 1:
            raise ValueError("adjacent union size must be a positive integer")
    if not config["erosion_reconstruction"]["radius_thickness_fractions"] or any(
            not 0 < v < .5 for v in config["erosion_reconstruction"]["radius_thickness_fractions"]):
        raise ValueError("invalid erosion fractions")
    seams = config["image_seams"]
    if (not 0 <= seams["probability_weight"] <= 1 or not 0 < seams["interior_fraction"] < .5
        or not 0 < seams["minimum_spacing_fraction"] < 1 or not 0 < seams["search_band_fraction"] < .5
        or seams["maximum_seams_per_axis"] < 1 or seams["minimum_native_extent"] < 3
        or seams["cut_half_width"] < 1):
        raise ValueError("invalid seam parameters")
    tiles = config["tiled_segmentation"]
    if not tiles["grids"] or any(not isinstance(v, int) or v < 2 for v in tiles["grids"]) or not 0 <= tiles["overlap"] < 1:
        raise ValueError("invalid tiling parameters")
    variants = config["variants"]
    if (not isinstance(variants["oriented"], bool)
        or not set(variants["source_methods"]) <= set(METHODS[:-1])
        or any(not -.5 < v < .5 for v in variants["border_height_fractions"])
        or any(not 0 < v <= .5 for v in variants["neutral_height_fractions"])):
        raise ValueError("invalid crop variants")
    if config["execution"]["parseq_batch_size"] < 1 or config["execution"]["cpu_threads"] < 1:
        raise ValueError("positive batch size and thread count required")
    for field in ("segmentation_checkpoint_sha256", "threshold_report_sha256", "threshold_protocol_config_sha256"):
        digest = config[field]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid hash: {field}")


def resolve_paths(path: Path, overrides: dict) -> dict[str, Path]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("path_base") != "repository_root":
        raise ValueError("path profile must use repository_root")
    output = {}
    for field in PATH_FIELDS:
        value = Path(overrides[field] if overrides.get(field) is not None else profile[field])
        output[field] = (value if value.is_absolute() else REPO / value).resolve()
    return output


@contextmanager
def measured_stage(output: Path, stage: str, device: str, details: dict):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    try:
        yield
    except Exception as error:
        write_json(output / "failure.json", {
            "result_class": "FAILED_RUN", "reproduction_mode": "approximate", "stage": stage,
            "error_type": type(error).__name__, "error": str(error),
            "out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
            "device": device, "precision": "float32", "model_set": [stage],
            "cuda_peak": _cuda_peak(device), "details": details, "traceback": traceback.format_exc()})
        raise
    else:
        write_json(output / f"{stage}_last_execution.json", {
            "device": device, "precision": "float32", "model_set": [stage],
            "cuda_peak": _cuda_peak(device), "seconds": time.monotonic() - start, **details})


def freeze(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"existing artifact differs: {path}; use a new output directory")
    else:
        write_json(path, value)


def run_tiles(samples, paths: dict, config: dict, plan_sha: str, output: Path,
              device: str, validation_sha: str) -> dict:
    """Separate model stage; every tile probability is hash checked on reuse."""
    records = {}
    model = None
    try:
        for position, sample in enumerate(samples, 1):
            path = output / "tiles" / f"{key(sample.sample_id)}.json"
            contract = {"plan_sha256": plan_sha, "sample_id": sample.sample_id, "image_sha256": sample.image_sha256}
            previous = cached(path, contract)
            if previous is not None:
                for row in previous["tiles"]:
                    read_tile(output, row)
                records[sample.sample_id] = previous
                print(f"[tiles] {position}/{len(samples)} {sample.sample_id} cached", flush=True)
                continue
            image = _image(sample, paths["data_root"])
            if model is None:
                model, transform, run = _model_and_run("letterbox", argparse.Namespace(
                    training_runs=paths["training_runs"]), training_config(), validation_sha)
                if run["checkpoint"]["sha256"] != config["segmentation_checkpoint_sha256"]:
                    raise RuntimeError("segmentation checkpoint differs from frozen mask source")
                model = model.eval().to(device)
            tiles = []
            for grid in config["tiled_segmentation"]["grids"]:
                windows = tile_windows((image.shape[1], image.shape[0]), grid, config["tiled_segmentation"]["overlap"])
                for index, window in enumerate(windows):
                    x1, y1, x2, y2 = window
                    rgb = image[y1:y2, x1:x2]
                    tensor = _tensor(rgb, transform).to(device)
                    with torch.inference_mode():
                        logits = model(tensor)
                    if logits.shape != (1, 1, config["mask_size"], config["mask_size"]):
                        raise RuntimeError(f"unexpected tile logit shape: {logits.shape}")
                    probability = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
                    probability *= letterbox_canvas_mask((rgb.shape[1], rgb.shape[0])).numpy()
                    relative = f"tiles/{key(sample.sample_id)}-g{grid}-{index}.npy"
                    from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _atomic_bytes
                    from io import BytesIO
                    encoded = BytesIO()
                    np.save(encoded, probability, allow_pickle=False)
                    _atomic_bytes(output / relative, encoded.getvalue())
                    tiles.append({"tile_id": f"g{grid}-{index}", "window": window, "path": relative,
                                  "file_sha256": sha(output / relative), "shape": list(probability.shape)})
                    del logits, tensor
            value = {"contract": contract, "reproduction_mode": "approximate", "tiles": tiles,
                     "model_peak": _cuda_peak(device), "input_shape": [1, 3, 512, 512], "batch_size": 1}
            write_json(path, value)
            records[sample.sample_id] = value
            print(f"[tiles] {position}/{len(samples)} {sample.sample_id} new={len(tiles)}", flush=True)
    finally:
        if model is not None:
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return records


def read_tile(output: Path, row: dict) -> np.ndarray:
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("tile artifact escapes output directory")
    path = output / relative
    if sha(path) != row["file_sha256"]:
        raise RuntimeError(f"tile probability hash changed: {path}")
    p = np.load(path, allow_pickle=False)
    if (list(p.shape) != row["shape"] or p.shape != (256, 256) or p.dtype != np.float32
        or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1):
        raise RuntimeError("invalid tile probability")
    return p


def prepare(samples, paths, config, plan_sha, output, tile_records, protocol_sha, reference):
    prepared = {}
    frozen = {row["sample_id"]: row for row in reference["thresholds"]["0.50"]["per_image"]}
    for position, sample in enumerate(samples, 1):
        shard_path = output / "prepared" / f"{key(sample.sample_id)}.json"
        sidecar = paths["probability_cache"] / "probabilities" / f"{key(sample.sample_id)}.json"
        contract = {"plan_sha256": plan_sha, "sample_id": sample.sample_id,
                    "image_sha256": sample.image_sha256, "probability_sidecar_sha256": sha(sidecar),
                    "tile_shard_sha256": sha(output / "tiles" / f"{key(sample.sample_id)}.json")
                    if sample.sample_id in tile_records else None}
        probability = _load_probability(paths["probability_cache"] / "probabilities" / f"{key(sample.sample_id)}.npy",
            json.loads(sidecar.read_text()), sample=sample, protocol_sha=protocol_sha)
        image = _image(sample, paths["data_root"])
        previous = cached(shard_path, contract)
        if previous is None:
            pool = mask_candidates(probability, image, config)
            for tile in tile_records.get(sample.sample_id, {}).get("tiles", []):
                add_tile_candidates(pool, read_tile(output, tile), tile["window"],
                                    config["tiled_segmentation"]["thresholds"], tile["tile_id"])
            add_border_variants(pool, config["variants"])
            rows = list(pool.rows.values())
            for row in rows:
                row["candidate_id"] = f"{key(sample.sample_id)}:{row['candidate_id']}"
                crop = render_crop(image, row)
                row["crop_shape"] = list(crop.shape)
                row["crop_rgb_sha256"] = crop_array_hash(crop)
            previous = {"contract": contract, "reproduction_mode": "approximate", "candidates": rows,
                        "counts_by_method": {method: sum(method in row["methods"] for row in rows) for method in METHODS}}
            write_json(shard_path, previous)
        rows = previous["candidates"]
        if len({row["candidate_id"] for row in rows}) != len(rows):
            raise RuntimeError("duplicate candidate IDs")
        actual = sorted((tuple(row["bbox"]), row["crop_rgb_sha256"]) for row in rows if "baseline" in row["methods"])
        expected = sorted((tuple(row["bbox"]), row["crop_rgb_sha256"]) for row in frozen[sample.sample_id]["predictions"])
        if actual != expected:
            raise RuntimeError(f"baseline crops differ from frozen reference: {sample.sample_id}")
        prepared[sample.sample_id] = previous
        print(f"[prepare] {position}/{len(samples)} {sample.sample_id} crops={len(rows)} "
              f"unique_pixels={len({r['crop_rgb_sha256'] for r in rows})}", flush=True)
    return prepared


def run_ocr(samples, prepared, paths, output, device, batch_size):
    recognizer = None
    all_readings = {}
    total_new = 0
    try:
        for position, sample in enumerate(samples, 1):
            shard_path = output / "ocr" / f"{key(sample.sample_id)}.json"
            contract = {"prepared_sha256": sha(output / "prepared" / f"{key(sample.sample_id)}.json"),
                        "device": device, "batch_size": batch_size,
                        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                        "parseq_adapter_sha256": sha(REPO / "recognition/parseq.py")}
            previous = cached(shard_path, contract)
            computed = 0
            rows = prepared[sample.sample_id]["candidates"]
            if previous is None:
                image = _image(sample, paths["data_root"])
                readings = {}
                cache_sources = {}
                pending = {}
                for row in rows:
                    crop_hash = row["crop_rgb_sha256"]
                    if crop_hash in readings or crop_hash in pending:
                        continue
                    for root in (output / "ocr_by_crop", paths["probability_cache"] / "ocr", paths["baseline_ocr_cache"]):
                        prior_path = root / f"{crop_hash}.json"
                        if prior_path.exists():
                            readings[crop_hash] = _ocr_record(prior_path, crop_hash)
                            cache_sources[crop_hash] = {"path": str(prior_path), "file_sha256": sha(prior_path),
                                "execution": readings[crop_hash].get("execution", "upstream; see resolved plan/cache protocol")}
                            break
                    else:
                        pending[crop_hash] = row
                if pending and recognizer is None:
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=paths["parseq_source"], runtime_source_archive=paths["parseq_archive"],
                        runtime_source_type=PARSEQ_ARCHIVE_SOURCE, checkpoint=paths["parseq_checkpoint"], device=device))
                pending_rows = list(pending.values())
                for start in range(0, len(pending_rows), batch_size):
                    batch = pending_rows[start:start + batch_size]
                    crops = [render_crop(image, row) for row in batch]
                    if any(crop_array_hash(crop) != row["crop_rgb_sha256"] for crop, row in zip(crops, batch)):
                        raise RuntimeError("prepared crop RGB changed")
                    values = recognizer.recognize_candidate_batch(crops)
                    if len(values) != len(batch):
                        raise RuntimeError("PARSeq returned incomplete batch")
                    for row, value in zip(batch, values):
                        crop_hash = row["crop_rgb_sha256"]
                        record = {"schema_version": 1, "crop_rgb_sha256": crop_hash,
                                  "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                                  "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                                  "execution": {"device": device, "batch_size": batch_size, "precision": "float32"},
                                  **value.to_dict()}
                        write_json(output / "ocr_by_crop" / f"{crop_hash}.json", record)
                        cache_sources[crop_hash] = {"path": str(output / "ocr_by_crop" / f"{crop_hash}.json"),
                            "file_sha256": sha(output / "ocr_by_crop" / f"{crop_hash}.json"), "execution": record["execution"]}
                        readings[crop_hash] = record
                        computed += 1
                    if (start // batch_size) % 25 == 0:
                        print(f"[PARSeq crops] {sample.sample_id} {min(start + batch_size, len(pending_rows))}/{len(pending_rows)}", flush=True)
                previous = {"contract": contract, "reproduction_mode": "approximate",
                            "rows": [{"candidate_id": row["candidate_id"],
                                      "crop_rgb_sha256": row["crop_rgb_sha256"],
                                      "raw_text": readings[row["crop_rgb_sha256"]]["raw_text"]} for row in rows],
                            "new_crops": computed, "model_peak": _cuda_peak(device),
                            "crop_cache_sources": cache_sources,
                            "model_input_shape": [batch_size, 3, 32, 128]}
                write_json(shard_path, previous)
            if [row["candidate_id"] for row in previous["rows"]] != [row["candidate_id"] for row in rows]:
                raise RuntimeError("OCR candidate order differs")
            all_readings[sample.sample_id] = {row["candidate_id"]: row for row in previous["rows"]}
            total_new += computed
            print(f"[PARSeq] {position}/{len(samples)} {sample.sample_id} computed_now={computed} "
                  f"computed_at_shard_creation={previous['new_crops']}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    return all_readings, total_new


def arm_definitions(include_tiles: bool) -> dict[str, list[str]]:
    methods = [m for m in METHODS if include_tiles or m != "tiled_segmentation"]
    arms = {method: [method] for method in methods}
    for method in methods[1:]:
        arms[f"baseline_plus_{method}"] = ["baseline", method]
    for end in range(2, len(methods)):
        arms[f"cumulative_{end}_{methods[end]}"] = methods[:end + 1]
    arms["all_methods"] = methods
    return arms


def score(samples, prepared, readings, paths, config, output, plan, include_tiles, reference):
    arms = arm_definitions(include_tiles)
    summary = {arm: {"correct": 0, "geometry": 0, "candidates": 0, "distinct_crop_pixels": 0,
                     "gt_words": 0, "gt_nonempty_normalized": 0} for arm in arms}
    per_image = []
    baseline_total = set()
    correct_total = {arm: set() for arm in arms}
    frozen = {row["sample_id"]: row for row in reference["thresholds"]["0.50"]["per_image"]}
    for sample in samples:
        image = _image(sample, paths["data_root"])
        gt, annotation_sha = _train_gt(sample, paths["data_root"], (image.shape[1], image.shape[0]))
        # The frozen aggregate stores GT instances, not annotation-file hashes.
        # Compare its exact semantic content; freeze the current file hash below.
        if gt != frozen[sample.sample_id]["ground_truth"]:
            raise RuntimeError(f"TRAIN annotations changed: {sample.sample_id}")
        from scripts.rescore_totaltext_highres_padding_symbols import normalize_word
        empty_count = sum(not normalize_word(word["text"]) for word in gt)
        rows = prepared[sample.sample_id]["candidates"]
        scored = {}
        for arm, methods in arms.items():
            selected = [row for row in rows if set(row["methods"]) & set(methods)]
            result = evaluate_pool(selected, readings[sample.sample_id], gt, config["matching_iou"])
            scored[arm] = result
            identity = {f"{sample.sample_id}:{index}" for index in result["correct_gt_indices"]}
            correct_total[arm].update(identity)
            total = summary[arm]
            total["correct"] += result["correct"]
            total["geometry"] += len(result["geometry_gt_indices"])
            total["candidates"] += result["candidates"]
            total["distinct_crop_pixels"] += result["distinct_crop_pixels"]
            total["gt_words"] += len(gt)
            total["gt_nonempty_normalized"] += len(gt) - empty_count
        baseline = set(scored["baseline"]["correct_gt_indices"])
        for arm, result in scored.items():
            current = set(result["correct_gt_indices"])
            result["new_vs_baseline_gt_indices"] = sorted(current - baseline)
            result["lost_vs_baseline_gt_indices"] = sorted(baseline - current)
        baseline_total.update(f"{sample.sample_id}:{index}" for index in baseline)
        value = {"sample_id": sample.sample_id, "annotation_sha256": annotation_sha,
                 "ground_truth": gt, "arms": scored,
                 "prepared_sha256": sha(output / "prepared" / f"{key(sample.sample_id)}.json"),
                 "ocr_sha256": sha(output / "ocr" / f"{key(sample.sample_id)}.json")}
        freeze(output / "evaluation" / f"{key(sample.sample_id)}.json", value)
        per_image.append(f"evaluation/{key(sample.sample_id)}.json")
    for arm, total in summary.items():
        total["rate_all_gt"] = total["correct"] / total["gt_words"] if total["gt_words"] else 0.
        total["rate_nonempty_gt"] = total["correct"] / total["gt_nonempty_normalized"] if total["gt_nonempty_normalized"] else 0.
        total["new_vs_baseline"] = len(correct_total[arm] - baseline_total)
        total["lost_vs_baseline"] = len(baseline_total - correct_total[arm])
        total["net_vs_baseline"] = total["correct"] - len(baseline_total)
        if arm in METHODS:
            others = [ids for method, ids in correct_total.items() if method in METHODS and method != arm]
            total["unique_vs_other_methods"] = len(correct_total[arm] - set().union(*others))
    if len(samples) == 125 and summary["baseline"]["gt_words"] != 983:
        raise RuntimeError("frozen full TRAIN denominator is not 983")
    creation_peaks = {}
    for stage, directory in (("segmentation", "tiles"), ("parseq", "ocr")):
        values = [json.loads((output / directory / f"{key(sample.sample_id)}.json").read_text())["model_peak"]
                  for sample in samples if (output / directory / f"{key(sample.sample_id)}.json").exists()]
        creation_peaks[stage] = {field: max((v[field] for v in values if v.get(field) is not None), default=None)
                                for field in ("peak_allocated_bytes", "peak_reserved_bytes")}
    result = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": config["claim_label"],
              "result_class": plan["result_class"], "plan_sha256": sha(output / "resolved_configuration.json"),
              "images": len(samples), "normalization": config["normalization"],
              "primary_metric": "normalized exact word recall; maximum one-to-one spatial IoU matching across all crops",
              "matching_iou": config["matching_iou"], "summary": summary, "arms": arms,
              "per_image": per_image, "correct_word_ids": {arm: sorted(ids) for arm, ids in correct_total.items()},
              "creation_time_model_peaks": creation_peaks,
              "table_5_claim": False}
    freeze(output / "results.json", result)
    lines = ["# Total-Text crop recovery", "", config["claim_label"], "",
             "Each care word is counted at most once; exact casefold + Unicode P/S removal and spatial IoU gate.", "",
             "| Method/pool | Correct / GT | New | Lost | Crop recipes | Distinct RGB crops |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, total in summary.items():
        lines.append(f"| {arm} | {total['correct']}/{total['gt_words']} | {total['new_vs_baseline']} | "
                     f"{total['lost_vs_baseline']} | {total['candidates']} | {total['distinct_crop_pixels']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:]), flush=True)
    print(f"Results: {output / 'results.json'}", flush=True)
    return result


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    paths = resolve_paths(args.paths, vars(args))
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("device must be cpu, auto or cuda:N")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select --device cpu for a CPU run")
    batch_size = args.batch_size if args.batch_size is not None else config["execution"]["parseq_batch_size"]
    threads = args.cpu_threads if args.cpu_threads is not None else config["execution"]["cpu_threads"]
    if batch_size < 1 or threads < 1 or (args.limit is not None and not 1 <= args.limit <= 125):
        raise ValueError("positive batch/threads and limit in 1..125 required")
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training = training_config()
    manifest_path = paths["data_root"] / training["train_manifest"]
    if sha(manifest_path) != training["train_manifest_sha256"]:
        raise RuntimeError("TRAIN manifest hash differs")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise RuntimeError("Only Total-Text TRAIN is permitted")
    _, val_ids, val_sha = _split_indices(manifest, training["training"]["seed"], 125)
    samples = [manifest.samples[index] for index in val_ids]
    if args.limit is not None:
        samples = samples[:args.limit]
    protocol_path = paths["probability_cache"] / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    # Upstream sidecars hash canonical JSON content, not the formatted file bytes.
    probability_contract_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (protocol.get("checkpoint_sha256") != config["segmentation_checkpoint_sha256"]
        or protocol.get("train_manifest_sha256") != sha(manifest_path)
        or protocol.get("config_sha256") != config["threshold_protocol_config_sha256"]
        or protocol.get("split") != "TRAIN_heldout_125"):
        raise RuntimeError("frozen probability protocol differs")
    reference_path = paths["probability_cache"] / "results.json"
    if sha(reference_path) != config["threshold_report_sha256"]:
        raise RuntimeError("frozen threshold reference changed")
    reference = json.loads(reference_path.read_text())
    reference_ids = {row["sample_id"] for row in reference["thresholds"]["0.50"]["per_image"]}
    if reference_ids != {manifest.samples[index].sample_id for index in val_ids}:
        raise RuntimeError("frozen probability cache uses another split")
    if sha(paths["parseq_checkpoint"]) != PARSEQ_CHECKPOINT_SHA256 or sha(paths["parseq_archive"]) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("qualified PARSeq artifacts missing or changed")
    if not paths["parseq_source"].is_dir():
        raise RuntimeError("qualified PARSeq runtime source missing")
    if not args.without_tiles and sha(paths["training_runs"] / "letterbox/best.pt") != config["segmentation_checkpoint_sha256"]:
        raise RuntimeError("qualified segmentation checkpoint missing or changed")
    plan = {"schema_version": 1, "reproduction_mode": "approximate", "config": config,
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TRAIN_DIAGNOSTIC",
            "claim_label": config["claim_label"], "config_sha256": sha(args.config),
            "paths": {name: str(value) for name, value in paths.items()},
            "paths_profile_sha256": sha(args.paths), "sample_ids": [s.sample_id for s in samples],
            "manifest_sha256": sha(manifest_path), "validation_ids_sha256": val_sha,
            "probability_protocol_sha256": sha(protocol_path), "probability_reference_sha256": sha(reference_path),
            "probability_sidecar_contract_sha256": probability_contract_sha,
            "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
            "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
            "tiles_enabled": not args.without_tiles,
            "execution": {"device": device, "batch_size": batch_size, "cpu_threads": threads,
                          "precision": "float32", "seed": config["seed"], "segmentation_batch_size": 1,
                          "model_loading": "sequential", "cuda_version": torch.version.cuda,
                          "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark,
                          "upstream_probability_execution": {k: protocol.get(k) for k in ("device", "parseq_batch_size")}},
            "software": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                         "opencv": cv2.__version__, "scipy": importlib.metadata.version("scipy"),
                         "pillow": importlib.metadata.version("Pillow"),
                         "torchvision": importlib.metadata.version("torchvision")},
            "code_sha256": {name: sha(REPO / name) for name in CODE_FILES},
            "ground_truth_use": "post-hoc evaluation only; no GT access in crop generation or OCR",
            "table_5_claim": False}
    digest = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output_dir.resolve() if args.output_dir else paths["output_root"] / digest
    if output == REPO or REPO in output.parents:
        raise ValueError("generated artifacts must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / "resolved_configuration.json", plan)
    plan_sha = sha(output / "resolved_configuration.json")
    print(f"Output: {output}\nDevice: {device}; images: {len(samples)}; tiles: {not args.without_tiles}", flush=True)
    if not args.without_tiles:
        with measured_stage(output, "segmentation", device, {"input_shape": [1, 3, 512, 512], "batch_size": 1}):
            tiles = run_tiles(samples, paths, config, plan_sha, output, device, val_sha)
    else:
        tiles = {}
    with measured_stage(output, "prepare", "cpu", {"models": []}):
        prepared = prepare(samples, paths, config, plan_sha, output, tiles, probability_contract_sha, reference)
    counts = {method: sum(value["counts_by_method"][method] for value in prepared.values()) for method in METHODS}
    print("Candidate counts: " + json.dumps(counts), flush=True)
    if args.prepare_only:
        write_json(output / "prepare_only.json", {"result_class": plan["result_class"], "counts_by_method": counts})
        return output
    with measured_stage(output, "parseq", device, {"input_shape": [batch_size, 3, 32, 128], "batch_size": batch_size}):
        readings, new_crops = run_ocr(samples, prepared, paths, output, device, batch_size)
    with measured_stage(output, "evaluation", "cpu", {"models": []}):
        score(samples, prepared, readings, paths, config, output, plan, not args.without_tiles, reference)
    write_json(output / "last_execution.json", {"new_ocr_crops": new_crops, "status": "SUCCESS"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    for name in PATH_FIELDS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--limit", type=int, help="1..125 images; always labeled SMOKE_TEST")
    parser.add_argument("--without-tiles", action="store_true", help="explicit partial suite using cached masks only")
    parser.add_argument("--prepare-only", action="store_true", help="generate and audit crops, do not run PARSeq")
    args = parser.parse_args()
    output = run(args)
    print(f"Total-Text crop recovery reports: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
