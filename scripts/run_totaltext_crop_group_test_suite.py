#!/usr/bin/env python3
"""Official TEST-300: frozen threshold/window crops, PARSeq, groups, 14 selectors."""
from __future__ import annotations

import argparse
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

from localize.crop_recovery import METHODS, mask_candidates, add_tile_candidates, render_crop
from recognition.crop_recovery_metric import evaluate_pool
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts import analyze_totaltext_crop_word_groups as grouping_runner
from scripts import run_crop_group_word_selection as selectors
from scripts import run_totaltext_crop_recovery_suite as crops
from scripts.crop_clip_runtime import sha, write_json, _cuda_peak
from scripts.evaluate_totaltext_highres_contour_padding import _ground_truth, crop_array_hash
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _freeze_probability, _load_probability
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _image, _tensor, _model_and_run
from scripts.train_totaltext_v4_image_frequency_ablation import _config as training_config
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/totaltext_crop_group_test_suite_v1.json"
PATHS = REPO / "configs/approximate/totaltext_crop_group_test_local_paths_v1.json"
GROUP_CONFIG = REPO / "configs/approximate/totaltext_crop_word_groups_test_v1.json"
# Qualified source-stage version of the completed TEST run before the independent
# CLIP channel-format correction. Its source-generation dependencies are checked
# separately below; the selector runtime is intentionally outside that boundary.
REUSABLE_SOURCE_RUNNER_SHA256 = "e791d8e547f62ea49005eb4da25438a9bfb421cca6d201150a47258c2f614afa"
PRIOR_SELECTOR_RUNTIME_SHA256 = "c02168ec7ad7f3752fc6cb63d324d5a6aa3bf926785e1df49002c60d404ebd59"
RGB_FIXED_SELECTOR_RUNTIME_SHA256 = "1fcf69e95dd13909c35d01a8aac425cf3f69a7ec1759671501d1c495b8215fae"


def read_config(path):
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("TEST configuration must belong to configs/approximate")
    value = json.loads(path.read_text())
    # Reuse existing numerical generator validation without changing its TRAIN defaults.
    validation = dict(value, split="TRAIN_heldout_125", threshold_report_sha256="0" * 64,
                      threshold_protocol_config_sha256="0" * 64)
    crops.validate_config(validation)
    if (value["split"] != "official_TEST_300" or value["test_images"] != 300
        or value["test_care_words"] != 2204 or value["parameter_fitting"] is not False
        or value["source_methods"] != ["baseline", "threshold_tree", "tiled_segmentation"]
        or value["variants"]["oriented"] or value["variants"]["border_height_fractions"]
        or value["variants"]["neutral_height_fractions"]):
        raise ValueError("expected frozen TEST-300 thresholds/windows without final variations")
    return value


def test_samples(path, cfg, limit=None):
    if sha(path) != cfg["test_manifest_sha256"]:
        raise ValueError("official TEST manifest hash changed")
    manifest = load_segmentation_manifest(path, require_checksums=True)
    if (manifest.dataset != "total_text" or manifest.split_role != "test"
        or len(manifest.samples) != cfg["test_images"]):
        raise ValueError("expected official TEST-300; TRAIN manifests are forbidden")
    if limit is not None and not 1 <= limit <= len(manifest.samples):
        raise ValueError("--limit must be between 1 and official TEST image count")
    return manifest.samples[:limit] if limit is not None else manifest.samples


def whole_probabilities(samples, paths, cfg, validation_sha, output, protocol_sha, device):
    model = None
    try:
        for position, sample in enumerate(samples, 1):
            key = crops.key(sample.sample_id)
            path = paths["probability_cache"] / "probabilities" / f"{key}.npy"
            metadata_path = path.with_suffix(".json")
            if metadata_path.exists():
                _load_probability(path, json.loads(metadata_path.read_text()), sample=sample, protocol_sha=protocol_sha)
            else:
                if path.exists():
                    raise RuntimeError(f"orphan probability without provenance: {path}")
                if model is None:
                    model, transform, report = _model_and_run("letterbox", argparse.Namespace(
                        training_runs=paths["training_runs"]), training_config(), validation_sha)
                    if report["checkpoint"]["sha256"] != cfg["segmentation_checkpoint_sha256"]:
                        raise RuntimeError("segmentation checkpoint differs from frozen TRAIN selection")
                    model = model.eval().float().to(device)
                image = _image(sample, paths["data_root"])
                with torch.inference_mode():
                    logits = model(_tensor(image, transform).to(device))
                if tuple(logits.shape) != (1, 1, cfg["mask_size"], cfg["mask_size"]):
                    raise RuntimeError(f"unexpected whole-image logit shape: {logits.shape}")
                probability = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
                probability *= letterbox_canvas_mask((image.shape[1], image.shape[0])).numpy()
                _freeze_probability(path, metadata_path, probability, sample=sample, protocol_sha=protocol_sha)
                del logits
            print(f"[TEST whole-mask] {position}/{len(samples)} {sample.sample_id}", flush=True)
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def prepare(samples, paths, cfg, output, plan_sha, protocol_sha, tiles):
    prepared = {}
    for position, sample in enumerate(samples, 1):
        key = crops.key(sample.sample_id)
        path = output / "prepared" / f"{key}.json"
        sidecar = paths["probability_cache"] / "probabilities" / f"{key}.json"
        contract = {"plan_sha256": plan_sha, "sample_id": sample.sample_id,
                    "image_sha256": sample.image_sha256, "probability_sidecar_sha256": sha(sidecar),
                    "tile_shard_sha256": sha(output / "tiles" / f"{key}.json")}
        value = crops.cached(path, contract)
        if value is None:
            image = _image(sample, paths["data_root"])
            probability = _load_probability(sidecar.with_suffix(".npy"), json.loads(sidecar.read_text()),
                sample=sample, protocol_sha=protocol_sha)
            pool = mask_candidates(probability, image, cfg, include_auxiliary=False)
            for tile in tiles[sample.sample_id]["tiles"]:
                add_tile_candidates(pool, crops.read_tile(output, tile), tile["window"],
                                    cfg["tiled_segmentation"]["thresholds"], tile["tile_id"])
            rows = list(pool.rows.values())
            for row in rows:
                row["candidate_id"] = f"{key}:{row['candidate_id']}"
                rgb = render_crop(image, row)
                row.update(crop_shape=list(rgb.shape), crop_rgb_sha256=crop_array_hash(rgb))
            value = {"contract": contract, "reproduction_mode": "approximate", "candidates": rows,
                     "counts_by_method": {method: sum(method in r["methods"] for r in rows) for method in METHODS}}
            write_json(path, value)
        if any(r["quad"] is not None or r["neutral_fraction"] != 0
               or not set(r["methods"]) <= set(cfg["source_methods"]) for r in value["candidates"]):
            raise RuntimeError("TEST candidates contain prohibited auxiliary/final variations")
        prepared[sample.sample_id] = value
        print(f"[TEST crops] {position}/{len(samples)} {sample.sample_id} crops={len(value['candidates'])}", flush=True)
    return prepared


def evaluate_source(samples, prepared, readings, paths, cfg, output, partial):
    """TEST annotations enter only after all crop and original OCR artifacts exist."""
    summary = {method: {"correct": 0, "candidates": 0, "gt_words": 0} for method in
               cfg["source_methods"] + ["all_methods"]}
    for sample in samples:
        key = crops.key(sample.sample_id)
        image = _image(sample, paths["data_root"])
        gt, annotation_sha = _ground_truth(sample, paths["data_root"], (image.shape[1], image.shape[0]))
        arms = {}
        for method, total in summary.items():
            rows = [r for r in prepared[sample.sample_id]["candidates"]
                    if method == "all_methods" or method in r["methods"]]
            result = evaluate_pool(rows, readings[sample.sample_id], gt, cfg["matching_iou"])
            arms[method] = result
            total["correct"] += result["correct"]
            total["candidates"] += result["candidates"]
            total["gt_words"] += len(gt)
        crops.freeze(output / "evaluation" / f"{key}.json", {
            "sample_id": sample.sample_id, "annotation_sha256": annotation_sha, "ground_truth": gt,
            "arms": arms, "prepared_sha256": sha(output / "prepared" / f"{key}.json"),
            "ocr_sha256": sha(output / "ocr" / f"{key}.json")})
    if not partial and summary["baseline"]["gt_words"] != cfg["test_care_words"]:
        raise RuntimeError("full TEST care-word denominator changed")
    crops.freeze(output / "results.json", {"schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": cfg["claim_label"], "result_class": "SMOKE_TEST" if partial else "APPROXIMATE_TEST_DIAGNOSTIC",
        "plan_sha256": sha(output / "resolved_configuration.json"), "images": len(samples),
        "summary": summary, "table_5_claim": False})


def qualify_reusable_source(source, cfg, config_path, manifest_path, samples, device, local):
    """Reuse complete original TEST inference after an isolated selector fix."""
    plan_path = source / "resolved_configuration.json"
    result_path = source / "results.json"
    if not plan_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"reusable TEST source is incomplete: {source}")
    plan = json.loads(plan_path.read_text())
    result = json.loads(result_path.read_text())
    if (plan.get("reproduction_mode") != "approximate"
        or plan["config"].get("split") != "official_TEST_300"
        or plan["config_sha256"] != sha(config_path)
        or plan["manifest_sha256"] != sha(manifest_path)
        or plan["sample_ids"] != [sample.sample_id for sample in samples]
        or plan["execution"]["device"] != device
        or plan["code_sha256"].get("scripts/run_totaltext_crop_group_test_suite.py")
           != REUSABLE_SOURCE_RUNNER_SHA256
        or result["plan_sha256"] != sha(plan_path)
        or result["images"] != len(samples)
        or result["summary"]["baseline"]["gt_words"] != cfg["test_care_words"]
        or result["table_5_claim"] is not False):
        raise RuntimeError(f"reusable TEST source contract differs: {source}")
    for field in ("data_root", "training_runs", "parseq_source", "parseq_archive", "parseq_checkpoint"):
        if plan["paths"][field] != str(local[field]):
            raise RuntimeError(f"reusable TEST source path changed: {field}")
    # Source production, OCR, and metric implementations must have identical
    # bytes. The new CLIP group runtime is solely a downstream selector input.
    for name in crops.CODE_FILES:
        if plan["code_sha256"].get(name) != sha(REPO / name):
            raise RuntimeError(f"reusable TEST source dependency changed: {name}")
    required = ("prepared", "ocr", "evaluation", "tiles")
    if any(len(list((source / name).glob("*.json"))) != len(samples) for name in required):
        raise RuntimeError("reusable TEST source per-image shards are incomplete")
    return plan


def reuse_cross_scores(source, group_dir, output, sample_ids, device):
    """Copy verified frozen PARSeq scores; CLIP scores require recomputation.

    The two pinned runtime source hashes differ only in the CLIP image-channel
    declaration. All other selector source files and score contracts must match.
    """
    prior = source.parent / "selectors"
    prior_plan_path = prior / "resolved_configuration.json"
    if not prior_plan_path.is_file():
        return 0
    old = json.loads(prior_plan_path.read_text())
    runtime = "recognition/group_word_selection_runtime.py"
    if (old.get("source") != str(source) or old.get("grouping") != str(group_dir)
        or old.get("source_plan_sha256") != sha(source / "resolved_configuration.json")
        or old.get("grouping_plan_sha256") != sha(group_dir / "resolved_configuration.json")
        or old.get("config_sha256") != sha(selectors.DEFAULT_CONFIG)
        or old.get("sample_ids") != sample_ids
        or old["execution"]["device"] != device
        or old["code_sha256"].get(runtime) != PRIOR_SELECTOR_RUNTIME_SHA256
        or sha(REPO / runtime) != RGB_FIXED_SELECTOR_RUNTIME_SHA256
        or json.loads((prior / "cross_parseq_last_execution.json").read_text()).get("status") != "SUCCESS"):
        raise RuntimeError("previous cross-PARSeq score provenance differs")
    for name, digest in old["code_sha256"].items():
        if name != runtime and sha(REPO / name) != digest:
            raise RuntimeError(f"previous cross-PARSeq source changed: {name}")
    scores = prior / "cross_parseq"
    if len(list(scores.glob("*.json"))) != len(sample_ids):
        raise RuntimeError("previous cross-PARSeq scores are incomplete")
    total = 0
    for sid in sample_ids:
        filename = f"{crops.key(sid)}.json"
        old_input = prior / "inputs" / filename
        record = json.loads((scores / filename).read_text())
        if record["contract"]["input_sha256"] != sha(old_input):
            raise RuntimeError(f"previous cross-PARSeq input changed: {sid}")
        crops.freeze(output / "cross_parseq" / filename, record)
        total += len(record["rows"])
    return total


def run(args):
    cfg = read_config(args.config)
    local = selectors.resolve_local(args.paths)
    samples = test_samples(local["test_manifest"], cfg, args.limit)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "cuda:0") or (device.startswith("cuda") and not torch.cuda.is_available()):
        raise ValueError("select cpu or an available cuda:0")
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    cv2.setNumThreads(1)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training = training_config()
    if sha(local["train_manifest"]) != training["train_manifest_sha256"]:
        raise RuntimeError("TRAIN qualification manifest changed")
    train_manifest = load_segmentation_manifest(local["train_manifest"], require_checksums=True)
    if train_manifest.split_role != "train":
        raise RuntimeError("training qualification manifest is not TRAIN")
    _, _, validation_sha = _split_indices(train_manifest, training["training"]["seed"], 125)
    for path, expected in ((local["training_runs"] / "letterbox/best.pt", cfg["segmentation_checkpoint_sha256"]),
                           (local["parseq_checkpoint"], PARSEQ_CHECKPOINT_SHA256),
                           (local["parseq_archive"], PARSEQ_RUNTIME_ARCHIVE_SHA256)):
        if sha(path) != expected:
            raise RuntimeError(f"qualified model artifact changed: {path}")
    # Fail before expensive segmentation if the complete selector setup is unavailable.
    selector_cfg = selectors.config(selectors.DEFAULT_CONFIG)
    if local["clip_model_dir"].name != selector_cfg["clip"]["revision"]:
        raise ValueError("CLIP local snapshot must match the full pinned revision")
    clip_identity = selectors.clip_snapshot_file_hashes(local["clip_model_dir"])
    if not {"pytorch_model.bin", "model.safetensors"} & set(clip_identity):
        raise FileNotFoundError("local CLIP weights missing")
    plan = {"schema_version": 1, "reproduction_mode": "approximate", "config": cfg,
            "claim_label": cfg["claim_label"], "config_sha256": sha(args.config),
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TEST_DIAGNOSTIC",
            "sample_ids": [s.sample_id for s in samples], "manifest_sha256": sha(local["test_manifest"]),
            "validation_ids_sha256": validation_sha, "paths_profile_sha256": sha(args.paths),
            "paths_profile": {k: str(v) for k, v in local.items()}, "clip_files_sha256": clip_identity,
            "group_config_sha256": sha(GROUP_CONFIG), "selector_config_sha256": sha(selectors.DEFAULT_CONFIG),
            "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
            "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
            "code_sha256": {name: sha(REPO / name) for name in set(crops.CODE_FILES + selectors.CODE + grouping_runner.CODE +
                 ("scripts/run_totaltext_crop_group_test_suite.py",))},
            "execution": {"device": device, "precision": "float32", **cfg["execution"],
                "python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                "numpy": np.__version__, "opencv": cv2.__version__, "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
            "ground_truth_use": "post-hoc only; no TEST parameter fitting", "table_5_claim": False}
    identity = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    base = args.output.resolve() if args.output else local["output_root"] / identity
    if base == REPO or REPO in base.parents:
        raise ValueError("generated TEST artifacts must remain outside repository")
    reuse = local.get("reusable_source")
    source = reuse if reuse and reuse.exists() and args.limit is None else base / "source"
    reused_source = source == reuse
    paths = {k: v for k, v in local.items() if k in crops.PATH_FIELDS}
    paths.update(probability_cache=source / "probability_cache", baseline_ocr_cache=source / "ocr_by_crop")
    protocol = {"schema_version": 1, "split": cfg["split"], "checkpoint_sha256": cfg["segmentation_checkpoint_sha256"],
                "manifest_sha256": sha(local["test_manifest"]), "experiment_identity": identity, "device": device}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    plan.update(paths={k: str(v) for k, v in paths.items()}, probability_sidecar_contract_sha256=protocol_sha)
    if reused_source:
        qualified = qualify_reusable_source(source, cfg, args.config, local["test_manifest"], samples, device, local)
        plan["reused_source_plan_sha256"] = sha(source / "resolved_configuration.json")
        plan["reused_source_runner_sha256"] = qualified["code_sha256"]["scripts/run_totaltext_crop_group_test_suite.py"]
        new = 0
        print(f"Reusing complete TEST crop/OCR source: {source}", flush=True)
    else:
        crops.freeze(source / "resolved_configuration.json", plan)
        crops.freeze(paths["probability_cache"] / "protocol.json", protocol)
        print(f"Official TEST: {len(samples)} images; output={base}; device={device}", flush=True)
        with crops.measured_stage(source, "whole_segmentation", device, {"input_shape": [1, 3, 512, 512], "batch_size": 1}):
            whole_probabilities(samples, paths, cfg, validation_sha, source, protocol_sha, device)
        with crops.measured_stage(source, "tile_segmentation", device, {"input_shape": [1, 3, 512, 512], "batch_size": 1}):
            tiles = crops.run_tiles(samples, paths, cfg, sha(source / "resolved_configuration.json"), source, device, validation_sha)
        with crops.measured_stage(source, "prepare", "cpu", {"models": []}):
            prepared = prepare(samples, paths, cfg, source, sha(source / "resolved_configuration.json"), protocol_sha, tiles)
        with crops.measured_stage(source, "parseq", device, {"input_shape": [cfg["execution"]["parseq_batch_size"], 3, 32, 128], "batch_size": cfg["execution"]["parseq_batch_size"]}):
            readings, new = crops.run_ocr(samples, prepared, paths, source, device, cfg["execution"]["parseq_batch_size"])
        with crops.measured_stage(source, "evaluation", "cpu", {"models": []}):
            evaluate_source(samples, prepared, readings, paths, cfg, source, args.limit is not None)
    crops.freeze(base / "resolved_configuration.json", plan)
    with crops.measured_stage(base, "grouping", "cpu", {"models": []}):
        group_dir = grouping_runner.run(argparse.Namespace(config=GROUP_CONFIG, source=source,
            paths=args.paths, output=None, limit=args.limit))
    reused_cross_pairs = reuse_cross_scores(source, group_dir, base / "selectors",
        [sample.sample_id for sample in samples], device) if reused_source else 0
    if reused_cross_pairs:
        print(f"Reusing {reused_cross_pairs} frozen PARSeq text/crop scores; recalculating all CLIP images", flush=True)
    with crops.measured_stage(base, "selectors", device, {"models": ["PARSeq", "CLIP"], "model_loading": "sequential"}):
        result_dir = selectors.run(argparse.Namespace(config=selectors.DEFAULT_CONFIG, paths=args.paths,
            source=source, grouping=group_dir, grouping_variant=None, image_manifest=local["test_manifest"],
            clip_model_dir=local["clip_model_dir"], device=device, output=base / "selectors", limit=args.limit, cached_only=False))
    write_json(base / "last_execution.json", {"status": "SUCCESS", "new_ocr_crops": new,
        "reused_source": reused_source, "reused_cross_pairs": reused_cross_pairs,
        "source": str(source),
        "results": str(result_dir / "results.json"), "summary": str(result_dir / "summary.md")})
    print(f"TEST comparison: {result_dir / 'summary.md'}", flush=True)
    return result_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--paths", type=Path, default=PATHS)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="explicit smoke test only; omit for all 300 images")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
