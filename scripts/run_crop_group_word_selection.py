#!/usr/bin/env python3
"""Run dataset-independent selectors on frozen crop-group artifacts, then evaluate."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import cv2

from recognition.group_word_selection import (CHEAP_METHODS, ProbabilityMass, add_clip_scores,
    add_cross_scores, aligned_probability, cross_requests, group_evidence, normalize, select_groups)
from recognition.group_word_selection_runtime import clip_group_scores, forced_likelihood_batch
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from recognition.crop_word_group_metric import grouped_oracle
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.analyze_totaltext_crop_word_groups import discover_source, freeze
from scripts.crop_clip_runtime import (_cuda_peak, cached, clip_snapshot_file_hashes,
                                      read_image, sha, write_json)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _load_probability
from scripts.run_totaltext_crop_recovery_suite import read_tile
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/crop_group_word_selection_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/crop_group_word_selection_local_paths_v1.json"
CODE = ("recognition/group_word_selection.py", "recognition/group_word_selection_runtime.py",
        "scripts/run_crop_group_word_selection.py", "recognition/parseq.py", "scripts/crop_clip_runtime.py",
        "recognition/crop_word_group_metric.py",
        "scripts/analyze_totaltext_crop_word_groups.py", "recognition/scene_text_v1_evaluation.py",
        "scripts/evaluate_totaltext_highres_contour_padding.py",
        "scripts/evaluate_totaltext_letterbox_threshold_sweep_train.py",
        "scripts/run_totaltext_crop_recovery_suite.py", "segmentation/data.py")


def validate_config(value: dict) -> None:
    if (value.get("schema_version") != 1 or value.get("reproduction_mode") != "approximate"
        or value.get("table_5_claim") is not False or tuple(value.get("cheap_methods", [])) != CHEAP_METHODS
        or value["execution"]["precision"] != "float32" or value["evaluation"]["matching_iou"] != .5
        or value["combination"] != "equal-weight geometric means of dimensionless scores in [0,1]"
        or "weights" in value or "dataset" in value
        or value["clip"]["views"] != ["full_image", "group_reference_crop"]
        or not re.fullmatch(r"[0-9a-f]{40}", value["clip"]["revision"])):
        raise ValueError("invalid isolated approximate general selector configuration")
    for section, fields in (("cross_parseq", ("word_budget", "crop_budget")),
                            ("execution", ("cpu_threads", "cross_batch_size", "clip_batch_size"))):
        if any(not isinstance(value[section][field], int) or value[section][field] < 1 for field in fields):
            raise ValueError("budgets/batches/threads must be positive integers")
    if not value["clip"]["templates"] or any(template.count("{text}") != 1 for template in value["clip"]["templates"].values()):
        raise ValueError("CLIP prompts must include exactly one {text}")


def config(path: Path) -> dict:
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("selector configuration must belong to configs/approximate")
    value = json.loads(path.read_text())
    validate_config(value)
    return value


def resolve_local(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("path_base") != "repository_root":
        raise ValueError("invalid local path profile")
    return {field: (Path(item) if Path(item).is_absolute() else REPO / item).resolve()
            for field, item in value.items() if field not in ("schema_version", "path_base")}


def discover_grouping(source: Path, variant: str) -> Path:
    eligible = []
    for path in source.glob("word_groups/*/results.json"):
        value = json.loads(path.read_text())
        source_ids = json.loads((source / "resolved_configuration.json").read_text())["sample_ids"]
        if variant in value.get("summary", {}) and value.get("images") == len(source_ids):
            eligible.append(path)
    if not eligible:
        raise FileNotFoundError("no completed crop grouping found; first run bash scripts/test_totaltext_groups.sh or pass --grouping")
    return max(eligible, key=lambda path: (path.stat().st_mtime_ns, str(path))).parent


@contextmanager
def measured(output: Path, stage: str, device: str, details: dict):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    try:
        yield
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "stage": stage,
            "reproduction_mode": "approximate", "error": str(error), "error_type": type(error).__name__,
            "out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError), "traceback": traceback.format_exc(),
            "device": device, "precision": "float32", "peak": _cuda_peak(device), **details})
        raise
    else:
        write_json(output / f"{stage}_last_execution.json", {"status": "SUCCESS", "seconds": time.monotonic() - start,
            "device": device, "precision": "float32", "peak": _cuda_peak(device), **details})


def crop_rgb(image, row):
    x1, y1, x2, y2 = row["bbox"]
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    if not crop.size or crop_array_hash(crop) != row["crop_rgb_sha256"]:
        raise RuntimeError(f"original crop pixels changed: {row['candidate_id']}")
    return crop


def prepare_context(sid, source, grouping, variant, source_plan, samples, output, selector_config):
    key = sid.replace(":", "__")
    prepared_path, ocr_path = source / "prepared" / f"{key}.json", source / "ocr" / f"{key}.json"
    grouping_path = grouping / "groups" / f"{key}.json"
    prepared, ocr, group_shard = (json.loads(path.read_text()) for path in (prepared_path, ocr_path, grouping_path))
    if (group_shard["sample_id"] != sid or group_shard["prepared_sha256"] != sha(prepared_path)
        or ocr["contract"]["prepared_sha256"] != sha(prepared_path)):
        raise RuntimeError(f"source grouping/OCR contract differs: {sid}")
    groups = group_shard["variants"][variant]["groups"]
    ids = {cid for group in groups for cid in group["candidate_ids"]}
    original = {row["candidate_id"]: row for row in prepared["candidates"] if row["candidate_id"] in ids}
    if set(original) != ids:
        raise RuntimeError("group references missing source crop")
    allowed = set(json.loads((grouping / "resolved_configuration.json").read_text())["config"]["source_methods"])
    frozen_ocr = {row["candidate_id"]: row for row in ocr["rows"]}
    rows, provenance, loaded = [], {}, {}
    for cid, row in sorted(original.items()):
        if row["quad"] is not None or row["neutral_fraction"] != 0:
            raise ValueError("selector battery accepts unmodified source crops only")
        crop_hash = row["crop_rgb_sha256"]
        if crop_hash not in loaded:
            origin = ocr["crop_cache_sources"][crop_hash]
            path = Path(origin["path"])
            if sha(path) != origin["file_sha256"]:
                raise RuntimeError(f"original OCR cache changed: {path}")
            reading = json.loads(path.read_text())
            if (reading["crop_rgb_sha256"] != crop_hash
                or reading["parseq_checkpoint_sha256"] != source_plan["parseq_checkpoint_sha256"]
                or reading["parseq_source_archive_sha256"] != source_plan["parseq_runtime_archive_sha256"]):
                raise RuntimeError("original PARSeq cache identity differs")
            loaded[crop_hash] = reading
            provenance[crop_hash] = {"path": str(path), "sha256": sha(path)}
        reading = loaded[crop_hash]
        if reading["raw_text"] != frozen_ocr[cid]["raw_text"]:
            raise RuntimeError("original OCR reading differs from aggregate")
        rows.append({"candidate_id": cid, "bbox": row["bbox"], "crop_rgb_sha256": crop_hash,
                     "methods": [method for method in row["methods"] if method in allowed],
                     "origins": [origin for origin in row["origins"] if origin["method"] in allowed],
                     "ocr": {field: reading[field] for field in
                             ("raw_text", "token_probabilities", "token_log_probabilities")}})
    sample = samples[sid]
    image, image_sha = read_image(Path(source_plan["paths"]["data_root"]), sample,
                                  prepared["contract"]["image_sha256"])
    probability_root = Path(source_plan["paths"]["probability_cache"]) / "probabilities"
    p_path, sidecar = probability_root / f"{key}.npy", probability_root / f"{key}.json"
    if sha(sidecar) != prepared["contract"]["probability_sidecar_sha256"]:
        raise RuntimeError("whole probability sidecar changed")
    whole = _load_probability(p_path, json.loads(sidecar.read_text()), sample=sample,
                               protocol_sha=source_plan["probability_sidecar_contract_sha256"])
    tiles_path = source / "tiles" / f"{key}.json"
    tiles, tile_provenance = [], []
    if prepared["contract"]["tile_shard_sha256"] is not None:
        if sha(tiles_path) != prepared["contract"]["tile_shard_sha256"]:
            raise RuntimeError("tile probability contract changed")
        for tile in json.loads(tiles_path.read_text())["tiles"]:
            tiles.append((read_tile(source, tile), tile["window"]))
            tile_provenance.append({"path": tile["path"], "sha256": tile["file_sha256"]})
    probability = aligned_probability(whole, tiles, (image.shape[1], image.shape[0]))
    evidence = group_evidence(rows, groups, allowed_methods=allowed,
                             mass=ProbabilityMass(probability, (image.shape[1], image.shape[0])))
    requests, cross_plan = cross_requests(evidence, rows,
        word_budget=selector_config["cross_parseq"]["word_budget"], crop_budget=selector_config["cross_parseq"]["crop_budget"])
    context = {"sample_id": sid, "image_sha256": image_sha, "rows": rows, "groups": groups,
               "evidence": evidence, "cross_requests": requests, "cross_plan": cross_plan,
               "contract": {"prepared_sha256": sha(prepared_path), "ocr_sha256": sha(ocr_path),
                            "group_shard_sha256": sha(grouping_path), "ocr_sources": provenance,
                            "whole_probability_sha256": sha(p_path), "tiles": tile_provenance,
                            "aligned_probability_sha256": sha256(probability.tobytes()).hexdigest()},
               "ground_truth_use": "none"}
    context_path = output / "inputs" / f"{key}.json"
    freeze(context_path, context)
    return context


def cross_stage(contexts, samples, source_plan, output, device, cfg):
    recognizer = None
    batch_size = cfg["execution"]["cross_batch_size"]
    try:
        for position, context in enumerate(contexts, 1):
            sid = context["sample_id"]
            key = sid.replace(":", "__")
            path = output / "cross_parseq" / f"{key}.json"
            contract = {"input_sha256": sha(output / "inputs" / f"{key}.json"), "device": device,
                        "batch_size": batch_size, "checkpoint": source_plan["parseq_checkpoint_sha256"],
                        "archive": source_plan["parseq_runtime_archive_sha256"], "scoring": cfg["cross_parseq"]}
            previous = cached(path, contract)
            if previous is None:
                if recognizer is None:
                    paths = source_plan["paths"]
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=Path(paths["parseq_source"]), runtime_source_archive=Path(paths["parseq_archive"]),
                        runtime_source_type=PARSEQ_ARCHIVE_SOURCE, checkpoint=Path(paths["parseq_checkpoint"]), device=device))
                image, _ = read_image(Path(source_plan["paths"]["data_root"]), samples[sid], context["image_sha256"])
                by_id = {row["candidate_id"]: row for row in context["rows"]}
                results = []
                requests = context["cross_requests"]
                for start in range(0, len(requests), batch_size):
                    batch = requests[start:start + batch_size]
                    crops = [crop_rgb(image, by_id[row["candidate_id"]]) for row in batch]
                    values = forced_likelihood_batch(recognizer, crops,
                        [row["raw_text"] for row in batch], [row["crop_rgb_sha256"] for row in batch])
                    results.extend({"request_id": row["request_id"], **value} for row, value in zip(batch, values))
                previous = {"contract": contract, "rows": results, "model_peak": _cuda_peak(device)}
                write_json(path, previous)
            if [row["request_id"] for row in previous["rows"]] != [row["request_id"] for row in context["cross_requests"]]:
                raise RuntimeError("cross PARSeq request ordering differs")
            add_cross_scores(context["evidence"], context["cross_plan"],
                             {row["request_id"]: row["mean_log_probability_with_eos"] for row in previous["rows"]})
            unsupported = sum(row.get("status") == "UNSUPPORTED_LENGTH" for row in previous["rows"])
            print(f"[cross-PARSeq] {position}/{len(contexts)} {sid} pairs={len(previous['rows'])} unsupported_length={unsupported}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
        del recognizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def clip_stage(contexts, samples, source_plan, output, device, cfg, model_dir, identity):
    model = processor = None
    batch_size = cfg["execution"]["clip_batch_size"]
    methods = {f"clip_{view}_{template}" for view in ("full", "group") for template in cfg["clip"]["templates"]}
    try:
        for position, context in enumerate(contexts, 1):
            sid = context["sample_id"]
            key = sid.replace(":", "__")
            path = output / "clip" / f"{key}.json"
            contract = {"input_sha256": sha(output / "inputs" / f"{key}.json"), "clip": identity,
                        "device": device, "batch_size": batch_size, "precision": "float32"}
            previous = cached(path, contract)
            if previous is None:
                if model is None:
                    from transformers import CLIPModel, CLIPProcessor
                    model, loading = CLIPModel.from_pretrained(str(model_dir), local_files_only=True,
                        use_safetensors=(model_dir / "model.safetensors").exists(), output_loading_info=True)
                    if any(loading.get(field) for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                        raise RuntimeError(f"CLIP checkpoint loading is incomplete: {loading}")
                    model = model.eval().float().to(device)
                    vision = model.config.vision_config
                    if (vision.patch_size != 32 or vision.hidden_size != 768
                        or vision.num_hidden_layers != 12 or model.config.projection_dim != 512):
                        raise RuntimeError("CLIP is not the configured qualified ViT-B/32")
                    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
                image, _ = read_image(Path(source_plan["paths"]["data_root"]), samples[sid], context["image_sha256"])
                scores = clip_group_scores(model, processor, image, context["evidence"],
                    templates=cfg["clip"]["templates"], device=device, batch_size=batch_size)
                previous = {"contract": contract, "scores": scores, "model_peak": _cuda_peak(device),
                            "checkpoint_loading": "checked missing/unexpected/mismatched keys and errors"}
                write_json(path, previous)
            methods.update(add_clip_scores(context["evidence"], previous["scores"]))
            print(f"[CLIP full/group] {position}/{len(contexts)} {sid}", flush=True)
    finally:
        del model, processor
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return sorted(methods)


def evaluate_predictions(predictions: list[dict], gt: list[dict], iou_min: float) -> dict:
    edges = [(pi, gi) for pi, prediction in enumerate(predictions) for gi, word in enumerate(gt)
             if normalize(word["text"]) and normalize(prediction["raw_text"]) == normalize(word["text"])
             and box_iou(prediction["bbox"], word["bbox"]) >= iou_min]
    matched = maximum_matching(len(predictions), len(gt), edges)
    return {"predictions": len(predictions), "correct": len(matched), "correct_gt_indices": sorted(matched),
            "unmatched_predictions": len(predictions) - len(matched)}


def run(args):
    cfg = config(args.config)
    local = resolve_local(args.paths)
    variant = args.grouping_variant or cfg["grouping_variant"]
    if args.grouping:
        grouping = args.grouping.resolve()
        group_plan = json.loads((grouping / "resolved_configuration.json").read_text())
        source = Path(group_plan["source"])
        if args.source and args.source.resolve() != source:
            raise ValueError("--source differs from --grouping source")
    else:
        source = args.source.resolve() if args.source else discover_source(local["crop_suite_paths"])
        grouping = discover_grouping(source, variant)
        group_plan = json.loads((grouping / "resolved_configuration.json").read_text())
    source_plan = json.loads((source / "resolved_configuration.json").read_text())
    group_result = json.loads((grouping / "results.json").read_text())
    if (source_plan["reproduction_mode"] != "approximate" or group_plan["reproduction_mode"] != "approximate"
        or group_plan["source_plan_sha256"] != sha(source / "resolved_configuration.json")
        or group_result["plan_sha256"] != sha(grouping / "resolved_configuration.json")
        or variant not in group_result["summary"]):
        raise RuntimeError("source/grouping experiment contract differs")
    manifest_path = args.image_manifest.resolve() if args.image_manifest else local["image_manifest"]
    if sha(manifest_path) != source_plan["manifest_sha256"]:
        raise RuntimeError("image manifest differs from source run")
    samples = {sample.sample_id: sample for sample in load_segmentation_manifest(manifest_path, require_checksums=True).samples}
    ids = group_plan["sample_ids"]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        ids = ids[:args.limit]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device != "cpu" and device != "cuda:0":
        raise ValueError("single-device run requires cpu or cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(cfg["execution"]["seed"])
    np.random.seed(cfg["execution"]["seed"])
    torch.manual_seed(cfg["execution"]["seed"])
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_dir = args.clip_model_dir.resolve() if args.clip_model_dir else local["clip_model_dir"]
    clip_identity = None
    if not args.cached_only:
        if model_dir.name != cfg["clip"]["revision"]:
            raise ValueError("local CLIP directory must be named by its pinned full commit SHA")
        files = clip_snapshot_file_hashes(model_dir)
        if (not all(name in files for name in ("config.json", "preprocessor_config.json", "tokenizer.json"))
            or not {"model.safetensors", "pytorch_model.bin"} & set(files)):
            raise FileNotFoundError("qualified local CLIP snapshot incomplete; pass --clip-model-dir")
        clip_identity = {**cfg["clip"], "files_sha256": files}
        for field, expected in (("parseq_checkpoint", PARSEQ_CHECKPOINT_SHA256), ("parseq_archive", PARSEQ_RUNTIME_ARCHIVE_SHA256)):
            if sha(Path(source_plan["paths"][field])) != expected:
                raise RuntimeError(f"qualified PARSeq artifact changed: {field}")
    plan = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": cfg["claim_label"],
            "table_5_claim": False, "result_class": "SMOKE_TEST" if args.limit else (
                "APPROXIMATE_TEST_DIAGNOSTIC" if source_plan["config"]["split"] == "official_TEST_300"
                else "APPROXIMATE_TRAIN_DIAGNOSTIC"),
            "source": str(source), "grouping": str(grouping), "grouping_variant": variant,
            "dataset": source_plan["config"]["dataset"], "split": source_plan["config"]["split"],
            "source_plan_sha256": sha(source / "resolved_configuration.json"),
            "grouping_plan_sha256": sha(grouping / "resolved_configuration.json"),
            "sample_ids": ids, "config": cfg, "config_sha256": sha(args.config),
            "paths_profile_sha256": sha(args.paths), "manifest_sha256": sha(manifest_path),
            "clip_identity": clip_identity, "cached_only": args.cached_only,
            "software": {"opencv": cv2.__version__, **{package: importlib.metadata.version(package) for package in
                         ("torch", "transformers", "numpy", "scipy", "Pillow")}},
            "code_sha256": {path: sha(REPO / path) for path in CODE},
            "execution": {"device": device, **cfg["execution"], "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                          "torch": torch.__version__, "cuda_build": torch.version.cuda, "python": sys.version,
                          "numpy": np.__version__, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                          "deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True},
            "ground_truth_use": "evaluation after all selection artifacts are frozen; never used for scores or parameter choice"}
    identity = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() if args.output else local["output_root"] / identity
    if output == REPO or REPO in output.parents:
        raise ValueError("generated artifacts must stay outside the repository")
    freeze(output / "resolved_configuration.json", plan)
    print(f"Output: {output}; device={device}; images={len(ids)}; cached_only={args.cached_only}", flush=True)
    contexts = []
    with measured(output, "prepare", "cpu", {"model_set": []}):
        for position, sid in enumerate(ids, 1):
            contexts.append(prepare_context(sid, source, grouping, variant, source_plan, samples, output, cfg))
            print(f"[prepare selectors] {position}/{len(ids)} {sid}", flush=True)
    methods = list(CHEAP_METHODS)
    if not args.cached_only:
        with measured(output, "cross_parseq", device, {"model_set": ["PARSeq"], "input_shape": [cfg["execution"]["cross_batch_size"], 3, 32, 128]}):
            cross_stage(contexts, samples, source_plan, output, device, cfg)
        methods.append("cross_parseq")
        with measured(output, "clip", device, {"model_set": ["CLIP ViT-B/32"], "batch_size": cfg["execution"]["clip_batch_size"], "input_shape": [cfg["execution"]["clip_batch_size"], 3, 224, 224]}):
            methods.extend(clip_stage(contexts, samples, source_plan, output, device, cfg, model_dir, clip_identity))
    with measured(output, "selection", "cpu", {"model_set": []}):
        for context in contexts:
            predictions = {method: select_groups(context["evidence"], method) for method in methods}
            freeze(output / "selection" / f"{context['sample_id'].replace(':', '__')}.json",
                   {"sample_id": context["sample_id"], "evidence": context["evidence"], "methods": predictions,
                    "ground_truth_use": "none", "input_sha256": sha(output / "inputs" / f"{context['sample_id'].replace(':', '__')}.json")})
    # All images/methods have been selected before any evaluation shard is opened.
    totals, per_image, denominator, oracle, shortlist_oracle = {}, [], 0, 0, 0
    with measured(output, "evaluation", "cpu", {"model_set": []}):
        for context in contexts:
            sid = context["sample_id"]
            key = sid.replace(":", "__")
            upstream_path = source / "evaluation" / f"{key}.json"
            upstream = json.loads(upstream_path.read_text())
            if (upstream["prepared_sha256"] != context["contract"]["prepared_sha256"]
                or upstream["ocr_sha256"] != context["contract"]["ocr_sha256"]):
                raise RuntimeError("evaluation source changed after selection")
            gt = upstream["ground_truth"]
            denominator += len(gt)
            upstream_group_path = grouping / "evaluation" / f"{key}.json"
            upstream_group = json.loads(upstream_group_path.read_text())
            if (upstream_group["groups_sha256"] != context["contract"]["group_shard_sha256"]
                or upstream_group["source_evaluation_sha256"] != sha(upstream_path)):
                raise RuntimeError("group evaluation contract differs from selected inputs")
            image_oracle = upstream_group["variants"][variant]["correct"]
            oracle += image_oracle
            short_groups = []
            for group in context["evidence"]:
                words = context["cross_plan"][group["group_id"]]["words"]
                short_groups.append({"candidate_ids": [cid for cid, item in group["candidates"].items()
                                                       if item["word"] in words]})
            short_correct = len(grouped_oracle(short_groups,
                {int(gi): ids for gi, ids in upstream["arms"]["all_methods"]["witnesses"].items()}, len(gt)))
            shortlist_oracle += short_correct
            selected_path = output / "selection" / f"{key}.json"
            selections = json.loads(selected_path.read_text())["methods"]
            baseline_ids = {row["candidate_id"] for row in context["rows"] if "baseline" in row["methods"]}
            selections["baseline_original"] = [{"raw_text": row["ocr"]["raw_text"], "bbox": row["bbox"]}
                                               for row in context["rows"] if row["candidate_id"] in baseline_ids]
            scored = {method: evaluate_predictions(predictions, gt, cfg["evaluation"]["matching_iou"])
                      for method, predictions in selections.items()}
            if any(values["correct"] > image_oracle for method, values in scored.items() if method != "baseline_original"):
                raise RuntimeError("selector exceeds its distinct-crop/group oracle")
            baseline = set(scored["baseline_original"]["correct_gt_indices"])
            for method, result in scored.items():
                result["recovered_vs_baseline"] = len(set(result["correct_gt_indices"]) - baseline)
                result["lost_vs_baseline"] = len(baseline - set(result["correct_gt_indices"]))
                totals.setdefault(method, Counter()).update({k: v for k, v in result.items() if isinstance(v, int)})
            evaluation_path = output / "evaluation" / f"{key}.json"
            freeze(evaluation_path, {"sample_id": sid, "selection_sha256": sha(selected_path),
                                    "source_evaluation_sha256": sha(upstream_path), "gt_words": len(gt), "methods": scored})
            per_image.append(str(evaluation_path.relative_to(output)))
    for result in totals.values():
        result["correct_rate"] = result["correct"] / denominator if denominator else 0.
        result["exact_precision"] = result["correct"] / result["predictions"] if result["predictions"] else 0.
    peaks = {}
    for stage in ("cross_parseq", "clip"):
        if not args.cached_only:
            records = [json.loads((output / stage / f"{sid.replace(':', '__')}.json").read_text())["model_peak"] for sid in ids]
            peaks[stage] = {field: max((r[field] for r in records if r.get(field) is not None), default=None)
                            for field in ("peak_allocated_bytes", "peak_reserved_bytes")}
    unsupported_cross = 0 if args.cached_only else sum(
        row.get("status") == "UNSUPPORTED_LENGTH" for sid in ids for row in
        json.loads((output / "cross_parseq" / f"{sid.replace(':', '__')}.json").read_text())["rows"])
    result = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": cfg["claim_label"],
              "result_class": plan["result_class"], "table_5_claim": False,
              "plan_sha256": sha(output / "resolved_configuration.json"), "images": len(ids), "gt_words": denominator,
              "group_oracle_correct": oracle, "cached_only": args.cached_only, "summary": {k: dict(v) for k, v in totals.items()},
              "cross_shortlist_oracle_correct": shortlist_oracle,
              "cross_parseq_unsupported_length_pairs": unsupported_cross,
              "per_image": per_image, "creation_model_peaks": peaks,
              "skipped_methods": ["cross_parseq", "clip_full_raw", "clip_full_written", "clip_group_raw", "clip_group_written"] if args.cached_only else [],
              "parameter_fitting": False, "normalization": cfg["evaluation"]["normalization"]}
    freeze(output / "results.json", result)
    lines = ["# Crop-group word selection", "", cfg["claim_label"], "",
             f"Images: {len(ids)}; care words: {denominator}; grouped oracle: {oracle}. Parameter fitting: none.", "",
             "| Method | Correct | Predictions | Recovered vs baseline | Lost vs baseline | Exact precision |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, values in totals.items():
        lines.append(f"| {method} | {values['correct']} | {values['predictions']} | {values['recovered_vs_baseline']} | {values['lost_vs_baseline']} | {values['exact_precision']:.4f} |")
    lines += ["", "All GT is post-hoc. No claim of transfer until independent evaluation.",
              "CLIP/cross PARSeq stages skipped explicitly." if args.cached_only else "All configured selectors executed."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(output / "last_execution.json", {"status": "SUCCESS", "methods": list(totals)})
    print(json.dumps({"report": str(output / "results.json"), "summary": result["summary"]}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--grouping", type=Path)
    parser.add_argument("--grouping-variant")
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--clip-model-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cached-only", action="store_true", help="explicitly skip CLIP and cross-PARSeq model inference")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
