"""Pinned-model, fixed-box PARSeq/CLIP/caption TRAIN diagnostic.

Run as python -m scripts.run_clip_word_verification --model vit_b32 --device cpu.
No packages or weights are downloaded, and missing artifacts fail explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from recognition.clip_word_verifier import auc, score_words, selections, word_pools
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.crop_clip_runtime import (_cuda_peak, cached, clip_snapshot_file_hashes,
                                       crop_from_row, read_image, sha, write_json)
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct, _train_gt
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
CODE = ("recognition/clip_word_verifier.py", "scripts/run_clip_word_verification.py",
        "recognition/group_word_selection_runtime.py", "scripts/crop_clip_runtime.py",
        "recognition/scene_text_v1_evaluation.py",
        "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py", "segmentation/data.py")


def load_config(path: Path) -> dict:
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("CLIP verification configuration must be isolated in configs/approximate")
    cfg = json.loads(path.read_text())
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
        or cfg.get("claim_label") != "approximate reproduction" or cfg.get("table_5_claim") is not False
        or cfg.get("candidate_pool") != "threshold_only_pad0_raw_parseq"
        or cfg.get("prediction_geometry") != "unchanged_threshold0.5_pad0_baseline"
        or cfg["execution"]["precision"] != "float32" or cfg["evaluation"]["matching_iou"] != .5):
        raise ValueError("invalid approximate fixed-box CLIP verification protocol")
    for field in ("batch_size", "cpu_threads", "seed"):
        if type(cfg["execution"][field]) is not int or cfg["execution"][field] < (0 if field == "seed" else 1):
            raise ValueError("invalid execution parameter")
    if (not np.isfinite(cfg["agreement_margin"]) or cfg["agreement_margin"] < 0
        or not np.isfinite(cfg["surround_expansion"]) or cfg["surround_expansion"] < 1
        or type(cfg["max_caption_words"]) is not int or cfg["max_caption_words"] < 1):
        raise ValueError("invalid context or gate parameters")
    if (any(v.count("{text}") != 1 for v in cfg["templates"].values())
        or cfg["context_template"].count("{text}") != 1 or cfg["context_template"].count("{caption}") != 1
        or not cfg["ensemble_templates"] or not set(cfg["ensemble_templates"]) <= cfg["templates"].keys()
        or not {"raw", "written"} <= cfg["templates"].keys()):
        raise ValueError("invalid CLIP prompt templates")
    for model in cfg["models"].values():
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("model revision must be a full immutable commit")
    return cfg


def paths_profile(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("path_base") != "repository_root":
        raise ValueError("invalid local path profile")
    def resolve(item):
        return (Path(item) if Path(item).is_absolute() else REPO / item).resolve()
    return {k: ({name: resolve(item) for name, item in v.items()} if k == "models" else resolve(v))
            for k, v in value.items() if k not in ("schema_version", "path_base")}


def prepare(cfg: dict, paths: dict, limit: int | None) -> tuple[list, list, dict]:
    source_plan = paths["source"] / "plan.json"
    if sha(source_plan) != cfg["source_plan_sha256"]:
        raise ValueError("frozen crop/PARSeq source plan changed")
    plan = json.loads(source_plan.read_text())
    if (plan["dataset"] != "total_text" or plan["reproduction_mode"] != "approximate"
        or plan["gt_used_in_selection"] is not False or len(plan["sample_ids"]) != cfg["expected_images"]
        or len(set(plan["sample_ids"])) != len(plan["sample_ids"])):
        raise ValueError("source must be the complete frozen Total-Text TRAIN population")
    if sha(paths["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("source TRAIN manifest changed")
    manifest = load_segmentation_manifest(paths["manifest"], require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("CLIP diagnostic only accepts the TRAIN manifest")
    samples = {sample.sample_id: sample for sample in manifest.samples}
    if sha(paths["captions"]) != cfg["captions_sha256"]:
        raise ValueError("frozen scene descriptions changed")
    captions = json.loads(paths["captions"].read_text())
    by_id = {row["sample_id"]: row for row in captions["samples"]}
    if (len(by_id) != len(captions["samples"]) or set(by_id) != set(plan["sample_ids"])
        or captions["test_loaded"] is not False or captions["reproduction_mode"] != "approximate"):
        raise ValueError("scene descriptions do not cover exactly the frozen TRAIN images")
    ids = plan["sample_ids"][:limit] if limit else plan["sample_ids"]
    contexts = []
    for sid in ids:
        key = sid.replace(":", "__")
        pp, op = paths["source"] / "prepared" / f"{key}.json", paths["source"] / "ocr" / f"{key}.json"
        prepared, ocr = json.loads(pp.read_text()), json.loads(op.read_text())
        contract = ocr["contract"]
        if (prepared["contract"]["plan_sha256"] != cfg["source_plan_sha256"]
            or prepared["contract"]["sample_id"] != sid or contract["prepared_sha256"] != sha(pp)
            or contract["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256
            or contract["parseq_source_archive_sha256"] != PARSEQ_RUNTIME_ARCHIVE_SHA256):
            raise ValueError(f"frozen PARSeq input contract differs: {sid}")
        expected = prepared["contract"]["image_sha256"]
        if by_id[sid]["image_sha256"] != expected or samples[sid].image_sha256 != expected:
            raise ValueError(f"caption/crop/original image checksum differs: {sid}")
        image, _ = read_image(paths["data_root"], samples[sid], expected)
        pools = word_pools(prepared, ocr)
        for pool in pools:
            crop_from_row(image, pool["reference"])
        # Shuffle over the FULL frozen population, even for a one-image smoke.
        next_sid = plan["sample_ids"][(plan["sample_ids"].index(sid)+1) % len(plan["sample_ids"])]
        contexts.append({"sample_id": sid, "image_sha256": expected, "pools": pools,
            "caption": by_id[sid]["text"], "shuffled_caption": by_id[next_sid]["text"],
            "shuffled_caption_sample_id": next_sid,
            "prepared_sha256": sha(pp), "ocr_sha256": sha(op), "ground_truth_use": "none"})
    if not limit and sum(len(c["pools"]) for c in contexts) != cfg["expected_anchors"]:
        raise ValueError("frozen anchor population changed")
    return [samples[sid] for sid in ids], contexts, captions


def model_identity(path: Path, spec: dict) -> dict:
    if path.name != spec["revision"]:
        raise ValueError("CLIP snapshot directory must be named by its full commit SHA")
    for field in (spec["weight_file"], "config.json", "preprocessor_config.json", "tokenizer.json"):
        if not (path / field).is_file():
            raise FileNotFoundError(f"pinned {spec['model_id']} snapshot missing {field}: {path}; download revision {spec['revision']}")
    hashes = clip_snapshot_file_hashes(path)
    if spec["weight_sha256"] and hashes[spec["weight_file"]] != spec["weight_sha256"]:
        raise ValueError("CLIP checkpoint checksum differs from declared artifact")
    return {**spec, "local_path": str(path), "files_sha256": hashes,
            "upstream_weight_checksum_verified": spec["weight_sha256"] is not None}


def evaluate(samples: list, contexts: list, shards: list, *, paths: dict, cfg: dict) -> dict:
    """Annotations are opened only after all GT-blind selections are persisted."""
    aggregate, per_image, diagnostic_rows = {}, [], []
    oracle_count = 0
    for sample, context, shard in zip(samples, contexts, shards):
        image, _ = read_image(paths["data_root"], sample, context["image_sha256"])
        gt, annotation_sha = _train_gt(sample, paths["data_root"], (image.shape[1], image.shape[0]))
        pools, selected = context["pools"], shard["selections"]
        predictions = {arm: [{"bbox": pool["reference"]["bbox"], "text": choices[str(pool["anchor_rank"])]}
                             for pool in pools] for arm, choices in selected.items()}
        matched = {arm: _correct(rows, gt, normalize=False) for arm, rows in predictions.items()}
        baseline = matched["baseline"]
        edges = [(pi, gi) for pi, pool in enumerate(pools) for gi, word in enumerate(gt)
                 if box_iou(pool["reference"]["bbox"], word["bbox"]) >= cfg["evaluation"]["matching_iou"]
                 and any(text and text.casefold() == word["text"].casefold() for text in pool["words"])]
        oracle = maximum_matching(len(pools), len(gt), edges)
        oracle_count += len(oracle)
        diagnostics = []
        for pool in pools:
            key = str(pool["anchor_rank"])
            overlaps = [gi for gi, word in enumerate(gt) if box_iou(pool["reference"]["bbox"], word["bbox"]) >= .5]
            row = {"sample_id": sample.sample_id, "anchor_rank": pool["anchor_rank"],
                   "overlapping_gt_indices": overlaps, "baseline_text": pool["baseline_text"],
                   "word_scores": shard["scored"]["scores"][key], "parseq_scores":
                   {word: entry["parseq"] for word, entry in pool["words"].items()},
                   "baseline_parseq": pool["baseline_parseq"]}
            if len(overlaps) == 1:
                target = gt[overlaps[0]]["text"]
                row.update({"target": target, "baseline_correct": bool(pool["baseline_text"])
                            and pool["baseline_text"].casefold() == target.casefold(),
                            "caption_contains_target": bool(re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)",
                                                                      context["caption"], flags=re.I))})
            diagnostics.append(row)
        diagnostic_rows.extend(diagnostics)
        for arm, ids in matched.items():
            entry = aggregate.setdefault(arm, Counter())
            entry.update({"exact": len(ids), "care_words": len(gt), "predictions": len(pools),
                "recovered": len(ids - baseline), "damaged": len(baseline - ids),
                "changed_words": sum(choices != selected["baseline"][key] for key, choices in selected[arm].items())})
        per_image.append({"sample_id": sample.sample_id, "annotation_sha256": annotation_sha,
            "ground_truth": gt, "predictions": predictions,
            "matched_gt_indices": {arm: sorted(ids) for arm, ids in matched.items()},
            "oracle_matched_gt_indices": sorted(oracle), "correctness_diagnostics": diagnostics})
    complete = len(samples) == cfg["expected_images"]
    if complete and (aggregate["baseline"]["exact"] != cfg["expected_baseline_exact"]
                     or aggregate["baseline"]["care_words"] != cfg["expected_care_words"]):
        raise ValueError("frozen baseline exactness/denominator changed")
    for entry in aggregate.values():
        entry["exact_rate"] = entry["exact"] / entry["care_words"] if entry["care_words"] else None
        entry["net"] = entry["recovered"] - entry["damaged"]
    features = sorted({feature for row in diagnostic_rows for scores in row["word_scores"].values() for feature in scores})
    correctness, pairs = {}, {}
    eligible = [row for row in diagnostic_rows if "target" in row]
    for feature in ["parseq", *features]:
        labels, values = [], []
        wins, count, edit1_wins, edit1_count = 0., 0, 0., 0
        for row in eligible:
            by_word = (row["parseq_scores"] if feature == "parseq" else
                       {word: scores[feature] for word, scores in row["word_scores"].items()})
            value = row["baseline_parseq"] if feature == "parseq" else by_word.get(row["baseline_text"])
            if value is not None:
                labels.append(row["baseline_correct"])
                values.append(value)
            good = [word for word, value in by_word.items() if word and value is not None
                    and word.casefold() == row["target"].casefold()]
            bad = [word for word, value in by_word.items() if word and value is not None
                   and word.casefold() != row["target"].casefold()]
            from rapidfuzz.distance import Levenshtein
            for a in good:
                for b in bad:
                    win = float(by_word[a] > by_word[b]) + .5 * float(by_word[a] == by_word[b])
                    wins += win
                    count += 1
                    if Levenshtein.distance(a.casefold(), b.casefold()) == 1:
                        edit1_wins += win
                        edit1_count += 1
        correctness[feature] = {"auroc": auc(labels, values), "anchors": len(values),
            "positive": sum(labels), "negative": len(labels)-sum(labels),
            "excluded_empty_or_missing_score": len(eligible)-len(values), "calibrated": False}
        pairs[feature] = {"correct_word_wins": wins, "pairs": count,
            "rate": wins/count if count else None, "one_edit_pairs": edit1_count,
            "one_edit_rate": edit1_wins/edit1_count if edit1_count else None}
    return {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "result_class": "APPROXIMATE_TRAIN_DIAGNOSTIC" if complete else "SMOKE_TEST",
        "table_5_claim": False, "test_loaded": False, "images": len(samples),
        "fixed_geometry_oracle_exact": oracle_count, "summary": aggregate,
        "baseline_correctness": correctness, "candidate_pair_discrimination": pairs,
        "correctness_geometry": {"eligible_anchors": len(eligible),
            "no_gt_overlap": sum(not r["overlapping_gt_indices"] for r in diagnostic_rows),
            "ambiguous_gt_overlap": sum(len(r["overlapping_gt_indices"])>1 for r in diagnostic_rows),
            "caption_contains_target_anchors": sum(r["caption_contains_target"] for r in eligible)},
        "per_image": per_image, "parameter_fitting": False, "ground_truth_used_in_selection": False}


def run(args: argparse.Namespace) -> Path:
    cfg, paths = load_config(args.config), paths_profile(args.paths)
    if args.model not in cfg["models"] or args.model not in paths["models"]:
        raise ValueError("model missing from explicit configuration/path profile")
    output = paths["output_root"] / args.model / (f"smoke-{args.limit}" if args.limit else "train125")
    stage = "preflight"
    try:
        if args.limit is not None and not 1 <= args.limit <= cfg["expected_images"]:
            raise ValueError("limit must be within the frozen TRAIN population")
        if args.device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", args.device):
            raise ValueError("explicit device must be cpu or cuda[:index]")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA is unavailable; no automatic CPU fallback")
        samples, contexts, captions = prepare(cfg, paths, args.limit)
        spec = cfg["models"][args.model]
        identity = model_identity(paths["models"][args.model], spec)
        seed = cfg["execution"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(cfg["execution"]["cpu_threads"])
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(cfg["execution"]["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        software = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "Pillow", "rapidfuzz")}
        plan = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TRAIN_DIAGNOSTIC",
            "resolved_config": cfg, "resolved_paths": {k: ({n: str(p) for n, p in v.items()} if k == "models" else str(v))
                                                        for k, v in paths.items()},
            "config_sha256": sha(args.config), "paths_profile_sha256": sha(args.paths),
            "model": identity, "device": args.device, "software": software, "python": platform.python_version(),
            "platform": platform.platform(), "code_sha256": {path: sha(REPO/path) for path in CODE},
            "caption_model": {k: captions[k] for k in ("model_id", "revision", "generation", "model_files_sha256", "runtime_sha256")},
            "inputs": contexts, "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "backends": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                         "cudnn_deterministic": True, "cudnn_benchmark": False, "tf32": False}}
        plan_path = output / "resolved_configuration.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("existing resolved configuration differs; use a new output_root")
        write_json(plan_path, plan)
        shards, model, processor = [], None, None
        for index, (sample, context) in enumerate(zip(samples, contexts), 1):
            shard_path = output / "scores" / (sample.sample_id.replace(":", "__") + ".json")
            contract = {"plan_sha256": sha(plan_path), "sample_id": sample.sample_id}
            shard = cached(shard_path, contract)
            if shard is None:
                if model is None:
                    stage = "model_loading"
                    from transformers import CLIPModel, CLIPProcessor
                    if args.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(args.device)
                    model, info = CLIPModel.from_pretrained(str(paths["models"][args.model]), local_files_only=True,
                        use_safetensors=spec["weight_file"].endswith("safetensors"), output_loading_info=True)
                    if any(info.values()):
                        raise RuntimeError(f"CLIP checkpoint is not an exact load: {info}")
                    vc = model.config.vision_config
                    actual = (vc.patch_size, vc.hidden_size, vc.num_hidden_layers, model.config.projection_dim)
                    if actual != tuple(spec[k] for k in ("patch_size", "hidden_size", "layers", "projection_dim")):
                        raise ValueError("CLIP architecture differs from declared model version")
                    model = model.eval().float().to(args.device)
                    processor = CLIPProcessor.from_pretrained(str(paths["models"][args.model]), local_files_only=True)
                stage = "CLIP_scoring"
                start = time.monotonic()
                image, _ = read_image(paths["data_root"], sample, context["image_sha256"])
                scored = score_words(model, processor, image, context["pools"], caption=context["caption"],
                    shuffled_caption=context["shuffled_caption"], config=cfg, device=args.device)
                selected = selections(scored, cfg)
                shard = {"contract": contract, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                    "scored": scored, "selections": selected, "ground_truth_use": "none",
                    "runtime": {"seconds": time.monotonic()-start, "cuda_peak": _cuda_peak(args.device),
                        "precision": "float32", "model_set": [spec["model_id"]], "device": args.device,
                        "batch_size": cfg["execution"]["batch_size"],
                        "processed_image_shape": [3, model.config.vision_config.image_size, model.config.vision_config.image_size],
                        "max_text_tokens": model.config.text_config.max_position_embeddings}}
                write_json(shard_path, shard)
            if shard["scored"]["pools"] != context["pools"] or shard["selections"] != selections(shard["scored"], cfg):
                raise RuntimeError("cached scores/selection contract differs")
            shards.append(shard)
            print(f"[{args.model}] {index}/{len(samples)} {sample.sample_id} anchors={len(context['pools'])}", flush=True)
        del model, processor
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        stage = "evaluation"
        result = evaluate(samples, contexts, shards, paths=paths, cfg=cfg)
        result.update({"plan_sha256": sha(plan_path), "model": identity, "device": args.device,
            "execution_seconds": sum(s["runtime"]["seconds"] for s in shards),
            "cuda_peak": {key: max((s["runtime"]["cuda_peak"][key] for s in shards
                                    if s["runtime"]["cuda_peak"][key] is not None), default=None)
                          for key in ("peak_allocated_bytes", "peak_reserved_bytes")}})
        write_json(output / "results.json", result)
        print(json.dumps({"results": str(output/"results.json"), "summary": result["summary"]}), flush=True)
        return output
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "stage": stage,
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "device": args.device, "precision": "float32", "batch_size": cfg["execution"]["batch_size"],
            "out_of_memory": isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)),
            "cuda_peak": _cuda_peak(args.device) if not args.device.startswith("cuda") or torch.cuda.is_available() else None})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO/"configs/approximate/clip_word_verification_train_v1.json")
    parser.add_argument("--paths", type=Path, default=REPO/"configs/approximate/clip_word_verification_local_paths_v1.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
