"""Extend a completed v1 CLIP run with isolated experimental verification rules."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from recognition.clip_innovative_verifier import (GlyphScorer, counterfactual_scores,
                                                 font_identity, innovative_selections)
from scripts import run_clip_word_verification as base
from scripts.crop_clip_runtime import _cuda_peak, cached, read_image, sha, write_json


REPO = base.REPO
CODE = ("recognition/clip_innovative_verifier.py", "scripts/run_clip_innovative_verification.py", *base.CODE)


def load_config(path: Path) -> tuple[dict, dict]:
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("innovative CLIP configuration must be in configs/approximate")
    config = json.loads(path.read_text())
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("claim_label") != "approximate reproduction" or config.get("table_5_claim") is not False
        or config["execution"]["precision"] != "float32"):
        raise ValueError("invalid approximate innovative CLIP configuration")
    for value in (config["font_size"], config["render_padding"], config["execution"]["batch_size"],
                  config["execution"]["cpu_threads"]):
        if type(value) is not int or value < 1:
            raise ValueError("invalid rendering/runtime parameter")
    for name in ("letter_effect_margin", "glyph_margin"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError("invalid experimental margin")
    for polarity in config["polarities"]:
        for name in ("foreground", "background"):
            if len(polarity[name]) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in polarity[name]):
                raise ValueError("invalid RGB rendering polarity")
        if polarity["foreground"] == polarity["background"]:
            raise ValueError("rendered letters must be visible")
    if not config["polarities"] or not config["font_sha256"]:
        raise ValueError("fonts and polarities must be declared")
    return config, base.load_config(REPO / config["base_config"])


def local_paths(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("path_base") != "repository_root":
        raise ValueError("invalid innovative path profile")
    def resolve(item):
        return (Path(item) if Path(item).is_absolute() else REPO / item).resolve()
    return {"output_root": resolve(value["output_root"]),
            "fonts": {name: resolve(p) for name, p in value["fonts"].items()}}


def checked_base(root: Path, base_config: dict, model_name: str) -> tuple[dict, dict, list, list]:
    plan = json.loads((root / "resolved_configuration.json").read_text())
    spec = base_config["models"][model_name]
    if (plan["reproduction_mode"] != "approximate" or plan["resolved_config"] != base_config
        or plan["model"]["model_id"] != spec["model_id"] or plan["model"]["revision"] != spec["revision"]):
        raise ValueError("base model/config differs from the declared experiment")
    if not (root / "results.json").is_file():
        raise FileNotFoundError("complete the v1 scoring/evaluation before innovative extensions")
    for name, expected in plan["code_sha256"].items():
        if sha(REPO/name) != expected:
            raise ValueError(f"base implementation changed since source run: {name}; run v1 in a new output_root")
    paths = {k: ({name: Path(p) for name, p in value.items()} if k == "models" else Path(value))
             for k, value in plan["resolved_paths"].items()}
    count = len(plan["inputs"])
    samples, contexts, _ = base.prepare(base_config, paths, count if count < base_config["expected_images"] else None)
    if contexts != plan["inputs"]:
        raise ValueError("base input provenance/pools changed")
    return plan, paths, samples, contexts


def run(args: argparse.Namespace) -> Path:
    config, base_config = load_config(args.config)
    paths = local_paths(args.paths)
    output = paths["output_root"] / args.model / ("counterfactual" if args.counterfactual_only else "all-methods")
    stage = "preflight"
    try:
        if args.device != "cpu" and args.device != "cuda:0":
            raise ValueError("choose explicit cpu or cuda:0")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA unavailable; no automatic device/precision substitution")
        source, source_paths, samples, contexts = checked_base(args.base_run, base_config, args.model)
        identity = {} if args.counterfactual_only else font_identity(config, paths["fonts"])
        execution = config["execution"]
        random.seed(execution["seed"])
        np.random.seed(execution["seed"])
        torch.manual_seed(execution["seed"])
        torch.set_num_threads(execution["cpu_threads"])
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(execution["seed"])
        torch.use_deterministic_algorithms(execution["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        plan = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "resolved_config": config, "base_config": base_config, "device": args.device,
            "counterfactual_only": args.counterfactual_only, "fonts": identity,
            "base_run": str(args.base_run.resolve()), "base_plan_sha256": sha(args.base_run/"resolved_configuration.json"),
            "base_results_sha256": sha(args.base_run/"results.json"), "model": source["model"],
            "sample_ids": [s.sample_id for s in samples], "code_sha256": {p: sha(REPO/p) for p in CODE},
            "software": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "Pillow")},
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "tf32": False,
            "source_shards_sha256": {s.sample_id: sha(args.base_run/"scores"/(s.sample_id.replace(":", "__")+".json"))
                                     for s in samples}, "ground_truth_use_in_selection": "none"}
        plan_path = output / "resolved_configuration.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("innovative resolved configuration differs; use a fresh output_root")
        write_json(plan_path, plan)
        shards, model, scorer = [], None, None
        for index, (sample, context) in enumerate(zip(samples, contexts), 1):
            key = sample.sample_id.replace(":", "__")
            original_path = args.base_run / "scores" / (key+".json")
            if sha(original_path) != plan["source_shards_sha256"][sample.sample_id]:
                raise RuntimeError("base score shard changed during execution")
            original = json.loads(original_path.read_text())
            if (original["contract"] != {"sample_id": sample.sample_id, "plan_sha256": plan["base_plan_sha256"]}
                or original["scored"]["pools"] != context["pools"] or original["ground_truth_use"] != "none"):
                raise ValueError("base score contract differs")
            contract = {"plan_sha256": sha(plan_path), "sample_id": sample.sample_id,
                        "base_score_sha256": plan["source_shards_sha256"][sample.sample_id]}
            path = output/"scores"/(key+".json")
            shard = cached(path, contract)
            if shard is None:
                if not args.counterfactual_only and scorer is None:
                    stage = "model_loading"
                    from transformers import CLIPModel, CLIPProcessor
                    spec = base_config["models"][args.model]
                    model_path = source_paths["models"][args.model]
                    if base.model_identity(model_path, spec) != source["model"]:
                        raise ValueError("local CLIP snapshot changed since base run")
                    if args.device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(args.device)
                    model, info = CLIPModel.from_pretrained(str(model_path), local_files_only=True,
                        use_safetensors=spec["weight_file"].endswith("safetensors"), output_loading_info=True)
                    if any(info.values()):
                        raise RuntimeError(f"CLIP load incomplete: {info}")
                    model = model.eval().float().to(args.device)
                    processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
                    scorer = GlyphScorer(model, processor, config=config, base_config=base_config,
                                         font_paths=paths["fonts"], device=args.device)
                stage = "scoring"
                start = time.monotonic()
                image, _ = read_image(source_paths["data_root"], sample, context["image_sha256"])
                scored = (counterfactual_scores(original["scored"]) if args.counterfactual_only
                          else scorer.score(original["scored"], image))
                selected = innovative_selections(scored, base_config, config)
                if any(selected[name] != choice for name, choice in original["selections"].items()):
                    raise RuntimeError("an innovative arm changed a frozen v1 control")
                shard = {"contract": contract, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                    "scored": scored, "selections": selected, "ground_truth_use": "none",
                    "runtime": {"seconds": time.monotonic()-start, "device": args.device,
                        "precision": "float32", "batch_size": execution["batch_size"],
                        "model_set": [] if args.counterfactual_only else [source["model"]["model_id"]],
                        "processed_image_shape": None if args.counterfactual_only else [3, scorer.size, scorer.size],
                        "cuda_peak": _cuda_peak(args.device)}}
                write_json(path, shard)
            if (shard["scored"]["pools"] != context["pools"] or
                shard["selections"] != innovative_selections(shard["scored"], base_config, config)):
                raise RuntimeError("innovative cached selection/input contract differs")
            shards.append(shard)
            print(f"[innovative {args.model}] {index}/{len(samples)} {sample.sample_id}", flush=True)
        del scorer, model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        stage = "evaluation"
        result = base.evaluate(samples, contexts, shards, paths=source_paths, cfg=base_config)
        result.update({"plan_sha256": sha(plan_path), "model": source["model"], "device": args.device,
            "counterfactual_only": args.counterfactual_only,
            "execution_seconds": sum(s["runtime"]["seconds"] for s in shards),
            "cuda_peak": {k: max((s["runtime"]["cuda_peak"][k] for s in shards
                                   if s["runtime"]["cuda_peak"][k] is not None), default=None)
                          for k in ("peak_allocated_bytes", "peak_reserved_bytes")}})
        write_json(output/"results.json", result)
        print(json.dumps({"results": str(output/"results.json"), "summary": result["summary"]}), flush=True)
        return output
    except Exception as error:
        write_json(output/"failure.json", {"status": "FAILED", "stage": stage,
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "device": args.device, "precision": "float32", "batch_size": config["execution"]["batch_size"],
            "out_of_memory": isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)),
            "cuda_peak": _cuda_peak(args.device) if args.device == "cpu" or torch.cuda.is_available() else None})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO/"configs/approximate/clip_innovative_verification_train_v2.json")
    parser.add_argument("--paths", type=Path, default=REPO/"configs/approximate/clip_innovative_verification_local_paths_v2.json")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--counterfactual-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
