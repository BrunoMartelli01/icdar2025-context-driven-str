#!/usr/bin/env python3
"""Qualify the pinned APPROXIMATE MPNet and all named scoring interpretations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import get_required, load_config, resolved_yaml
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scoring.strategies import STRATEGIES, decide
from scripts.check_table5_assets import check_mpnet


def _embedding_record(vector: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(vector, dtype="<f8")
    return {
        "dtype": "little-endian float64",
        "l2_norm": float(np.linalg.norm(canonical)),
        "sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "shape": list(canonical.shape),
    }


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if get_required(cfg, "run.table_5_claim") is not False:
        raise ValueError("Scoring qualification cannot use a Table 5 claim configuration")
    if tuple(float(get_required(cfg, "scoring." + name)) for name in ("alpha", "beta", "tau")) != (
        0.6,
        0.4,
        0.8,
    ):
        raise ValueError("Scoring qualification requires the paper-specified 0.6/0.4/0.8 values")

    fixture_root = "qualification.mpnet_scoring."
    t1 = str(get_required(cfg, fixture_root + "T1"))
    t2 = str(get_required(cfg, fixture_root + "T2"))
    t3 = str(get_required(cfg, fixture_root + "T3"))
    repeated_atol = float(get_required(cfg, fixture_root + "repeated_batch_atol"))
    batch_single_atol = float(get_required(cfg, fixture_root + "batch_single_atol"))
    asset = check_mpnet(argparse.Namespace(hf_cache=args.hf_cache))

    model_started = time.perf_counter()
    scorer = MPNetSemanticSimilarity(
        MPNetApproxConfig(
            model_id=str(get_required(cfg, "scoring.embedder_model_id")),
            revision=str(get_required(cfg, "scoring.embedder_revision")),
            normalize_embeddings=bool(get_required(cfg, "scoring.normalize_embeddings")),
            precision=str(get_required(cfg, "scoring.embedder_precision")),
            device=str(get_required(cfg, "scoring.embedder_device")),
        )
    )
    model_load_seconds = time.perf_counter() - model_started
    texts = [t1, t3, t2]
    inference_started = time.perf_counter()
    batch = scorer.embeddings(texts)
    repeated = scorer.embeddings(texts)
    singles = np.concatenate([scorer.embeddings([text]) for text in texts], axis=0)
    inference_seconds = time.perf_counter() - inference_started
    repeated_difference = float(np.max(np.abs(batch - repeated)))
    batch_single_difference = float(np.max(np.abs(batch - singles)))
    if repeated_difference > repeated_atol:
        raise RuntimeError("Repeated MPNet batch output exceeded the frozen tolerance")
    if batch_single_difference > batch_single_atol:
        raise RuntimeError("MPNet batch-versus-single output exceeded the frozen tolerance")

    s1, s3 = scorer.compare_candidates(t1, t3, t2)
    lexical_1 = lexical_similarity(t1, t2, str(get_required(cfg, "scoring.lexical_mode")))
    lexical_3 = lexical_similarity(t3, t2, str(get_required(cfg, "scoring.lexical_mode")))
    decisions = {
        strategy: decide(
            strategy,
            T1=t1,
            T2=t2,
            T3=t3,
            S1=s1,
            S3=s3,
            L1=lexical_1.score,
            L3=lexical_3.score,
        ).to_dict()
        for strategy in STRATEGIES
    }
    release = scorer.release()
    resolved = resolved_yaml(cfg)
    return {
        "schema_version": 1,
        "success": True,
        "status": "approximate reproduction",
        "result_class": "MPNet/scoring qualification; not a Table 5 result",
        "artifact": asset,
        "configuration": {
            "path": str(args.config.resolve()),
            "resolved_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
            "resolved_yaml": resolved,
        },
        "inputs": {"T1": t1, "T2": t2, "T3": t3},
        "embeddings": {
            label: _embedding_record(batch[index])
            for index, label in enumerate(("T1", "T3", "T2"))
        },
        "determinism": {
            "repeated_batch_max_abs_difference": repeated_difference,
            "repeated_batch_atol": repeated_atol,
            "batch_single_max_abs_difference": batch_single_difference,
            "batch_single_atol": batch_single_atol,
        },
        "scores": {
            "S1": s1,
            "S3": s3,
            "L1": asdict(lexical_1),
            "L3": asdict(lexical_3),
        },
        "decisions": decisions,
        "model_metadata": scorer.metadata,
        "release": release,
        "timing": {
            "model_load_seconds": model_load_seconds,
            "embedding_qualification_seconds": inference_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "torch": importlib.metadata.version("torch"),
            "rapidfuzz": importlib.metadata.version("rapidfuzz"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = qualify(args)
    except Exception as error:
        report = {
            "schema_version": 1,
            "success": False,
            "status": "failed",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
