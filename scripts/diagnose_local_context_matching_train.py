#!/usr/bin/env python3
"""TRAIN-only diagnostic for practical local caption matching.

Candidate generation, PARSeq outputs, BLIP captions and MPNet are frozen before
GT is read.  This script does not train, does not load TEST and does not invoke
CUDA.  Image-wide results use the complete diagnostic proposal pool and are not
an end-to-end Table 5 metric.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Sequence

import numpy as np

from caption.blip2_captioner import (
    BLIP2_MODEL_ID, BLIP2_REVISION, DEFAULT_GENERATION, DEFAULT_PROMPT,
    PRACTICAL_ANTI_REPETITION_GENERATION, stable_hash,
)
from localize.multihypothesis_word_proposals import box_iou
from scoring.local_context import (
    LOCAL_VARIANTS, TAU, repetition_diagnostics, score_local_candidates,
)
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scripts.diagnose_blip_context_candidate_ranking_train import (
    EXPECTED_CANDIDATES, EXPECTED_ELIGIBLE, EXPECTED_EXACT_SELECTABLE,
    EXPECTED_GT_WORDS, EXPECTED_IMAGES, EXPECTED_SOURCE_REPORT_SHA256,
    _load_candidates_without_gt,
)
from scripts.check_table5_assets import check_mpnet


FIELDS = ("whole_caption", *LOCAL_VARIANTS, "mean_token_log_probability")
EXPECTED_GT_REFERENCE_SHA256 = "68c6a8279949a4de141cca0d3ed363b52a01c6a53cac62943cbe687fdc4fe462"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--gt-reference-report", type=Path, required=True)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _load_captions(path: Path) -> tuple[dict[str, str], dict[str, Any], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    contract = report.get("caption_contract", {})
    generation = contract.get("generation")
    if generation == DEFAULT_GENERATION and "generation_profile" not in contract:
        profile = "repository-v1"
    elif (
        generation == PRACTICAL_ANTI_REPETITION_GENERATION
        and contract.get("generation_profile") == "practical-antirepetition-v1"
    ):
        profile = "practical-antirepetition-v1"
    else:
        raise RuntimeError("caption generation profile is not recognized")
    if (
        report.get("sample_count") != EXPECTED_IMAGES
        or report.get("test_data_loaded") is not False
        or report.get("gt_loaded") is not False
        or report.get("one_caption_per_image") is not True
        or contract.get("caption_count_per_image") != 1
        or contract.get("model_id") != BLIP2_MODEL_ID
        or contract.get("revision") != BLIP2_REVISION
        or contract.get("prompt") != DEFAULT_PROMPT
        or contract.get("prompt_sha256") != stable_hash(DEFAULT_PROMPT)
        or contract.get("generation_sha256") != stable_hash(generation)
        or contract.get("device") != "cuda:0"
        or contract.get("dtype") != "float16"
        or contract.get("placement") != "single_device"
        or contract.get("quantization") != "none"
    ):
        raise RuntimeError("TRAIN caption manifest contract mismatch")
    captions: dict[str, str] = {}
    for row in report["samples"]:
        sample_id = str(row["sample_id"])
        shard_path = path.parent / row["shard_path"]
        if _sha(shard_path) != row["shard_sha256"]:
            raise RuntimeError(f"caption shard SHA mismatch: {sample_id}")
        if sample_id in captions or not str(row["caption"]).strip():
            raise RuntimeError("duplicate or empty caption")
        captions[sample_id] = str(row["caption"])
    if len(captions) != EXPECTED_IMAGES:
        raise RuntimeError("caption population is incomplete")
    return captions, report, profile


def _intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _whole_scores(texts: Sequence[str], caption: str, scorer: MPNetSemanticSimilarity) -> list[dict[str, Any]]:
    embeddings = scorer.embeddings([*texts, caption])
    caption_embedding = embeddings[-1]
    output = []
    for text, embedding in zip(texts, embeddings[:-1]):
        semantic = float(np.clip(np.dot(embedding, caption_embedding), -1.0, 1.0))
        lexical = float(lexical_similarity(text, caption, "raw").score)
        output.append({
            "semantic_similarity": semantic, "lexical_similarity": lexical,
            "confidence": 0.6 * semantic + 0.4 * lexical, "window": caption,
        })
    return output


def _score_candidates(
    candidates: list[dict[str, Any]], captions: dict[str, str], scorer: MPNetSemanticSimilarity,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_sample[str(row["sample_id"])].append(row)
    if set(by_sample) != set(captions):
        raise RuntimeError("candidate and caption populations differ")
    diagnostics = []
    for number, sample_id in enumerate(sorted(by_sample, key=int), 1):
        rows = by_sample[sample_id]
        texts = [str(row["ocr"]["raw_text"]) for row in rows]
        whole = _whole_scores(texts, captions[sample_id], scorer)
        local = {
            variant: score_local_candidates(texts, captions[sample_id], scorer, variant=variant)
            for variant in LOCAL_VARIANTS
        }
        for index, row in enumerate(rows):
            row["practical_context"] = {
                "whole_caption": whole[index],
                **{variant: local[variant][index] for variant in LOCAL_VARIANTS},
            }
        diagnostics.append(repetition_diagnostics(captions[sample_id]))
        print(f"[local context/no-GT] {number}/{EXPECTED_IMAGES} sample_id={sample_id} candidates={len(rows)}", flush=True)
    if sum(map(len, by_sample.values())) != EXPECTED_CANDIDATES:
        raise RuntimeError("candidate population changed during scoring")
    return dict(by_sample), _caption_quality(diagnostics)


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()), "median": float(np.median(array)),
        "mean": float(array.mean()), "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def _caption_quality(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fractions = [float(row["duplicate_ngram_fraction"]) for row in rows]
    lengths = [float(row["token_count"]) for row in rows]
    return {
        "captions": len(rows), "repeated_4gram_captions": sum(row["has_duplicate_ngram"] for row in rows),
        "duplicate_fraction_ge_0_10": sum(value >= 0.10 for value in fractions),
        "duplicate_fraction_ge_0_25": sum(value >= 0.25 for value in fractions),
        "token_count": _quantiles(lengths), "duplicate_4gram_fraction": _quantiles(fractions),
    }


def _value(row: dict[str, Any], field: str) -> float:
    if field == "mean_token_log_probability":
        value = row["ocr"].get(field)
        return float(value) if value is not None else float("-inf")
    return float(row["practical_context"][field]["confidence"])


def _rank(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-_value(row, field), str(row["candidate_id"])))


def _evaluate(
    by_sample: dict[str, list[dict[str, Any]]], source: dict[str, Any],
    gt_reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in gt_reference["per_word"]:
        gt_by_sample[str(word["sample_id"])].append({
            "gt_bbox": list(word["gt_bbox"]), "gt_text": str(word["gt_text"]),
        })
    if sum(map(len, gt_by_sample.values())) != EXPECTED_GT_WORDS:
        raise RuntimeError("complete TRAIN GT reference cardinality mismatch")
    eligible = []
    for word in source["gt_words"]:
        if word["eligible_iou_ge_0_50_candidate_exists"]:
            eligible.append(word)
    rankings: dict[str, dict[str, Any]] = {}
    for field in FIELDS:
        values = []
        for word in eligible:
            associated = [
                row for row in by_sample[str(word["sample_id"])]
                if _intersection(row["bbox"], word["gt_bbox"]) > 0
            ]
            ranked = _rank(associated, field)
            ious = [box_iou(row["bbox"], word["gt_bbox"]) for row in ranked]
            values.append({
                "top1_ge_0_50": ious[0] >= 0.5,
                "top3_ge_0_50": any(value >= 0.5 for value in ious[:3]),
                "top5_ge_0_50": any(value >= 0.5 for value in ious[:5]),
                "top1_ge_0_75": ious[0] >= 0.75,
                "top1_exact": str(ranked[0]["ocr"]["raw_text"]).casefold() == str(word["gt_text"]).casefold(),
                "first_good_rank": next(index for index, value in enumerate(ious, 1) if value >= 0.5),
                "accepted": field != "mean_token_log_probability" and _value(ranked[0], field) >= TAU,
            })
        accepted = [row for row in values if row["accepted"]]
        rankings[field] = {
            "eligible_gt_words": len(values),
            "top1_iou_ge_0_50": sum(row["top1_ge_0_50"] for row in values),
            "top3_iou_ge_0_50": sum(row["top3_ge_0_50"] for row in values),
            "top5_iou_ge_0_50": sum(row["top5_ge_0_50"] for row in values),
            "top1_iou_ge_0_75": sum(row["top1_ge_0_75"] for row in values),
            "top1_parseq_exact": sum(row["top1_exact"] for row in values),
            "median_rank_first_iou_ge_0_50": float(median(row["first_good_rank"] for row in values)),
            "tau_0_8_accepted": len(accepted),
            "tau_0_8_accepted_iou_ge_0_50": sum(row["top1_ge_0_50"] for row in accepted),
            "tau_0_8_accepted_parseq_exact": sum(row["top1_exact"] for row in accepted),
        }
    image_wide = {}
    for field in FIELDS[:-1]:
        rows = []
        for sample_id, candidates in by_sample.items():
            top = _rank(candidates, field)[0]
            words = gt_by_sample[sample_id]
            spatial = any(
                str(top["ocr"]["raw_text"]).casefold() == str(word["gt_text"]).casefold()
                and box_iou(top["bbox"], word["gt_bbox"]) >= 0.5 for word in words
            )
            text_member = any(
                str(top["ocr"]["raw_text"]).casefold() == str(word["gt_text"]).casefold()
                for word in words
            )
            accepted = _value(top, field) >= TAU
            rows.append((accepted, spatial, text_member, _value(top, field)))
        accepted_rows = [row for row in rows if row[0]]
        image_wide[field] = {
            "images": len(rows), "no_fallback_images": len(accepted_rows),
            "fallback_images": len(rows) - len(accepted_rows),
            "accepted_spatial_exact": sum(row[1] for row in accepted_rows),
            "accepted_text_gt_member": sum(row[2] for row in accepted_rows),
            "accepted_not_spatial_exact": sum(not row[1] for row in accepted_rows),
            "accepted_not_text_gt_member": sum(not row[2] for row in accepted_rows),
            "confidence": _quantiles([row[3] for row in rows]),
        }
    return rankings, image_wide


def main() -> int:
    args = _args()
    if args.device != "cpu":
        raise ValueError("local context matching is CPU-only and requires --device cpu")
    source_path = args.source_report.resolve()
    if _sha(source_path) != EXPECTED_SOURCE_REPORT_SHA256:
        raise RuntimeError("frozen TRAIN source-report SHA256 mismatch")
    gt_reference_path = args.gt_reference_report.resolve()
    if _sha(gt_reference_path) != EXPECTED_GT_REFERENCE_SHA256:
        raise RuntimeError("frozen complete TRAIN GT-reference SHA256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    asset = check_mpnet(argparse.Namespace(hf_cache=args.hf_cache.resolve()))
    candidates = _load_candidates_without_gt(args.candidate_artifact_dir.resolve())
    captions, caption_report, profile = _load_captions(args.caption_manifest.resolve())
    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    try:
        by_sample, caption_quality = _score_candidates(candidates, captions, scorer)
        runtime = scorer.metadata
    finally:
        release = scorer.release()
    # GT is first read only after all candidate scores are frozen.
    source = json.loads(source_path.read_text(encoding="utf-8"))
    gt_reference = json.loads(gt_reference_path.read_text(encoding="utf-8"))
    if (
        source.get("scope", {}).get("images") != EXPECTED_IMAGES
        or source.get("scope", {}).get("gt_words") != EXPECTED_GT_WORDS
        or source.get("scope", {}).get("eligible_gt_words_with_candidate_iou_ge_0_50") != EXPECTED_ELIGIBLE
    ):
        raise RuntimeError("TRAIN source scope mismatch")
    rankings, image_wide = _evaluate(by_sample, source, gt_reference)
    if rankings["mean_token_log_probability"]["top1_iou_ge_0_50"] != 323:
        raise RuntimeError("mean-token-logprob control mismatch")
    report = {
        "schema_version": 1,
        "mode": "PRACTICAL_LOCAL_CONTEXT_MATCHING_TRAIN_V1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "table_5_claim": False,
        "scope": {"split": "TRAIN", "images": EXPECTED_IMAGES, "gt_words": EXPECTED_GT_WORDS},
        "scientific_boundary": {
            "test_loaded": False, "training": False, "cuda_used_by_this_script": False,
            "proposal_generator_changed": False, "parseq_rerun": False,
            "gt_first_loaded_after_candidate_scoring": True,
            "image_wide_pool": "unpruned multihypothesis diagnostic pool; not E2E",
        },
        "caption": {
            "profile": profile, "quality": caption_quality,
            "manifest": str(args.caption_manifest.resolve()),
            "manifest_sha256": _sha(args.caption_manifest.resolve()),
            "contract": caption_report["caption_contract"],
        },
        "matching": {
            "whole_caption": "repository control: one full caption",
            "exact_length": "paired max 0.6*S+0.4*L over windows with candidate token length",
            "neighbor_lengths": "paired max over candidate token length +/-1",
            "alpha": 0.6, "beta": 0.4, "tau": TAU,
            "lexical": "RapidFuzz raw ratio/window; no partial_ratio",
            "semantic": "normalized MPNet cosine/window",
        },
        "ranking": rankings,
        "image_wide_gate_diagnostic": image_wide,
        "mpnet": {"asset": asset, "runtime": runtime, "release": release},
        "provenance": {
            "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "gt_reference_report_sha256": EXPECTED_GT_REFERENCE_SHA256,
        },
    }
    output = args.output.resolve()
    _write_exclusive(output, report)
    print(json.dumps({
        "caption": report["caption"], "ranking": rankings,
        "image_wide_gate_diagnostic": image_wide,
    }, indent=2, sort_keys=True), flush=True)
    print(f"JSON {output}\nSHA256 {_sha(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
