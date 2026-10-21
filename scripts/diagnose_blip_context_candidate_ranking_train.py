#!/usr/bin/env python3
"""CPU-only context ranking of frozen TRAIN word-proposal OCR candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from localize.multihypothesis_word_proposals import box_iou
from scoring.context_candidate_ranking import (
    ALL_SCORE_FIELDS, CONTEXT_SCORE_FIELDS, aggregate, distribution,
    evaluate_word, flip_diagnostics, intersection_area,
)
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scripts.check_table5_assets import check_mpnet


EXPECTED_SOURCE_REPORT_SHA256 = "532b5da717a5a9ca11ca79b924cfb652a3a57b432931f9ae64f3e98362ea2960"
EXPECTED_GENERATION_MANIFEST_SHA256 = "4ced95784912cc1799146fd934caa6d0e1914fd6230563145e5a662efbcfaed7"
EXPECTED_CANDIDATE_METADATA_SHA256 = "7adef23d47138724f5e5d168997ab3e72e8997efefdfe60baa99a04773cec60c"
EXPECTED_OCR_MANIFEST_SHA256 = "ca5f0351644b02e7b57688a2bac931363e32c3b0c7c5863f78bba3421f98f237"
EXPECTED_CANDIDATES = 58_707
EXPECTED_IMAGES = 229
EXPECTED_GT_WORDS = 849
EXPECTED_ELIGIBLE = 735
EXPECTED_EXACT_SELECTABLE = 667


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_candidates_without_gt(artifact_dir: Path) -> list[dict[str, Any]]:
    generation_path = artifact_dir / "generation-manifest.json"
    metadata_path = artifact_dir / "candidates.jsonl"
    ocr_manifest_path = artifact_dir / "ocr-batch-0064/manifest.json"
    if _sha_file(generation_path) != EXPECTED_GENERATION_MANIFEST_SHA256:
        raise RuntimeError("candidate-generation manifest SHA256 mismatch")
    if _sha_file(metadata_path) != EXPECTED_CANDIDATE_METADATA_SHA256:
        raise RuntimeError("candidate metadata SHA256 mismatch")
    if _sha_file(ocr_manifest_path) != EXPECTED_OCR_MANIFEST_SHA256:
        raise RuntimeError("PARSeq OCR manifest SHA256 mismatch")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    ocr_manifest = json.loads(ocr_manifest_path.read_text(encoding="utf-8"))
    if (
        generation.get("candidate_count") != EXPECTED_CANDIDATES
        or generation.get("sample_count") != EXPECTED_IMAGES
        or generation.get("gt_loaded") is not False
        or generation.get("test_loaded") is not False
        or ocr_manifest.get("candidate_count") != EXPECTED_CANDIDATES
    ):
        raise RuntimeError("frozen candidate/OCR artifact cardinality or split mismatch")
    metadata = [
        json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    ocr_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(ocr_manifest["shards"], 1):
        path = artifact_dir / record["path"]
        if _sha_file(path) != record["sha256"]:
            raise RuntimeError(f"PARSeq OCR shard SHA256 mismatch: {record['path']}")
        shard = json.loads(path.read_text(encoding="utf-8"))
        for row in shard["results"]:
            candidate_id = str(row["candidate_id"])
            if candidate_id in ocr_by_id:
                raise RuntimeError("duplicate candidate in PARSeq OCR shards")
            ocr_by_id[candidate_id] = row
        if index % 100 == 0 or index == len(ocr_manifest["shards"]):
            print(f"[verify frozen PARSeq artifacts] {index}/{len(ocr_manifest['shards'])}", flush=True)
    if len(metadata) != EXPECTED_CANDIDATES or len(ocr_by_id) != EXPECTED_CANDIDATES:
        raise RuntimeError("candidate/OCR population is incomplete")
    candidates = []
    for row in metadata:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in ocr_by_id:
            raise RuntimeError(f"missing PARSeq OCR candidate: {candidate_id}")
        candidates.append({**row, "ocr": ocr_by_id[candidate_id]})
    return candidates


def _load_captions(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    contract = report.get("caption_contract", {})
    if (
        report.get("sample_count") != EXPECTED_IMAGES
        or report.get("test_data_loaded") is not False
        or report.get("gt_loaded") is not False
        or report.get("one_caption_per_image") is not True
        or contract.get("caption_count_per_image") != 1
        or contract.get("device") != "cuda:0"
        or contract.get("dtype") != "float16"
        or contract.get("placement") != "single_device"
        or contract.get("quantization") != "none"
        or report.get("source", {}).get("parseq_candidate_ranking_report_sha256")
        != EXPECTED_SOURCE_REPORT_SHA256
        or report.get("source", {}).get("candidate_generation_manifest_sha256")
        != EXPECTED_GENERATION_MANIFEST_SHA256
    ):
        raise RuntimeError("BLIP-2 TRAIN caption manifest contract mismatch")
    output: dict[str, str] = {}
    for row in report["samples"]:
        sample_id = str(row["sample_id"])
        if sample_id in output or not str(row["caption"]).strip():
            raise RuntimeError("duplicate or empty BLIP-2 TRAIN caption")
        shard = path.parent / row["shard_path"]
        if _sha_file(shard) != row["shard_sha256"]:
            raise RuntimeError(f"BLIP-2 caption shard SHA mismatch: {sample_id}")
        output[sample_id] = str(row["caption"])
    if len(output) != EXPECTED_IMAGES:
        raise RuntimeError("BLIP-2 caption population is incomplete")
    return output, report


def _attach_context_scores(
    candidates: list[dict[str, Any]], captions: dict[str, str], scorer: MPNetSemanticSimilarity,
) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    if set(by_sample) != set(captions):
        raise RuntimeError("caption and candidate image populations differ")
    scored = []
    for image_index, sample_id in enumerate(sorted(by_sample, key=int), 1):
        rows = by_sample[sample_id]
        caption = captions[sample_id]
        texts = [str(row["ocr"]["raw_text"]) for row in rows]
        embeddings = scorer.embeddings(texts + [caption])
        caption_embedding = embeddings[-1]
        similarities = np.clip(embeddings[:-1] @ caption_embedding, -1.0, 1.0)
        for row, semantic in zip(rows, similarities):
            lexical = lexical_similarity(row["ocr"]["raw_text"], caption, "raw").score
            semantic_value = float(semantic)
            scored.append({
                **row,
                "context_scores": {
                    "semantic_similarity": semantic_value,
                    "lexical_similarity": float(lexical),
                    "fixed_context_score": 0.6 * semantic_value + 0.4 * float(lexical),
                },
            })
        print(
            f"[CPU context scoring/no-GT] {image_index}/229 sample_id={sample_id} "
            f"candidates={len(rows)}", flush=True,
        )
    if len(scored) != EXPECTED_CANDIDATES:
        raise RuntimeError("context scoring candidate cardinality changed")
    return scored


def _assert_baseline_reproduced(source: dict[str, Any], ranking: dict[str, Any]) -> None:
    expected = source["ranking"]["mean_token_log_probability"]
    actual = ranking["mean_token_log_probability"]
    for field in (
        "eligible_gt_words", "top1_ge_0_50", "top3_ge_0_50", "top5_ge_0_50",
        "top1_ge_0_75", "top1_parseq_exact_casefold",
        "median_rank_first_iou_ge_0_50",
    ):
        if actual[field] != expected[field]:
            raise RuntimeError(f"frozen mean-token-logprob baseline mismatch: {field}")


def main() -> int:
    args = _args()
    if args.device != "cpu":
        raise ValueError("context candidate ranking must use exactly --device cpu")
    source_path = args.source_report.resolve()
    if _sha_file(source_path) != EXPECTED_SOURCE_REPORT_SHA256:
        raise RuntimeError("frozen PARSeq candidate-ranking report SHA256 mismatch")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mpnet_asset = check_mpnet(argparse.Namespace(hf_cache=args.hf_cache.resolve()))

    # Candidate metadata, OCR and BLIP captions contain no GT.  All S/L/C values
    # are fixed before the combined source report (which contains GT) is opened.
    candidates = _load_candidates_without_gt(args.candidate_artifact_dir.resolve())
    captions, caption_report = _load_captions(args.caption_manifest.resolve())
    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    try:
        candidates = _attach_context_scores(candidates, captions, scorer)
        mpnet_metadata = scorer.metadata
    finally:
        release = scorer.release()

    # Post-hoc boundary: the frozen source report's GT-derived word rows are first
    # accessed only after all candidate-only context scores have been computed.
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("scope", {}).get("images") != EXPECTED_IMAGES
        or source.get("scope", {}).get("gt_words") != EXPECTED_GT_WORDS
        or source.get("scope", {}).get("eligible_gt_words_with_candidate_iou_ge_0_50")
        != EXPECTED_ELIGIBLE
    ):
        raise RuntimeError("frozen source report scope mismatch")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    by_id = {}
    for row in candidates:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
        by_id[row["candidate_id"]] = row

    rows = []
    for source_row in source["gt_words"]:
        if not source_row["eligible_iou_ge_0_50_candidate_exists"]:
            continue
        evaluated = evaluate_word(
            by_sample[str(source_row["sample_id"])],
            gt_bbox=source_row["gt_bbox"], gt_text=source_row["gt_text"],
        )
        if evaluated is None:
            raise RuntimeError("eligible GT word lost its IoU>=0.50 candidate")
        rows.append({
            "sample_id": str(source_row["sample_id"]),
            "gt_word_index": int(source_row["gt_word_index"]),
            **evaluated,
        })
    if len(rows) != EXPECTED_ELIGIBLE:
        raise RuntimeError("eligible GT population changed")
    ranking = aggregate(rows)
    if ranking["fixed_context_score"]["exact_selectable_gt_words"] != EXPECTED_EXACT_SELECTABLE:
        raise RuntimeError("exact-selectable GT population changed")
    _assert_baseline_reproduced(source, ranking)

    values: dict[str, dict[str, list[float]]] = {
        field: {"correct_iou_ge_0_50": [], "incorrect_iou_lt_0_50": []}
        for field in CONTEXT_SCORE_FIELDS
    }
    for word in rows:
        gt_bbox = word["gt_bbox"]
        for candidate in by_sample[word["sample_id"]]:
            if intersection_area(candidate["bbox"], gt_bbox) <= 0:
                continue
            label = (
                "correct_iou_ge_0_50"
                if box_iou(candidate["bbox"], gt_bbox) >= 0.50
                else "incorrect_iou_lt_0_50"
            )
            for field in CONTEXT_SCORE_FIELDS:
                values[field][label].append(float(candidate["context_scores"][field]))
    distributions = {
        field: {label: distribution(scores) for label, scores in classes.items()}
        for field, classes in values.items()
    }
    flips = flip_diagnostics(rows)
    fixed_flips = flips["fixed_context_score"]
    decision = (
        "GO"
        if fixed_flips["top1_ge_0_50"]["net"] > 0
        and fixed_flips["top1_parseq_exact_casefold"]["net"] > 0
        else "NO_GO"
    )
    candidate_scores = [
        {
            "candidate_id": row["candidate_id"],
            "sample_id": str(row["sample_id"]),
            **row["context_scores"],
        }
        for row in candidates
    ]
    report = {
        "schema_version": 1,
        "mode": "BLIP_CONTEXT_CANDIDATE_RANKING_TRAIN_V1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "table_5_claim": False,
        "scope": {
            "split": "TRAIN", "images": EXPECTED_IMAGES,
            "gt_words": EXPECTED_GT_WORDS, "eligible_gt_words": EXPECTED_ELIGIBLE,
            "exact_selectable_gt_words": EXPECTED_EXACT_SELECTABLE,
        },
        "scientific_boundary": {
            "proposal_generator_changed": False,
            "training": False, "test_loaded": False, "tau_applied": False,
            "gt_first_accessed_after_all_candidate_scores_frozen": True,
            "context_and_parseq_scores_combined": False,
        },
        "scores": {
            "S": "normalized all-mpnet-base-v2 cosine(transcription, full-image caption)",
            "L": "RapidFuzz raw ratio(transcription, full-image caption)/100",
            "C": "0.6*S + 0.4*L",
            "baseline": "PARSeq mean token log-probability",
            "ranking": "descending one score only; candidate_id ascending tie-break",
            "candidate_association": "positive bbox intersection with GT, post-hoc only",
        },
        "ranking": ranking,
        "score_distributions": distributions,
        "improved_worsened_vs_mean_token_log_probability": flips,
        "decision": {
            "value": decision,
            "predeclared_rule": (
                "GO iff fixed C has positive net flips versus PARSeq mean-token-logprob "
                "for both top1 IoU>=0.50 and top1 PARSeq exact; otherwise NO_GO"
            ),
        },
        "caption_cache": {
            "path": str(args.caption_manifest.resolve()),
            "sha256": _sha_file(args.caption_manifest.resolve()),
            "contract": caption_report["caption_contract"],
        },
        "mpnet": {"asset": mpnet_asset, "runtime": mpnet_metadata, "release": release},
        "provenance": {
            "source_report": str(source_path),
            "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "candidate_generation_manifest_sha256": EXPECTED_GENERATION_MANIFEST_SHA256,
            "candidate_metadata_sha256": EXPECTED_CANDIDATE_METADATA_SHA256,
            "ocr_manifest_sha256": EXPECTED_OCR_MANIFEST_SHA256,
        },
        "candidate_scores": candidate_scores,
        "eligible_gt_rows": rows,
    }
    _write_exclusive(output, report)
    summary = {
        "ranking": ranking,
        "improved_worsened": flips,
        "decision": decision,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"JSON {output}", flush=True)
    print(f"SHA256 {_sha_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
