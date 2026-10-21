#!/usr/bin/env python3
"""TRAIN-only local-context gate on frozen v1 and BALANCED proposal sets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from PIL import Image

from localize.multihypothesis_word_proposals import box_iou
from localize.practical_candidate_selection_v2 import select_anchor_complete
from scoring.local_context import LOCAL_VARIANTS, TAU, score_local_candidates
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scripts.check_table5_assets import check_mpnet
from scripts.diagnose_blip_context_candidate_ranking_train import (
    EXPECTED_CANDIDATES, _load_candidates_without_gt,
)
from scripts.diagnose_local_context_matching_train import (
    EXPECTED_GT_REFERENCE_SHA256, EXPECTED_IMAGES, _load_captions,
)
from scripts.diagnose_multihypothesis_word_proposal_oracle_train import _load_probability
from scripts.evaluate_practical_candidate_selection_v2_train import (
    EXPECTED_CONTEXT_REPORT_SHA256, EXPECTED_PROBABILITY_MANIFEST_SHA256,
    _attach_base_features, _candidate_provenance_and_features,
)
from segmentation.data import load_segmentation_manifest


EXPECTED_V1_GENERATION_SHA256 = "262f9f74f55f8e0f1d59bc12b1204fdf23b9c574636b39708b5a0db3977a0b17"
EXPECTED_V1_METADATA_SHA256 = "bebfe5d4bdfac3081782a119fbd658704e0b18e8f3cec8ce2a138e14e1fa44e4"
EXPECTED_V1_OCR_SHA256 = "c5277f1de49e0eef0d49b011a5cf978dfaf45562f80f040e1e32a4a5e08d6c99"
EXPECTED_V1_COUNT = 1203
EXPECTED_BALANCED_COUNT = 1095
EXPECTED_BALANCED_V1 = 527
EXPECTED_BALANCED_ALTERNATIVE = 568
FIELDS = ("whole_caption", *LOCAL_VARIANTS)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--probability-manifest", type=Path, required=True)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--v1-artifact-dir", type=Path, required=True)
    parser.add_argument("--old-context-report", type=Path, required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--gt-reference-report", type=Path, required=True)
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


def _write(path: Path, value: Any) -> None:
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


def _contained(root: Path, relative: str | Path) -> Path:
    path = (root / Path(relative)).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes root")
    return path


def _load_v1(root: Path) -> list[dict[str, Any]]:
    expected = {
        root / "generation-manifest.json": EXPECTED_V1_GENERATION_SHA256,
        root / "v1-candidates.jsonl": EXPECTED_V1_METADATA_SHA256,
        root / "ocr-batch-0064" / "manifest.json": EXPECTED_V1_OCR_SHA256,
    }
    for path, digest in expected.items():
        if _sha(path) != digest:
            raise RuntimeError(f"frozen v1 artifact SHA mismatch: {path}")
    rows = [json.loads(line) for line in (root / "v1-candidates.jsonl").read_text().splitlines()]
    manifest = json.loads((root / "ocr-batch-0064" / "manifest.json").read_text())
    ocr = {}
    for record in manifest["shards"]:
        path = root / "ocr-batch-0064" / Path(record["path"]).name
        if _sha(path) != record["sha256"]:
            raise RuntimeError("v1 OCR shard SHA mismatch")
        for value in json.loads(path.read_text())["results"]:
            ocr[str(value["candidate_id"])] = value
    if len(rows) != EXPECTED_V1_COUNT or len(ocr) != EXPECTED_V1_COUNT:
        raise RuntimeError("frozen v1 candidate population mismatch")
    return [{**row, "ocr": ocr[str(row["candidate_id"])]} for row in rows]


def _whole_scores(texts: Sequence[str], caption: str, scorer: MPNetSemanticSimilarity) -> list[dict[str, Any]]:
    embeddings = scorer.embeddings([*texts, caption])
    output = []
    for text, embedding in zip(texts, embeddings[:-1]):
        semantic = float(np.clip(np.dot(embedding, embeddings[-1]), -1.0, 1.0))
        lexical = float(lexical_similarity(text, caption, "raw").score)
        output.append({"semantic_similarity": semantic, "lexical_similarity": lexical,
                       "confidence": 0.6 * semantic + 0.4 * lexical, "window": caption})
    return output


def _attach_scores(
    selections: dict[str, dict[str, tuple[dict[str, Any], ...]]],
    captions: dict[str, str], scorer: MPNetSemanticSimilarity,
) -> None:
    for number, sample_id in enumerate(sorted(captions, key=int), 1):
        unique: dict[str, dict[str, Any]] = {}
        for branch in selections:
            for row in selections[branch][sample_id]:
                unique[str(row["candidate_id"])] = row
        rows = list(unique.values())
        texts = [str(row["ocr"]["raw_text"]) for row in rows]
        whole = _whole_scores(texts, captions[sample_id], scorer)
        local = {variant: score_local_candidates(texts, captions[sample_id], scorer, variant=variant)
                 for variant in LOCAL_VARIANTS}
        for index, row in enumerate(rows):
            row["practical_context"] = {"whole_caption": whole[index], **{
                variant: local[variant][index] for variant in LOCAL_VARIANTS
            }}
        score_by_id = {
            str(row["candidate_id"]): row["practical_context"] for row in rows
        }
        for branch in selections:
            for row in selections[branch][sample_id]:
                row["practical_context"] = score_by_id[str(row["candidate_id"])]
        print(f"[balanced local context/no-GT] {number}/{EXPECTED_IMAGES} sample_id={sample_id} unique={len(rows)}", flush=True)


def _evaluate(
    selections: dict[str, dict[str, tuple[dict[str, Any], ...]]],
    gt_reference: dict[str, Any],
) -> dict[str, Any]:
    gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gt_reference["per_word"]:
        gt[str(row["sample_id"])].append({"bbox": row["gt_bbox"], "text": row["gt_text"]})
    if sum(map(len, gt.values())) != 849:
        raise RuntimeError("TRAIN GT population mismatch")
    output = {}
    for branch, by_sample in selections.items():
        output[branch] = {}
        for field in FIELDS:
            rows = []
            for sample_id, candidates in by_sample.items():
                selected = max(
                    enumerate(candidates),
                    key=lambda item: (item[1]["practical_context"][field]["confidence"], -item[0]),
                )[1]
                confidence = float(selected["practical_context"][field]["confidence"])
                text = str(selected["ocr"]["raw_text"])
                spatial = any(
                    text.casefold() == str(word["text"]).casefold()
                    and box_iou(selected["bbox"], word["bbox"]) >= 0.5 for word in gt[sample_id]
                )
                member = any(text.casefold() == str(word["text"]).casefold() for word in gt[sample_id])
                rows.append({"accepted": confidence >= TAU, "spatial": spatial, "member": member,
                             "confidence": confidence})
            accepted = [row for row in rows if row["accepted"]]
            correct = sum(row["spatial"] for row in accepted)
            output[branch][field] = {
                "images": len(rows), "no_fallback": len(accepted),
                "fallback": len(rows) - len(accepted),
                "accepted_spatial_exact": correct,
                "accepted_text_gt_member": sum(row["member"] for row in accepted),
                "false_accept_spatial": len(accepted) - correct,
                "accepted_spatial_precision": correct / len(accepted) if accepted else None,
                "top1_spatial_exact_before_gate": sum(row["spatial"] for row in rows),
                "top1_text_gt_member_before_gate": sum(row["member"] for row in rows),
                "confidence_median": float(np.median([row["confidence"] for row in rows])),
                "confidence_max": max(row["confidence"] for row in rows),
            }
    return output


def main() -> int:
    args = _args()
    if args.device != "cpu":
        raise ValueError("balanced local-context diagnostic requires --device cpu")
    output = args.output.resolve()
    probability_path = args.probability_manifest.resolve()
    old_context_path = args.old_context_report.resolve()
    gt_path = args.gt_reference_report.resolve()
    if _sha(probability_path) != EXPECTED_PROBABILITY_MANIFEST_SHA256:
        raise RuntimeError("probability manifest SHA mismatch")
    if _sha(old_context_path) != EXPECTED_CONTEXT_REPORT_SHA256:
        raise RuntimeError("old context report SHA mismatch")
    if _sha(gt_path) != EXPECTED_GT_REFERENCE_SHA256:
        raise RuntimeError("GT reference report SHA mismatch")
    asset = check_mpnet(argparse.Namespace(hf_cache=args.hf_cache.resolve()))
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if manifest.split_role != "train" or len(manifest.samples) != EXPECTED_IMAGES:
        raise RuntimeError("requires TRAIN-229")
    probabilities = json.loads(probability_path.read_text())
    probability_by_id = {str(row["sample_id"]): row for row in probabilities["samples"]}
    anchors = _load_v1(args.v1_artifact_dir.resolve())
    alternatives = _load_candidates_without_gt(args.candidate_artifact_dir.resolve())
    if len(alternatives) != EXPECTED_CANDIDATES:
        raise RuntimeError("alternative population mismatch")
    old_context = json.loads(old_context_path.read_text())
    context_by_id = {str(row["candidate_id"]): float(row["fixed_context_score"])
                     for row in old_context["candidate_scores"]}
    anchors_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alternatives_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        anchors_by_sample[str(row["sample_id"])].append(row)
    for row in alternatives:
        alternatives_by_sample[str(row["sample_id"])].append(row)
    selections = {"practical_remap_v1": {}, "balanced_anchor_mean": {}}
    data_root = args.data_root.resolve()
    for number, sample in enumerate(manifest.samples, 1):
        sample_id = str(sample.sample_id)
        with Image.open(_contained(data_root, sample.image_path)) as image:
            width, height = image.size
        probability = _load_probability(probability_path.parent, probability_by_id[sample_id])
        sample_alternatives = _candidate_provenance_and_features(
            candidates=alternatives_by_sample[sample_id], probability=probability,
            width=width, height=height, context_by_id=context_by_id,
        )
        sample_anchors = anchors_by_sample[sample_id]
        _attach_base_features(sample_anchors, sample_alternatives)
        selections["practical_remap_v1"][sample_id] = tuple(sample_anchors)
        selections["balanced_anchor_mean"][sample_id] = select_anchor_complete(
            [dict(row) for row in sample_anchors], [dict(row) for row in sample_alternatives],
            variant="BALANCED_ANCHOR_MEAN",
        )
        print(f"[reconstruct frozen proposal sets/no-GT] {number}/{EXPECTED_IMAGES} sample_id={sample_id}", flush=True)
    balanced = [row for rows in selections["balanced_anchor_mean"].values() for row in rows]
    if (
        sum(len(rows) for rows in selections["practical_remap_v1"].values()) != EXPECTED_V1_COUNT
        or len(balanced) != EXPECTED_BALANCED_COUNT
        or sum(row["source"] == "v1_exact" for row in balanced) != EXPECTED_BALANCED_V1
        or sum(row["source"] != "v1_exact" for row in balanced) != EXPECTED_BALANCED_ALTERNATIVE
    ):
        raise RuntimeError("reconstructed frozen proposal-set control mismatch")
    captions, caption_report, profile = _load_captions(args.caption_manifest.resolve())
    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    try:
        _attach_scores(selections, captions, scorer)
        runtime = scorer.metadata
    finally:
        release = scorer.release()
    # GT is loaded only after proposal reconstruction and context scoring finish.
    result = _evaluate(selections, json.loads(gt_path.read_text()))
    report = {
        "schema_version": 1, "mode": "BALANCED_LOCAL_CONTEXT_TRAIN_V1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC", "table_5_claim": False,
        "scope": {"split": "TRAIN", "images": EXPECTED_IMAGES, "gt_words": 849},
        "scientific_boundary": {"test_loaded": False, "training": False, "cuda_used_by_script": False,
                                "gt_first_loaded_after_selection_and_scoring": True,
                                "T1_full_image_excluded": True, "tau": TAU},
        "caption": {"profile": profile, "manifest_sha256": _sha(args.caption_manifest.resolve()),
                    "contract": caption_report["caption_contract"]},
        "proposal_sets": {"practical_remap_v1": EXPECTED_V1_COUNT,
                          "balanced_anchor_mean": EXPECTED_BALANCED_COUNT},
        "results": result,
        "mpnet": {"asset": asset, "runtime": runtime, "release": release},
        "provenance": {"probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
                       "old_context_report_sha256": EXPECTED_CONTEXT_REPORT_SHA256,
                       "gt_reference_report_sha256": EXPECTED_GT_REFERENCE_SHA256},
    }
    _write(output, report)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"JSON {output}\nSHA256 {_sha(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
