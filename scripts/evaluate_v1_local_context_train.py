#!/usr/bin/env python3
"""CPU-only TRAIN evaluation of whole-caption versus local-caption v1 routing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from rapidfuzz.fuzz import ratio

from localize.multihypothesis_word_proposals import box_iou
from localize.practical_candidate_selection import intersection_area
from localize.practical_remap_protocol import train_task22_word_boxes
from scoring.local_context import LOCAL_VARIANTS, TAU, repetition_diagnostics, score_local_candidates
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scripts.diagnose_local_context_matching_train import _load_captions
from scripts.diagnose_parseq_candidate_ranking_train import _task22_word_texts
from segmentation.data import load_segmentation_manifest


FIELDS = ("whole_caption", *LOCAL_VARIANTS)
EXPECTED_V1_CANDIDATES = 1203


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative artifact path required")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes root")
    return path


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


def _whole(texts: Sequence[str], caption: str, scorer: MPNetSemanticSimilarity) -> list[dict[str, float | str]]:
    embeddings = scorer.embeddings([*texts, caption])
    output = []
    for text, embedding in zip(texts, embeddings[:-1]):
        semantic = float(np.clip(np.dot(embedding, embeddings[-1]), -1.0, 1.0))
        lexical = float(lexical_similarity(text, caption, "raw").score)
        output.append({"window": caption, "semantic_similarity": semantic,
                       "lexical_similarity": lexical,
                       "confidence": 0.6 * semantic + 0.4 * lexical})
    return output


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--caption-manifest", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.device != "cpu":
        raise RuntimeError("local-context TRAIN evaluation requires CPU")
    input_path, input_root = args.input_manifest.resolve(), args.input_manifest.resolve().parent
    caption_path, output = args.caption_manifest.resolve(), args.output.resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("mode") != "V1_LOCAL_CONTEXT_TRAIN_PERSISTENT_V1"
            or existing.get("scope") != {"split": "TRAIN", "images": 229, "gt_words": 849}
            or existing.get("inputs", {}).get("compact_manifest_sha256") != _sha(input_path)
            or existing.get("caption", {}).get("manifest_sha256") != _sha(caption_path)
        ):
            raise RuntimeError("existing local-context report does not match current inputs")
        print(f"[v1 local context] verified reusable report: {output}", flush=True)
        print(f"SHA256 {_sha(output)}", flush=True)
        return 0
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("scope") != {"split": "TRAIN", "images": 229, "test_loaded": False, "gt_loaded": False}:
        raise RuntimeError("compact input scope mismatch")
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("requires TRAIN-229")
    captions, caption_source, profile = _load_captions(caption_path)
    source_by_id = {str(row["sample_id"]): row for row in source["samples"]}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for sample in manifest.samples:
        ref = source_by_id[sample.sample_id]
        cp, op = _contained(input_root, ref["candidate_shard"]), _contained(input_root, ref["ocr_shard"])
        if _sha(cp) != ref["candidate_shard_sha256"] or _sha(op) != ref["ocr_shard_sha256"]:
            raise RuntimeError(f"compact shard SHA mismatch: {sample.sample_id}")
        cvalue, ovalue = json.loads(cp.read_text()), json.loads(op.read_text())
        ocr = {str(row["candidate_id"]): row for row in ovalue["results"]}
        rows = [{**row, "ocr": ocr[str(row["candidate_id"])]} for row in cvalue["candidates"]]
        if len(rows) != ref["candidate_count"] or len(ocr) != len(rows):
            raise RuntimeError(f"compact shard cardinality mismatch: {sample.sample_id}")
        candidates[sample.sample_id] = rows
    if (
        sum(map(len, candidates.values())) != source["candidate_count"]
        or source["candidate_count"] != EXPECTED_V1_CANDIDATES
    ):
        raise RuntimeError("compact candidate population mismatch")

    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    try:
        for number, sample in enumerate(manifest.samples, 1):
            rows = candidates[sample.sample_id]
            texts = [str(row["ocr"]["raw_text"]) for row in rows]
            whole = _whole(texts, captions[sample.sample_id], scorer) if texts else []
            local = {name: score_local_candidates(texts, captions[sample.sample_id], scorer, variant=name)
                     for name in LOCAL_VARIANTS}
            for index, row in enumerate(rows):
                row["context"] = {"whole_caption": whole[index], **{
                    name: local[name][index] for name in LOCAL_VARIANTS
                }}
            print(f"[v1 local context/no-GT] {number}/229 sample_id={sample.sample_id} crops={len(rows)}", flush=True)
        runtime = scorer.metadata
    finally:
        release = scorer.release()

    # GT is first opened after every candidate has been scored.
    data_root = args.data_root.resolve()
    gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in manifest.samples:
        boxes = train_task22_word_boxes(_contained(data_root, sample.mask_path),
                                        _contained(data_root, sample.validity_annotation_path))
        texts = _task22_word_texts(_contained(data_root, sample.validity_annotation_path))
        if len(boxes) != len(texts):
            raise RuntimeError(f"TRAIN GT mismatch: {sample.sample_id}")
        gt[sample.sample_id] = [{"bbox": list(box.bbox), "text": text} for box, text in zip(boxes, texts)]
    if sum(map(len, gt.values())) != 849:
        raise RuntimeError("expected 849 TRAIN GT words")

    best_ious = []
    wordness = Counter()
    available_predictions = 0
    for sample in manifest.samples:
        rows, words = candidates[sample.sample_id], gt[sample.sample_id]
        for word in words:
            best = max((box_iou(row["bbox"], word["bbox"]) for row in rows), default=0.0)
            best_ious.append(best)
            if any(intersection_area(row["bbox"], word["bbox"]) > 0 for row in rows):
                available_predictions += 1
        for row in rows:
            overlaps = []
            for word in words:
                inter = intersection_area(row["bbox"], word["bbox"])
                area = max(1, (word["bbox"][2] - word["bbox"][0]) * (word["bbox"][3] - word["bbox"][1]))
                overlaps.append((box_iou(row["bbox"], word["bbox"]), inter / area))
            significant = sum(coverage >= .25 for _, coverage in overlaps)
            best = max((iou for iou, _ in overlaps), default=0.0)
            category = (
                "MULTIWORD_MERGE" if significant >= 2 else
                "WORD_GOOD" if best >= .5 else
                "WORD_PARTIAL" if significant == 1 else "UNMATCHED"
            )
            wordness[category] += 1
    iou25 = sum(value >= .25 for value in best_ious)
    iou50 = sum(value >= .50 for value in best_ious)
    iou75 = sum(value >= .75 for value in best_ious)
    observed_control = {
        "available": available_predictions, "iou25": iou25, "iou50": iou50,
        "iou75": iou75, **{name: wordness[name] for name in (
            "WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED"
        )},
    }
    expected_control = {
        "available": 745, "iou25": 679, "iou50": 568, "iou75": 324,
        "WORD_GOOD": 552, "WORD_PARTIAL": 130,
        "MULTIWORD_MERGE": 37, "UNMATCHED": 484,
    }
    if observed_control != expected_control:
        raise RuntimeError(
            f"frozen v1 geometry/wordness control mismatch: {observed_control}"
        )
    precision, recall = iou50 / EXPECTED_V1_CANDIDATES, iou50 / 849
    geometry = {
        "gt_words": 849, "available_predictions": available_predictions,
        "gt_with_proposal_iou_ge_0_25": iou25,
        "gt_with_proposal_iou_ge_0_50": iou50,
        "gt_with_proposal_iou_ge_0_75": iou75,
        "retained_proposals": EXPECTED_V1_CANDIDATES,
        "proposal_precision_iou_ge_0_50": precision,
        "recall_iou_ge_0_50": recall,
        "f1_proposal_level_iou_ge_0_50": 2 * precision * recall / (precision + recall),
        "average_proposals_per_image": EXPECTED_V1_CANDIDATES / 229,
        "wordness": {name: wordness[name] for name in (
            "WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED"
        )},
    }

    gate = {}
    ranking = {}
    for field in FIELDS:
        image_rows = []
        for sample in manifest.samples:
            rows = candidates[sample.sample_id]
            selected = max(enumerate(rows), key=lambda item: (item[1]["context"][field]["confidence"], -item[0]))[1] if rows else None
            confidence = float(selected["context"][field]["confidence"]) if selected else 0.0
            spatial = bool(selected) and any(
                selected["ocr"]["raw_text"].casefold() == word["text"].casefold()
                and box_iou(selected["bbox"], word["bbox"]) >= .5 for word in gt[sample.sample_id]
            )
            image_rows.append({"accepted": confidence >= TAU, "spatial": spatial, "confidence": confidence})
        accepted = [row for row in image_rows if row["accepted"]]
        gate[field] = {
            "images": 229, "no_fallback": len(accepted), "fallback": 229 - len(accepted),
            "accepted_spatial_exact": sum(row["spatial"] for row in accepted),
            "false_accept_spatial": sum(not row["spatial"] for row in accepted),
            "accepted_spatial_precision": (sum(row["spatial"] for row in accepted) / len(accepted)) if accepted else None,
            "confidence_median": float(np.median([row["confidence"] for row in image_rows])),
        }

        eligible = []; exact = 0; fuzzy = 0
        for sample in manifest.samples:
            rows = candidates[sample.sample_id]
            ordered = sorted(rows, key=lambda row: (-row["context"][field]["confidence"], row["candidate_id"]))
            for word in gt[sample.sample_id]:
                geometric = [row for row in rows if box_iou(row["bbox"], word["bbox"]) >= .5]
                best = max(rows, key=lambda row: box_iou(row["bbox"], word["bbox"]), default=None)
                if best and intersection_area(best["bbox"], word["bbox"]) > 0:
                    text = str(best["ocr"]["raw_text"])
                    exact += text.casefold() == word["text"].casefold()
                    fuzzy += ratio(text.casefold(), word["text"].casefold()) >= 80
                if geometric:
                    ranks = [index + 1 for index, row in enumerate(ordered) if row in geometric]
                    eligible.append(min(ranks))
        ranking[field] = {
            "eligible_gt_words": len(eligible),
            "top1_iou_ge_0_50": sum(rank <= 1 for rank in eligible),
            "top3_iou_ge_0_50": sum(rank <= 3 for rank in eligible),
            "top5_iou_ge_0_50": sum(rank <= 5 for rank in eligible),
            "median_rank_first_iou_ge_0_50": float(np.median(eligible)) if eligible else None,
            "geometric_best_parseq_exact_casefold_all_849": exact,
            "geometric_best_parseq_fuzzy_ge_0_8_all_849": fuzzy,
        }
    controls = {(value["geometric_best_parseq_exact_casefold_all_849"],
                 value["geometric_best_parseq_fuzzy_ge_0_8_all_849"])
                for value in ranking.values()}
    if controls != {(502, 578)}:
        raise RuntimeError(f"frozen v1 PARSeq control mismatch: {sorted(controls)}")
    quality = [repetition_diagnostics(captions[sample.sample_id]) for sample in manifest.samples]
    report = {
        "schema_version": 1, "mode": "V1_LOCAL_CONTEXT_TRAIN_PERSISTENT_V1",
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC", "table_5_claim": False,
        "scope": {"split": "TRAIN", "images": 229, "gt_words": 849},
        "scientific_boundary": {"test_loaded": False, "training": False,
                                "gt_first_loaded_after_context_scoring": True,
                                "T1_full_image_excluded": True, "tau": TAU},
        "caption": {"profile": profile, "manifest_sha256": _sha(caption_path),
                    "repeated_4gram_captions": sum(row["has_duplicate_ngram"] for row in quality),
                    "median_token_count": float(np.median([row["token_count"] for row in quality]))},
        "inputs": {"compact_manifest_sha256": _sha(input_path), "candidate_count": source["candidate_count"]},
        "v1_control": geometry, "gate": gate, "ranking": ranking,
        "mpnet": {"runtime": runtime, "release": release},
    }
    _write_exclusive(output, report)
    print(json.dumps({"caption": report["caption"], "gate": gate, "ranking": ranking}, indent=2), flush=True)
    print(f"JSON {output}", flush=True)
    print(f"SHA256 {_sha(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
