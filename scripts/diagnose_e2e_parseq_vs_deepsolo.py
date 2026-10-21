#!/usr/bin/env python3
"""Compare frozen E2E PARSeq proposals with raw and reduced DeepSolo outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from localize.practical_candidate_selection import box_iou
from recognition.icdar13_annotations import load_icdar13_recognition_samples


EXPECTED_E2E_SHA256 = "e6db8027e1d0f6ec3bf5e3b574c431ad1779857330825cb14bf766f505747eee"
EXPECTED_IMAGES = 233
EXPECTED_WORDS = 1095
BRANCHES = ("faithful", "balanced")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--recognition-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
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


def detection_envelope(detection: dict[str, Any]) -> list[float]:
    points = detection.get("geometry", {}).get("boundary_points", [])
    if len(points) < 3 or any(not isinstance(point, list) or len(point) != 2 for point in points):
        raise ValueError("DeepSolo detection lacks a valid boundary polygon")
    xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def exact_spatial_edges(
    predictions: Sequence[dict[str, Any]], words: Sequence[dict[str, Any]], *, threshold: float = 0.5,
) -> list[tuple[int, int]]:
    edges = []
    for prediction_index, prediction in enumerate(predictions):
        for word_index, word in enumerate(words):
            if (
                str(prediction["text"]).casefold() == str(word["text"]).casefold()
                and box_iou(prediction["bbox"], word["bbox"]) >= threshold
            ):
                edges.append((prediction_index, word_index))
    return edges


def maximum_word_matching(
    prediction_count: int, word_count: int, edges: Sequence[tuple[int, int]],
) -> set[int]:
    """Return GT indices in a deterministic maximum bipartite matching."""
    adjacency = {index: [] for index in range(prediction_count)}
    for prediction, word in edges:
        adjacency[prediction].append(word)
    for values in adjacency.values():
        values.sort()
    assigned: dict[int, int] = {}

    def visit(prediction: int, seen: set[int]) -> bool:
        for word in adjacency[prediction]:
            if word in seen:
                continue
            seen.add(word)
            if word not in assigned or visit(assigned[word], seen):
                assigned[word] = prediction
                return True
        return False

    for prediction in range(prediction_count):
        visit(prediction, set())
    if any(not 0 <= word < word_count for word in assigned):
        raise RuntimeError("matching produced an invalid GT index")
    return set(assigned)


def _summary(rows: Sequence[dict[str, Any]], branch: str) -> dict[str, Any]:
    def count(field: str) -> int:
        return sum(bool(row[branch][field]) for row in rows)

    parseq = {index for index, row in enumerate(rows) if row[branch]["parseq_exact_iou_ge_0_50"]}
    deep_spatial = {index for index, row in enumerate(rows) if row["deepsolo_exact_iou_ge_0_50"]}
    deep_text = {index for index, row in enumerate(rows) if row["deepsolo_exact_text_in_image"]}
    return {
        "gt_words": len(rows),
        "parseq_exact_iou_ge_0_50": len(parseq),
        "parseq_exact_iou_ge_0_50_one_to_one": count("parseq_one_to_one_match"),
        "deepsolo_raw_exact_iou_ge_0_50": len(deep_spatial),
        "deepsolo_raw_exact_iou_ge_0_50_one_to_one": sum(row["deepsolo_one_to_one_match"] for row in rows),
        "parseq_correct_deepsolo_no_spatial_exact": len(parseq - deep_spatial),
        "parseq_correct_deepsolo_no_exact_text_anywhere_in_image": len(parseq - deep_text),
        "both_spatially_correct": len(parseq & deep_spatial),
        "deepsolo_spatially_correct_parseq_not": len(deep_spatial - parseq),
        "selected_T3_exact_iou_ge_0_50": count("selected_T3_exact_iou_ge_0_50"),
        "selected_T3_correct_but_deepsolo_selected_wrong": count("selected_T3_correct_deepsolo_selected_wrong"),
    }


def main() -> int:
    args = _args()
    report_path, artifact_root = args.e2e_report.resolve(), args.artifact_dir.resolve()
    if _sha(report_path) != EXPECTED_E2E_SHA256:
        raise RuntimeError("frozen E2E report SHA256 mismatch")
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        source.get("scope") != {"gt_words": EXPECTED_WORDS, "images": EXPECTED_IMAGES, "split": "official_test_233"}
        or source.get("scientific_boundary", {}).get("accuracy") is not None
        or source.get("scientific_boundary", {}).get("table_5_claim") is not False
    ):
        raise RuntimeError("E2E report scientific boundary mismatch")
    samples, dataset = load_icdar13_recognition_samples(
        args.recognition_manifest.resolve(), args.data_root.resolve()
    )
    rows = []
    for number, sample in enumerate(samples, start=1):
        sample_id = sample.sample_id
        routing_path = artifact_root / "04-routing" / f"{int(sample_id):04d}.json"
        deep_path = artifact_root / "05-deepsolo" / f"{int(sample_id):04d}.json"
        routing, deep = json.loads(routing_path.read_text()), json.loads(deep_path.read_text())
        if routing["sample_id"] != sample_id or deep["sample_id"] != sample_id:
            raise RuntimeError(f"sample shard mismatch: {sample_id}")
        response = deep.get("response")
        if not deep.get("invoked") or response is None or response.get("success") is not True:
            raise RuntimeError(f"expected successful frozen DeepSolo response: {sample_id}")
        words = [{"bbox": list(word.bbox_xyxy), "text": word.transcription} for word in sample.words]
        detections = [{
            "text": detection["text"], "bbox": detection_envelope(detection),
            "source_index": detection["source_index"], "score": detection["score"],
        } for detection in response["detections"]]
        deep_edges = exact_spatial_edges(detections, words)
        deep_matched = maximum_word_matching(len(detections), len(words), deep_edges)
        by_branch = {}
        for branch in BRANCHES:
            proposals = routing[branch]["proposals"]
            parseq_edges = exact_spatial_edges(proposals, words)
            parseq_matched = maximum_word_matching(len(proposals), len(words), parseq_edges)
            selected = routing[branch]["route"]["selected_crop"]
            values = []
            for word_index, word in enumerate(words):
                parseq_candidates = [
                    proposal for proposal in proposals
                    if proposal["text"].casefold() == word["text"].casefold()
                    and box_iou(proposal["bbox"], word["bbox"]) >= 0.5
                ]
                selected_correct = bool(
                    selected is not None
                    and str(selected["text"]).casefold() == word["text"].casefold()
                    and box_iou(selected["bbox"], word["bbox"]) >= 0.5
                )
                values.append({
                    "parseq_exact_iou_ge_0_50": bool(parseq_candidates),
                    "parseq_one_to_one_match": word_index in parseq_matched,
                    "parseq_candidate_ids": [value["candidate_id"] for value in parseq_candidates],
                    "selected_T3_exact_iou_ge_0_50": selected_correct,
                    "selected_T3_correct_deepsolo_selected_wrong": (
                        selected_correct and response["selected_output"].casefold() != word["text"].casefold()
                    ),
                })
            by_branch[branch] = values
        deep_texts = {value["text"].casefold() for value in detections}
        for word_index, word in enumerate(words):
            row = {
                "sample_id": sample_id, "gt_word_index": word_index + 1,
                "gt_text": word["text"], "gt_bbox": word["bbox"],
                "deepsolo_exact_text_in_image": word["text"].casefold() in deep_texts,
                "deepsolo_exact_iou_ge_0_50": any(edge[1] == word_index for edge in deep_edges),
                "deepsolo_one_to_one_match": word_index in deep_matched,
                "deepsolo_selected_output": response["selected_output"],
                "faithful": by_branch["faithful"][word_index],
                "balanced": by_branch["balanced"][word_index],
            }
            rows.append(row)
        print(f"[PARSeq vs DeepSolo] {number}/{EXPECTED_IMAGES} sample_id={sample_id}", flush=True)
    if len(rows) != EXPECTED_WORDS:
        raise RuntimeError("GT word population mismatch")
    summaries = {branch: _summary(rows, branch) for branch in BRANCHES}
    examples = {
        branch: [
            row for row in rows
            if row[branch]["parseq_exact_iou_ge_0_50"] and not row["deepsolo_exact_iou_ge_0_50"]
        ][:40]
        for branch in BRANCHES
    }
    output = {
        "schema_version": 1,
        "classification": "APPROXIMATE_DIAGNOSTIC",
        "scope": {"split": "official_test_233", "images": EXPECTED_IMAGES, "gt_words": EXPECTED_WORDS},
        "comparison": summaries,
        "examples_parseq_correct_deepsolo_no_spatial_exact": examples,
        "per_word": rows,
        "matching": {
            "parseq_correct": "casefold exact transcription AND proposal bbox IoU>=0.50; reports existence and one-to-one matching",
            "deepsolo_correct": "casefold exact transcription AND axis-aligned boundary-polygon envelope IoU>=0.50; reports existence and one-to-one matching",
            "text_only": "same-image raw DeepSolo detection exact casefold, without geometry",
            "gt_used": "post-hoc diagnostic only",
        },
        "provenance": {
            "e2e_report": str(report_path), "e2e_report_sha256": EXPECTED_E2E_SHA256,
            "artifact_dir": str(artifact_root), "recognition_dataset": dataset,
            "source_sha256": _sha(Path(__file__).resolve()),
        },
        "scientific_boundary": {"table_5_claim": False, "accuracy": None},
    }
    _write_atomic(args.output.resolve(), output)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"JSON {args.output.resolve()}\nSHA256 {_sha(args.output.resolve())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
