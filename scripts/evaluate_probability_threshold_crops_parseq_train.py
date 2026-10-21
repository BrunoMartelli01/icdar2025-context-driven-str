#!/usr/bin/env python3
"""GPU PARSeq evaluation of frozen TRAIN threshold-sweep crop artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from localize.practical_remap_protocol import _box_iou, train_task22_word_boxes
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from segmentation.data import load_segmentation_manifest


EXPECTED_GEOMETRY_REPORT_SHA256 = "49038549de0ba4ce1b90e9c09e78cb261deb1b1ba8c467997010973aabc09913"
EXPECTED_CROP_MANIFEST_SHA256 = "f8cd3b710e3a93e4aabbfb4d6ecb095b292501c03c7eec8d94fa95659e70e193"
EXPECTED_TRAIN_MANIFEST_SHA256 = "3aeecf40411326f0721e5d4eea1bf74a9357eab1c9da54a77d0624b6e24f7c14"
EXPECTED_THRESHOLD = 0.30
EXPECTED_IMAGES = 229
EXPECTED_GT_WORDS = 849
TASK22_LINE = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+).*"(.*)"\s*$')


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _area(box: Sequence[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _wordness(box: Sequence[int], words: Sequence[Sequence[int]]) -> str:
    significant = sum(
        _intersection(box, word) / max(1, _area(word)) >= 0.25 for word in words
    )
    best_iou = max((_box_iou(box, word) for word in words), default=0.0)
    return (
        "MULTIWORD_MERGE" if significant >= 2 else
        "WORD_GOOD" if best_iou >= 0.50 else
        "WORD_PARTIAL" if significant == 1 else "UNMATCHED"
    )


def _texts(path: Path) -> tuple[str, ...]:
    groups, current = [], []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            if current:
                groups.append("".join(current)); current = []
            continue
        if raw.lstrip().startswith("#"):
            continue
        match = TASK22_LINE.match(raw)
        if match is None:
            raise ValueError(f"malformed Task2.2 record: {path}")
        current.append(match.group(4))
    if current:
        groups.append("".join(current))
    return tuple(groups)


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_crop(root: Path, row: dict[str, Any]) -> np.ndarray:
    relative = Path(row["crop_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("crop path escapes cache")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or _sha_file(path) != row["crop_sha256"]:
        raise RuntimeError(f"crop SHA/path mismatch: {row['candidate_id']}")
    with Image.open(path) as opened:
        return np.ascontiguousarray(np.asarray(opened.convert("RGB")))


def _run_or_resume(
    *, candidates: list[dict[str, Any]], crop_root: Path, artifact_dir: Path,
    batch_size: int, parseq_source: Path, parseq_source_archive: Path,
    parseq_checkpoint: Path, device: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    batch_count = math.ceil(len(candidates) / batch_size)
    recognizer = None
    release = {"model_reference_released": False, "transform_reference_released": False}
    try:
        for batch_index in range(batch_count):
            start = batch_index * batch_size
            end = min(len(candidates), start + batch_size)
            batch = candidates[start:end]
            ids = [row["candidate_id"] for row in batch]
            path = artifact_dir / f"batch-{batch_index:05d}.json"
            if path.exists():
                shard = json.loads(path.read_text(encoding="utf-8"))
                if (
                    shard.get("candidate_ids") != ids
                    or shard.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
                    or shard.get("parseq_runtime_revision") != PARSEQ_RUNTIME_REVISION
                    or shard.get("parseq_runtime_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
                    or [row.get("candidate_id") for row in shard.get("results", ())] != ids
                ):
                    raise RuntimeError(f"existing OCR shard provenance mismatch: {path}")
                state = "verified"
            else:
                if recognizer is None:
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=parseq_source,
                        checkpoint=parseq_checkpoint,
                        device=device,
                        runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=parseq_source_archive,
                    ))
                results = recognizer.recognize_candidate_batch(
                    [_load_crop(crop_root, row) for row in batch]
                )
                shard = {
                    "schema_version": 1,
                    "candidate_ids": ids,
                    "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                    "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
                    "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                    "results": [
                        {"candidate_id": row["candidate_id"], "raw_text": result.raw_text}
                        for row, result in zip(batch, results)
                    ],
                }
                _write_atomic(path, shard)
                state = "written"
            print(
                f"[PARSeq threshold crops] {batch_index + 1}/{batch_count} "
                f"crops={end}/{len(candidates)} state={state}", flush=True,
            )
    finally:
        if recognizer is not None:
            release = recognizer.release()

    output: dict[str, str] = {}
    shards = []
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        end = min(len(candidates), start + batch_size)
        path = artifact_dir / f"batch-{batch_index:05d}.json"
        shard = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = [row["candidate_id"] for row in candidates[start:end]]
        if shard.get("candidate_ids") != expected_ids:
            raise RuntimeError("final OCR shard ordering mismatch")
        for row in shard["results"]:
            if row["candidate_id"] in output:
                raise RuntimeError("duplicate threshold-crop OCR result")
            output[row["candidate_id"]] = str(row["raw_text"])
        shards.append({"path": str(path), "sha256": _sha_file(path)})
    if len(output) != len(candidates):
        raise RuntimeError("threshold-crop OCR population incomplete")
    return output, {"batch_size": batch_size, "shards": shards, "release": release}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ocr-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.device != "cuda:0":
        raise ValueError("qualified threshold-crop PARSeq evaluation requires cuda:0")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    geometry_path = args.geometry_report.resolve()
    crop_manifest_path = args.crop_manifest.resolve()
    train_manifest_path = args.train_manifest.resolve()
    if _sha_file(geometry_path) != EXPECTED_GEOMETRY_REPORT_SHA256:
        raise RuntimeError("frozen threshold geometry report SHA256 mismatch")
    if _sha_file(crop_manifest_path) != EXPECTED_CROP_MANIFEST_SHA256:
        raise RuntimeError("frozen selected crop manifest SHA256 mismatch")
    if _sha_file(train_manifest_path) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("TRAIN manifest SHA256 mismatch")
    if _sha_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError("qualified PARSeq checkpoint SHA256 mismatch")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    if (
        geometry["selection_rule"]["selected_threshold"] != EXPECTED_THRESHOLD
        or geometry["selected_crop_manifest"]["sha256"] != EXPECTED_CROP_MANIFEST_SHA256
        or crop_manifest.get("selected_threshold") != EXPECTED_THRESHOLD
        or crop_manifest.get("sample_count") != EXPECTED_IMAGES
        or crop_manifest.get("test_loaded") is not False
        or crop_manifest.get("ocr_called") is not False
    ):
        raise RuntimeError("selected threshold/crop contract mismatch")
    candidates = [
        {**crop, "sample_id": str(sample["sample_id"])}
        for sample in crop_manifest["samples"] for crop in sample["crops"]
    ]
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("duplicate crop candidate IDs")

    # No GT is opened before all frozen crops have PARSeq outputs.
    ocr, ocr_manifest = _run_or_resume(
        candidates=candidates, crop_root=crop_manifest_path.parent,
        artifact_dir=args.ocr_artifact_dir.resolve(), batch_size=args.batch_size,
        parseq_source=args.parseq_source.resolve(),
        parseq_source_archive=args.parseq_source_archive.resolve(),
        parseq_checkpoint=args.parseq_checkpoint.resolve(), device=args.device,
    )

    manifest = load_segmentation_manifest(train_manifest_path)
    if (
        manifest.split != "official-training-all-229-author-protocol"
        or manifest.split_role != "train" or len(manifest.samples) != EXPECTED_IMAGES
    ):
        raise RuntimeError("PARSeq diagnostic requires TRAIN-229 only")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_sample.setdefault(candidate["sample_id"], []).append(candidate)
    per_word = []
    wordness = Counter()
    data_root = args.data_root.resolve()
    for image_index, sample in enumerate(manifest.samples, 1):
        target_path = (data_root / sample.mask_path).resolve()
        annotation_path = (data_root / sample.validity_annotation_path).resolve()
        words = tuple(row.bbox for row in train_task22_word_boxes(target_path, annotation_path))
        texts = _texts(annotation_path)
        if len(words) != len(texts):
            raise RuntimeError(f"GT geometry/text mismatch: {sample.sample_id}")
        sample_candidates = by_sample[sample.sample_id]
        category_by_id = {
            row["candidate_id"]: _wordness(row["padded_bbox_original"], words)
            for row in sample_candidates
        }
        wordness.update(category_by_id.values())
        for word_index, (word_box, truth) in enumerate(zip(words, texts), 1):
            best = max(
                sample_candidates,
                key=lambda row: (_box_iou(row["padded_bbox_original"], word_box), -row["rank"]),
                default=None,
            )
            available = best is not None and _intersection(best["padded_bbox_original"], word_box) > 0
            text = ocr[best["candidate_id"]] if available else None
            fuzzy = ratio(text.casefold(), truth.casefold()) / 100 if text is not None else None
            per_word.append({
                "sample_id": sample.sample_id,
                "gt_word_index": word_index,
                "gt_text": truth,
                "available": available,
                "candidate_id": best["candidate_id"] if available else None,
                "bbox_iou": _box_iou(best["padded_bbox_original"], word_box) if available else 0.0,
                "crop_wordness": category_by_id[best["candidate_id"]] if available else None,
                "parseq_text": text,
                "exact_casefold": text is not None and text.casefold() == truth.casefold(),
                "rapidfuzz_casefold": fuzzy,
                "rapidfuzz_casefold_ge_0_8": fuzzy is not None and fuzzy >= 0.80,
            })
        if image_index % 25 == 0 or image_index == EXPECTED_IMAGES:
            print(f"[PARSeq post-hoc GT] {image_index}/229", flush=True)
    if len(per_word) != EXPECTED_GT_WORDS:
        raise RuntimeError(f"expected 849 TRAIN GT words, got {len(per_word)}")
    exact = sum(row["exact_casefold"] for row in per_word)
    fuzzy = sum(row["rapidfuzz_casefold_ge_0_8"] for row in per_word)
    good = [row for row in per_word if row["crop_wordness"] == "WORD_GOOD"]
    expected_wordness = next(
        row for row in geometry["threshold_table"] if row["threshold"] == EXPECTED_THRESHOLD
    )
    for name in ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED"):
        if wordness[name] != expected_wordness[name]:
            raise RuntimeError(f"selected crop wordness drift: {name}")
    result = {
        "gt_words": EXPECTED_GT_WORDS,
        "available_predictions": sum(row["available"] for row in per_word),
        "exact_casefold": exact,
        "exact_rate": exact / EXPECTED_GT_WORDS,
        "rapidfuzz_casefold_ge_0_8": fuzzy,
        "fuzzy_rate": fuzzy / EXPECTED_GT_WORDS,
        "WORD_GOOD_matched_gt_words": len(good),
        "exact_on_WORD_GOOD": sum(row["exact_casefold"] for row in good),
        "exact_on_WORD_GOOD_rate": (
            sum(row["exact_casefold"] for row in good) / len(good) if good else 0.0
        ),
    }
    report = {
        "schema_version": 1,
        "classification": "APPROXIMATE_DIAGNOSTIC",
        "table_5_claim": False,
        "scope": {"split": "TRAIN", "images": 229, "gt_words": 849},
        "scientific_boundary": {
            "test_loaded": False, "training": False, "segmentation_rerun": False,
            "GT_first_opened_after_all_crop_OCR": True,
        },
        "selected_threshold": EXPECTED_THRESHOLD,
        "result": result,
        "frozen_comparators": {
            "GT_BOX_PARSeq_exact": 836,
            "practical_remap_v1_PARSeq_exact": 502,
        },
        "crop_wordness": dict(wordness),
        "ocr_artifacts": ocr_manifest,
        "provenance": {
            "geometry_report_sha256": EXPECTED_GEOMETRY_REPORT_SHA256,
            "crop_manifest_sha256": EXPECTED_CROP_MANIFEST_SHA256,
            "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
            "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
            "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        },
        "per_word": per_word,
    }
    _write_atomic(output, report)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"JSON {output}", flush=True)
    print(f"SHA256 {_sha_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
