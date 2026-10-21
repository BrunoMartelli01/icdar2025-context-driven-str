#!/usr/bin/env python3
"""GPU PARSeq ranking diagnostic for frozen TRAIN multi-hypothesis proposals.

Candidate generation uses only cached native probability maps and RGB images.
GT files are opened only after candidate generation and all PARSeq outputs are
complete.  OCR is resumable through deterministic, atomic batch shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image

from localize.multihypothesis_word_proposals import (
    BBOX_VARIANTS,
    DEDUPLICATION_IOU,
    THRESHOLDS,
    deduplicate_candidates,
    enumerate_native_hypotheses,
    materialize_bbox_variants,
)
from localize.practical_remap_protocol import train_task22_word_boxes
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256,
    PARSEQ_RUNTIME_REVISION,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)
from recognition.parseq_candidate_ranking import aggregate_rankings, evaluate_gt_word
from scripts.diagnose_multihypothesis_word_proposal_oracle_train import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PROBABILITY_MANIFEST_SHA256,
    EXPECTED_V1_SHA256,
    _load_probability,
)
from scripts.evaluate_supervised_word_boundary_train import _sha256_file
from segmentation.data import load_segmentation_manifest


MODE = "PARSEQ_CANDIDATE_RANKING_TRAIN_V1"
GENERATOR_THRESHOLD_ORDER = (0.50, 0.40, 0.30, 0.20)
MAX_SPAN = 12
EXPECTED_SPLIT = "official-training-all-229-author-protocol"
TASK22_LINE = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+).*"(.*)"\s*$')


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _contained(root: Path, relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("artifact path must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes declared root")
    return resolved


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_text_exclusive(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _task22_word_texts(path: Path) -> tuple[str, ...]:
    groups: list[str] = []
    current: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            if current:
                groups.append("".join(current)); current = []
            continue
        if raw.lstrip().startswith("#"):
            continue
        match = TASK22_LINE.match(raw)
        if match is None:
            raise ValueError(f"malformed Task2.2 transcription record: {path}")
        current.append(match.group(4))
    if current:
        groups.append("".join(current))
    return tuple(groups)


def _generator_contract(probability_manifest: Path) -> dict[str, Any]:
    core = Path(__file__).resolve().parents[1] / "localize/multihypothesis_word_proposals.py"
    return {
        "probability_manifest": str(probability_manifest),
        "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
        "thresholds": list(THRESHOLDS),
        "canonical_threshold_order": list(GENERATOR_THRESHOLD_ORDER),
        "max_contiguous_components": MAX_SPAN,
        "line_robust": False,
        "bbox_variants": list(BBOX_VARIANTS),
        "deduplication_iou_greater_than_or_equal": DEDUPLICATION_IOU,
        "deduplication_rule": "stable greedy; keep first; no GT score",
        "top10": False,
        "generator_source": str(core),
        "generator_source_sha256": _sha256_file(core),
    }


def _generate_artifacts(
    *, artifact_dir: Path, probability_manifest: Path, probability_report: dict[str, Any],
    manifest: Any, data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = _generator_contract(probability_manifest)
    generation_manifest_path = artifact_dir / "generation-manifest.json"
    metadata_path = artifact_dir / "candidates.jsonl"
    if artifact_dir.exists():
        if not generation_manifest_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("existing artifact directory is incomplete; preserve it for inspection")
        generation = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        if generation.get("generator_contract") != contract:
            raise RuntimeError("existing candidate artifacts use a different generator contract")
        if _sha256_file(metadata_path) != generation.get("candidate_metadata_sha256"):
            raise RuntimeError("existing candidate metadata SHA mismatch")
        candidates = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        if len(candidates) != generation.get("candidate_count"):
            raise RuntimeError("existing candidate metadata count mismatch")
        print(f"[candidate generation] verified reusable artifacts: {len(candidates)} candidates", flush=True)
        return candidates, generation

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="parseq-candidate-ranking-generation-", dir=artifact_dir.parent))
    candidate_stream = None
    records: list[dict[str, Any]] = []
    samples = []
    probability_by_id = {str(value["sample_id"]): value for value in probability_report["samples"]}
    try:
        crops_root = staging / "crops"
        crops_root.mkdir()
        candidate_metadata = staging / "candidates.jsonl"
        candidate_stream = candidate_metadata.open("w", encoding="utf-8")
        for sample_number, sample in enumerate(manifest.samples, 1):
            probability_record = probability_by_id[sample.sample_id]
            image_path = _contained(data_root, sample.image_path)
            if _sha256_file(image_path) != probability_record["image_sha256"]:
                raise RuntimeError(f"TRAIN RGB SHA mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            probability = _load_probability(probability_manifest.parent, probability_record)
            master = []
            for threshold in GENERATOR_THRESHOLD_ORDER:
                mask = np.ascontiguousarray((probability >= threshold).astype(np.uint8))
                hypotheses = enumerate_native_hypotheses(mask, max_components=MAX_SPAN)
                master.extend(
                    value for value in materialize_bbox_variants(
                        hypotheses, threshold=threshold,
                        width=rgb.shape[1], height=rgb.shape[0],
                    ) if value["source_kind"] == "same_line"
                )
            candidates = deduplicate_candidates(master, threshold=DEDUPLICATION_IOU)
            sample_dir = crops_root / f"{int(sample.sample_id):04d}"
            sample_dir.mkdir()
            sample_records = []
            for index, candidate in enumerate(candidates, 1):
                candidate_id = f"{int(sample.sample_id):04d}:{index:05d}"
                bbox = tuple(int(value) for value in candidate["bbox_original"])
                x1, y1, x2, y2 = bbox
                crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                if crop.size == 0:
                    raise RuntimeError(f"empty proposal crop: {candidate_id}")
                crop_path = sample_dir / f"candidate-{index:05d}.png"
                Image.fromarray(crop, mode="RGB").save(crop_path)
                record = {
                    "candidate_id": candidate_id,
                    "sample_id": sample.sample_id,
                    "candidate_index": index,
                    "bbox": list(bbox),
                    "crop_path": str(crop_path.relative_to(staging)),
                    "crop_sha256": _sha256_file(crop_path),
                    "crop_shape_hwc": list(crop.shape),
                    "source_threshold": candidate["threshold"],
                    "source_kind": candidate["source_kind"],
                    "source_line_ids": list(candidate["source_line_ids"]),
                    "source_span_start": candidate["start"],
                    "source_span_end": candidate["end"],
                    "source_raw_component_ids": list(candidate["raw_component_ids"]),
                    "source_bbox_native": list(candidate["bbox_native"]),
                    "bbox_variant": candidate["bbox_variant"],
                    "adaptive_pad_original": candidate["adaptive_pad_original"],
                }
                candidate_stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                records.append(record); sample_records.append(record)
            samples.append({
                "sample_id": sample.sample_id,
                "image_sha256": probability_record["image_sha256"],
                "candidate_count": len(sample_records),
                "candidate_ids_sha256": _json_digest([value["candidate_id"] for value in sample_records]),
            })
            print(
                f"[candidate generation/no-GT] {sample_number}/229 "
                f"sample_id={sample.sample_id} candidates={len(sample_records)}",
                flush=True,
            )
        candidate_stream.flush(); os.fsync(candidate_stream.fileno()); candidate_stream.close()
        candidate_stream = None
        generation = {
            "schema_version": 1,
            "artifact": "parseq-candidate-ranking-train-v1-crops",
            "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
            "gt_loaded": False,
            "test_loaded": False,
            "generator_contract": contract,
            "sample_count": len(samples),
            "candidate_count": len(records),
            "candidate_metadata": "candidates.jsonl",
            "candidate_metadata_sha256": _sha256_file(candidate_metadata),
            "samples": samples,
        }
        _write_json_atomic(staging / "generation-manifest.json", generation)
        os.replace(staging, artifact_dir)
        print(f"[candidate generation/no-GT] COMPLETE candidates={len(records)}", flush=True)
        return records, generation
    except BaseException:
        if candidate_stream is not None:
            candidate_stream.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_crop(artifact_dir: Path, candidate: dict[str, Any]) -> np.ndarray:
    path = _contained(artifact_dir, candidate["crop_path"])
    if _sha256_file(path) != candidate["crop_sha256"]:
        raise RuntimeError(f"proposal crop SHA mismatch: {candidate['candidate_id']}")
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    if list(rgb.shape) != candidate["crop_shape_hwc"]:
        raise RuntimeError(f"proposal crop shape mismatch: {candidate['candidate_id']}")
    return np.ascontiguousarray(rgb)


def _validate_ocr_shard(
    path: Path, candidate_ids: list[str], batch_size: int,
) -> list[dict[str, Any]]:
    shard = json.loads(path.read_text(encoding="utf-8"))
    if (
        shard.get("candidate_ids") != candidate_ids
        or shard.get("batch_size") != batch_size
        or shard.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
        or shard.get("parseq_runtime_revision") != PARSEQ_RUNTIME_REVISION
        or shard.get("parseq_runtime_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or shard.get("parseq_adapter_sha256") != _sha256_file(
            Path(__file__).resolve().parents[1] / "recognition/parseq.py"
        )
        or [value["candidate_id"] for value in shard.get("results", [])] != candidate_ids
    ):
        raise RuntimeError(f"OCR shard provenance/cardinality mismatch: {path}")
    return shard["results"]


def _run_or_resume_ocr(
    *, artifact_dir: Path, candidates: list[dict[str, Any]], batch_size: int,
    parseq_source: Path, parseq_source_archive: Path, parseq_checkpoint: Path, device: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    ocr_dir = artifact_dir / f"ocr-batch-{batch_size:04d}"
    ocr_dir.mkdir(exist_ok=True)
    batch_count = math.ceil(len(candidates) / batch_size)
    pending = []
    for batch_index in range(batch_count):
        path = ocr_dir / f"batch-{batch_index:06d}.json"
        if not path.exists():
            pending.append(batch_index)
    recognizer = None
    parseq_release = {"model_reference_released": False, "transform_reference_released": False}
    if pending:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=parseq_source,
            checkpoint=parseq_checkpoint,
            device=device,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=parseq_source_archive,
        ))
    try:
        for batch_index in range(batch_count):
            start, end = batch_index * batch_size, min(len(candidates), (batch_index + 1) * batch_size)
            batch_candidates = candidates[start:end]
            candidate_ids = [value["candidate_id"] for value in batch_candidates]
            path = ocr_dir / f"batch-{batch_index:06d}.json"
            if path.exists():
                _validate_ocr_shard(path, candidate_ids, batch_size)
                state = "verified"
            else:
                if recognizer is None:
                    raise RuntimeError("missing PARSeq runtime for pending OCR shard")
                results = recognizer.recognize_candidate_batch([
                    _load_crop(artifact_dir, value) for value in batch_candidates
                ])
                shard = {
                    "schema_version": 1,
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "candidate_ids": candidate_ids,
                    "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                    "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
                    "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                    "parseq_adapter_sha256": _sha256_file(
                        Path(__file__).resolve().parents[1] / "recognition/parseq.py"
                    ),
                    "results": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "transcription": result.raw_text,
                            "raw_text": result.raw_text,
                            "sequence_confidence_native": result.sequence_confidence_native,
                            "sequence_confidence_native_aggregation":
                                result.sequence_confidence_native_aggregation,
                            "mean_token_log_probability": result.mean_token_log_probability,
                            "minimum_token_probability": result.minimum_token_probability,
                            "predicted_length": result.predicted_length,
                        }
                        for candidate, result in zip(batch_candidates, results)
                    ],
                }
                _write_json_atomic(path, shard)
                state = "written"
            print(
                f"[PARSeq batch] {batch_index + 1}/{batch_count} "
                f"candidates={end}/{len(candidates)} state={state}", flush=True,
            )
    finally:
        if recognizer is not None:
            parseq_release = recognizer.release()

    output: dict[str, dict[str, Any]] = {}
    shard_records = []
    for batch_index in range(batch_count):
        start, end = batch_index * batch_size, min(len(candidates), (batch_index + 1) * batch_size)
        candidate_ids = [value["candidate_id"] for value in candidates[start:end]]
        path = ocr_dir / f"batch-{batch_index:06d}.json"
        results = _validate_ocr_shard(path, candidate_ids, batch_size)
        for result in results:
            if result["candidate_id"] in output:
                raise RuntimeError("duplicate candidate OCR result")
            output[result["candidate_id"]] = result
        shard_records.append({
            "path": str(path.relative_to(artifact_dir)), "sha256": _sha256_file(path),
            "candidate_count": len(results),
        })
    if set(output) != {value["candidate_id"] for value in candidates}:
        raise RuntimeError("OCR result population differs from candidate population")
    ocr_manifest = {
        "batch_size": batch_size, "batch_count": batch_count,
        "candidate_count": len(output),
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
        "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "parseq_release": parseq_release,
        "shards": shard_records,
    }
    _write_json_atomic(ocr_dir / "manifest.json", ocr_manifest)
    return output, ocr_manifest


def main() -> int:
    args = _args()
    if args.device != "cuda:0":
        raise ValueError("qualified PARSeq ranking command requires exactly --device cuda:0")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    probability_manifest = args.probability_manifest.resolve()
    train_manifest = args.train_manifest.resolve()
    data_root = args.data_root.resolve()
    artifact_dir = args.artifact_dir.resolve()
    v1_path = args.v1_calibration.resolve()
    if _sha256_file(probability_manifest) != EXPECTED_PROBABILITY_MANIFEST_SHA256:
        raise RuntimeError("TRAIN probability manifest SHA256 mismatch")
    if _sha256_file(v1_path) != EXPECTED_V1_SHA256:
        raise RuntimeError("frozen v1 calibration SHA256 mismatch")
    if _sha256_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError("qualified PARSeq checkpoint SHA256 mismatch")
    probability_report = json.loads(probability_manifest.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(train_manifest)
    if (
        probability_report.get("sample_count") != 229
        or probability_report.get("test_data_loaded") is not False
        or probability_report.get("gt_loaded") is not False
        or probability_report.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or probability_report.get("train_manifest", {}).get("sha256") != _sha256_file(train_manifest)
        or manifest.split != EXPECTED_SPLIT
        or manifest.split_role != "train"
        or len(manifest.samples) != 229
    ):
        raise RuntimeError("requires frozen TRAIN-229 probability cache and manifest")

    candidates, generation = _generate_artifacts(
        artifact_dir=artifact_dir,
        probability_manifest=probability_manifest,
        probability_report=probability_report,
        manifest=manifest,
        data_root=data_root,
    )
    ocr_by_id, ocr_manifest = _run_or_resume_ocr(
        artifact_dir=artifact_dir, candidates=candidates, batch_size=args.batch_size,
        parseq_source=args.parseq_source.resolve(),
        parseq_source_archive=args.parseq_source_archive.resolve(),
        parseq_checkpoint=args.parseq_checkpoint.resolve(), device=args.device,
    )

    # GT is first opened here, after every candidate and OCR output is frozen.
    v1_report = json.loads(v1_path.read_text(encoding="utf-8"))
    source_by_id = {str(value["sample_id"]): value for value in v1_report["train_source"]["samples"]}
    candidates_by_sample: dict[str, list[dict[str, Any]]] = {}
    combined_candidates = []
    for candidate in candidates:
        combined = {**candidate, "ocr": ocr_by_id[candidate["candidate_id"]]}
        candidates_by_sample.setdefault(candidate["sample_id"], []).append(combined)
        combined_candidates.append(combined)

    eligible_rows = []
    all_words = []
    for sample_number, sample in enumerate(manifest.samples, 1):
        source = source_by_id[sample.sample_id]
        target_path = _contained(data_root, sample.mask_path)
        annotation_path = _contained(data_root, sample.validity_annotation_path)
        if (
            _sha256_file(target_path) != source["mask_sha256"]
            or _sha256_file(annotation_path) != source["annotation_sha256"]
        ):
            raise RuntimeError(f"TRAIN GT identity mismatch: {sample.sample_id}")
        boxes = train_task22_word_boxes(target_path, annotation_path)
        texts = _task22_word_texts(annotation_path)
        if len(boxes) != len(texts) or len(boxes) != source["word_box_count"]:
            raise RuntimeError(f"TRAIN GT word/transcription mismatch: {sample.sample_id}")
        for word_index, (word, text) in enumerate(zip(boxes, texts), 1):
            evaluation = evaluate_gt_word(
                candidates_by_sample.get(sample.sample_id, ()),
                gt_bbox=word.bbox, gt_text=text,
            )
            row = {
                "sample_id": sample.sample_id, "gt_word_index": word_index,
                "eligible_iou_ge_0_50_candidate_exists": evaluation is not None,
            }
            if evaluation is not None:
                row.update(evaluation); eligible_rows.append(row)
            all_words.append(row)
        if sample_number % 25 == 0 or sample_number == 229:
            print(f"[ranking evaluation/post-hoc GT] {sample_number}/229", flush=True)
    if len(all_words) != 849:
        raise RuntimeError(f"expected 849 TRAIN GT words, got {len(all_words)}")

    ranking = aggregate_rankings(eligible_rows)
    report = {
        "schema_version": 1,
        "mode": MODE,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "table_5_claim": False,
        "scope": {
            "split": "TRAIN", "images": 229, "gt_words": len(all_words),
            "eligible_gt_words_with_candidate_iou_ge_0_50": len(eligible_rows),
        },
        "scientific_boundary": {
            "segmentation_rerun": False, "test_loaded": False,
            "training": False, "routing": False, "score_combination": False,
            "gt_first_loaded_after_candidate_and_ocr_completion": True,
        },
        "generator": generation["generator_contract"],
        "artifacts": {
            "directory": str(artifact_dir),
            "generation_manifest": str(artifact_dir / "generation-manifest.json"),
            "generation_manifest_sha256": _sha256_file(artifact_dir / "generation-manifest.json"),
            "candidate_metadata": str(artifact_dir / "candidates.jsonl"),
            "candidate_metadata_sha256": generation["candidate_metadata_sha256"],
            "ocr": ocr_manifest,
        },
        "parseq_scoring": {
            "A": "sequence_confidence_native",
            "B": "mean_token_log_probability",
            "C": "minimum_token_probability",
            "sequence_confidence_native": (
                "upstream PARSeq product of tokenizer-filtered greedy probabilities; includes EOS when present"
            ),
            "token_statistics": "decoded character positions only; EOS excluded",
            "empty_prediction_policy_B_C": "None sorts after every finite score",
            "ranking_order": "descending single score; candidate_id ascending exact-tie break",
            "geometric_association": "strictly positive bbox intersection with GT; post-hoc only",
        },
        "ranking": ranking,
        "gt_words": all_words,
        "candidates": combined_candidates,
        "provenance": {
            "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
            "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
            "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
            "v1_calibration_sha256": EXPECTED_V1_SHA256,
        },
        "source_sha256": {
            str(Path(__file__).resolve()): _sha256_file(Path(__file__).resolve()),
            str(Path(__file__).resolve().parents[1] / "recognition/parseq_candidate_ranking.py"):
                _sha256_file(Path(__file__).resolve().parents[1] / "recognition/parseq_candidate_ranking.py"),
            str(Path(__file__).resolve().parents[1] / "recognition/parseq.py"):
                _sha256_file(Path(__file__).resolve().parents[1] / "recognition/parseq.py"),
        },
    }
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_exclusive(output, content)
    print(json.dumps({"scope": report["scope"], "ranking": ranking}, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {_sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
