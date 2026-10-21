#!/usr/bin/env python3
"""Recognize frozen native Algorithm 1 crops and score official TEST geometry/OCR."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)
from recognition.scene_text_v1_evaluation import evaluate_images


EXPECTED_IMAGES = {"ctw1500": 500, "icdar2015": 500, "total_text": 300}


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(EXPECTED_IMAGES), required=True)
    parser.add_argument("--algorithm1-run", type=Path, required=True)
    parser.add_argument("--v1-run", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_sources(args: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    algorithm1_run = args.algorithm1_run.resolve()
    result_path = algorithm1_run / "results.json"
    protocol_path = algorithm1_run / "protocol.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (result.get("classification") != "APPROXIMATE_PAPER_ALGORITHM1"
        or protocol.get("dataset") != args.dataset
        or result["summary"]["images"] != EXPECTED_IMAGES[args.dataset]
        or result["summary"]["dataset"] != args.dataset):
        raise ValueError("Algorithm 1 source does not match the requested official TEST run")
    if len(result["sample_shards"]) != EXPECTED_IMAGES[args.dataset]:
        raise ValueError("Algorithm 1 source has an incomplete sample list")
    protocol_sha = file_sha(protocol_path)
    rows = []
    seen = set()
    for relative in result["sample_shards"]:
        shard = json.loads((algorithm1_run / relative).read_text(encoding="utf-8"))
        sample_id = shard["sample_id"]
        if shard["protocol_sha256"] != protocol_sha or sample_id in seen:
            raise ValueError(f"invalid Algorithm 1 sample shard: {sample_id}")
        seen.add(sample_id)
        rows.append(shard)
    if sum(len(row["proposals"]) for row in rows) != result["summary"]["retained_crops"]:
        raise ValueError("Algorithm 1 crop count differs from its summary")
    return rows, {
        "algorithm1_results": str(result_path),
        "algorithm1_results_sha256": file_sha(result_path),
        "algorithm1_protocol_sha256": protocol_sha,
        "algorithm1_geometry": result["summary"]["geometry"],
    }, protocol


def _v1_reuse(args: argparse.Namespace, rows: list[dict]) -> tuple[dict, dict]:
    run = args.v1_run.resolve()
    result_path = run / "results.json"
    protocol_path = run / "protocol.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if result["summary"]["dataset"] != args.dataset or result["summary"]["images"] != EXPECTED_IMAGES[args.dataset]:
        raise ValueError("V1 comparison run does not match dataset/official TEST count")
    if (protocol["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256
        or protocol["parseq_source_archive_sha256"] != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("V1 PARSeq identity differs from the qualified runtime")
    protocol_sha = file_sha(protocol_path)
    reuse = {}
    for row in rows:
        key = row["sample_id"].replace(":", "__")
        frozen = json.loads((run / "proposals" / f"{key}.json").read_text(encoding="utf-8"))
        if (frozen["protocol_sha256"] != protocol_sha
            or frozen["sample_id"] != row["sample_id"]
            or frozen["image_sha256"] != row["image_sha256"]):
            raise ValueError(f"V1 provenance mismatch: {key}")
        if frozen["segmentation"]["native_mask_sha256"] != row["native_mask_array_sha256"]:
            raise ValueError(f"Algorithm 1/V1 mask identity mismatch: {key}")
        for proposal in frozen["proposals"]:
            ocr_path = run / "ocr" / f"{key}__{int(proposal['rank']):02d}.json"
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            if ocr["protocol_sha256"] != protocol_sha or ocr["sample_id"] != row["sample_id"]:
                raise ValueError(f"V1 OCR provenance mismatch: {ocr_path}")
            reuse[(row["sample_id"], tuple(proposal["bbox"]))] = {
                "raw_text": ocr["raw_text"],
                "sequence_confidence_native": ocr["sequence_confidence_native"],
                "mean_token_log_probability": ocr["mean_token_log_probability"],
                "minimum_token_probability": ocr["minimum_token_probability"],
                "predicted_length": ocr["predicted_length"],
                "reused_from": str(ocr_path),
                "reused_from_sha256": file_sha(ocr_path),
            }
    return reuse, {
        "v1_results": str(result_path),
        "v1_results_sha256": file_sha(result_path),
        "v1_protocol_sha256": protocol_sha,
        "v1_geometry": result["summary"]["geometry"],
        "v1_recognition": result["summary"].get("recognition"),
    }


def main() -> int:
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if file_sha(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint SHA256 mismatch")
    if file_sha(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq source archive SHA256 mismatch")
    rows, source, _ = _load_sources(args)
    reuse, v1_source = _v1_reuse(args, rows)
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        previous = report.get("protocol", {})
        if (previous.get("dataset") != args.dataset
            or previous.get("algorithm1_results_sha256") != source["algorithm1_results_sha256"]
            or previous.get("v1_results_sha256") != v1_source["v1_results_sha256"]
            or previous.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
            or previous.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256):
            raise ValueError("completed report provenance differs from current inputs")
        print(json.dumps(report["summary"], indent=2), flush=True)
        print(f"[reuse] {result_path} SHA256 {file_sha(result_path)}", flush=True)
        return 0
    protocol = {
        "schema_version": 1,
        "classification": "APPROXIMATE_PAPER_ALGORITHM1",
        "result_class": "official TEST Algorithm 1 crops + qualified PARSeq; not Table 5",
        "dataset": args.dataset,
        "images": EXPECTED_IMAGES[args.dataset],
        "annotation_granularity": "text_line" if args.dataset == "ctw1500" else "word",
        "matching": "maximum one-to-one matching; bbox IoU>=0.50",
        "strict_exact": "case-sensitive; punctuation retained",
        "casefold_exact": "Unicode casefold; punctuation retained",
        "algorithm1_results_sha256": source["algorithm1_results_sha256"],
        "v1_results_sha256": v1_source["v1_results_sha256"],
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "device": args.device,
        "batch_size": args.batch_size,
        "ground_truth_use": "post-hoc evaluation only",
        "table_5_claim": False,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing output has a different protocol")
    else:
        write_json(protocol_path, protocol)
    protocol_sha = file_sha(protocol_path)

    missing = []
    reused_count = 0
    for row in rows:
        key = row["sample_id"].replace(":", "__")
        for proposal in row["proposals"]:
            ocr_path = output / "ocr" / f"{key}__{int(proposal['rank']):02d}.json"
            crop_path = args.algorithm1_run.resolve() / proposal["crop_file"]
            if file_sha(crop_path) != proposal["crop_file_sha256"]:
                raise ValueError(f"Algorithm 1 crop SHA256 mismatch: {crop_path}")
            if ocr_path.exists():
                ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
                if ocr["protocol_sha256"] != protocol_sha or ocr["crop_file_sha256"] != proposal["crop_file_sha256"]:
                    raise ValueError(f"existing OCR shard mismatch: {ocr_path}")
                continue
            prior = reuse.get((row["sample_id"], tuple(proposal["padded_bbox_original"])))
            if prior is not None:
                write_json(ocr_path, {
                    "schema_version": 1, "protocol_sha256": protocol_sha,
                    "sample_id": row["sample_id"], "rank": proposal["rank"],
                    "crop_file_sha256": proposal["crop_file_sha256"],
                    "source": "identical_image_and_bbox_v1_reuse", **prior,
                })
                reused_count += 1
            else:
                missing.append((row, proposal, crop_path, ocr_path))
    print(f"[{args.dataset}] retained={sum(len(r['proposals']) for r in rows)} reused={reused_count} new={len(missing)}", flush=True)

    if missing:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source.resolve(),
            checkpoint=args.parseq_checkpoint.resolve(),
            device=args.device,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive.resolve(),
        ))
        try:
            for start in range(0, len(missing), args.batch_size):
                batch = missing[start:start + args.batch_size]
                crops = []
                for _, _, crop_path, _ in batch:
                    with Image.open(crop_path) as opened:
                        crops.append(np.ascontiguousarray(opened.convert("RGB")))
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(batch):
                    raise RuntimeError("PARSeq returned an unexpected batch length")
                for (row, proposal, _, ocr_path), prediction in zip(batch, predictions):
                    write_json(ocr_path, {
                        "schema_version": 1, "protocol_sha256": protocol_sha,
                        "sample_id": row["sample_id"], "rank": proposal["rank"],
                        "crop_file_sha256": proposal["crop_file_sha256"],
                        "source": "qualified_parseq_inference", **prediction.to_dict(),
                    })
                print(f"[{args.dataset} PARSeq] {min(start + len(batch), len(missing))}/{len(missing)}", flush=True)
        finally:
            recognizer.release()

    evaluated = []
    per_sample = []
    for row in rows:
        key = row["sample_id"].replace(":", "__")
        predictions = []
        for proposal in row["proposals"]:
            ocr = json.loads((output / "ocr" / f"{key}__{int(proposal['rank']):02d}.json").read_text(encoding="utf-8"))
            predictions.append({
                "rank": proposal["rank"],
                "bbox": proposal["padded_bbox_original"],
                "text": ocr["raw_text"],
                "confidence": ocr["sequence_confidence_native"],
            })
        value = {"sample_id": row["sample_id"], "ground_truth": row["ground_truth"], "predictions": predictions}
        evaluated.append(value)
        per_sample.append(value)
    summary = evaluate_images(evaluated, text_available=True)
    if summary["geometry"]["iou_ge_0_50_one_to_one"] != source["algorithm1_geometry"]["iou_ge_0_50_one_to_one"]:
        raise RuntimeError("recognition scoring changed the frozen Algorithm 1 geometry")
    report = {
        "schema_version": 1,
        "classification": "APPROXIMATE_PAPER_ALGORITHM1",
        "protocol": protocol,
        "sources": {**source, **v1_source},
        "summary": summary,
        "per_sample": per_sample,
        "ocr_reused_this_run": reused_count,
    }
    write_json(result_path, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"JSON {result_path} SHA256 {file_sha(result_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
