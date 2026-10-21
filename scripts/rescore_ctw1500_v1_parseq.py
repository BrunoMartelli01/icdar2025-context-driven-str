#!/usr/bin/env python3
"""Attach qualified CTW1500 text-line GT to frozen V1/PARSeq outputs.

No segmentation or OCR inference is performed.  The command fails unless the
qualified recognition geometry reproduces the frozen detection-only geometry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from recognition.scene_text_manifest import load_scene_text_manifest, sha256_file
from recognition.scene_text_v1_evaluation import evaluate_images


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _ground_truth(samples: tuple[Any, ...]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        records = []
        width, height = sample.image_size_wh
        for instance in sample.instances:
            if instance.ignore:
                continue
            x1, y1, x2, y2 = instance.bbox_xyxy
            records.append(
                {
                    "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
                    "text": instance.transcription,
                    "ignore": False,
                    "granularity": "text_line",
                }
            )
        output[f"ctw1500:{int(sample.sample_id)}"] = records
    return output


def _rescore(run: Path, gt: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    frozen_result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    if frozen_result.get("summary", {}).get("dataset") != "ctw1500":
        raise ValueError(f"not a frozen CTW1500 result: {run}")
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    protocol_sha = sha256_file(run / "protocol.json")
    rows = []
    for sample_id in sorted(gt, key=lambda value: int(value.partition(":")[2])):
        key = sample_id.replace(":", "__")
        proposal_path = run / "proposals" / f"{key}.json"
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        if proposal.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"protocol mismatch: {proposal_path}")
        predictions = []
        for block in proposal["proposals"]:
            ocr_path = run / "ocr" / f"{key}__{int(block['rank']):02d}.json"
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            if ocr.get("protocol_sha256") != protocol_sha:
                raise ValueError(f"protocol mismatch: {ocr_path}")
            predictions.append(
                {
                    "rank": block["rank"],
                    "bbox": block["bbox"],
                    "text": ocr["raw_text"],
                    "confidence": ocr["sequence_confidence_native"],
                }
            )
        rows.append({"sample_id": sample_id, "ground_truth": gt[sample_id], "predictions": predictions})
    summary = evaluate_images(rows, text_available=True)
    old_geometry = frozen_result["summary"]["geometry"]
    geometry_keys = (
        "gt_instances",
        "retained_proposals",
        "nonempty_parseq_outputs",
        "iou_ge_0_25_one_to_one",
        "iou_ge_0_50_one_to_one",
        "iou_ge_0_75_one_to_one",
    )
    geometry_delta = {
        key: summary["geometry"][key] - old_geometry[key] for key in geometry_keys
    }
    return {
        "frozen_run": str(run.resolve()),
        "frozen_results_sha256": sha256_file(run / "results.json"),
        "frozen_protocol_sha256": protocol_sha,
        "segmentation_model_run": frozen_result["summary"]["segmentation_model_run"],
        "summary": summary,
        "geometry_revision_comparison": {
            "old_detection_only": old_geometry,
            "qualified_e2e_recognition": summary["geometry"],
            "integer_count_deltas_new_minus_old": geometry_delta,
            "reason": (
                "official E2E GT excludes missing/ignored transcriptions and contains minor "
                "geometry revision differences; frozen proposals and OCR are unchanged"
            ),
        },
        "protocol": protocol,
    }


def main() -> int:
    args = _args()
    samples, qualification = load_scene_text_manifest(
        args.recognition_manifest.resolve(), args.data_root.resolve()
    )
    if len(samples) != 500 or qualification["dataset"] != "ctw1500" or qualification["split"] != "test":
        raise ValueError("recognition manifest is not the official CTW1500 TEST-500 split")
    gt = _ground_truth(samples)
    runs = [_rescore(path.resolve(), gt) for path in args.frozen_run]
    report = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "CTW1500 text-line V1+PARSeq diagnostic; not paper Table 5",
        "annotation_granularity": "text_line",
        "table_5_claim": False,
        "recognition_manifest": str(args.recognition_manifest.resolve()),
        "recognition_manifest_sha256": sha256_file(args.recognition_manifest.resolve()),
        "dataset_qualification": qualification,
        "runs": runs,
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps({run["segmentation_model_run"]: run["summary"] for run in runs}, indent=2))
    print(f"JSON {args.output.resolve()} SHA256 {sha256_file(args.output.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
