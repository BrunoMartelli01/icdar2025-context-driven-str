#!/usr/bin/env python3
"""Collect resumable ICDAR13 APPROXIMATE traces without computing Table 5 accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from pipeline.config import load_config, resolved_yaml
from recognition.icdar13_annotations import load_icdar13_recognition_samples


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _summarize_result(
    result_path: Path,
    *,
    expected_image_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    value = json.loads(result_path.read_text(encoding="utf-8"))
    claims = value.get("claims", {})
    if claims.get("table_5") is not False:
        raise RuntimeError("A batch trace attempted to claim Table 5")
    if value.get("reproduction_mode") != "APPROXIMATE":
        raise RuntimeError("A batch trace is not explicitly APPROXIMATE")
    image_sha256 = value.get("image", value.get("inputs", {})).get(
        "file_sha256", value.get("image", value.get("inputs", {})).get("image_sha256")
    )
    if value.get("success") and image_sha256 != expected_image_sha256:
        raise RuntimeError("Batch result image SHA256 mismatch")
    if value.get("success"):
        config_sha256 = value.get("configuration", {}).get("resolved_yaml_sha256")
        if config_sha256 != expected_config_sha256:
            raise RuntimeError("Batch result configuration SHA256 mismatch")
    fallback = value.get("fallback", {})
    return {
        "success": bool(value.get("success")),
        "result_sha256": _sha256_file(result_path),
        "final_prediction": value.get("final_prediction"),
        "confidence": value.get("scores", {}).get("C"),
        "fallback_needed": fallback.get("needs_fallback"),
        "fallback_invoked": fallback.get("invoked"),
        "fallback_string_count": len(fallback.get("recognized_strings", [])),
        "localized_block_count": len(
            value.get("segmentation", {}).get("localized_blocks", [])
        ),
        "intermediates_recorded": bool(
            value.get("segmentation", {}).get("mask_artifact")
        ),
        "error": value.get("error"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--end-id", type=int, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--deepsolo-python", type=Path, required=True)
    parser.add_argument("--deepsolo-runtime-manifest", type=Path, required=True)
    parser.add_argument("--deepsolo-source-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (1 <= args.start_id <= args.end_id <= 233):
        raise ValueError("Expected 1 <= start-id <= end-id <= 233")
    samples, dataset_report = load_icdar13_recognition_samples(
        args.manifest, args.data_root
    )
    selected = [
        sample for sample in samples
        if args.start_id <= int(sample.sample_id) <= args.end_id
    ]
    cfg = load_config(args.config)
    config_text = resolved_yaml(cfg)
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "batch-contract.json"
    index_path = output_root / "index.json"
    contract = {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "approximate reproduction trace collection; no accuracy metric",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "dataset": dataset_report,
        "configuration": {
            "path": str(args.config.resolve()),
            "resolved_yaml_sha256": config_sha256,
        },
        "artifacts": {
            "segmentation_checkpoint": {
                "path": str(args.segmentation_checkpoint.resolve()),
                "sha256": _sha256_file(args.segmentation_checkpoint),
            },
            "parseq_source": str(args.parseq_source.resolve()),
            "parseq_source_archive": {
                "path": str(args.parseq_source_archive.resolve()),
                "sha256": _sha256_file(args.parseq_source_archive),
            },
            "parseq_checkpoint": {
                "path": str(args.parseq_checkpoint.resolve()),
                "sha256": _sha256_file(args.parseq_checkpoint),
            },
            "deepsolo_python": str(args.deepsolo_python.absolute()),
            "deepsolo_runtime_manifest": {
                "path": str(args.deepsolo_runtime_manifest.resolve()),
                "sha256": _sha256_file(args.deepsolo_runtime_manifest),
            },
            "deepsolo_source_root": str(args.deepsolo_source_root.resolve()),
        },
        "scientific_boundary": {
            "accuracy": None,
            "evaluation_status": "UNRESOLVED",
            "blocking_ambiguity_ids": ["R-003", "S-001", "S-003", "D-002", "DS-002", "E-001"],
            "prohibition": "Trace collection cannot be reported as a Table 5 result.",
        },
    }
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise RuntimeError("Existing batch contract differs; refusing mixed provenance")
    else:
        _write_json_atomic(contract, contract_path)

    existing_index: dict[str, Any] = {}
    if index_path.exists():
        stored_index = json.loads(index_path.read_text(encoding="utf-8"))
        existing_index = {
            str(record["sample_id"]): record for record in stored_index.get("samples", [])
        }

    failures_in_request = 0
    for offset, sample in enumerate(selected, start=1):
        sample_dir = output_root / "samples" / "{:04d}".format(int(sample.sample_id))
        result_path = sample_dir / "result.json"
        log_path = sample_dir / "run.log"
        if result_path.exists():
            summary = _summarize_result(
                result_path,
                expected_image_sha256=sample.image_sha256,
                expected_config_sha256=config_sha256,
            )
            print(
                "[{}/{}] sample {} already recorded; success={}".format(
                    offset, len(selected), sample.sample_id, summary["success"]
                ),
                flush=True,
            )
        else:
            if sample_dir.exists() and any(sample_dir.iterdir()):
                archive_root = output_root / "incomplete-attempts"
                archive_root.mkdir(parents=True, exist_ok=True)
                attempt = 1
                while True:
                    archived = archive_root / "{:04d}-attempt-{:03d}".format(
                        int(sample.sample_id), attempt
                    )
                    if not archived.exists():
                        break
                    attempt += 1
                os.replace(sample_dir, archived)
                print(
                    "Archived incomplete sample {} attempt at {}".format(
                        sample.sample_id, archived
                    ),
                    flush=True,
                )
            sample_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m", "scripts.run_table5_smoke",
                "--config", str(args.config.resolve()),
                "--image", str(sample.image_path),
                "--output", str(result_path),
                "--segmentation-checkpoint", str(args.segmentation_checkpoint.resolve()),
                "--parseq-source", str(args.parseq_source.resolve()),
                "--parseq-source-archive", str(args.parseq_source_archive.resolve()),
                "--parseq-checkpoint", str(args.parseq_checkpoint.resolve()),
                "--deepsolo-python", str(args.deepsolo_python.absolute()),
                "--deepsolo-runtime-manifest", str(args.deepsolo_runtime_manifest.resolve()),
                "--deepsolo-source-root", str(args.deepsolo_source_root.resolve()),
            ]
            print(
                "[{}/{}] sample {} starting".format(
                    offset, len(selected), sample.sample_id
                ),
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    cwd=REPOSITORY_ROOT,
                )
            if not result_path.is_file():
                summary = {
                    "success": False,
                    "result_sha256": None,
                    "error": {
                        "type": "MissingResultError",
                        "message": "Child process exited {} without result.json".format(
                            completed.returncode
                        ),
                    },
                }
            else:
                summary = _summarize_result(
                    result_path,
                    expected_image_sha256=sample.image_sha256,
                    expected_config_sha256=config_sha256,
                )
                if bool(completed.returncode == 0) != bool(summary["success"]):
                    raise RuntimeError(
                        "Child exit/result success mismatch for sample {}".format(
                            sample.sample_id
                        )
                    )
            print(
                "[{}/{}] sample {} finished; success={}".format(
                    offset, len(selected), sample.sample_id, summary["success"]
                ),
                flush=True,
            )

        record = {
            "sample_id": sample.sample_id,
            "image_sha256": sample.image_sha256,
            "annotation_sha256": sample.annotation_sha256,
            "ground_truth_word_count": len(sample.words),
            "result": str(result_path),
            "log": str(log_path),
            **summary,
        }
        existing_index[sample.sample_id] = record
        ordered = [existing_index[key] for key in sorted(existing_index, key=int)]
        index = {
            "schema_version": 1,
            "reproduction_mode": "APPROXIMATE",
            "result_class": "approximate reproduction trace collection; no accuracy metric",
            "claims": {"table_5": False, "table_6": False, "table_7": False},
            "batch_contract": {
                "path": str(contract_path),
                "sha256": _sha256_file(contract_path),
            },
            "requested_range": [args.start_id, args.end_id],
            "recorded_sample_count": len(ordered),
            "successful_sample_count": sum(bool(item.get("success")) for item in ordered),
            "failed_sample_count": sum(not bool(item.get("success")) for item in ordered),
            "fallback_invoked_count": sum(item.get("fallback_invoked") is True for item in ordered),
            "no_fallback_count": sum(item.get("fallback_invoked") is False for item in ordered),
            "accuracy": None,
            "evaluation_status": "UNRESOLVED",
            "samples": ordered,
        }
        _write_json_atomic(index, index_path)
        if not summary.get("success"):
            failures_in_request += 1

    print(index_path)
    return 1 if failures_in_request else 0


if __name__ == "__main__":
    raise SystemExit(main())
