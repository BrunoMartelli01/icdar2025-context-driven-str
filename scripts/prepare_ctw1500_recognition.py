#!/usr/bin/env python3
"""Qualify official CTW1500 end-to-end annotations against frozen images.

The segmentation dataset remains untouched.  This command installs only the
separate text-line transcription annotations and emits checksum-frozen
recognition manifests.  CTW1500 is an out-of-paper practical extension.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any
from zipfile import BadZipFile, ZipFile

from PIL import Image

from recognition.scene_text_datasets import CTW1500_MISSING_TRANSCRIPTION
from recognition.scene_text_manifest import (
    build_scene_text_manifest,
    load_scene_text_manifest,
    sha256_file,
)


EXPECTED = {"train": 1000, "test": 500}
GEOMETRY_IOU_MINIMUM = 0.50


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-archive", type=Path, required=True)
    parser.add_argument("--detection-archive", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    return parser.parse_args()


def _canonical_member_id(name: str) -> int | None:
    path = Path(name)
    if path.suffix.lower() != ".txt" or not path.stem.isascii() or not path.stem.isdecimal():
        return None
    return int(path.stem)


def discover_annotation_members(
    archive: ZipFile, expected: dict[str, int] = EXPECTED
) -> dict[str, dict[int, str]]:
    """Find the official numeric TXT records without trusting archive paths."""

    result: dict[str, dict[int, str]] = {"train": {}, "test": {}}
    for info in archive.infolist():
        if info.is_dir():
            continue
        identifier = _canonical_member_id(info.filename)
        if identifier is None:
            continue
        split = "train" if 1 <= identifier <= 1000 else "test" if 1001 <= identifier <= 1500 else None
        if split is None:
            continue
        if identifier in result[split]:
            raise ValueError(
                f"duplicate CTW1500 {split} annotation ID {identifier}: "
                f"{result[split][identifier]!r}, {info.filename!r}"
            )
        payload = archive.read(info)
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(f"annotation is not UTF-8: {info.filename}") from error
        rows = [
            row
            for row in csv.reader(text.splitlines(), skipinitialspace=True)
            if row and any(field.strip() for field in row)
        ]
        if not rows:
            raise ValueError(f"empty annotation: {info.filename}")
        is_e2e = (
            len(rows[0]) == 1
            and rows[0][0].strip().isascii()
            and rows[0][0].strip().isdecimal()
        )
        records = rows[1:] if is_e2e else rows
        coordinate_count = 28 if is_e2e else 32
        # Detection-only labels lack the field after the fixed geometry prefix.
        for line_number, fields in enumerate(records, start=2 if is_e2e else 1):
            if len(fields) < coordinate_count:
                raise ValueError(
                    f"detection-only/malformed annotation {info.filename}:{line_number}"
                )
            if not is_e2e and (
                len(fields) < coordinate_count + 1
                or not ",".join(fields[coordinate_count:]).strip()
            ):
                raise ValueError(
                    f"detection-only/malformed annotation {info.filename}:{line_number}"
                )
            try:
                [int(value.strip()) for value in fields[:coordinate_count]]
            except ValueError as error:
                raise ValueError(
                    f"non-numeric CTW1500 geometry {info.filename}:{line_number}"
                ) from error
        result[split][identifier] = info.filename
    for split, count in expected.items():
        if len(result[split]) != count:
            raise ValueError(
                f"official CTW1500 {split} annotation count must be {count}; "
                f"found {len(result[split])}"
            )
    return result


def inspect_annotation_anomalies(
    archive: ZipFile, members: dict[str, dict[int, str]]
) -> dict[str, Any]:
    """Record upstream E2E inconsistencies without inventing missing labels."""

    report: dict[str, Any] = {}
    for split, mapping in members.items():
        count_mismatches = []
        missing_transcriptions = []
        ignored_transcriptions = 0
        physical_instances = 0
        for identifier, member in sorted(mapping.items()):
            rows = [
                row
                for row in csv.reader(
                    archive.read(member).decode("utf-8-sig").splitlines(),
                    skipinitialspace=True,
                )
                if row and any(field.strip() for field in row)
            ]
            declared = int(rows[0][0].strip())
            records = rows[1:]
            physical_instances += len(records)
            if declared != len(records):
                count_mismatches.append(
                    {"sample_id": identifier, "declared": declared, "physical": len(records)}
                )
            for record_index, row in enumerate(records, start=1):
                transcription = ",".join(row[28:]).strip() if len(row) > 28 else ""
                if not transcription:
                    missing_transcriptions.append(
                        {"sample_id": identifier, "record_index": record_index}
                    )
                elif transcription == "###":
                    ignored_transcriptions += 1
        report[split] = {
            "physical_instances": physical_instances,
            "declared_count_mismatch_count": len(count_mismatches),
            "declared_count_mismatches": count_mismatches,
            "missing_transcription_count": len(missing_transcriptions),
            "missing_transcriptions": missing_transcriptions,
            "official_ignore_transcription_count": ignored_transcriptions,
            "missing_transcription_policy": (
                "ASSUMPTION: preserve geometry and exclude from recognition denominator; "
                "no transcription is invented"
            ),
        }
    return report


def _bbox_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _maximum_assignment(matrix: list[list[float]], threshold: float) -> dict[int, int]:
    assigned: dict[int, int] = {}

    def visit(left: int, seen: set[int]) -> bool:
        candidates = sorted(
            (right for right, value in enumerate(matrix[left]) if value >= threshold),
            key=lambda right: (-matrix[left][right], right),
        )
        for right in candidates:
            if right in seen:
                continue
            seen.add(right)
            previous = assigned.get(right)
            if previous is None or visit(previous, seen):
                assigned[right] = left
                return True
        return False

    for left in range(len(matrix)):
        visit(left, set())
    return {left: right for right, left in assigned.items()}


def _detection_records(archive: ZipFile, split: str) -> dict[str, list[dict[str, Any]]]:
    member = f"ctw1500/textdet_{split}.json"
    try:
        document = json.loads(archive.read(member))
    except KeyError as error:
        raise ValueError(f"detection mirror lacks {member}") from error
    output: dict[str, list[dict[str, Any]]] = {}
    for item in document.get("data_list", []):
        identifier = str(int(Path(str(item["img_path"])).stem))
        records = []
        for instance in item.get("instances", []):
            polygon = [float(value) for value in instance["polygon"]]
            xs, ys = polygon[0::2], polygon[1::2]
            records.append(
                {
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "ignore": bool(instance.get("ignore", False)),
                }
            )
        output[identifier] = records
    return output


def _verify_geometry(samples: tuple[Any, ...], detection: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ious: list[float] = []
    match_counts = {threshold: 0 for threshold in (0.50, 0.75, 0.90, 0.95)}
    ignore_mismatches = 0
    missing_transcription_overrides = 0
    total = 0
    for sample in samples:
        identifier = str(int(sample.sample_id))
        expected = detection.get(identifier)
        if expected is None:
            raise ValueError(f"detection mirror lacks image ID {identifier}")
        observed = [
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "ignore": bool(instance.ignore),
                "transcription": instance.transcription,
            }
            for instance in sample.instances
            for x1, y1, x2, y2 in [instance.bbox_xyxy]
        ]
        if len(observed) > len(expected):
            raise ValueError(
                f"recognition has more instances for {identifier}: recognition={len(observed)}, "
                f"detection={len(expected)}"
            )
        matrix = [[_bbox_iou(left["bbox"], right["bbox"]) for right in expected] for left in observed]
        for threshold in match_counts:
            match_counts[threshold] += len(_maximum_assignment(matrix, threshold))
        assignment = _maximum_assignment(matrix, GEOMETRY_IOU_MINIMUM)
        if len(assignment) != len(observed):
            best = [max(row, default=0.0) for row in matrix]
            raise ValueError(
                f"geometry mismatch for {identifier}: matched={len(assignment)}/{len(observed)}, "
                f"minimum_best_iou={min(best, default=0.0):.6f}"
            )
        for left, right in assignment.items():
            ious.append(matrix[left][right])
            if observed[left]["ignore"] != expected[right]["ignore"]:
                if observed[left]["transcription"] == CTW1500_MISSING_TRANSCRIPTION:
                    missing_transcription_overrides += 1
                else:
                    ignore_mismatches += 1
        total += len(observed)
    ordered = sorted(ious)
    return {
        "sample_count": len(samples),
        "instance_count": total,
        "matched_at_bbox_iou_ge_0_50": match_counts[0.50],
        "matched_at_bbox_iou_ge_0_75": match_counts[0.75],
        "matched_at_bbox_iou_ge_0_90": match_counts[0.90],
        "matched_at_bbox_iou_ge_0_95": match_counts[0.95],
        "minimum_bbox_iou": min(ious),
        "median_bbox_iou": statistics.median(ious),
        "p05_bbox_iou": ordered[int(0.05 * (len(ordered) - 1))],
        "ignore_flag_mismatches": ignore_mismatches,
        "missing_transcription_ignore_overrides": missing_transcription_overrides,
        "unmatched_detection_instances": sum(len(values) for values in detection.values()) - total,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _install_annotations(
    archive: ZipFile,
    members: dict[str, dict[int, str]],
    prepared_root: Path,
) -> dict[str, Path]:
    installed: dict[str, Path] = {}
    for split, mapping in members.items():
        target = prepared_root / "datasets" / "ctw1500" / split / "recognition_annotations"
        stage = target.with_name(target.name + f".stage-{os.getpid()}")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        for identifier, member in sorted(mapping.items()):
            (stage / f"{identifier}.txt").write_bytes(archive.read(member))
        if target.exists():
            target_files = {path.name: sha256_file(path) for path in target.glob("*.txt")}
            stage_files = {path.name: sha256_file(path) for path in stage.glob("*.txt")}
            if target_files != stage_files:
                shutil.rmtree(stage)
                raise ValueError(f"existing recognition annotations differ: {target}")
            shutil.rmtree(stage)
        else:
            stage.rename(target)
        installed[split] = target
    return installed


def main() -> int:
    args = _args()
    annotation_archive = args.annotation_archive.resolve()
    detection_archive = args.detection_archive.resolve()
    prepared_root = args.prepared_root.resolve()
    for path in (annotation_archive, detection_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not prepared_root.is_dir():
        raise FileNotFoundError(prepared_root)

    try:
        with ZipFile(annotation_archive) as annotations:
            bad = annotations.testzip()
            if bad is not None:
                raise ValueError(f"corrupt annotation archive member: {bad}")
            members = discover_annotation_members(annotations)
            upstream_anomalies = inspect_annotation_anomalies(annotations, members)
            installed = _install_annotations(annotations, members, prepared_root)
    except BadZipFile as error:
        raise ValueError(f"invalid recognition annotation ZIP: {annotation_archive}") from error

    annotation_sha = sha256_file(annotation_archive)
    detection_sha = sha256_file(detection_archive)
    reports: dict[str, Any] = {}
    with ZipFile(detection_archive) as detection_zip:
        for split in ("train", "test"):
            image_dir = prepared_root / "datasets" / "ctw1500" / split / "images"
            manifest_path = prepared_root / "manifests" / f"ctw1500_recognition_{split}.json"
            manifest = build_scene_text_manifest(
                dataset="ctw1500",
                split=split,
                data_root=prepared_root,
                image_directory=image_dir,
                annotation_directory=installed[split],
                output=manifest_path,
            )
            manifest["acquisition"] = {
                "classification": "OFFICIAL_UPSTREAM_RECOGNITION_ANNOTATIONS",
                "source_repository": "https://github.com/Yuliang-Liu/Curve-Text-Detector",
                "source_file_id": "1tHf-n86z4zFRaRNEc02EO-hZINZqkQpz",
                "source_archive": annotation_archive.name,
                "source_archive_sha256": annotation_sha,
                "geometry_reference_archive": detection_archive.name,
                "geometry_reference_archive_sha256": detection_sha,
            }
            _atomic_json(manifest_path, manifest)
            samples, identity = load_scene_text_manifest(manifest_path, prepared_root)
            geometry = _verify_geometry(samples, _detection_records(detection_zip, split))
            reports[split] = {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "identity": identity,
                "geometry_crosscheck": geometry,
                "care_instances": sum(
                    not instance.ignore for sample in samples for instance in sample.instances
                ),
                "ignored_instances": sum(
                    instance.ignore for sample in samples for instance in sample.instances
                ),
            }

    qualification = {
        "schema_version": 1,
        "status": "qualified",
        "success": True,
        "classification": "ASSUMPTION",
        "result_class": "CTW1500 recognition dataset qualification; out-of-paper extension",
        "annotation_granularity": "text_line",
        "table_5_claim": False,
        "annotation_archive_sha256": annotation_sha,
        "detection_archive_sha256": detection_sha,
        "geometry_iou_minimum": GEOMETRY_IOU_MINIMUM,
        "upstream_annotation_anomalies": upstream_anomalies,
        "splits": reports,
    }
    output = prepared_root / "qualification" / "ctw1500-recognition-v1.json"
    _atomic_json(output, qualification)
    print(json.dumps(qualification, indent=2), flush=True)
    print(f"JSON {output} SHA256 {sha256_file(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
