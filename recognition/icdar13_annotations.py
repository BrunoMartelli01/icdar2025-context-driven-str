"""Verified ICDAR13 Task 1 recognition annotations for APPROXIMATE evaluation work.

This module freezes dataset identity only.  It deliberately defines no recognition
metric, matching rule, normalization, lexicon, or Table 5 denominator because those
paper-level choices remain unresolved under E-001 and DS-002.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ARTIFACT_ID = "ART-ICDAR13-M11-APPROX-RECOGNITION-MANIFEST"
EXPECTED_SAMPLE_COUNT = 233
EXPECTED_WORD_COUNT = 1095
EXPECTED_ANNOTATION_IDENTITY = (
    "89747036f9dc09b22b3cd35887be0e6f0d694122c099f23f9a6034b8996abd41"
)


class ICDAR13RecognitionManifestError(ValueError):
    """Raised when manifest metadata or physical annotation bytes drift."""


@dataclass(frozen=True)
class ICDAR13Word:
    bbox_xyxy: tuple[int, int, int, int]
    transcription: str


@dataclass(frozen=True)
class ICDAR13RecognitionSample:
    sample_id: str
    image_path: Path
    image_sha256: str
    annotation_path: Path
    annotation_sha256: str
    image_size_wh: tuple[int, int]
    words: tuple[ICDAR13Word, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _resolve_beneath(root: Path, relative: str, *, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ICDAR13RecognitionManifestError(
            "{} escapes the data root: {}".format(label, relative)
        ) from error
    return candidate


def parse_task1_annotation(
    path: Path, *, image_size_wh: tuple[int, int]
) -> tuple[ICDAR13Word, ...]:
    """Parse the acquired five-column Task 1 CSV-like text format without normalization."""

    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ICDAR13RecognitionManifestError(
            "Could not read Task 1 annotation {}: {}".format(path, error)
        ) from error
    width, height = image_size_wh
    words: list[ICDAR13Word] = []
    for line_number, row in enumerate(
        csv.reader(text.splitlines(), skipinitialspace=True), start=1
    ):
        if len(row) != 5:
            raise ICDAR13RecognitionManifestError(
                "Malformed Task 1 annotation {} line {}: expected 5 fields, got {}".format(
                    path, line_number, len(row)
                )
            )
        try:
            x_min, y_min, x_max, y_max = (int(value.strip()) for value in row[:4])
        except ValueError as error:
            raise ICDAR13RecognitionManifestError(
                "Non-integer Task 1 coordinates in {} line {}".format(path, line_number)
            ) from error
        if not (0 <= x_min <= x_max < width and 0 <= y_min <= y_max < height):
            raise ICDAR13RecognitionManifestError(
                "Out-of-bounds Task 1 box in {} line {} for image {}x{}".format(
                    path, line_number, width, height
                )
            )
        transcription = row[4]
        if not transcription:
            raise ICDAR13RecognitionManifestError(
                "Empty Task 1 transcription in {} line {}".format(path, line_number)
            )
        words.append(
            ICDAR13Word(
                bbox_xyxy=(x_min, y_min, x_max, y_max),
                transcription=transcription,
            )
        )
    if not words:
        raise ICDAR13RecognitionManifestError(
            "Task 1 annotation contains no word records: {}".format(path)
        )
    return tuple(words)


def load_icdar13_recognition_samples(
    manifest_path: Path, data_root: Path
) -> tuple[tuple[ICDAR13RecognitionSample, ...], dict[str, Any]]:
    """Fail closed unless all 233 image/annotation pairs match the frozen manifest."""

    manifest_path = manifest_path.resolve()
    data_root = data_root.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    expected_top_level = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "artifact_id": MANIFEST_ARTIFACT_ID,
        "dataset": "ICDAR13-FST-Task1",
        "split": "official_test_233",
        "sample_order": "numeric_id_ascending",
        "expected_sample_count": EXPECTED_SAMPLE_COUNT,
        "expected_word_count": EXPECTED_WORD_COUNT,
    }
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            raise ICDAR13RecognitionManifestError(
                "Recognition manifest mismatch at {}: expected {!r}, got {!r}".format(
                    key, expected, manifest.get(key)
                )
            )
    if manifest.get("table_5_claim") is not False:
        raise ICDAR13RecognitionManifestError("Recognition manifest must forbid Table 5 claims")

    source = manifest.get("source_image_manifest", {})
    source_path = (REPOSITORY_ROOT / str(source.get("path", ""))).resolve()
    if _sha256_file(source_path) != source.get("sha256"):
        raise ICDAR13RecognitionManifestError("Source image manifest SHA256 mismatch")
    source_manifest = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_samples = source_manifest.get("samples", [])
    expected_ids = [str(index) for index in range(1, EXPECTED_SAMPLE_COUNT + 1)]
    if [str(sample.get("id")) for sample in source_samples] != expected_ids:
        raise ICDAR13RecognitionManifestError(
            "Source image manifest must contain numeric IDs 1 through 233 in order"
        )

    annotation_template = str(manifest.get("annotation_path_template", ""))
    samples: list[ICDAR13RecognitionSample] = []
    annotation_identities: list[dict[str, str]] = []
    for source_sample, sample_id in zip(source_samples, expected_ids):
        image_path = _resolve_beneath(
            data_root, str(source_sample["image"]), label="image path"
        )
        annotation_path = _resolve_beneath(
            data_root,
            annotation_template.format(id=sample_id),
            label="annotation path",
        )
        if not image_path.is_file() or image_path.is_symlink():
            raise ICDAR13RecognitionManifestError(
                "Missing or non-regular recognition image: {}".format(image_path)
            )
        if not annotation_path.is_file() or annotation_path.is_symlink():
            raise ICDAR13RecognitionManifestError(
                "Missing or non-regular Task 1 annotation: {}".format(annotation_path)
            )
        image_digest = _sha256_file(image_path)
        if image_digest != source_sample.get("image_sha256"):
            raise ICDAR13RecognitionManifestError(
                "Recognition image SHA256 mismatch for sample {}".format(sample_id)
            )
        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image.verify()
        except OSError as error:
            raise ICDAR13RecognitionManifestError(
                "Unreadable recognition image {}: {}".format(image_path, error)
            ) from error
        words = parse_task1_annotation(annotation_path, image_size_wh=image_size)
        annotation_digest = _sha256_file(annotation_path)
        annotation_identities.append(
            {"id": sample_id, "annotation_sha256": annotation_digest}
        )
        samples.append(
            ICDAR13RecognitionSample(
                sample_id=sample_id,
                image_path=image_path,
                image_sha256=image_digest,
                annotation_path=annotation_path,
                annotation_sha256=annotation_digest,
                image_size_wh=image_size,
                words=words,
            )
        )

    word_count = sum(len(sample.words) for sample in samples)
    if word_count != EXPECTED_WORD_COUNT:
        raise ICDAR13RecognitionManifestError(
            "Expected {} word records; found {}".format(EXPECTED_WORD_COUNT, word_count)
        )
    identity = hashlib.sha256(_canonical_json(annotation_identities)).hexdigest()
    expected_identity = manifest.get("annotation_identity", {}).get("sha256")
    if expected_identity != EXPECTED_ANNOTATION_IDENTITY or identity != expected_identity:
        raise ICDAR13RecognitionManifestError("Task 1 annotation identity SHA256 mismatch")

    report = {
        "schema_version": 1,
        "status": "ready_for_approximate_protocol_development",
        "artifact_id": MANIFEST_ARTIFACT_ID,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "data_root": str(data_root),
        "sample_count": len(samples),
        "word_count": word_count,
        "annotation_identity_sha256": identity,
        "table_5_claim": False,
        "evaluation_status": "UNRESOLVED",
        "blocking_ambiguity_ids": ["DS-002", "E-001", "D-002"],
    }
    return tuple(samples), report


__all__ = [
    "EXPECTED_ANNOTATION_IDENTITY",
    "EXPECTED_SAMPLE_COUNT",
    "EXPECTED_WORD_COUNT",
    "ICDAR13RecognitionManifestError",
    "ICDAR13RecognitionSample",
    "ICDAR13Word",
    "load_icdar13_recognition_samples",
    "parse_task1_annotation",
]
