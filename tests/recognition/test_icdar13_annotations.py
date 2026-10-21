from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recognition.icdar13_annotations import (
    ICDAR13RecognitionManifestError,
    _canonical_json,
    parse_task1_annotation,
)


def test_task1_parser_preserves_case_punctuation_and_quoted_comma(tmp_path: Path) -> None:
    annotation = tmp_path / "gt.txt"
    annotation.write_bytes(
        b'\xef\xbb\xbf1, 2, 8, 9, "Tiredness"\r\n'
        b'0, 0, 4, 5, "HELLO, World!"\r\n'
    )
    words = parse_task1_annotation(annotation, image_size_wh=(10, 12))
    assert [word.bbox_xyxy for word in words] == [(1, 2, 8, 9), (0, 0, 4, 5)]
    assert [word.transcription for word in words] == ["Tiredness", "HELLO, World!"]


@pytest.mark.parametrize(
    "row,error_fragment",
    [
        ('1,2,3,"missing field"\n', "expected 5 fields"),
        ('a,2,3,4,"text"\n', "Non-integer"),
        ('1,2,30,4,"text"\n', "Out-of-bounds"),
        ('1,2,3,4,""\n', "Empty"),
    ],
)
def test_task1_parser_fails_closed(row: str, error_fragment: str, tmp_path: Path) -> None:
    annotation = tmp_path / "gt.txt"
    annotation.write_text(row, encoding="utf-8")
    with pytest.raises(ICDAR13RecognitionManifestError, match=error_fragment):
        parse_task1_annotation(annotation, image_size_wh=(10, 12))


def test_annotation_identity_has_an_explicit_canonical_serialization() -> None:
    records = [
        {"id": "1", "annotation_sha256": "a" * 64},
        {"id": "2", "annotation_sha256": "b" * 64},
    ]
    expected_bytes = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert _canonical_json(records) == expected_bytes
    assert hashlib.sha256(_canonical_json(records)).hexdigest() == hashlib.sha256(
        expected_bytes
    ).hexdigest()
