from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts.prepare_ctw1500_recognition import (
    _maximum_assignment,
    _verify_geometry,
    discover_annotation_members,
)


def _record(text: str = "word") -> str:
    coordinates = [0, 0, 12, 4] + [value for point in range(14) for value in (point, point % 2)]
    return ",".join(str(value) for value in coordinates) + f",{text}\n"


def test_discovers_train_and_test_recognition_annotations(tmp_path: Path) -> None:
    archive_path = tmp_path / "annotations.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("ctw/train/text_label_curve/0000001.txt", _record("train one"))
        archive.writestr("ctw/train/text_label_curve/0000002.txt", _record("train two"))
        archive.writestr("ctw/test/text_label_curve/0001001.txt", _record("test"))
        archive.writestr("ctw/test/test.txt", "1001.jpg\n")
    with ZipFile(archive_path) as archive:
        members = discover_annotation_members(archive, {"train": 2, "test": 1})
    assert sorted(members["train"]) == [1, 2]
    assert sorted(members["test"]) == [1001]


def test_discovers_official_e2e_count_header_format(tmp_path: Path) -> None:
    archive_path = tmp_path / "annotations.zip"
    absolute = [value for point in range(14) for value in (point, point % 2)]
    record = ",".join(str(value) for value in absolute) + ',"MEXICO, D. F."\n'
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("ctw1500_e2e_train/1.txt", "1\n" + record)
        archive.writestr("ctw1500_e2e_test/1001.txt", "1\n" + record)
    with ZipFile(archive_path) as archive:
        members = discover_annotation_members(archive, {"train": 1, "test": 1})
    assert members["train"][1].endswith("/1.txt")
    assert members["test"][1001].endswith("/1001.txt")


def test_rejects_detection_only_annotation(tmp_path: Path) -> None:
    archive_path = tmp_path / "annotations.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("0000001.txt", ",".join("0" for _ in range(32)) + "\n")
    with ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="detection-only"):
            discover_annotation_members(archive, {"train": 1, "test": 0})


def test_geometry_assignment_is_maximum_and_deterministic() -> None:
    matrix = [[0.95, 0.91], [0.94, 0.10]]
    assert _maximum_assignment(matrix, 0.90) == {0: 1, 1: 0}


def test_geometry_crosscheck_reports_ignore_policy_differences() -> None:
    instances = (
        SimpleNamespace(bbox_xyxy=(1, 2, 11, 6), ignore=False, transcription="one"),
        SimpleNamespace(bbox_xyxy=(20, 2, 30, 6), ignore=True, transcription="###"),
    )
    sample = SimpleNamespace(sample_id="1001", instances=instances)
    detection = {
        "1001": [
            {"bbox": [1, 2, 11, 6], "ignore": False},
            {"bbox": [20, 2, 30, 6], "ignore": True},
        ]
    }
    report = _verify_geometry((sample,), detection)
    assert report["matched_at_bbox_iou_ge_0_90"] == 2
    detection["1001"][1]["ignore"] = False
    report = _verify_geometry((sample,), detection)
    assert report["ignore_flag_mismatches"] == 1
