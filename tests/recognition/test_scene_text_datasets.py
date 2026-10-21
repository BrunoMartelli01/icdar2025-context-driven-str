from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from recognition.scene_text_datasets import (
    SceneTextAnnotationError,
    parse_ctw1500_test_annotation,
    parse_ctw1500_train_xml,
    parse_icdar2015_annotation,
    parse_totaltext_annotation,
)
from recognition.scene_text_manifest import (
    SceneTextManifestError,
    _canonical_id,
    build_scene_text_manifest,
    load_scene_text_manifest,
)


def test_ctw_test_numeric_ids_pair_across_zero_padding() -> None:
    assert _canonical_id(Path("1001.jpg"), "ctw1500") == "1001"
    assert _canonical_id(Path("0001001.txt"), "ctw1500") == "1001"


def test_icdar2015_preserves_numeric_and_comma_transcriptions(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_bytes(
        b"\xef\xbb\xbf0,0,10,0,10,8,0,8,123\r\n"
        b"1,1,9,1,9,7,1,7,HELLO, WORLD\r\n"
        b"2,2,8,2,8,6,2,6,###\r\n"
    )
    instances = parse_icdar2015_annotation(path, image_size_wh=(10, 8))
    assert [item.transcription for item in instances] == ["123", "HELLO,WORLD", "###"]
    assert [item.ignore for item in instances] == [False, False, True]
    assert all(item.granularity == "word" for item in instances)


def test_totaltext_variable_polygon_and_orientation(tmp_path: Path) -> None:
    path = tmp_path / "poly_gt_img1.txt"
    path.write_text(
        "x: [[1 5 9 7 2]], y: [[1 0 3 8 7]], ornt: [u'c'], "
        "transcriptions: [u\"O'Brien\"]\n"
        "x: [[0 4 4 0]], y: [[0 0 4 4]], ornt: [u'#'], "
        "transcriptions: [u'###']\n",
        encoding="utf-8",
    )
    instances = parse_totaltext_annotation(path, image_size_wh=(10, 10))
    assert len(instances[0].polygon_xy) == 5
    assert instances[0].transcription == "O'Brien"
    assert instances[0].orientation == "c"
    assert instances[1].ignore is True


def test_totaltext_accepts_official_single_point_ignore_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "poly_gt_img1304.txt"
    path.write_text(
        "x: [[718]], y: [[303]], ornt: [u'#'], transcriptions: [u'#']\n",
        encoding="utf-8",
    )
    instance = parse_totaltext_annotation(path, image_size_wh=(1200, 800))[0]
    assert instance.ignore is True
    assert instance.polygon_xy == ((718, 303),)


def test_totaltext_accepts_official_missing_orientation(tmp_path: Path) -> None:
    path = tmp_path / "poly_gt_img1492.txt"
    path.write_text(
        "x: [[799 841 841 802]], y: [[323 313 335 343]], "
        "ornt: [], transcriptions: [u'and']\n",
        encoding="utf-8",
    )
    instance = parse_totaltext_annotation(path, image_size_wh=(1000, 600))[0]
    assert instance.orientation is None
    assert instance.ignore is False


def test_totaltext_accepts_empty_transcription_only_for_ignore(tmp_path: Path) -> None:
    path = tmp_path / "poly_gt_img1143.txt"
    path.write_text(
        "x: [[333 373 368 329]], y: [[222 215 241 246]], "
        "ornt: [u'#'], transcriptions: []\n",
        encoding="utf-8",
    )
    instance = parse_totaltext_annotation(path, image_size_wh=(1000, 600))[0]
    assert instance.ignore is True
    assert instance.transcription == "#"


def test_totaltext_accepts_official_multiline_polygon_record(tmp_path: Path) -> None:
    path = tmp_path / "poly_gt_img623.txt"
    path.write_text(
        "x: [[101 130 178 232 274 364 435 486 518 496\n"
        " 476 458 424 364 324 286 258 216 198]], "
        "y: [[201 155 110 90 88 113 158 215 254 318\n"
        " 325 287 255 200 181 177 184 220 262]], "
        "ornt: [u'c'], transcriptions: [u'HARANA']\n",
        encoding="utf-8",
    )
    instance = parse_totaltext_annotation(path, image_size_wh=(1000, 600))[0]
    assert len(instance.polygon_xy) == 19
    assert instance.transcription == "HARANA"


def test_ctw1500_test_relative_offsets_and_numeric_text(tmp_path: Path) -> None:
    path = tmp_path / "1001.txt"
    offsets = [0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0,
               12, 4, 10, 4, 8, 4, 6, 4, 4, 4, 2, 4, 0, 4]
    path.write_text(
        ",".join(str(value) for value in [5, 6, 12, 4, *offsets]) + ",2025\n",
        encoding="utf-8",
    )
    instance = parse_ctw1500_test_annotation(path, image_size_wh=(30, 20))[0]
    assert instance.polygon_xy[0] == (5, 6)
    assert instance.polygon_xy[-1] == (5, 10)
    assert instance.transcription == "2025"
    assert instance.granularity == "text_line"


def test_ctw1500_end_to_end_absolute_points_and_quoted_comma(tmp_path: Path) -> None:
    path = tmp_path / "1001.txt"
    points = [value for point in range(14) for value in (5 + point, 6 + point % 2)]
    path.write_text(
        "1\n" + ",".join(str(value) for value in points) + ',"MEXICO, D. F."\n',
        encoding="utf-8",
    )
    instance = parse_ctw1500_test_annotation(path, image_size_wh=(30, 20))[0]
    assert instance.polygon_xy[0] == (5, 6)
    assert instance.polygon_xy[-1] == (18, 7)
    assert instance.transcription == "MEXICO, D. F."


def test_ctw1500_end_to_end_missing_text_is_excluded_without_invention(tmp_path: Path) -> None:
    path = tmp_path / "1442.txt"
    points = [value for point in range(14) for value in (5 + point, 6 + point % 2)]
    path.write_text(
        "2\n" + ",".join(str(value) for value in points) + "\n",
        encoding="utf-8",
    )
    instance = parse_ctw1500_test_annotation(path, image_size_wh=(30, 20))[0]
    assert instance.transcription == "<missing-transcription>"
    assert instance.ignore is True


def test_ctw1500_xml_does_not_depend_on_child_order(tmp_path: Path) -> None:
    path = tmp_path / "0001.xml"
    path.write_text(
        "<root><box><label>curved text</label><segs>1,1,9,1,9,5,1,5</segs></box>"
        "<box><segs>2,2,8,2,8,6,2,6</segs><transcription>###</transcription></box></root>",
        encoding="utf-8",
    )
    instances = parse_ctw1500_train_xml(path, image_size_wh=(10, 10))
    assert instances[0].transcription == "curved text"
    assert instances[1].ignore is True


def test_ctw1500_train_txt_manifest_uses_end_to_end_parser(tmp_path: Path) -> None:
    root = tmp_path / "ctw"
    images = root / "images"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (30, 20), "white").save(images / "1.jpg")
    offsets = [
        0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0,
        12, 4, 10, 4, 8, 4, 6, 4, 4, 4, 2, 4, 0, 4,
    ]
    (annotations / "0000001.txt").write_text(
        ",".join(str(value) for value in [5, 6, 12, 4, *offsets]) + ",curved text\n",
        encoding="utf-8",
    )
    output = root / "manifest.json"
    report = build_scene_text_manifest(
        dataset="ctw1500",
        split="train",
        data_root=root,
        image_directory=images,
        annotation_directory=annotations,
        output=output,
        enforce_official_count=False,
    )
    assert report["parser"] == "ctw1500_relative14_txt_v1"
    samples, _ = load_scene_text_manifest(output, root)
    assert samples[0].instances[0].transcription == "curved text"


@pytest.mark.parametrize(
    "content,parser",
    [
        ("0,0,1,0,1,1,0,1\n", parse_icdar2015_annotation),
        ("x: [[1 2]], y: [[1]], ornt: [u'h'], transcriptions: [u'x']\n", parse_totaltext_annotation),
    ],
)
def test_parsers_fail_closed(content: str, parser, tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SceneTextAnnotationError):
        parser(path, image_size_wh=(10, 10))


def test_manifest_round_trip_and_checksum_guard(tmp_path: Path) -> None:
    root = tmp_path / "data"
    images = root / "train_images"
    annotations = root / "train_gt"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (12, 10), "white").save(images / "img_1.jpg")
    (annotations / "gt_img_1.txt").write_text(
        "0,0,10,0,10,8,0,8,word\n", encoding="utf-8"
    )
    output = tmp_path / "manifest.json"
    report = build_scene_text_manifest(
        dataset="icdar2015",
        split="train",
        data_root=root,
        image_directory=images,
        annotation_directory=annotations,
        output=output,
        enforce_official_count=False,
    )
    assert report["sample_count"] == 1
    samples, qualification = load_scene_text_manifest(output, root)
    assert samples[0].instances[0].transcription == "word"
    assert qualification["table_5_claim"] is False

    (annotations / "gt_img_1.txt").write_text(
        "0,0,10,0,10,8,0,8,changed\n", encoding="utf-8"
    )
    with pytest.raises(SceneTextManifestError, match="SHA256 mismatch"):
        load_scene_text_manifest(output, root)


def test_manifest_rejects_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(SceneTextManifestError, match="escapes data root"):
        build_scene_text_manifest(
            dataset="total_text",
            split="test",
            data_root=root,
            image_directory=tmp_path / "outside_images",
            annotation_directory=root / "annotations",
            output=tmp_path / "manifest.json",
            enforce_official_count=False,
        )


def test_written_manifest_is_machine_readable_and_forbids_claim(tmp_path: Path) -> None:
    root = tmp_path / "data"
    images, annotations = root / "test_images", root / "test_gt"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (20, 12), "white").save(images / "img1.jpg")
    (annotations / "poly_gt_img1.txt").write_text(
        "x: [[1 10 10 1]], y: [[1 1 8 8]], ornt: [u'h'], transcriptions: [u'TEXT']\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.json"
    build_scene_text_manifest(
        dataset="total_text",
        split="test",
        data_root=root,
        image_directory=images,
        annotation_directory=annotations,
        output=output,
        enforce_official_count=False,
    )
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["table_5_claim"] is False
    assert raw["evaluation_protocol"] == "UNRESOLVED"
