from pathlib import Path

from scripts.export_scene_text_algorithm1_crops import _other_gt


def test_totaltext_posthoc_gt_clips_qualified_out_of_bounds_polygon(tmp_path: Path) -> None:
    image = tmp_path / "test" / "images" / "4.jpg"
    annotation = tmp_path / "test" / "annotations" / "4.txt"
    image.parent.mkdir(parents=True)
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        "x: [[500 511 511 500]], y: [[210 210 216 216]], "
        "ornt: [u'h'], transcriptions: [u'EDGE']\n",
        encoding="utf-8",
    )

    words = _other_gt("total_text", image, "total_text:4", (510, 336))

    assert words == [{
        "bbox": [500, 210, 510, 217],
        "text": "EDGE",
        "granularity": "word",
    }]
