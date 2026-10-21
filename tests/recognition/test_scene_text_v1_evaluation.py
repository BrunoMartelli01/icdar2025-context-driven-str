from __future__ import annotations

import pytest

from recognition.scene_text_v1_evaluation import box_iou, evaluate_images, maximum_matching
from scripts.evaluate_scene_text_v1_parseq import _ground_truth


def test_box_iou_uses_exclusive_coordinates() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_maximum_matching_does_not_reuse_a_prediction_or_gt() -> None:
    matched = maximum_matching(2, 2, [(0, 0), (0, 1), (1, 0)])
    assert matched == {0, 1}


def test_word_evaluation_requires_geometry_and_text() -> None:
    report = evaluate_images(
        [
            {
                "ground_truth": [
                    {"bbox": [0, 0, 10, 10], "text": "SHOP"},
                    {"bbox": [20, 0, 30, 10], "text": "CAFE"},
                ],
                "predictions": [
                    {"rank": 1, "bbox": [0, 0, 10, 10], "text": "shop"},
                    {"rank": 2, "bbox": [20, 0, 30, 10], "text": "CAF"},
                ],
            }
        ],
        text_available=True,
    )
    assert report["geometry"]["iou_ge_0_50_one_to_one"] == 2
    assert report["recognition"]["exact_case_sensitive_spatial_one_to_one"] == 0
    assert report["recognition"]["exact_casefold_spatial_one_to_one"] == 1
    assert report["recognition"]["rapidfuzz_casefold_ge_0_8_spatial_one_to_one"] == 2
    assert report["recognition"]["image_level_lexical_exact_multiset_matches"] == 1


def test_detection_only_evaluation_never_claims_recognition() -> None:
    report = evaluate_images(
        [{
            "ground_truth": [{"bbox": [0, 0, 10, 10], "text": None}],
            "predictions": [{"rank": 1, "bbox": [0, 0, 10, 10], "text": "anything"}],
        }],
        text_available=False,
    )
    assert report["geometry"]["iou_ge_0_50_one_to_one"] == 1
    assert report["recognition"] is None
    assert "detection-only" in report["recognition_unavailable_reason"]


def test_ctw_geometry_is_clipped_to_original_image() -> None:
    care, ignored = _ground_truth(
        "ctw1500",
        "ctw1500:1001",
        None,  # ignored for CTW source geometry
        100,
        50,
        {
            "1001": [
                {"bbox": [-2, 3, 105, 60], "text": None, "ignore": False,
                 "granularity": "text_line"},
                {"bbox": [0, 0, 5, 5], "text": None, "ignore": True,
                 "granularity": "text_line"},
            ]
        },
    )
    assert care == [
        {"bbox": [0, 3, 100, 50], "text": None, "ignore": False,
         "granularity": "text_line"}
    ]
    assert ignored == 1
