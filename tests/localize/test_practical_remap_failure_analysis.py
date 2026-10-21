from __future__ import annotations

import pytest

from localize.practical_remap_failure_analysis import _pair_metrics, classify_failure


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"foreground_recall": 0.09, "pre_cap_has_valid_group": False, "oracle_union_iou": 0.9,
          "associated_group_count": 3, "best_group_gt_word_count": 1}, "NO_FOREGROUND"),
        ({"foreground_recall": 0.8, "pre_cap_has_valid_group": True, "oracle_union_iou": 0.9,
          "associated_group_count": 3, "best_group_gt_word_count": 1}, "TOP10_TRUNCATION"),
        ({"foreground_recall": 0.8, "pre_cap_has_valid_group": False, "oracle_union_iou": 0.5,
          "associated_group_count": 2, "best_group_gt_word_count": 1}, "GROUPING_UNDERMERGE"),
        ({"foreground_recall": 0.8, "pre_cap_has_valid_group": False, "oracle_union_iou": 0.5,
          "associated_group_count": 1, "best_group_gt_word_count": 2}, "GROUPING_OVERMERGE"),
        ({"foreground_recall": 0.8, "pre_cap_has_valid_group": False, "oracle_union_iou": 0.49,
          "associated_group_count": 1, "best_group_gt_word_count": 1}, "GEOMETRY_LIMIT"),
        ({"foreground_recall": 0.8, "pre_cap_has_valid_group": False, "oracle_union_iou": 0.5,
          "associated_group_count": 1, "best_group_gt_word_count": 1}, "OTHER"),
    ],
)
def test_failure_categories_have_a_single_declared_priority(
    kwargs: dict[str, float | bool | int], expected: str
) -> None:
    assert classify_failure(best_retained_group_iou=0.0, **kwargs) == expected  # type: ignore[arg-type]


def test_pair_metric_mapping_uses_exclusive_xyxy_boxes() -> None:
    metrics = _pair_metrics((0, 0, 10, 10), (14, 1, 24, 11))
    assert metrics["horizontal_gap_over_reference_height"] == pytest.approx(0.4)
    assert metrics["vertical_overlap_over_min_height"] == pytest.approx(0.9)
    assert metrics["center_y_distance_over_reference_height"] == pytest.approx(0.1)
    assert metrics["height_ratio"] == pytest.approx(1.0)
