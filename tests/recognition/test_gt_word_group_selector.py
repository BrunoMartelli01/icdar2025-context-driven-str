import json
from pathlib import Path

import pytest

from diagnostics.gt_word_group_selector import build_gt_groups, load_config


def test_gt_oracle_makes_one_group_per_word_and_shares_multiword_crop():
    rows = [{"candidate_id": "a", "bbox": [0, 0, 10, 10]},
            {"candidate_id": "b", "bbox": [1, 0, 11, 10]},
            {"candidate_id": "other", "bbox": [30, 0, 40, 10]}]
    gt = [{"text": "first", "bbox": [0, 0, 10, 10]},
          {"text": "second", "bbox": [1, 0, 11, 10]},
          {"text": "missing", "bbox": [60, 0, 70, 10]}]
    groups = build_gt_groups(rows, gt, iou_min=.5)
    assert len(groups) == len(gt) == 3
    assert groups[0]["candidate_ids"] == ["a", "b"]
    assert groups[1]["candidate_ids"] == ["a", "b"]
    assert groups[2]["candidate_ids"] == []
    assert groups[2]["reference_bbox"] is None
    # The GT transcription is not used in membership or medoid selection.
    changed = json.loads(json.dumps(gt))
    changed[0]["text"] = "arbitrary"
    assert build_gt_groups(rows, changed, iou_min=.5) == groups


def test_gt_oracle_median_reference_comes_from_candidate_and_is_order_stable():
    rows = [{"candidate_id": "wide", "bbox": [0, 0, 10, 10]},
            {"candidate_id": "median", "bbox": [1, 0, 11, 10]},
            {"candidate_id": "narrow", "bbox": [2, 0, 12, 10]}]
    gt = [{"text": "word", "bbox": [1, 0, 11, 10]}]
    a = build_gt_groups(rows, gt, iou_min=.5)
    b = build_gt_groups(list(reversed(rows)), gt, iou_min=.5)
    assert a == b
    assert a[0]["reference_candidate_id"] == "median"
    assert a[0]["reference_bbox"] == [1, 0, 11, 10]


def test_gt_oracle_config_is_explicitly_approximate():
    path = Path(__file__).resolve().parents[2] / "configs/approximate/gt_word_group_selector_v1.json"
    assert load_config(path)["result_class"] == "ORACLE_TEST_DIAGNOSTIC"
    with pytest.raises(ValueError, match="configs/approximate"):
        load_config(Path(__file__))
