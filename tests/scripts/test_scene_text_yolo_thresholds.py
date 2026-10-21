import copy
import json
import math

import numpy as np
import pytest

from recognition.yolo_word_refinement import joint_score, nms
from scripts.analyze_scene_text_yolo_thresholds import (
    DEFAULT_CONFIG, line_diagnostics, score_subset, shared_control, threshold_grid, validate_config,
)
from scripts.run_crop_group_word_selection import evaluate_predictions


def row(cid, bbox, text, detector=0.9, character=0.9):
    return {"candidate_id": cid, "bbox": bbox, "raw_text": text,
        "detector_score": detector, "token_probabilities": [character] * len(text),
        "token_log_probabilities": [math.log(character)] * len(text)}


def test_frozen_grid_and_strict_mode_boundaries():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    grid = validate_config(cfg)
    assert len(grid) == 95 and grid[0] == 0.05 and grid[-1] == 0.99
    assert 0.6 in grid
    changed = copy.deepcopy(cfg)
    changed["reproduction_mode"] = "strict"
    with pytest.raises(ValueError):
        validate_config(changed)
    for change in ({"step": 0}, {"minimum": float("nan")}, {"step": 0.03}, {"maximum": 1.1}):
        altered = copy.deepcopy(cfg)
        altered["threshold_grid"].update(change)
        with pytest.raises(ValueError):
            threshold_grid(altered)


def test_descending_nms_can_be_filtered_afterward_for_all_thresholds():
    rng = np.random.default_rng(0)
    rows = [row(str(i), [i % 4, i % 3, 30 + i % 4, 20 + i % 3],
                ("Hello", "HELLO!", "world", "other")[i % 4],
                float(rng.uniform(0.05, 1)), float(rng.uniform(0.05, 1))) for i in range(40)]
    rows += [row("tie-a", [60, 0, 80, 20], "same"),
             row("tie-b", [60, 0, 80, 20], "SAME"),
             row("separate", [90, 0, 110, 20], "same")]
    survivors = nms(rows, 0.5, same_text=True, joint=True)
    thresholds = [i / 100 for i in range(5, 100)] + [joint_score(r) for r in rows]
    for threshold in thresholds:
        before = nms([r for r in rows if joint_score(r) >= threshold], 0.5, same_text=True, joint=True)
        after = [r for r in survivors if joint_score(r) >= threshold]
        assert [r["candidate_id"] for r in before] == [r["candidate_id"] for r in after]


def test_filtered_matching_preserves_one_to_one_counts_and_matching_order():
    rows = [row("duplicate", [0, 0, 10, 10], "hello"),
            row("first", [0, 0, 10, 10], "HELLO!"),
            row("second", [20, 0, 30, 10], "world")]
    gt = [{"text": "Hello", "bbox": [0, 0, 10, 10]},
          {"text": "World", "bbox": [20, 0, 30, 10]}]
    edges = [(0, 0), (1, 0), (2, 1)]
    for indices in ([0, 1, 2], [2, 1], [0, 1], [2], []):
        result = score_subset(indices, len(gt), edges, edges, ["hello", "hello", "world"])
        reference = evaluate_predictions([rows[i] for i in indices], gt, 0.5)
        assert result["correct"] == reference["correct"]
        assert result["correct_gt_indices"] == reference["correct_gt_indices"]


def test_line_diagnostic_requires_proper_tokens_and_spatial_support():
    gt = [{"text": "HARBOR POINT", "bbox": [0, 0, 100, 20]},
          {"text": "FERRY", "bbox": [0, 30, 50, 50]}]
    rows = [row("a", [0, 0, 45, 20], "Harbor"), row("b", [50, 0, 100, 20], "POINT"),
            row("noise", [150, 0, 190, 20], "POINT"),
            row("substring", [0, 0, 50, 20], "HARB"), row("c", [0, 30, 50, 50], "FERRY")]
    result = line_diagnostics(rows, gt, [1], 0.5)
    assert result["counts"]["multi_token"] == {"gt": 1, "correct": 0,
        "missed_with_fragment": 1, "missed_with_two_distinct_fragments": 1}
    assert result["counts"]["single_token"] == {"gt": 1, "correct": 1}
    assert [f["candidate_id"] for f in result["failures"][0]["fragments"]] == ["a", "b"]


def test_shared_threshold_requires_improvement_on_every_dataset():
    data = {name: {"controls": {"joint_text_nms": [
        {"threshold": 0.6, "hmean": 0.5}, {"threshold": 0.65, "hmean": 0.52},
        {"threshold": 0.7, "hmean": 0.55}]}} for name in ("a", "b", "c")}
    baseline = {name: {"hmean": 0.5} for name in data}
    data["c"]["controls"]["joint_text_nms"][-1]["hmean"] = 0.49
    assert shared_control(data, baseline)["threshold"] == 0.65
    data["c"]["controls"]["joint_text_nms"][1]["hmean"] = 0.5
    assert shared_control(data, baseline) is None


def test_pre_ocr_detector_bound_keeps_every_joint_admitted_crop():
    for detector in (0.05, 0.36, 0.4, 0.8, 1.0):
        for character in (0.05, 0.5, 0.9, 1.0):
            candidate = row("a", [0, 0, 10, 10], "hello", detector, character)
            for threshold in (0.4, 0.6, 0.8, joint_score(candidate)):
                if joint_score(candidate) >= threshold:
                    assert math.sqrt(detector) >= threshold
