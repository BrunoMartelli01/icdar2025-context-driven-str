"""Meaningful boundaries for validation-selected DeepSolo filtering."""

import pytest

from recognition.deepsolo_icdar2015_gap import evaluate_scenes, paired_interval, select_policy


def prediction(index, text, score, bbox=None, ignored=False, invalid=False):
    return dict(instance_id=index, text=text, score=score, recognition_confidence=0.95,
                bbox=bbox or [0, 0, 10, 10], ignored=ignored,
                invalid_envelope={} if invalid else None)


def test_calibration_uses_micro_counts_and_keeps_boundary():
    scenes = {"a": ([prediction(0, "A!", 0.3), prediction(1, "bad", 0.2)],
                     [dict(text="a", bbox=[0, 0, 10, 10])])}
    chosen, grid = select_policy(scenes, [0.2, 0.3, 0.4], [0.0])
    assert chosen["policy"]["score_threshold"] == 0.3
    assert chosen["metrics"]["f1"] == 1.0
    assert len(grid) == 3


def test_symbol_only_invalid_and_ignored_do_not_disappear():
    rows = [prediction(0, "A!", 0.3), prediction(1, "!!!", 0.9),
            prediction(2, "a", 0.9, invalid=True), prediction(3, "noise", 0.9, ignored=True)]
    scenes = {"a": (rows, [dict(text="a", bbox=[0, 0, 10, 10]),
                            dict(text="!!!", bbox=[10, 10, 20, 20])])}
    policy = select_policy(scenes, [0.3], [0.0])[0]["policy"]
    result = evaluate_scenes(scenes, policy, iou_threshold=0.5, height_breaks=[12, 24])
    assert (result["metrics"]["tp"], result["metrics"]["fp"], result["metrics"]["fn"]) == (1, 2, 1)
    assert result["metrics"]["ignored_outputs"] == 1
    assert result["metrics"]["deepsolo_images"] == 1
    assert result["spatial_diagnostic_counts"]["predictions"] == 3


def test_empty_predictions_preserve_all_images_and_targets():
    scenes = {"a": ([], [dict(text="x", bbox=[0, 0, 10, 10])])}
    policy = select_policy(scenes, [0.3], [0.0])[0]["policy"]
    result = evaluate_scenes(scenes, policy, iou_threshold=0.5, height_breaks=[12, 24])
    assert result["metrics"]["f1"] == 0.0
    assert result["original_word_height"]["small"]["gt"] == 1
    assert result["metrics"]["total_images"] == 1


def test_paired_uncertainty_rejects_unpaired_and_replays_identity():
    rows = {"a": {"counts": dict(tp=2, detected_words=3, gt_words=4)}}
    assert paired_interval(rows, rows, repetitions=20, seed=1)["gain95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        paired_interval(rows, {}, repetitions=20, seed=1)
