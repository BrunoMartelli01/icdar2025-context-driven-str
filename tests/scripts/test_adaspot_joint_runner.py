import json

import pytest
import torch

from scripts import run_adaspot_joint as runner
from recognition.adaspot_gate import prediction_correctness


def test_hash_partition_is_reproducible_and_manifest_order_independent():
    ids = [f"total_text:{i}" for i in range(20)]
    a = runner.partition(ids, seed=202503, calibration_images=4)
    b = runner.partition(ids[::-1], seed=202503, calibration_images=4)
    assert a == b and list(map(len, a)) == [16, 2, 2]
    assert set(sum((list(v) for v in a), [])) == set(ids)
    assert not any(set(a[i]) & set(a[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
    with pytest.raises(ValueError):
        runner.partition(ids + ids[:1], seed=202503, calibration_images=4)


def test_correctness_matching_does_not_mark_both_duplicate_predictions_true():
    sample = {"ground_truth": [{"text": "Hi!", "bbox": [0, 0, 4, 4]}],
        "predictions": [{"bbox": [0, 0, 4, 4], "light": {"text": "hi!"}} for _ in range(2)]}
    assert sum(prediction_correctness(sample, reading_key="light")) == 1
    sample["predictions"][0]["light"]["text"] = "hi"
    assert prediction_correctness(sample, reading_key="light") == [0, 1]


def test_train_only_calibration_recovers_deep_gain_at_one_call():
    row = {"sample_id": "train:synthetic", "ground_truth": [
        {"text": "easy", "bbox": [0, 0, 5, 5]}, {"text": "hard", "bbox": [10, 0, 15, 5]}],
        "predictions": [
            {"bbox": [0, 0, 5, 5], "instance_id": 1, "gate_logit": 3.,
             "light": {"text": "easy", "minimum_probability": .9}, "deep": {"text": "easy", "minimum_probability": .9}},
            {"bbox": [10, 0, 15, 5], "instance_id": 2, "gate_logit": -3.,
             "light": {"text": "wrong", "minimum_probability": .2}, "deep": {"text": "hard", "minimum_probability": .9}}],
        "light_correctness_labels": [1, 0]}
    calibrated = runner.calibrate([row], [row], {"temperature_log_bounds": [-3., 3.],
        "routing_thresholds": [0., .5, 1.], "calibration_bins": 5})
    assert calibrated["selection"]["exact_words"] == 2
    assert calibrated["selection"]["hmean"] == 1.
    assert calibrated["selection"]["deep_calls"] == 1


def test_empty_reading_is_explicitly_not_emitted_but_gt_stays():
    row = {"sample_id": "test:empty", "ground_truth": [{"text": "word", "bbox": [0, 0, 3, 3]}],
        "predictions": [{"instance_id": 1, "bbox": [0, 0, 3, 3], "gate_logit": 0.,
            "light": {"text": "", "minimum_probability": 0.}, "deep": {"text": "", "minimum_probability": 0.}}]}
    light, _ = runner.score_arm([row], "light")
    adaptive, routes = runner.score_arm([row], "adaptive", {"temperature": 1., "reject_threshold": 0.,
        "keep_threshold": 0., "deep_threshold": 0.})
    assert light["aggregate"]["proposals"] == 1 and adaptive["aggregate"]["proposals"] == 0
    assert light["aggregate"]["gt_words"] == adaptive["aggregate"]["gt_words"] == 1
    assert not routes[0][0]["emitted"]


def test_oom_is_reported_and_raised_without_fallback(tmp_path, monkeypatch):
    def fail(args):
        args._output_created = True
        raise torch.cuda.OutOfMemoryError("controlled unit-test OOM")
    monkeypatch.setattr(runner, "run", fail)
    monkeypatch.setattr("sys.argv", ["run", "--mode", "approximate", "--stage", "smoke", "--config", "test.json",
        "--source-run", str(tmp_path), "--output-root", str(tmp_path), "--device", "cuda:0"])
    with pytest.raises(torch.cuda.OutOfMemoryError):
        runner.main()
    receipt = json.loads((tmp_path / "failure.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["oom"]
    assert not (tmp_path / "results.json").exists()
