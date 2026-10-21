import pytest

from scripts.diagnose_adaspot_phase0 import (confidence_order, mixed_predictions,
    paired_changes, routing, summarize, write_json)


def prediction(text, bbox=(0, 0, 10, 10), confidence=.5, digest="a"):
    return {"text": text, "bbox": list(bbox), "minimum_token_probability": confidence,
        "crop_rgb_sha256": digest}


def sample(predictions):
    return {"sample_id": "s", "ground_truth": [{"text": "Word!", "bbox": [0, 0, 10, 10]}],
        "predictions": predictions}


def test_duplicates_and_missing_predictions_keep_denominator():
    result = summarize([sample([prediction("word!"), prediction("WORD!")])])
    assert result["aggregate"]["exact_casefold_iou_ge_50"] == 1
    assert result["aggregate"]["proposals"] == 2
    assert result["aggregate"]["precision"] == .5
    missing = summarize([sample([])])["aggregate"]
    assert missing["gt_words"] == 1 and missing["recall"] == 0


def test_punctuation_and_geometry_are_both_required():
    for p in (prediction("Word"), prediction("Word!", bbox=(20, 20, 30, 30))):
        assert summarize([sample([p])])["aggregate"]["exact_casefold_iou_ge_50"] == 0


def test_routing_rejects_different_pixels_and_coordinate_changes():
    weak = [prediction("x")]
    for strong in ([prediction("Word!", digest="b")], [prediction("Word!", bbox=(1, 1, 11, 11))], []):
        with pytest.raises(ValueError): mixed_predictions(weak, strong, {0})


def test_empty_first_ties_stable_and_probability_validation():
    assert confidence_order([prediction("a"), prediction("b"), prediction("", confidence=None)]) == [2, 0, 1]
    with pytest.raises(ValueError): confidence_order([prediction("a", confidence=float("nan"))])


def test_paired_recoveries_and_regressions():
    weak = summarize([sample([prediction("wrong")])])
    strong = summarize([sample([prediction("word!")])])
    assert paired_changes(weak, strong)["recovered_ids"] == [["s", 0]]
    assert paired_changes(strong, weak)["damaged"] == 1
    gate = routing(weak, strong, {"budgets": [0, 1], "seed": 42, "random_repeats": 2})
    assert gate["rows"][0]["confidence"]["exact_casefold_iou_ge_50"] == 0
    assert gate["rows"][1]["confidence"]["exact_casefold_iou_ge_50"] == 1
    assert gate["rows"][1]["random_exact_counts"] == [1, 1]


def test_immutable_protocol_cannot_be_rebound(tmp_path):
    p = tmp_path / "protocol.json"
    write_json(p, {"mode": "approximate"}); write_json(p, {"mode": "approximate"})
    with pytest.raises(ValueError): write_json(p, {"mode": "strict"})
