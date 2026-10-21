"""TRAIN word-identity comparison and checkpoint reuse guards; CPU only."""

import torch
import pytest

from scripts.evaluate_totaltext_letterbox_500_words_train import (
    _changes, _config, _verify_identical_weights,
)


def test_word_audit_is_frozen_to_train_and_pad_zero():
    config = _config()
    assert config["split"] == "TRAIN_heldout_125"
    assert config["gt_words"] == 983
    assert config["probability_threshold"] == 0.5
    assert config["padding_native_pixels"] == 0
    assert config["area_filter"] == "disabled"
    assert config["maximum_proposals"] == "disabled"


def test_per_word_change_audit_distinguishes_recovery_from_damage():
    gt = [{"bbox": [0, 0, 10, 10], "text": "ONE"},
          {"bbox": [20, 0, 30, 10], "text": "TWO"}]
    source = [{"sample_id": "total_text:1", "ground_truth": gt,
               "predictions": [{"bbox": gt[0]["bbox"], "text": "ONE"},
                               {"bbox": gt[1]["bbox"], "text": "TWOX"}],
               "matched_gt_indices": {"exact_casefold_iou_ge_50": [0]}}]
    control = [{"sample_id": "total_text:1", "ground_truth": gt,
                "predictions": [{"bbox": gt[0]["bbox"], "text": "ONEX"},
                                {"bbox": gt[1]["bbox"], "text": "TWO"}],
                "matched_gt_indices": {"exact_casefold_iou_ge_50": [1]}}]
    result = _changes(source, control, "exact_casefold_iou_ge_50")
    assert (result["gained"], result["lost"], result["net"]) == (1, 1, 0)
    assert result["gained_words"][0]["gt"]["text"] == "TWO"
    assert result["gained_words"][0]["control_best_spatial"]["text"] == "TWO"
    assert result["lost_words"][0]["source_best_spatial"]["text"] == "ONE"


def test_reusing_sep5_ocr_requires_identical_model_tensors(tmp_path):
    source = tmp_path / "source.pt"
    sep5 = tmp_path / "sep5.pt"
    torch.save({"epoch": 71, "model_state_dict": {"w": torch.tensor([1.0, 2.0])}}, source)
    torch.save({"epoch": 71, "model_state_dict": {"w": torch.tensor([1.0, 2.0])}}, sep5)
    assert _verify_identical_weights(source, sep5) == 1
    torch.save({"epoch": 71, "model_state_dict": {"w": torch.tensor([1.0, 3.0])}}, sep5)
    with pytest.raises(ValueError, match="not identical"):
        _verify_identical_weights(source, sep5)
