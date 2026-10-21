"""Focused CPU checks for the isolated practical word-F1/separator ablation."""

import torch
import pytest

from models.unet_adv import PracticalTotalTextSeparatorHeadAGUNet
from scripts.evaluate_totaltext_letterbox_pad0_test import _config as letterbox_config
from scripts.evaluate_totaltext_letterbox_threshold065_test import CONFIG_PATH as THRESHOLD_CONFIG
from scripts.train_totaltext_letterbox_word_f1_separator_head import (
    _config, _train_separator_epoch, balanced_separator_bce, selection_key,
)


def test_threshold_065_is_separate_from_existing_05_protocol():
    assert letterbox_config()["probability_threshold"] == 0.5
    frozen = letterbox_config(THRESHOLD_CONFIG, expected_threshold=0.65)
    assert frozen["probability_threshold"] == 0.65
    assert frozen["threshold_selection_report_sha256"]
    with pytest.raises(ValueError, match="protocol changed"):
        letterbox_config(THRESHOLD_CONFIG)


def test_two_output_head_preserves_first_channel_initialization():
    torch.manual_seed(17)
    control = PracticalTotalTextSeparatorHeadAGUNet(
        skip_floor=0.25, encoder_pretrained=False).eval()
    assert control.head.weight.shape == (2, 64, 1, 1)
    assert control.architecture_identity != "practical-totaltext-v4-highres-agunet-v1"
    assert sum(p.numel() for p in control.parameters()) == 58155146
    with torch.inference_mode():
        output = control(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 2, 16, 16)


def test_balanced_separator_loss_requires_valid_zero_target_and_gradients():
    logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
    separator = torch.tensor([[[[1., 0.], [0., 0.]]]])
    valid = torch.ones_like(separator)
    loss = balanced_separator_bce(logits, separator, valid)
    assert torch.isclose(loss, torch.tensor(0.69314718), atol=1e-6)
    loss.backward()
    assert logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[0, 0, 0, 1] > 0
    with pytest.raises(ValueError, match="outside valid"):
        balanced_separator_bce(logits.detach(), separator,
                               torch.tensor([[[[0., 1.], [1., 1.]]]]))


def test_word_f1_selection_prefers_fewer_fusions_before_more_recall():
    _config()
    baseline = {"proposal_f1_iou_50": 0.5,
                "gt_words_touched_by_fused_components": 10,
                "word_recall_iou_50": 0.7, "foreground_iou": 0.8}
    fewer_fusions = {**baseline, "gt_words_touched_by_fused_components": 9,
                     "word_recall_iou_50": 0.6}
    assert selection_key(fewer_fusions, 20) > selection_key(baseline, 10)
    assert selection_key(baseline, 10) > selection_key(baseline, 11)
    higher_f1 = {**baseline, "proposal_f1_iou_50": 0.5001,
                 "gt_words_touched_by_fused_components": 100}
    assert selection_key(higher_f1, 30) > selection_key(baseline, 10)


def test_two_head_training_step_uses_both_valid_targets_on_cpu():
    tiny = torch.nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.Adam(tiny.parameters(), lr=1e-4)
    mask = torch.zeros(2, 1, 256, 256)
    mask[:, :, 30:80, 30:80] = 1
    separator = torch.zeros_like(mask)
    separator[:, :, 80:82, 30:80] = 1
    batch = {"image": torch.rand(2, 3, 256, 256), "mask": mask,
             "separation": separator, "validity": torch.ones_like(mask),
             "sample_id": ["a", "b"]}
    before = tiny.weight.detach().clone()
    metrics = _train_separator_epoch(tiny, [batch], optimizer, torch.device("cpu"))
    assert metrics["steps"] == 1 and metrics["samples"] == 2
    assert metrics["loss"] > 0 and not torch.equal(before, tiny.weight)
