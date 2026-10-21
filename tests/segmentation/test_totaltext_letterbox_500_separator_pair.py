"""Two frozen letterbox continuation branches; CPU-only unit checks."""

import torch

from scripts.train_totaltext_letterbox_500_separator_pair import (
    BRANCHES, _config, _contract,
)
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config
from segmentation.training import separation_weighted_bce_with_logits


def test_same_epoch71_fork_and_500_epoch_target_for_both_branches():
    pair = _config()
    assert pair["fork_checkpoint_epoch"] == 71
    assert pair["source_completed_epochs"] == 100
    assert pair["total_epochs_including_fork_prefix"] == 500
    assert BRANCHES == {"letterbox_500_control": 3.0, "letterbox_500_sep5": 5.0}
    assert source_config()["training"]["separation_weight"] == 3.0


def test_branch_contracts_share_source_and_split_but_not_loss_weight():
    pair, source = _config(), source_config()
    control = _contract(pair, source, "letterbox_500_control", "architecture", "validation", [0, 1])
    strong = _contract(pair, source, "letterbox_500_sep5", "architecture", "validation", [0, 1])
    assert control["source_checkpoint_sha256"] == strong["source_checkpoint_sha256"]
    assert control["validation_ids_sha256"] == strong["validation_ids_sha256"]
    assert control["train_indices_sha256"] == strong["train_indices_sha256"]
    assert control["letterbox"] == strong["letterbox"]
    assert control["separation_weight"] == 3.0
    assert strong["separation_weight"] == 5.0


def test_stronger_penalty_increases_interword_false_positive_gradient():
    target = torch.zeros(1, 1, 1, 2)
    valid = torch.ones_like(target)
    separator = torch.tensor([[[[1.0, 0.0]]]])
    gradients = []
    for weight in (3.0, 5.0):
        logits = torch.zeros_like(target, requires_grad=True)
        loss = separation_weighted_bce_with_logits(
            logits, target, valid, separator, separation_weight=weight)
        loss.backward()
        gradients.append(logits.grad.detach().clone())
    assert gradients[1][0, 0, 0, 0] > gradients[0][0, 0, 0, 0]
    assert gradients[1][0, 0, 0, 1] < gradients[0][0, 0, 0, 1]
