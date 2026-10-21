"""Explicit class-balanced spatial Gap supervision, approximate experiment."""

import torch
from torch.nn import functional as F


def decoder_loss(logits, gap_target, text_target, validity, teacher_text, config):
    if not (logits.shape == gap_target.shape == text_target.shape == validity.shape == teacher_text.shape):
        raise ValueError("Gap, Text, validity and teacher must have matching shapes")
    if (not torch.isfinite(logits).all() or not torch.isfinite(teacher_text).all()
            or any(torch.any((x != 0) & (x != 1)) for x in (gap_target, text_target, validity))
            or torch.any((gap_target > 0) & ((text_target > 0) | (validity == 0)))):
        raise ValueError("Invalid Gap supervision or nonfinite predictions")
    valid, positive = validity.bool(), gap_target.bool()
    foreground = teacher_text.detach() >= config["teacher_text_threshold"]
    strata = (valid & positive & foreground, valid & positive & ~foreground,
              valid & ~positive & foreground, valid & ~positive & ~foreground)
    bce = F.binary_cross_entropy_with_logits(logits, gap_target, reduction="none")
    # Missing strata contribute zero; fixed denominator4 keeps the objective
    # defined for sparse/empty-Gap batches without silently changing weights.
    terms = [bce[s].mean() if s.any() else logits.sum() * 0 for s in strata]
    balanced = sum(terms) / 4
    probability = logits.sigmoid() * validity
    dice = 1 - (2 * (probability * gap_target).sum() + config["smooth"]) / (
        probability.sum() + gap_target.sum() + config["smooth"])
    text = valid & text_target.bool()
    protection = bce[text].mean() if text.any() else logits.sum() * 0
    if not valid.any():
        dice = logits.sum() * 0
    total = config["balanced_bce_weight"] * balanced + config["dice_weight"] * dice + config["text_protection_weight"] * protection
    return total, {"balanced_bce": balanced, "dice": dice, "text_protection": protection}
