"""Approximate, isolated loss controls for the ICDAR13 word-mask refit."""
from __future__ import annotations

import math
import torch

from segmentation.icdar13_word_refit import word_loss
from segmentation.word_multitask import masked_bce_dice


def matched_word_loss(logits, batch, config, branch):
    """Keep targets identical; restore the source Total-Text BCE+Dice exactly."""
    if branch == "refit_bce":
        return word_loss(logits, batch, config)
    if branch != "totaltext_bce_dice":
        raise ValueError("Unknown explicit ICDAR13 loss branch")
    target, valid, separator, kernel = (batch[k] for k in ("mask", "validity", "separation", "kernel"))
    if logits.shape != (len(target), 2, *target.shape[-2:]):
        raise ValueError("Aligned full-word/kernel logits required")
    if torch.any(kernel > target) or torch.any(kernel > valid):
        raise ValueError("Kernel must stay inside valid full-word foreground")
    if torch.any((separator != 0) & (separator != 1)) or torch.any((separator > 0) & ((target > 0) | (valid == 0))):
        raise ValueError("Separator must be valid non-text background")
    options = config["source_loss"]
    text = masked_bce_dice(logits[:, :1], target, valid,
        pixel_weight=1 + (config["separation_weight"] - 1) * separator, **options)
    core = masked_bce_dice(logits[:, 1:2], kernel, valid, **options)
    return text + config["kernel_loss_weight"] * core


def emit_words(rows, minimum_probability):
    """Explicit no-word rejection, before evaluation; preserve raw rows separately."""
    if not math.isfinite(minimum_probability) or not 0 <= minimum_probability <= 1:
        raise ValueError("Finite sequence-probability cutoff in [0,1] required")
    result = []
    for row in rows:
        probability = row["sequence_confidence_native"]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid native PARSeq sequence confidence")
        if probability >= minimum_probability and any(c.isalnum() for c in row["raw_text"].casefold()):
            result.append(row)
    return result


def choose_selection(rows):
    """Stable predeclared tie order; this helper receives selection rows only."""
    if not rows or any(r["split"] != "selection" for r in rows):
        raise ValueError("Model/policy selection may read only the selection partition")
    return max(enumerate(rows), key=lambda pair: (pair[1]["counts"]["f1"], -pair[0]))[1]
