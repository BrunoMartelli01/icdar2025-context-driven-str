"""TRAIN-only word reader supervision and matched text/kernel objectives."""

from collections import Counter
import math

import numpy as np
import torch
from torch.nn import functional as F

from models.unet_auxiliary_ocr import axis_feature_crops
from segmentation.word_multitask import masked_bce_dice


def vocabulary(texts):
    """Blank=0, Unicode casefold characters derived only from fit transcripts."""
    values = sorted({ch for text in texts for ch in text.casefold()})
    if not values:
        raise ValueError("Fit transcripts contain no characters")
    return values


def word_supervision(owners, validity, texts, characters, *, max_length, time_steps,
                     max_words, epoch):
    owners, validity = np.asarray(owners), np.asarray(validity, bool)
    if (owners.ndim != 2 or owners.shape != validity.shape or owners.dtype.kind not in "iu"
            or np.any(owners < 0) or max_length < 1 or time_steps < 1 or max_words < 1 or epoch < 0):
        raise ValueError("Invalid native owner supervision/settings")
    present = sorted(int(v) for v in np.unique(owners[owners > 0]))
    if any(v > len(texts) for v in present):
        raise ValueError("Owner ID is not aligned to the care transcript order")
    lookup = {ch: i+1 for i, ch in enumerate(characters)}
    if len(lookup) != len(characters):
        raise ValueError("Duplicate vocabulary characters")
    rows, counts = [], Counter(expected_words=len(texts), native_missing=len(texts)-len(present))
    for owner in present:
        active = (owners == owner) & validity
        if not active.any():
            counts["ignored_owner"] += 1
            continue
        text = texts[owner-1].casefold()
        if not text:
            counts["empty_transcript"] += 1
            continue
        if any(ch not in lookup for ch in text):
            counts["unsupported_character"] += 1
            continue
        repeated = sum(a == b for a, b in zip(text, text[1:]))
        if len(text) > max_length or len(text) + repeated > time_steps:
            counts["ctc_length_infeasible"] += 1
            continue
        y, x = np.where(active)
        rows.append({"owner_id": owner, "text": text,
            "tokens": [lookup[ch] for ch in text],
            "bbox": [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]})
    counts["eligible_words"] = len(rows)
    # Rotate through annotation order, independent of outcomes. Dense images
    # cannot permanently hide their later words behind the per-image budget.
    if rows:
        offset = (epoch * max_words) % len(rows)
        rows = (rows[offset:] + rows[:offset])[:max_words]
    counts["selected_words"] = len(rows)
    return rows, dict(counts)


def ctc_word_losses(log_probabilities, token_lists):
    """Deterministic CPU CTC, normalized per character then equal per word.

    No zero_infinity masking: infeasible labels fail before loss; nonfinite loss
    propagates. The CPU transfer is an explicit recorded hardware deviation.
    """
    t, n, classes = log_probabilities.shape
    if n != len(token_lists) or not n or not torch.isfinite(log_probabilities).all():
        raise ValueError("Finite TNC logits and one nonempty label per crop required")
    for tokens in token_lists:
        if (not tokens or min(tokens) < 1 or max(tokens) >= classes
                or len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])) > t):
            raise ValueError("Invalid/infeasible CTC label; blank is never a target")
    lengths = torch.tensor([len(v) for v in token_lists], dtype=torch.long)
    target = torch.tensor([i for row in token_lists for i in row], dtype=torch.long)
    loss = F.ctc_loss(log_probabilities.cpu(), target,
        torch.full((n,), t, dtype=torch.long), lengths, blank=0,
        reduction="none", zero_infinity=False) / lengths
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite CTC loss")
    return loss.to(log_probabilities.device)


def auxiliary_loss(features, reader, supervision, *, output_hw, detach, stats=None):
    values = []
    for i, rows in enumerate(supervision):
        if rows:
            crops = axis_feature_crops(features[i:i+1], [r["bbox"] for r in rows],
                                      output_hw=output_hw, detach=detach)
            logits = reader(crops)
            values.append(ctc_word_losses(logits, [r["tokens"] for r in rows]).mean())
            if stats is not None:
                for sequence, row in zip(logits.detach().argmax(-1).T.cpu().tolist(), rows):
                    decoded, previous = [], -1
                    for token in sequence:
                        if token and token != previous: decoded.append(token)
                        previous = token
                    stats["reader_crop_words"] += 1
                    stats["reader_greedy_correct_words"] += int(decoded == row["tokens"])
    if values:
        return torch.stack(values).mean()
    # A genuinely unsupervised objective has no gradient path to the U-Net.
    return sum(p.sum()*0 for p in reader.parameters())


def segmentation_objective(words, batch, config):
    opts = {k: config[k] for k in ("bce_weight", "dice_weight", "smooth")}
    valid = batch["validity"]
    text = masked_bce_dice(words[:, :1], batch["mask"], valid, **opts,
        pixel_weight=1 + (config["separation_weight"]-1)*batch["separation"])
    kernel = masked_bce_dice(words[:, 1:2], batch["kernel"], valid, **opts)
    if not math.isfinite(config["lambda_text"]) or not math.isfinite(config["lambda_kernel"]):
        raise ValueError("Nonfinite task weight")
    return config["lambda_text"]*text + config["lambda_kernel"]*kernel


def selection_key(candidate):
    a = candidate["aggregate"]
    return a["exact_casefold_iou_ge_50"], a["hmean"], -candidate["epoch"]
