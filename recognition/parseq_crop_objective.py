"""Differentiable local boxes and frozen PARSeq teacher forcing, opt-in only."""

import torch
from torch.nn import functional as F


def bounded_boxes(boxes, raw_deltas, shape, *, max_fraction):
    if boxes.shape != raw_deltas.shape or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected matching Nx4 boxes and deltas")
    if not 0 < max_fraction < .5 or not torch.isfinite(raw_deltas).all():
        raise ValueError("Invalid bounded box deltas")
    size = boxes[:, 2:] - boxes[:, :2]
    if not (size > 0).all():
        raise ValueError("Empty source box")
    corrected = boxes + raw_deltas.tanh() * max_fraction * torch.cat((size, size), 1)
    return torch.stack((corrected[:, 0].clamp(0, shape[1]), corrected[:, 1].clamp(0, shape[0]),
                        corrected[:, 2].clamp(0, shape[1]), corrected[:, 3].clamp(0, shape[0])), 1)


def sample_boxes(rgb, boxes, output_size):
    """Exclusive continuous xyxy; align_corners=False pixel-center sampling.

    Run on CPU even for CUDA geometry/PARSeq: CPU grid_sample backward preserves
    the declared deterministic policy; differentiable transfers retain box grads.
    This bilinear training transform differs from official bicubic PIL inference.
    """
    if rgb.device.type != "cpu" or boxes.device.type != "cpu":
        raise ValueError("Differentiable sampling requires explicit CPU tensors")
    if rgb.ndim != 4 or rgb.shape[1] != 3 or boxes.shape != (len(rgb), 4):
        raise ValueError("Sampling input/box shapes differ")
    if not torch.isfinite(boxes).all() or not (boxes[:, 2:] > boxes[:, :2]).all():
        raise ValueError("Invalid continuous crop")
    oh, ow = output_size
    if min(oh, ow) < 1:
        raise ValueError("Empty output crop")
    h, w = rgb.shape[-2:]
    x = (torch.arange(ow, dtype=boxes.dtype) + .5) / ow
    y = (torch.arange(oh, dtype=boxes.dtype) + .5) / oh
    gx = boxes[:, 0, None] + x * (boxes[:, 2]-boxes[:, 0])[:, None]
    gy = boxes[:, 1, None] + y * (boxes[:, 3]-boxes[:, 1])[:, None]
    grid = torch.stack((gx[:, None, :].expand(-1, oh, -1) * (2/w)-1,
                        gy[:, :, None].expand(-1, -1, ow) * (2/h)-1), -1)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=False)


def frozen_parseq_loss(recognizer, crops, texts):
    """Causal CE on exact supported GT strings plus EOS, equal mean per word.

    Unknown characters/empty/too-long labels fail explicitly; the runner counts
    those as unsupervised OCR training examples without dropping evaluation GT.
    Freezing weights must not disable autograd to crop coordinates.
    """
    system = recognizer.model
    model, tokenizer = system.model, system.tokenizer
    if any(p.requires_grad for p in system.parameters()) or system.training:
        raise ValueError("PARSeq must be eval and all of its weights frozen")
    if len(texts) != len(crops) or not texts or any(not text for text in texts):
        raise ValueError("Nonempty one-to-one OCR targets required")
    if any(len(text) > model.max_label_length for text in texts):
        raise ValueError("OCR target exceeds qualified PARSeq label length")
    tokens = tokenizer.encode(texts, device=crops.device)
    context, target = tokens[:, :-1], tokens[:, 1:]
    length = context.shape[1]
    mask = torch.triu(torch.ones(length, length, device=crops.device, dtype=torch.bool), diagonal=1)
    memory = model.encode(crops * 2 - 1)
    decoded = model.decode(context, memory, tgt_mask=mask,
                           tgt_padding_mask=context == tokenizer.pad_id, tgt_query_mask=mask)
    logits = model.head(decoded)
    losses = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=tokenizer.pad_id, reduction="none")
    valid = target != tokenizer.pad_id
    return ((losses * valid).sum(1) / valid.sum(1)).mean()
