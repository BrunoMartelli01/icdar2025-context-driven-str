"""Experimental crop geometry learned through a frozen PARSeq recognizer.

Training uses a bilinear differentiable RGB proxy. Evaluation always returns
integer boxes for the unchanged, qualified PIL/PARSeq recognition transform.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecognitionCropRefiner(nn.Module):
    def __init__(self, *, translation, log_scale):
        super().__init__()
        self.translation, self.log_scale = translation, log_scale
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(260, 64), nn.ReLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, context, boxes):
        """Boxes and output use context-relative [0,1] exclusive xyxy."""
        if context.ndim != 4 or context.shape[1] != 4 or boxes.shape != (len(context), 4):
            raise ValueError("Invalid crop-refiner tensor shapes")
        inputs = F.interpolate(context, (32, 128), mode="bilinear", align_corners=False)
        delta = self.head(torch.cat((self.features(inputs), boxes), dim=1)).tanh()
        center = (boxes[:, :2]+boxes[:, 2:])/2
        size = boxes[:, 2:]-boxes[:, :2]
        if torch.any(size <= 0): raise ValueError("Empty source crop")
        center = center + delta[:, :2]*self.translation*size
        size = size*(delta[:, 2:]*self.log_scale).exp()
        return torch.cat((center-size/2, center+size/2), dim=1).clamp(0, 1)


def sample_boxes(images, boxes, size=(32, 128)):
    """Pixel-center sampling, align_corners=False, exclusive normalized boxes."""
    if images.ndim != 4 or boxes.shape != (len(images), 4):
        raise ValueError("Sampler tensor shapes differ")
    h, w = size
    y = (torch.arange(h, device=boxes.device, dtype=boxes.dtype)+.5)/h
    x = (torch.arange(w, device=boxes.device, dtype=boxes.dtype)+.5)/w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    start = boxes[:, None, None, :2]
    extent = (boxes[:, 2:]-boxes[:, :2])[:, None, None]
    grid = start + torch.stack((xx, yy), -1)[None]*extent
    return F.grid_sample(images, grid*2-1, mode="bilinear", padding_mode="border", align_corners=False)


def box_loss(boxes, truth, neighbors, valid_neighbors):
    """Coverage, neighbor contamination, and bounded geometry supervision."""
    def intersection(a, b):
        return (torch.minimum(a[..., 2:], b[..., 2:])-torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    target_area = (truth[:, 2:]-truth[:, :2]).prod(-1).clamp_min(1e-6)
    coverage = 1-intersection(boxes, truth)/target_area
    contamination = intersection(boxes[:, None], neighbors) * valid_neighbors
    contamination = contamination.sum(1)/target_area
    regression = F.smooth_l1_loss(boxes, truth, reduction="none").mean(1)
    return coverage.mean(), contamination.mean(), regression.mean()


def frozen_parseq_loss(system, images, texts):
    """Forward autoregressive teacher forcing, including EOS, no weight updates.

    Uses the pinned official runtime's encode/decode/head and canonical causal
    mask. This is an explicit training proxy; deployment retains AR+refine1.
    """
    if any(p.requires_grad for p in system.parameters()):
        raise ValueError("Recognition weights must be frozen")
    tokenizer, model = system.tokenizer, system.model
    targets = tokenizer.encode(list(texts), device=images.device)
    if targets.shape[1]-2 > model.max_label_length:
        raise ValueError("Training transcription exceeds qualified PARSeq length")
    inputs, outputs = targets[:, :-1], targets[:, 1:]
    length = inputs.shape[1]
    mask = torch.triu(torch.ones(length, length, device=images.device, dtype=torch.bool), 1)
    padding = (inputs == tokenizer.pad_id) | (inputs == tokenizer.eos_id)
    memory = model.encode(images)
    decoded = model.decode(inputs, memory, tgt_mask=mask, tgt_padding_mask=padding, tgt_query_mask=mask)
    logits = model.head(decoded)
    return F.cross_entropy(logits.flatten(0, 1), outputs.flatten(), ignore_index=tokenizer.pad_id)
