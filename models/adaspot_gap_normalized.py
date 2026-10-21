"""TRAIN-normalized linear/nonlinear spatial Gap heads, isolated from M/K."""

import torch
from torch import nn
from torch.nn import functional as F


class NormalizedGapHead(nn.Module):
    def __init__(self, kind, mean, scale, *, hidden, initial_bias):
        super().__init__()
        if (kind not in ("linear", "nonlinear") or mean.shape != (576,) or scale.shape != (576,)
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0)):
            raise ValueError("Finite 576-dimensional TRAIN normalization and declared head required")
        self.kind = kind
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.first = nn.Linear(576, hidden if kind == "nonlinear" else 1)
        if kind == "nonlinear":
            self.last = nn.Linear(hidden, 1)
            nn.init.zeros_(self.last.weight); nn.init.constant_(self.last.bias, initial_bias)
        else:
            nn.init.zeros_(self.first.weight); nn.init.constant_(self.first.bias, initial_bias)

    def forward(self, patches):
        if patches.ndim != 2 or patches.shape[1] != 576:
            raise ValueError("Channel-major 3x3 feature patches required")
        values = self.first((patches - self.mean) / self.scale)
        return (self.last(F.relu(values)) if self.kind == "nonlinear" else values)[:, 0]

    def native_logits(self, features):
        if features.ndim != 3 or features.shape[0] != 64:
            raise ValueError("Expected 64-channel native features")
        # Normalization is applied AFTER zero padding in patch space. Subtract
        # every offset's mean through the bias, including out-of-image offsets.
        weight = (self.first.weight / self.scale).reshape(-1, 64, 3, 3)
        bias = self.first.bias - self.first.weight @ (self.mean / self.scale)
        values = F.conv2d(features[None], weight, bias, padding=1)
        if self.kind == "nonlinear":
            values = F.conv2d(F.relu(values), self.last.weight[:, :, None, None], self.last.bias)
        return values[0, 0]
