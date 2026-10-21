"""Opt-in local word models; no changes to the qualified Text+Kernel model."""

import torch
from torch import nn


class LocalWordModel(nn.Module):
    """Matched RGB/Text/prompt networks for three independent experiments."""

    def __init__(self, kind, channels):
        super().__init__()
        if kind not in ("mask", "affinity", "geometry") or type(channels) is not int or channels < 1:
            raise ValueError("Invalid local word architecture")
        self.kind = kind
        self.features = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.output = (nn.Linear(channels + 4, 4) if kind == "geometry"
                       else nn.Conv2d(channels, 1 if kind == "mask" else 4, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, boxes=None):
        if inputs.ndim != 4 or inputs.shape[1] != 5 or not torch.isfinite(inputs).all():
            raise ValueError("Expected finite RGB/Text/kernel-prompt NCHW input")
        features = self.features(inputs)
        if self.kind == "geometry":
            if boxes is None or boxes.shape != (len(inputs), 4):
                raise ValueError("Geometry head requires one local exclusive box per input")
            scale = inputs.new_tensor([inputs.shape[-1], inputs.shape[-2]] * 2)
            return self.output(torch.cat((features.mean((-2, -1)), boxes / scale), 1))
        return self.output(features)
