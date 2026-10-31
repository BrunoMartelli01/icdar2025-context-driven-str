"""Approximate center/extent instance decoder on qualified frozen U-Net features.

This is an output-representation experiment, inspired by Objects as Points.
It does not use a YOLO model or replace the shared Attention-Gated U-Net.
"""

import math

import torch
from torch import nn


class DenseWordInstances(nn.Module):
    def __init__(self, *, input_channels, width, groups, dilations, initial_probability,
                 initial_size):
        super().__init__()
        if width % groups or not 0 < initial_probability < 1 or min(initial_size) <= 0:
            raise ValueError("Explicit valid dense-instance architecture required")
        layers = []
        for dilation in dilations:
            layers.extend([nn.Conv2d(input_channels, width, 3, padding=dilation,
                                     dilation=dilation, bias=False),
                           nn.GroupNorm(groups, width), nn.GELU()])
            input_channels = width
        self.tower = nn.Sequential(*layers)
        self.center = nn.Conv2d(width, 1, 1)
        self.offset = nn.Conv2d(width, 2, 1)
        self.extent = nn.Conv2d(width, 2, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.center.bias, math.log(initial_probability/(1-initial_probability)))
        with torch.no_grad():
            self.extent.bias.copy_(torch.tensor(initial_size).log())

    def forward(self, features):
        if features.ndim != 4 or not torch.isfinite(features).all():
            raise ValueError("Finite BxCxHxW U-Net features required")
        hidden = self.tower(features)
        return dict(center=self.center(hidden), offset=self.offset(hidden),
                    log_size=self.extent(hidden))
