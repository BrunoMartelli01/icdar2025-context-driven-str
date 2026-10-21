"""Approximate Gap-specific decoder; qualified Text/Kernel path stays frozen."""

from copy import deepcopy
import math

import torch
from torch import nn


class TrainableGapDecoder(nn.Module):
    def __init__(self, qualified_source, *, initial_bias):
        super().__init__()
        if not math.isfinite(initial_bias) or initial_bias >= 0:
            raise ValueError("Negative finite initial Gap bias required")
        self.source = qualified_source.eval()
        for parameter in self.source.parameters():
            parameter.requires_grad_(False)
        # Real feature adaptation, initialized from the existing complete decoder.
        # Deep copies deliberately do not alias any source tensors or BN buffers.
        self.decoder = nn.ModuleDict({name: deepcopy(getattr(self.source, name))
                                     for name in ("bottleneck", "up3", "up2", "up1", "up0")})
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(True)
        self.gap = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.gap.weight)
        nn.init.constant_(self.gap.bias, initial_bias)

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()  # Frozen parameters alone would not freeze BN statistics.
        return self

    def branch_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("source.")}

    def load_branch(self, state):
        own = self.state_dict()
        expected = {k for k in own if not k.startswith("source.")}
        if set(state) != expected:
            raise ValueError("Complete exact Gap branch required")
        if any(own[k].shape != v.shape or own[k].dtype != v.dtype for k, v in state.items()):
            raise ValueError("Gap branch tensor shape/dtype mismatch")
        own.update(state)
        self.load_state_dict(own, strict=True)

    def forward_evidence(self, image, *, teacher=True):
        if image.ndim != 4 or image.shape[1] != 3 or any(v % 32 for v in image.shape[-2:]):
            raise ValueError("Expected RGB batch with dimensions divisible by32")
        with torch.no_grad():
            e0, e1, e2, e3, e4 = self.source._encode(image)
            if teacher:
                feature = self.source.up3(self.source.bottleneck(e4), e3)
                feature = self.source.up2(feature, e2)
                feature = self.source.up1(feature, e1)
                feature = self.source.up0(feature, e0)
                words = self.source.head(feature)
            else:
                words = None
        feature = self.decoder["up3"](self.decoder["bottleneck"](e4), e3)
        feature = self.decoder["up2"](feature, e2)
        feature = self.decoder["up1"](feature, e1)
        feature = self.decoder["up0"](feature, e0)
        return {"words": words, "gap": self.gap(feature)}

    def forward(self, image):
        return self.forward_evidence(image, teacher=False)["gap"]
