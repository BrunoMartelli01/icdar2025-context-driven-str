"""Practical feature-wavelet skip ablation; the paper-faithful model is untouched."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.unet_adv import PracticalTotalTextHighResAGUNet


def haar_details(features: torch.Tensor) -> torch.Tensor:
    """Three fixed, local Haar detail bands of an encoder feature map."""
    if features.ndim != 4 or features.shape[-2] % 2 or features.shape[-1] % 2:
        raise ValueError("Haar input must be BxCxHxW with even spatial dimensions")
    batch, channels, height, width = features.shape
    pixels = F.pixel_unshuffle(features, 2).reshape(
        batch, channels, 4, height // 2, width // 2)
    a, b, c, d = pixels.unbind(dim=2)
    horizontal = (a + b - c - d) * 0.5
    vertical = (a - b + c - d) * 0.5
    diagonal = (a - b - c + d) * 0.5
    return torch.cat((horizontal, vertical, diagonal), dim=1)


class PracticalTotalTextFrequencyGatedSkipAGUNet(PracticalTotalTextHighResAGUNet):
    """Gate feature-level Haar details with decoder context at the /2 skip.

    The scalar residual gain starts at zero, so the new path cannot silently
    alter the initial baseline function. It is a practical ablation, not the
    paper AG-UNet architecture.
    """

    PRACTICAL_IDENTITY = "practical-totaltext-v4-feature-haar-gated-skip-v1"

    def __init__(self, *, skip_floor: float = 0.25,
                 encoder_pretrained: bool = True) -> None:
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
        self.frequency_projection = nn.Conv2d(64 * 3, 64, kernel_size=1, bias=False)
        self.frequency_gate = nn.Conv2d(128 + 64, 64, kernel_size=1, bias=True)
        self.frequency_gain = nn.Parameter(torch.zeros(()))
        nn.init.kaiming_normal_(self.frequency_projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.frequency_gate.bias)

    def gated_skip(self, shallow: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        if shallow.ndim != 4 or shallow.shape[1] != 64 or decoder.ndim != 4 or decoder.shape[1] != 128:
            raise ValueError("Expected /2 skip with 64 channels and decoder with 128")
        detail = haar_details(shallow)
        if decoder.shape[-2:] != detail.shape[-2:]:
            raise ValueError("Decoder and wavelet details must share /4 geometry")
        shallow_low = F.avg_pool2d(shallow, 2)
        gate = torch.sigmoid(self.frequency_gate(torch.cat((shallow_low, decoder), dim=1)))
        residual = gate * self.frequency_projection(detail)
        residual = F.interpolate(residual, size=shallow.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return shallow + self.frequency_gain * residual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB image tensor Bx3xHxW")
        height, width = image.shape[-2:]
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError("Input dimensions must be >=32 and divisible by 32")
        e0, e1, e2, e3, e4 = self._encode(image)
        bottleneck = self.bottleneck(e4)
        d3 = self.up3(bottleneck, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, self.gated_skip(e0, d1))
        logits = self.head(d0)
        if logits.shape[-2:] != (height // 2, width // 2):
            raise RuntimeError("Native /2 mask resolution changed")
        return logits


__all__ = ["PracticalTotalTextFrequencyGatedSkipAGUNet", "haar_details"]
