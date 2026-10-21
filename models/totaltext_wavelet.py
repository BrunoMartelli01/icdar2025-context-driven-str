"""Tiny fixed-Haar detail adapter for the practical native-256 Total-Text model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.unet_adv import PracticalTotalTextHighResAGUNet


class PracticalTotalTextWaveletHighResAGUNet(PracticalTotalTextHighResAGUNet):
    """Inject local high-frequency detail into the /2 skip, preserving RGB ResNet."""

    PRACTICAL_IDENTITY = "practical-totaltext-v4-highres-haar-adapter-v1"

    def __init__(self, *, skip_floor: float = 0.25, encoder_pretrained: bool = True):
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
        kernels = torch.tensor([
            [[1, 1], [-1, -1]],
            [[1, -1], [1, -1]],
            [[1, -1], [-1, 1]],
        ], dtype=torch.float32).unsqueeze(1) / 2.0
        self.register_buffer("haar_kernels", kernels, persistent=True)
        self.wavelet_adapter = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        nn.init.zeros_(self.wavelet_adapter.weight)

    def _encode(self, x: torch.Tensor):
        encoded = super()._encode(x)
        # RGB is already normalized; the fixed luminance projection still
        # measures local differences. No trainable changes to the frozen stem.
        gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
        detail = F.conv2d(gray, self.haar_kernels, stride=2)
        if detail.shape[-2:] != encoded[0].shape[-2:]:
            raise RuntimeError("Haar and encoder /2 skip geometry differ")
        encoded[0] = encoded[0] + self.wavelet_adapter(detail)
        return encoded


__all__ = ["PracticalTotalTextWaveletHighResAGUNet"]
