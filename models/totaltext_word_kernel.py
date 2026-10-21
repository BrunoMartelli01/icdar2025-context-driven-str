"""Isolated practical full-word/kernel head; the existing models are unchanged."""

import torch
from torch import nn

from models.unet_adv import PracticalTotalTextHighResAGUNet


class PracticalTotalTextWordKernelAGUNet(PracticalTotalTextHighResAGUNet):
    PRACTICAL_IDENTITY = "practical-totaltext-v4-word-kernel-agunet-v1"

    def __init__(self, *, skip_floor=0.25, encoder_pretrained=True):
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
        original = self.head
        self.head = nn.Conv2d(original.in_channels, 2, 1, bias=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight[:1].copy_(original.weight)
            self.head.bias[:1].copy_(original.bias)
