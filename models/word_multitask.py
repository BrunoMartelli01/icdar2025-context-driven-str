"""Small experimental heads on the existing Attention U-Net shared decoder."""

import torch
from torch import nn

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
from models.unet_adv import PracticalTotalTextHighResAGUNet


VARIANTS = {"text": 1, "text_kernel": 2, "text_kernel_center": 3}


class WordMultitaskAGUNet(PracticalTotalTextWordKernelAGUNet):
    """Text/kernel/center logits, in that order; no new decoder or encoder.

    The two common channels retain the existing kernel model's initialization.
    An additional 1x1 output costs just 64 weights and one bias.
    """

    def __init__(self, *, skip_floor=0.25, encoder_pretrained=True):
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
        original = self.head
        self.head = nn.Conv2d(original.in_channels, 3, 1, bias=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.weight[:2].copy_(original.weight)
            self.head.bias[:2].copy_(original.bias)

    @property
    def architecture_identity(self):
        return "experimental-word-text-kernel-center-agunet-v1"


def build_word_model(variant, *, skip_floor, encoder_pretrained):
    classes = {"text": PracticalTotalTextHighResAGUNet,
               "text_kernel": PracticalTotalTextWordKernelAGUNet,
               "text_kernel_center": WordMultitaskAGUNet}
    if variant not in classes:
        raise ValueError("Unknown word multi-task variant: {}".format(variant))
    return classes[variant](skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
