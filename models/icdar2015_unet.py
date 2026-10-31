"""Isolated approximate ICDAR2015 transfer of the qualified Total-Text U-Net."""

from models.unet_adv import PracticalTotalTextHighResAGUNet


class PracticalICDAR2015AGUNet(PracticalTotalTextHighResAGUNet):
    """Reuse the single-output decoder and residual attention, with a new identity.

    No Total-Text weights are implicitly transferred. The training entry point
    loads a checksum-pinned ImageNet encoder and trains on ICDAR2015 TRAIN only.
    """

    PRACTICAL_IDENTITY = "practical-icdar2015-totaltext-transfer-agunet-v1"

    def __init__(self, *, skip_floor: float = 0.25, encoder_pretrained: bool = True):
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
