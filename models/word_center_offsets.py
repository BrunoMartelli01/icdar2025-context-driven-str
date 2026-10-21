"""Experimental auxiliary U-Net: each text pixel points toward its word center.

This is an APPROXIMATE extension, not a paper segmentation model. The existing
text U-Net supplies the shape; this separate network supplies a location cue.
"""

import torch
from torch import nn

from models.unet_adv import PracticalTotalTextHighResAGUNet


class WordCenterOffsetAGUNet(PracticalTotalTextHighResAGUNet):
    """Return Bx2x(H/2)x(W/2) raw x/y offsets; apply tanh before voting.

Offsets are normalized by native width and height, respectively. The ResNet50,
attention decoder and frozen encoder/BN behavior are inherited unchanged.
The two-channel head starts at zero displacement. No text mask is predicted.
"""

    PRACTICAL_IDENTITY = "experimental-word-center-offset-agunet-v1"

    def __init__(self, *, skip_floor, encoder_pretrained):
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
        self.head = nn.Conv2d(self.head.in_channels, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initialize_from_text_model(self, model):
        """Strictly copy a loaded classic backbone, explicitly resetting its head.

        The source is never mutated. This is warm-start initialization, not
        loading a trained offset checkpoint. Offset checkpoints load strictly
        using this class's own architecture identity.
        """
        if (not isinstance(model, PracticalTotalTextHighResAGUNet)
            or model.head.out_channels != 1 or model.skip_floor != self.skip_floor):
            raise ValueError("Warm start requires a single-output practical text U-Net with the same skip_floor")
        state = dict(model.state_dict())
        if state["head.weight"].shape != (1, self.head.in_channels, 1, 1):
            raise ValueError("Unexpected text prediction head schema")
        state["head.weight"] = torch.zeros_like(self.head.weight)
        state["head.bias"] = torch.zeros_like(self.head.bias)
        destination = self.state_dict()
        if any(value.dtype != destination[key].dtype for key, value in state.items() if key in destination):
            raise ValueError("Warm start requires matching tensor dtypes; precision changes must be explicit")
        self.load_state_dict(state, strict=True)
        return {"classification": "ASSUMPTION", "source_architecture": model.architecture_identity,
                "reset_tensors": ["head.weight", "head.bias"],
                "initialization": "zero normalized displacement; strict backbone copy"}

