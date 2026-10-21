"""An isolated APPROXIMATE centerline/radius head on frozen word features."""

import math

import torch
from torch import nn

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet


class CenterlineRadiusAGUNet(PracticalTotalTextWordKernelAGUNet):
    """Preserve every source word/kernel tensor, including BatchNorm buffers.

    This tests information in existing frozen features. It does not fine-tune
    the segmentation backbone and is not an implementation of TextSnake.
    """

    PRIVATE_PREFIX = "centerline_private."

    def __init__(self, config):
        hidden = config["private_head_channels"]
        if type(hidden) is not int or hidden < 1:
            raise ValueError("Positive private head channel count required")
        super().__init__(skip_floor=config["skip_floor"], encoder_pretrained=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.centerline_private = nn.Sequential(
            nn.Conv2d(128, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 2, 1))
        line_bias, initial_radius = config["initial_line_bias"], config["initial_radius"]
        radius_floor = config["loss"]["radius_floor"]
        if (not all(math.isfinite(value) for value in (line_bias, initial_radius, radius_floor))
                or radius_floor <= 0 or initial_radius <= radius_floor):
            raise ValueError("Finite initial line bias and radius above its positive floor required")
        radius_delta = initial_radius - radius_floor
        # Stable inverse softplus, including large explicitly configured radii.
        radius_bias = radius_delta + math.log(-math.expm1(-radius_delta))
        nn.init.zeros_(self.centerline_private[-1].weight)
        with torch.no_grad():
            self.centerline_private[-1].bias.copy_(torch.tensor([line_bias, radius_bias]))
        self.train(self.training)

    @property
    def architecture_identity(self):
        return "experimental-frozen-word-centerline-radius-agunet-v1"

    def train(self, mode=True):
        super().train(False)
        # The parent remains in eval: no source BatchNorm buffers may move.
        if hasattr(self, "centerline_private"):
            self.centerline_private.train(mode)
        return self

    def initialize_control_state(self, state):
        own = self.state_dict()
        original = {key for key in own if not key.startswith(self.PRIVATE_PREFIX)}
        if original != set(state):
            raise ValueError("Source must contain every original word/kernel tensor and no others")
        for key, value in state.items():
            if own[key].shape != value.shape or own[key].dtype != value.dtype:
                raise ValueError("Control tensor shape/dtype mismatch: " + key)
            own[key] = value
        self.load_state_dict(own, strict=True)
        return {"copied_tensors": len(original), "new_tensors": sorted(set(own) - original)}

    def forward_evidence(self, image):
        if (image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 32
                or any(value % 32 for value in image.shape[-2:])):
            raise ValueError("Expected RGB dimensions divisible by 32")
        with torch.no_grad():
            e0, e1, e2, e3, e4 = self._encode(image)
            feature = self.up3(self.bottleneck(e4), e3)
            feature = self.up2(feature, e2)
            feature = self.up1(feature, e1)
            feature = self.up0(feature, e0)
            words = self.head(feature)
            private = torch.cat((feature, e0), dim=1)
        auxiliary = self.centerline_private(private)
        return {"words": words, "line": auxiliary[:, :1], "radius_raw": auxiliary[:, 1:2]}

    def forward(self, image):
        return self.forward_evidence(image)["words"]
