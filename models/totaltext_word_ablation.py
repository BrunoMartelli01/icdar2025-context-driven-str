"""Isolated practical word-training ablations; author models are unchanged."""

import torch
from torch import nn

from models.unet_adv import PracticalTotalTextHighResAGUNet


BRANCHES = ("control", "pixel_link", "word_balanced", "word_zoom", "deep_supervision")


class PracticalTotalTextWordAGUNet(PracticalTotalTextHighResAGUNet):
    def __init__(self, branch, *, link_initial_bias, encoder_pretrained=False):
        if branch not in BRANCHES:
            raise ValueError("Unknown word ablation branch")
        super().__init__(skip_floor=0.25, encoder_pretrained=encoder_pretrained)
        self.branch = branch
        if branch == "pixel_link":
            self.links = nn.Conv2d(64, 8, 1)
            nn.init.zeros_(self.links.weight)
            nn.init.constant_(self.links.bias, link_initial_bias)
        if branch == "deep_supervision":
            self.aux128 = nn.Conv2d(128, 1, 1)
            self.aux64 = nn.Conv2d(256, 1, 1)

    @property
    def architecture_identity(self):
        return "practical-totaltext-word-ablation-v1:" + self.branch

    def initialize_base_state(self, state):
        """Strictly transfer every base tensor; explicitly initialize only new heads."""
        own = self.state_dict()
        extra = set(own) - set(state)
        expected = ({"links.weight", "links.bias"} if self.branch == "pixel_link" else
                    {"aux128.weight", "aux128.bias", "aux64.weight", "aux64.bias"}
                    if self.branch == "deep_supervision" else set())
        if extra != expected or set(state) - set(own):
            raise ValueError("Unexpected base checkpoint keys")
        for name, value in state.items():
            if value.shape != own[name].shape:
                raise ValueError(f"Base checkpoint tensor shape mismatch: {name}")
            own[name] = value
        self.load_state_dict(own, strict=True)

    def forward_evidence(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 32 or any(v % 32 for v in x.shape[-2:]):
            raise ValueError("Expected RGB input dimensions divisible by32")
        e0, e1, e2, e3, e4 = self._encode(x)
        d3 = self.up3(self.bottleneck(e4), e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)
        foreground = self.head(d0)
        if foreground.shape[-2:] != tuple(v // 2 for v in x.shape[-2:]):
            raise RuntimeError("Native decoder endpoint changed")
        return {"foreground": foreground,
                "links": self.links(d0) if self.branch == "pixel_link" else None,
                "auxiliary": (self.aux128(d1), self.aux64(d2))
                if self.training and self.branch == "deep_supervision" else ()}

    def forward(self, x):
        return self.forward_evidence(x)["foreground"]
