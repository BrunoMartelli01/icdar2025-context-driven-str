"""Isolated AdaSpot text/kernel/gap experiment on the qualified word backbone."""

import math

import torch
from torch import nn

from models.unet_word_research import WordResearchAGUNet


BRANCHES = ("gap", "control")


class AdaSpotGapAGUNet(WordResearchAGUNet):
    """One new 65-parameter gap head; all source word/kernel tensors retained."""

    def __init__(self, branch, *, skip_floor, initial_gap_bias):
        if branch not in BRANCHES or not math.isfinite(initial_gap_bias) or initial_gap_bias >= 0:
            raise ValueError("Expected gap/control branch and finite negative initial gap bias")
        super().__init__("control", skip_floor=skip_floor, embedding_dimensions=4,
                         encoder_pretrained=False)
        self.branch = branch
        if branch == "gap":
            self.gap = nn.Conv2d(64, 1, 1)
            nn.init.zeros_(self.gap.weight)
            nn.init.constant_(self.gap.bias, initial_gap_bias)

    @property
    def architecture_identity(self):
        return "experimental-adaspot-text-kernel-{}-v1".format(self.branch)

    def initialize_kernel_state(self, state):
        own = self.state_dict()
        common = {key for key in own if not key.startswith("gap.")}
        if common != set(state):
            raise ValueError("Warm start requires every qualified word/kernel tensor and no others")
        for key, value in state.items():
            if own[key].shape != value.shape or own[key].dtype != value.dtype:
                raise ValueError("Warm-start tensor shape/dtype mismatch: " + key)
            own[key] = value
        self.load_state_dict(own, strict=True)
        return {"copied_tensors": len(common), "new_tensors": sorted(set(own) - common),
                "strict_state_dict_load": True}

    def forward_evidence(self, image):
        evidence = super().forward_evidence(image)
        evidence["gap"] = self.gap(evidence["features"]) if self.branch == "gap" else None
        return evidence

    def forward(self, image):
        evidence = self.forward_evidence(image)
        return (torch.cat((evidence["words"], evidence["gap"]), dim=1)
                if evidence["gap"] is not None else evidence["words"])
