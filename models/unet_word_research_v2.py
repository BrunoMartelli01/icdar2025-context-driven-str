"""Isolated APPROXIMATE corrective experiments, with protected word features."""

import math
import torch
from torch import nn

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet


BRANCHES = ("reference_control", "control", "glyph_isolated", "glyph_staged",
            "links_isolated", "threshold_frozen", "db_core")
PRIVATE = ("glyph_private.", "links_private.", "threshold_private.", "db_private.")


def private_head(channels, hidden, outputs):
    return nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(hidden, outputs, 1))


class WordResearchV2AGUNet(PracticalTotalTextWordKernelAGUNet):
    def __init__(self, branch, config):
        if branch not in BRANCHES:
            raise ValueError("Unknown corrective branch")
        super().__init__(skip_floor=config["skip_floor"], encoder_pretrained=False)
        self.branch, self.research_config, self.research_epoch = branch, config, 0
        # Preserve the original encoder's requires_grad policy, including its BN.
        self.original_trainability = {k: p.requires_grad for k, p in self.named_parameters()}
        hidden = config["private_head_channels"]
        if branch.startswith("glyph_"):
            self.glyph_private = private_head(128, hidden, 1)
            nn.init.zeros_(self.glyph_private[-1].weight)
            nn.init.constant_(self.glyph_private[-1].bias, config["glyph"]["initial_bias"])
        if branch == "links_isolated":
            self.links_private = private_head(128, hidden, 8)
        if branch == "threshold_frozen":
            self.threshold_private = private_head(128, hidden, 1)
            low, high = config["adaptive"]["threshold_range"]
            fraction = (config["postprocessing"]["text_threshold"] - low) / (high - low)
            nn.init.zeros_(self.threshold_private[-1].weight)
            nn.init.constant_(self.threshold_private[-1].bias, math.log(fraction / (1 - fraction)))
        if branch == "db_core":
            self.db_private = private_head(128, hidden, 2)
            nn.init.zeros_(self.db_private[-1].weight)
            nn.init.zeros_(self.db_private[-1].bias)
        self.set_research_epoch(0)

    @property
    def architecture_identity(self):
        return "experimental-unet-word-research-v2:" + self.branch

    @property
    def shared_trainable(self):
        return self.branch == "control" or (self.branch == "glyph_staged" and
            self.research_epoch > self.research_config["glyph"]["warmup_epochs"])

    def set_research_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Nonnegative research epoch required")
        self.research_epoch = epoch
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(PRIVATE) or
                             (self.shared_trainable and self.original_trainability.get(name, False)))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not self.shared_trainable:
            super().train(False)
            for name in ("glyph_private", "links_private", "threshold_private", "db_private"):
                if hasattr(self, name):
                    getattr(self, name).train(mode)
        return self

    def initialize_control_state(self, state):
        own = self.state_dict()
        original = {k for k in own if not k.startswith(PRIVATE)}
        if original != set(state):
            raise ValueError("Source must contain every original word/kernel tensor")
        for key, value in state.items():
            if own[key].shape != value.shape or own[key].dtype != value.dtype:
                raise ValueError("Control tensor shape/dtype mismatch: " + key)
            own[key] = value
        self.load_state_dict(own, strict=True)
        return {"copied_tensors": len(original), "new_tensors": sorted(set(own) - original)}

    def forward_evidence(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 32 or any(v % 32 for v in image.shape[-2:]):
            raise ValueError("Expected RGB dimensions divisible by32")
        e0, e1, e2, e3, e4 = self._encode(image)
        f = self.up3(self.bottleneck(e4), e3)
        f = self.up2(f, e2); f = self.up1(f, e1); f = self.up0(f, e0)
        words = self.head(f)
        private = torch.cat((f, e0), dim=1)
        db = self.db_private(private) if self.branch == "db_core" else None
        return {"words": words, "features": f,
                "glyph": self.glyph_private(private) if self.branch.startswith("glyph_") else None,
                "links": self.links_private(private) if self.branch == "links_isolated" else None,
                "threshold": self.threshold_private(private) if self.branch == "threshold_frozen" else None,
                "db_probability": words[:, 1:2] + db[:, :1] if db is not None else None,
                "db_threshold": db[:, 1:2] if db is not None else None}

    def forward(self, image):
        return self.forward_evidence(image)["words"]
