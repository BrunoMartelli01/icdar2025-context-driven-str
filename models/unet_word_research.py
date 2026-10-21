"""Isolated APPROXIMATE heads; existing word and strict models are unchanged."""

import torch
from torch import nn
from torch.nn import functional as F

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet


BRANCHES = ("control", "glyph_word", "embedding", "adaptive_threshold", "quality")


class MaskQualityHead(nn.Module):
    """Crop features plus predicted support -> bbox quality, coverage, purity.

    Neither owner maps nor ground truth may be passed to this forward method.
    """

    def __init__(self, channels=64):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels + 1, 16, 3, padding=1), nn.ReLU())
        self.predict = nn.Linear(16, 3)

    def forward(self, features, masks):
        if features.ndim != 4 or masks.shape != (len(features), 1, *features.shape[-2:]):
            raise ValueError("Quality head requires aligned crop features and predicted masks")
        # Spatial mean has deterministic backward; CUDA adaptive-pool backward
        # would violate the experiment's deterministic-algorithm contract.
        return self.predict(self.features(torch.cat((features, masks), dim=1)).mean(dim=(-2, -1)))


class WordResearchAGUNet(PracticalTotalTextWordKernelAGUNet):
    def __init__(self, branch, *, skip_floor, embedding_dimensions, encoder_pretrained=False):
        if branch not in BRANCHES or type(embedding_dimensions) is not int or embedding_dimensions < 1:
            raise ValueError("Invalid research branch/embedding dimension")
        super().__init__(skip_floor=skip_floor, encoder_pretrained=encoder_pretrained)
        self.branch = branch
        if branch == "glyph_word":
            self.glyph = nn.Conv2d(64, 1, 1)
        if branch == "embedding":
            self.embedding = nn.Conv2d(64, embedding_dimensions, 1)
        if branch == "adaptive_threshold":
            self.threshold = nn.Conv2d(64, 1, 1)
            nn.init.zeros_(self.threshold.weight)
            nn.init.zeros_(self.threshold.bias)
        if branch == "quality":
            self.quality = MaskQualityHead()
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("quality."))

    @property
    def architecture_identity(self):
        return "experimental-unet-word-research-v1:" + self.branch

    def initialize_kernel_state(self, state):
        """Copy ALL original tensors strictly; explicitly retain new head init."""
        original = {k: v for k, v in self.state_dict().items()
                    if not k.startswith(("glyph.", "embedding.", "threshold.", "quality."))}
        if set(original) != set(state):
            raise ValueError("Source must be a full qualified Text + Kernel state")
        own = self.state_dict()
        for key, value in state.items():
            if own[key].shape != value.shape or own[key].dtype != value.dtype:
                raise ValueError("Kernel state shape/dtype mismatch: " + key)
            own[key] = value
        self.load_state_dict(own, strict=True)
        return {"copied_tensors": len(state), "new_tensors": sorted(set(own) - set(state))}

    def train(self, mode=True):
        super().train(mode)
        if self.branch == "quality":
            # Frozen weights must also have frozen BatchNorm running statistics.
            super().train(False)
            self.quality.train(mode)
        return self

    def forward_evidence(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 32 or any(v % 32 for v in image.shape[-2:]):
            raise ValueError("Expected RGB dimensions divisible by 32")
        e0, e1, e2, e3, e4 = self._encode(image)
        feature = self.up3(self.bottleneck(e4), e3)
        feature = self.up2(feature, e2)
        feature = self.up1(feature, e1)
        feature = self.up0(feature, e0)
        return {"words": self.head(feature), "features": feature,
                "glyph": self.glyph(feature) if self.branch == "glyph_word" else None,
                "embedding": self.embedding(feature) if self.branch == "embedding" else None,
                "threshold": self.threshold(feature) if self.branch == "adaptive_threshold" else None}

    def forward(self, image):
        return self.forward_evidence(image)["words"]

    def quality_scores(self, features, labels, *, context_pixels, pool_size):
        if features.ndim != 3 or labels.ndim != 2 or tuple(features.shape[-2:]) != labels.shape:
            raise ValueError("Quality scoring requires aligned native features and integer labels")
        if type(context_pixels) is not int or context_pixels < 0 or type(pool_size) is not int or pool_size < 1:
            raise ValueError("Invalid quality crop context/pool size")
        import numpy as np
        ids, crops, supports = [], [], []
        for instance in np.unique(labels[labels > 0]):
            yy, xx = np.where(labels == instance)
            x1, x2 = max(0, int(xx.min()) - context_pixels), min(labels.shape[1], int(xx.max()) + 1 + context_pixels)
            y1, y2 = max(0, int(yy.min()) - context_pixels), min(labels.shape[0], int(yy.max()) + 1 + context_pixels)
            mask = torch.as_tensor(labels[y1:y2, x1:x2] == instance, dtype=features.dtype, device=features.device)
            crops.append(F.adaptive_avg_pool2d(features[:, y1:y2, x1:x2], pool_size))
            supports.append(F.adaptive_avg_pool2d(mask[None], pool_size))
            ids.append(int(instance))
        if not ids:
            return ids, features.new_empty((0, 3))
        return ids, self.quality(torch.stack(crops), torch.stack(supports))
