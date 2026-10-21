"""Isolated APPROXIMATE text adaptation, scale fusion and progressive kernels."""

import torch
from torch import nn
from torch.nn import functional as F

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet


BRANCHES = ("reference", "control", "scale_fusion", "progressive_kernels",
            "fusion_progressive", "pretrained_control", "pretrained_fusion_progressive")


class CompactScaleFusion(nn.Module):
    """Spatially weighted /8,/4,/2 decoder features; zero residual at initialization.

    Inspired by DBNet++ scale fusion, not its ASF implementation or checkpoint.
    """

    def __init__(self, channels=32):
        super().__init__()
        self.project = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in (256, 128, 64))
        self.weights = nn.Conv2d(3 * channels, 3, 1)
        self.residual = nn.Conv2d(channels, 64, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        size = features[-1].shape[-2:]
        projected = []
        for layer, x in zip(self.project, features):
            x = layer(x)
            height, width = x.shape[-2:]
            if size[0] % height or size[1] % width:
                raise ValueError("Fusion requires integral decoder scale ratios")
            # Exact nearest expansion using expand/reduction autograd. Avoid
            # CUDA bilinear backward's nondeterministic atomic accumulation.
            scale_y, scale_x = size[0] // height, size[1] // width
            x = x[:, :, :, None, :, None].expand(-1, -1, -1, scale_y, -1, scale_x)
            projected.append(x.reshape(x.shape[0], x.shape[1], *size))
        weights = self.weights(torch.cat(projected, dim=1)).softmax(dim=1)
        fused = sum(x * weights[:, i:i + 1] for i, x in enumerate(projected))
        return features[-1] + self.residual(fused), weights


class DetectionMethodsAGUNet(PracticalTotalTextWordKernelAGUNet):
    def __init__(self, branch, *, skip_floor=0.25, fusion_channels=32):
        if branch not in BRANCHES:
            raise ValueError("Unknown detection-method branch")
        super().__init__(skip_floor=skip_floor, encoder_pretrained=False)
        self.branch = branch
        self.has_fusion = "fusion" in branch
        self.has_progressive = "progressive" in branch
        if self.has_fusion:
            self.scale_fusion = CompactScaleFusion(fusion_channels)
        if self.has_progressive:
            self.kernel_scales = nn.Conv2d(64, 2, 1)
        # Inference reference keeps the historical encoder. All experimental
        # controls adapt the encoder with frozen BN affine/running statistics.
        self.encoder.requires_grad_(branch != "reference")
        for layer in self.encoder.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.requires_grad_(False)
        if hasattr(self.encoder, "fc"):
            self.encoder.fc.requires_grad_(False)  # Classification layer is not executed.

    @property
    def architecture_identity(self):
        return "experimental-unet-detection-methods-v1:" + self.branch

    def initialize_source(self, state):
        new_prefixes = ("scale_fusion.", "kernel_scales.")
        current = self.state_dict()
        original = {k: v for k, v in current.items() if not k.startswith(new_prefixes)}
        if set(state) != set(original):
            raise ValueError("Source tensor names differ from the complete original model")
        for name, value in state.items():
            if value.shape != original[name].shape or value.dtype != original[name].dtype:
                raise ValueError("Source tensor shape/dtype differs: " + name)
            current[name] = value
        if self.has_progressive:
            current["kernel_scales.weight"] = state["head.weight"][1:2].repeat(2, 1, 1, 1)
            current["kernel_scales.bias"] = state["head.bias"][1:2].repeat(2)
        self.load_state_dict(current, strict=True)

    def _encode(self, image):
        # The original author's frozen _encode uses no_grad. The explicit
        # experimental override is essential for actual encoder adaptation.
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(image)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        return e0, e1, e2, e3, self.encoder.layer4(e3)

    def forward_evidence(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or any(s < 32 or s % 32 for s in image.shape[-2:]):
            raise ValueError("Expected Bx3xHxW RGB with dimensions divisible by32")
        e0, e1, e2, e3, e4 = self._encode(image)
        d3 = self.up3(self.bottleneck(e4), e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)
        weights = None
        if self.has_fusion:
            d0, weights = self.scale_fusion((d2, d1, d0))
        words = self.head(d0)
        logits = torch.cat((words, self.kernel_scales(d0)), dim=1) if self.has_progressive else words
        return {"logits": logits, "scale_weights": weights}

    def forward(self, image):
        return self.forward_evidence(image)["logits"]
