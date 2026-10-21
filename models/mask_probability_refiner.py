"""Small identity-initialized Text/Kernel correctors (research extensions).

Inspired by global/context and iterative refinement literature, not implementations
of CascadePSP, PointRend or SegRefiner. No targets enter this inference interface.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing, dilation=1):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=dilation, dilation=dilation),
                         nn.GroupNorm(4, outgoing), nn.ReLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1),
                         nn.GroupNorm(4, outgoing), nn.ReLU())


class TinyContextUNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoders = nn.ModuleList([Block(8, width), Block(width, width*2),
                                      Block(width*2, width*4), Block(width*4, width*8)])
        self.context = nn.Conv2d(width*8, width*8, 1)
        self.decoders = nn.ModuleList([Block(width*12, width*4), Block(width*6, width*2),
                                      Block(width*3, width)])

    def forward(self, x):
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = x + self.context(x.mean((-2, -1), keepdim=True))
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            # Exact integer nearest expansion supports deterministic CUDA backward.
            x = decoder(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode="nearest"), skip), 1))
        return x


class MaskProbabilityRefiner(nn.Module):
    def __init__(self, architecture, *, width=16, steps=1, text_threshold=.65,
                 kernel_threshold=.5, epsilon=1e-4, max_logit_delta=12.):
        super().__init__()
        if architecture not in ("shallow", "dilated", "unet") or width < 4 or width % 4 or steps < 1:
            raise ValueError("Unsupported refiner architecture/width/steps")
        self.architecture, self.steps = architecture, steps
        self.thresholds = (text_threshold, kernel_threshold)
        self.epsilon, self.max_logit_delta = epsilon, max_logit_delta
        if not 0 < epsilon < .5 or max_logit_delta <= 0 or any(not 0 < t < 1 for t in self.thresholds):
            raise ValueError("Invalid numerical/probability contract")
        if architecture == "unet":
            self.features = TinyContextUNet(width)
        else:
            rates = [1, 1, 1] if architecture == "shallow" else [1, 2, 4, 8, 16]
            layers = []
            for index, rate in enumerate(rates):
                layers += [nn.Conv2d(8 if index == 0 else width, width, 3,
                                     padding=rate, dilation=rate), nn.ReLU()]
            self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb, probabilities, canvas):
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or probabilities.shape != (len(rgb), 2, *rgb.shape[-2:])
                or canvas.shape != (len(rgb), 1, *rgb.shape[-2:])
                or rgb.shape[-1] % 8 or rgb.shape[-2] % 8):
            raise ValueError("Expected aligned Bx3 RGB, Bx2 probabilities, Bx1 canvas, multiples of8")
        current, outputs = probabilities, []
        thresholds = current.new_tensor(self.thresholds)[None, :, None, None]
        for _ in range(self.steps):
            binary = (current >= thresholds).to(current.dtype)
            features = self.features(torch.cat((rgb, current, binary, canvas.to(rgb.dtype)), 1))
            delta = self.max_logit_delta * torch.tanh(self.head(features)/self.max_logit_delta)
            reference_logits = torch.logit(current.clamp(self.epsilon, 1-self.epsilon))
            logits = reference_logits + delta
            # Parenthesized difference is EXACTLY zero at delta=0, including p=0/1.
            # Autograd remains nonzero there, unlike a special-case off switch.
            current = (current + (logits.sigmoid()-reference_logits.sigmoid())).clamp(0, 1)
            outputs.append({"probabilities": current, "logits": logits, "delta": delta})
        return outputs
