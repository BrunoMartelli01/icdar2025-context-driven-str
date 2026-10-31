"""Research word-box field on the complete attention U-Net decoder.

Dense positive word regions supervise four boundary distances. Learned subcell
offsets retain words smaller than a feature cell; this is not an exact FCOS/EAST
implementation. No detector or recognition transformer enters this backbone.
"""

import math

import torch
from torch import nn


class WordBoxField(nn.Module):
    def __init__(self, *, input_channels, width, groups, dilations,
                 initial_probability, initial_distances):
        super().__init__()
        if width % groups or not 0 < initial_probability < 1:
            raise ValueError('Explicit valid field architecture required')
        if len(initial_distances) != 4 or min(initial_distances) <= 0:
            raise ValueError('Four positive initial normalized boundary distances required')
        layers = []
        for dilation in dilations:
            layers.extend([nn.Conv2d(input_channels, width, 3, padding=dilation,
                dilation=dilation, bias=False), nn.GroupNorm(groups, width), nn.GELU()])
            input_channels = width
        self.tower = nn.Sequential(*layers)
        self.classification = nn.Conv2d(width, 1, 1)
        self.centerness = nn.Conv2d(width, 1, 1)
        self.distances = nn.Conv2d(width, 4, 1)
        self.offsets = nn.Conv2d(width, 2, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.classification.bias,
                          math.log(initial_probability/(1-initial_probability)))
        with torch.no_grad():
            self.distances.bias.copy_(torch.tensor(initial_distances).expm1().log())

    def forward(self, features):
        hidden = self.tower(features)
        return dict(class_logits=self.classification(hidden),
            center_logits=self.centerness(hidden), raw_distances=self.distances(hidden),
            offset_logits=self.offsets(hidden))


class WordBoxFieldUNet(nn.Module):
    def __init__(self, backbone, field, *, train_decoder):
        super().__init__()
        self.backbone, self.field = backbone, field
        self.set_decoder_trainable(train_decoder)

    def set_decoder_trainable(self, enabled):
        self.train_decoder = bool(enabled)
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(self.train_decoder and not name.startswith('encoder.'))
        self.backbone.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.encoder.eval()
        if not self.train_decoder:
            self.backbone.eval()
        return self

    def forward(self, image):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_decoder):
            evidence = self.backbone.forward_evidence(image)
            features = torch.cat((evidence['features'], evidence['words'].sigmoid()), 1)
        result = self.field(features)
        result['segmentation'] = evidence['words']
        return result
