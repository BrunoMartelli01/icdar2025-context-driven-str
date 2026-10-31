"""Approximate jointly trainable attention U-Net and dense word instances."""

import torch
from torch import nn
from torch.nn import functional as F


class JointWordInstanceUNet(nn.Module):
    def __init__(self, backbone, instance_head, *, train_decoder):
        super().__init__()
        self.backbone = backbone
        self.instance_head = instance_head
        self.train_decoder = bool(train_decoder)
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
            features = torch.cat((F.avg_pool2d(evidence['features'], 2),
                                  F.avg_pool2d(evidence['words'].sigmoid(), 2)), 1)
        result = self.instance_head(features)
        result['segmentation'] = evidence['words']
        return result
