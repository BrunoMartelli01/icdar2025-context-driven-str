"""Isolated APPROXIMATE ResNet50 geometric proposal decoder.

The direct multi-scale fusion contains no U-Net decoder or attention skip path.
All encoder tensors must come from a declared checkpoint, with strict loading.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


def deterministic_resize(value, size):
    """CPU bilinear backward is an explicit HARDWARE_DEVIATION, with gradients."""
    device = value.device
    return F.interpolate(value.cpu(), size=size, mode='bilinear', align_corners=False).to(device)


class GeometricProposalDecoder(nn.Module):
    """Shared frozen ResNet50 -> direct lateral fusion -> text/kernel/geometry."""
    architecture_identity = 'approximate-resnet50-direct-geometry-decoder-v1'
    requires_unet = False

    def __init__(self, encoder_state, *, channels, native_size):
        super().__init__()
        if channels < 4 or native_size < 16:
            raise ValueError('Invalid explicit geometry dimensions')
        self.native_size = native_size
        self.encoder = resnet50(weights=None)
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.lateral = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in (64, 256, 512, 1024, 2048))
        self.fusion = nn.Sequential(nn.Conv2d(5*channels, 64, 3, padding=2, dilation=2),
                                    nn.GroupNorm(8,64), nn.ReLU(),
                                    nn.Conv2d(64,64,3,padding=2,dilation=2), nn.GroupNorm(8,64), nn.ReLU())
        self.word = nn.Conv2d(64, 2, 1)
        self.geometry = nn.Conv2d(64, 3, 1)

    def train(self, mode=True):
        super().train(mode); self.encoder.eval(); return self

    def forward_evidence(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[-2:] != (2*self.native_size, 2*self.native_size):
            raise ValueError('Explicit square RGB/native coordinate contract violated')
        with torch.no_grad():
            a=self.encoder.relu(self.encoder.bn1(self.encoder.conv1(image)))
            b=self.encoder.layer1(self.encoder.maxpool(a)); c=self.encoder.layer2(b)
            d=self.encoder.layer3(c); e=self.encoder.layer4(d)
        size=(self.native_size//2,)*2
        # Frozen-feature resize needs no gradient. Lateral projections then learn.
        fused=[projection(F.interpolate(x,size=size,mode='bilinear',align_corners=False))
               for projection,x in zip(self.lateral,(a,b,c,d,e))]
        feature=deterministic_resize(self.fusion(torch.cat(fused,1)),(self.native_size,)*2)
        return dict(words=self.word(feature), geometry=self.geometry(feature), features=feature)

    def forward(self, image):
        return self.forward_evidence(image)['words']


class UNetGeometryControl(nn.Module):
    """Fresh qualified U-Net decoder, same frozen encoder and dense objectives.

    The supplied model must be newly initialized, NOT warm-started on word masks.
    Only the encoder is loaded; the complete resulting state loads strictly.
    """
    architecture_identity = 'approximate-fresh-unet-geometry-control-v1'
    requires_unet = True

    def __init__(self, fresh_unet, encoder_state):
        super().__init__()
        fresh_unet.encoder.load_state_dict(encoder_state, strict=True)
        self.network=fresh_unet
        self.geometry=nn.Conv2d(64,3,1)

    def forward_evidence(self, image):
        evidence=self.network.forward_evidence(image)
        return {**evidence,'geometry':self.geometry(evidence['features'])}

    def forward(self,image):
        return self.forward_evidence(image)['words']
