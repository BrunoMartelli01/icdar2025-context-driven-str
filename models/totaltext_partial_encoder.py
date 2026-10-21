"""Isolated encoder fine-tuning ablation, never used by the author protocol."""

import torch

from models.unet_adv import PracticalTotalTextHighResAGUNet


class PracticalTotalTextPartialEncoderAGUNet(PracticalTotalTextHighResAGUNet):
    def __init__(self, *, branch, encoder_pretrained=False):
        if branch not in {"frozen_control", "partial_encoder"}:
            raise ValueError("Unknown partial-encoder branch")
        super().__init__(skip_floor=0.25, encoder_pretrained=encoder_pretrained)
        self.branch = branch
        if branch == "partial_encoder":
            self.encoder.layer3.requires_grad_(True)
            self.encoder.layer4.requires_grad_(True)
        # Parent train() leaves the ENTIRE encoder in eval mode. Convolutional
        # parameters still receive gradients; all BN affine/statistics stay fixed.
        for module in self.encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(False)

    @property
    def architecture_identity(self):
        return "practical-totaltext-partial-encoder-v1:" + self.branch

    def _encode(self, x):
        with torch.no_grad():
            e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
            e1 = self.encoder.layer1(self.encoder.maxpool(e0))
            e2 = self.encoder.layer2(e1)
        if self.branch == "partial_encoder":
            e3 = self.encoder.layer3(e2)
            e4 = self.encoder.layer4(e3)
        else:
            with torch.no_grad():
                e3 = self.encoder.layer3(e2)
                e4 = self.encoder.layer4(e3)
        return [e0, e1, e2, e3, e4]


def build_partial_encoder_optimizer(model, training_config, encoder_lr=1e-6):
    """Fresh Adam on both branches; no incompatible source optimizer transfer."""
    decoder = [p for name, p in model.named_parameters()
               if p.requires_grad and not name.startswith("encoder.")]
    encoder = [p for name, p in model.named_parameters()
               if p.requires_grad and name.startswith("encoder.")]
    groups = [{"params": decoder, "lr": training_config.learning_rate,
               "group_name": "decoder"}]
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr,
                       "group_name": "encoder_layer3_layer4"})
    return torch.optim.Adam(groups, weight_decay=training_config.weight_decay,
                            betas=training_config.adam_betas,
                            eps=training_config.adam_epsilon)
