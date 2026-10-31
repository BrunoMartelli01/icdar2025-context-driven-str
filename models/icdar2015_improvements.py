"""Approximate U-Net norm ablations; keep historical architectures byte-identical."""

import torch
from torch import nn

from models.icdar2015_unet import PracticalICDAR2015AGUNet


class ImprovementAGUNet(PracticalICDAR2015AGUNet):
    def __init__(self, *, normalization, max_groups, skip_floor, source_state=None):
        if normalization not in {"batchnorm", "groupnorm"}:
            raise ValueError("Explicit batchnorm/groupnorm variant required")
        if max_groups < 1:
            raise ValueError("Positive GroupNorm group limit required")
        super().__init__(encoder_pretrained=False, skip_floor=skip_floor)
        # Load the complete historical model before an explicit norm migration.
        # Never use strict=False or ignore unexpected checkpoint tensors.
        if source_state is not None:
            self.load_state_dict(source_state, strict=True)
        self.normalization_variant = normalization
        self.normalization_migration = []
        if normalization == "groupnorm":
            for name, module in list(self.named_modules()):
                if name.startswith("encoder.") or not isinstance(module, nn.BatchNorm2d):
                    continue
                groups = max(g for g in range(1, min(max_groups, module.num_features)+1)
                             if module.num_features % g == 0)
                replacement = nn.GroupNorm(groups, module.num_features, eps=module.eps, affine=True)
                with torch.no_grad():
                    replacement.weight.copy_(module.weight)
                    replacement.bias.copy_(module.bias)
                parent_name, child = name.rsplit(".", 1)
                setattr(self.get_submodule(parent_name), child, replacement)
                self.normalization_migration.append({"name": name, "channels": module.num_features,
                    "groups": groups, "affine_parameters_copied": True,
                    "removed_buffers": ["running_mean", "running_var", "num_batches_tracked"]})
            if len(self.normalization_migration) != 21:
                raise RuntimeError("Unexpected number of decoder/bottleneck norm conversions")

    @property
    def architecture_identity(self):
        return "approximate-icdar2015-word-improvement-v1-" + self.normalization_variant
