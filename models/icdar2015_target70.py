"""Approximate ICDAR U-Net continuation and explicit partial-encoder gradients."""

import torch

from models.icdar2015_improvements import ImprovementAGUNet


class Target70AGUNet(ImprovementAGUNet):
    def __init__(self, *, encoder_stages, skip_floor, max_groups, source_state=None):
        if tuple(encoder_stages) not in ((), ("layer3", "layer4")):
            raise ValueError("Explicit frozen or layer3/layer4 encoder setting required")
        super().__init__(normalization="batchnorm",max_groups=max_groups,skip_floor=skip_floor,source_state=source_state)
        self.encoder_stages=tuple(encoder_stages)
        for name in self.encoder_stages:self.encoder.get_submodule(name).requires_grad_(True)
        for module in self.encoder.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.requires_grad_(False)

    @property
    def architecture_identity(self):
        return "approximate-icdar2015-target70-agunet-v1:"+("partial_encoder" if getattr(self,"encoder_stages",()) else "frozen_encoder")

    def _encode(self,x):
        # Parent _encode uses no_grad for the whole encoder; overriding it is
        # essential. Merely changing requires_grad would not fine-tune anything.
        with torch.no_grad():
            e0=self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
            e1=self.encoder.layer1(self.encoder.maxpool(e0))
            e2=self.encoder.layer2(e1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and bool(self.encoder_stages)):
            e3=self.encoder.layer3(e2);e4=self.encoder.layer4(e3)
        return [e0,e1,e2,e3,e4]


def restored_optimizer(network, payload, config, branch):
    t=config["training"]
    decoder=[p for name,p in network.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
    encoder=[p for name,p in network.named_parameters() if p.requires_grad and name.startswith("encoder.")]
    optimizer=torch.optim.Adam(decoder,lr=t["learning_rate"],weight_decay=t["weight_decay"],
        betas=tuple(t["adam_betas"]),eps=t["adam_epsilon"])
    # Decoder parameter ordering is unchanged. Restore its complete state before
    # explicitly adding previously frozen encoder parameters with fresh moments.
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if len(optimizer.param_groups)!=1:raise ValueError("Source decoder optimizer must have exactly one group")
    group=optimizer.param_groups[0]
    if (group["lr"]!=t["learning_rate"] or group["weight_decay"]!=t["weight_decay"]
            or tuple(group["betas"])!=tuple(t["adam_betas"]) or group["eps"]!=t["adam_epsilon"]):
        raise ValueError("Source Adam settings differ from the qualified continuation")
    group.update(group_name="decoder",initial_lr=t["learning_rate"])
    if encoder:
        optimizer.add_param_group({"params":encoder,"lr":branch["encoder_learning_rate"],
            "initial_lr":branch["encoder_learning_rate"],"group_name":"encoder_layer3_layer4"})
    return optimizer
