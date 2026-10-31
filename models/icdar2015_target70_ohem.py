"""Approximate continuation: explicit full feature-encoder gradients, frozen BN."""

import torch

from models.icdar2015_target70 import Target70AGUNet


class OHEMAGUNet(Target70AGUNet):
    def __init__(self,*,full_encoder,skip_floor,max_groups,source_state=None):
        if not isinstance(full_encoder,bool):raise ValueError("Explicit full encoder boolean required")
        super().__init__(encoder_stages=["layer3","layer4"],skip_floor=skip_floor,max_groups=max_groups,source_state=source_state)
        self.full_encoder=full_encoder
        if full_encoder:
            for name in ("conv1","layer1","layer2"):self.encoder.get_submodule(name).requires_grad_(True)
        for module in self.encoder.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.requires_grad_(False)

    @property
    def architecture_identity(self):
        return "approximate-icdar2015-target70-ohem-v1:"+("full_encoder" if getattr(self,"full_encoder",False) else "partial_encoder")

    def _encode(self,x):
        if not self.full_encoder:return super()._encode(x)
        e0=self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e1=self.encoder.layer1(self.encoder.maxpool(e0));e2=self.encoder.layer2(e1)
        e3=self.encoder.layer3(e2);e4=self.encoder.layer4(e3)
        return [e0,e1,e2,e3,e4]


def restored_optimizer(network,payload,config,branch):
    t=config["training"]
    decoder=[p for name,p in network.named_parameters() if p.requires_grad and not name.startswith("encoder.")]
    deep=[p for name,p in network.named_parameters() if p.requires_grad and name.startswith(("encoder.layer3.","encoder.layer4."))]
    early=[p for name,p in network.named_parameters() if p.requires_grad and name.startswith("encoder.") and not name.startswith(("encoder.layer3.","encoder.layer4."))]
    old=payload["optimizer_state_dict"]
    if len(old["param_groups"])!=2 or [g["group_name"] for g in old["param_groups"]]!=["decoder","encoder_layer3_layer4"]:
        raise ValueError("Qualified decoder/deep-encoder Adam groups required")
    for group in old["param_groups"]:
        if (group["weight_decay"]!=t["weight_decay"] or tuple(group["betas"])!=tuple(t["adam_betas"])
                or group["eps"]!=t["adam_epsilon"]):raise ValueError("Source Adam settings changed")
    optimizer=torch.optim.Adam([{"params":decoder},{"params":deep}],lr=t["learning_rate"],
        weight_decay=t["weight_decay"],betas=tuple(t["adam_betas"]),eps=t["adam_epsilon"])
    optimizer.load_state_dict(old)
    # Restart only the declared LR schedule; retain every inherited moment/step.
    for group,name,rate in zip(optimizer.param_groups,("decoder","encoder_layer3_layer4"),
            (t["learning_rate"],branch["encoder_learning_rate"])):
        group.update(group_name=name,lr=rate,initial_lr=rate)
    if early:
        optimizer.add_param_group({"params":early,"group_name":"encoder_early",
            "lr":branch["early_encoder_learning_rate"],"initial_lr":branch["early_encoder_learning_rate"]})
    return optimizer
