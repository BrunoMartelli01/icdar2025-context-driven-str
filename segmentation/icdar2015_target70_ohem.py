"""Per-image weighted hard-negative BCE with explicit negative-only behavior."""

from dataclasses import asdict,dataclass
from hashlib import sha256
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from segmentation.icdar2015_target70 import (Target70TrainingConfig,training_config as base_training_config)


@dataclass(frozen=True)
class OHEMTrainingConfig(Target70TrainingConfig):
    def __post_init__(self):
        if self.loss!="masked_OHEM_BCEWithLogitsLoss":raise ValueError("Explicit OHEM BCE required")
        fields=asdict(self);fields["loss"]="masked_BCEWithLogitsLoss"
        Target70TrainingConfig(**fields)


def training_config(config,branch):
    fields=asdict(base_training_config(config,branch));fields["loss"]=config["training"]["loss"]
    return OHEMTrainingConfig(**fields)


class TrainingPixels(Dataset):
    """Keep the identical base tensors; omit unused native owner-map replay."""
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return self.base[index]


def ohem_loss(logits,batch,*,negative_ratio,separation_weight,ignore_negative_weight):
    if not math.isfinite(negative_ratio) or negative_ratio<1:raise ValueError("Hard-negative ratio must be finite and >=1")
    target,valid,separator,ignored=[batch[k].to(logits.device) for k in ("mask","validity","separation","ignored_negative")]
    if any(x.shape!=logits.shape for x in (target,valid,separator,ignored)):raise ValueError("OHEM map shape mismatch")
    if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite OHEM logits")
    if any(torch.any((x!=0)&(x!=1)) for x in (target,valid,separator,ignored)):raise ValueError("OHEM supervision must be binary")
    if (torch.any((target>0)&((valid==0)|(ignored>0)|(separator>0)))
            or torch.any((ignored>0)&(valid==0))):raise ValueError("Invalid OHEM supervision")
    if min(separation_weight,ignore_negative_weight)<1:raise ValueError("Explicit positive source pixel weights required")
    weights=valid*(1+(separation_weight-1)*separator)+(ignore_negative_weight-1)*ignored
    per_pixel=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
    values=[];stats={"positive_pixels":0,"available_negative_pixels":0,"selected_negative_pixels":0}
    for index in range(len(logits)):
        positive=(target[index]>0)&(valid[index]>0);negative=(target[index]==0)&(valid[index]>0)
        pos=per_pixel[index][positive]*weights[index][positive]
        neg=per_pixel[index][negative]*weights[index][negative]
        npos,nneg=pos.numel(),neg.numel()
        if not npos and not nneg:raise ValueError("Image has no valid OHEM pixels")
        count=min(nneg,int(math.floor(negative_ratio*npos))) if npos else nneg
        selected=torch.topk(neg,count,sorted=False).indices if count else None
        numerator=pos.sum();denominator=weights[index][positive].sum()
        if count:
            numerator=numerator+neg[selected].sum();denominator=denominator+weights[index][negative][selected].sum()
        values.append(numerator/denominator)
        stats["positive_pixels"]+=npos;stats["available_negative_pixels"]+=nneg;stats["selected_negative_pixels"]+=count
    return torch.stack(values).mean(),stats


def train_epoch(network,batches,optimizer,config,branch,*,device,max_steps=None):
    network.train();optimizer.zero_grad(set_to_none=True)
    accumulation=branch["gradient_accumulation"];window=window_samples=0;samples=steps=microbatches=0;total=0.;ids=[]
    statistics={"positive_pixels":0,"available_negative_pixels":0,"selected_negative_pixels":0}
    def update():
        nonlocal window,window_samples,steps
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:parameter.grad.div_(window_samples)
        torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],branch["gradient_clip_norm"],error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1;window=window_samples=0
    for batch in batches:
        if max_steps is not None and steps>=max_steps:break
        images=batch["image"].to(device);logits=network(images)
        loss,stats=ohem_loss(logits,batch,negative_ratio=config["training"]["hard_negative_ratio"],
            separation_weight=config["training"]["separation_weight"],ignore_negative_weight=branch["ignore_negative_weight"])
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite OHEM loss")
        count=len(images);(loss*count).backward();window+=1;window_samples+=count
        samples+=count;microbatches+=1;total+=float(loss.detach())*count;ids.extend(batch["sample_id"])
        for key in statistics:statistics[key]+=stats[key]
        if window==accumulation:update()
    if window:update()
    if not samples:raise ValueError("No OHEM training samples")
    return {"loss":total/samples,"samples":samples,"steps":steps,"microbatches":microbatches,
        "sample_order_sha256":sha256("".join(x+"\n" for x in ids).encode()).hexdigest(),"ohem_pixel_counts":statistics}


def word_selection_key(raw,emitted,geometry,epoch):
    return (raw["alphanumeric_casefold"]["f1"],emitted["alphanumeric_casefold"]["f1"],geometry["proposal_f1_iou_50"],-epoch)
