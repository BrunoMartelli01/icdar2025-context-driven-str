"""Approximate ICDAR2013 full-word / kernel transfer with qualified geometry."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from models.adaspot_gap import AdaSpotGapAGUNet
from segmentation.icdar2015_improvements import make_transform
from segmentation.totaltext_word_kernel import native_owner_map,make_word_kernels
from segmentation.training import separation_weighted_bce_with_logits
from scripts.train_totaltext_letterbox_word_f1_separator_head import balanced_separator_bce


class WordRefitUNet(AdaSpotGapAGUNet):
    """Exactly the source state keys; only the declared encoder gradient regime changes."""
    def __init__(self,*,partial_encoder,skip_floor,initial_gap_bias):
        super().__init__('control',skip_floor=skip_floor,initial_gap_bias=initial_gap_bias)
        self.partial_encoder=bool(partial_encoder)
        if self.partial_encoder:
            self.encoder.layer3.requires_grad_(True);self.encoder.layer4.requires_grad_(True)
        for module in self.encoder.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.requires_grad_(False)

    @property
    def architecture_identity(self):
        return 'approximate-icdar13-word-kernel-refit-v1:'+('partial' if self.partial_encoder else 'frozen')

    def _encode(self,x):
        with torch.no_grad():
            e0=self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
            e1=self.encoder.layer1(self.encoder.maxpool(e0));e2=self.encoder.layer2(e1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.partial_encoder):
            e3=self.encoder.layer3(e2);e4=self.encoder.layer4(e3)
        return [e0,e1,e2,e3,e4]


class WordDataset(Dataset):
    def __init__(self,records,config,*,augment):
        self.records=records;self.config=config
        self.transform=make_transform(config,config['image_hw'],augment=augment)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(r['image_path']) as p:rgb=p.convert('RGB')
        images={}
        for name in ['mask','validity','separator','owners']:
            with Image.open(r[name+'_path']) as p:images[name]=p.convert('I' if name=='owners' else 'L')
        x,y,v,s=self.transform.transform_with_validity_and_separation(rgb,images['mask'],images['validity'],images['separator'],images['owners'],index)
        owners=native_owner_map(images['owners'],self.transform,index,y,v)
        kernel,_=make_word_kernels(owners,distance_fraction=self.config['kernel_distance_fraction'])
        return dict(image=x,mask=y,validity=v,separation=s,kernel=torch.from_numpy((kernel>0).astype(np.float32))[None],
            owners=torch.from_numpy(owners),sample_id=r['sample_id'])


def word_loss(logits,batch,config):
    target,valid,separator,kernel=[batch[k].to(logits.device) for k in ['mask','validity','separation','kernel']]
    if logits.shape!=(len(target),2,*target.shape[-2:]) or torch.any(kernel>target) or torch.any(kernel>valid):
        raise ValueError('Real full-word and kernel supervision required')
    full=separation_weighted_bce_with_logits(logits[:,:1],target,valid,separator,separation_weight=config['separation_weight'])
    core=balanced_separator_bce(logits[:,1:2],kernel,valid)
    return full+config['kernel_loss_weight']*core


def strict_fallback_count(total,maximum_fraction):
    """Largest integer strictly below the user's maximum fallback fraction."""
    from fractions import Fraction
    r=Fraction(str(maximum_fraction))
    if total<1 or not 0<r<1:raise ValueError('Positive population and fraction in(0,1) required')
    return (r.numerator*total-1)//r.denominator


def padded_rows(rows,image_wh,padding):
    if padding<0:raise ValueError('Nonnegative padding required')
    out=[]
    for r in rows:
        a,b,c,d=r['bbox'];box=[max(0,a-padding),max(0,b-padding),min(image_wh[0],c+padding),min(image_wh[1],d+padding)]
        out.append(dict(r,bbox=box))
    return out
