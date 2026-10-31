"""Pinned external crop teachers; no guessed checkpoint architecture mapping."""
from __future__ import annotations

import codecs
import argparse
import hashlib
from pathlib import Path
import types

import numpy as np
import torch
from torch.nn import functional as F


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load_ccd(config, device, role='teacher'):
    """Load EVERY tensor in the official CCD-S teacher backbone strictly.

    Only the original DINO projection head is excluded from this feature-only
    teacher. The exact excluded keys and all copied keys are written in audit.
    A single import of an initialization utility is replaced with PyTorch's
    equivalent; all initialized backbone tensors are then overwritten strictly.
    """
    source, checkpoint = Path(config['source']), Path(config['checkpoint'])
    if sha256(source)!=config['source_sha256'] or sha256(checkpoint)!=config['checkpoint_sha256']:
        raise ValueError('CCD source/checkpoint SHA256 mismatch')
    original=source.read_text()
    old='from Dino.modules.utils import trunc_normal_'
    if original.count(old)!=1:
        raise ValueError('Unexpected pinned CCD import')
    patched=original.replace(old,'from torch.nn.init import trunc_normal_')
    module=types.ModuleType('pinned_ccd_vit')
    exec(compile(patched,str(source),'exec'),module.__dict__)
    # Narrow numeric allowlist for the official checkpoint's optimizer metadata;
    # no arbitrary classes or custom model objects are unpickled.
    torch.serialization.add_safe_globals([np.core.multiarray.scalar,np.dtype,
                                          type(np.dtype('float64')),codecs.encode])
    checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    source_state=checkpoint_data[role]
    prefix='module.backbone.'
    state={k[len(prefix):]:v for k,v in source_state.items() if k.startswith(prefix)}
    excluded=sorted(k for k in source_state if not k.startswith(prefix))
    if any(not k.startswith(('module.head.','module.segmentation.')) for k in excluded):
        raise ValueError('Unrecognized excluded CCD tensor')
    model=module.vit_small(patch_size=4,img_size=[32,128])
    model.load_state_dict(state,strict=True)
    model.eval().to(device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    audit={'source_revision':config['revision'],'source_sha256':sha256(source),
           'checkpoint_sha256':sha256(checkpoint),'checkpoint_epoch':checkpoint_data['epoch'],
           'strict_backbone_load':True,'copied_keys':sorted(state),'excluded_projection_keys':excluded,
           'engineering_patch':'trunc_normal_ import only; all backbone state strictly overwritten',
           'role':'frozen official CCD-S crop feature teacher; primary reader remains full RGB PARSeq'}
    return model,audit


def load_ccd_glyph(config,device):
    backbone,audit=load_ccd(config,device,role='student')
    source=Path(config['segmentation_source'])
    if sha256(source)!=config['segmentation_source_sha256']:raise ValueError('CCD segmentation source changed')
    module=types.ModuleType('pinned_ccd_segmentation');exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    model=module.SegHead(in_channels=384,mla_channels=128,mlahead_channels=64,num_classes=2)
    data=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)['student']
    prefix='module.segmentation.';state={k[len(prefix):]:v for k,v in data.items() if k.startswith(prefix)}
    model.load_state_dict(state,strict=True);model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    return backbone,model,{'backbone':audit,'segmentation_strict_load':True,'segmentation_keys':sorted(state),
                           'source_sha256':sha256(source),'role':'frozen CCD student glyph pseudo-targets for SIGA-inspired auxiliary supervision'}


def load_mae(config,device):
    source=Path(config['source']);checkpoint=Path(config['checkpoint'])
    if sha256(source)!=config['source_sha256'] or sha256(checkpoint)!=config['checkpoint_sha256']:
        raise ValueError('MAERec teacher source/checkpoint changed')
    code=source.read_text();old='from mmocr.registry import MODELS'
    if code.count(old)!=1:raise ValueError('Unexpected MAERec registry import')
    # The registry only decorates these classes; no MMOCR pipeline is imported.
    replacement='class _Registry:\n    def register_module(self):\n        return lambda cls: cls\nMODELS = _Registry()'
    module=types.ModuleType('pinned_maerec_vit');exec(compile(code.replace(old,replacement),str(source),'exec'),module.__dict__)
    torch.serialization.add_safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),codecs.encode,argparse.Namespace])
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    model=module.VisionTransformer(img_size=(32,128),patch_size=(4,4),embed_dim=384,depth=12,num_heads=6,
                                    mlp_ratio=4.,qkv_bias=True,pretrained=None)
    keys=set(model.state_dict());state={k:v for k,v in data.items() if k in keys};excluded=sorted(set(data)-keys)
    if any(not k.startswith(('decoder_','mask_token')) for k in excluded):raise ValueError('Unexpected excluded MAE parameter')
    model.load_state_dict(state,strict=True);model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    return model,{'strict_encoder_load':True,'copied_keys':sorted(state),'excluded_pretraining_decoder_keys':excluded,
                  'source_sha256':sha256(source),'checkpoint_sha256':sha256(checkpoint),'revision':config['revision'],
                  'engineering_patch':'registry decorator only; upstream pretrained/strict=False path is not executed',
                  'role':'Union14M MAE pretrained ViT-S crop teacher; remove CLS token; PARSeq remains primary reader'}


def character_vectors(features, lengths):
    """Pool column-aligned features to transcription-length character bins.

    Uniform horizontal bins are an explicit APPROXIMATE alignment assumption,
    not the original CCD learned character masks or SIGA architecture.
    """
    if features.ndim==3:
        features=features.transpose(1,2).reshape(len(features),-1,8,32)
    values=[]
    for item,length in zip(features,lengths):
        if length:
            # Explicit means avoid nondeterministic CUDA adaptive-pool backward.
            columns=item.mean(-2)
            width=columns.shape[-1]
            bins=[columns[:,(i*width)//length:((i+1)*width+length-1)//length].mean(-1)
                  for i in range(length)]
            values.append(torch.stack(bins))
    return torch.cat(values) if values else features.new_empty((0,features.shape[1]))
