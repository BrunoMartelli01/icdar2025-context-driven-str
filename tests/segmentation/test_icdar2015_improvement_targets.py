"""Word-mask separation and geometry across explicit ICDAR2015 resolutions."""

from pathlib import Path
import json

import numpy as np
from PIL import Image
import pytest
import torch

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.icdar2015_transfer import make_targets, load_config, transform as old_transform
from segmentation.icdar2015_improvements import (
    ImprovementTransformConfig, make_transform, native_target_audit_maps,
    padding_geometry, remap_native_box, instance_balanced_loss,
)


ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/"configs/approximate/icdar2015_unet_improvement_v1.json").read_text())


def inputs():
    rgb=Image.new("RGB",(320,180),(124,116,104))
    words=tuple(SceneTextInstance(points,"word",False,"word") for points in (
        ((20,40),(100,40),(100,80),(20,80)), ((100,40),(180,40),(180,80),(100,80))))
    mask,valid,sep,owner,_=make_targets(rgb.size,words)
    return rgb,mask,valid,sep,owner.convert("I")


@pytest.mark.parametrize("augment",[False,True])
def test_square512_matches_qualified_transform_exactly(augment):
    previous=old_transform(load_config(ROOT/"configs/approximate/icdar2015_totaltext_transfer_v1.json"),augment=augment)
    current=make_transform(CONFIG,[512,512],augment=augment)
    previous.set_epoch(5);current.set_epoch(5)
    expected=previous.transform_with_validity_and_separation(*inputs(),9)
    actual=current.transform_with_validity_and_separation(*inputs(),9)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))


@pytest.mark.parametrize("shape",[(512,512),(768,768),(1024,1024),(576,1024)])
def test_independent_audit_replays_native_guard_and_aspect(shape):
    rgb,mask,valid,sep,owner=inputs()
    tx=make_transform(CONFIG,shape,augment=False)
    _,target,valid_t,separator=tx.transform_with_validity_and_separation(rgb,mask,valid,sep,owner,0)
    raw,final,ids,valid_np=native_target_audit_maps(mask,valid,sep,owner,shape,threshold=.5)
    assert np.array_equal(final,target[0].numpy().astype(bool))
    assert np.array_equal(valid_np,valid_t[0].numpy().astype(bool))
    assert not np.any(final & separator[0].numpy().astype(bool))
    assert set(np.unique(ids[final]))=={1,2}
    assert target.shape==(1,shape[0]//2,shape[1]//2)


def test_rectangular_geometry_preserves_original_pixels_and_round_trip():
    assert padding_geometry((1280,720),(576,1024))==(1280,720,0,0)
    assert remap_native_box([40,20,80,40],(1280,720),(576,1024))==[100,50,200,100]
    assert remap_native_box([20,60,40,80],(1280,720),(512,512))==[100,20,200,120]
    with pytest.raises(ValueError):
        ImprovementTransformConfig((512,512),(128,128),0,0,(-15,15))
    with pytest.raises(ValueError,match="approximate"):
        ImprovementTransformConfig((512,512),(256,256),0,0,(-15,15),reproduction_mode="strict")


def test_each_word_gets_equal_positive_bce_gradient_despite_area():
    logits=torch.zeros(1,1,3,6,requires_grad=True)
    owners=torch.zeros(1,3,6,dtype=torch.int64);owners[0,0,0]=1;owners[0,1,:4]=2
    target=(owners>0).float()[:,None];valid=torch.ones_like(target);sep=torch.zeros_like(target)
    loss=instance_balanced_loss(logits,target,valid,sep,owners,separation_weight=3,
        dice_weight=0,dice_margin=2,epsilon=1e-6)
    loss.backward()
    assert torch.allclose(logits.grad[0,0][owners[0]==1].sum(),logits.grad[0,0][owners[0]==2].sum())


def test_padding_and_ignore_have_zero_loss_gradient_with_local_dice():
    logits=torch.zeros(1,1,4,6,requires_grad=True);owners=torch.zeros(1,4,6,dtype=torch.int64)
    owners[0,1,1]=1;target=(owners>0).float()[:,None];valid=torch.ones_like(target);valid[:,:,0,:]=0
    loss=instance_balanced_loss(logits,target,valid,torch.zeros_like(target),owners,
        separation_weight=3,dice_weight=.5,dice_margin=2,epsilon=1e-6)
    loss.backward();assert torch.isfinite(loss);assert not logits.grad[:,:,0,:].any()
