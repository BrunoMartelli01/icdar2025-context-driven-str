import copy

import pytest
import torch

from models.unet_auxiliary_ocr import AuxiliaryCTCHead, axis_feature_crops
from segmentation.unet_auxiliary_ocr import auxiliary_loss


def test_axis_coordinates_identity_and_partial_box():
    feature=torch.arange(20,dtype=torch.float32).reshape(1,1,4,5).requires_grad_()
    actual=axis_feature_crops(feature,[[0,0,5,4]],output_hw=[4,5],detach=False)
    torch.testing.assert_close(actual,feature)
    subset=axis_feature_crops(feature,[[1,1,4,3]],output_hw=[2,3],detach=False)
    torch.testing.assert_close(subset,feature[:,:,1:3,1:4])
    subset.sum().backward()
    expected=torch.zeros_like(feature);expected[:,:,1:3,1:4]=1
    torch.testing.assert_close(feature.grad,expected)


@pytest.mark.parametrize('boxes',[[[1,1,1,2]], [[-1,0,3,2]], [[0,0,6,2]], [[0,0,float('nan'),2]],[]])
def test_bad_boxes_fail(boxes):
    with pytest.raises(ValueError):
        axis_feature_crops(torch.ones(1,2,4,5),boxes,output_hw=[2,3],detach=False)


def test_detach_changes_only_gradient_not_sampled_values():
    f=torch.randn(1,3,9,11,requires_grad=True)
    connected=axis_feature_crops(f,[[.2,.3,8.7,7.2]],output_hw=[4,8],detach=False)
    detached=axis_feature_crops(f,[[.2,.3,8.7,7.2]],output_hw=[4,8],detach=True)
    assert torch.equal(connected,detached)
    assert connected.requires_grad and not detached.requires_grad


def test_ctc_trains_reader_in_both_arms_but_shared_features_only_when_connected():
    torch.manual_seed(32)
    features=torch.randn(1,4,8,12,requires_grad=True)
    a=AuxiliaryCTCHead(5,channels=4,hidden=4);b=copy.deepcopy(a)
    rows=[[{'bbox':[1,1,11,7],'tokens':[1,2,2,3]}]]
    off=auxiliary_loss(features,a,rows,output_hw=[4,16],detach=True)
    on=auxiliary_loss(features,b,rows,output_hw=[4,16],detach=False)
    torch.testing.assert_close(off,on,rtol=0,atol=0)
    assert torch.autograd.grad(off,features,allow_unused=True,retain_graph=True)[0] is None
    grad=torch.autograd.grad(on,features,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    off.backward();on.backward()
    for p,q in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(p.grad,q.grad)


def test_no_ocr_supervision_has_no_feature_gradient():
    f=torch.randn(1,4,8,12,requires_grad=True)
    reader=AuxiliaryCTCHead(4,channels=4,hidden=4)
    loss=auxiliary_loss(f,reader,[[]],output_hw=[4,16],detach=False)
    assert loss.item()==0
    assert torch.autograd.grad(loss,f,allow_unused=True)[0] is None


def test_detached_auxiliary_preserves_segmentation_optimizer_update():
    from segmentation.unet_auxiliary_ocr import segmentation_objective
    torch.manual_seed(37)
    trunk=torch.nn.Conv2d(3,4,3,padding=1);word_head=torch.nn.Conv2d(4,2,1)
    paired=copy.deepcopy(torch.nn.ModuleList([trunk,word_head]))
    reader=AuxiliaryCTCHead(4,channels=4,hidden=4)
    x=torch.randn(1,3,6,9);target=torch.zeros(1,1,6,9);target[:,:,1:5,2:7]=1
    batch=dict(mask=target,kernel=target,validity=torch.ones_like(target),separation=torch.zeros_like(target))
    config=dict(lambda_text=1.,lambda_kernel=1.,bce_weight=1.,dice_weight=1.,smooth=1.,separation_weight=3.)
    opt=torch.optim.AdamW(list(trunk.parameters())+list(word_head.parameters())+list(reader.parameters()),lr=1e-3)
    reference=torch.optim.AdamW(paired.parameters(),lr=1e-3)
    features=trunk(x);loss=segmentation_objective(word_head(features),batch,config)
    aux=auxiliary_loss(features,reader,[[dict(bbox=[1,1,8,5],tokens=[1,2])]],output_hw=[4,12],detach=True)
    (loss+.1*aux).backward();opt.step()
    control_loss=segmentation_objective(paired[1](paired[0](x)),batch,config)
    control_loss.backward();reference.step()
    for a,b in zip(list(trunk.parameters())+list(word_head.parameters()),paired.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
