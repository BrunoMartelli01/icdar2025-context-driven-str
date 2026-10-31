import torch

from segmentation.icdar13_matched_training import matched_word_loss
from segmentation.uniform_instance_objective import training_loss


def sample():
    target=torch.zeros(1,1,8,8);target[0,0,1,1]=1;target[0,0,4:7,4:7]=1
    owners=target[:,0].long();owners[:,4:7,4:7]=2
    batch=dict(mask=target,validity=torch.ones_like(target),separation=torch.zeros_like(target),
        kernel=target.clone(),owners=owners)
    config=dict(separation_weight=3.,kernel_loss_weight=1.,source_loss=dict(bce_weight=1.,dice_weight=1.,smooth=1.))
    return batch,config


def test_legacy_loss_and_gradients_are_exact_without_explicit_intervention():
    batch,config=sample();a=torch.randn(1,2,8,8,requires_grad=True);b=a.detach().clone().requires_grad_()
    x=training_loss(a,batch,config);y=matched_word_loss(b,batch,config,'totaltext_bce_dice')
    assert torch.equal(x,y);x.backward();y.backward();assert torch.equal(a.grad,b.grad)


def test_equal_total_positive_gradient_per_owner_despite_different_word_areas():
    batch,config=sample();config['instance_balanced_text']=dict(dice_weight=0.,dice_margin=2,epsilon=1e-6)
    logits=torch.zeros(1,2,8,8,requires_grad=True);training_loss(logits,batch,config).backward()
    gradient=logits.grad[0,0];owners=batch['owners'][0]
    assert torch.allclose(gradient[owners==1].sum(),gradient[owners==2].sum())
    assert gradient[owners==1].sum()<0


def test_kernel_gradient_is_unchanged_and_invalid_pixels_have_zero_gradient():
    batch,config=sample();a=torch.randn(1,2,8,8,requires_grad=True);b=a.detach().clone().requires_grad_()
    matched_word_loss(a,batch,config,'totaltext_bce_dice').backward()
    config['instance_balanced_text']=dict(dice_weight=.5,dice_margin=2,epsilon=1e-6)
    training_loss(b,batch,config).backward();assert torch.equal(a.grad[:,1],b.grad[:,1])
    batch={k:torch.zeros_like(v) for k,v in batch.items()};c=torch.zeros(1,2,8,8,requires_grad=True)
    loss=training_loss(c,batch,config);assert torch.isfinite(loss);loss.backward();assert torch.count_nonzero(c.grad)==0
