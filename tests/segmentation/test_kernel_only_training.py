import torch
import pytest
from segmentation.kernel_only_training import kernel_loss,train_kernel_epoch

OPTIONS={'bce_weight':1.,'dice_weight':1.,'smooth':1.}


def test_only_kernel_and_validity_affect_loss_and_gradients():
    logits=torch.zeros(1,1,3,3,requires_grad=True)
    batch={'kernel':torch.ones_like(logits),'validity':torch.ones_like(logits),'mask':torch.zeros_like(logits)}
    batch['validity'][:,:,0,0]=0
    loss=kernel_loss(logits,batch,OPTIONS);gradient=torch.autograd.grad(loss,logits)[0]
    changed={**batch,'mask':torch.full_like(logits,float('nan')),'separation':torch.full_like(logits,float('inf')),'gap_target':torch.randn_like(logits)}
    other=kernel_loss(logits,changed,OPTIONS)
    assert torch.equal(loss,other) and torch.equal(gradient,torch.autograd.grad(other,logits)[0])
    assert gradient[0,0,0,0]==0 and torch.count_nonzero(gradient)==8


def test_all_invalid_skips_adam_and_weight_decay():
    model=torch.nn.Conv2d(1,1,1);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.5)
    before={k:v.clone() for k,v in model.state_dict().items()}
    batch={'image':torch.ones(1,1,3,3),'kernel':torch.ones(1,1,3,3),'validity':torch.zeros(1,1,3,3),'sample_id':['fixture']}
    result=train_kernel_epoch(model,[batch],opt,torch.device('cpu'),{'loss':OPTIONS})
    assert result['steps']==0 and result['skipped_all_invalid_batches']==1 and not opt.state
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_real_update_is_exclusively_kernel_and_errors_propagate():
    model=torch.nn.Conv2d(1,1,1);opt=torch.optim.Adam(model.parameters(),lr=.01)
    before=model.weight.clone()
    batch={'image':torch.ones(1,1,3,3),'kernel':torch.ones(1,1,3,3),'validity':torch.ones(1,1,3,3),'sample_id':['fixture']}
    result=train_kernel_epoch(model,[batch],opt,torch.device('cpu'),{'loss':OPTIONS})
    assert result['active_loss_components']==['kernel'] and result['steps']==1
    assert not torch.equal(before,model.weight)
    class Broken(torch.nn.Module):
        def forward(self,x):raise torch.cuda.OutOfMemoryError('fixture OOM')
    with pytest.raises(torch.cuda.OutOfMemoryError):train_kernel_epoch(Broken(),[batch],opt,torch.device('cpu'),{'loss':OPTIONS})
