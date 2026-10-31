"""Full feature gradients must reach early layers while source inference is exact."""

import torch

from models.icdar2015_target70 import Target70AGUNet
from models.icdar2015_target70_ohem import OHEMAGUNet,restored_optimizer


def test_full_encoder_preserves_source_logits_and_enables_every_feature_stage():
    torch.set_num_threads(2);torch.manual_seed(81)
    source=Target70AGUNet(encoder_stages=["layer3","layer4"],skip_floor=.25,max_groups=32).eval()
    full=OHEMAGUNet(full_encoder=True,skip_floor=.25,max_groups=32,source_state=source.state_dict()).eval()
    image=torch.randn(2,3,64,96)
    with torch.no_grad():assert torch.equal(source(image),full(image))
    before={k:v.clone() for k,v in full.encoder.named_buffers()};full.train();full(image).square().mean().backward()
    for stage in ("conv1","layer1","layer2","layer3","layer4"):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in full.encoder.get_submodule(stage).parameters())
    for module in full.encoder.modules():
        if isinstance(module,torch.nn.BatchNorm2d):assert not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert all(torch.equal(v,before[k]) for k,v in full.encoder.named_buffers())
    assert all(not p.requires_grad for p in full.encoder.fc.parameters())


def test_inherited_decoder_and_deep_moments_are_restored_early_group_is_fresh():
    torch.set_num_threads(2)
    source=OHEMAGUNet(full_encoder=False,skip_floor=.25,max_groups=32)
    decoder=[p for n,p in source.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
    deep=[p for n,p in source.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    old=torch.optim.Adam([{'params':decoder,'group_name':'decoder','lr':2e-6},{'params':deep,'group_name':'encoder_layer3_layer4','lr':2e-7}],weight_decay=.01)
    source.head.bias.grad=torch.ones_like(source.head.bias)
    deep_param=next(p for p in source.encoder.layer3.parameters() if p.requires_grad);deep_param.grad=torch.ones_like(deep_param)
    old.step();old.zero_grad(set_to_none=True)
    full=OHEMAGUNet(full_encoder=True,skip_floor=.25,max_groups=32,source_state=source.state_dict())
    config={'training':{'learning_rate':1e-5,'weight_decay':.01,'adam_betas':[.9,.999],'adam_epsilon':1e-8}}
    branch={'encoder_learning_rate':1e-6,'early_encoder_learning_rate':1e-6}
    optimizer=restored_optimizer(full,{'optimizer_state_dict':old.state_dict()},config,branch)
    new_deep=next(p for p in full.encoder.layer3.parameters() if p.requires_grad)
    assert torch.equal(optimizer.state[full.head.bias]['exp_avg'],old.state[source.head.bias]['exp_avg'])
    assert torch.equal(optimizer.state[new_deep]['exp_avg'],old.state[deep_param]['exp_avg'])
    assert [g['lr'] for g in optimizer.param_groups]==[1e-5,1e-6,1e-6]
    assert all(p not in optimizer.state for p in optimizer.param_groups[2]['params'])
