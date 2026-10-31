"""Actual gradients and preserved decoder optimizer state, without downloads."""

import torch

from models.icdar2015_improvements import ImprovementAGUNet
from models.icdar2015_target70 import Target70AGUNet,restored_optimizer


def test_frozen_predictions_and_actual_partial_encoder_gradients():
    torch.set_num_threads(2);torch.manual_seed(7)
    source=ImprovementAGUNet(normalization="batchnorm",max_groups=32,skip_floor=.25).eval()
    network=Target70AGUNet(encoder_stages=[],max_groups=32,skip_floor=.25,source_state=source.state_dict()).eval()
    image=torch.randn(2,3,64,96)
    with torch.no_grad():assert torch.equal(source(image),network(image))
    del network
    network=Target70AGUNet(encoder_stages=["layer3","layer4"],max_groups=32,skip_floor=.25,source_state=source.state_dict()).train()
    before={name:value.clone() for name,value in network.encoder.named_buffers()}
    network(image).square().mean().backward()
    for stage in ("layer3","layer4"):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in network.encoder.get_submodule(stage).parameters())
    assert all(p.grad is None for name,p in network.encoder.named_parameters() if not name.startswith(("layer3.","layer4.")))
    for module in network.encoder.modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            assert not module.training and all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert all(torch.equal(value,before[name]) for name,value in network.encoder.named_buffers())


def test_source_decoder_adam_moments_are_restored_encoder_moments_are_fresh():
    torch.set_num_threads(2)
    network=Target70AGUNet(encoder_stages=[],max_groups=32,skip_floor=.25)
    config={"training":{"learning_rate":1e-5,"weight_decay":.01,"adam_betas":[.9,.999],"adam_epsilon":1e-8}}
    old=torch.optim.Adam([p for p in network.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01)
    network.head.bias.grad=torch.ones_like(network.head.bias);old.step();old.zero_grad(set_to_none=True)
    expected={k:v.clone() for k,v in old.state[network.head.bias].items()}
    payload={"optimizer_state_dict":old.state_dict()}
    for stage in ("layer3","layer4"):
        network.encoder.get_submodule(stage).requires_grad_(True)
    for module in network.encoder.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.requires_grad_(False)
    optimizer=restored_optimizer(network,payload,config,{"encoder_learning_rate":1e-6})
    assert len(optimizer.param_groups)==2
    assert all(torch.equal(optimizer.state[network.head.bias][k],value) for k,value in expected.items())
    assert all(parameter not in optimizer.state for parameter in optimizer.param_groups[1]["params"])
