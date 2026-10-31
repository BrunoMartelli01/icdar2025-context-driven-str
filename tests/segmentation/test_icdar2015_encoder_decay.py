import copy

import pytest
import torch

from segmentation.icdar2015_encoder_decay import restored_optimizer


def source():
    network = torch.nn.Module()
    network.decoder = torch.nn.Linear(2,1)
    network.encoder = torch.nn.Linear(2,2)
    optimizer = torch.optim.Adam([
        dict(params=network.decoder.parameters(),group_name='decoder'),
        dict(params=network.encoder.parameters(),group_name='encoder_layer3_layer4')],lr=1e-5,weight_decay=.01)
    sum(p.square().sum() for p in network.parameters()).backward();optimizer.step()
    return network,dict(optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))


CONFIG = dict(training=dict(learning_rate=1e-5,weight_decay=.01,adam_betas=[.9,.999],adam_epsilon=1e-8))


@pytest.mark.parametrize('decay',[0.,.01])
def test_decoder_state_preserved_encoder_state_explicitly_reset(decay):
    network,payload=source();old=copy.deepcopy(payload)
    optimizer=restored_optimizer(network,payload,CONFIG,dict(encoder_learning_rate=1e-5,encoder_weight_decay=decay))
    for p,pid in zip(network.decoder.parameters(),old['optimizer_state_dict']['param_groups'][0]['params']):
        assert all(torch.equal(v,old['optimizer_state_dict']['state'][pid][k]) for k,v in optimizer.state[p].items())
    assert all(p not in optimizer.state for p in network.encoder.parameters())
    assert [g['weight_decay'] for g in optimizer.param_groups]==[.01,decay]
    assert [g['initial_lr'] for g in optimizer.param_groups]==[1e-5,1e-5]
    assert len(payload['optimizer_state_dict']['state'])==len(old['optimizer_state_dict']['state'])


def test_zero_data_gradient_preserves_unregularized_encoder():
    initial,payload=source();zero=copy.deepcopy(initial);regularized=copy.deepcopy(initial)
    for network,decay in [(zero,0.),(regularized,.01)]:
        optimizer=restored_optimizer(network,payload,CONFIG,dict(encoder_learning_rate=1e-5,encoder_weight_decay=decay))
        for p in network.parameters():p.grad=torch.zeros_like(p)
        optimizer.step()
    assert all(torch.equal(a,b) for a,b in zip(zero.encoder.parameters(),initial.encoder.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(regularized.encoder.parameters(),initial.encoder.parameters()))


@pytest.mark.parametrize('corruption',['count','source_decay','group_name','undeclared_decay'])
def test_incompatible_optimizer_fails_closed(corruption):
    network,payload=source();branch=dict(encoder_learning_rate=1e-5,encoder_weight_decay=0.)
    if corruption=='count':payload['optimizer_state_dict']['param_groups'][0]['params'].pop()
    if corruption=='source_decay':payload['optimizer_state_dict']['param_groups'][1]['weight_decay']=.02
    if corruption=='group_name':payload['optimizer_state_dict']['param_groups'][0]['group_name']='other'
    if corruption=='undeclared_decay':branch['encoder_weight_decay']=.003
    with pytest.raises(ValueError):restored_optimizer(network,payload,CONFIG,branch)
