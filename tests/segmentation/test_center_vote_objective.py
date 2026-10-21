from dataclasses import replace

import numpy as np
import pytest
import torch

from segmentation.word_center_offsets import make_word_center_offsets, word_center_offset_loss, train_word_center_offset_epoch
from segmentation.word_center_votes import CenterVoteLossParams, center_vote_objective, train_center_vote_epoch


@pytest.fixture
def params():
    return CenterVoteLossParams(beta=.01,regression_weight=1.,consensus_weight=.5,neighbor_weight=.5,
        separation_margin_fraction=.1,centroid_tolerance_pixels=.001,competition_chunk_pixels=2048)


def example(owners=None, validity=None):
    if owners is None:
        owners = np.zeros((16,24),np.int32)
        owners[3:13,2:10] = 1;owners[3:13,14:22] = 2
    validity = np.ones_like(owners,bool) if validity is None else validity
    targets,_ = make_word_center_offsets(owners,validity)
    return torch.from_numpy(targets)[None],torch.from_numpy(owners)[None],torch.from_numpy(validity)[None,None]


def objective(raw,data,params):
    return center_vote_objective(raw,*data,params=params)


def test_exact_targets_have_zero_objective_and_scatter_in_rectangular_coordinates(params):
    data = example();raw = torch.atanh(data[0]).requires_grad_()
    stats = objective(raw,data,params)
    assert stats['loss'] < 1e-12 and stats['endpoint_error_native_pixels'] < 1e-5
    assert stats['endpoint_scatter_native_pixels'] < 1e-5 and stats['wrong_neighbor_vote_rate'] == 0
    assert stats['supervised_images'] == 1 and stats['surviving_words'] == stats['competing_words'] == 2
    stats['loss'].backward()
    assert torch.isfinite(raw.grad).all()


def test_consensus_detects_scattered_votes_with_unchanged_average_center(params):
    data = example();shift = data[0].clone()
    mask = data[1][0] == 1
    yy,xx = torch.where(mask)
    shift[0,0,yy,xx] += torch.where(xx%2==0,.05,-.05)
    stats = objective(torch.atanh(shift),data,params)
    assert stats['consensus_loss'] > .001 and stats['endpoint_scatter_native_pixels'] > 0
    assert objective(torch.atanh(data[0]),data,params)['consensus_loss'] < 1e-12


def test_coherent_wrong_centers_are_penalized_and_neighbor_gradients_point_back(params):
    data = example();target,owners,valid = data
    # Move every left-word vote to the right centroid and vice versa.
    swapped = target.clone()
    swapped[0,0,owners[0]==1] += 12/24
    swapped[0,0,owners[0]==2] -= 12/24
    raw = torch.atanh(swapped).requires_grad_()
    stats = objective(raw,data,params)
    assert stats['consensus_loss'] < 1e-12
    assert stats['separation_loss'] > .1 and stats['wrong_neighbor_vote_rate'] == 1
    stats['separation_loss'].backward()
    assert torch.all(raw.grad[0,0,owners[0]==1] > 0)
    assert torch.all(raw.grad[0,0,owners[0]==2] < 0)


def test_no_neighbor_and_coincident_centers_are_explicit_not_invented(params):
    owners = np.zeros((4,4),np.int32);owners[0,0] = owners[3,3] = 1
    data = example(owners)
    stats = objective(torch.atanh(data[0]),data,params)
    assert stats['competing_words'] == stats['coincident_centroid_words'] == 0
    assert stats['separation_loss'] == stats['wrong_neighbor_vote_rate'] == 0
    owners[0,3] = owners[3,0] = 2
    data = example(owners)
    stats = objective(torch.atanh(data[0]),data,params)
    assert stats['coincident_centroid_words'] == 2 and stats['competing_words'] == 0
    assert stats['loss'] < 1e-12


def test_equal_word_and_image_aggregation_does_not_weight_large_words(params):
    owners = np.zeros((16,24),np.int32);owners[0,0] = 1;owners[3:13,2:22] = 2
    target,owner,valid = example(owners)
    shifted = target.clone()
    shifted[0,:,owner[0]==1] += .1;shifted[0,:,owner[0]==2] += .3
    stats = objective(torch.atanh(shifted),(target,owner,valid),params)
    assert stats['offset_loss'].item() == pytest.approx(.195)
    raw = torch.cat((torch.atanh(shifted),torch.atanh(target),torch.zeros_like(target)))
    target = target.repeat(3,1,1,1);owner = owner.repeat(3,1,1);valid = valid.repeat(3,1,1,1);owner[2] = 0
    combined = objective(raw,(target,owner,valid),params)
    assert combined['offset_loss'].item() == pytest.approx(.195/2)
    assert combined['supervised_images'] == 2 and combined['surviving_words'] == 4


def test_invalid_background_and_ignored_pixels_have_zero_gradient(params):
    data = example();valid = data[2].numpy()[0,0].copy();valid[3:6,2:10] = False
    data = example(data[1].numpy()[0],valid)
    raw = torch.zeros_like(data[0],requires_grad=True)
    objective(raw,data,params)['loss'].backward()
    inactive = (data[1]==0) | ~data[2][:,0]
    assert not raw.grad.permute(0,2,3,1)[inactive].any()
    empty = (data[0],data[1]*0,data[2])
    raw = torch.randn_like(raw,requires_grad=True)
    stats = objective(raw,empty,params);stats['loss'].backward()
    assert stats['supervised_images'] == 0 and stats['loss'] == 0 and not raw.grad.any()


def test_zero_extra_weights_preserve_v1_loss_and_optimizer_step(params):
    data = example();torch.manual_seed(123)
    raw = torch.randn_like(data[0],requires_grad=True)
    control = replace(params,consensus_weight=0.,neighbor_weight=0.)
    stats = objective(raw,data,control)
    reference = word_center_offset_loss(raw,*data,beta=params.beta)
    assert torch.equal(stats['loss'],reference)
    assert torch.equal(torch.autograd.grad(stats['loss'],raw,retain_graph=True)[0],torch.autograd.grad(reference,raw)[0])
    a = torch.nn.Conv2d(3,2,1);b = torch.nn.Conv2d(3,2,1);b.load_state_dict(a.state_dict())
    batch = {'image':torch.randn(1,3,16,24),'center_offsets':data[0],'owners_native':data[1],
             'validity':data[2],'sample_id':['same:1']}
    oa = torch.optim.Adam(a.parameters(),lr=1e-4,weight_decay=.01)
    ob = torch.optim.Adam(b.parameters(),lr=1e-4,weight_decay=.01)
    train_center_vote_epoch(a,[batch],oa,torch.device('cpu'),params=control)
    train_word_center_offset_epoch(b,[batch],ob,torch.device('cpu'),beta=params.beta)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_empty_batches_skip_weight_decay_but_count_samples(params):
    target,owners,valid = example()
    batch = {'image':torch.randn(1,3,16,24),'center_offsets':target,'owners_native':owners*0,
             'validity':valid,'sample_id':['empty:1']}
    model = torch.nn.Conv2d(3,2,1);before = model.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(),lr=.1,weight_decay=.5)
    stats = train_center_vote_epoch(model,[batch],optimizer,torch.device('cpu'),params=params)
    assert stats['empty_batches_without_update'] == 1 and stats['steps'] == 0 and stats['samples'] == 1
    assert torch.equal(before,model.weight) and not optimizer.state


def test_chunking_does_not_change_nearest_rivals_or_gradients(params):
    owners = np.zeros((16,24),np.int32)
    owners[1:6,1:9] = 1;owners[1:6,13:21] = 2;owners[10:15,7:15] = 3
    data = example(owners);torch.manual_seed(12)
    raw = (torch.atanh(data[0])+torch.randn_like(data[0])*.2).requires_grad_()
    a = objective(raw,data,replace(params,competition_chunk_pixels=1))
    b = objective(raw,data,params)
    assert torch.equal(a['separation_loss'],b['separation_loss'])
    assert torch.equal(torch.autograd.grad(a['loss'],raw,retain_graph=True)[0],torch.autograd.grad(b['loss'],raw)[0])


def test_objective_passes_double_precision_gradient_check(params):
    data = example();data = (data[0].double(),data[1],data[2])
    torch.manual_seed(51)
    raw = (torch.atanh(data[0])+torch.randn_like(data[0])*.25).requires_grad_()
    # Check a small subset of free coordinates while holding the full geometry.
    free = raw[0,:,4,4].detach().clone().requires_grad_()
    fixed = raw.detach()
    def loss(value):
        modified = fixed.clone();modified[0,:,4,4] = value
        return objective(modified,data,params)['loss']
    assert torch.autograd.gradcheck(loss,(free,),eps=1e-6,atol=1e-5)


@pytest.mark.parametrize('problem',['nan','shape','owners','noncentroid','range'])
def test_invalid_or_misaligned_targets_fail(problem,params):
    target,owners,valid = example();raw = torch.zeros_like(target)
    if problem=='nan':raw[0,0,0,0] = float('nan')
    if problem=='shape':raw = raw[:,:1]
    if problem=='owners':owners = owners.float()
    if problem=='noncentroid':target[0,0,4,4] += .1
    if problem=='range':target[0,0,0,0] = 1
    with pytest.raises(ValueError):objective(raw,(target,owners,valid),params)


@pytest.mark.parametrize('field,value',[('beta',0),('consensus_weight',-1),('neighbor_weight',float('nan')),
    ('separation_margin_fraction',1),('centroid_tolerance_pixels',0),('competition_chunk_pixels',True)])
def test_invalid_scientific_loss_settings_fail(params,field,value):
    with pytest.raises(ValueError):replace(params,**{field:value})
