import pytest
import torch

from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_decoder import TrainableGapDecoder
from segmentation.adaspot_gap_decoder import decoder_loss


@pytest.fixture
def options():
    return dict(teacher_text_threshold=.65, balanced_bce_weight=.5, dice_weight=.5,
                text_protection_weight=.25, smooth=1.)


def test_real_decoder_is_independent_and_teacher_stays_exact():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    source = AdaSpotGapAGUNet("control", skip_floor=.25, initial_gap_bias=-2.).eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad(): reference = source.forward_evidence(image)["words"]
    net = TrainableGapDecoder(source, initial_bias=-2.)
    frozen = {k: v.clone() for k, v in source.state_dict().items()}
    net.train()
    assert not net.source.training and net.decoder.training
    assert net.decoder["bottleneck"][0].weight.data_ptr() != source.bottleneck[0].weight.data_ptr()
    optimizer = torch.optim.SGD((p for p in net.parameters() if p.requires_grad), lr=.01)
    for step in range(2):
        evidence = net.forward_evidence(image)
        assert torch.equal(evidence["words"], reference)
        assert evidence["gap"].shape == (2, 1, 32, 32)
        if step == 0: assert torch.all(evidence["gap"] == -2.)
        optimizer.zero_grad(set_to_none=True)
        evidence["gap"].square().mean().backward()
        if step == 1: assert next(net.decoder["up0"].parameters()).grad.abs().sum() > 0
        optimizer.step()
    assert all(p.grad is None for p in source.parameters())
    assert all(torch.equal(v, frozen[k]) for k, v in source.state_dict().items())
    branch = net.branch_state()
    net.load_branch(branch)
    with pytest.raises(ValueError, match="Complete exact"): net.load_branch({})


def test_loss_rewards_real_corridor_and_protects_word(options):
    target = torch.tensor([[[[1., 0., 0., 0.]]]])
    text = torch.tensor([[[[0., 1., 0., 0.]]]])
    valid = torch.ones_like(target)
    teacher = torch.tensor([[[[.9, .9, .1, .1]]]])
    logits = torch.zeros_like(target, requires_grad=True)
    loss, _ = decoder_loss(logits, target, text, valid, teacher, options)
    loss.backward()
    assert logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[0, 0, 0, 1] > logits.grad[0, 0, 0, 2] > 0
    bad = torch.tensor([[[[-4., 4., 4., 4.]]]])
    good = -bad
    assert decoder_loss(good, target, text, valid, teacher, options)[0] < decoder_loss(bad, target, text, valid, teacher, options)[0]


def test_ignored_pixels_have_no_gradient_even_all_invalid(options):
    logits = torch.randn(1, 1, 2, 2, requires_grad=True)
    zero = torch.zeros_like(logits)
    loss, _ = decoder_loss(logits, zero, zero, zero, zero, options)
    assert loss == 0
    loss.backward()
    assert torch.all(logits.grad == 0)


def test_gap_inside_text_is_rejected(options):
    one = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="Invalid Gap"):
        decoder_loss(one, one, one, one, one, options)
