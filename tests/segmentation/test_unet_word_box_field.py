import math

import pytest
import torch

from models.unet_word_box_field import WordBoxField, WordBoxFieldUNet
from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.unet_word_box_field import word_box_targets, word_box_loss, word_box_proposals


def test_subcell_word_keeps_exact_original_box_and_each_word_has_unit_weight():
    boxes = torch.tensor([[.101, .101, .102, .102], [.3, .3, .8, .6]])
    t = word_box_targets(boxes, torch.ones(8, 8), core_fraction=.5)
    assert t['represented_words'] == 2
    for owner in [1, 2]:
        selected = t['owners'].flatten()[t['indices']] == owner
        assert torch.allclose(t['weights'][selected].sum(), torch.tensor(1.))
    xy = torch.stack((t['indices'] % 8, t['indices']//8), -1)
    anchors = (xy+t['offsets'])/8
    reconstructed = torch.cat((anchors-t['distances'][:, :2],
                               anchors+t['distances'][:, 2:]), -1)
    assert torch.allclose(reconstructed, t['boxes'], atol=1e-7)
    with pytest.raises(ValueError, match='collide'):
        word_box_targets([[.1, .1, .2, .2], [.11, .11, .21, .21]], torch.ones(8, 8), core_fraction=.5)


def test_real_unet_backward_updates_decoder_and_field_but_freezes_encoder():
    torch.set_num_threads(2)
    backbone = WordRefitUNet(partial_encoder=False, skip_floor=.25, initial_gap_bias=-2.)
    head = WordBoxField(input_channels=66, width=8, groups=2, dilations=[1],
        initial_probability=.01, initial_distances=[.1, .05, .1, .05])
    net = WordBoxFieldUNet(backbone, head, train_decoder=True).train()
    output = net(torch.randn(1, 3, 64, 64))
    target = word_box_targets([[.2, .2, .7, .4]], torch.ones(32, 32), core_fraction=.5)
    loss, _ = word_box_loss(output, [target], dict(focal_alpha=.25, focal_gamma=2.,
        classification_weight=1., centerness_weight=1., giou_weight=2., offset_weight=1.))
    loss.backward()
    assert head.distances.weight.grad.abs().sum() > 0
    assert backbone.up0.block.conv1.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backbone.encoder.parameters())
    assert torch.isfinite(loss)


def test_exact_box_prediction_returns_original_rectangular_rgb_frame():
    score = torch.full((1, 1, 10, 10), -20.)
    score[0, 0, 4, 4] = 20.
    distance = torch.empty(1, 4, 10, 10)
    for channel, value in enumerate([.1, .05, .1, .05]):
        distance[:, channel].fill_(math.log(math.expm1(value)))
    output = dict(class_logits=score, center_logits=torch.full_like(score, 20.),
                  raw_distances=distance, offset_logits=torch.zeros(1, 2, 10, 10))
    rows = word_box_proposals(output, [200, 100], dict(score_threshold=.2,
        maximum_distance=2., padding=0, nms_iou=.5, maximum_candidates=50))
    assert len(rows) == 1
    assert all(abs(actual-wanted) <= 1 for actual, wanted in zip(rows[0]['bbox'], [70, 30, 110, 50]))
