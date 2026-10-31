import math

import pytest
import torch

from models.unet_dense_instances import DenseWordInstances
from segmentation.unet_dense_instances import center_targets, dense_instance_loss, instance_proposals


def test_tiny_word_preserved_and_colliding_centers_rejected():
    target = center_targets([[.101, .101, .102, .102]], torch.ones(8, 8),
                            radius_fraction=1/6, minimum_radius=0)
    assert len(target['index']) == 1 and target['heatmap'].eq(1).sum() == 1
    with pytest.raises(ValueError, match='collide'):
        center_targets([[.1, .1, .2, .2], [.11, .11, .21, .21]], torch.ones(8, 8),
                       radius_fraction=1/6, minimum_radius=0)


def test_finite_gradients_and_ignore_only_negative_centers():
    head = DenseWordInstances(input_channels=4, width=8, groups=2, dilations=[1, 2],
                              initial_probability=.01, initial_size=[.1, .1])
    features = torch.randn(1, 4, 16, 16)
    output = head(features)
    output['center'].retain_grad()
    validity = torch.ones(16, 16)
    validity[0] = 0
    target = center_targets([[.25, .25, .5, .5]], validity,
                            radius_fraction=1/6, minimum_radius=0)
    settings = dict(focal_alpha=2., focal_beta=4., log_size_bounds=[-9., math.log(2)],
                    center_weight=1., size_weight=1., offset_weight=1., giou_weight=1.)
    loss, _ = dense_instance_loss(output, [target], settings)
    loss.backward()
    assert output['center'].grad[0, 0, 0].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.extent.weight.grad.abs().sum() > 0


def test_perfect_center_extent_roundtrip_to_rectangular_rgb():
    center = torch.full((1, 1, 100, 100), -20.)
    center[0, 0, 45, 40] = 10.
    offset = torch.zeros(1, 2, 100, 100)
    # 200x100 RGB: square canvas has top padding50.
    extent = torch.zeros(1, 2, 100, 100)
    extent[:, 0] = math.log(.2)
    extent[:, 1] = math.log(.1)
    output = dict(center=center, offset=offset, log_size=extent)
    rows = instance_proposals(output, [200, 100], peak_threshold=.5,
        maximum_candidates=10, nms_iou=.5, padding=0, log_size_bounds=[-9., math.log(2)])
    assert len(rows) == 1
    box = rows[0]['bbox']
    assert abs(box[0]-61) <= 1 and abs(box[1]-31) <= 1
    assert abs(box[2]-101) <= 1 and abs(box[3]-51) <= 1


def test_care_center_positive_survives_an_ignored_bbox_concavity():
    center = torch.zeros(1, 1, 8, 8, requires_grad=True)
    output = dict(center=center, offset=torch.zeros(1, 2, 8, 8, requires_grad=True),
                  log_size=torch.zeros(1, 2, 8, 8, requires_grad=True))
    target = center_targets([[.25, .25, .5, .5]], torch.zeros(8, 8),
                            radius_fraction=1/6, minimum_radius=0)
    settings = dict(focal_alpha=2., focal_beta=4., log_size_bounds=[-9., math.log(2)],
                    center_weight=1., size_weight=0., offset_weight=0., giou_weight=0.)
    loss, _ = dense_instance_loss(output, [target], settings)
    loss.backward()
    assert center.grad.count_nonzero() == 1
    assert center.grad.flatten()[target['index'][0]] < 0


def test_lexical_emission_keeps_native_confidence_and_inclusive_boundaries():
    from recognition.deepsolo_lexicon import DeepSoloLexicon
    from scripts.evaluate_icdar13_dense_instances import apply_policy
    dictionary = DeepSoloLexicon(['HELLO'], {'HELLO': 'Hello'}, distance_exclusive=1.5)
    raw = [dict(raw_text='HEL1O', detector_score=.1, sequence_confidence_native=.7,
                confidence=.93, token_probabilities=[.93]*5),
           dict(raw_text='...', detector_score=.9, sequence_confidence_native=.9,
                confidence=.9, token_probabilities=[.9]*3)]
    policy = dict(peak_threshold=.1, sequence_threshold=.7, maximum_candidates=50,
                  lexical='reject')
    rows, trace = apply_policy(raw, policy, dictionary)
    assert len(rows) == 1 and rows[0]['raw_text'] == 'Hello'
    assert rows[0]['confidence'] == .93 and rows[0]['token_probabilities'] == [.93]*5
    assert raw[0]['raw_text'] == 'HEL1O' and trace[0]['original_text'] == 'HEL1O'
