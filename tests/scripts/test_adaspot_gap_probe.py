import numpy as np
import pytest
import torch

from scripts.train_adaspot_gap_probe import class_weights, head_probability, sample_pixels


def test_sampling_preserves_all_positives_excludes_ignore_and_is_repeatable():
    target = np.zeros((8, 8), bool); target[2, 2:5] = True
    valid = np.ones_like(target); valid[0] = False
    text = np.zeros_like(target); text[:, :4] = True
    a = sample_pixels(target, valid, text, np.random.default_rng(7), 4)
    b = sample_pixels(target, valid, text, np.random.default_rng(7), 4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert set(a[0]) == set(np.flatnonzero(target))
    joined = np.concatenate(a)
    assert len(joined) == len(set(joined)) and valid.ravel()[joined].all()
    assert not target.ravel()[np.concatenate(a[1:])].any()


def test_positive_outside_validity_fails_and_negative_only_image_survives():
    target = np.zeros((4, 4), bool); target[0, 0] = True
    with pytest.raises(ValueError): sample_pixels(target, ~target, target, np.random.default_rng(1), 2)
    groups = sample_pixels(np.zeros_like(target), np.ones_like(target), target, np.random.default_rng(1), 2)
    assert not len(groups[0]) and len(groups[1]) == 1 and len(groups[2]) == 2


def test_global_class_weights_equalize_three_strata_and_reject_missing():
    classes = torch.tensor([0, 1, 1, 2, 2, 2])
    weights, counts = class_weights(classes)
    assert counts.tolist() == [1, 2, 3]
    assert torch.allclose(weights * counts, torch.tensor([2., 2., 2.]))
    with pytest.raises(ValueError): class_weights(torch.tensor([0, 1]))


def test_linear_probe_is_equivalent_to_native_conv_and_features_stay_frozen():
    torch.manual_seed(1)
    head = torch.nn.Linear(64, 1)
    features = torch.randn(64, 7, 9)
    expected = head(features.permute(1, 2, 0)).sigmoid()[..., 0]
    actual = head_probability(head, features)
    assert torch.allclose(actual, expected, atol=1e-7)
    actual.mean().backward()
    assert head.weight.grad is not None and features.grad is None
