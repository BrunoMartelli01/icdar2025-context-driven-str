import cv2
import numpy as np
import pytest
import torch

from models.adaspot_mask_align import (MaskAlignOptions, mask_align_geometry,
    native_grid_to_rgb, sample_mask_aligned)


OPTIONS = MaskAlignOptions(.5, .25, .5)


def rectangle():
    mask = np.zeros((40, 64), bool)
    mask[12:25, 5:57] = True
    return mask


def test_straight_cubic_and_bin_centers():
    mask = rectangle()
    geometry = mask_align_geometry(mask, view="mask", options=OPTIONS)
    assert geometry.audit["method"] == "cubic_centerline"
    grid = geometry.grid((8, 32))
    assert grid.shape == (8, 32, 2)
    assert np.all(np.diff(grid[4, :, 0]) > 0)
    np.testing.assert_allclose(grid[:, :, 1].mean(), 18., atol=.2)
    assert grid[4, 0, 0] < 6 and grid[4, -1, 0] > 55


def test_curved_word_has_curved_sampling_grid():
    mask = np.zeros((80, 100), np.uint8)
    points = np.array([[x, round(25 + .012 * (x - 50) ** 2)] for x in range(10, 91)], np.int32)
    cv2.polylines(mask, [points], False, 1, thickness=11)
    geometry = mask_align_geometry(mask.astype(bool), view="mask", options=OPTIONS)
    assert geometry.audit["method"] == "cubic_centerline"
    center = geometry.grid((1, 32))[0]
    assert center[0, 1] - center[16, 1] > 10
    assert center[-1, 1] - center[16, 1] > 10


@pytest.mark.parametrize("shape,reason", [("branch", "branched_skeleton"),
    ("ring", "cycle_or_degenerate_skeleton"), ("disconnected", "disconnected_mask"),
    ("singleton", "cycle_or_degenerate_skeleton")])
def test_fallback_keeps_instance_and_reason(shape, reason):
    mask = np.zeros((60, 60), np.uint8)
    if shape == "branch":
        cv2.line(mask, (10, 12), (50, 12), 1, 7)
        cv2.line(mask, (30, 12), (30, 50), 1, 7)
    elif shape == "ring":
        cv2.circle(mask, (30, 30), 18, 1, 5)
    elif shape == "disconnected":
        mask[5:12, 5:12] = 1
        mask[30:40, 30:40] = 1
    else:
        mask[12, 20] = 1
    geometry = mask_align_geometry(mask.astype(bool), view="mask", options=OPTIONS)
    assert geometry.audit["method"] == "rotated_rectangle"
    assert geometry.audit["fallback_reason"] == reason
    assert np.isfinite(geometry.grid((16, 64))).all()


@pytest.mark.parametrize("view", ["axis", "rotated", "mask"])
def test_translation_preserves_geometry(view):
    mask = rectangle()
    moved = np.pad(mask, ((3, 7), (6, 2)))
    a = mask_align_geometry(mask, view=view, options=OPTIONS).grid((8, 32))
    b = mask_align_geometry(moved, view=view, options=OPTIONS).grid((8, 32))
    np.testing.assert_allclose(b, a + [6, 3], atol=1e-4)


def test_letterbox_inverse_half_pixels_and_odd_padding():
    grid = np.array([[[0., 0.], [3., 3.]]], np.float32)
    np.testing.assert_allclose(native_grid_to_rgb(grid, (8, 5), (4, 4)),
        [[[.5, -.5], [6.5, 5.5]]])


@pytest.mark.parametrize("cpu_backward", [False, True])
def test_bilinear_coordinates_and_exact_feature_gradients(cpu_backward):
    y, x = torch.meshgrid(torch.arange(10), torch.arange(12), indexing="ij")
    features = (x + 10 * y).float()[None, None].requires_grad_()
    grids = torch.tensor([[[[2.25, 3.5], [7.5, 6.25]]]])
    output = sample_mask_aligned(features, grids, deterministic_cpu_backward=cpu_backward)
    torch.testing.assert_close(output.flatten(), torch.tensor([37.25, 70.]))
    output.sum().backward()
    assert features.grad.sum().item() == pytest.approx(2.)
    assert features.grad[0, 0, 3, 2].item() == pytest.approx(.375)
    assert int((features.grad != 0).sum()) == 8


def test_full_mask_axis_identity_and_zero_padding():
    mask = np.ones((8, 12), bool)
    grid = mask_align_geometry(mask, view="axis", options=OPTIONS).grid((8, 12))
    features = torch.randn(1, 3, 8, 12)
    sampled = sample_mask_aligned(features, torch.from_numpy(grid[None]), deterministic_cpu_backward=False)
    torch.testing.assert_close(sampled, features, atol=2e-6, rtol=1e-5)
    outside = torch.full((1, 1, 1, 2), -10.)
    assert sample_mask_aligned(features, outside, deterministic_cpu_backward=False).abs().sum() == 0


@pytest.mark.parametrize("mask", [np.zeros((8, 8), bool), np.ones((8, 8), np.uint8), np.ones((8,), bool)])
def test_invalid_masks_fail(mask):
    with pytest.raises(ValueError):
        mask_align_geometry(mask, view="mask", options=OPTIONS)


def test_gpu_forward_and_explicit_deterministic_gradient():
    if not torch.cuda.is_available():
        pytest.skip("CUDA qualification requires actual GPU")
    old = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        features = torch.randn(1, 4, 40, 64)
        grid = torch.from_numpy(mask_align_geometry(rectangle(), view="mask", options=OPTIONS).grid((8, 32))[None])
        expected = sample_mask_aligned(features, grid, deterministic_cpu_backward=False)
        cuda_features = features.cuda().requires_grad_()
        actual = sample_mask_aligned(cuda_features, grid.cuda(), deterministic_cpu_backward=False)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)
        deterministic = sample_mask_aligned(cuda_features, grid.cuda(), deterministic_cpu_backward=True)
        deterministic.square().mean().backward()
        assert cuda_features.grad.is_cuda and torch.isfinite(cuda_features.grad).all()
        assert cuda_features.grad.abs().sum() > 0
    finally:
        torch.use_deterministic_algorithms(old)
