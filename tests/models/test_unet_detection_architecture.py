import gc

import pytest
import torch

from models.unet_detection_methods import DetectionMethodsAGUNet


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous); gc.collect()


@pytest.mark.parametrize("branch,channels", [("scale_fusion", 2), ("progressive_kernels", 4), ("fusion_progressive", 4)])
def test_strict_warm_start_preserves_original_logits_and_parameter_budget(branch, channels):
    torch.manual_seed(11)
    source = DetectionMethodsAGUNet("reference").eval()
    model = DetectionMethodsAGUNet(branch).eval()
    model.initialize_source(source.state_dict())
    image = torch.rand(2, 3, 64, 96)
    with torch.inference_mode():
        expected = source(image); actual = model(image)
    assert actual.shape == (2, channels, 32, 48) and torch.equal(expected, actual[:, :2])
    if channels == 4:
        assert torch.equal(actual[:, 1], actual[:, 2]) and torch.equal(actual[:, 1], actual[:, 3])
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in source.parameters())
    assert extra < 20000
    if branch == "progressive_kernels": assert extra == 130
    bad = dict(source.state_dict()); bad.pop("head.bias")
    with pytest.raises(ValueError, match="names"): model.initialize_source(bad)


def test_encoder_actually_receives_gradients_and_bn_remains_frozen():
    model = DetectionMethodsAGUNet("fusion_progressive").train()
    before = model.encoder.bn1.running_mean.clone()
    logits = model(torch.rand(2, 3, 64, 64))
    logits.square().mean().backward()
    assert model.encoder.conv1.weight.grad is not None and model.encoder.conv1.weight.grad.abs().sum() > 0
    assert model.scale_fusion.residual.weight.grad.abs().sum() > 0
    assert model.kernel_scales.weight.grad.abs().sum() > 0
    assert not model.encoder.bn1.training and not model.encoder.bn1.weight.requires_grad
    assert torch.equal(before, model.encoder.bn1.running_mean)
