"""Full architecture compatibility and strict warm-start behavior."""

import pytest
import torch

from models.adaspot_gap import AdaSpotGapAGUNet


def test_real_shared_backbone_strict_transfer_shapes_and_frozen_encoder():
    model = AdaSpotGapAGUNet("gap", skip_floor=.25, initial_gap_bias=-2)
    source = {k: v for k, v in model.state_dict().items() if not k.startswith("gap.")}
    report = model.initialize_kernel_state(source)
    assert report["strict_state_dict_load"] and report["new_tensors"] == ["gap.bias", "gap.weight"]
    assert sum(p.numel() for p in model.gap.parameters()) == 65
    model.train()
    assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    model.eval()
    with torch.inference_mode():
        image = torch.randn(2, 3, 64, 64)
        evidence = model.forward_evidence(image)
        output = model(image)
    assert output.shape == (2, 3, 32, 32)
    assert torch.equal(output[:, :2], evidence["words"])
    assert torch.all(output[:, 2:] == -2)
    with pytest.raises(ValueError, match="every qualified"):
        model.initialize_kernel_state({k: v for k, v in source.items() if k != "head.bias"})
    with pytest.raises(ValueError, match="shape/dtype"):
        model.initialize_kernel_state({**source, "head.bias": torch.zeros(3)})


@pytest.mark.parametrize("branch,bias", [("other", -2), ("gap", float("nan")), ("gap", 0)])
def test_invalid_architecture_selection_fails(branch, bias):
    with pytest.raises(ValueError):
        AdaSpotGapAGUNet(branch, skip_floor=.25, initial_gap_bias=bias)
