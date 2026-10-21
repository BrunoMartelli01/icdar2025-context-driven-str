from __future__ import annotations

import torch

from models.totaltext_frequency_gated_skip import haar_details, PracticalTotalTextFrequencyGatedSkipAGUNet


def test_haar_details_constant_is_zero() -> None:
    features = torch.ones((1, 2, 4, 4))
    details = haar_details(features)
    assert details.shape == (1, 6, 2, 2)
    assert torch.count_nonzero(details) == 0


def test_haar_details_detect_directional_change() -> None:
    features = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    details = haar_details(features)
    assert details[0, 0, 0, 0] == 1.0
    assert details[0, 1, 0, 0] == 0.0
    assert details[0, 2, 0, 0] == 0.0


def test_zero_gain_preserves_original_skip_and_accepts_gradient() -> None:
    model = PracticalTotalTextFrequencyGatedSkipAGUNet(encoder_pretrained=False)
    shallow = torch.randn((1, 64, 8, 8))
    decoder = torch.randn((1, 128, 4, 4))
    selected = model.gated_skip(shallow, decoder)
    assert torch.equal(selected, shallow)
    selected.sum().backward()
    assert model.frequency_gain.grad is not None
    assert torch.isfinite(model.frequency_gain.grad)


def test_model_native_output_shape() -> None:
    model = PracticalTotalTextFrequencyGatedSkipAGUNet(encoder_pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.zeros((1, 3, 64, 64)))
    assert logits.shape == (1, 1, 32, 32)
