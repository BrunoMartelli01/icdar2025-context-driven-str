from pathlib import Path

import numpy as np
import pytest
import torch

from models.adaspot_joint import CorrectnessGate, FeatureOCRHead, qualified_parseq_parts, spatial_encoding
from recognition.adaspot_gate import fit_temperature, calibration_metrics, route_cached
from segmentation.adaspot_joint import label_support, masked_dice, match_owner_texts


@pytest.fixture(scope="module")
def runtime():
    base = Path("/extra1/bmartelli/context-str-artifacts/parseq")
    source = base / "parseq-1902db043c029a7e03a3818c616c06600af574be"
    if not source.exists():
        pytest.skip("Qualified local PARSeq runtime unavailable")
    return qualified_parseq_parts(source, base / (source.name + ".tar.gz"), base / "parseq-bb5792a6.pt")


def small_head(runtime, depth, ar):
    return FeatureOCRHead(runtime[0], runtime[1], depth=depth, autoregressive=ar,
        channels=8, embed_dim=24, num_heads=4, mlp_ratio=2, dropout=0., max_label_length=4)


def test_real_light_and_four_layer_deep_gradients(runtime):
    light, deep = small_head(runtime, 1, False), small_head(runtime, 4, True)
    a = torch.randn(2, 8, 8, 32, requires_grad=True)
    b = torch.randn(2, 8, 16, 64, requires_grad=True)
    first, refined = light.light(light.memory(a))
    tokens = runtime[1].encode(["ab", "Cd"])
    logits = deep.teacher(deep.memory(b), tokens)
    assert first.shape == refined.shape == (2, 5, 95)
    assert logits.shape == (2, 3, 95)
    (refined.square().mean() + logits.square().mean()).backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert all(layer.cross_attn.in_proj_weight.grad.abs().sum() > 0 for layer in deep.decoder.layers)
    assert deep.decoder.layers[0].linear1.weight.data_ptr() != deep.decoder.layers[1].linear1.weight.data_ptr()


def test_deep_teacher_cannot_see_future_tokens(runtime):
    head = small_head(runtime, 4, True).eval()
    memory = head.memory(torch.randn(1, 8, 2, 4))
    a, b = runtime[1].encode(["abcd"]), runtime[1].encode(["abxy"])
    left, right = head.teacher(memory, a), head.teacher(memory, b)
    torch.testing.assert_close(left[:, :3], right[:, :3])
    assert not torch.allclose(left[:, 3:], right[:, 3:])


def test_official_checkpoint_copied_with_strict_complete_subset(runtime):
    head = FeatureOCRHead(runtime[0], runtime[1], depth=4, autoregressive=True,
        channels=64, embed_dim=384, num_heads=12, mlp_ratio=4, dropout=.1, max_label_length=25)
    report = head.initialize_parseq(runtime[3])
    assert report["strict_load"] and len(report["copied_keys"]) == 26
    assert any(k.startswith("decoder.layers.3.") for k in report["new_keys"])
    for key in report["copied_keys"]:
        torch.testing.assert_close(head.state_dict()[key], runtime[3][key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        small_head(runtime, 1, False).initialize_parseq(runtime[3])


def test_gate_stop_gradient_is_real():
    cues = torch.randn(3, 7, requires_grad=True)
    gate = CorrectnessGate(8)
    gate(cues).sum().backward()
    assert cues.grad is None
    assert gate.layers[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        gate(torch.randn(3, 6))


def test_token_probabilities_and_eos_exclusion(runtime):
    head = small_head(runtime, 1, False)
    tokens = runtime[1].encode(["ab"])[0, 1:]
    logits = torch.full((1, 3, 95), -10.)
    logits[0, torch.arange(3), tokens] = 10.
    reading = head.read(logits)[0]
    assert reading["text"] == "ab" and reading["eos_reached"]
    assert len(reading["character_probabilities"]) == 2


def test_no_silent_charset_or_length_rewrite(runtime):
    tokenizer = runtime[1]
    assert label_support("Hello!", tokenizer, 25) is None
    assert label_support("é", tokenizer, 25) == "unsupported_character"
    assert label_support("a" * 26, tokenizer, 25) == "overlength_annotation"


def test_one_to_one_training_match_preserves_false_positives():
    owners = np.zeros((20, 30), int)
    owners[2:8, 3:12] = 1
    proposals = [{"bbox_native": [3, 2, 12, 8]}, {"bbox_native": [3, 2, 12, 8]}]
    matched, audit = match_owner_texts(proposals, owners, {1: "word"}, iou_threshold=.5)
    assert len(matched) == 1 and audit["native_owners"] == 1
    with pytest.raises(ValueError):
        match_owner_texts(proposals, owners, {}, iou_threshold=.5)


def test_dice_validity_and_empty_supervision():
    logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
    target, valid = torch.ones_like(logits), torch.zeros_like(logits)
    zero = masked_dice(logits, target, valid, smooth=1.)
    zero.backward()
    assert zero.item() == 0 and logits.grad.abs().sum() == 0
    assert masked_dice(logits, target, torch.ones_like(valid), smooth=1.).item() > 0


def test_temperature_improves_or_preserves_nll_and_reports_single_class():
    fit = fit_temperature([8., -8., 8., -8.], [1, 0, 0, 1], log_bounds=[-3, 3])
    assert fit["temperature"] > 1 and fit["nll_after"] <= fit["nll_before"]
    metric = calibration_metrics([1., 1.], [0, 1], temperature=1., bins=10)
    assert metric["auroc"] == .5
    metric = calibration_metrics([-1., -2.], [0, 0], temperature=1., bins=10)
    assert metric["auroc"] is None and metric["undefined_auroc_reason"]


def test_routing_threshold_boundaries_and_rejection_denominator():
    sample = {"ground_truth": [{"text": "keep", "bbox": [0, 0, 2, 2]}],
        "predictions": [{"instance_id": i, "gate_logit": float(z),
            "light": {"text": "light", "minimum_probability": .8},
            "deep": {"text": "deep", "minimum_probability": .7}} for i, z in enumerate([-10, 0, 10])]}
    routed, decisions = route_cached(sample, temperature=1., reject_threshold=.25, keep_threshold=.75, deep_threshold=.7)
    assert [d["route"] for d in decisions] == ["reject", "escalate", "keep"]
    assert [p["text"] for p in routed["predictions"]] == ["deep", "light"]
    assert routed["ground_truth"] == sample["ground_truth"]
    _, decisions = route_cached(sample, temperature=1., reject_threshold=.5, keep_threshold=.5, deep_threshold=0)
    assert decisions[1]["route"] == "keep"


def test_spatial_positions_retain_both_axes():
    values = spatial_encoding(3, 4, 24, "cpu", torch.float32)
    assert values.shape == (1, 12, 24)
    assert not torch.equal(values[:, 0], values[:, 1])
    assert not torch.equal(values[:, 0], values[:, 4])
