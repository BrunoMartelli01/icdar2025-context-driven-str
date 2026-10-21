import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from scripts.compare_yolo_parseq_four_datasets import DEFAULT_CONFIG, sample_indices, validate_config
from scripts.neural_macs import DenseMacCounter, letterbox_shape, multihead_attention_macs


def dense_operator_flops(function):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        with torch.inference_mode():
            function()
    # Torch also counts attention scaling multiplies; these are outside our
    # explicitly declared dense-MAC convention, alongside other pointwise ops.
    dense = {"aten::mm", "aten::bmm", "aten::addmm", "aten::conv2d"}
    return sum(event.flops for event in prof.key_averages() if event.key in dense)


def test_config_preserves_four_complete_datasets_and_approximate_boundary():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    validate_config(cfg)
    assert cfg["datasets"]["total_text"] == 300
    for key, value in (("reproduction_mode", "strict"), ("matching_iou", 0.25)):
        altered = copy.deepcopy(cfg)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_config(altered)


def test_evenly_spaced_profile_selection_is_deterministic_and_bounded():
    assert sample_indices(300, 4) == [0, 99, 199, 299]
    assert sample_indices(2, 4) == [0, 1]
    with pytest.raises(ValueError):
        sample_indices(0, 4)


def test_letterbox_shape_matches_qualified_actual_preprocessor():
    from ultralytics.data.augment import LetterBox
    for width, height in ((1280, 720), (640, 480), (669, 500), (180, 240), (101, 27), (3, 3)):
        actual = LetterBox(new_shape=(640, 640), auto=True, stride=32)(
            image=np.zeros((height, width, 3), dtype=np.uint8))
        assert letterbox_shape((width, height), 640, 32) == [1, 3, *actual.shape[:2]]


def test_conv_and_linear_count_matches_torch_operator_profiler():
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.Flatten(), nn.Linear(4 * 6 * 6, 5, bias=False)).eval()
    x = torch.zeros((2, 3, 8, 8))
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    with DenseMacCounter(model) as counter, torch.inference_mode():
        counted = model(x)
    assert 2 * counter.total == dense_operator_flops(lambda: model(x))
    assert torch.equal(counted, model(x))
    assert all(torch.equal(v, weights[k]) for k, v in model.state_dict().items())
    assert not counter.handles and not any(m._forward_hooks for m in model.modules())


def test_mha_functional_projections_and_attention_products_are_counted_once():
    torch.set_num_threads(1)
    model = nn.MultiheadAttention(8, 2, batch_first=True).eval()
    q, kv = torch.zeros((2, 3, 8)), torch.zeros((2, 5, 8))
    with DenseMacCounter(model) as counter, torch.inference_mode():
        model(q, kv, kv)
    assert counter.total == multihead_attention_macs(2, 3, 5, 8)
    assert 2 * counter.total == dense_operator_flops(lambda: model(q, kv, kv))
    assert list(counter.counts) == ["multihead_attention"]


def test_vit_attention_products_add_to_actual_linear_projection_counts():
    from timm.models.vision_transformer import Attention
    model = Attention(8, num_heads=2).eval()
    # Use explicit matmuls only in this synthetic independent profiler comparison.
    model.fused_attn = False
    x = torch.zeros((2, 5, 8))
    with DenseMacCounter(model) as counter, torch.inference_mode():
        model(x)
    assert counter.counts["vit_attention_products"] == 2 * 2 * 5 * 5 * 8
    assert 2 * counter.total == dense_operator_flops(lambda: model(x))


def test_profile_hooks_are_removed_after_forward_failure():
    model = nn.Linear(8, 4)
    with pytest.raises(RuntimeError):
        with DenseMacCounter(model):
            model(torch.zeros((2, 7)))
    assert not model._forward_hooks
