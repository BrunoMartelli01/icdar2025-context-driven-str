from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
from typing import Any, Iterator

import numpy as np
import pytest
import torch

from models.ag_unet_approx import (
    ARCHITECTURE_CONTRACT_SCHEMA_VERSION,
    ARCHITECTURE_IDENTITY,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT,
    EXPECTED_STATE_DICT_TENSOR_COUNT,
    ApproximateAGUNet,
    state_dict_schema_fingerprint,
)
from segmentation.training import seed_everything


ROOT = Path(__file__).resolve().parents[2]


def _model(*, freeze_encoder: bool = False) -> ApproximateAGUNet:
    return ApproximateAGUNet(encoder_weights=None, freeze_encoder=freeze_encoder)


def _global_rng_and_determinism_state() -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


@contextmanager
def _preserve_global_rng_and_determinism_state() -> Iterator[None]:
    state = _global_rng_and_determinism_state()
    try:
        yield
    finally:
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        torch.use_deterministic_algorithms(
            state["deterministic_algorithms"], warn_only=state["deterministic_warn_only"]
        )
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


def _assert_global_states_equal(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert before["python_rng"] == after["python_rng"]
    before_numpy, after_numpy = before["numpy_rng"], after["numpy_rng"]
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(before["torch_rng"], after["torch_rng"])
    if before["cuda_rng"] is None:
        assert after["cuda_rng"] is None
    else:
        assert len(before["cuda_rng"]) == len(after["cuda_rng"])
        assert all(
            torch.equal(expected, actual)
            for expected, actual in zip(before["cuda_rng"], after["cuda_rng"])
        )
    for name in (
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ):
        assert before[name] == after[name]


def test_frozen_architecture_identity_topology_and_machine_summary() -> None:
    model = _model()
    summary = model.architecture_summary()
    artifact = json.loads(
        (ROOT / "provenance/AG_UNET_ARCHITECTURE_M8.json").read_text(encoding="utf-8")
    )
    assert summary == artifact
    assert summary["architecture_identity"] == ARCHITECTURE_IDENTITY
    assert summary["schema_version"] == ARCHITECTURE_CONTRACT_SCHEMA_VERSION
    assert summary["modules"] == {
        "encoder": {
            "type": "torchvision.models.resnet50",
            "registered_but_not_executed": ["encoder.avgpool", "encoder.fc"],
            "stages": {
                "e0": {"module": "encoder.conv1+bn1+relu", "channels": [3, 64]},
                "e1": {"module": "encoder.maxpool+layer1", "channels": [64, 256]},
                "e2": {"module": "encoder.layer2", "channels": [256, 512]},
                "e3": {"module": "encoder.layer3", "channels": [512, 1024]},
                "e4": {"module": "encoder.layer4", "channels": [1024, 2048]},
            },
        },
        "bottleneck": {
            "module": "bottleneck",
            "type": "3x3 Conv2d(no bias)+BatchNorm2d+ReLU",
            "channels": [2048, 512],
        },
        "decoder3": model._decoder_contract(512, 1024, 256, 512),
        "decoder2": model._decoder_contract(512, 512, 128, 256),
        "decoder1": model._decoder_contract(256, 256, 64, 128),
        "decoder0": model._decoder_contract(128, 64, 32, 64),
        "head": {"module": "head", "type": "1x1 Conv2d", "channels": [64, 1]},
    }
    assert summary["initialization"]["selected_encoder_weights"] == "IMAGENET1K_V2"
    assert summary["initialization"]["selected_encoder_weights_classification"] == (
        "REPOSITORY_SPECIFIED"
    )
    assert summary["initialization"]["paper_encoder_weights"] == "UNRESOLVED (A-006)"


def test_exact_parameter_count_and_state_dict_schema_fingerprint() -> None:
    model = _model()
    assert model.parameter_count == EXPECTED_PARAMETER_COUNT == 48_454_733
    assert model.trainable_parameter_count == EXPECTED_PARAMETER_COUNT
    state = model.state_dict()
    assert len(state) == EXPECTED_STATE_DICT_TENSOR_COUNT == 456
    assert state_dict_schema_fingerprint(state) == EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT
    assert model.state_dict_schema_fingerprint == EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT
    assert next(iter(state)) == "encoder.conv1.weight"
    assert next(reversed(state)) == "head.bias"


def test_figure_derived_shape_contract_and_every_attention_skip_at_256() -> None:
    model = _model().eval()
    expected = model.architecture_summary()["canonical_tensors"]
    with torch.inference_mode():
        logits, trace = model.forward_with_trace(torch.zeros(1, 3, 256, 256))
    actual = {name: list(tensor.shape) for name, tensor in trace.items()}
    assert actual == expected
    assert list(logits.shape) == expected["logits"] == [1, 1, 128, 128]
    for level in (3, 2, 1, 0):
        gate = trace[f"gate{level}"]
        skip = trace[f"skip{level}"]
        coefficient = trace[f"attention{level}"]
        assert gate.shape[-2:] == skip.shape[-2:] == coefficient.shape[-2:]
        assert coefficient.shape[1] == 1
        assert torch.all((coefficient >= 0) & (coefficient <= 1))


@pytest.mark.parametrize("height,width", [(32, 32), (64, 96), (96, 160)])
def test_supported_divisible_input_sizes_have_half_resolution_output(
    height: int, width: int
) -> None:
    model = _model().eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, height, width))
    assert tuple(logits.shape) == (1, 1, height // 2, width // 2)


@pytest.mark.parametrize("height,width", [(16, 32), (32, 16), (63, 64), (64, 65), (65, 96)])
def test_unsupported_spatial_sizes_fail_actionably(height: int, width: int) -> None:
    with pytest.raises(ValueError, match=rf"at least 32 and divisible by 32.*{height}x{width}"):
        _model()(torch.zeros(1, 3, height, width))


def test_invalid_layout_batch_and_dtype_fail_actionably() -> None:
    model = _model()
    with pytest.raises(ValueError, match="Bx3xHxW"):
        model(torch.zeros(1, 1, 64, 64))
    with pytest.raises(ValueError, match="at least one image"):
        model(torch.zeros(0, 3, 64, 64))
    with pytest.raises(ValueError, match="floating-point dtype"):
        model(torch.zeros(1, 3, 64, 64, dtype=torch.uint8))
    with pytest.raises(ValueError, match="input dtype must match model parameters"):
        model(torch.zeros(1, 3, 64, 64, dtype=torch.float64))


def test_cpu_forward_backward_reaches_encoder_bottleneck_attention_decoder_and_head() -> None:
    model = _model().train()
    logits = model(torch.randn(2, 3, 64, 64))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert model.encoder.conv1.weight.grad is not None
    assert model.encoder.fc.weight.grad is None
    assert model.bottleneck[0].weight.grad is not None
    assert model.decoder3.attention.skip_projection[0].weight.grad is not None
    assert model.decoder2.attention.skip_projection[0].weight.grad is not None
    assert model.decoder1.attention.skip_projection[0].weight.grad is not None
    assert model.decoder0.attention.skip_projection[0].weight.grad is not None
    assert model.decoder0.decode[1][0].weight.grad is not None
    assert model.head.weight.grad is not None


def test_fixed_seed_initialization_is_deterministic_for_the_complete_state() -> None:
    before = _global_rng_and_determinism_state()
    with _preserve_global_rng_and_determinism_state():
        seed_everything(123, deterministic_algorithms=True)
        first = _model()
        seed_everything(123, deterministic_algorithms=True)
        second = _model()
        assert first.state_dict().keys() == second.state_dict().keys()
        for key, tensor in first.state_dict().items():
            assert torch.equal(tensor, second.state_dict()[key]), key
    _assert_global_states_equal(before, _global_rng_and_determinism_state())


def test_frozen_encoder_is_explicit_and_does_not_change_architecture_schema() -> None:
    trainable = _model(freeze_encoder=False)
    frozen = _model(freeze_encoder=True)
    assert not any(parameter.requires_grad for parameter in frozen.encoder.parameters())
    assert frozen.trainable_parameter_count == 22_897_701
    assert frozen.parameter_count == trainable.parameter_count == EXPECTED_PARAMETER_COUNT
    assert frozen.state_dict_schema_fingerprint == trainable.state_dict_schema_fingerprint


def test_strict_checkpoint_round_trip_and_all_incompatibility_failures() -> None:
    torch.manual_seed(1)
    source = _model()
    payload = source.make_checkpoint_payload()
    torch.manual_seed(2)
    restored = _model()
    restored.load_compatible_checkpoint(payload)
    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key]), key

    wrong_identity = dict(payload, architecture_identity="different-architecture")
    with pytest.raises(ValueError, match="architecture_identity mismatch"):
        restored.load_compatible_checkpoint(wrong_identity)

    wrong_contract_schema = dict(payload, architecture_contract_schema_version=999)
    with pytest.raises(ValueError, match="architecture_contract_schema_version mismatch"):
        restored.load_compatible_checkpoint(wrong_contract_schema)

    wrong_checkpoint_schema = dict(payload, checkpoint_schema_version=999)
    with pytest.raises(ValueError, match="checkpoint_schema_version mismatch"):
        restored.load_compatible_checkpoint(wrong_checkpoint_schema)

    wrong_fingerprint = dict(payload, state_dict_schema_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="state_dict_schema_fingerprint mismatch"):
        restored.load_compatible_checkpoint(wrong_fingerprint)

    missing_state = dict(payload["model_state_dict"])
    missing_state.pop("head.bias")
    with pytest.raises(ValueError, match=r"missing keys.*head\.bias"):
        restored.load_compatible_checkpoint(dict(payload, model_state_dict=missing_state))

    unexpected_state = dict(payload["model_state_dict"], **{"not_a_layer.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="unexpected keys.*not_a_layer.weight"):
        restored.load_compatible_checkpoint(dict(payload, model_state_dict=unexpected_state))

    wrong_shape_state = dict(payload["model_state_dict"])
    wrong_shape_state["head.weight"] = torch.zeros(2, 64, 1, 1)
    with pytest.raises(ValueError, match="shape mismatches.*head.weight"):
        restored.load_compatible_checkpoint(dict(payload, model_state_dict=wrong_shape_state))

    wrong_dtype_state = dict(payload["model_state_dict"])
    wrong_dtype_state["head.weight"] = wrong_dtype_state["head.weight"].double()
    with pytest.raises(ValueError, match="dtype mismatches.*head.weight"):
        restored.load_compatible_checkpoint(dict(payload, model_state_dict=wrong_dtype_state))

    with pytest.raises(ValueError, match="model_state_dict must be a mapping"):
        restored.load_compatible_checkpoint(dict(payload, model_state_dict=[]))
    non_string_key_state = dict(payload["model_state_dict"])
    non_string_key_state[1] = torch.zeros(1)
    with pytest.raises(ValueError, match="state keys must be strings"):
        restored.load_compatible_checkpoint(dict(payload, model_state_dict=non_string_key_state))
    with pytest.raises(ValueError, match="prohibits non-strict"):
        restored.load_state_dict(payload["model_state_dict"], strict=False)
