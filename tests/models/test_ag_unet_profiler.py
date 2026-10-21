from __future__ import annotations

import torch

from scripts import profile_ag_unet


def test_cuda_context_is_initialized_by_allocation_before_peak_reset(monkeypatch) -> None:
    events: list[str] = []
    cuda_device = object()

    class Initializer:
        def __del__(self) -> None:
            events.append("initializer_deleted")

    class FakeTensor:
        shape = (1, 3, 256, 256)

    class FakeLogits:
        shape = (1, 1, 128, 128)

    class FakeModel:
        def eval(self):
            return self

        def to(self, *, device, dtype):
            assert device is cuda_device
            assert dtype is torch.float32
            events.append("model_to")
            return self

        def __call__(self, image):
            assert isinstance(image, FakeTensor)
            events.append("forward")
            return FakeLogits()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "device", lambda value: cuda_device)

    def empty(size, *, device):
        assert size == 1
        assert device is cuda_device
        events.append("initializer_allocated")
        return Initializer()

    def synchronize(device):
        assert device is cuda_device
        events.append("synchronized")

    def empty_cache() -> None:
        events.append("cache_emptied")

    def reset_peak_memory_stats(device) -> None:
        assert device is cuda_device
        events.append("peaks_reset")

    def zeros(shape, *, device, dtype):
        assert shape == (1, 3, 256, 256)
        assert device is cuda_device
        assert dtype is torch.float32
        events.append("input_allocated")
        return FakeTensor()

    def build_model(**kwargs):
        assert kwargs == {"encoder_weights": "IMAGENET1K_V2", "freeze_encoder": False}
        return FakeModel()

    def get_device_name(device):
        assert device is cuda_device
        return "mock RTX 3070"

    def max_memory_allocated(device):
        assert device is cuda_device
        return 123

    def max_memory_reserved(device):
        assert device is cuda_device
        return 456

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(profile_ag_unet, "ApproximateAGUNet", build_model)
    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_memory_allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", max_memory_reserved)

    result = profile_ag_unet.profile_cuda_forward(device="cuda:0", precision="float32")

    assert events[:5] == [
        "initializer_allocated",
        "initializer_deleted",
        "synchronized",
        "cache_emptied",
        "peaks_reset",
    ]
    assert events.index("peaks_reset") < events.index("model_to") < events.index("input_allocated")
    assert result["success"] is True
    assert result["peak_allocated_bytes"] == 123
    assert result["peak_reserved_bytes"] == 456
