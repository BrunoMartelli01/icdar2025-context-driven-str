from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from caption.blip2_captioner import (
    BLIP2_REVISION, BLIP2ApproxConfig, BLIP2Captioner, CaptionResult, DEFAULT_GENERATION,
    DEFAULT_PROMPT, canonical_torch_device, stable_hash,
)
from caption.cache import CachedCaptioner, CaptionCache, caption_cache_key, image_sha256


def _result(text="caption"):
    return CaptionResult(
        text, "Salesforce/blip2-flan-t5-xl", BLIP2_REVISION,
        stable_hash(DEFAULT_PROMPT), dict(DEFAULT_GENERATION), stable_hash(DEFAULT_GENERATION),
        "float16", "cuda:0", "single_device", "none", 0.1, 123, 456,
    )


def test_candidate_is_pinned_and_quantization_is_not_automatic():
    config = BLIP2ApproxConfig()
    assert config.revision == BLIP2_REVISION
    assert config.quantization == "none"


def test_caption_result_preserves_generation_and_provenance_metadata():
    class Inputs(dict):
        def to(self, *args, **kwargs):
            return self
    class Processor:
        def __call__(self, **kwargs):
            return Inputs(pixel_values=torch.zeros(1, 3, 224, 224), input_ids=torch.ones(1, 2, dtype=torch.long))
        def batch_decode(self, output, **kwargs):
            return [" raw caption "]
    class Model:
        def generate(self, **kwargs):
            assert kwargs["num_beams"] == 3 and kwargs["do_sample"] is False
            return torch.ones(1, 2, dtype=torch.long)
    captioner = BLIP2Captioner(BLIP2ApproxConfig())
    captioner._processor, captioner._model = Processor(), Model()
    result = captioner.caption(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.raw_caption == "raw caption"
    assert result.revision == BLIP2_REVISION
    assert result.prompt_hash == stable_hash(DEFAULT_PROMPT)
    assert result.success and not result.oom


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_single_device_uses_one_valid_torch_device_for_model_and_inputs(monkeypatch, device):
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    calls = {}

    class Inputs(dict):
        def to(self, *, device, dtype):
            calls["input_device"] = device
            calls["input_dtype"] = dtype
            return self

    class Processor:
        def __call__(self, **kwargs):
            return Inputs(pixel_values=torch.zeros(1, 3, 2, 2), input_ids=torch.ones(1, 1, dtype=torch.long))
        def batch_decode(self, output, **kwargs):
            return ["caption"]

    class Model:
        def to(self, device):
            calls["model_device"] = device
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            return torch.ones(1, 1, dtype=torch.long)

    def load_model(*args, **kwargs):
        calls["load_kwargs"] = kwargs
        return Model()

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(Blip2ForConditionalGeneration, "from_pretrained", load_model)
    if device.startswith("cuda"):
        calls["cuda_order"] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch,
            "empty",
            lambda *shape, device: (
                calls["cuda_order"].append(("initialize", device)), object()
            )[1],
        )
        monkeypatch.setattr(
            torch.cuda,
            "empty_cache",
            lambda: calls["cuda_order"].append(("empty_cache", None)),
        )
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                            lambda index: (
                                calls["cuda_order"].append(("reset", index)),
                                calls.setdefault("reset_index", index),
                            ))
        monkeypatch.setattr(torch.cuda, "synchronize",
                            lambda index: calls.setdefault("sync_index", index))
        monkeypatch.setattr(torch.cuda, "max_memory_allocated",
                            lambda index: calls.setdefault("allocated_index", index) or 123)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved",
                            lambda index: calls.setdefault("reserved_index", index) or 456)

    captioner = BLIP2Captioner(BLIP2ApproxConfig(device=device))
    result = captioner.caption(np.zeros((2, 2, 3), dtype=np.uint8))
    expected = canonical_torch_device(device)
    assert calls["model_device"] == expected
    assert calls["input_device"] == expected
    assert "device_map" not in calls["load_kwargs"]
    assert result.device == str(expected)
    if device.startswith("cuda"):
        assert calls["cuda_order"] == [
            ("initialize", expected), ("empty_cache", None), ("reset", 0)
        ]
        assert type(calls["reset_index"]) is int
        assert type(calls["sync_index"]) is int
        assert type(calls["allocated_index"]) is int
        assert type(calls["reserved_index"]) is int
        assert calls["reset_index"] == calls["sync_index"] == 0
        assert calls["allocated_index"] == calls["reserved_index"] == 0
        assert result.peak_allocated_bytes == 123
        assert result.peak_reserved_bytes == 456


def test_cpu_offload_uses_accelerate_device_map_and_reports_placement(monkeypatch):
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    calls = {}

    class Inputs(dict):
        def to(self, *, device, dtype):
            calls["input_device"] = device
            calls["input_dtype"] = dtype
            return self

    class Processor:
        def __call__(self, **kwargs):
            return Inputs(pixel_values=torch.zeros(1, 3, 2, 2), input_ids=torch.ones(1, 1, dtype=torch.long))
        def batch_decode(self, output, **kwargs):
            return ["caption"]

    class Model:
        hf_device_map = {"vision_model": 0, "language_model": "cpu"}
        def to(self, device):
            raise AssertionError("cpu_offload must not call model.to")
        def eval(self):
            return self
        def generate(self, **kwargs):
            return torch.ones(1, 1, dtype=torch.long)

    def load_model(*args, **kwargs):
        calls["load_kwargs"] = kwargs
        return Model()

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    monkeypatch.setattr(Blip2ForConditionalGeneration, "from_pretrained", load_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "empty", lambda *shape, device: object())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda index: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda index: 456)

    result = BLIP2Captioner(BLIP2ApproxConfig(
        device="cuda:0", dtype="float16", placement="cpu_offload", quantization="none",
    )).caption(np.zeros((2, 2, 3), dtype=np.uint8))

    assert calls["load_kwargs"]["device_map"] == "auto"
    assert "quantization_config" not in calls["load_kwargs"]
    assert calls["input_device"] == torch.device("cuda:0")
    assert result.hardware_deviation.startswith("HARDWARE_DEVIATION")
    assert result.device_placement_summary["module_count_by_device"] == {"cuda:0": 1, "cpu": 1}
    assert result.load_time_seconds is not None
    assert result.generation_time_seconds is not None
    assert result.total_latency_seconds == result.latency_seconds
    assert result.cpu_peak_rss_bytes > 0


def test_cpu_offload_requires_cuda_execution_device():
    with pytest.raises(ValueError, match="requires a CUDA execution device"):
        BLIP2ApproxConfig(device="cpu", placement="cpu_offload")


@pytest.mark.parametrize("device", ["", "gpu", "cuda:1", " cuda:0", "cuda:0 "])
def test_invalid_or_noncanonical_devices_are_rejected(device):
    with pytest.raises(ValueError, match="Invalid device"):
        BLIP2ApproxConfig(device=device)


def test_profiler_preserves_json_error_schema_for_invalid_device(monkeypatch, tmp_path):
    from scripts import profile_blip2

    image_path, output_path = tmp_path / "image.png", tmp_path / "profile.json"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(sys, "argv", [
        "profile_blip2", "--image", str(image_path), "--output", str(output_path),
        "--device", "cuda:1", "--dtype", "float16", "--placement", "single_device",
        "--quantization", "none", "--model-lifecycle", "standalone_blip2_only",
    ])
    with pytest.raises(SystemExit, match="2"):
        profile_blip2.main()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["success"] is False and payload["oom"] is False
    assert payload["error_type"] == "ValueError"
    assert payload["device"] == "cuda:1"
    assert "peak_allocated_bytes" in payload and "peak_reserved_bytes" in payload


def test_cache_key_depends_on_all_required_fields(tmp_path: Path):
    fields = ["image", BLIP2_REVISION, stable_hash(DEFAULT_PROMPT), stable_hash(DEFAULT_GENERATION)]
    base = caption_cache_key(*fields)
    for index in range(4):
        changed = fields.copy()
        changed[index] += "x"
        assert caption_cache_key(*changed) != base

    calls = 0
    def generate():
        nonlocal calls
        calls += 1
        return _result()
    cache = CaptionCache(tmp_path)
    kwargs = dict(
        image_bytes=b"pixels", model_revision=BLIP2_REVISION,
        prompt_hash=stable_hash(DEFAULT_PROMPT),
        generation_config_hash=stable_hash(DEFAULT_GENERATION), generate=generate,
    )
    first, was_cached = cache.get_or_generate(**kwargs)
    second, now_cached = cache.get_or_generate(**kwargs)
    assert not was_cached and now_cached and calls == 1
    assert first.raw_caption == second.raw_caption == "caption"


def test_cached_captioner_avoids_second_generation(tmp_path: Path):
    class Generator:
        config = BLIP2ApproxConfig()
        calls = 0
        def caption(self, image):
            self.calls += 1
            return _result()
    generator = Generator()
    cached = CachedCaptioner(generator, CaptionCache(tmp_path))
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    assert cached.caption(image).raw_caption == "caption"
    assert not cached.last_cache_hit
    assert cached.caption(image).raw_caption == "caption"
    assert cached.last_cache_hit and generator.calls == 1
