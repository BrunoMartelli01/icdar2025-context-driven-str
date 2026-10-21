"""Pinned APPROXIMATE BLIP-2 integration and auditable inference metadata."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image


BLIP2_MODEL_ID = "Salesforce/blip2-flan-t5-xl"
BLIP2_REVISION = "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"
DEFAULT_PROMPT = (
    "Describe this image comprehensively, focusing on all visible text elements. "
    "Mention the environment, text-bearing objects, positions, and how the text "
    "relates to the scene context."
)
DEFAULT_GENERATION = {
    "max_new_tokens": 80,
    "min_new_tokens": 40,
    "length_penalty": 1.0,
    "num_beams": 3,
    "do_sample": False,
}
PRACTICAL_ANTI_REPETITION_GENERATION = {
    "max_new_tokens": 80,
    "min_new_tokens": 20,
    "length_penalty": 1.0,
    "num_beams": 3,
    "do_sample": False,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.1,
}
GENERATION_PROFILES = {
    "repository-v1": DEFAULT_GENERATION,
    "practical-antirepetition-v1": PRACTICAL_ANTI_REPETITION_GENERATION,
}
CANONICAL_DEVICES = ("cpu", "cuda", "cuda:0")


def canonical_torch_device(value: str):
    """Validate the supported CLI/config spellings and return ``torch.device``."""
    import torch

    if value not in CANONICAL_DEVICES:
        raise ValueError(
            f"Invalid device {value!r}; expected one of: {', '.join(CANONICAL_DEVICES)}"
        )
    return torch.device(value)


def cuda_device_index(device) -> int | None:
    """Resolve a CUDA device for allocator APIs without passing device strings."""
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    resolved = device.index if device.index is not None else torch.cuda.current_device()
    return int(resolved)


def stable_hash(value: Any) -> str:
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cpu_peak_rss_bytes() -> int:
    """Return Linux process peak RSS; Ubuntu reports ``ru_maxrss`` in KiB."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


@dataclass(frozen=True)
class BLIP2ApproxConfig:
    model_id: str = BLIP2_MODEL_ID
    revision: str = BLIP2_REVISION
    prompt: str = DEFAULT_PROMPT
    generation: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_GENERATION))
    device: str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    placement: Literal["single_device", "cpu_offload"] = "single_device"
    quantization: Literal["none", "8bit", "4bit"] = "none"

    def __post_init__(self) -> None:
        if self.model_id != BLIP2_MODEL_ID or self.revision != BLIP2_REVISION:
            raise ValueError("APPROXIMATE BLIP-2 must use the provenance-defined candidate")
        resolved_device = canonical_torch_device(self.device)
        if self.placement == "cpu_offload" and resolved_device.type != "cuda":
            raise ValueError("cpu_offload requires a CUDA execution device")


@dataclass(frozen=True)
class CaptionResult:
    raw_caption: str
    model_id: str
    revision: str
    prompt_hash: str
    generation_config: dict[str, Any]
    generation_config_hash: str
    dtype: str
    device: str
    placement: str
    quantization: str
    latency_seconds: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    success: bool = True
    oom: bool = False
    status: str = "approximate reproduction"
    latency_scope: str = "caption call; includes model load when cold, plus preprocessing/generation/decode"
    load_time_seconds: float | None = None
    generation_time_seconds: float | None = None
    total_latency_seconds: float | None = None
    cpu_peak_rss_bytes: int | None = None
    device_placement_summary: dict[str, Any] = field(default_factory=dict)
    hardware_deviation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BLIP2Captioner:
    """Loads only the pinned repository candidate; no automatic quantization."""

    def __init__(
        self,
        config: BLIP2ApproxConfig | None = None,
        *,
        repo_id: str = BLIP2_MODEL_ID,
        revision: str | None = None,
        device: str = "cpu",
        max_len: int = 80,
        min_len: int = 40,
    ) -> None:
        # Keyword compatibility is retained for the historical demo runner, but
        # it is forcibly pinned to the same approximate candidate and records the
        # fully resolved values in every result.
        if config is None:
            generation = dict(DEFAULT_GENERATION)
            generation.update(max_new_tokens=max_len, min_new_tokens=min_len)
            config = BLIP2ApproxConfig(
                model_id=repo_id,
                revision=revision or BLIP2_REVISION,
                generation=generation,
                device=device,
            )
        if isinstance(config, str):
            repo_id, config = config, None
            generation = dict(DEFAULT_GENERATION)
            generation.update(max_new_tokens=max_len, min_new_tokens=min_len)
            config = BLIP2ApproxConfig(
                model_id=repo_id, revision=revision or BLIP2_REVISION,
                generation=generation, device=device,
            )
        self.config = config
        self.device = canonical_torch_device(config.device)
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

        dtype = getattr(torch, self.config.dtype)
        kwargs: dict[str, Any] = {
            "revision": self.config.revision,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.config.quantization != "none":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.config.quantization == "8bit",
                load_in_4bit=self.config.quantization == "4bit",
            )
            kwargs["device_map"] = "auto"
        elif self.config.placement == "cpu_offload":
            # Transformers delegates this immutable-revision checkpoint to
            # Accelerate, which infers a module map across the selected GPU and
            # CPU. No quantization or alternate checkpoint is introduced.
            kwargs["device_map"] = "auto"
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id, revision=self.config.revision
        )
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.model_id, **kwargs
        )
        if self.config.placement == "single_device" and self.config.quantization == "none":
            self._model.to(self.device)
        self._model.eval()

    def _device_placement_summary(self) -> dict[str, Any]:
        raw_map = getattr(self._model, "hf_device_map", None)
        if raw_map is None:
            return {"method": "torch.nn.Module.to", "target": str(self.device)}
        normalized: dict[str, str] = {}
        counts: dict[str, int] = {}
        for module, placement in raw_map.items():
            label = f"cuda:{placement}" if isinstance(placement, int) else str(placement)
            normalized[str(module)] = label
            counts[label] = counts.get(label, 0) + 1
        if self.config.placement == "cpu_offload":
            if "cpu" not in counts or not any(label.startswith("cuda") for label in counts):
                raise RuntimeError(
                    "cpu_offload requires the resolved Accelerate map to contain both CUDA and CPU modules"
                )
        return {
            "method": "transformers.device_map=auto (Accelerate dispatch)",
            "module_count_by_device": counts,
            "hf_device_map": normalized,
        }

    def caption(self, image_rgb: np.ndarray) -> CaptionResult:
        import torch

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("BLIP-2 expects an RGB uint8 HxWx3 array")
        cuda_index = cuda_device_index(self.device)
        if cuda_index is not None:
            initialization_tensor = torch.empty(1, device=self.device)
            del initialization_tensor
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_index)
        started = time.perf_counter()
        load_started = time.perf_counter()
        self._load()
        load_time = time.perf_counter() - load_started
        placement_summary = self._device_placement_summary()
        inputs = self._processor(
            images=Image.fromarray(image_rgb, mode="RGB"),
            text=self.config.prompt,
            return_tensors="pt",
        )
        if self.config.placement == "single_device" and self.config.quantization == "none":
            inputs = inputs.to(device=self.device, dtype=getattr(torch, self.config.dtype))
        elif self.config.placement == "cpu_offload":
            inputs = inputs.to(device=self.device, dtype=getattr(torch, self.config.dtype))
        else:
            inputs = inputs.to(next(self._model.parameters()).device)
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(**inputs, **self.config.generation)
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
        generation_time = time.perf_counter() - generation_started
        caption = self._processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        elapsed = time.perf_counter() - started
        deviations = []
        if self.config.dtype != "float32":
            deviations.append(
                "HARDWARE_DEVIATION: {} BLIP-2 inference".format(self.config.dtype)
            )
        if self.config.placement == "cpu_offload":
            deviations.append("HARDWARE_DEVIATION: Accelerate CPU offload")
        if self.config.quantization != "none":
            deviations.append(
                "HARDWARE_DEVIATION: {} BLIP-2 quantization".format(
                    self.config.quantization
                )
            )
        return CaptionResult(
            raw_caption=caption,
            model_id=self.config.model_id,
            revision=self.config.revision,
            prompt_hash=stable_hash(self.config.prompt),
            generation_config=dict(self.config.generation),
            generation_config_hash=stable_hash(self.config.generation),
            dtype=self.config.dtype,
            device=str(self.device),
            placement=self.config.placement,
            quantization=self.config.quantization,
            latency_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(cuda_index) if cuda_index is not None else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(cuda_index) if cuda_index is not None else None,
            load_time_seconds=load_time,
            generation_time_seconds=generation_time,
            total_latency_seconds=elapsed,
            cpu_peak_rss_bytes=cpu_peak_rss_bytes(),
            device_placement_summary=placement_summary,
            hardware_deviation="; ".join(deviations) if deviations else None,
        )

    def __call__(self, image_rgb: np.ndarray) -> str:
        """Compatibility surface; scientific callers should retain ``CaptionResult``."""
        return self.caption(image_rgb).raw_caption

    def release(self) -> dict[str, Any]:
        """Drop the dispatched model/processor before an isolated fallback starts."""
        had_model, had_processor = self._model is not None, self._processor is not None
        self._model = None
        self._processor = None
        return {
            "model_reference_released": had_model,
            "processor_reference_released": had_processor,
        }
