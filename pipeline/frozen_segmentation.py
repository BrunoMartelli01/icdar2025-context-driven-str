"""Frozen author-protocol segmentation adapter for downstream recognition.

The adapter reuses the already-qualified M10 preprocessing and checkpoint loader.
It thresholds the native 128x128 logits at 0.5, then nearest-neighbor backprojects
that binary prediction to the original image geometry.  Evaluation validity masks
are deliberately outside this inference boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from segmentation.config import load_segmentation_config
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint


FROZEN_ARCHITECTURE = "author-clarified-ag-unet-2048x1024-v1"
FROZEN_LOGITS_SHAPE = (128, 128)
FROZEN_THRESHOLD = 0.5


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def backproject_binary_mask(mask: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor backproject a 2-D binary mask without changing its values."""

    if mask.ndim != 2 or tuple(mask.shape) != FROZEN_LOGITS_SHAPE:
        raise ValueError("Frozen downstream mask must have shape 128x128")
    if mask.dtype != np.uint8 or not set(np.unique(mask).tolist()).issubset({0, 1}):
        raise ValueError("Frozen downstream mask must be binary uint8")
    height, width = output_hw
    if height < 1 or width < 1:
        raise ValueError("Backprojection output dimensions must be positive")
    tensor = torch.from_numpy(mask).to(torch.float32)[None, None]
    projected = F.interpolate(tensor, size=(height, width), mode="nearest")
    return projected[0, 0].to(torch.uint8).numpy()


@dataclass(frozen=True)
class FrozenSegmentationConfig:
    evaluation_config: Path
    checkpoint: Path
    device: str = "cuda:0"


class FrozenAuthorProtocolSegmenter:
    """One-shot inference wrapper for the frozen ``epoch-0098.pt`` baseline."""

    def __init__(self, config: FrozenSegmentationConfig) -> None:
        resolved = load_segmentation_config(config.evaluation_config)
        expected = {
            "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction",
            "architecture.identity": FROZEN_ARCHITECTURE,
            "architecture.freeze_encoder": True,
            "architecture.frozen_batchnorm_state": True,
            "transforms.image_size": [256, 256],
            "transforms.mask_size": [128, 128],
            "transforms.normalization": "imagenet",
            "evaluation.probability_threshold": FROZEN_THRESHOLD,
            "evaluation.threshold_comparison": "greater_than_or_equal",
            "evaluation.threshold_resolution": "native_128x128_logits",
            "evaluation.dont_care": "foreground",
            "evaluation.valid_mask": "all_pixels",
        }
        for dotted, wanted in expected.items():
            current: Any = resolved
            for part in dotted.split("."):
                current = getattr(current, part)
            comparable = list(current) if isinstance(wanted, list) else current
            if comparable != wanted:
                raise ValueError(
                    "Frozen segmentation config mismatch at {}: expected {!r}, got {!r}".format(
                        dotted, wanted, current
                    )
                )

        model = _model_from_config(resolved)
        checkpoint_record = validate_checkpoint(
            config.checkpoint.resolve(),
            model,
            str(resolved.checkpoint.sha256),
            FROZEN_ARCHITECTURE,
        )
        if not all(not parameter.requires_grad for parameter in model.encoder.parameters()):
            raise RuntimeError("Frozen segmentation encoder unexpectedly has trainable parameters")

        self.config = config
        self.segmentation_config = resolved
        self.model = model.eval().to(torch.device(config.device))
        self.checkpoint_record = checkpoint_record

    @torch.inference_mode()
    def __call__(self, image_rgb: np.ndarray) -> dict[str, Any]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("Frozen segmentation expects an RGB uint8 HxWx3 array")
        image = Image.fromarray(image_rgb, mode="RGB")
        inputs = _preprocess(image).unsqueeze(0).to(self.config.device)
        logits = self.model(inputs)
        if tuple(logits.shape) != (1, 1, *FROZEN_LOGITS_SHAPE):
            raise RuntimeError(
                "Frozen segmentation expected logits [1,1,128,128], got {}".format(
                    list(logits.shape)
                )
            )
        probability_native = (
            logits.sigmoid()[0, 0].to(torch.float32).cpu().numpy()
        )
        probability_native = np.ascontiguousarray(probability_native)
        native = np.ascontiguousarray(
            (probability_native >= FROZEN_THRESHOLD).astype(np.uint8)
        )
        projected = backproject_binary_mask(native, image_rgb.shape[:2])
        return {
            "mask": projected,
            "mask_native": native,
            # Additive, audit-oriented output.  The paper-faithful binary path
            # above remains byte-identical; practical multi-threshold proposal
            # diagnostics consume these probabilities without a second model
            # inference.
            "probability_native": probability_native,
            "metadata": {
                "architecture": FROZEN_ARCHITECTURE,
                "checkpoint": dict(self.checkpoint_record),
                "input_shape": list(inputs.shape),
                "logits_shape": list(logits.shape),
                "native_mask_shape": list(native.shape),
                "native_mask_sha256": _array_sha256(native),
                "native_probability_shape": list(probability_native.shape),
                "native_probability_dtype": probability_native.dtype.str,
                "native_probability_sha256": _array_sha256(probability_native),
                "output_mask_shape": list(projected.shape),
                "output_mask_sha256": _array_sha256(projected),
                "probability_activation": "sigmoid",
                "threshold": FROZEN_THRESHOLD,
                "threshold_comparison": "greater_than_or_equal",
                "backprojection": "nearest after binary threshold",
                "validity_mask": "not loaded or applied; segmentation evaluation only",
                "encoder_frozen": True,
                "classification": "approximate reproduction",
            },
        }

    def release(self) -> dict[str, Any]:
        had_model = self.model is not None
        self.model = None
        return {"model_reference_released": had_model}


__all__ = [
    "FROZEN_ARCHITECTURE",
    "FROZEN_LOGITS_SHAPE",
    "FROZEN_THRESHOLD",
    "FrozenAuthorProtocolSegmenter",
    "FrozenSegmentationConfig",
    "backproject_binary_mask",
]
