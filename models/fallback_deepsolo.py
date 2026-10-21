"""Retired in-process fallback hook.

DeepSolo is dependency-incompatible with the main environment. Scientific and
approximate callers must use the versioned adapters in :mod:`deepsolo.adapter`.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def run_deepsolo_once(img_rgb_np: np.ndarray, config_path: Optional[str] = None, ckpt_path: Optional[str] = None) -> str:
    raise RuntimeError(
        "The in-process DeepSolo/MMOCR stub is disabled. Use the versioned isolated "
        "DeepSoloSubprocessAdapter with an image path, runtime manifest, explicit "
        "candidate ID, and explicit reduction policy."
    )


def maybe_deepsolo(img_rgb_np: np.ndarray, config_path: Optional[str] = None, ckpt_path: Optional[str] = None) -> str:
    """Fail explicitly instead of silently fabricating an empty fallback output."""
    return run_deepsolo_once(img_rgb_np, config_path=config_path, ckpt_path=ckpt_path)
