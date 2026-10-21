"""
models/recognizers.py

Lightweight, pluggable recognizer pool for T1 (full image) and T3 (crop) text recognition.

Adapters provided:
  • TrOCRRecognizer (HF: microsoft/trocr-base-printed)
  • ParseqRecognizer (qualified official runtime/checkpoint; APPROXIMATE only)
  • TesseractRecognizer (pytesseract)    [optional; fallback if installed]
  • DummyRecognizer (always returns empty string) — demo-only last resort

Demo policy (not the scientific M5 path):
  • The pool tries models in the order given by `cfg["models"]`.
  • If a requested adapter is unavailable, we raise a clear error unless
    a working alternative exists (then we warn & continue).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import warnings

import numpy as np
import cv2


# ---------- Base interface ----------

class BaseRecognizer:
    """Interface all recognizers must implement."""

    name: str = "base"

    def predict(self, img_rgb_np: np.ndarray) -> str:
        """Return a single text string for the given RGB image."""
        raise NotImplementedError


# ---------- Tesseract adapter (optional) ----------

class TesseractRecognizer(BaseRecognizer):
    name = "tesseract"

    def __init__(self, lang: str = "eng") -> None:
        try:
            import pytesseract  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "pytesseract is not installed or Tesseract binary is missing."
            ) from e
        self.lang = lang

    def predict(self, img_rgb_np: np.ndarray) -> str:
        import pytesseract
        # Convert to grayscale and apply mild threshold to help OCR
        gray = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(thr, lang=self.lang, config="--psm 6")
        return txt.strip()


# ---------- TrOCR (HF) adapter ----------

class TrOCRRecognizer(BaseRecognizer):
    name = "trocr"

    def __init__(self, repo_id: str = "microsoft/trocr-base-printed", revision: Optional[str] = None, device: str = "cuda"):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except Exception as e:
            raise RuntimeError(
                "transformers (TrOCR) not available. Install transformers, torchvision, and torch."
            ) from e

        self.device = device
        self._torch = __import__("torch")
        self.proc = TrOCRProcessor.from_pretrained(repo_id, revision=revision)
        self.model = VisionEncoderDecoderModel.from_pretrained(repo_id, revision=revision).to(device)
        self.model.eval()

    def predict(self, img_rgb_np: np.ndarray) -> str:
        torch = self._torch
        # Convert to PIL via cv2 → RGB already
        from PIL import Image
        pil = Image.fromarray(img_rgb_np)
        with torch.inference_mode():
            inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
            out_ids = self.model.generate(**inputs, max_new_tokens=64)
            text = self.proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()


# ---------- PARSeq (HF) adapter (optional) ----------

class ParseqRecognizer(BaseRecognizer):
    name = "parseq"

    def __init__(self, runtime_source: str, checkpoint: str, device: str = "cuda"):
        from pathlib import Path
        from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer

        self.adapter = ParseqApproxRecognizer(
            ParseqApproxConfig(Path(runtime_source), Path(checkpoint), device)
        )

    def predict(self, img_rgb_np: np.ndarray) -> str:
        return self.adapter.recognize(
            img_rgb_np, input_role="T3_localized_crop"
        ).raw_text

    def predict_role(self, img_rgb_np: np.ndarray, role: str) -> str:
        return self.adapter.recognize(img_rgb_np, input_role=role).raw_text


# ---------- Dummy adapter ----------

class DummyRecognizer(BaseRecognizer):
    name = "dummy"

    def predict(self, img_rgb_np: np.ndarray) -> str:
        return ""


# ---------- Pool ----------

class RecognizerPool:
    """
    RecognizerPool(cfg) manages {full, crop} predictions.

    cfg example:
        {
          "models": ["trocr", "tesseract"],  # priority order
          "trocr": {"repo_id": "microsoft/trocr-base-printed", "revision": null},
          "parseq": {"repo_id": "baudm/parseq", "variant": "small"},
          "device": "cuda"
        }
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        device = cfg.get("device", "cuda")
        requested: List[str] = cfg.get("models", ["trocr"])

        self.adapters: List[BaseRecognizer] = []
        for name in requested:
            try:
                if name.lower() == "trocr":
                    params = cfg.get("trocr", {})
                    self.adapters.append(
                        TrOCRRecognizer(
                            repo_id=params.get("repo_id", "microsoft/trocr-base-printed"),
                            revision=params.get("revision", None),
                            device=device,
                        )
                    )
                elif name.lower() == "parseq":
                    params = cfg.get("parseq", {})
                    self.adapters.append(
                        ParseqRecognizer(
                            runtime_source=params["runtime_source"],
                            checkpoint=params["checkpoint"],
                            device=device,
                        )
                    )
                elif name.lower() == "tesseract":
                    params = cfg.get("tesseract", {})
                    self.adapters.append(TesseractRecognizer(lang=params.get("lang", "eng")))
                elif name.lower() == "dummy":
                    self.adapters.append(DummyRecognizer())
                else:
                    raise ValueError(f"Unknown recognizer '{name}'")
            except Exception as e:
                warnings.warn(f"Could not initialize recognizer '{name}': {e}")

        if not self.adapters:
            warnings.warn("No recognizers available; using DummyRecognizer.")
            self.adapters = [DummyRecognizer()]

        # Demo-only policy. Scientific M5 code constructs the provenance-locked
        # adapter directly and never catches model-loading failures.
        self.primary = self.adapters[0]

    def predict_full(self, img_rgb_np: np.ndarray) -> str:
        """Predict on the full image using the primary adapter."""
        if isinstance(self.primary, ParseqRecognizer):
            return self.primary.predict_role(img_rgb_np, "T1_full_image")
        return self.primary.predict(img_rgb_np)

    def predict_crop(self, crop_rgb_np: np.ndarray) -> str:
        """Predict on a crop using the primary adapter."""
        if isinstance(self.primary, ParseqRecognizer):
            return self.primary.predict_role(crop_rgb_np, "T3_localized_crop")
        return self.primary.predict(crop_rgb_np)
