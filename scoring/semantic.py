"""Pinned APPROXIMATE MPNet sentence embedding integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


MPNET_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MPNET_WEIGHT_SHA256 = "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658"


@dataclass(frozen=True)
class MPNetApproxConfig:
    model_id: str = MPNET_MODEL_ID
    revision: str = MPNET_REVISION
    pooling: Literal["mean"] = "mean"
    normalize_embeddings: bool = True
    truncation: bool = True
    max_sequence_length: int = 384
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.model_id != MPNET_MODEL_ID or self.revision != MPNET_REVISION:
            raise ValueError("APPROXIMATE MPNet must use the provenance-defined candidate")
        if self.pooling != "mean" or not self.truncation:
            raise ValueError("This candidate is qualified only with mean pooling and truncation")


class MPNetSemanticSimilarity:
    def __init__(self, config: MPNetApproxConfig, *, model: Any | None = None) -> None:
        self.config = config
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                config.model_id,
                revision=config.revision,
                device=config.device,
                model_kwargs={"torch_dtype": getattr(torch, config.precision)},
            )
        model.max_seq_length = config.max_sequence_length
        self.model = model

    def embeddings(self, texts: list[str]) -> np.ndarray:
        values = self.model.encode(
            [text or "" for text in texts],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float64)

    def compare_candidates(self, t1: str, t3: str, caption: str) -> tuple[float, float]:
        e1, e3, e2 = self.embeddings([t1, t3, caption])
        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            return 0.0 if denominator == 0.0 else float(np.clip(np.dot(a, b) / denominator, -1, 1))
        return cosine(e1, e2), cosine(e3, e2)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "status": "approximate reproduction",
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "weight_sha256": MPNET_WEIGHT_SHA256,
            "pooling": self.config.pooling,
            "normalization": self.config.normalize_embeddings,
            "truncation": self.config.truncation,
            "max_sequence_length": self.config.max_sequence_length,
            "precision": self.config.precision,
            "device": self.config.device,
        }

    def release(self) -> dict[str, Any]:
        had_model = self.model is not None
        self.model = None
        return {"model_reference_released": had_model}
