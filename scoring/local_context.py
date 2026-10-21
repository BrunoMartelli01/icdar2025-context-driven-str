"""Training-free local caption matching for a separate practical approximation.

The paper-faithful whole-caption scorer is intentionally not changed.  This
module compares an STR string with short, contiguous windows from the caption
and maximizes the *paired* ``0.6*S + 0.4*L`` score over one shared window.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

import numpy as np
from rapidfuzz.fuzz import ratio


TOKEN_PATTERN = re.compile(r"\w+(?:['’.-]\w+)*|[^\w\s]", re.UNICODE)
ALPHA = 0.6
BETA = 0.4
TAU = 0.8
LOCAL_VARIANTS = ("exact_length", "neighbor_lengths")


class Embedder(Protocol):
    def embeddings(self, texts: list[str]) -> np.ndarray: ...


def caption_tokens(text: str) -> tuple[str, ...]:
    """Tokenize without external NLP state; punctuation remains auditable."""
    return tuple(TOKEN_PATTERN.findall(text or ""))


def candidate_window_lengths(candidate: str, *, variant: str) -> tuple[int, ...]:
    if variant not in LOCAL_VARIANTS:
        raise ValueError(f"unknown local-context variant: {variant}")
    count = len(caption_tokens(candidate))
    if count == 0:
        return ()
    if variant == "exact_length":
        return (count,)
    return tuple(sorted({max(1, count - 1), count, count + 1}))


def caption_windows(caption: str, lengths: Sequence[int]) -> tuple[str, ...]:
    tokens = caption_tokens(caption)
    values: list[str] = []
    seen: set[str] = set()
    for length in sorted(set(int(value) for value in lengths)):
        if length <= 0:
            raise ValueError("caption-window lengths must be positive")
        if length > len(tokens):
            continue
        for start in range(len(tokens) - length + 1):
            value = " ".join(tokens[start:start + length])
            if value not in seen:
                seen.add(value)
                values.append(value)
    return tuple(values)


def score_local_candidates(
    candidate_texts: Sequence[str], caption: str, embedder: Embedder, *, variant: str,
) -> tuple[dict[str, Any], ...]:
    """Score candidates against same-window semantic and lexical evidence.

    The candidate list order and earliest-window tie rule are deterministic.
    Empty candidates fail closed with a zero confidence.
    """
    if variant not in LOCAL_VARIANTS:
        raise ValueError(f"unknown local-context variant: {variant}")
    texts = [str(value or "") for value in candidate_texts]
    per_candidate_windows = [
        caption_windows(caption, candidate_window_lengths(text, variant=variant))
        for text in texts
    ]
    unique_values: list[str] = []
    index: dict[str, int] = {}
    for value in [*texts, *(window for values in per_candidate_windows for window in values)]:
        if value and value not in index:
            index[value] = len(unique_values)
            unique_values.append(value)
    embeddings = (
        np.asarray(embedder.embeddings(unique_values), dtype=np.float64)
        if unique_values else np.empty((0, 0), dtype=np.float64)
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(unique_values):
        raise RuntimeError("embedder returned an invalid local-context matrix")

    output = []
    for text, windows in zip(texts, per_candidate_windows):
        best: dict[str, Any] | None = None
        for window_index, window in enumerate(windows):
            semantic = float(np.clip(
                np.dot(embeddings[index[text]], embeddings[index[window]]), -1.0, 1.0
            ))
            lexical = float(ratio(text, window)) / 100.0
            confidence = ALPHA * semantic + BETA * lexical
            row = {
                "window": window,
                "window_index": window_index,
                "semantic_similarity": semantic,
                "lexical_similarity": lexical,
                "confidence": confidence,
            }
            if best is None or confidence > best["confidence"]:
                best = row
        output.append(best or {
            "window": "", "window_index": None, "semantic_similarity": 0.0,
            "lexical_similarity": 0.0, "confidence": 0.0,
        })
    return tuple(output)


def repetition_diagnostics(text: str, *, ngram_size: int = 4) -> dict[str, Any]:
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least two")
    tokens = caption_tokens(text.casefold())
    ngrams = [tokens[index:index + ngram_size] for index in range(max(0, len(tokens) - ngram_size + 1))]
    duplicate_count = len(ngrams) - len(set(ngrams))
    return {
        "token_count": len(tokens),
        "ngram_size": ngram_size,
        "ngram_count": len(ngrams),
        "duplicate_ngram_count": duplicate_count,
        "duplicate_ngram_fraction": duplicate_count / len(ngrams) if ngrams else 0.0,
        "has_duplicate_ngram": duplicate_count > 0,
    }


__all__ = [
    "ALPHA", "BETA", "LOCAL_VARIANTS", "TAU", "candidate_window_lengths",
    "caption_tokens", "caption_windows", "repetition_diagnostics",
    "score_local_candidates",
]
