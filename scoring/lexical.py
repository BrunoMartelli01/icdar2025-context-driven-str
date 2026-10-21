"""Explicit lexical preprocessing modes around the repository's literal ratio path."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from rapidfuzz.fuzz import ratio


LexicalMode = Literal["raw", "lowercase", "normalized_whitespace", "unicode_normalized"]


@dataclass(frozen=True)
class LexicalResult:
    score: float
    mode: LexicalMode
    left_processed: str
    right_processed: str
    function: str = "rapidfuzz.fuzz.ratio/100"


def preprocess(text: str | None, mode: LexicalMode) -> str:
    value = text or ""
    if mode == "raw":
        return value
    if mode == "lowercase":
        return value.lower()
    if mode == "normalized_whitespace":
        return re.sub(r"\s+", " ", value).strip()
    if mode == "unicode_normalized":
        return unicodedata.normalize("NFC", value)
    raise ValueError(f"Unknown lexical mode: {mode}")


def lexical_similarity(left: str | None, right: str | None, mode: LexicalMode = "raw") -> LexicalResult:
    a, b = preprocess(left, mode), preprocess(right, mode)
    return LexicalResult(float(ratio(a, b)) / 100.0, mode, a, b)

