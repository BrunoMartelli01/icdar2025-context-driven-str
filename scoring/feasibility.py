"""Diagnostic for literal whole-string fuzzy matching at the paper threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .lexical import lexical_similarity


@dataclass(frozen=True)
class FeasibilityResult:
    recognized_text: str
    caption: str
    lexical_score: float
    required_lexical_at_perfect_semantic: float
    optimistic_confidence: float
    reaches_tau_even_if_semantic_is_one: bool


def diagnose(recognized_text: str, caption: str) -> FeasibilityResult:
    lexical = lexical_similarity(recognized_text, caption, "raw").score
    required = (0.8 - 0.6 * 1.0) / 0.4
    optimistic = 0.6 + 0.4 * lexical
    return FeasibilityResult(recognized_text, caption, lexical, required, optimistic, optimistic >= 0.8)


def representative_diagnostic() -> list[dict[str, object]]:
    caption = (
        "The image shows an outdoor street scene with a large red restaurant sign "
        "above a doorway, smaller notices on the windows, parked bicycles, pedestrians, "
        "and surrounding buildings in daylight."
    )
    return [asdict(diagnose(text, caption)) for text in ("CAFE", "RESTAURANT", "OPEN 24 HOURS")]


if __name__ == "__main__":
    import json
    print(json.dumps(representative_diagnostic(), indent=2, sort_keys=True))

