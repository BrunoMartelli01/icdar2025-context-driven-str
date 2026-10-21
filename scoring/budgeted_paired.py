"""Opt-in approximate paired routing; faithful scoring defaults stay unchanged."""
from dataclasses import replace
import math

from scoring.strategies import decide


def budget_threshold(scores, nonempty, max_fallback_fraction):
    """Largest observed cutoff respecting an image-call budget, without labels.

    Equal scores are never split. Empty predictions always require fallback.
    The TRAIN budget is not a guarantee for a different score distribution.
    """
    scores, nonempty = list(scores), list(nonempty)
    if not scores or len(scores) != len(nonempty):
        raise ValueError("nonempty aligned population required")
    if not 0 <= max_fallback_fraction < 1:
        raise ValueError("budget must be in [0,1)")
    if any(not math.isfinite(c) or not -.6 <= c <= 1 for c in scores):
        raise ValueError("invalid confidence")
    budget = math.floor(len(scores) * max_fallback_fraction)
    if sum(not ok for ok in nonempty) > budget:
        raise ValueError("mandatory empty-output fallbacks exceed budget")
    candidates = sorted(set(c for c, ok in zip(scores, nonempty) if ok))
    feasible = [t for t in candidates
                if sum(not ok or c < t for c, ok in zip(scores, nonempty)) <= budget]
    return max(feasible)


def route(*, tau, **scores_and_texts):
    if not math.isfinite(tau) or not -.6 <= tau <= 1:
        raise ValueError("invalid frozen threshold")
    result = decide("selected_candidate_paired", **scores_and_texts)
    return replace(result, tau=tau,
                   needs_fallback=not result.selected_output.strip() or result.C < tau)
