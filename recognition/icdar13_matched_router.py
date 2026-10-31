"""GT-blind OCR observations and exact image budgets for the native IC13 gate."""
from fractions import Fraction
import math

from recognition.group_word_selection import reading_confidence
from segmentation.icdar13_matched_training import emit_words


def router_rows(raw, cutoff):
    """Add the qualified native confidence without changing text, boxes or filtering."""
    return [dict(r, confidence=reading_confidence(r)["confidence_balanced"],
                 transcript_probability=r["sequence_confidence_native"]) for r in emit_words(raw, cutoff)]


def validation_fallback_count(total, test_total, requested):
    if min(total, test_total) < 1 or not 0 <= requested <= test_total:
        raise ValueError("Valid explicit TEST and heldout image populations required")
    return round(Fraction(total * requested, test_total))


def exact_accepted(scores, tie_keys, fallback_images):
    """Select an observable rank, splitting equal scores by the fixed RGB hash."""
    if set(scores) != set(tie_keys) or not 0 <= fallback_images <= len(scores):
        raise ValueError("Matched complete populations and a valid exact budget required")
    if any(v is not None and not math.isfinite(v) for v in scores.values()):
        raise ValueError("Finite scores required")
    order = sorted((s for s, v in scores.items() if v is not None), key=lambda s: (-scores[s], tie_keys[s], s))
    count = len(scores) - fallback_images
    if len(order) < count:
        raise ValueError("Exact fallback budget is infeasible under the eligibility guard")
    return order[:count]
