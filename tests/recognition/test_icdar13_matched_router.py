import math
import pytest

from recognition.icdar13_matched_router import router_rows, validation_fallback_count, exact_accepted
from recognition.unet_token_routing import selected_rows


def test_confidence_adapter_matches_native_interface_without_mutating_ocr():
    raw = [dict(raw_text="AB", token_probabilities=[.98, .97], token_log_probabilities=[math.log(.98), math.log(.97)], sequence_confidence_native=.91),
           dict(raw_text="&", token_probabilities=[.99], token_log_probabilities=[math.log(.99)], sequence_confidence_native=.99)]
    expected = selected_rows(raw, dict(minimum_transcript_probability=.9))[0]
    actual = router_rows(raw, .9)
    assert len(actual) == 1 and actual[0] == expected
    assert "confidence" not in raw[0]


def test_user_budget_is_exact_and_validation_uses_rounded_same_fraction():
    scores = {str(i):float(i) for i in range(233)}
    assert len(exact_accepted(scores, scores, 61)) == 172
    assert validation_fallback_count(23, 233, 61) == 6
    assert validation_fallback_count(4, 233, 61) == 1


def test_ties_are_rgb_ordered_and_ineligible_images_cannot_be_silently_accepted():
    scores = dict(a=1., b=1., c=None)
    assert exact_accepted(scores, dict(a="z", b="a", c="b"), 2) == ["b"]
    with pytest.raises(ValueError): exact_accepted(scores, dict(a="z", b="a", c="b"), 0)
