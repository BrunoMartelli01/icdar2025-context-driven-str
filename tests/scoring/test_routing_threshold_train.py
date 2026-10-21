import math

import pytest

from diagnostics.routing_threshold_train import separability, token_labels


def row(c, correct, *, good=None, bad=None, eligible=True):
    return {"C": c, "correct": correct, "eligible": eligible,
            "good_words": int(correct) if good is None else good,
            "bad_words": int(not correct) if bad is None else bad}


def test_threshold_excludes_incorrect_ties_and_keeps_higher_correct():
    result = separability([row(.2, True), row(.4, False), row(.4, True), row(.6, True)])
    safe = result["zero_observed_errors_train"]
    assert safe["threshold"] == math.nextafter(.4, math.inf)
    assert safe["accepted_units"] == safe["correct_units"] == 1
    assert safe["incorrect_units"] == 0
    assert safe["correct_units_lost"] == 2
    assert not result["all_correct_separable_from_all_incorrect"]


def test_no_useful_threshold_does_not_claim_perfect_precision():
    result = separability([row(.3, True), row(.8, False)])
    assert not result["useful_zero_error_threshold_exists"]
    assert result["zero_observed_errors_train"]["precision_units"] is None


def test_perfect_separation_and_empty_guard():
    result = separability([row(.2, False), row(.5, True), row(.7, True), row(1., False, eligible=False)])
    assert result["all_correct_separable_from_all_incorrect"]
    assert result["zero_observed_errors_train"]["correct_units"] == 2
    assert result["eligible_units"] == 3


def test_mixed_phrase_must_not_be_accepted_as_all_correct():
    result = separability([row(.7, False, good=3, bad=1), row(.8, True, good=2, bad=0)])
    assert result["correct_word_C"]["count"] == 5
    assert result["incorrect_word_C"]["count"] == 1
    assert result["zero_observed_errors_train"]["correct_words"] == 2


def test_token_multiplicity_casefold_and_punctuation():
    result = token_labels("SHOP shop shop, ROAD", [{"text": "shop"}, {"text": "road"}])
    assert [x["correct"] for x in result] == [True, False, False, True]


def test_no_candidates_and_nonfinite_fail_closed():
    assert not separability([])["useful_zero_error_threshold_exists"]
    with pytest.raises(ValueError):
        separability([row(float("nan"), True)])
