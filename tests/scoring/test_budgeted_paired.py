import pytest

from scoring.budgeted_paired import budget_threshold, route
from scoring.strategies import decide


def test_budget_and_ties():
    assert budget_threshold([.1,.2,.3,.4], [True]*4, .3) == .2
    assert budget_threshold([.1,.1,.3,.4], [True]*4, .3) == .1


def test_empty_outputs_count_toward_budget():
    assert budget_threshold([.9,.2,.3,.4], [False,True,True,True], .3) == .2
    with pytest.raises(ValueError, match="mandatory"):
        budget_threshold([.1,.2], [False,True], .3)


def test_invalid_population():
    for scores, valid, budget in [([],[],.3), ([float('nan')],[True],.3),
                                ([.2],[],.3), ([.2],[True],1.)]:
        with pytest.raises(ValueError):
            budget_threshold(scores, valid, budget)


def test_paired_lexical_and_faithful_default_unchanged():
    args = dict(T1="shop",T2="the full caption",T3="other",S1=.6,S3=.5,L1=.1,L3=1.)
    result = route(tau=.4, **args)
    assert result.L == .1 and result.C == pytest.approx(.4)
    assert not result.needs_fallback
    assert decide("selected_candidate_paired", **args).tau == .8
    assert decide("selected_candidate_paired", **args).needs_fallback


def test_empty_guard_and_semantic_tie():
    result = route(tau=.1,T1="",T2="caption",T3="word",S1=.9,S3=.9,L1=1.,L3=.2)
    assert result.semantic_winner == "T1" and result.needs_fallback


def test_order_independent_label_free_calibration():
    scores = [.2,.7,.1,.5,.3,.8,.4,.9,.6,1.]
    tau = budget_threshold(scores, [True]*10, .3)
    assert tau == .4
    assert tau == budget_threshold(scores[::-1], [True]*10, .3)
