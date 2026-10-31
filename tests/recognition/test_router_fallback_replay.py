import pytest

from recognition.router_fallback_replay import replace_fallback, replay, ordered_frontier


def counts(tp, predictions, gt):
    return dict(tp=tp, fp=predictions-tp, fn=gt-tp, detected_words=predictions,
                gt_words=gt, all_outputs=predictions, ignored_outputs=0)


def example():
    return {"a": dict(light=counts(1, 2, 2), deepsolo=counts(0, 1, 2)),
            "b": dict(light=counts(0, 0, 1), deepsolo=counts(1, 1, 1))}


def test_frozen_decision_changes_only_fallback():
    original = example()
    updated = replace_fallback(original, {"a": counts(2, 2, 2), "b": counts(1, 1, 1)})
    assert replay(updated, ["a"])["f1"] == replay(original, ["a"])["f1"]
    assert replay(updated, ["b"])["f1"] > replay(original, ["b"])["f1"]
    assert original["a"]["deepsolo"]["tp"] == 0


def test_complete_prefix_uses_frozen_rgb_ties_and_keeps_ineligible_fallback():
    b = example()
    b["c"] = dict(light=counts(1, 1, 1), deepsolo=counts(0, 0, 1))
    scores = dict(a=1., b=1., c=None)
    curve = ordered_frontier(b, scores, dict(a="z", b="a", c="0"))
    assert len(curve) == 3
    assert curve[1]["boundary_tie_key"] == "a"
    assert curve[1]["f1"] == replay(b, ["b"])["f1"]
    assert curve[-1]["deepsolo_images"] == 1


def test_population_and_unknown_decisions_fail():
    with pytest.raises(ValueError):
        replace_fallback(example(), {"a": counts(1, 1, 2)})
    with pytest.raises(ValueError):
        replace_fallback(example(), {"a": counts(1, 1, 1), "b": counts(1, 1, 1)})
    with pytest.raises(ValueError):
        replay(example(), ["unknown"])
