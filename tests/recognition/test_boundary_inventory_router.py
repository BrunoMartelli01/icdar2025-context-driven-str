"""Quota and protected-decision guarantees for bounded inventory corrections."""
import pytest

from recognition.boundary_inventory_router import select


@pytest.mark.parametrize('fraction', [.2, .3, .5])
@pytest.mark.parametrize('width', [0., .025, .05, .1])
def test_only_borderline_decisions_can_change(fraction, width):
    ids = [f'{i:03d}' for i in range(100)]
    base = {s: float(100-i) for i, s in enumerate(ids)}
    reverse = {s: -v for s, v in base.items()}
    k, w = round(len(ids)*fraction), round(len(ids)*width)
    out = select(base, reverse, ids, fraction, {'half_width_fraction': width, 'audit_weight': 1.})
    assert len(out) == len(set(out)) == k
    assert set(ids[:max(0, k-w)]) <= set(out)
    assert not set(ids[k+w:])&set(out)
    assert len(set(out)-set(ids[:k])) <= w


def test_zero_weight_preserves_ties_and_original_rank():
    scores = {'b': .5, 'a': .5, 'c': .3, 'd': None}
    assert select(scores, {'a': 0., 'b': 1., 'c': 2., 'd': None}, list(scores), .5,
                  {'half_width_fraction': .1, 'audit_weight': 0.}) == ['a', 'b']


def test_missing_audit_score_fails_without_silent_fallback():
    with pytest.raises(ValueError):
        select({'a': 1., 'b': .5}, {'a': 1., 'b': None}, ['a', 'b'], .5,
               {'half_width_fraction': .1, 'audit_weight': .5})


def test_unfilled_quota_fails():
    with pytest.raises(ValueError):
        select({'a': None, 'b': None}, {'a': None, 'b': None}, ['a', 'b'], .5,
               {'half_width_fraction': .1, 'audit_weight': .5})
