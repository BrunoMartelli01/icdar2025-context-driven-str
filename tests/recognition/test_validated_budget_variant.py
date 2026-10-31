import pytest

from scripts.select_validated_budget_variant import choose, choose_no_extra_neural


def record(label, *, f1, tp, fp, fn, maps=0, rereads=0, ids=('a', 'b')):
    return dict(label=label, ids=list(ids), ground_truth_words=[tp + fn] * len(ids),
        values=[dict(tp=tp, fp=fp, fn=fn) for _ in ids], f1=f1,
        deepsolo_images=1, additional_map_models=maps, extra_reader_crops=rereads)


def test_selector_prefers_lower_neural_cost_at_equal_validation_quality():
    expensive = record('map', f1=0.8, tp=4, fp=1, fn=1, maps=1)
    reread = record('reread', f1=0.8, tp=4, fp=1, fn=1, rereads=1)
    light = record('light', f1=0.8, tp=4, fp=1, fn=1)
    winner, uncertainty = choose([expensive, reread, light], 100, 7)
    assert winner['label'] == 'light'
    assert uncertainty['eligible_count'] == 3


def test_selector_does_not_trade_away_clear_validation_quality():
    good = record('good', f1=0.8, tp=4, fp=1, fn=1, maps=1)
    poor = record('poor', f1=0.0, tp=0, fp=5, fn=5)
    winner, _ = choose([good, poor], 100, 7)
    assert winner['label'] == 'good'


def test_selector_rejects_mismatched_validation_or_budget():
    a = record('a', f1=0.8, tp=4, fp=1, fn=1)
    b = record('b', f1=0.8, tp=4, fp=1, fn=1, ids=('a', 'c'))
    with pytest.raises(ValueError, match='populations and budgets'):
        choose([a, b], 100, 7)
    b = dict(a, label='b', deepsolo_images=0)
    with pytest.raises(ValueError, match='populations and budgets'):
        choose([a, b], 100, 7)


def test_no_extra_neural_selects_best_eligible_and_rejects_empty_pool():
    expensive = record('map', f1=0.9, tp=4, fp=1, fn=1, maps=1)
    eligible = record('light', f1=0.8, tp=4, fp=1, fn=1)
    lesser = record('lesser', f1=0.7, tp=4, fp=1, fn=1)
    winner, meta = choose_no_extra_neural([expensive, eligible, lesser])
    assert winner['label'] == 'light'
    assert meta['eligible_count'] == 2
    with pytest.raises(ValueError, match='no-extra-neural'):
        choose_no_extra_neural([expensive])
