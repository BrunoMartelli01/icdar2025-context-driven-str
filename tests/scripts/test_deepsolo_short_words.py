import pytest

from scripts.evaluate_deepsolo_short_words import filter_population
from recognition.deepsolo_word_f1 import word_counts


def p(text, **kw):
    return dict(text=text, ignored=False, **kw)


def test_three_character_boundary_is_applied_after_symbol_removal_on_both_sides():
    preds = [p('AT!'), p('ABC'), p('123'), p('a-b'), p('ééé')]
    gt = [dict(text=t) for t in ('at', 'abc!', '123', 'ab', 'ééé')]
    kept_p, kept_g, report = filter_population(preds, gt, 3)
    assert [r['text'] for r in kept_p] == ['ABC', '123', 'ééé']
    assert [r['text'] for r in kept_g] == ['abc!', '123', 'ééé']
    assert report['excluded_gt_words'] == report['excluded_care_predictions'] == 2


def test_duplicates_ignore_and_invalid_predictions_remain_auditable():
    preds = [p('word'), p('word'), p('word', invalid_envelope={}), dict(p('at'), ignored=True)]
    kept_p, kept_g, report = filter_population(preds, [dict(text='word'), dict(text='at')], 3)
    metric = word_counts(kept_p, kept_g)
    assert (metric['tp'], metric['fp'], metric['fn']) == (1, 2, 0)
    assert report['excluded_all_predictions'] == 1 and report['excluded_care_predictions'] == 0


def test_no_remaining_gt_does_not_make_long_predictions_disappear():
    kept_p, kept_g, _ = filter_population([p('wrong')], [dict(text='at')], 3)
    assert not kept_g
    assert word_counts(kept_p, kept_g)['fp'] == 1


@pytest.mark.parametrize('minimum', [0, -1, True, 3.5])
def test_invalid_minimum_fails(minimum):
    with pytest.raises(ValueError):
        filter_population([], [], minimum)
