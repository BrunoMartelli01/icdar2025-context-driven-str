from collections import defaultdict
import itertools

import pytest

from recognition.deepsolo_word_f1 import (
    aggregate, apply_policy, beam_decode, error_inventory, fuse_views, normalize_word,
    recognition_confidence, word_counts,
)


def pred(text, **kw):
    return dict(text=text, ignored=False, bbox=[0, 0, 10, 10], instance_id=0,
                score=.7, recognition_confidence=.8, beam_text='other', **kw)


def test_symbol_normalization_duplicates_and_empty_population():
    p = [pred('WORD!'), pred('word'), pred('word'), pred('#')]
    g = [dict(text='word'), dict(text='WORD'), dict(text='!')]
    r = word_counts(p, g)
    assert (r['tp'], r['fp'], r['fn']) == (2, 2, 1)
    assert (r['symbol_only_gt'], r['symbol_only_predictions']) == (1, 1)
    assert aggregate([r])['f1'] == 4/7
    assert normalize_word("L'été-2026") == 'lété2026'
    assert aggregate([word_counts([], [])])['f1'] is None
    assert aggregate([word_counts([], [dict(text='word')])])['f1'] == 0
    assert word_counts([pred('')], [dict(text='word')])['fp'] == 1


def test_invalid_envelope_ignored_and_no_word_splitting():
    p = [pred('word', invalid_envelope={}), dict(pred('word'), ignored=True), pred('two words')]
    r = word_counts(p, [dict(text='word'), dict(text='two'), dict(text='words')])
    assert (r['tp'], r['fp'], r['fn'], r['ignored_outputs']) == (0, 2, 3, 1)


def test_ctc_matches_exhaustive_path_marginalization():
    rows = []
    for a, b, blank1, blank2 in ((.4, .1, .3, .2), (.4, .2, .2, .2), (.2, .3, .1, .4)):
        rows.append([a, b]+[0.]*34+[blank1, blank2])
    mass = defaultdict(float)
    for path in itertools.product((0, 1, 36, 37), repeat=3):
        text, previous, probability = '', None, 1.
        for step, token in enumerate(path):
            probability *= rows[step][token]
            if token < 36 and token != previous:
                text += 'ab'[token]
            previous = token if token < 36 else None
        mass[text] += probability
    assert beam_decode(rows, 100) == max(mass, key=mass.get)


def test_ctc_repeated_letters_require_blank():
    row = lambda i: [float(j == i) for j in range(38)]
    assert beam_decode([row(0), row(0), row(37), row(0)], 10) == 'aa'
    assert beam_decode([row(0), row(36), row(0)], 10) == 'aa'
    assert recognition_confidence([row(0), row(37), row(0)]) == 1.


@pytest.mark.parametrize('rows', [[], [[.1]*38], [[float('nan')]+[0.]*37], [[1.]+[0.]*36]])
def test_ctc_rejects_invalid_probabilities(rows):
    with pytest.raises(ValueError):
        beam_decode(rows, 10)


def test_threshold_boundaries_and_spatial_deduplication():
    policy = dict(decoder='greedy', score_threshold=.7, recognition_threshold=.8, same_text_nms_iou=.5)
    p = [pred('word'), dict(pred('word'), instance_id=1),
         dict(pred('word'), instance_id=2, bbox=[20, 0, 30, 10])]
    assert len(apply_policy(p, policy)) == 2
    assert len(apply_policy(p, dict(policy, same_text_nms_iou=None))) == 3
    assert not apply_policy(p, dict(policy, score_threshold=.700001))
    assert not apply_policy(p, dict(policy, recognition_threshold=.800001))


def test_error_inventory_separates_localization_and_recognition():
    p = [pred('wrong')]
    g = [dict(text='right', bbox=[0, 0, 10, 10]), dict(text='missing', bbox=[20, 0, 30, 10])]
    r = error_inventory(p, g, .5)
    assert (r['localized_words'], r['unlocalized_words'], r['localized_wrong_transcription']) == (1, 1, 1)


def test_fusion_preserves_spatial_repeats_and_selects_confidence_without_gt():
    policy = dict(decoder='greedy', score_threshold=.1, recognition_threshold=0.,
                  same_text_nms_iou=None, fusion_iou=.5, recognition_rank_power=1)
    native = [pred('word'), dict(pred('word'), bbox=[20, 0, 30, 10], instance_id=1)]
    large = [dict(pred('different'), recognition_confidence=.99, ignored=True)]
    result = fuse_views(native, large, policy)
    assert [p['text'] for p in result] == ['different', 'word']
    # Evaluation-only ignore flags do not alter inference decisions.
    assert [p['text'] for p in fuse_views(native, [dict(large[0], ignored=False)], policy)] == ['different', 'word']
    assert len(fuse_views(native+[dict(native[0], instance_id=2)], [], policy)) == 3
