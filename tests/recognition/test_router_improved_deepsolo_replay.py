"""Metric and frozen-routing boundaries for improved-fallback comparisons."""
import itertools

import numpy as np
import pytest

from scripts.replay_router_improved_deepsolo import (
    conditional_interval, frozen_predictions, independent_counts, independent_route, metric_counts,
)
from recognition.trainable_image_router import branch_f1, oracle


def pred(text, **kwargs):
    return dict(text=text, ignored=False, **kwargs)


@pytest.mark.parametrize('metric,tp', [('casefold_exact', 0), ('casefold_alnum', 1)])
def test_only_declared_metric_removes_punctuation(metric, tp):
    predictions, gt = [pred('CAFÉ!')], [dict(text='café')]
    value = metric_counts(predictions, gt, metric)
    assert value['tp'] == tp and value['gt_words'] == value['detected_words'] == 1
    assert value == independent_counts(predictions, gt, metric)


@pytest.mark.parametrize('metric', ['casefold_exact', 'casefold_alnum'])
def test_duplicates_ignored_outputs_and_invalid_envelopes_remain_explicit(metric):
    predictions = [pred('A'), pred('a'), pred('A', invalid_envelope={}), dict(text='B', ignored=True)]
    gt = [dict(text='a')]
    value = metric_counts(predictions, gt, metric)
    assert value['tp'] == 1 and value['fp'] == 2 and value['ignored_outputs'] == 1
    assert value == independent_counts(predictions, gt, metric)


def test_symbol_only_and_empty_predictions_do_not_disappear_from_denominators():
    predictions = [pred('!'), pred('')]; gt = [dict(text='!'), dict(text='')]
    exact = metric_counts(predictions, gt, 'casefold_exact')
    alnum = metric_counts(predictions, gt, 'casefold_alnum')
    assert exact['tp'] == 1 and alnum['tp'] == 0
    assert alnum['detected_words'] == alnum['gt_words'] == 2
    assert alnum == independent_counts(predictions, gt, 'casefold_alnum')


def test_unicode_accents_are_preserved_and_words_are_not_split():
    value = metric_counts([pred('CAFÉ!'), pred('two words')], [dict(text='cafe'), dict(text='two'), dict(text='words')], 'casefold_alnum')
    assert value['tp'] == 0 and value['detected_words'] == 2 and value['gt_words'] == 3


def test_unknown_metric_fails():
    with pytest.raises(ValueError):
        metric_counts([], [], 'unspecified')


def test_profile_thresholds_do_not_consult_ignore_flags():
    rows = [dict(pred('a'), score=.4, recognition_confidence=.9, instance_id=0),
            dict(pred('b'), score=.4, recognition_confidence=.899, instance_id=1)]
    policy = dict(resolution='native', decoder='greedy', score_threshold=.4, recognition_threshold=.9, same_text_nms_iou=None)
    assert [p['text'] for p in frozen_predictions({'native': rows}, policy)] == ['a']
    rows[0]['ignored'] = True
    assert [p['text'] for p in frozen_predictions({'native': rows}, policy)] == ['a']


def test_frozen_routing_keeps_unaccepted_images_and_empty_light_predictions():
    empty = metric_counts([], [dict(text='a')], 'casefold_exact')
    full = metric_counts([pred('a')], [dict(text='a')], 'casefold_exact')
    branches = {'a': dict(light=empty, deepsolo=full), 'b': dict(light=full, deepsolo=empty)}
    value = independent_route(branches, ['a', 'b'], ['a'])
    assert value['images'] == 2 and value['tp'] == 0 and value['fn'] == 2
    assert value['accuracy_tn0'] == 0
    with pytest.raises(ValueError):
        independent_route(branches, ['a', 'b'], ['a', 'a'])


def test_oracle_recomputed_for_replacement_fallback_matches_exhaustive_search():
    gt = [dict(text='a')]; full = metric_counts([pred('a')], gt, 'casefold_exact'); empty = metric_counts([], gt, 'casefold_exact')
    branches = {'a': dict(light=empty, deepsolo=full), 'b': dict(light=full, deepsolo=empty), 'c': dict(light=full, deepsolo=full)}
    ids = sorted(branches); result = oracle(branches, ids, ids, 1)
    assert result['accepted_ids'] == ['b']
    assert result['f1'] == max(float(branch_f1(branches, ids, ss)) for ss in itertools.combinations(ids, 1))


def test_zero_bypass_has_zero_paired_change():
    full = metric_counts([pred('a')], [dict(text='a')], 'casefold_exact')
    value = conditional_interval({'a': dict(light=full, deepsolo=full)}, ['a'], [], np.ones((20, 1)))
    assert value['lower95_pp'] == value['upper95_pp'] == 0
