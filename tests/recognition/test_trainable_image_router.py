import itertools

import numpy as np
import pytest

from recognition.trainable_image_router import branch_f1, fit_model, oracle, predict, regression_targets, score


def counts(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'detected_words': tp+fp, 'gt_words': tp+fn}


def test_fractional_oracle_matches_exhaustive_selection_at_every_budget():
    branches = {'a': {'light': counts(1, 0, 0), 'deepsolo': counts(0, 2, 1)},
        'b': {'light': counts(1, 1, 1), 'deepsolo': counts(2, 0, 0)},
        'c': {'light': counts(2, 0, 1), 'deepsolo': counts(2, 3, 1)},
        'd': {'light': counts(0, 0, 1), 'deepsolo': counts(1, 0, 0)}}
    ids = list(branches)
    for n in range(5):
        expected = max(branch_f1(branches, ids, subset) for subset in itertools.combinations(ids, n))
        actual = oracle(branches, ids, ids, n)
        assert branch_f1(branches, ids, actual['accepted_ids']) == expected
        assert actual['certificate_max_residual'] == '0'


def test_oracle_respects_inference_eligibility_and_population():
    branches = {'a': {'light': counts(0, 0, 1), 'deepsolo': counts(0, 2, 1)},
                'b': {'light': counts(1, 0, 0), 'deepsolo': counts(1, 0, 0)}}
    assert oracle(branches, list(branches), ['b'], 1)['accepted_ids'] == ['b']
    with pytest.raises(ValueError, match='cardinality'): oracle(branches, list(branches), ['b'], 2)


def test_targets_use_branch_count_differences_and_reject_extra_test_labels():
    branch = {'train': {'light': counts(3, 1, 2), 'deepsolo': counts(4, 3, 1)}}
    assert regression_targets(branch, ['train']).tolist() == [[-1., -3.]]
    with pytest.raises(ValueError, match='exactly'): regression_targets({**branch, 'test': branch['train']}, ['train'])


@pytest.mark.parametrize('kind', ['ridge', 'trees', 'oracle_logistic', 'oracle_trees'])
def test_portable_models_match_fit_and_do_not_update_on_inference(kind):
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.column_stack([np.sin(x[:, 0]), x[:, 1]/10])
    if kind.startswith('oracle'): y = (x[:, 0] > 20).astype(int)
    recipe = {'kind': kind, 'alpha': 10., 'C': 1., 'max_depth': 2, 'min_samples_leaf': 2}
    settings = {'seed': 12, 'trees': 8, 'logistic_max_iter': 1000, 'logistic_tolerance': 1e-7}
    model = fit_model(x, y, [str(i) for i in range(20)], recipe, settings)
    assert model['mean'] == x.mean(axis=0).tolist()
    before = model['mean'].copy(); result = predict(model, [[1e6, 1e6]])
    assert np.isfinite(result).all() and model['mean'] == before


def test_count_regression_ranking_matches_fractional_gain_definition():
    predictions = np.array([[1., 2.], [0., -3.], [-1., 0.]])
    assert score(predictions, 'ridge', .8).tolist() == pytest.approx([.4, 2.4, -2.])
    assert score(np.array([.2, .8]), 'oracle_logistic', .8).tolist() == [.2, .8]


def test_fold_training_cannot_receive_validation_or_test_ids():
    from scripts.run_trainable_image_router import train_one
    with pytest.raises(ValueError, match='outside'):
        train_one({}, {'fit_ids': ['train']}, {}, {}, ['test'], {}, '', .5)


def test_feature_vectors_exclude_outcomes_and_make_missing_context_explicit():
    from recognition.trainable_image_router import feature_dict
    settings = {'lower_quantile': .2, 'control_features': ['base_proxy_f1'],
                'head_features': [], 'context_features': ['caption']}
    head = {'coverage': .5, 'components': []}
    r = {'word_evidence': {'words': [{'native': .8, 'clip_rank': .5, 'crop_agreement': 1., 'visual': .7}],
        'predicted_word_count': 1, 'has_empty_reading': False, 'native_error_proxy': .2, 'visual_error_proxy': .3},
        'regions': {'heads': {'text': head, 'kernel': head}, 'missing_count': 1, 'missing_mass': 1., 'missing_consensus': 1.},
        'scores': {'base_proxy_f1': .7}}
    context = {'scores': {'caption': None}}
    baseline = feature_dict(r, context, [100, 50], settings)
    changed = feature_dict({**r, 'ground_truth': 'WRONG', 'deepsolo_tp': 10000},
        {**context, 'ground_truth': 'OTHER'}, [100, 50], settings)
    assert baseline == changed
    vector, schema, eligible = baseline
    assert eligible and vector['context_caption_missing'] == 1.
    assert 'context_caption' not in schema['core'] and 'context_caption' in schema['full']


def test_test_vectors_require_existing_frozen_choices_before_extraction(tmp_path):
    from scripts.run_trainable_image_router import features
    (tmp_path/'resolved_configuration.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='frozen_choices'):
        features({}, {}, tmp_path, 'test', {})
