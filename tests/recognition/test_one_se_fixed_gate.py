import json

import pytest

from scripts.select_one_se_fixed_gate import choose


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def trial(score, models, heuristic):
    return dict(selection_score=score, models=models, heuristic=heuristic, groups=[], scalars=30 if models else 0,
        reference=.8 if models else None, guard=False,
        crossfit=dict(accepted_ids=['a', 'b', 'c']),
        result=dict(accepted_ids=['d'], combined=dict(f1=.91)))


def test_one_se_prefers_cheaper_trained_router_within_bootstrap_uncertainty(tmp_path):
    parent = tmp_path/'parent'; source = tmp_path/'source'; view = 'seen'
    write(parent/'resolved_configuration.json', dict(crossfit_config=dict(source_view=view), source_root=str(source)))
    write(parent/'data'/view/'case.json', dict(fit_ids=['a', 'b', 'c'], validation_ids=['d'],
        map_available=False, map_extra_models=0))
    details = [dict(target_name='expensive', extra_reader_crops=1), dict(target_name='cheap', extra_reader_crops=0)]
    write(parent/'targets.json', details)
    write(parent/'models'/view/'expensive'/'trials.json', [trial(.875, ['head.joblib'], None),
        trial(.99, [], 'weakest_sequence')])
    write(parent/'models'/view/'cheap'/'trials.json', [trial(.86, [], 'expected_errors')])
    branches = {}
    for sid, tp in zip(['a', 'b', 'c', 'd'], [10, 7, 8, 10]):
        counts = dict(tp=tp, fp=10-tp, fn=10-tp)
        branches[sid] = dict(light=counts, deepsolo=counts)
    write(source/'models'/view/'expensive'/'train_branches.json', branches)
    selected, info = choose(parent, dict(minimum_validation_f1=.9), dict(seed=42, bootstrap_draws=2000))
    assert info['bootstrap_standard_error'] > .015
    assert info['candidate_count'] == 2  # The non-trained sequence-only control is excluded.
    assert selected['target_name'] == 'cheap'
    assert selected['winner']['heuristic'] == 'expected_errors'


def test_one_se_requires_a_trained_candidate(tmp_path):
    write(tmp_path/'resolved_configuration.json', dict(crossfit_config=dict(source_view='seen'), source_root='unused'))
    write(tmp_path/'data/seen/case.json', dict(fit_ids=['a'], validation_ids=['b'], map_available=False, map_extra_models=0))
    write(tmp_path/'targets.json', [dict(target_name='only', extra_reader_crops=0)])
    write(tmp_path/'models/seen/only/trials.json', [trial(.9, [], 'weakest_sequence')])
    with pytest.raises(ValueError, match='trained candidate'):
        choose(tmp_path, dict(minimum_validation_f1=.9), dict(seed=42, bootstrap_draws=10))
