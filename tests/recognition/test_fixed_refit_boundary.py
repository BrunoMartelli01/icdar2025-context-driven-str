import json

import pytest

from scripts.evaluate_fixed_expert_budget import expected_frontend_checkpoint
from scripts.icdar_budget_data import sha


def test_detector_refit_must_bind_to_the_exact_frozen_gate_and_checkpoint(tmp_path):
    gate = tmp_path/'gate'; gate.mkdir()
    for name in ('selection_frozen.json', 'deployment_qualification.json', 'emission_frozen.json'):
        (gate/name).write_text(name)
    expert = tmp_path/'expert.json'; expert.write_text(json.dumps(dict(winner=dict(checkpoint_sha256='old'))))
    checkpoint = tmp_path/'checkpoint.pt'; checkpoint.write_bytes(b'new weights')
    contract = dict(status='PASS', mode='approximate', test_used_for_selection=False,
        gate_root=str(gate), gate_selection_sha256=sha(gate/'selection_frozen.json'),
        gate_qualification_sha256=sha(gate/'deployment_qualification.json'),
        emission_policy_sha256=sha(gate/'emission_frozen.json'), checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha(checkpoint))
    path = tmp_path/'refit.json'; path.write_text(json.dumps(contract))
    plan = dict(config=dict(expert_frozen=str(expert)))
    assert expected_frontend_checkpoint({}, plan, gate) == 'old'
    config = dict(frontend_refit_contract=str(path), pinned_inputs={str(path): sha(path), str(checkpoint): sha(checkpoint)})
    assert expected_frontend_checkpoint(config, plan, gate) == sha(checkpoint)
    (gate/'selection_frozen.json').write_text('changed')
    with pytest.raises(ValueError, match='not qualified'):
        expected_frontend_checkpoint(config, plan, gate)


def test_detector_refit_without_an_explicit_pin_fails(tmp_path):
    expert = tmp_path/'expert.json'; expert.write_text(json.dumps(dict(winner=dict(checkpoint_sha256='old'))))
    with pytest.raises(ValueError, match='explicit pinned input'):
        expected_frontend_checkpoint(dict(frontend_refit_contract='untracked', pinned_inputs={}),
            dict(config=dict(expert_frozen=str(expert))), tmp_path)
