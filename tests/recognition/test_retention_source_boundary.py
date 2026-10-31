import json

import pytest

from scripts.evaluate_fixed_expert_retention import source_paths


def test_crossfit_gate_inherits_only_qualified_parent_observations(tmp_path):
    common = tmp_path/'common'; common.mkdir(); (common/'cases.json').write_text('{}')
    parent = tmp_path/'parent'; parent.mkdir()
    (parent/'resolved_configuration.json').write_text(json.dumps(dict(source_root=str(common))))
    obs = parent/'test_observations.json'; obs.write_text('{}')
    child = tmp_path/'child'; child.mkdir()
    resolved, observations = source_paths(child, dict(source_root=str(parent)))
    assert resolved == common and observations == obs
    middle = tmp_path/'middle'; middle.mkdir()
    (middle/'resolved_configuration.json').write_text(json.dumps(dict(source_root=str(parent))))
    resolved, observations = source_paths(child, dict(source_root=str(middle)))
    assert resolved == common and observations == obs


def test_crossfit_gate_rejects_cyclic_source_ancestry(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'resolved_configuration.json').write_text(json.dumps(dict(source_root=str(source))))
    with pytest.raises(ValueError, match='ancestry'):
        source_paths(tmp_path, dict(source_root=str(source)))
