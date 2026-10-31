"""Load one independently qualified, validation-selected approximate pipeline."""

from pathlib import Path

from recognition.fusion_deployment_factory import load_fusion_budget_pipeline
from scripts.icdar_budget_data import read, sha


def load_validated_budget_variant(run_root, case):
    root = Path(run_root)
    qualification = read(root / 'deployment_qualification.json')
    if (qualification['status'] != 'PASS' or
            sha(qualification['audit_path']) != qualification['audit_sha256']):
        raise ValueError('An independently qualified validation selector is required')
    for path, digest in qualification['artifact_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed qualified selector artifact: ' + path)
    frozen = read(root / 'selection_frozen.json')
    if (frozen['status'] != 'FROZEN' or frozen['mode'] != 'approximate' or
            frozen['test_used_for_selection'] or
            frozen['config_sha256'] != sha(root / 'resolved_configuration.json')):
        raise ValueError('A frozen validation-only approximate selector is required')
    if case not in frozen['selections']:
        raise ValueError('Unknown explicitly configured dataset/expert case')
    return load_fusion_budget_pipeline(frozen['selections'][case]['selected_root'])
