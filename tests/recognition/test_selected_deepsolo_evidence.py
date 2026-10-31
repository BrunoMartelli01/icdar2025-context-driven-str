import json

import pytest

from recognition.selected_deepsolo_evidence import inspect_fresh_export
from scripts.icdar_budget_data import sha


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def test_fresh_export_verifies_one_smoke_plus_remaining_selected_images(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    prior = tmp_path / 'prior.json'; plan = tmp_path / 'plan.json'; protocol = raw / 'protocol.json'
    r1 = raw / 'receipt_1.json'; r2 = raw / 'receipt_all.json'
    decision = dict(fallback_ids=['d:1', 'd:2'], accepted_ids=['d:3'],
                    ground_truth_used=False, fallback_outputs_loaded=False)
    save(prior, decision)
    save(plan, dict(decision_sha256=sha(prior), selected_before_inference=True,
        ground_truth_paths_included=False, population={'test': {'d': [
            dict(sample_id='d:1', image_path='/1', image_sha256='x'),
            dict(sample_id='d:2', image_path='/2', image_sha256='y')]}}))
    binding = dict(plan_sha256=sha(plan)); save(protocol, dict(binding=binding, gpu='test-gpu'))
    for sid, image_hash in (('d:1', 'x'), ('d:2', 'y')):
        save(raw / (sid.replace(':', '__') + '.json'), dict(status='SUCCESS', ground_truth_used=False,
            contract=dict(binding, sample_id=sid, image_sha256=image_hash), precision='float32', batch_size=1,
            gpu_peak_allocated_bytes=10, gpu_peak_reserved_bytes=20, seconds=0.1, input_chw=[3, 2, 2]))
    files = {p.name: sha(p) for p in raw.glob('d__*.json')}
    save(r1, dict(status='SUCCESS', binding=binding, new_model_inferences=1, reused=0,
        files_sha256={'d__1.json': files['d__1.json']}))
    save(r2, dict(status='SUCCESS', binding=binding, new_model_inferences=1, reused=1, files_sha256=files))
    paths = [plan, protocol, prior, r1, r2]
    config = dict(raw_test_root=str(raw), test_protocol=str(protocol),
        fresh_deepsolo=dict(plan=str(plan), prior_decisions=str(prior), dataset='d',
                            receipts=[str(r1), str(r2)]), pinned_inputs={str(p): sha(p) for p in paths})
    evidence = inspect_fresh_export(config, decision)
    assert evidence['actual_new_deepsolo_neural_inferences'] == 2
    assert evidence['gpu_peak_allocated_bytes'] == 10
    assert evidence['gpu_peak_reserved_bytes'] == 20
    (raw / 'd__2.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed fresh DeepSolo'):
        inspect_fresh_export(config, decision)
