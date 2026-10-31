import json
import hashlib
import sys
from pathlib import Path

import pytest

from scripts.run_deepsolo_word_f1_study import save, validate_config
from scripts.analyze_deepsolo_word_f1 import load_report
from scripts import run_deepsolo_word_f1_study as study
from types import SimpleNamespace


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/approximate/deepsolo_word_f1_resolution_v1.json').read_text())


@pytest.mark.parametrize('change', [
    {'reproduction_mode': 'strict'}, {'table_5_claim': True},
    {'inference_floor': .9}, {'precision': 'float16'}, {'batch_size': 2},
    {'recognition_thresholds': [float('nan')]}, {'resolutions': {'bad': [0, 1024]}},
])
def test_experiment_rejects_changed_scientific_boundaries(change):
    with pytest.raises(ValueError):
        validate_config(dict(config(), **change))


def test_evidence_is_exclusive_and_idempotent(tmp_path):
    path = tmp_path/'shard.json'
    save(path, {'status': 'SUCCESS'})
    save(path, {'status': 'SUCCESS'})
    with pytest.raises(ValueError, match='Refuse to overwrite'):
        save(path, {'status': 'FAILED'})


@pytest.mark.parametrize('status,limit', [('FAILED', None), ('SUCCESS', 1)])
def test_failed_or_partial_run_cannot_enter_full_dataset_comparison(tmp_path, status, limit):
    (tmp_path/'results.json').write_text(json.dumps({'status': status}))
    (tmp_path/'resolved_configuration.json').write_text(json.dumps(
        {'reproduction_mode': 'approximate', 'limit': limit}))
    with pytest.raises(ValueError, match='complete successful'):
        load_report(tmp_path, 10)


@pytest.mark.parametrize('message,oom', [('CUDA out of memory', True), ('incompatible checkpoint', False)])
def test_model_failure_aborts_and_preserves_structured_receipt(tmp_path, monkeypatch, message, oom):
    plan = tmp_path/'source.json'; plan.write_text('{}')
    cfg = config(); cfg['source_plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest()
    cfg_path = tmp_path/'approximate'/'config.json'; cfg_path.parent.mkdir()
    cfg_path.write_text(json.dumps(cfg))
    output = tmp_path/'output'
    argv = ['study', '--mode', 'approximate', '--stage', 'export', '--config', str(cfg_path),
            '--source-plan', str(plan), '--output-root', str(output),
            '--runtime', str(tmp_path/'runtime'), '--source-code-root', str(tmp_path/'upstream'),
            '--data-root', str(tmp_path/'data')]
    monkeypatch.setattr(sys, 'argv', argv)

    def fail(*args):
        raise RuntimeError(message)

    monkeypatch.setattr(study, 'export', fail)
    with pytest.raises(RuntimeError, match=message):
        study.main()
    receipts = list(output.glob('failure_*.json'))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt['status'] == 'FAILED' and receipt['oom'] is oom
    assert message in receipt['traceback']
    assert not (output/'results.json').exists()


@pytest.mark.parametrize('sizes', [(1024, 1892), (1536, 2838)])
def test_resolution_change_updates_inverse_coordinates_as_well_as_input(sizes):
    # ViTAE scales its normalized point to input pixels and then uses cached
    # construction-time resize to restore original pixels. A stale native value
    # gives x=900 instead of x=600 for the large input in this regression.
    model = SimpleNamespace(min_size_test=1024, max_size_test=1892)
    cfg = SimpleNamespace(INPUT=SimpleNamespace(MIN_SIZE_TEST=sizes[0], MAX_SIZE_TEST=sizes[1]))
    assert study.configure_model_resize(model, cfg) == list(sizes)
    original_width = 800
    point_in_resized_image = .75 * sizes[0]
    restored = point_in_resized_image * original_width / model.min_size_test
    assert restored == 600.
    # Returning to native after a larger view must also restore native mapping.
    study.configure_model_resize(model, SimpleNamespace(INPUT=SimpleNamespace(MIN_SIZE_TEST=1024, MAX_SIZE_TEST=1892)))
    assert model.min_size_test == 1024 and model.max_size_test == 1892
