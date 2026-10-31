import copy
from pathlib import Path

import pytest

from scripts import run_detailed_caption_comparison as runner


def config():
    return dict(mode='approximate', claim_label='approximate reproduction',
        caption_model_is_paper_blip2=False, caption_batch_size=2,
        router=dict(alpha=.6, beta=.4, tau=.8, primary_gate='paper_selected', lexical_mode='raw'),
        caption_profile=dict(caption_model='qwen'))


@pytest.mark.parametrize('key,value', [('mode', 'strict'), ('caption_model_is_paper_blip2', True),
    ('caption_batch_size', 4), ('router', dict(alpha=.5, beta=.5, tau=.8))])
def test_ablation_never_enters_strict_or_changes_frozen_scoring(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError): runner.validate_config(c)


def t1_fixture(tmp_path):
    baseline = tmp_path / 'baseline'; root = tmp_path / 'new'; sample = dict(sample_id='image:1')
    old = dict(config_sha256='old', source_sha256={}, runtime=dict(original=True), assets={}, samples={'image:1': sample})
    plan = dict(config_sha256='new', source_sha256={}, runtime=dict(original=False), assets={},
        samples={'image:1': sample}, config=dict(baseline_root=str(baseline)))
    payload = dict(result=dict(raw_text='Native!'), batch_size=1, precision='float32', peak_allocated_bytes=12)
    runner.seal(baseline / 'resolved_configuration.json', old)
    runner.seal(runner.cache_path(baseline, 't1', 'image:1'),
        dict(contract=runner.contract(old, 't1', sample), payload=payload, payload_sha256=runner.digest(payload)))
    return baseline, root, sample, old, plan, payload


def test_frozen_t1_reuse_preserves_original_payload_and_proves_origin(tmp_path):
    baseline, root, sample, old, plan, payload = t1_fixture(tmp_path)
    runner.reuse_t1(root, plan)
    assert runner.load_cache(root, plan, 't1', sample) == payload
    d = runner.read(runner.cache_path(root, 't1', 'image:1'))
    assert d['reuse_provenance']['source_sha256'] == runner.sha(runner.cache_path(baseline, 't1', 'image:1'))
    assert d['reuse_provenance']['neural_inference_rerun'] is False
    runner.reuse_t1(root, plan)


def test_changed_reused_t1_fails_even_with_valid_local_payload_digest(tmp_path):
    _, root, sample, _, plan, payload = t1_fixture(tmp_path)
    changed = copy.deepcopy(payload); changed['result']['raw_text'] = 'Different'
    runner.seal(runner.cache_path(root, 't1', 'image:1'),
        dict(contract=runner.contract(plan, 't1', sample), payload=changed, payload_sha256=runner.digest(changed)))
    with pytest.raises(ValueError, match='differs from its frozen source'): runner.reuse_t1(root, plan)


def test_corrupt_original_native_t1_is_not_reused(tmp_path):
    baseline, root, _, _, plan, _ = t1_fixture(tmp_path)
    p = runner.cache_path(baseline, 't1', 'image:1'); d = runner.read(p)
    d['payload']['result']['raw_text'] = 'Corrupted'
    import json
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='Invalid inference cache'): runner.reuse_t1(root, plan)
    assert not runner.cache_path(root, 't1', 'image:1').exists()


def test_partial_native_batch_fails_without_changing_context_or_outputs(tmp_path, monkeypatch):
    import caption.qwen_batch_scene as backend
    released = []
    class Model:
        def __init__(self, *args): pass
        def release(self): released.append(True)
        def generate(self, *args): raise AssertionError('Must not regenerate a partial batch')
    monkeypatch.setattr(backend, 'LocalQwenBatchScene', Model)
    monkeypatch.setattr(runner, 'load_cache', lambda root, plan, stage, sample: {} if sample['sample_id'] == 'a' else None)
    plan = dict(config=dict(caption_profile=dict(min_pixels=1, max_pixels=4, placement='single_device'),
        device='cuda:0', caption_batch_size=2), assets=dict(qwen=dict(snapshot='test-only')),
        samples={s: dict(sample_id=s) for s in ('a', 'b')})
    with pytest.raises(ValueError, match='Incomplete native batch'): runner.generate(tmp_path, plan)
    assert released == [True]
