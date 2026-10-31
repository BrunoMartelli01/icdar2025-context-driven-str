import copy
from types import SimpleNamespace

import pytest
import torch

from segmentation.icdar2015_totaltext_weights import (
    fresh_optimizer, raw_word_selection_key, verify_disjoint_manifests, verify_source_metadata,
)


def source_metadata():
    spec = dict(architecture='source', epoch=263, checkpoint_sha256='checkpoint', manifest_sha256='manifest')
    contract = dict(architecture='source', manifest_sha256='manifest')
    payload = dict(architecture='source', epoch=263, run_contract=contract)
    report = dict(status='SUCCESS', training_complete=True, checkpoint=dict(sha256='checkpoint', epoch=263), run_contract=contract)
    return payload, report, spec


@pytest.mark.parametrize('corruption', ['incomplete', 'architecture', 'manifest', 'epoch', 'contract'])
def test_source_provenance_fails_closed(corruption):
    payload, report, spec = source_metadata()
    verify_source_metadata(payload, report, spec)
    payload = copy.deepcopy(payload); report = copy.deepcopy(report)
    if corruption == 'incomplete': report['training_complete'] = False
    if corruption == 'architecture': payload['architecture'] = 'other'
    if corruption == 'manifest': report['run_contract']['manifest_sha256'] = 'other'
    if corruption == 'epoch': payload['epoch'] = 264
    if corruption == 'contract': payload['run_contract']['extra'] = 'changed'
    with pytest.raises(ValueError, match='provenance'): verify_source_metadata(payload, report, spec)


def test_fresh_optimizer_groups_and_initial_rates():
    model = torch.nn.Module(); model.encoder = torch.nn.Linear(2, 2); model.decoder = torch.nn.Linear(2, 1)
    model.encoder.bias.requires_grad_(False)
    cfg = dict(training=dict(learning_rate=1e-4, weight_decay=.01, adam_betas=[.9,.999], adam_epsilon=1e-8))
    optimizer = fresh_optimizer(model, cfg, dict(encoder_learning_rate=1e-5))
    assert not optimizer.state
    assert [g['initial_lr'] for g in optimizer.param_groups] == [1e-4,1e-5]
    assert [g['group_name'] for g in optimizer.param_groups] == ['decoder','encoder_layer3_layer4']
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters() if p.requires_grad}


def test_raw_quality_precedes_rejection_and_earliest_ties():
    metric = lambda f: dict(alphanumeric_casefold=dict(f1=f))
    geo = dict(proposal_f1_iou_50=.8)
    assert raw_word_selection_key(metric(.5), metric(.55), geo, 10) > raw_word_selection_key(metric(.49), metric(.7), geo, 1)
    assert raw_word_selection_key(metric(.5), metric(.55), geo, 0) > raw_word_selection_key(metric(.5), metric(.55), geo, 1)


def test_exposure_overlap_rejected(monkeypatch, tmp_path):
    import segmentation.icdar2015_totaltext_weights as module
    sample = lambda h: SimpleNamespace(image_sha256=h)
    monkeypatch.setattr(module, 'file_sha256', lambda p:'pinned')
    monkeypatch.setattr(module, 'load_segmentation_manifest', lambda *a,**kw: SimpleNamespace(samples=[sample('a'),sample('b')]))
    config = dict(datasets=dict(total_text=dict(manifest_sha256='pinned',images=2)))
    assert verify_disjoint_manifests(config,SimpleNamespace(samples=[sample('c')]),tmp_path/'m')['exact_image_hash_overlap'] == 0
    with pytest.raises(ValueError, match='overlap'):
        verify_disjoint_manifests(config,SimpleNamespace(samples=[sample('b')]),tmp_path/'m')
