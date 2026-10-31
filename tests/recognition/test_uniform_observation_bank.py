from pathlib import Path

import pytest

from recognition.uniform_observation_bank import observation_sources
from scripts.run_uniform_unet_training import write, read
from segmentation.uniform_unet_fusion import file_hash


def fixture(tmp_path):
    source, bank = tmp_path/'deployment', tmp_path/'bank'
    plan = dict(partitions=dict(fit=['a', 'b'], selection=['v']))
    write(source/'resolved_configuration.json', plan)
    original = {s: dict(path='full-fit-'+s, sha256='original') for s in ['a', 'b', 'v']}
    write(source/'export_train/index.json', original)
    index, folds = {}, []
    for i, sid in enumerate(['a', 'b']):
        folder = tmp_path/('fold_'+str(i))
        sample = folder/'sample.json'
        write(sample, dict(ground_truth_used=False, decoder_fit_excluded=True, maps_path='explicit-map.npz'))
        index[sid] = dict(path=str(sample), sha256=file_hash(sample))
        audit = folder/'heldout/independent_audit.json'
        write(audit, dict(status='PASS'))
        folds.append(dict(root=str(folder), audit_sha256=file_hash(audit)))
    write(bank/'index.json', index)
    write(bank/'independent_audit.json', dict(status='PASS', decoder_fit_excluded=True,
        index_sha256=file_hash(bank/'index.json'), source_training_plan_sha256=file_hash(source/'resolved_configuration.json'),
        images=2, folds=folds, upstream_encoder_exposure=True))
    config = dict(fit_export_overrides=dict(icdar13=str(bank)), decoder_folds=2,
        pinned_inputs={str(p): file_hash(p) for p in [bank/'index.json', bank/'independent_audit.json']})
    return source, bank, config, plan, original


def test_decoder_bank_replaces_only_fit_preserving_deployment_validation(tmp_path):
    source, bank, config, plan, original = fixture(tmp_path)
    index, metadata = observation_sources(config, 'icdar13', source, plan)
    assert index['v'] == original['v'] and index['a'] != original['a']
    assert metadata['decoder_fit_excluded'] and not metadata['upstream_expert_image_excluded']
    index, metadata = observation_sources({}, 'icdar13', source, plan)
    assert index == original and not metadata['decoder_fit_excluded']


def test_incomplete_bank_cannot_fall_back_to_fit_seen_predictions(tmp_path):
    source, bank, config, plan, _ = fixture(tmp_path)
    write(bank/'index.json', {})
    config['pinned_inputs'][str(bank/'index.json')] = file_hash(bank/'index.json')
    with pytest.raises(ValueError, match='cover the original FIT exactly'):
        observation_sources(config, 'icdar13', source, plan)


def test_changed_bank_or_fold_qualification_is_rejected(tmp_path):
    source, bank, config, plan, _ = fixture(tmp_path)
    write(bank/'index.json', {})
    with pytest.raises(ValueError, match='Pinned'):
        observation_sources(config, 'icdar13', source, plan)
    source, bank, config, plan, _ = fixture(tmp_path/'second')
    write(tmp_path/'second/fold_0/heldout/independent_audit.json', dict(status='FAILED'))
    with pytest.raises(ValueError, match='neural-fold qualification'):
        observation_sources(config, 'icdar13', source, plan)


def test_two_view_frontend_requires_matching_fit_and_validation_sources(tmp_path):
    source, bank, config, plan, _ = fixture(tmp_path)
    policy=tmp_path/'frontend.json';write(policy,dict(kind='two_view_fixture'))
    digest=file_hash(policy);config['observation_frontend_policy']=str(policy)
    config['pinned_inputs'][str(policy)]=digest
    with pytest.raises(ValueError,match='both required'):
        observation_sources(config,'icdar13',source,plan)
    vb=tmp_path/'validation';sample=vb/'sample.json'
    write(sample,dict(ground_truth_used=False,maps_path='native-map.npz',frontend_policy_sha256=digest))
    write(source/'complete.json',dict(checkpoint_sha256='qualified-full-fit-model'))
    write(vb/'index.json',{'v':dict(path=str(sample),sha256=file_hash(sample))})
    write(vb/'independent_audit.json',dict(status='PASS',images=1,index_sha256=file_hash(vb/'index.json'),
        source_training_plan_sha256=file_hash(source/'resolved_configuration.json'),
        source_checkpoint_sha256='qualified-full-fit-model',frontend_policy_sha256=digest))
    config['validation_export_overrides']={'icdar13':str(vb)}
    for p in (vb/'index.json',vb/'independent_audit.json'):config['pinned_inputs'][str(p)]=file_hash(p)
    with pytest.raises(ValueError,match='another observation frontend'):
        observation_sources(config,'icdar13',source,plan)
    audit=read(bank/'independent_audit.json');audit['frontend_policy_sha256']=digest;write(bank/'independent_audit.json',audit)
    index=read(bank/'index.json')
    for item in index.values():
        p=Path(item['path']);s=read(p);s['frontend_policy_sha256']=digest;write(p,s);item['sha256']=file_hash(p)
    write(bank/'index.json',index);audit['index_sha256']=file_hash(bank/'index.json');write(bank/'independent_audit.json',audit)
    for p in (bank/'index.json',bank/'independent_audit.json'):config['pinned_inputs'][str(p)]=file_hash(p)
    result,meta=observation_sources(config,'icdar13',source,plan)
    assert result['v']['path']==str(sample) and meta['frontend_policy_sha256']==digest
    va=read(vb/'independent_audit.json');va['source_checkpoint_sha256']='wrong-model';write(vb/'independent_audit.json',va)
    config['pinned_inputs'][str(vb/'independent_audit.json')]=file_hash(vb/'independent_audit.json')
    with pytest.raises(ValueError,match='validation exactly'):
        observation_sources(config,'icdar13',source,plan)
