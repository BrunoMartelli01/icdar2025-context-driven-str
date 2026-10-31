"""Fail-closed selection of qualified native or decoder-heldout FIT observations."""

from pathlib import Path

from scripts.run_uniform_unet_training import read
from scripts.icdar_budget_data import key
from segmentation.uniform_unet_fusion import file_hash


def observation_sources(config, dataset, source, plan):
    source = Path(source)
    original = read(source/'export_train/index.json')
    fit, val = plan['partitions']['fit'], plan['partitions']['selection']
    if set(fit)&set(val) or not set(fit+val).issubset(original):
        raise ValueError('Complete disjoint native FIT/selection observations required')
    index = {s: original[s] for s in fit+val}
    bank_path = config.get('fit_export_overrides', {}).get(dataset)
    metadata = dict(decoder_fit_excluded=False, upstream_expert_image_excluded=False,
        selection_expert='qualified full-FIT deployment U-Net', fit_images=len(fit))
    frontend = config.get('observation_frontend_policy')
    validation_bank = config.get('validation_export_overrides', {}).get(dataset)
    expected_frontend = None
    if frontend is not None:
        expected_frontend = file_hash(frontend)
        if config['pinned_inputs'].get(frontend) != expected_frontend:
            raise ValueError('Pinned observation frontend policy required')
        if bank_path is None or validation_bank is None:
            raise ValueError('Matching FIT and validation banks are both required')
    elif validation_bank is not None:
        raise ValueError('Validation replacement requires an explicit matching frontend policy')
    if validation_bank is not None:
        vb = Path(validation_bank)
        for name in ('index.json', 'independent_audit.json'):
            p = vb/name
            if config['pinned_inputs'].get(str(p)) != file_hash(p):
                raise ValueError('Pinned full-FIT validation bank qualification required')
        va, vi = read(vb/'independent_audit.json'), read(vb/'index.json')
        if (va['status'] != 'PASS' or va['images'] != len(val) or set(vi) != set(val) or
            va['index_sha256'] != file_hash(vb/'index.json') or
            va['source_training_plan_sha256'] != file_hash(source/'resolved_configuration.json') or
            va['source_checkpoint_sha256'] != read(source/'complete.json')['checkpoint_sha256'] or
            va['frontend_policy_sha256'] != expected_frontend):
            raise ValueError('Matching full-FIT validation bank must cover validation exactly')
        for sid, entry in vi.items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Changed two-view validation observation')
            sample = read(entry['path'])
            if (sample['ground_truth_used'] or 'maps_path' not in sample or
                sample['frontend_policy_sha256'] != expected_frontend):
                raise ValueError('Annotation-free matching validation observation required')
        index.update(vi)
        metadata.update(selection_expert='qualified matching full-FIT two-view U-Net/PARSeq',
            validation_bank=str(vb), frontend_policy_sha256=expected_frontend)
    if bank_path is None:
        return index, metadata
    bank = Path(bank_path)
    for name in ('independent_audit.json', 'index.json'):
        path = bank/name
        if config['pinned_inputs'].get(str(path)) != file_hash(path):
            raise ValueError('Pinned decoder-heldout bank qualification required: '+str(path))
    audit = read(bank/'independent_audit.json')
    replacement = read(bank/'index.json')
    if (audit['status'] != 'PASS' or not audit['decoder_fit_excluded'] or
            audit['index_sha256'] != file_hash(bank/'index.json') or
            audit['source_training_plan_sha256'] != file_hash(source/'resolved_configuration.json') or
            audit['images'] != len(fit) or set(replacement) != set(fit)):
        raise ValueError('Qualified decoder-heldout bank must cover the original FIT exactly')
    if expected_frontend is not None and audit.get('frontend_policy_sha256') != expected_frontend:
        raise ValueError('FIT bank uses another observation frontend')
    if len(audit['folds']) != config['decoder_folds']:
        raise ValueError('All declared neural decoder folds must be qualified')
    for fold in audit['folds']:
        path = Path(fold['root'])/'heldout/independent_audit.json'
        if file_hash(path) != fold['audit_sha256'] or read(path)['status'] != 'PASS':
            raise ValueError('Changed independent neural-fold qualification')
    for sid, entry in replacement.items():
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Changed decoder-heldout observation')
        sample = read(entry['path'])
        if sample['ground_truth_used'] or not sample['decoder_fit_excluded'] or 'maps_path' not in sample:
            raise ValueError('Decoder-heldout annotation-free observation and explicit maps required')
        if expected_frontend is not None and sample.get('frontend_policy_sha256') != expected_frontend:
            raise ValueError('FIT observation uses another frontend')
    index.update(replacement)
    metadata.update(decoder_fit_excluded=True, bank=str(bank), bank_index_sha256=file_hash(bank/'index.json'),
        bank_audit_sha256=file_hash(bank/'independent_audit.json'), upstream_encoder_exposure=audit['upstream_encoder_exposure'],
        caution='Conditional decoder folds; shared exposed upstream initialization and global validation epoch/emission selection; not fully nested OOF')
    return index, metadata


def observation_map_path(source, sample_id, sample):
    return Path(sample['maps_path']) if 'maps_path' in sample else Path(source)/'export_train/maps'/(key(sample_id)+'.npz')
