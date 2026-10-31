"""Freeze conditional U-Net decoder folds using the existing YOLO image splits."""

import argparse
import copy
from pathlib import Path

from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def check_fold(parent, manifest, fold):
    parts = parent['partitions']; records = parent['records']
    fit, hold = set(fold['fit_ids']), set(fold['heldout_ids'])
    original = set(parts['fit']); validation = set(parts['selection'] + parts['calibration'])
    if (len(fit) != len(fold['fit_ids']) or len(hold) != len(fold['heldout_ids']) or
            fit & hold or fit | hold != original or validation & original or
            validation != set(manifest['validation_ids']) or original != set(manifest['source_fit_ids'])):
        raise ValueError('Fold does not preserve the qualified FIT and validation split')
    def rgb(ids):
        return {records[s]['rgb_sha256'] for s in ids}
    if rgb(hold) & (rgb(fit | validation) | rgb(parent['tt_ids'])):
        raise ValueError('A heldout image duplicates supervised decoder fitting or validation RGB')
    return dict(fit=sorted(fit), heldout=sorted(hold), validation=sorted(validation))


def run(path):
    if path.resolve().parent != REPO / 'configs/approximate':
        raise ValueError('Explicit approximate U-Net fold configuration required')
    c = read(path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed frozen U-Net fold source: ' + file)
    parent = read(c['parent_plan']); manifest = read(c['yolo_folds_manifest'])
    base = read(c['training_template']); expert = read(c['frozen_expert'])
    audit = read(c['training_audit'])
    if (parent['mode'] != 'approximate' or audit['status'] != 'PASS' or
            expert['status'] != 'FROZEN' or expert['test_used_for_selection'] or
            expert['choice']['selected_epoch'] != c['fixed_epochs'] or
            sha(expert['choice']['checkpoint']) != expert['choice']['checkpoint_sha256'] or
            manifest['test_used']):
        raise ValueError('Existing U-Net expert and YOLO fold identities must be qualified')
    if [f['fold'] for f in manifest['folds']] != list(range(c['folds'])):
        raise ValueError('Unexpected number of decoder folds')
    splits = [check_fold(parent, manifest, f) for f in manifest['folds']]
    held = [s for x in splits for s in x['heldout']]
    if sorted(held) != sorted(parent['partitions']['fit']):
        raise ValueError('The conditional decoder-heldout population must be covered once')
    root = Path(c['output_root'])
    if root.exists():
        raise ValueError('Preserve the existing U-Net fold run')
    outputs = [REPO / 'configs/approximate' / (c['generated_config_prefix'] + str(i) + '_v1.json')
               for i in range(c['folds'])]
    smoke_config = REPO / 'configs/approximate' / (c['generated_config_prefix'] + 'smoke_v1.json')
    if any(p.exists() for p in outputs + [smoke_config]):
        raise ValueError('Generated fold configurations already exist')
    root.mkdir(parents=True, exist_ok=False)
    plans = []
    for i, (fold, split, config_path) in enumerate(zip(manifest['folds'], splits, outputs)):
        parent_copy = copy.deepcopy(parent)
        parent_copy['partitions'] = dict(parent['partitions'], fit=split['fit'],
            selection=split['validation'], calibration=[], decoder_heldout=split['heldout'])
        parent_copy['decoder_oof'] = dict(fold=i, heldout_ids=split['heldout'],
            supervised_decoder_excludes_heldout=True, checkpoint_selection_excludes_heldout=True,
            upstream_text_encoder_includes_icdar13_train=True,
            scope='Decoder-fold exclusion only; pretrained encoder has known ICDAR2013 TRAIN exposure')
        fold_plan = root / 'fold_plans' / ('fold_' + str(i) + '.json')
        seal(fold_plan, parent_copy)
        cfg = copy.deepcopy(base); old_parent = cfg['parent_plan']; cfg['parent_plan'] = str(fold_plan)
        cfg['output_root'] = str(root / 'training'); cfg['arms'] = ['text_encoder']
        cfg['training']['epochs'] = c['fixed_epochs']; cfg['validation_interval'] = c['fixed_epochs']
        cfg['pinned_inputs'].pop(old_parent, None)
        cfg['pinned_inputs'][str(fold_plan)] = sha(fold_plan)
        cfg['pinned_inputs'][c['yolo_folds_manifest']] = sha(c['yolo_folds_manifest'])
        cfg['pinned_inputs'][c['training_audit']] = sha(c['training_audit'])
        cfg['pinned_inputs'][c['frozen_expert']] = sha(c['frozen_expert'])
        cfg['scientific_choices'] = c['scientific_choices']
        seal(config_path, cfg)
        plans.append(dict(fold=i, fit_ids=split['fit'], heldout_ids=split['heldout'],
            validation_ids=split['validation'], parent_plan=str(fold_plan),
            parent_plan_sha256=sha(fold_plan), config=str(config_path), config_sha256=sha(config_path)))
    smoke = read(outputs[0]); smoke['output_root'] = str(root / 'training')
    seal(smoke_config, smoke)
    seal(root / 'folds_frozen.json', dict(status='FROZEN', mode='approximate',
        claim_label='approximate reproduction', folds=plans, config_sha256=sha(path),
        fixed_epochs=c['fixed_epochs'], test_used_for_selection=False,
        decoder_heldout_only=True, upstream_encoder_exposure=True,
        scope='Five supervised decoder folds, conditional on the IC13-exposed pretrained encoder'))
    print('FROZEN', len(plans), 'folds', len(held), 'heldout TRAIN images', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    run(p.parse_args().config)
