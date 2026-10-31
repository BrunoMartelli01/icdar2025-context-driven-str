"""Qualify a predeclared fixed-epoch detector refit against its frozen gate."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def qualify(training_root, gate_root, output):
    plan = read(training_root/'resolved_configuration.json'); c = plan['config']
    complete = read(training_root/'completion.json'); manifest = read(training_root/'dataset_manifest.json')
    gate = read(gate_root/'selection_frozen.json'); q = read(gate_root/'deployment_qualification.json')
    if (plan['mode'] != 'approximate' or plan['smoke'] or plan['test_used'] or
            complete['status'] != 'COMPLETE' or complete['test_used'] or complete['smoke'] or
            c.get('fit_protocol') != 'all_train_fixed_epochs' or c['candidate_epochs'] != [c['epochs']] or
            c['train_args']['val'] is not False or set(c['arms']) != {'head_only'} or
            gate['test_used_for_selection'] or q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']):
        raise ValueError('A frozen gate and completed fixed-epoch APPROXIMATE final refit are required')
    for p, digest in q['artifact_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Changed gate qualification artifact')
    for name in ('selection_frozen.json', 'deployment_qualification.json', 'emission_frozen.json'):
        p = gate_root/name
        if c['pinned_inputs'].get(str(p)) != sha(p):
            raise ValueError('Refit was not predeclared against this frozen gate and emission policy')
    original = read(c['original_plan']); parts = original['partitions']['icdar13']
    train = sorted(parts['fit']+parts['selection']+parts['calibration'])
    validation = sorted(parts['selection']+parts['calibration'])
    if (manifest['roles']['train'] != train or manifest['roles']['validation'] != validation or
            set(manifest['records']) != set(train) or not manifest['validation_overlaps_training'] or
            set(validation)-set(train) or manifest['test_used']):
        raise ValueError('Final fitting image/validation-shadow manifest differs')
    records = {r['sample_id']: r for r in original['records']['icdar13']['train']}
    test_rgb = {r['rgb_sha256'] for r in original['records']['icdar13']['test']}
    if test_rgb & {records[s]['rgb_sha256'] for s in train}:
        raise ValueError('Final refit TRAIN and TEST RGB overlap')
    for sid in train:
        r = records[sid]; saved = manifest['records'][sid]
        if (r['split'] != 'train' or saved['role'] != 'train' or
                sha(r['image_path']) != saved['original_image_sha256'] or
                sha(r['annotation_path']) != saved['annotation_sha256']):
            raise ValueError('Changed final refit image or annotation')
    training = complete['arms']['head_only']; epochs = training['epochs']
    if (len(epochs) != c['epochs'] or [v['epoch'] for v in epochs] != list(range(1, c['epochs']+1)) or
            not all(v['frozen_exact'] for v in epochs) or not training['initialization']['strict'] or
            any(training['initialization'][k] for k in ('missing', 'unexpected', 'wrong_shapes'))):
        raise ValueError('Incomplete strict initialization or fixed epoch/backbone audit')
    relative = 'head_only/candidates/epoch_'+format(c['epochs'], '03d')+'.pt'
    checkpoint = training_root/relative
    if set(training['checkpoint_sha256']) != {relative} or sha(checkpoint) != training['checkpoint_sha256'][relative]:
        raise ValueError('A single exact final epoch checkpoint is required')
    source = YOLO(c['checkpoint']).model.float().cpu().state_dict()
    fitted = YOLO(str(checkpoint)).model.float().cpu().state_dict()
    if set(source) != set(fitted) or any(source[k].shape != fitted[k].shape for k in source):
        raise ValueError('Refit state keys or shapes differ')
    frozen = tuple('model.'+str(i)+'.' for i in range(c['arms']['head_only']))
    changed = [k for k in source if not torch.equal(source[k], fitted[k])]
    if any(k.startswith(frozen) for k in changed) or not any(k.endswith(('.weight', '.bias')) for k in changed):
        raise ValueError('Refit frozen backbone or learned head audit failed')
    report = dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        training_root=str(training_root), training_completion_sha256=sha(training_root/'completion.json'),
        training_manifest_sha256=sha(training_root/'dataset_manifest.json'),
        training_configuration_sha256=sha(training_root/'resolved_configuration.json'),
        gate_root=str(gate_root), gate_selection_sha256=sha(gate_root/'selection_frozen.json'),
        gate_qualification_sha256=sha(gate_root/'deployment_qualification.json'),
        emission_policy_sha256=sha(gate_root/'emission_frozen.json'),
        fit_images=len(train), validation_shadow_images=len(validation), fixed_epoch=c['epochs'],
        state_tensors=len(source), changed_tensors=len(changed), frozen_tensors=sum(k.startswith(frozen) for k in source),
        peak_allocated=max(v['allocated'] for v in epochs), peak_reserved=max(v['reserved'] for v in epochs),
        out_of_memory=training['out_of_memory'], test_used_for_selection=False,
        historical_test_exposure=True,
        scope='The fixed detector refit is bound to the previously selected OOF-trained gate; the gate has not been refitted to this checkpoint. Validation shadow is not independent.')
    seal(output, report)
    print(__import__('json').dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--gate-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); qualify(a.training_root, a.gate_root, a.output)
