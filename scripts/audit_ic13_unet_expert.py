"""Audit U-Net/PARSeq exported maps, crops and decoder-heldout boundaries."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.adapted_word_policy import emit
from recognition.common_budget_pipeline import map_features
from recognition.group_word_selection import reading_confidence
from recognition.unet_token_routing import unet_proposals
from segmentation.icdar13_word_refit import padded_rows
from scripts.crop_clip_runtime import array_sha
from scripts.icdar_budget_data import read, sha, key
from scripts.run_yolo_f1_quality_study import seal


def audit(root):
    plan = read(root / 'resolved_configuration.json'); c = plan['config']
    if plan['mode'] != 'approximate' or plan['ground_truth_used'] or plan['test_used_for_selection']:
        raise ValueError('GT-free approximate expert export required')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed U-Net export source')
    for file, digest in plan['source_sha256'].items():
        if sha(root / 'code_snapshot' / file) != digest:
            raise ValueError('Changed saved U-Net export source code')
    full = read(c['full_training_plan']); policy_source = read(c['source_policy']); frozen = read(c['frozen_expert'])
    if (read(c['source_expert_audit'])['status'] != 'PASS' or
            policy_source['test_used_for_selection'] or frozen['test_used_for_selection']):
        raise ValueError('Source U-Net qualification changed')
    original = read(c['original_plan']); split = 'test' if c['kind'] == 'test' else 'train'
    records = {r['sample_id']: r for r in original['records']['icdar13'][split]}
    ids = c['prediction_ids']; index = read(root / 'index.json'); complete = read(root / 'completion.json')
    if (complete['status'] != 'COMPLETE' or complete['ground_truth_used'] or
            set(index) != set(ids) or len(ids) != len(set(ids)) or complete['images'] != len(ids)):
        raise ValueError('Incomplete qualified U-Net export')
    if c['kind'] == 'fold':
        train_root = Path(c['fold_training_root']); training = read(train_root / 'resolved_configuration.json')
        if (read(train_root / 'independent_audit.json')['status'] != 'PASS' or
                set(ids) != set(training['partitions']['decoder_heldout']) or
                set(ids) & (set(training['fitting_ids']) | set(training['partitions']['selection'])) or
                plan['checkpoint_sha256'] != sha(train_root / 'text_encoder/last.pt')):
            raise ValueError('Decoder-heldout scope changed')
    elif plan['checkpoint_sha256'] != policy_source['choice']['checkpoint_sha256']:
        raise ValueError('Full-fitting U-Net checkpoint changed')
    role_by_id = {}
    for role in ('fit', 'selection', 'calibration'):
        role_by_id.update({sid: role for sid in full['partitions'][role]})
    settings = dict(policy_source['choice']['policy']); settings['minimum_transcript_probability'] = 0.0
    pins = {str(root / name): sha(root / name) for name in ('resolved_configuration.json', 'index.json', 'completion.json')}
    checks = dict(images=0, crops=0, predicted_map_replays=0, decoder_heldout_images=0)
    for sid in ids:
        record = records[sid]; entry = index[sid]; sample_path = Path(entry['path'])
        if sha(sample_path) != entry['sha256'] or sha(record['image_path']) != record['image_sha256']:
            raise ValueError('Changed U-Net sample or original RGB')
        sample = read(sample_path); mp = Path(sample['map_path'])
        if (sample['ground_truth_used'] or sample['image_sha256'] != record['image_sha256'] or
                sample['checkpoint_sha256'] != plan['checkpoint_sha256'] or
                sample['frozen_expert_sha256'] != sha(c['frozen_expert']) or
                sample['map_sha256'] != sha(mp) or sample['image_excluded'] or
                sample['decoder_fit_excluded'] != (c['kind'] == 'fold') or not sample['upstream_encoder_exposure']):
            raise ValueError('Changed U-Net identity or decoder exclusion classification')
        with np.load(mp, allow_pickle=False) as z:
            prob = z['probabilities']
        proposals, _, _ = unet_proposals(prob, record['image_wh'], settings)
        proposals = padded_rows(proposals, record['image_wh'], settings['padding'])
        raw = sample['raw_rows']
        if len(proposals) != len(raw):
            raise ValueError('U-Net connected-component candidate count changed')
        with Image.open(record['image_path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        for proposal, row in zip(proposals, raw):
            a, b, d, e = row['bbox']
            if (proposal['bbox'] != row['bbox'] or proposal['instance_id'] != row['instance_id'] or
                    abs(proposal['detector_score'] - row['detector_score']) > 1e-12 or
                    array_sha(np.ascontiguousarray(rgb[b:e, a:d])) != row['crop_rgb_sha256'] or
                    abs(reading_confidence(row)['confidence_balanced'] - row['confidence']) > 1e-12):
                raise ValueError('U-Net proposal, original crop or PARSeq confidence differs')
            checks['crops'] += 1
        expected, _ = emit(raw, None, frozen['winner']['policy'])
        if expected != sample['rows']:
            raise ValueError('Frozen initial emission changed')
        np.testing.assert_allclose(map_features(prob, record['image_wh'], raw, c['map_settings']),
                                   sample['map_summary'], rtol=1e-12, atol=1e-12)
        if c['kind'] == 'seen':
            source = Path(c['source_expert_root']) / 'selected_expert' / role_by_id[sid] / 'samples' / (key(sid) + '.json')
            saved = read(source)
            if saved['ground_truth_used_for_prediction'] or saved['image_sha256'] != record['image_sha256']:
                raise ValueError('Changed existing full U-Net source')
            original_raw = [dict(r, confidence=reading_confidence(r)['confidence_balanced']) for r in saved['raw_rows']]
            if original_raw != raw or saved['maps_sha256'] != sha(mp):
                raise ValueError('Existing U-Net raw output differs')
            pins[str(source)] = sha(source)
        checks['images'] += 1; checks['predicted_map_replays'] += 1
        checks['decoder_heldout_images'] += int(c['kind'] == 'fold')
        pins[str(sample_path)] = sha(sample_path); pins[str(mp)] = sha(mp)
    pins.update(c['pinned_inputs'])
    path = root / 'independent_audit.json'
    seal(path, dict(status='PASS', mode='approximate', checks=checks, ground_truth_used=False,
        upstream_encoder_exposure=True, decoder_fit_excluded=c['kind'] == 'fold',
        auditor_sha256=sha(__file__), scope='Independent predicted map, crop and conditional decoder-split replay'))
    seal(root / 'deployment_qualification.json', dict(status='PASS', audit_path=str(path),
        audit_sha256=sha(path), artifact_sha256=pins))
    print('PASS', checks, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    audit(p.parse_args().run_root)
