"""Audit native word survival without dropping any evaluation care annotation."""

import argparse
from pathlib import Path

import torch

from scripts.run_uniform_unet_training import read, write
from segmentation.icdar13_word_refit import WordDataset
from segmentation.separated_word_gt import audit_word_owners
from segmentation.uniform_unet_fusion import file_hash


def audit(root):
    plan = read(root/'resolved_configuration.json')
    c = plan['config']
    torch.set_num_threads(c['cpu_threads'])
    lookup = {r['sample_id']: r for r in plan['train_records']}
    ids = plan['partitions']['fit']+plan['partitions']['selection']
    records = [lookup[s] for s in ids]
    dataset = WordDataset(records, c, augment=False)
    rows = []
    for i, r in enumerate(records):
        batch = dataset[i]
        value = audit_word_owners(batch['owners'].numpy(), r['care_words'])
        if value['unexpected_word_ids'] or value['interword_contact_pairs']:
            raise ValueError('Unexpected ownership/contact topology: '+r['sample_id'])
        rows.append(dict(sample_id=r['sample_id'], **value))
    report = dict(status='PASS', images=len(rows), care_words=sum(r['words'] for r in rows),
        native_missing_words=sum(len(r['missing_word_ids']) for r in rows),
        native_fragmented_words=sum(len(r['fragmented_word_ids']) for r in rows),
        rows=rows, source_sha256=file_hash(__file__),
        policy='Native loss/fragmentation is explicitly counted, not removed from evaluation care denominators; no source masks overwritten',
        image_hw=c['image_hw'], native_hw=[v//2 for v in c['image_hw']])
    write(root/'native_target_audit.json', report)
    print('TARGETS', plan['dataset'], {k: report[k] for k in ['images', 'care_words', 'native_missing_words', 'native_fragmented_words']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    audit(p.parse_args().run_root)
