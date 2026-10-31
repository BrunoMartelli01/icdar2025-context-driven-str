"""Check the upper-encoder arm against the completed frozen-encoder control."""

import argparse
import copy
from pathlib import Path

from scripts.run_uniform_photometric_pilot import prepare
from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash


def run(path):
    c = read(path)
    prepare(path)
    for role in ('smoke', 'full'):
        for ds in c['dataset_order']:
            root = Path(c['output_root'])/'upper_encoder'/role/ds
            reference = Path(c['frozen_reference_root'])/role/ds
            plan = read(root/'resolved_configuration.json')
            control = read(reference/'resolved_configuration.json')
            same = copy.deepcopy(plan['config'])
            del same['encoder_adaptation']
            del same['training']['encoder_learning_rate']
            same['minimum_free_disk_bytes'] = control['config']['minimum_free_disk_bytes']
            if same != control['config'] or any(plan[k] != control[k]
                    for k in ('initialization', 'partitions', 'train_records')):
                raise ValueError('Undeclared change beyond encoder gradients/LR and storage reserve')
            assert read(reference/'independent_training_audit.json')['status'] == 'PASS'
            plan['matched_control'] = dict(root=str(reference),
                plan_sha256=file_hash(reference/'resolved_configuration.json'),
                complete_sha256=file_hash(reference/'complete.json'),
                audit_sha256=file_hash(reference/'independent_training_audit.json'))
            write(root/'resolved_configuration.json', plan)
    print('Matched frozen control verified; only upper encoder training/LR and storage reserve change.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    run(p.parse_args().config)
