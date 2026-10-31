"""Prepare the shared geometric-augmentation intervention against the old control."""

import argparse
import copy
from pathlib import Path

from scripts.run_uniform_unet_training import prepare,read,write
from scripts.icdar_budget_data import key
from segmentation.uniform_unet_fusion import file_hash,aggregate_counts


def run(path,smoke):
    c=read(path);prepare(path,smoke);role='smoke' if smoke else 'full';parent=Path(c['output_root'])/role
    for ds in c['dataset_order']:
        folder=parent/ds;plan=read(folder/'resolved_configuration.json');control=Path(c['matched_control_root'])/ds
        reference=read(control/'resolved_configuration.json');same=copy.deepcopy(plan['config'])
        for field in c['augmentation_training_keys']:same['training'][field]=reference['config']['training'][field]
        same['minimum_free_disk_bytes']=reference['config']['minimum_free_disk_bytes']
        if same!=reference['config'] or plan['initialization']!=reference['initialization'] or plan['train_records']!=reference['train_records']:
            raise ValueError('Geometric arm differs from control beyond declared augmentation and disk reserve')
        expected_parts={name:(ids[:(8 if name=='fit' else 4)] if smoke else ids) for name,ids in reference['partitions'].items()}
        if plan['partitions']!=expected_parts:raise ValueError('Geometric control partition mismatch')
        assert read(control/'independent_training_audit.json')['status']=='PASS'
        expected=aggregate_counts(read(control/'selection/epoch_000/samples'/(key(s)+'.json'))['counts'] for s in plan['partitions']['selection'])
        plan.update(experiment='uniform_geometric_training_v17',continuation_control=dict(kind='matched_original_initialization',
            expected_initial_validation=expected,source_plan_sha256=file_hash(control/'resolved_configuration.json')))
        write(folder/'resolved_configuration.json',plan)
    print('Geometric control verified; only declared training augmentation and disk reserve differ.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.config,a.smoke)
