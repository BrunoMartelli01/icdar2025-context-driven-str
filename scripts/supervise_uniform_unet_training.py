"""Resource-aware sequential smoke/audit/full training; never interrupts other jobs."""

import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.run_uniform_unet_training import read, write


def run(config_path):
    config = read(config_path)
    base = Path(config['output_root'])
    with (base/'training_supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = [(role, dataset) for role in ('smoke', 'full') for dataset in config['dataset_order']]
        for role, dataset in stages:
            root = base/role/dataset
            if not (root/'resolved_configuration.json').exists():
                raise ValueError('Prepare every case before supervision: '+str(root))
            while not (root/'complete.json').exists():
                free_disk = shutil.disk_usage(base).free
                if free_disk < config['shared']['minimum_free_disk_bytes']:
                    write(base/'training_supervisor_status.json', dict(status='FAILED_DISK_RESERVE', stage=role, dataset=dataset, free_disk_bytes=free_disk))
                    raise RuntimeError('Disk reserve exhausted; no model was substituted')
                response = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
                free_gpu = int(response.stdout.strip().splitlines()[0])*1024**2
                if free_gpu < config['shared']['minimum_free_gpu_bytes']:
                    write(base/'training_supervisor_status.json', dict(status='WAITING_GPU', stage=role, dataset=dataset,
                        free_gpu_bytes=free_gpu, required_gpu_bytes=config['shared']['minimum_free_gpu_bytes'], free_disk_bytes=free_disk))
                    time.sleep(20)
                    continue
                write(base/'training_supervisor_status.json', dict(status='TRAINING', stage=role, dataset=dataset,
                    free_gpu_bytes=free_gpu, free_disk_bytes=free_disk))
                snapshot = base/role/'code_snapshot'
                with (base/(role+'_'+dataset+'_train.log')).open('a') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'scripts.run_uniform_unet_training',
                        '--stage', 'train', '--run-root', str(root)], cwd=snapshot,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (root/'independent_training_audit.json').exists():
                with (base/(role+'_'+dataset+'_audit.log')).open('a') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'scripts.audit_uniform_unet_training',
                        '--run-root', str(root)], stdout=log, stderr=subprocess.STDOUT, check=True)
            if read(root/'independent_training_audit.json')['status'] != 'PASS':
                raise RuntimeError('Training audit did not pass: '+str(root))
        write(base/'training_supervisor_status.json', dict(status='ALL_TRAINING_AUDITED', cases=stages,
            note='Router fitting and full TEST evaluation are subsequent required stages'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.config)
    except BaseException as exc:
        config = read(args.config)
        write(Path(config['output_root'])/f'training_supervisor_failure_{time.time_ns()}.json',
              dict(status='FAILED', error=repr(exc)))
        raise
