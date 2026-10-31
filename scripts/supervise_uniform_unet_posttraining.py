"""Qualify common emissions and word-risk/fusion gates as each U-Net completes."""

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_uniform_unet_training import read, write


def run(training_config, router_config):
    config = read(training_config)
    base = Path(config['output_root'])
    with (base/'posttraining_supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def state(status, dataset, stage):
            write(base/'posttraining_supervisor_status.json', dict(status=status, dataset=dataset, stage=stage))
        def launch(dataset, stage, command, gpu=False):
            if gpu:
                while True:
                    output = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                        check=True, capture_output=True, text=True).stdout
                    free = int(output.strip().splitlines()[0])*1024**2
                    if free >= config['shared']['minimum_free_gpu_bytes']:
                        break
                    state('WAITING_GPU', dataset, stage)
                    time.sleep(20)
            state('RUNNING', dataset, stage)
            with (base/('post_'+dataset+'_'+stage+'.log')).open('a') as log:
                subprocess.run([sys.executable, '-u', '-m']+command, stdout=log,
                    stderr=subprocess.STDOUT, check=True)
        for dataset in config['dataset_order']:
            root = base/'full'/dataset
            while not (root/'independent_training_audit.json').exists():
                if list(base.glob('training_supervisor_failure_*.json')):
                    raise RuntimeError('U-Net supervisor failed; do not continue to router or TEST')
                state('WAITING_TRAINING', dataset, 'training_audit')
                time.sleep(20)
            if read(root/'independent_training_audit.json')['status'] != 'PASS':
                raise RuntimeError('Independent U-Net audit did not pass')
            if not (root/'emission_selection/frozen.json').exists():
                launch(dataset, 'emission_selection', ['scripts.export_uniform_unet', '--run-root', str(root), '--stage', 'selection'], True)
            if not (root/'export_train/complete.json').exists():
                launch(dataset, 'train_export', ['scripts.export_uniform_unet', '--run-root', str(root), '--stage', 'train'], True)
            if not (root/'router/selection_frozen.json').exists():
                launch(dataset, 'router_fit', ['scripts.train_uniform_risk_fusion', '--config', str(router_config), '--dataset', dataset])
            if not (root/'router/independent_router_audit.json').exists():
                launch(dataset, 'router_audit', ['scripts.audit_uniform_risk_fusion', '--config', str(router_config), '--dataset', dataset])
            if read(root/'router/independent_router_audit.json')['status'] != 'PASS':
                raise RuntimeError('Independent word-risk/fusion audit did not pass')
            if not (root/'export_test/complete.json').exists():
                launch(dataset, 'test_frontend_export', ['scripts.export_uniform_unet', '--run-root', str(root),
                    '--stage', 'test', '--frozen-router', str(root/'router/selection_frozen.json')], True)
        state('ALL_ROUTERS_AND_TEST_FRONTENDS_QUALIFIED', 'all', 'evaluation_pending')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--training-config', type=Path, required=True)
    p.add_argument('--router-config', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args.training_config, args.router_config)
    except BaseException as exc:
        base = Path(read(args.training_config)['output_root'])
        write(base/f'posttraining_supervisor_failure_{time.time_ns()}.json', dict(status='FAILED', error=repr(exc)))
        write(base/'posttraining_supervisor_status.json', dict(status='FAILED', error=repr(exc)))
        raise
