"""Run the frozen U-Net folds sequentially, then audit and export each fold."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def status(root, **value):
    target = root / 'supervisor_status.json'
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(dict(recorded_unix=time.time(), **value), indent=2, sort_keys=True) + '\n')
    temporary.replace(target)


def gpu_check(root, label, minimum):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip()
    if len(output.splitlines()) != 1 or int(output.split(',')[-2].strip()) * 1024**2 < minimum:
        raise RuntimeError('Insufficient free memory on the declared single GPU')
    seal(root / 'hardware' / (label + '.json'), dict(observed=output, recorded_unix=time.time()))


def run(config_path, existing_pid):
    config = read(config_path); root = Path(config['output_root'])
    manifest = read(root / 'folds_frozen.json'); export_template = read(config['export_template'])
    if (config['mode'] != 'approximate' or manifest['status'] != 'FROZEN' or
            sha(config['export_template']) != config['template_sha256']):
        raise ValueError('Changed frozen supervisor or export template')
    worker = root / 'worker_snapshot'
    if worker.exists():
        raise ValueError('Preserve the existing supervisor snapshot; inspect its status before resuming')
    worker.mkdir()
    for package in export_template['snapshot_packages']:
        for source in (REPO / package).rglob('*.py'):
            target = worker / source.relative_to(REPO); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copytree(REPO / 'configs', worker / 'configs')
    seal(root / 'supervisor_source.json', dict(config_sha256=sha(config_path),
        supervisor_sha256=sha(__file__), worker_source_sha256={str(p.relative_to(worker)): sha(p)
            for p in worker.rglob('*.py')}))
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def execute(args, cwd, log, stage, fold):
        with log.open('x') as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', *args], cwd=cwd, env=env,
                                     stdout=output, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    status(root, status='RUNNING', stage=stage, fold=fold, child_pid=child.pid)
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f'{stage} fold {fold} failed with exit {child.returncode}; see {log}')
            except BaseException:
                if child.poll() is None:
                    child.terminate(); child.wait()
                raise
    try:
        for detail in manifest['folds']:
            i = detail['fold']; training = root / 'training' / ('fold_' + str(i))
            complete_path = training / 'text_encoder/complete.json'
            if not complete_path.exists() and i == 0 and existing_pid:
                command_path = Path('/proc') / str(existing_pid) / 'cmdline'
                while command_path.exists():
                    if complete_path.exists():
                        break
                    if str(training) not in command_path.read_bytes().decode().replace('\0', ' '):
                        raise RuntimeError('The original training PID no longer identifies the expected fold')
                    status(root, status='RUNNING', stage='existing_training', fold=i, child_pid=existing_pid)
                    time.sleep(10)
                if not complete_path.exists():
                    raise RuntimeError('Existing fold training stopped without a successful completion; no automatic retry')
            elif not complete_path.exists():
                gpu_check(root, 'train_fold_' + str(i), config['minimum_free_gpu_bytes'])
                execute(['scripts.train_icdar13_text_encoder_words', '--run-root', str(training),
                         '--stage', 'train', '--arm', 'text_encoder'], training / 'code_snapshot',
                        root / ('fold_' + str(i) + '_train.log'), 'training', i)
            if read(complete_path)['status'] != 'COMPLETE':
                raise ValueError('Fold training is incomplete')
            audit = training / 'independent_audit.json'
            if not audit.exists():
                execute(['scripts.audit_icdar13_text_encoder_words', '--run-root', str(training),
                         '--output', str(audit)], training / 'code_snapshot',
                        root / ('fold_' + str(i) + '_audit.log'), 'training_audit', i)
            if read(audit)['status'] != 'PASS':
                raise ValueError('Training audit failed')
            export = dict(export_template, kind='fold', prediction_ids=detail['heldout_ids'],
                output_root=str(root / 'fold_exports' / ('fold_' + str(i))),
                fold_training_root=str(training), fixed_epochs=config['fixed_epochs'])
            export['pinned_inputs'] = dict(export_template['pinned_inputs'])
            for file in [training / 'resolved_configuration.json', complete_path, audit, training / 'text_encoder/last.pt']:
                export['pinned_inputs'][str(file)] = sha(file)
            export_path = worker / 'configs/approximate' / ('ic13_unet_fold_export_' + str(i) + '.json')
            seal(export_path, export)
            export_root = Path(export['output_root'])
            gpu_check(root, 'export_fold_' + str(i), config['minimum_free_gpu_bytes'])
            execute(['scripts.export_ic13_unet_expert', '--config', str(export_path)], worker,
                    root / ('fold_' + str(i) + '_export.log'), 'heldout_export', i)
            execute(['scripts.audit_ic13_unet_expert', '--run-root', str(export_root)], worker,
                    root / ('fold_' + str(i) + '_export_audit.log'), 'export_audit', i)
            if read(export_root / 'independent_audit.json')['status'] != 'PASS':
                raise ValueError('Fold export audit failed')
        status(root, status='FOLDS_COMPLETE', stage='router_training_pending', folds_completed=len(manifest['folds']))
    except BaseException as error:
        status(root, status='FAILED', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--existing-pid', type=int)
    a = p.parse_args(); run(a.config, a.existing_pid)
