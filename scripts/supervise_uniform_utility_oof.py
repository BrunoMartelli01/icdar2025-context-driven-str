"""After all neural banks qualify, refit/audit the same common v3 gate on CPU."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash

REPO = Path(__file__).resolve().parents[1]


def freeze_operating_points(c, plans, gates):
    """Choose shared fraction using only frozen validation metrics, before TEST."""
    fractions = sorted(c['alternate_fallback_fractions'])
    common = next((f for f in fractions if all(
        str(math.ceil(f*len(plans[d]['partitions']['test']))) in gates[d]['policies'] and
        gates[d]['policies'][str(math.ceil(f*len(plans[d]['partitions']['test'])))]['counts']['f1'] > plans[d]['target']
        for d in plans)), None)
    return dict(status='FROZEN', mode='approximate', test_used_for_selection=False,
        primary_quotas={d: p['budget'] for d, p in plans.items()},
        validation_recommended_quotas={d: gates[d]['validation_recommended_fallback_images'] for d in plans},
        validation_common_fraction=common,
        validation_common_quotas={d: math.ceil(common*len(p['partitions']['test'])) for d, p in plans.items()} if common is not None else None,
        descriptive_test_grid='Predeclared control points; TEST cannot select a deployment quota')


def prepare(path):
    c = read(path)
    if c['mode'] != 'approximate' or path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate supervision profile required')
    for p, digest in c['pinned_inputs'].items():
        if file_hash(p) != digest:
            raise ValueError('Changed utility supervisor input: '+p)
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'source_snapshot'; manifest = {}
    for package in c['snapshot_packages']:
        for src in (REPO/package).rglob('*.py'):
            dst = snapshot/src.relative_to(REPO); dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst); manifest[str(src.relative_to(REPO))] = file_hash(dst)
            if file_hash(src) != manifest[str(src.relative_to(REPO))]:
                raise ValueError('Utility source changed during freeze')
    shutil.copytree(REPO/'configs', snapshot/'configs')
    write(root/'source_manifest.json', manifest)
    write(root/'resolved_configuration.json', c)
    write(root/'contract.json', dict(status='FROZEN', config_sha256=file_hash(path),
        source_manifest_sha256=file_hash(root/'source_manifest.json'), test_used_for_selection=False))


def run(path):
    c = read(path); root = Path(c['output_root']); contract = read(root/'contract.json')
    if c['mode'] != 'approximate' or contract['config_sha256'] != file_hash(path):
        raise ValueError('Changed frozen utility supervisor profile')
    for p, digest in c['pinned_inputs'].items():
        if file_hash(p) != digest:
            raise ValueError('Changed supervisor input: '+p)
    def status(**fields):
        write(root/'supervisor_status.json', dict(recorded_unix=time.time(), supervisor_pid=os.getpid(), **fields))
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        neural = Path(c['decoder_root'])
        while not all((neural/d/'bank/independent_audit.json').exists() for d in c['dataset_order']):
            failures = sorted(neural.glob('supervisor_failure_*.json'))
            if failures:
                raise RuntimeError('Neural bank supervisor failed; inspect '+str(failures[-1]))
            launch = read(neural/'launch.json')
            if not (Path('/proc')/str(launch['pid'])).exists():
                raise RuntimeError('Neural supervisor is no longer live and qualified banks are incomplete')
            status(status='WAITING_QUALIFIED_NEURAL_BANKS', observed=read(neural/'supervisor_status.json'))
            time.sleep(20)
        target = Path(c['target_config'])
        if not target.exists():
            fit = read(c['base_utility_config'])
            fit['output_root'] = c['result_root']; fit['decoder_folds'] = c['decoder_folds']
            fit['fit_export_overrides'] = {d: str(neural/d/'bank') for d in c['dataset_order']}
            for d in c['dataset_order']:
                for name in ('index.json', 'independent_audit.json'):
                    p = neural/d/'bank'/name; fit['pinned_inputs'][str(p)] = file_hash(p)
            fit['scientific_choices']['ASSUMPTION'] += ' Replace native FIT observations only by complete independently qualified conditional decoder-heldout banks, with full-FIT deployment experts on validation/TEST; same runtime model family for all3.'
            fit['scientific_choices']['UNRESOLVED'] = 'Upstream encoder/initial decoder exposure, global validation epoch/emission selection, TRAIN vocabulary exposure, nonnested stacking, historical TEST research and final generalization/cost.'
            write(target, fit)
            write(root/'generated_config.json', dict(path=str(target), sha256=file_hash(target),
                neural_source_manifest_sha256=file_hash(neural/'source_manifest.json')))
        if read(root/'generated_config.json')['sha256'] != file_hash(target):
            raise ValueError('Changed generated frozen common v3 config')
        def execute(module, dataset, args):
            logfile = root/(module.rsplit('.', 1)[-1]+'_'+dataset+'.log')
            with logfile.open('a') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', module, '--config', str(target),
                    '--dataset', dataset, *args], cwd=root/'source_snapshot', stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status(status='RUNNING', module=module, dataset=dataset, child_pid=child.pid)
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError('Utility stage failed; inspect '+str(logfile))
        for d in c['dataset_order']:
            result = Path(c['result_root'])/d
            if not (result/'selection_frozen.json').exists(): execute('scripts.train_uniform_utility_fusion', d, [])
            if not (result/'independent_audit.json').exists(): execute('scripts.audit_uniform_utility_fusion', d, [])
        points = root/'operating_points_before_test.json'
        if not points.exists():
            fit = read(target)
            plans = {d: read(Path(fit['training_root'])/d/'resolved_configuration.json') for d in c['dataset_order']}
            gates = {d: read(Path(c['result_root'])/d/'selection_frozen.json') for d in c['dataset_order']}
            if any(read(Path(c['result_root'])/d/'independent_audit.json')['status'] != 'PASS' for d in c['dataset_order']):
                raise ValueError('Every common router must qualify before TEST')
            write(points, freeze_operating_points(fit, plans, gates))
        for d in c['dataset_order']:
            result = Path(c['result_root'])/d/'evaluation'
            args = ['--evaluation-config', c['evaluation_config']]
            if not (result/'results.json').exists(): execute('scripts.evaluate_uniform_utility_fusion', d, args)
            if not (result/'independent_audit.json').exists(): execute('scripts.audit_uniform_utility_evaluation', d, args)
        status(status='ALL_V3_TEST_RESULTS_AUDITED', result_root=c['result_root'], goal_complete=False,
            next_stage='Assess primary/validation-frozen operating points, scientific limits and actual online cost before declaring completion')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'run'], required=True); a = p.parse_args()
    if a.stage == 'prepare': prepare(a.config)
    else:
        try: run(a.config)
        except BaseException as exc:
            write(Path(read(a.config)['output_root'])/('failure_'+str(time.time_ns())+'.json'), dict(error=repr(exc), status='FAILED'))
            raise
