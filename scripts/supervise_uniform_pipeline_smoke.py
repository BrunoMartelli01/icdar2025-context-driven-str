"""Qualify the RGB-only chain on three small batches without stopping other jobs."""

import argparse
import fcntl
from pathlib import Path
import os
import subprocess
import time

from scripts.run_uniform_unet_training import read, write
from segmentation.uniform_unet_fusion import file_hash
from pipeline.uniform_word_pipeline import input_records, batch_quota


def audit(folder, manifest, reference_quota, reference_images, hard_budget=None):
    """Recount real calls and annotation-free handoff independently of run()."""
    records = input_records(read(manifest)); ids = {r['sample_id'] for r in records}
    complete = read(folder/'complete.json'); decisions = read(folder/'decisions_before_fallback.json')
    predictions = read(folder/'predictions.json'); index = read(folder/'frontend/index.json')
    if hard_budget is None:
        k = batch_quota(reference_quota, reference_images, len(ids))
    else:
        assert hard_budget['reference_images'] == reference_images
        assert 0 <= reference_quota <= hard_budget['maximum_fallback_images']
        assert hard_budget['batch_rounding'] == 'floor'
        k = reference_quota*len(ids)//reference_images
        assert k*reference_images <= hard_budget['maximum_fallback_images']*len(ids)
        assert decisions['hard_budget'] == complete['hard_budget'] == hard_budget
    selected, accepted = set(decisions['fallback_ids']), set(decisions['accepted_ids'])
    assert complete['status'] == 'COMPLETE' and not complete['ground_truth_used']
    assert decisions['status'] == 'FROZEN' and not decisions['ground_truth_used'] and not decisions['fallback_outputs_loaded']
    assert selected.isdisjoint(accepted) and selected|accepted == ids and len(selected) == k
    assert set(predictions) == set(index) == ids and complete['images'] == len(ids)
    assert complete['deepsolo_images'] == complete['actual_new_deepsolo_calls'] == k
    assert complete['reused_deepsolo_outputs'] == 0 and complete['actual_new_unet_calls'] == len(ids)
    assert complete['total_images'] == len(ids) and abs(complete['deepsolo_percentage']-100*k/len(ids)) < 1e-12
    crops = 0; peaks = []
    for sid in ids:
        assert file_hash(index[sid]['path']) == index[sid]['sha256']
        sample = read(index[sid]['path']); crops += len(sample['raw_rows'])
        assert not sample['ground_truth_used'] and file_hash(sample['maps_path']) == sample['maps_sha256']
        assert not predictions[sid]['ground_truth_used']
        assert predictions[sid]['action'] == ('deepsolo' if sid in selected else 'light')
        if sid in selected:
            raw = read(folder/'fallback'/(sid.replace(':', '__')+'.json'))
            assert raw['status'] == 'SUCCESS' and not raw['ground_truth_used']
            assert raw['contract']['sample_id'] == sid
            peaks.append((raw['gpu_peak_allocated_bytes'], raw['gpu_peak_reserved_bytes']))
    assert complete['actual_new_parseq_crop_calls'] == crops
    assert file_hash(folder/'predictions.json') == complete['predictions_sha256']
    plan = read(folder/'fallback_plan.json')
    assert plan['selected_before_inference'] and not plan['ground_truth_paths_included']
    samples = next(iter(plan['population']['test'].values()))
    assert {r['sample_id'] for r in samples} == selected
    assert all(set(r) == {'sample_id', 'image_path', 'image_sha256'} for r in samples)
    write(folder/'independent_engineering_audit.json', dict(status='PASS', result_class='smoke-test result',
        mode='approximate', images=len(ids), deepsolo_images=k, total_images=len(ids), deepsolo_percentage=100*k/len(ids),
        actual_new_deepsolo_calls=k, actual_new_unet_calls=len(ids), actual_parseq_crop_calls=crops,
        frontend_peak_allocated=complete['frontend']['peak_allocated'], frontend_peak_reserved=complete['frontend']['peak_reserved'],
        fallback_peak_allocated=max((a for a, _ in peaks), default=0), fallback_peak_reserved=max((b for _, b in peaks), default=0),
        complete_sha256=file_hash(folder/'complete.json'), source_sha256=file_hash(__file__),
        no_ground_truth_in_runtime_input=True, out_of_memory=False,
        scope='Real neural calls and exact selected-only isolated handoff on a small RGB-only batch. Shared GPU activity allowed; no full-dataset quality or general speed claim.'))


def run(config_path):
    c = read(config_path); root = Path(c['output_root'])
    if c['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate engineering profile required')
    for path, digest in c['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed engineering input: '+path)
    root.mkdir(parents=True, exist_ok=True)
    def status(**fields):
        write(root/'supervisor_status.json', dict(recorded_unix=time.time(), supervisor_pid=os.getpid(), **fields))
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        dependency = Path(c['neural_root'])
        while not (dependency/'smoke/heldout/independent_audit.json').exists():
            if list(dependency.glob('supervisor_failure_*.json')):
                raise RuntimeError('Neural engineering smoke failed; preserve and inspect its failure')
            pid = read(dependency/'launch.json')['pid']
            if not (Path('/proc')/str(pid)).exists():
                raise RuntimeError('Neural supervisor stopped before engineering qualification')
            status(status='WAITING_NEURAL_SMOKE', observed=read(dependency/'supervisor_status.json'))
            time.sleep(20)
        if read(dependency/'smoke/heldout/independent_audit.json')['status'] != 'PASS':
            raise ValueError('Neural engineering prerequisite did not qualify')
        for job in c['jobs']:
            folder = root/job['dataset']
            if not (folder/'complete.json').exists():
                if folder.exists():
                    raise ValueError('Preserve incomplete pipeline smoke; inspect before an explicit retry')
                while True:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True).stdout.strip()
                    if len(gpu.splitlines()) != 1:
                        raise RuntimeError('Expected the declared single-GPU host')
                    if int(gpu.split(',')[-2].strip())*1024**2 >= c['minimum_free_gpu_bytes']:
                        break
                    status(status='WAITING_GPU', dataset=job['dataset'], observed=gpu); time.sleep(20)
                processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                    capture_output=True, text=True, check=True).stdout
                write(root/(job['dataset']+'_resource_before.json'), dict(gpu=gpu, compute_processes=processes,
                    scope='Other active GPU jobs preserved; timing is engineering-only'))
                with (root/(job['dataset']+'.log')).open('a') as log:
                    child = subprocess.Popen([c['main_python'], '-u', '-m', 'scripts.infer_uniform_word_pipeline',
                        '--config', c['deployment_config'], '--manifest', job['manifest'], '--dataset', job['dataset'],
                        '--policy-quota', str(job['policy_quota']), '--output', str(folder)],
                        cwd=c['source_snapshot'], stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        status(status='RUNNING', dataset=job['dataset'], child_pid=child.pid); time.sleep(10)
                    if child.returncode:
                        raise RuntimeError('Live pipeline smoke failed; inspect '+str(root/(job['dataset']+'.log')))
            if not (folder/'independent_engineering_audit.json').exists():
                audit(folder, Path(job['manifest']), job['policy_quota'], job['reference_images'],
                      c.get('hard_budgets', {}).get(job['dataset']))
        status(status='ALL_THREE_RGB_CHAINS_QUALIFIED', result_class='smoke-test result', goal_complete=False,
            next_stage='v3 quality/target audit and matched idle-GPU final cost evaluation remain required')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True); a = p.parse_args()
    try: run(a.config)
    except BaseException as exc:
        write(Path(read(a.config)['output_root'])/('failure_'+str(time.time_ns())+'.json'), dict(status='FAILED', error=repr(exc)))
        raise
