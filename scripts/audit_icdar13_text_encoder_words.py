"""Check source identities, frozen encoders, fitting scopes and word counts."""

import argparse
import copy
import hashlib
import math
from pathlib import Path

import torch

from scripts.icdar_budget_data import read, sha, key
from scripts.audit_icdar13_dense_instances import count
from scripts.train_icdar13_text_encoder_words import model
from scripts.run_yolo_f1_quality_study import seal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    plan = read(root/'resolved_configuration.json')
    c = plan['config']
    torch.set_num_threads(c['cpu_threads'])
    for path, digest in c['pinned_inputs'].items():
        assert sha(path) == digest, path
    for name, digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest, name
    assert not set(plan['fitting_ids']) & set(plan['partitions']['selection']+
        plan['partitions']['calibration']+plan['partitions']['test'])
    lookup = {r['sample_id']: r for r in plan['train_records']+plan['test_records']}
    results = {}
    for arm in c['arms']:
        out = root/arm
        complete = read(out/'complete.json')
        assert complete['status'] == 'COMPLETE'
        assert complete['fit_ids'] == plan['fitting_ids']
        assert complete['checkpoint_sha256'] == sha(out/'best.pt')
        last = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        best = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        assert last['plan_sha256'] == best['plan_sha256'] == sha(root/'resolved_configuration.json')
        history = read(out/'epochs.json')
        assert history == last['history'] and len(history) == complete['epochs']
        cpu_plan = copy.deepcopy(plan)
        cpu_plan['config']['device'] = 'cpu'
        initial, receipt = model(cpu_plan, arm)
        assert receipt == read(out/'initialization.json')
        original = initial.state_dict()
        encoder_keys = [name for name in original if name.startswith('encoder.')]
        for name in encoder_keys:
            assert torch.equal(original[name], last['model_state_dict'][name]), name
            assert torch.equal(original[name], best['model_state_dict'][name]), name
        assert not torch.equal(original['up0.block.conv1.weight'],
                               last['model_state_dict']['up0.block.conv1.weight'])
        initial.load_state_dict(best['model_state_dict'], strict=True)
        expected_steps = math.ceil(math.ceil(len(plan['fitting_ids'])/
            c['training']['physical_batch_size'])/c['training']['gradient_accumulation'])
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row['images'] == len(plan['fitting_ids'])
            assert row['steps'] == expected_steps
            generator = torch.Generator().manual_seed(c['seed']+epoch)
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            order = [plan['fitting_ids'][i] for i in torch.randperm(
                len(plan['fitting_ids']), generator=generator).tolist()]
            assert hashlib.sha256(('\n'.join(order)+'\n').encode()).hexdigest() == row['order_sha256']
        options = []
        checked = 0
        for directory in sorted((out/'selection').glob('epoch_*')):
            metrics = read(directory/'metrics.json')
            epoch = int(directory.name.split('_')[-1])
            options.append((epoch, metrics['counts']))
            assert metrics['deepsolo_images'] == 0 and metrics['deepsolo_percentage'] == 0.
            for sid in plan['partitions']['selection']:
                data = read(directory/'samples'/(key(sid)+'.json'))
                assert not data['ground_truth_used_for_prediction']
                assert data['image_sha256'] == lookup[sid]['image_sha256']
                assert data['maps_sha256'] == sha(directory/'maps'/(key(sid)+'.npz'))
                assert count(data['rows'], lookup[sid]) == data['counts']
                checked += 1
        selected = max(options, key=lambda point: (point[1]['f1'], -point[0]))
        assert selected[0] == complete['selected_epoch'] == best['epoch']
        assert selected[1] == complete['best_selection'] == best['selection']
        results[arm] = dict(epochs=len(history), encoder_tensors_unchanged=len(encoder_keys),
            selection_image_replays=checked, selected_epoch=selected[0],
            peak_allocated=max(row['gpu_allocated'] for row in history),
            peak_reserved=max(row['gpu_reserved'] for row in history))
        del initial, last, best
    seal(args.output, dict(status='PASS', checks=results, mode='approximate',
        actual_new_deepsolo_recognizer_calls=0, code_sha256=sha(__file__),
        scope='Exact unchanged encoder state, real decoder updates, every epoch order/step, policy/model scope and all saved word counts. Not an independent neural refit.'))
    print('PASS', results, flush=True)


if __name__ == '__main__':
    main()
