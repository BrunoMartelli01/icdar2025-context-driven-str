"""Independently check learned states, complete epochs and validation word counts."""

import argparse
import hashlib
import math
from pathlib import Path

import torch

from scripts.run_uniform_unet_training import read, write, make_model
from segmentation.uniform_unet_fusion import file_hash, annotation, word_counts, aggregate_counts
from scripts.icdar_budget_data import predictions, key


def audit(root):
    plan = read(root/'resolved_configuration.json')
    complete = read(root/'complete.json')
    c = plan['config']
    torch.set_num_threads(c['cpu_threads'])
    manifest = read(root.parent/'source_manifest.json')
    for name, digest in manifest.items():
        if file_hash(root.parent/'code_snapshot'/name) != digest:
            raise ValueError('Changed frozen source: '+name)
    last = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    assert last['plan_sha256'] == best['plan_sha256'] == file_hash(root/'resolved_configuration.json')
    assert file_hash(root/'best.pt') == complete['checkpoint_sha256']
    original_device = c['device']; c['device'] = 'cpu'
    net, receipt = make_model(plan)
    c['device'] = original_device
    assert receipt == read(root/'initialization.json')
    initial = net.state_dict()
    parameters = dict(net.named_parameters())
    adapted = 'encoder_adaptation' in c
    frozen = [k for k in initial if k.startswith('encoder.') and
              (not adapted or k not in parameters or not parameters[k].requires_grad)]
    assert all(torch.equal(initial[k], last['model_state_dict'][k]) and
               torch.equal(initial[k], best['model_state_dict'][k]) for k in frozen)
    if adapted:
        for stage in ('layer3', 'layer4'):
            trained = [k for k, p in parameters.items() if p.requires_grad and k.startswith('encoder.'+stage+'.')]
            assert trained and any(not torch.equal(initial[k], last['model_state_dict'][k]) for k in trained)
        groups = last['optimizer_state_dict']['param_groups']
        assert [g['lr'] for g in groups] == [c['training']['learning_rate'], c['training']['encoder_learning_rate']]
    assert not torch.equal(initial['up0.block.conv1.weight'], last['model_state_dict']['up0.block.conv1.weight'])
    assert not torch.equal(initial['head.weight'][0], last['model_state_dict']['head.weight'][0])
    assert not torch.equal(initial['head.weight'][1], last['model_state_dict']['head.weight'][1])
    history = read(root/'epochs.json')
    assert history == last['history'] and len(history) == complete['epochs']
    for epoch, row in enumerate(history, 1):
        assert row['epoch'] == epoch and row['images'] == len(plan['partitions']['fit'])
        t = c['training']
        assert row['steps'] == math.ceil(math.ceil(row['images']/t['physical_batch_size'])/t['gradient_accumulation'])
        generator = torch.Generator().manual_seed(c['seed']+epoch)
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        ids = [plan['partitions']['fit'][i] for i in torch.randperm(row['images'], generator=generator).tolist()]
        assert hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest() == row['order_sha256']
    lookup = {r['sample_id']: r for r in plan['train_records']}
    choices, checked = [], 0
    for directory in sorted((root/'selection').glob('epoch_*')):
        counts = []
        for sid in plan['partitions']['selection']:
            data = read(directory/'samples'/(key(sid)+'.json'))
            assert not data['ground_truth_used_for_prediction']
            assert data['image_sha256'] == lookup[sid]['image_sha256']
            assert file_hash(directory/'maps'/(key(sid)+'.npz')) == data['maps_sha256']
            if adapted:
                import numpy as np
                with np.load(directory/'maps'/(key(sid)+'.npz'), allow_pickle=False) as z:
                    assert z['encoder_context'].shape == (4096,) and np.isfinite(z['encoder_context']).all()
            truth, validity = annotation(lookup[sid])
            ignored = [p['ignored'] for p in predictions(data['rows'], lookup[sid], validity)]
            recount = word_counts(data['rows'], truth, ignored)
            assert recount == data['counts']
            counts.append(recount); checked += 1
        aggregate = aggregate_counts(counts)
        assert aggregate == read(directory/'metrics.json')['counts']
        choices.append((int(directory.name.split('_')[-1]), aggregate))
    selected = max(choices, key=lambda v: (v[1]['f1'], -v[0]))
    assert selected[0] == best['epoch'] == complete['selected_epoch']
    assert selected[1] == complete['best_selection']
    net.load_state_dict(best['model_state_dict'], strict=True)
    if adapted:
        import numpy as np
        from segmentation.icdar13_word_refit import WordDataset
        from segmentation.uniform_encoder_adaptation import validation_forward
        net.eval()
        probes = plan['partitions']['selection'][:2]
        data = WordDataset([lookup[s] for s in probes], c, augment=False)
        for i, sid in enumerate(probes):
            with torch.inference_mode():
                logits, context = validation_forward(net, data[i]['image'][None], c)
            mp = root/'selection'/f'epoch_{selected[0]:03d}'/'maps'/(key(sid)+'.npz')
            with np.load(mp, allow_pickle=False) as z:
                # CPU versus CUDA FP32 kernels are not generally bitwise equal.
                np.testing.assert_allclose(context[0].numpy(), z['encoder_context'], rtol=3e-4, atol=3e-4)
                np.testing.assert_allclose(logits[0].sigmoid().numpy(), z['probabilities'], rtol=3e-4, atol=3e-4)
    write(root/'independent_training_audit.json', dict(status='PASS', epochs=len(history),
        frozen_encoder_tensors=len(frozen), both_heads_learned=True, validation_image_recounts=checked,
        selected_epoch=selected[0], best_checkpoint_sha256=file_hash(root/'best.pt'),
        audit_source_sha256=file_hash(__file__), limitations='Word recount shares declared annotation/ignore primitives; no independent neural refit'))
    print('PASS', root, 'epochs', len(history), 'recounts', checked, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    audit(p.parse_args().run_root)
