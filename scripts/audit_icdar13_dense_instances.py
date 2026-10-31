"""Independent geometry, cache, fit-scope and lexical count audit."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects
import torch

from models.unet_dense_instances import DenseWordInstances
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_yolo_f1_quality_study import seal


def count(rows, record):
    truth, valid = annotation(record)
    pp = predictions(rows, record, valid)
    normalize = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
    care = [p for p in pp if not p['ignored']]
    emitted = Counter(normalize(p['text']) for p in care if normalize(p['text']))
    gt = Counter(normalize(p['text']) for p in truth if normalize(p['text']))
    tp = sum((emitted & gt).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp, detected_words=len(care),
                gt_words=len(truth), all_outputs=len(pp), ignored_outputs=len(pp)-len(care))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    plan = read(root/'resolved_configuration.json')
    c = plan['config']
    for name, digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest, name
    for path, digest in c['pinned_inputs'].items():
        assert sha(path) == digest, path
    fit_ids = plan['tt_ids']+plan['partitions']['fit']
    reserved_ids = plan['partitions']['selection']+plan['partitions']['calibration']+plan['partitions']['test']
    assert not set(fit_ids) & set(reserved_ids)
    checked = Counter()
    for sid, r in plan['records'].items():
        for field in ('image', 'validity', 'owners'):
            assert sha(r[field+'_path']) == r[field+'_sha256']
        with Image.open(r['owners_path']) as image:
            owners = np.asarray(image, np.int32)
        regions = find_objects(owners)
        assert all(region is not None for region in regions)
        boxes = [[region[1].start, region[0].start, region[1].stop, region[0].stop] for region in regions]
        assert boxes == r['boxes_original']
        width, height = r['image_wh']
        side = max(width, height)
        off = np.asarray([(side-width)//2, (side-height)//2]*2)
        expected = (np.asarray(boxes)+off)/side
        assert np.allclose(expected, r['boxes_canvas'], atol=5e-8, rtol=1e-7)
        centers = (expected[:, :2]+expected[:, 2:])/2*np.asarray(c['feature_hw'][::-1])
        cells = centers.astype(int)
        assert len({tuple(row) for row in cells}) == len(boxes) == r['positive_centers']
        checked['target_images'] += 1
        checked['target_words'] += len(boxes)
    for receipt_name in ('train_cache_receipt.json', 'test_cache_receipt.json'):
        path = root/receipt_name
        if not path.exists():
            continue
        receipt = read(path)
        assert receipt['status'] == 'PASS' and receipt['actual_new_deepsolo_inferences'] == 0
        for name, digest in receipt['files_sha256'].items():
            assert sha(name) == digest
            with np.load(name, allow_pickle=False) as saved:
                assert saved['features'].dtype == np.float32
                assert saved['features'].shape == (c['architecture']['input_channels'], *c['feature_hw'])
                assert saved['source_checkpoint_sha256'].item() == c['pinned_inputs'][c['source_checkpoint']]
                assert np.isfinite(saved['features']).all()
            checked['cache_files'] += 1
    completed = root/'head/complete.json'
    if completed.exists():
        complete = read(completed)
        assert complete['status'] == 'COMPLETE' and complete['fit_ids'] == fit_ids
        assert complete['selection_ids'] == plan['partitions']['selection']
        assert complete['checkpoint_sha256'] == sha(root/'head/best.pt')
        epochs = read(root/'head/epochs.json')
        assert len(epochs) == complete['epochs']
        resume_receipt = root.parent/'resume_v2_receipt.json'
        if root.name == 'full' and resume_receipt.exists():
            resume = read(resume_receipt)
            assert resume['checkpoint_sha256'] == sha(root/'failed_epoch2_checkpoint.pt')
            assert resume['plan_sha256'] == sha(root/'resolved_configuration.json')
            assert resume['patch_sha256'] == sha(root.parent/'writer_fix_v2.patch')
            executed = read(resume['source_manifest'])
            snapshot = Path(resume['source_manifest']).parent
            for name, digest in executed.items():
                assert sha(snapshot/name) == digest
            changed = [name for name, digest in plan['code_sha256'].items()
                       if executed.get(name) != digest]
            assert changed == ['scripts/run_icdar13_dense_instances.py']
            saved_resume = torch.load(root/'failed_epoch2_checkpoint.pt', map_location='cpu', weights_only=False)
            assert saved_resume['history'] == epochs[:resume['resume_epoch']]
            checked['qualified_resume'] += 1
        for epoch, row in enumerate(epochs, 1):
            assert row['epoch'] == epoch and row['images'] == len(fit_ids)
            generator = torch.Generator().manual_seed(c['seed']+epoch)
            # DataLoader consumes its base worker seed before RandomSampler.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            permutation = torch.randperm(len(fit_ids), generator=generator).tolist()
            order = [fit_ids[i] for i in permutation]
            assert hashlib.sha256(('\n'.join(order)+'\n').encode()).hexdigest() == row['order_sha256']
            checked['epoch_orders'] += 1
        head = DenseWordInstances(**c['architecture'])
        state = torch.load(root/'head/best.pt', map_location='cpu', weights_only=False)
        head.load_state_dict(state['model_state_dict'], strict=True)
        assert state['plan_sha256'] == sha(root/'resolved_configuration.json')
        eligible = [row for row in epochs if 'selection' in row]
        selected = max(enumerate(eligible), key=lambda item: (item[1]['selection']['f1'], -item[0]))[1]
        assert selected['epoch'] == state['epoch'] and selected['selection'] == state['selection']
    lookup = dict(plan['records'], **{r['sample_id']: r for r in plan['test_records']})
    for path in (root/'head/selection').glob('epoch_*/samples/*.json'):
        data = read(path)
        sid = next(s for s in plan['partitions']['selection'] if key(s)+'.json' == path.name)
        assert count(data['rows'], lookup[sid]) == data['counts']
        checked['epoch_selection_counts'] += 1
    for path in (root/'selected_expert').glob('*/samples/*.json'):
        data = read(path)
        assert count(data['rows'], lookup[data['sample_id']]) == data['counts']
        checked['final_expert_counts'] += 1
    seal(args.output, dict(status='PASS', checks=dict(checked),
        deepsolo_neural_inferences=0, code_sha256=sha(__file__),
        scope='Full geometry/cache identities and fitting scopes; independent epoch orders, model choice and lexical per-image counts. Not an independent neural refit.'))
    print('PASS', dict(checked), flush=True)


if __name__ == '__main__':
    main()
