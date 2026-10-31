"""Prepare, cache, train and qualify an approximate U-Net center/extent decoder."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from models.unet_dense_instances import DenseWordInstances
from segmentation.unet_dense_instances import center_targets, dense_instance_loss, instance_proposals
from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.icdar2015_improvements import make_transform, pad_to_aspect
from recognition.group_word_selection import reading_confidence
from recognition.icdar_budget_router import lexical_counts
from recognition.f1_quality_study import aggregate
from scripts.icdar_budget_data import annotation, predictions, read, sha, key
from scripts.train_icdar13_word_refit import recognizer, save_checkpoint
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def runtime(config):
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires its explicitly selected GPU')
    if torch.cuda.mem_get_info()[0] < config['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free GPU memory')


def prepare(args):
    if args.config.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate configuration required')
    config = read(args.config)
    assert config['mode'] == 'approximate' and config['claim_label'] == 'approximate reproduction'
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Pinned input changed: '+path)
    parent = read(config['parent_plan'])
    parts = parent['partitions']
    tt = parent['tt_records']
    ic = [r for r in parent['records'] if r['split'] == 'train']
    test = [r for r in parent['records'] if r['split'] == 'test']
    if args.smoke:
        parts = {role: ids[:(8 if role == 'fit' else 4)] for role, ids in parts.items()}
        included = {sid for role, ids in parts.items() if role != 'test' for sid in ids}
        ic = [r for r in ic if r['sample_id'] in included]
        tt = tt[:16]
    test_hashes = {r['rgb_sha256'] for r in test}
    if any(r['rgb_sha256'] in test_hashes for r in tt+ic):
        raise ValueError('TRAIN RGB overlaps the fixed ICDAR13 TEST population')
    root = Path(config['output_root'])/('smoke' if args.smoke else 'full')
    root.mkdir(parents=True, exist_ok=False)
    grid = config['feature_hw']
    records = {}
    count_words = dict(icdar13=0, total_text=0)
    for record in tt+ic:
        r = dict(record)
        for field in ('image', 'owners', 'validity'):
            if sha(r[field+'_path']) != r[field+'_sha256']:
                raise ValueError('Changed source '+r[field+'_path'])
        with Image.open(r['owners_path']) as image:
            owners = np.asarray(image, dtype=np.int32)
        objects = find_objects(owners)
        if any(region is None for region in objects):
            raise ValueError('Missing original owner identity: '+r['sample_id'])
        boxes = np.asarray([[region[1].start, region[0].start, region[1].stop, region[0].stop]
                            for region in objects], np.float32).reshape(-1, 4)
        width, height = r['image_wh']
        side = max(width, height)
        offset = np.asarray([(side-width)//2, (side-height)//2]*2, np.float32)
        normalized = (boxes+offset)/side
        with Image.open(r['validity_path']) as image:
            valid = np.asarray(pad_to_aspect(image.convert('L'), config['image_hw'], fill=0).
                               resize((grid[1], grid[0]), Image.Resampling.NEAREST)) > 0
        target = center_targets(normalized, valid, **config['targets'])
        if len(target['index']) != len(objects):
            raise ValueError('Lost GT word in center targets')
        if r['sample_id'].startswith('icdar13:') and len(objects) != r['care_words']:
            raise ValueError('IC13 care/owner identity mismatch')
        domain = 'icdar13' if r['sample_id'].startswith('icdar13:') else 'total_text'
        count_words[domain] += len(boxes)
        r.update(boxes_original=boxes.tolist(), boxes_canvas=normalized.tolist(), domain=domain,
                 positive_centers=int(target['heatmap'].eq(1).sum()))
        records[r['sample_id']] = r
    plan = dict(mode='approximate', claim_label='approximate reproduction', config=config,
        config_sha256=sha(args.config), partitions=parts, records=records, test_records=test,
        tt_ids=[r['sample_id'] for r in tt], smoke=args.smoke,
        metric='Unicode casefold+isalnum lexical multiset micro-F1, native256 ignore, all care GT/images',
        expected_test_images=233, expected_test_care_words=1095,
        result_class='smoke-test result' if args.smoke else 'approximate reproduction',
        historical_test_exposure=True, test_used_for_selection=False,
        target_audit=dict(images=len(records), words=count_words, center_collisions=0,
                          lost_centers=0, icdar13_test_rgb_overlap=0),
        paths=dict(parseq_source=config['parseq_source'], parseq_archive=config['parseq_archive'],
                   parseq_checkpoint=config['parseq_checkpoint']))
    source = {}
    for directory in ('models', 'segmentation', 'scripts', 'recognition', 'localize',
                      'caption', 'scoring', 'pipeline', 'research'):
        for path in (REPO/directory).rglob('*.py'):
            relative = str(path.relative_to(REPO))
            source[relative] = sha(path)
            destination = root/'code_snapshot'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    plan['code_sha256'] = source
    seal(root/'resolved_configuration.json', plan)
    print('PREPARED', root, plan['target_audit'], flush=True)
    return root


def source_net(plan):
    c = plan['config']
    if sha(c['source_checkpoint']) != c['pinned_inputs'][c['source_checkpoint']]:
        raise ValueError('Source checkpoint changed')
    net = WordRefitUNet(partial_encoder=False, skip_floor=c['skip_floor'],
                        initial_gap_bias=c['initial_gap_bias'])
    state = torch.load(c['source_checkpoint'], map_location='cpu', weights_only=False)
    net.load_state_dict(state['model_state_dict'], strict=True)
    net.requires_grad_(False)
    return net.to(c['device']).eval()


def image_tensor(record, config):
    if sha(record['image_path']) != record['image_sha256']:
        raise ValueError('Source RGB changed')
    with Image.open(record['image_path']) as image:
        rgb = image.convert('RGB')
    transform = make_transform(config, config['image_hw'], augment=False)
    value, _ = transform(rgb, Image.new('L', rgb.size, 0), 0)
    return value


def feature_path(root, sid):
    return root/'feature_cache'/(key(sid)+'.npz')


def cache(plan, root, test=False):
    c = plan['config']
    if shutil.disk_usage(root).free < c['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient disk space for FP32 feature caches')
    net = source_net(plan)
    records = ([r for r in plan['test_records'] if r['sample_id'] in plan['partitions']['test']]
               if test else list(plan['records'].values()))
    files = {}
    start = time.monotonic()
    for i, r in enumerate(records, 1):
        path = feature_path(root, r['sample_id'])
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved['image_sha256'].item() != r['image_sha256']:
                    raise ValueError('Cache/RGB mismatch')
            files[str(path)] = sha(path)
            continue
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            x = image_tensor(r, c)[None].to(c['device'])
            evidence = net.forward_evidence(x)
            features = torch.cat((F.avg_pool2d(evidence['features'], 2),
                                  F.avg_pool2d(evidence['words'].sigmoid(), 2)), 1)
            if features.shape != (1, c['architecture']['input_channels'], *c['feature_hw']):
                raise ValueError('U-Net feature shape mismatch')
            array = features[0].cpu().numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=array, image_sha256=r['image_sha256'],
                            source_checkpoint_sha256=c['pinned_inputs'][c['source_checkpoint']])
        seal(root/'feature_metadata'/(key(r['sample_id'])+'.json'), dict(
            image_sha256=r['image_sha256'], ground_truth_used_for_features=False,
            precision='float32', cache_dtype=str(array.dtype), batch_size=1,
            input_shape=list(x.shape), feature_shape=list(features.shape),
            gpu_allocated=torch.cuda.max_memory_allocated(),
            gpu_reserved=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(0),
            source_sha256=c['pinned_inputs'][c['source_checkpoint']], cache_sha256=sha(path)))
        files[str(path)] = sha(path)
        if i % 50 == 0 or i == len(records):
            print('CACHE', i, '/', len(records), 'seconds', time.monotonic()-start, flush=True)
    del net
    torch.cuda.empty_cache()
    seal(root/('test_cache_receipt.json' if test else 'train_cache_receipt.json'),
         dict(status='PASS', images=len(records), cache_dtype='float32', files_sha256=files,
              actual_new_deepsolo_inferences=0, gpu=torch.cuda.get_device_name(0)))


class CachedWords(Dataset):
    def __init__(self, plan, root, ids):
        self.plan, self.root, self.ids = plan, root, ids
        self.targets = {}
        c = plan['config']
        for sid in ids:
            r = plan['records'][sid]
            with Image.open(r['validity_path']) as image:
                valid = np.asarray(pad_to_aspect(image.convert('L'), c['image_hw'], fill=0).
                    resize(tuple(reversed(c['feature_hw'])), Image.Resampling.NEAREST)) > 0
            self.targets[sid] = center_targets(r['boxes_canvas'], valid, **c['targets'])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        with np.load(feature_path(self.root, sid), allow_pickle=False) as data:
            r = self.plan['records'][sid]
            if data['image_sha256'].item() != r['image_sha256']:
                raise ValueError('Training cache/RGB mismatch')
            features = torch.from_numpy(data['features'].copy())
        return features, self.targets[sid], sid


def collate(batch):
    return torch.stack([row[0] for row in batch]), [row[1] for row in batch], [row[2] for row in batch]


def new_head(config):
    return DenseWordInstances(**config['architecture']).to(config['device'])


def read_sample(plan, root, head, ocr, record, policy):
    c = plan['config']
    with np.load(feature_path(root, record['sample_id']), allow_pickle=False) as data:
        if data['image_sha256'].item() != record['image_sha256']:
            raise ValueError('Inference cache/RGB mismatch')
        features = torch.from_numpy(data['features'].copy())[None].to(c['device'])
    with torch.inference_mode():
        output = head(features)
    rows = instance_proposals(output, record['image_wh'],
        peak_threshold=policy['peak_threshold'], maximum_candidates=policy['maximum_candidates'],
        nms_iou=policy['nms_iou'], padding=policy['padding'], log_size_bounds=c['loss']['log_size_bounds'])
    with Image.open(record['image_path']) as image:
        rgb = np.ascontiguousarray(image.convert('RGB'))
    decoded = []
    for offset in range(0, len(rows), c['parseq_batch_size']):
        group = rows[offset:offset+c['parseq_batch_size']]
        crops = [np.ascontiguousarray(rgb[r['bbox'][1]:r['bbox'][3], r['bbox'][0]:r['bbox'][2]]) for r in group]
        readings = ocr.recognize_candidate_batch(crops)
        if len(readings) != len(group):
            raise RuntimeError('Incomplete PARSeq output')
        for row, value in zip(group, readings):
            reading = value.to_dict()
            merged = dict(row, **reading)
            merged['confidence'] = reading_confidence(merged)['confidence_balanced']
            decoded.append(merged)
    return decoded


def emitted(rows, policy):
    return [r for r in rows if r['detector_score'] >= policy['peak_threshold']
            and r['sequence_confidence_native'] >= policy['sequence_threshold']
            and any(char.isalnum() for char in r['raw_text'].casefold())][:policy['maximum_candidates']]


def evaluate(plan, root, head, ids, directory, policy, ocr):
    head.eval()
    lookup = dict(plan['records'], **{r['sample_id']: r for r in plan['test_records']})
    values = []
    torch.cuda.reset_peak_memory_stats()
    for sid in ids:
        record = lookup[sid]
        raw = read_sample(plan, root, head, ocr, record, policy)
        rows = emitted(raw, policy)
        gt, validity = annotation(record)
        counts = lexical_counts(predictions(rows, record, validity), gt)
        values.append(counts)
        seal(directory/'samples'/(key(sid)+'.json'), dict(sample_id=sid, rows=rows, raw_rows=raw,
            counts=counts, image_sha256=record['image_sha256'],
            ground_truth_used_for_prediction=False, feature_sha256=sha(feature_path(root, sid))))
    result = aggregate(values)
    seal(directory/'metrics.json', dict(counts=result, deepsolo_images=0,
        total_images=len(ids), deepsolo_percent=0.,
        gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved()))
    return result


def train(plan, root):
    c = plan['config']
    head = new_head(c)
    t = c['training']
    ids = plan['tt_ids']+plan['partitions']['fit']
    dataset = CachedWords(plan, root, ids)
    optimizer = torch.optim.AdamW(head.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    history = []
    best = -1.
    epochs = 1 if plan['smoke'] else t['epochs']
    start = time.monotonic()
    ocr = recognizer(plan)
    out = root/'head'
    if (out/'complete.json').exists():
        print('TRAIN already complete', flush=True)
        ocr.release()
        return
    if (out/'last.pt').exists():
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        if state['plan_sha256'] != sha(root/'resolved_configuration.json'):
            raise ValueError('Changed resume plan')
        head.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        history, best = state['history'], state['best']
    try:
        for epoch in range(len(history)+1, epochs+1):
            head.train()
            loader = DataLoader(dataset, batch_size=t['batch_size'], shuffle=True,
                generator=torch.Generator().manual_seed(c['seed']+epoch),
                num_workers=t['num_workers'], collate_fn=collate, pin_memory=True)
            total, seen, steps = 0., [], 0
            torch.cuda.reset_peak_memory_stats()
            for features, target, sids in loader:
                optimizer.zero_grad(set_to_none=True)
                output = head(features.to(c['device']))
                # Same geometrical objective, explicit domain weighting per image.
                losses = []
                for i, sid in enumerate(sids):
                    loss, _ = dense_instance_loss({k: v[i:i+1] for k, v in output.items()}, [target[i]], c['loss'])
                    weight = t['icdar_loss_weight'] if sid.startswith('icdar13:') else 1.
                    losses.append(loss*weight)
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), t['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(sids)
                seen.extend(sids)
                steps += 1
            if sorted(seen) != sorted(ids):
                raise ValueError('Incomplete fitting epoch')
            row = dict(epoch=epoch, loss=total/len(seen), images=len(seen), steps=steps,
                order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
                gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved())
            if epoch % c['validation_interval'] == 0 or epoch == epochs or epoch == 1:
                result = evaluate(plan, root, head, plan['partitions']['selection'],
                    out/'selection'/('epoch_'+str(epoch).zfill(3)), c['inference'], ocr)
                row['selection'] = result
                if result['f1'] > best:
                    best = result['f1']
                    save_checkpoint(out/'best.pt', dict(model_state_dict=head.state_dict(),
                        epoch=epoch, selection=result, plan_sha256=sha(root/'resolved_configuration.json')))
            history.append(row)
            save_checkpoint(out/'last.pt', dict(model_state_dict=head.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), history=history, best=best,
                plan_sha256=sha(root/'resolved_configuration.json')))
            progress_path = out/'epochs.json'
            temporary = progress_path.with_suffix('.json.partial')
            temporary.write_text(json.dumps(history, indent=2, sort_keys=True, allow_nan=False)+'\n')
            temporary.replace(progress_path)
            print('EPOCH', epoch, 'loss', row['loss'], 'selection', row.get('selection'),
                  'best', best, 'seconds', time.monotonic()-start, flush=True)
        seal(out/'complete.json', dict(status='COMPLETE', epochs=epochs,
            fit_ids=ids, selection_ids=plan['partitions']['selection'],
            best_selection_f1=best, checkpoint_sha256=sha(out/'best.pt'),
            actual_new_deepsolo_inferences=0, total_fit_images=len(ids)))
    finally:
        ocr.release()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'cache', 'train'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.run_root
    if args.stage == 'prepare':
        root = prepare(args)
        return
    if root is None:
        raise ValueError('An explicit prepared --run-root is required')
    plan = read(root/'resolved_configuration.json')
    runtime(plan['config'])
    try:
        {'cache': cache, 'train': train}[args.stage](plan, root)
    except BaseException:
        failure_path = root/('failure_'+args.stage+'_'+str(time.time_ns())+'.json')
        seal(failure_path, dict(status='FAILED', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
