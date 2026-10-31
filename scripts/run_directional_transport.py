"""APPROXIMATE parameter-matched decoder study with image-only native PARSeq.

CAL100 selects checkpoints and explicit postprocessing; TEST300 is evaluated
only after all selections are sealed. Historical TEST reuse is disclosed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.directional_transport_unet import DirectionalTransportUNet
from recognition.directional_transport_metrics import aggregate, counts
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.evaluate_cached_totaltext_padding import crop_hash
from segmentation.data import SegmentationDataset, file_sha256, load_segmentation_manifest
from segmentation.directional_transport import objective, transform
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import WordKernelDataset, expand_word_kernels
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]
ARMS = ('unet_continuation', 'global_detail', 'local_transport')


def baseline_arm(arm):
    return arm in ('unet_continuation', 'unet_reference')


def arms_for(c):
    if c['experiment_kind'] == 'directional_transport_continuation_v4':
        return ('unet_reference', 'global_detail')
    return ('unet_reference', 'global_detail', 'local_transport') if c['experiment_kind'] == 'directional_transport_frozen_v2' else ARMS


def continuation_contract(c, parent):
    """A continuation preserves optimizer, targets, data, architecture and seed."""
    for key in ('architecture', 'augmentation', 'targets', 'loss', 'seed', 'postprocessing', 'excluded_fit_ids'):
        if c[key] != parent[key]:
            raise ValueError('Continuation changes scientific invariant: ' + key)
    for key in ('batch_size', 'source_learning_rate', 'new_learning_rate', 'weight_decay',
                'adam_betas', 'adam_epsilon', 'gradient_clip', 'freeze_source'):
        if c['training'][key] != parent['training'][key]:
            raise ValueError('Continuation changes optimizer/training invariant: ' + key)
    for key in ('train_manifest', 'test_manifest', 'source_checkpoint', 'parseq_checkpoint', 'parseq_archive', 'split_pickle'):
        if c['input_sha256'][key] != parent['input_sha256'][key]:
            raise ValueError('Continuation changes input invariant: ' + key)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def qualified_path(path, historical_root, qualified_root):
    """Accept the two documented source roots, never guess a third location."""
    path, historical_root, qualified_root = map(Path, (path, historical_root, qualified_root))
    if qualified_root in path.parents:
        return str(path)
    return str(qualified_root / path.relative_to(historical_root))


def model_for(c, arm):
    a = c['architecture']
    if baseline_arm(arm):
        model = AdaSpotGapAGUNet('control', skip_floor=a['skip_floor'], initial_gap_bias=a['initial_gap_bias'])
        if c['training'].get('freeze_source', False):
            model.requires_grad_(False)
        return model
    model = DirectionalTransportUNet(skip_floor=a['skip_floor'], initial_gap_bias=a['initial_gap_bias'],
        widths=a['widths'], radii=a['radii'], detail_width=a['detail_width'], spatial=arm == 'local_transport')
    if c['training'].get('freeze_source', False):
        model.freeze_source()
    return model


def memory():
    torch.cuda.synchronize()
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())


def setup(c):
    r = c['runtime']
    if r['precision'] != 'float32' or r['device'] != 'cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('Explicit qualified CUDA FP32 runtime required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != r['cublas_workspace_config']:
        raise RuntimeError('Set configured CUBLAS_WORKSPACE_CONFIG before launch')
    torch.set_num_threads(r['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_everything(c['seed'], deterministic_algorithms=True)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < r['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free GPU memory; existing jobs are preserved')
    return dict(gpu=torch.cuda.get_device_name(0), total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__, numpy=np.__version__, deterministic_algorithms=True, tf32=False,
        compute_processes=subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout, **r)


def prepare(args):
    path = args.config.resolve()
    if args.mode != 'approximate' or REPO / 'configs/approximate' not in path.parents:
        raise ValueError('Isolated explicit APPROXIMATE configuration required')
    c = read(path)
    if c['reproduction_mode'] != args.mode or c['experiment_kind'] not in ('directional_transport_v1', 'directional_transport_frozen_v2', 'directional_transport_continuation_v4'):
        raise ValueError('Configuration kind/mode mismatch')
    if c['experiment_kind'] in ('directional_transport_frozen_v2', 'directional_transport_continuation_v4') and (
            not c['training']['freeze_source'] or c['training']['baseline_epochs'] != 0):
        raise ValueError('Frozen v2 requires preserved source and an untrained reference')
    for key, expected in c['input_sha256'].items():
        if file_sha256(Path(c['paths'][key])) != expected:
            raise ValueError('Input checksum mismatch: ' + key)
    if c['experiment_kind'] == 'directional_transport_continuation_v4':
        continuation_contract(c, read(c['paths']['parent_protocol'])['resolved_config'])
    a, tr, pp = c['architecture'], c['training'], c['postprocessing']
    if (len(a['widths']) != 4 or tr['epochs'] < 1 or tr['batch_size'] < 1
            or tr['evaluation_epochs'][0] != 0 or tr['evaluation_epochs'][-1] != tr['epochs']
            or sorted(set(tr['evaluation_epochs'])) != tr['evaluation_epochs']
            or tr['source_learning_rate'] <= 0 or tr['new_learning_rate'] <= 0
            or not 1 <= a['maximum_parameter_ratio'] <= 1.1
            or any(not 0 < t < 1 for t in pp['text_thresholds'])
            or any(not 0 <= t <= 1 for t in pp['confidence_thresholds'])):
        raise ValueError('Invalid predeclared architecture/training/selection budget')
    out = args.output.resolve()
    if out == REPO or REPO in out.parents:
        raise ValueError('Scientific outputs must be outside the repository')
    out.mkdir(parents=True, exist_ok=True)
    raw = pickle.load(Path(c['paths']['split_pickle']).open('rb'))
    old = Path(c['paths']['historical_word_root'])
    qualified = Path(c['paths']['word_root'])
    index = {}
    for split, entries in raw.items():
        index[split] = []
        for record in entries:
            if split == 'fit' and record['sample_id'] in c['excluded_fit_ids']:
                continue
            row = {k: record[k] for k in ('sample_id', 'image_path', 'image_sha256', 'annotation_path',
                                         'annotation_sha256', 'image_hw', 'words')}
            if split != 'test':
                for key in ('image_path', 'annotation_path'):
                    row[key] = qualified_path(row[key], old, qualified)
            for key in ('image', 'annotation'):
                if file_sha256(Path(row[key + '_path'])) != row[key + '_sha256']:
                    raise ValueError('Changed scientific input: ' + row[key + '_path'])
            index[split].append(row)
    if [len(index[k]) for k in ('fit', 'calibration', 'evaluation', 'test')] != [1028, 100, 125, 300]:
        raise ValueError('Qualified populations changed')
    fit_hashes = {r['image_sha256'] for r in index['fit']}
    for split in ('calibration', 'evaluation', 'test'):
        if fit_hashes & {r['image_sha256'] for r in index[split]}:
            raise ValueError('Exact RGB duplicate in new training versus ' + split)
    if args.smoke:
        index = {k: v[:c['smoke'][k]] for k, v in index.items()}
    code = {str(p.relative_to(REPO)): file_sha256(p)
        for folder in ('models', 'segmentation', 'localize', 'recognition', 'scripts')
        for p in sorted((REPO / folder).rglob('*.py'))}
    identity = dict(config_sha256=file_sha256(path), code_sha256=code, smoke=args.smoke,
                    index=index, source_checkpoint=c['input_sha256']['source_checkpoint'])
    hardware = setup(c)
    if (out / 'protocol.json').exists():
        if read(out / 'protocol.json')['identity'] != identity:
            raise ValueError('Resume code/config/input contract drift')
    else:
        write(out / 'protocol.json', dict(identity=identity, resolved_config=c, hardware=hardware,
            result_class='smoke-test result' if args.smoke else 'approximate reproduction; exploratory TEST300',
            selection_uses_test=False, historical_test_exploration=True, source_training_saw_duplicates=True,
            metric='micro F1; alphanumeric Unicode casefold; unpadded bbox IoU>=0.5; maximum one-to-one; '
                   'all care words and all emitted predictions count; no ignore neutralization; empty cannot match',
            claim='New research extension; not a strict reproduction and not a global novelty priority claim'))
        write(out / 'resolved_config.json', c)
        write(out / 'splits.json', {k: [r['sample_id'] for r in v] for k, v in index.items()})
        for name in code:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        shutil.copyfile(path, out / 'source_config.json')
    return c, out, index


class OCRCache:
    """Exact RGB pixels, pinned PARSeq and native AR/refine1; resume-safe cache."""
    def __init__(self, c, out):
        self.c, self.path, self.model = c, out / 'ocr.jsonl', None
        self.rows, self.new_calls, self.reused = {}, 0, 0
        if self.path.exists():
            for line in self.path.open():
                row = json.loads(line)
                key = row['crop_array_sha256']
                if key in self.rows and row != self.rows[key]:
                    raise ValueError('Conflicting OCR cache entries')
                self.rows[key] = row

    def image(self, rgb, proposals):
        pending = {}
        for p in proposals:
            x0, y0, x1, y1 = p['padded_bbox_original']
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            if not crop.size:
                raise ValueError('Empty RGB crop; failed sample must not disappear')
            key = crop_hash(crop)
            p['crop_array_sha256'] = key
            if key not in self.rows:
                pending.setdefault(key, crop)
            else:
                self.reused += 1
        if pending and self.model is None:
            p = self.c['paths']
            self.model = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(p['parseq_source']),
                checkpoint=Path(p['parseq_checkpoint']), device=self.c['runtime']['device'],
                runtime_source_type=PARSEQ_ARCHIVE_SOURCE, runtime_source_archive=Path(p['parseq_archive'])))
        keys = list(pending)
        batch_size = self.c['runtime']['ocr_batch_size']
        for start in range(0, len(keys), batch_size):
            batch = keys[start:start + batch_size]
            values = self.model.recognize_candidate_batch([pending[k] for k in batch])
            if len(values) != len(batch):
                raise RuntimeError('OCR cardinality mismatch')
            with self.path.open('a') as stream:
                for key, value in zip(batch, values):
                    row = {**asdict(value), 'crop_array_sha256': key}
                    self.rows[key] = row
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    self.new_calls += 1
        for p in proposals:
            row = self.rows[p['crop_array_sha256']]
            p.update(text=row['raw_text'], confidence=row['sequence_confidence_native'])

    def release(self):
        if self.model is not None:
            self.model.release()
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()


def select_predictions(proposals, confidence):
    return [p for p in proposals if p['confidence'] >= confidence]


def finite_probabilities(logits):
    if logits.ndim != 4 or logits.shape[:2] != (1, 2) or logits.shape[-2] != logits.shape[-1]:
        raise ValueError('Expected single-image two-channel square segmentation logits')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite segmentation logits must not become empty predictions')
    return logits[0].sigmoid().cpu().numpy()


def validate(model, c, entries, directory, ocr, *, policy=None):
    """Image-only prediction first; annotations are accessed only for scoring."""
    if (directory / 'complete.json').exists():
        return read(directory / 'complete.json')
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    t = transform(c, native_size=256, train=False)
    pp = c['postprocessing']
    policies = [policy] if policy else [dict(text_threshold=tt, padding256=padding, confidence=confidence)
        for tt in pp['text_thresholds'] for padding in pp['padding256'] for confidence in pp['confidence_thresholds']]
    pools, per_policy = {}, [[] for _ in policies]
    started = time.monotonic()
    with torch.inference_mode():
        for i, entry in enumerate(entries):
            sid = entry['sample_id']
            with Image.open(entry['image_path']) as opened:
                image = opened.convert('RGB')
            rgb = np.ascontiguousarray(image)
            image_aug, mask, _ = t._augment(image, Image.new('L', image.size), None, 0)
            tensor, _ = t._finalize(image_aug, mask)
            probabilities = finite_probabilities(model(tensor[None].cuda()))
            native = probabilities.shape[-1]
            canvas = np.asarray(letterbox_canvas_mask(image.size, native_size=native), bool)
            unique = {}
            for q in policies:
                key = str((q['text_threshold'], q['padding256']))
                if key in unique:
                    continue
                labels, _ = expand_word_kernels((probabilities[0] >= q['text_threshold']) & canvas,
                                                (probabilities[1] >= pp['kernel_threshold']) & canvas)
                proposals = parseq_word_proposals(labels, image.size,
                    padding=int(q['padding256'] * native / 256),
                    min_area=pp['minimum_area256'] * (native / 256) ** 2,
                    max_instances=pp['maximum_instances'])
                for proposal in proposals:
                    proposal['padded_bbox_original'] = proposal['bbox']
                    proposal['bbox'] = proposal['bbox_original']
                ocr.image(rgb, proposals)
                unique[key] = proposals
            pools[sid] = unique
            for k, q in enumerate(policies):
                pred = select_predictions(unique[str((q['text_threshold'], q['padding256']))], q['confidence'])
                per_policy[k].append(dict(sample_id=sid, **counts(pred, entry['words'], c['metric_iou_threshold'])))
            stem = sid.replace(':', '__')
            np.savez_compressed(directory / (stem + '.npz'), probabilities=probabilities)
            write(directory / (stem + '.json'), dict(sample_id=sid, image_sha256=entry['image_sha256'],
                annotation_sha256=entry['annotation_sha256'], image_hw=entry['image_hw'], pools=unique))
            if (i + 1) % 25 == 0:
                print(f'[CAL/eval] {directory.name} {i+1}/{len(entries)}', flush=True)
    summaries = [dict(policy=q, aggregate=aggregate(rows)) for q, rows in zip(policies, per_policy)]
    # Earliest predeclared policy wins exact ties. TEST never participates.
    best = max(range(len(summaries)), key=lambda k: summaries[k]['aggregate']['f1'])
    for sid, unique in pools.items():
        chosen = policies[best]
        pools[sid] = select_predictions(unique[str((chosen['text_threshold'], chosen['padding256']))], chosen['confidence'])
    write(directory / 'predictions.json', pools)
    write(directory / 'per_sample.json', per_policy[best])
    write(directory / 'policies.json', summaries)
    result = dict(**summaries[best], seconds=time.monotonic()-started, memory=memory(),
                  ocr_unique_inferred_cumulative=ocr.new_calls, annotations_used_for_prediction=False)
    write(directory / 'complete.json', result)
    ocr.release()
    return result


def save(path, model, optimizer, epoch, history, best, protocol_sha):
    payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch, history=history, best=best, protocol_sha256=protocol_sha,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate())
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train_arm(c, out, index, arm, ocr, smoke):
    directory = out / arm
    directory.mkdir(exist_ok=True)
    if (directory / 'training_complete.json').exists():
        return read(directory / 'training_complete.json')
    seed_everything(c['seed'], deterministic_algorithms=True)
    model = model_for(c, arm)
    source = torch.load(c['paths']['source_checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
    if baseline_arm(arm):
        model.load_state_dict(source, strict=True)
        initialization = dict(strict_load=True, copied_tensors=len(source), new_tensors=[])
    else:
        initialization = model.initialize_source(source)
    del source
    params = dict(total=sum(p.numel() for p in model.parameters()),
                  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if params['total'] > c['architecture']['baseline_parameters'] * c['architecture']['maximum_parameter_ratio']:
        raise ValueError('Architecture exceeds the predeclared parameter budget')
    base, new = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (new if name.startswith(('transport.', 'rgb_detail.', 'detail_fusion.')) else base).append(parameter)
    tr = c['training']
    groups = [dict(params=base, lr=tr['source_learning_rate'])]
    if new:
        groups.append(dict(params=new, lr=tr['new_learning_rate']))
    optimizer = torch.optim.AdamW(groups, weight_decay=tr['weight_decay'], betas=tuple(tr['adam_betas']), eps=tr['adam_epsilon'])
    native = 256 if baseline_arm(arm) else 512
    paired = transform(c, native_size=native, train=True)
    manifest = load_segmentation_manifest(Path(c['paths']['train_manifest']), require_checksums=True)
    dataset = WordKernelDataset(SegmentationDataset(manifest, Path(c['paths']['word_root']), transform=paired),
                                distance_fraction=c['targets']['distance_fraction'])
    by = {s.sample_id: i for i, s in enumerate(manifest.samples)}
    ids = [by[r['sample_id']] for r in index['fit']]
    protocol_sha = file_sha256(out / 'protocol.json')
    last = directory / 'last.pt'
    start, history, best = 0, [], None
    if last.exists():
        payload = torch.load(last, map_location='cpu', weights_only=False)
        if payload['protocol_sha256'] != protocol_sha:
            raise ValueError('Checkpoint protocol drift')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        start, history, best = payload['epoch'], payload['history'], payload['best']
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
        np.random.set_state(payload['numpy_rng'])
        random.setstate(payload['python_rng'])
    elif c.get('continuation') and not baseline_arm(arm):
        payload = torch.load(c['paths']['parent_checkpoint'], map_location='cpu', weights_only=False)
        if (payload['protocol_sha256'] != c['input_sha256']['parent_protocol']
                or payload['epoch'] != c['continuation']['epoch']):
            raise ValueError('Parent optimizer/checkpoint epoch or protocol mismatch')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        start = payload['epoch']
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
        np.random.set_state(payload['numpy_rng'])
        random.setstate(payload['python_rng'])
        initialization.update(parent_checkpoint_sha256=c['input_sha256']['parent_checkpoint'],
            parent_protocol_sha256=c['input_sha256']['parent_protocol'], parent_epoch=start,
            optimizer_and_rng_restored=True)
    model.cuda()
    for group in optimizer.param_groups:
        for p in group['params']:
            for key, value in optimizer.state[p].items():
                if torch.is_tensor(value) and key != 'step':
                    optimizer.state[p][key] = value.cuda()
    torch.cuda.reset_peak_memory_stats()
    write(directory / 'architecture.json', dict(arm=arm, parameters=params, initialization=initialization,
        native_resolution=native, precision='float32', batch_size=tr['batch_size'], input_shape=[3,512,512],
        encoder_frozen=True, source_frozen=tr.get('freeze_source', False), architecture_identity=model.architecture_identity))
    epochs = (tr.get('baseline_epochs', tr['epochs']) if baseline_arm(arm) else tr['epochs'])
    if smoke:
        epochs = start + 1 if c.get('continuation') and not baseline_arm(arm) else min(epochs, 1)
    evaluation_epochs = [e for e in ((start, epochs) if smoke else tr['evaluation_epochs']) if e <= epochs]
    for epoch in range(start, epochs + 1):
        if epoch > start:
            paired.set_epoch(epoch)
            generator = torch.Generator().manual_seed(c['seed'] + epoch)
            loader = DataLoader(Subset(dataset, ids), batch_size=tr['batch_size'], shuffle=True,
                generator=generator, num_workers=c['runtime']['num_workers'], pin_memory=True,
                persistent_workers=False)
            model.train()
            totals, samples, tick = {}, 0, time.monotonic()
            for i, batch in enumerate(loader):
                batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                evidence = (model.forward_evidence(batch['image'], return_edges=True)
                            if not baseline_arm(arm) else model.forward_evidence(batch['image']))
                loss, terms = objective(evidence, batch, c['loss'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite transport training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    tr['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                n = len(batch['image'])
                samples += n
                for key, value in dict(loss=loss, **terms).items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) * n
                if (i + 1) % 50 == 0:
                    print(f'[{arm}] epoch{epoch} batch{i+1}/{len(loader)} loss={float(loss.detach()):.4f}', flush=True)
            optimizer.zero_grad(set_to_none=True)
            del evidence, batch, loss, terms
            history.append(dict(epoch=epoch, updates=len(loader), samples=samples,
                **{k: v / samples for k, v in totals.items()}, seconds=time.monotonic()-tick, memory=memory()))
            write(directory / 'history.json', history)
        if epoch in evaluation_epochs:
            result = validate(model, c, index['calibration'], directory / f'cal-epoch-{epoch:03d}', ocr)
            if (c.get('continuation') and not baseline_arm(arm) and not smoke
                    and epoch == c['continuation']['epoch']):
                parent_point = Path(c['paths']['parent_protocol']).parent / arm / f'cal-epoch-{epoch:03d}' / 'complete.json'
                expected = read(parent_point)
                if result['aggregate'] != expected['aggregate'] or result['policy'] != expected['policy']:
                    raise ValueError('Continuation initial CAL counts/policy do not reproduce the parent')
            if tr.get('retain_evaluation_checkpoints', False):
                torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, protocol_sha256=protocol_sha),
                           directory / f'epoch-{epoch:03d}.pt')
            if best is None or result['aggregate']['f1'] > best['aggregate']['f1']:
                best = dict(epoch=epoch, **result)
                torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, protocol_sha256=protocol_sha),
                           directory / 'best.pt')
                write(directory / 'selection.json', best)
            print(f'[{arm}] epoch{epoch} CAL word F1={result["aggregate"]["f1"]:.6f} best_epoch={best["epoch"]}', flush=True)
        save(last, model, optimizer, epoch, history, best, protocol_sha)
    result = dict(arm=arm, best=best, parameters=params, epochs=epochs,
                  initial_epoch=c.get('continuation', {}).get('epoch', 0) if not baseline_arm(arm) else 0, memory=memory(),
                  checkpoint_sha256=file_sha256(directory / 'best.pt'))
    write(directory / 'training_complete.json', result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def final(c, out, index, ocr, smoke):
    arms = arms_for(c)
    selections = {arm: read(out / arm / 'training_complete.json') for arm in arms}
    write(out / 'selection_sealed_before_test.json', selections)
    results = {}
    for arm in arms:
        model = model_for(c, arm)
        payload = torch.load(out / arm / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.cuda().eval()
        results[arm] = {}
        for split in ('evaluation', 'test'):
            results[arm][split] = validate(model, c, index[split], out / arm / split, ocr,
                                          policy=selections[arm]['best']['policy'])
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
    baseline = results[arms[0]]['test']['aggregate']['f1']
    for arm in arms:
        results[arm]['test']['delta_f1_pp'] = 100 * (results[arm]['test']['aggregate']['f1'] - baseline)
    primary = c.get('primary_arm', 'local_transport')
    report = dict(result_class='smoke-test result' if smoke else 'approximate reproduction; exploratory TEST300',
        protocol_sha256=file_sha256(out / 'protocol.json'), selections=selections, results=results,
        target_70_achieved=not smoke and results[primary]['test']['aggregate']['f1'] >= .70,
        user_objective_achieved=not smoke and results[primary]['test']['aggregate']['f1'] >= .70
            and results[primary]['test']['aggregate']['f1'] > baseline
            and selections[primary]['best']['epoch'] > 0,
        ocr_unique_cached=len(ocr.rows), deepsolo='0/{} (0%)'.format(len(index['test'])))
    write(out / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['approximate'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--train-only', action='store_true', help='Finish training/CAL selection; defer all TEST evaluation')
    args = parser.parse_args()
    try:
        c, out, index = prepare(args)
        ocr = OCRCache(c, out)
        for arm in arms_for(c):
            train_arm(c, out, index, arm, ocr, args.smoke)
        if args.train_only:
            write(out / 'training_phase_complete.json', dict(status='SUCCESS', reproduction_mode='approximate', test_evaluated=False))
        else:
            final(c, out, index, ocr, args.smoke)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / 'failure.json', dict(error=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), oom=isinstance(error, torch.cuda.OutOfMemoryError),
            memory=memory() if torch.cuda.is_available() else None))
        raise


if __name__ == '__main__':
    main()
