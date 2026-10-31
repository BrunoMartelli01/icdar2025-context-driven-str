"""Shared validation-only crop-policy selection and RGB-only expert exports."""

import argparse
import copy
import itertools
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from scripts.run_uniform_unet_training import read, write
from scripts.train_icdar13_word_refit import recognizer, read_rows
from scripts.run_icdar13_dense_instances import runtime
from scripts.icdar_budget_data import key, predictions
from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.icdar13_matched_training import emit_words
from segmentation.icdar2015_improvements import make_transform
from segmentation.uniform_unet_fusion import file_hash, annotation, word_counts, aggregate_counts
from recognition.unet_expert_records import decorate_unet_rows


def checked_training(root):
    plan = read(root/'resolved_configuration.json')
    complete = read(root/'complete.json')
    audit = read(root/'independent_training_audit.json')
    if (plan['smoke'] or complete['status'] != 'COMPLETE' or audit['status'] != 'PASS' or
            file_hash(root/'best.pt') != complete['checkpoint_sha256'] or
            audit['best_checkpoint_sha256'] != complete['checkpoint_sha256']):
        raise ValueError('A full independently audited U-Net is required')
    return plan, complete


def model(plan, root):
    c = plan['config']
    net = WordRefitUNet(partial_encoder=False, skip_floor=c['skip_floor'], initial_gap_bias=c['initial_gap_bias'])
    state = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    if state['plan_sha256'] != file_hash(root/'resolved_configuration.json'):
        raise ValueError('Changed selected U-Net state/plan')
    net.load_state_dict(state['model_state_dict'], strict=True)
    return net.to(c['device']).eval()


def rows_with_sources(raw, minimum_probability, image_wh):
    raw = decorate_unet_rows(raw, image_wh)
    raw = [dict(r, source_row=i) for i, r in enumerate(raw)]
    return raw, emit_words(raw, minimum_probability)


def select(root):
    plan, complete = checked_training(root)
    config = read(root.parent/'resolved_configuration.json')
    output = root/'emission_selection'
    if output.exists():
        raise ValueError('Preserve existing emission selection')
    output.mkdir()
    runtime(plan['config'])
    ocr = recognizer(plan)
    lookup = {r['sample_id']: r for r in plan['train_records']}
    ids = plan['partitions']['selection']
    grid = config['emission_grid']
    map_root = root/'selection'/f"epoch_{complete['selected_epoch']:03d}"/'maps'
    trials, total_crops = [], 0
    try:
        for text, kernel, padding in itertools.product(grid['text_thresholds'], grid['kernel_thresholds'], grid['padding']):
            geometry = f'text{text}_kernel{kernel}_pad{padding}'
            settings = dict(plan['config']['inference'], text_threshold=text, kernel_threshold=kernel,
                            padding=padding, minimum_transcript_probability=0.)
            all_rows = {}
            for sid in ids:
                r = lookup[sid]
                mp = map_root/(key(sid)+'.npz')
                with np.load(mp, allow_pickle=False) as archive:
                    probability = archive['probabilities']
                _, raw, expansion = read_rows(probability, r, settings, ocr, plan['config']['parseq_batch_size'])
                raw, _ = rows_with_sources(raw, 0., r['image_wh'])
                all_rows[sid] = raw
                total_crops += len(raw)
                write(output/geometry/(key(sid)+'.json'), dict(raw_rows=raw,
                    image_sha256=r['image_sha256'], maps_sha256=file_hash(mp),
                    ground_truth_used_for_prediction=False, expansion=expansion))
            for cut in grid['sequence_thresholds']:
                counts = []
                for sid in ids:
                    r = lookup[sid]
                    rows = emit_words(all_rows[sid], cut)
                    truth, validity = annotation(r)
                    ignored = [p['ignored'] for p in predictions(rows, r, validity)]
                    counts.append(word_counts(rows, truth, ignored))
                trials.append(dict(policy=dict(settings, minimum_transcript_probability=cut),
                    geometry=geometry, counts=aggregate_counts(counts), deepsolo_images=0,
                    total_images=len(ids), deepsolo_percentage=0.))
            write(output/'trials.json', trials)
        selected = max(enumerate(trials), key=lambda p: (p[1]['counts']['f1'], -p[0]))[1]
        write(output/'frozen.json', dict(status='FROZEN', mode='approximate', selected=selected,
            training_plan_sha256=file_hash(root/'resolved_configuration.json'),
            checkpoint_sha256=complete['checkpoint_sha256'], selection_ids=ids,
            test_used_for_selection=False, actual_parseq_crop_inferences=total_crops,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            source_sha256=file_hash(__file__)))
        print('FROZEN EMISSION', plan['dataset'], selected['policy'], selected['counts'], flush=True)
    except BaseException as exc:
        write(output/f'failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(),
              out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError)))
        raise
    finally:
        ocr.release()
        torch.cuda.empty_cache()


def export(root, split, frozen_router):
    plan, complete = checked_training(root)
    policy_path = root/'emission_selection/frozen.json'
    frozen = read(policy_path)
    if (frozen['status'] != 'FROZEN' or frozen['test_used_for_selection'] or
            frozen['checkpoint_sha256'] != complete['checkpoint_sha256']):
        raise ValueError('Frozen validation-only emission policy required')
    if split == 'test':
        if frozen_router is None:
            raise ValueError('TEST export requires an explicitly frozen trained router')
        gate = read(frozen_router)
        if gate['status'] != 'FROZEN' or gate['test_used_for_selection']:
            raise ValueError('Router must be frozen before TEST export')
        audit = read(frozen_router.parent/'independent_router_audit.json')
        if (audit['status'] != 'PASS' or audit['model_sha256'] != gate['model_sha256'] or
                audit['frozen_selection_sha256'] != file_hash(frozen_router)):
            raise ValueError('Independent router qualification is required before TEST export')
    ids = plan['partitions']['fit']+plan['partitions']['selection'] if split == 'train' else plan['partitions']['test']
    lookup = {r['sample_id']: r for r in plan['train_records']+plan['test_records']}
    output = root/('export_'+split)
    if output.exists():
        raise ValueError('Preserve existing expert export')
    output.mkdir()
    write(output/'contract.json', dict(mode='approximate', checkpoint_sha256=complete['checkpoint_sha256'],
        plan_sha256=file_hash(root/'resolved_configuration.json'), policy_sha256=file_hash(policy_path),
        frozen_router_sha256=file_hash(frozen_router) if frozen_router else None,
        sample_ids=ids, ground_truth_feature_use=False, source_sha256=file_hash(__file__)))
    config = copy.deepcopy(plan['config'])
    runtime(config)
    net, ocr = model(plan, root), recognizer(plan)
    transform = make_transform(config, config['image_hw'], augment=False)
    policy = dict(frozen['selected']['policy'], minimum_transcript_probability=0.)
    index, crops = {}, 0
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        for sid in ids:
            r = lookup[sid]
            if file_hash(r['image_path']) != r['image_sha256']:
                raise ValueError('Changed export RGB')
            # This path deliberately opens no masks, annotation or GT validity.
            with Image.open(r['image_path']) as source:
                rgb = source.convert('RGB')
                x, _ = transform(rgb, Image.new('L', rgb.size, 0), 0)
            with torch.inference_mode():
                probability = net(x[None].to(config['device']))[0].sigmoid().cpu().numpy()
            _, raw, expansion = read_rows(probability, r, policy, ocr, config['parseq_batch_size'])
            raw, rows = rows_with_sources(raw, frozen['selected']['policy']['minimum_transcript_probability'], r['image_wh'])
            mp = output/'maps'/(key(sid)+'.npz')
            mp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mp, probabilities=probability)
            sample = output/'samples'/(key(sid)+'.json')
            write(sample, dict(raw_rows=raw, rows=rows, maps_sha256=file_hash(mp), image_sha256=r['image_sha256'],
                image_wh=r['image_wh'], rgb_sha256=r['rgb_sha256'], ground_truth_used=False, expansion=expansion))
            index[sid] = dict(path=str(sample), sha256=file_hash(sample))
            crops += len(raw)
        torch.cuda.synchronize()
        write(output/'index.json', index)
        write(output/'complete.json', dict(status='COMPLETE', images=len(ids), actual_unet_image_inferences=len(ids),
            actual_parseq_crop_inferences=crops, actual_new_deepsolo_calls=0, gpu=torch.cuda.get_device_name(0),
            precision='float32', image_hw=config['image_hw'], unet_batch_size=1, parseq_batch=config['parseq_batch_size'],
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            seconds=time.monotonic()-started, ground_truth_used=False,
            scope='Frontend export, model loading excluded; not complete online pipeline latency'))
    except BaseException as exc:
        write(output/f'failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(),
              out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError)))
        raise
    finally:
        ocr.release()
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--stage', choices=['selection', 'train', 'test'], required=True)
    p.add_argument('--frozen-router', type=Path)
    args = p.parse_args()
    if args.stage == 'selection':
        select(args.run_root)
    else:
        export(args.run_root, args.stage, args.frozen_router)
