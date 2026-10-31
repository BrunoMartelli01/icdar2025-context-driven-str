"""Read real cached DeepSolo boxes with the already qualified PARSeq model."""

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch

from localize.independent_word_slots import (
    detector_boundary_in_original, boundary_bbox, EmptyDetectorBoundaryError,
)
from recognition.fallback_reread import padded_box
from recognition.fold_safe_teacher import checked_response
from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_ARCHIVE_SOURCE
from scripts.crop_clip_runtime import array_sha
from scripts.icdar_budget_data import read, sha, key
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def run(config_path, smoke):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(config_path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for p, digest in c['pinned_inputs'].items():
        if sha(p) != digest:
            raise ValueError('Changed rereading input: '+p)
    source = read(c['original_plan']); protocol = read(c['raw_protocol'])
    split = c.get('split', 'train')
    if split not in ('train', 'test'):
        raise ValueError('Explicit supported rereading split required')
    records = {r['sample_id']: r for r in source['records']['icdar13'][split]}
    ids = sorted(records)
    if split == 'test':
        if c['decision_path'] not in c['pinned_inputs']:
            raise ValueError('TEST rereading requires pinned pre-fallback decisions')
        decision = read(c['decision_path'])
        if (decision['ground_truth_used'] or decision['fallback_outputs_loaded'] or
                decision['deepsolo_images'] != c['fallback_images'] or
                decision['total_images'] != len(records)):
            raise ValueError('Annotation-free exact-budget decisions required before rereading')
        ids = decision['fallback_ids']
        if len(ids) != c['fallback_images'] or len(set(ids)) != len(ids) or not set(ids) <= set(records):
            raise ValueError('Only frozen selected TEST fallback images may be reread')
    if smoke:
        ids = ids[:c['smoke_images']]
    root = Path(c['output_root'])/('smoke' if smoke else split)
    root.mkdir(parents=True, exist_ok=False)
    code = {}
    for package in ('scripts', 'recognition', 'localize', 'segmentation', 'models',
                    'pipeline', 'caption', 'scoring', 'research'):
        for path in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
            code[str(path.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    torch.set_num_threads(c['cpu_threads']); torch.manual_seed(c['seed'])
    np.random.seed(c['seed']); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if not torch.cuda.is_available():
        raise RuntimeError('Requested real CUDA qualification requires a visible GPU')
    free, total = torch.cuda.mem_get_info()
    if free < c['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free GPU memory for the declared PARSeq run')
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction', config=c, config_sha256=sha(config_path),
        source_sha256=code, smoke=smoke, record_ids=ids, test_predictions_generated=split == 'test',
        ground_truth_used=False, result_class='smoke-test result' if smoke else split.upper()+' inference preparation',
        hardware=dict(gpu=torch.cuda.get_device_name(c['device']),
            free_bytes_before_loading=free, total_bytes=total, precision='float32',
            device=c['device'], model_set='PARSeq only; DeepSolo outputs cached',
            max_input_shape=[c['batch_size'], 3, 32, 128]),
        approximate_choices=c['scientific_choices']))
    model = ParseqApproxRecognizer(ParseqApproxConfig(
        runtime_source=Path(c['parseq_source']), checkpoint=Path(c['parseq_checkpoint']),
        runtime_source_archive=Path(c['parseq_archive']), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        device=c['device']))
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic(); crop_count = 0
    pins = {}; index = {}; peaks = []
    try:
        for number, sid in enumerate(ids):
            record = records[sid]
            if sha(record['image_path']) != record['image_sha256']:
                raise ValueError('Changed RGB input')
            path = Path(c['raw_root'])/(key(sid)+'.json')
            envelope = read(path)
            response = checked_response(envelope, protocol['binding'], sid, record['image_sha256'])
            pins[str(path)] = sha(path)
            with Image.open(record['image_path']) as image:
                rgb = np.ascontiguousarray(image.convert('RGB'))
            items = []; rows = []
            for detection in response['detections']:
                if detection['score'] < c['score_threshold']:
                    continue
                points = detector_boundary_in_original(detection['geometry']['boundary_points'],
                    tuple(record['image_wh']), record['orientation'])
                row = dict(source_index=detection['source_index'], ctc_text=detection['text'],
                    score=detection['score'], boundary_original=points, rereadings={})
                try:
                    box = boundary_bbox(points, tuple(record['image_wh']))
                    row['bbox'] = box
                except EmptyDetectorBoundaryError as exc:
                    row.update(bbox=None, invalid_envelope=str(exc))
                    rows.append(row)
                    continue
                rows.append(row)
                for padding in c['paddings']:
                    crop_box = padded_box(box, record['image_wh'], padding)
                    a, b, d, e = crop_box
                    crop = np.ascontiguousarray(rgb[b:e, a:d])
                    items.append((row, padding, crop_box, crop))
            for offset in range(0, len(items), c['batch_size']):
                group = items[offset:offset+c['batch_size']]
                outputs = model.recognize_candidate_batch([g[3] for g in group])
                if len(outputs) != len(group):
                    raise RuntimeError('Incomplete actual PARSeq fallback reading')
                for (row, padding, box, crop), output in zip(group, outputs):
                    row['rereadings'][str(padding)] = dict(output.to_dict(), crop_bbox=box,
                        crop_rgb_sha256=array_sha(crop), crop_shape=list(crop.shape))
                crop_count += len(group)
            target = root/'samples'/(key(sid)+'.json')
            seal(target, dict(sample_id=sid, image_sha256=record['image_sha256'], rows=rows,
                fallback_sha256=sha(path), ground_truth_used=False,
                model_checkpoint_sha256=c['pinned_inputs'][c['parseq_checkpoint']]))
            index[sid] = dict(path=str(target), sha256=sha(target))
            peaks.append(dict(sample_id=sid, allocated=torch.cuda.max_memory_allocated(),
                              reserved=torch.cuda.max_memory_reserved()))
            if number % 50 == 0:
                print(json.dumps(dict(stage='reread', images=number+1, total=len(ids), crops=crop_count)), flush=True)
    except BaseException as exc:
        seal(root/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))
        raise
    finally:
        model.release(); torch.cuda.empty_cache()
    seal(root/'inputs.json', pins); seal(root/'index.json', index)
    seal(root/'completion.json', dict(status='COMPLETE', mode='approximate',
        claim_label='approximate reproduction', smoke=smoke, images=len(ids), crops=crop_count,
        actual_parseq_crop_inferences=crop_count, actual_new_deepsolo_inferences=0,
        scope='Offline '+split.upper()+' fallback rereading; logical DeepSolo routing, no online latency result',
        split=split, ground_truth_used=False,
        seconds=time.monotonic()-started, peaks=peaks, out_of_memory=False))
    print(json.dumps(dict(status='COMPLETE', images=len(ids), crops=crop_count, root=str(root))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args(); run(a.config, a.smoke)
