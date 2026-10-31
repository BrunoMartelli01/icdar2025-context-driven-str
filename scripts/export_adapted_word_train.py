"""GT-free expert outputs on TRAIN, or on TEST after gate qualification."""

import argparse
import gc
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from recognition.adapted_word_policy import emit
from recognition.fallback_reread import padded_box
from recognition.group_word_selection import reading_confidence
from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from scripts.crop_clip_runtime import array_sha
from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots
from scripts.icdar_budget_data import read, sha, key
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def run(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed expert export input: '+file)
    frozen = read(c['frozen_expert'])['winner']
    if frozen['policy']['lexical'] != 'none':
        raise ValueError('This export requires the selected vocabulary-free expert')
    completion = read(Path(c['training_root'])/'completion.json')
    if completion['status'] != 'COMPLETE' or completion['smoke']:
        raise ValueError('Completed full detector fitting required')
    checkpoint = Path(c['training_root'])/c['checkpoint_relative']
    expected = completion['arms']['head_only']['checkpoint_sha256'][c['checkpoint_relative']]
    if sha(checkpoint) != expected:
        raise ValueError('Changed selected or fold detector checkpoint')
    original = read(c['original_plan'])
    split = c.get('split', 'train')
    if split not in ('train', 'test'):
        raise ValueError('Explicit supported expert export split required')
    if split == 'test':
        router = Path(c['frozen_router'])
        for name in ('selection_frozen.json', 'deployment_qualification.json'):
            if str(router/name) not in c['pinned_inputs']:
                raise ValueError('TEST export requires pinned frozen gate and independent qualification')
        if (read(router/'selection_frozen.json')['test_used_for_selection'] or
                read(router/'deployment_qualification.json')['status'] != 'PASS'):
            raise ValueError('Qualified TRAIN-only gate required before TEST export')
    records = {r['sample_id']: r for r in original['records']['icdar13'][split]}
    train_records = {r['sample_id']: r for r in original['records']['icdar13']['train']}
    ids = c['prediction_ids']
    if len(ids) != len(set(ids)) or not set(ids) <= set(records):
        raise ValueError('Only explicitly declared distinct images in the selected split can be exported')
    training_manifest = read(Path(c['training_root'])/'dataset_manifest.json')
    if c['image_excluded']:
        if set(ids) & set(training_manifest['records']):
            raise ValueError('Heldout images used in fold training or checkpoint validation')
        seen_rgb = {train_records[s]['rgb_sha256'] for s in training_manifest['records']}
        if seen_rgb & {records[s]['rgb_sha256'] for s in ids}:
            raise ValueError('Heldout RGB duplicate used in fold fitting')
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    source = {}
    for package in ('scripts', 'recognition', 'localize', 'models', 'segmentation',
                    'pipeline', 'caption', 'scoring', 'research'):
        for file in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/file.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, dest)
            source[str(file.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    torch.set_num_threads(c['cpu_threads']); torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    if torch.cuda.mem_get_info()[0] < c['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free CUDA memory')
    seal(root/'resolved_configuration.json', dict(config=c, config_sha256=sha(path),
        mode='approximate', claim_label='approximate reproduction', source_sha256=source,
        checkpoint_sha256=expected, frozen_expert_sha256=sha(c['frozen_expert']),
        training_completion_sha256=sha(Path(c['training_root'])/'completion.json'),
        image_excluded=c['image_excluded'], split=split, test_used=split == 'test', ground_truth_used=False,
        hardware=dict(gpu=torch.cuda.get_device_name(0), precision='float32',
            model_set='YOLO then PARSeq, sequential', detector_batch_size=1,
            max_parseq_input_shape=[c['batch_size'], 3, 32, 128])))
    detector = reader = None; stage = 'detect'; memory = {}; timings = {}; generated = 0; manifest = {}
    started = time.monotonic()
    try:
        stage_started = time.monotonic(); torch.cuda.reset_peak_memory_stats(); detector = YOLO(str(checkpoint))
        for sid in ids:
            record = records[sid]
            if sha(record['image_path']) != record['image_sha256']:
                raise ValueError('Changed original RGB')
            with Image.open(record['image_path']) as im:
                rgb = np.ascontiguousarray(im.convert('RGB'))
            cfg = c['detector']
            output = detector.predict(np.ascontiguousarray(rgb[:, :, ::-1]),
                imgsz=cfg['image_size'], conf=cfg['confidence_floor'], iou=cfg['nms_iou'],
                max_det=cfg['max_detections'], device=c['device'], half=False, verbose=False)[0]
            slots, rejected = boxes_to_slots(output.boxes.xyxy.cpu().numpy(),
                output.boxes.conf.cpu().numpy(), output.boxes.cls.cpu().numpy(),
                tuple(record['image_wh']), cfg['confidence_floor'])
            seal(root/'detector'/(key(sid)+'.json'), dict(slots=slots, rejected=rejected,
                image_sha256=record['image_sha256'], ground_truth_used=False))
        torch.cuda.synchronize(); timings['detector_seconds'] = time.monotonic()-stage_started
        memory['detector'] = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        del detector; detector = None; gc.collect(); torch.cuda.empty_cache(); stage = 'parseq'
        stage_started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        reader = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['parseq_source']),
            checkpoint=Path(c['parseq_checkpoint']), runtime_source_archive=Path(c['parseq_archive']),
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE, device=c['device']))
        for sid in ids:
            record = records[sid]
            with Image.open(record['image_path']) as im:
                rgb = np.ascontiguousarray(im.convert('RGB'))
            slots = read(root/'detector'/(key(sid)+'.json'))['slots']; items = []; rows = []
            for slot in slots:
                box = padded_box(slot['bbox'], record['image_wh'], frozen['padding'])
                a, b, d, e = box
                items.append((slot, box, np.ascontiguousarray(rgb[b:e, a:d])))
            for offset in range(0, len(items), c['batch_size']):
                group = items[offset:offset+c['batch_size']]
                outputs = reader.recognize_candidate_batch([v[2] for v in group])
                if len(outputs) != len(group):
                    raise RuntimeError('Incomplete actual expert PARSeq export')
                for (slot, box, crop), value in zip(group, outputs):
                    raw = value.to_dict()
                    rows.append(dict(raw, bbox=box, detector_score=slot['score'],
                        confidence=reading_confidence(raw)['confidence_balanced'],
                        candidate_id=f"{sid}:{slot['source_index']}", crop_rgb_sha256=array_sha(crop)))
                generated += len(group)
            selected, _ = emit(rows, None, frozen['policy'])
            target = root/'samples'/(key(sid)+'.json')
            seal(target, dict(rows=selected, raw_rows=rows, ground_truth_used=False,
                image_sha256=record['image_sha256'], checkpoint_sha256=expected,
                frozen_expert_sha256=sha(c['frozen_expert']), image_excluded=c['image_excluded']))
            manifest[sid] = dict(path=str(target), sha256=sha(target))
        torch.cuda.synchronize(); timings['parseq_seconds'] = time.monotonic()-stage_started
        memory['parseq'] = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
    except BaseException as exc:
        seal(root/'failure.json', dict(stage=stage, error=repr(exc), traceback=traceback.format_exc(),
            out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError),
            allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()))
        raise
    finally:
        if reader is not None:
            reader.release()
        if detector is not None:
            del detector
        torch.cuda.empty_cache()
    seal(root/'index.json', manifest)
    seal(root/'completion.json', dict(status='COMPLETE', mode='approximate',
        claim_label='approximate reproduction', images=len(ids), actual_parseq_crop_inferences=generated,
        actual_new_deepsolo_inferences=0, memory=memory, out_of_memory=False,
        split=split, test_used=split == 'test', ground_truth_used=False, image_excluded=c['image_excluded'],
        timings=dict(timings, total_seconds=time.monotonic()-started,
            scope='Sequential offline stages include I/O and model loading; exclude DeepSolo and are not end-to-end online latency')))
    print('EXPORTED', root, len(ids), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.config)
