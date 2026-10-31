"""Choose an adapted word detector checkpoint/crop policy on TRAIN validation."""

import argparse
import gc
import importlib.metadata
import itertools
from pathlib import Path
import shutil
import traceback

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from recognition.adapted_word_policy import emit
from recognition.common_budget_pipeline import counts, micro
from recognition.fallback_reread import padded_box
from recognition.group_word_selection import reading_confidence
from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from scripts.crop_clip_runtime import array_sha
from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_lightweight_word_selection import dictionary
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def run(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(config_path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for name, digest in c['pinned_inputs'].items():
        if sha(name) != digest:
            raise ValueError('Changed expert selection input: '+name)
    training = read(Path(c['training_root'])/'completion.json')
    if training['status'] != 'COMPLETE':
        raise ValueError('Both training arms must complete before selection')
    if set(training['arms']) != set(c['expected_arms']):
        raise ValueError('Incomplete detector training arms')
    for value in training['arms'].values():
        if not value['initialization']['strict'] or len(value['epochs']) != c['expected_epochs']:
            raise ValueError('Strict full-duration detector training required')
    trained_environment = read(Path(c['training_root'])/'resolved_configuration.json')['packages']
    if any(importlib.metadata.version(p) != version for p, version in trained_environment.items()):
        raise ValueError('Training/selection package versions differ')
    a = read(Path(c['admission_root'])/'resolved_configuration.json')
    data = read(Path(c['admission_root'])/'train_data.json')
    ids = a['partitions']['validation']; records = a['records']
    lexicon = dictionary(a, data, a['partitions']['fit'])
    candidates = dict(source=dict(path=c['source_checkpoint'], epoch=0, arm='source'))
    for arm, item in training['arms'].items():
        for relative, digest in item['checkpoint_sha256'].items():
            path = Path(c['training_root'])/relative
            if sha(path) != digest:
                raise ValueError('Changed trained candidate')
            epoch = int(path.stem.split('_')[-1])
            candidates[f'{arm}_{epoch:03d}'] = dict(path=str(path), epoch=epoch, arm=arm)
    for candidate in candidates.values():
        candidate['sha256'] = sha(candidate['path'])
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    code = {}
    for package in ('recognition', 'scripts', 'localize', 'models', 'segmentation',
                    'pipeline', 'scoring', 'caption', 'research'):
        for file in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/file.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, dest)
            code[str(file.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    torch.set_num_threads(c['cpu_threads']); torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    if torch.cuda.mem_get_info()[0] < c['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free CUDA memory')
    seal(root/'resolved_configuration.json', dict(config=c, config_sha256=sha(config_path),
        mode='approximate', claim_label='approximate reproduction', candidates=candidates,
        source_sha256=code, validation_ids=ids, test_used_for_selection=False,
        training_completion_sha256=sha(Path(c['training_root'])/'completion.json'),
        packages=trained_environment,
        hardware=dict(gpu=torch.cuda.get_device_name(0), precision='float32', device=c['device'],
            model_set='Sequential single YOLO candidate then PARSeq only', detector_batch_size=1,
            max_parseq_input_shape=[c['batch_size'], 3, 32, 128])))
    detector = reader = None; stage = 'detection'; memory = {}; raw = {}
    try:
        for name, candidate in candidates.items():
            torch.cuda.reset_peak_memory_stats(); detector = YOLO(candidate['path'])
            for sid in ids:
                record = records[sid]
                if sha(record['image_path']) != record['image_sha256']:
                    raise ValueError('Changed validation RGB')
                with Image.open(record['image_path']) as im:
                    rgb = np.ascontiguousarray(im.convert('RGB'))
                settings = c['detector']; shapes = []
                hook = detector.model.register_forward_pre_hook(lambda m, x: shapes.append(list(x[0].shape)))
                try:
                    output = detector.predict(np.ascontiguousarray(rgb[:, :, ::-1]), imgsz=settings['image_size'],
                        conf=settings['confidence_floor'], iou=settings['nms_iou'],
                        max_det=settings['max_detections'], device=c['device'], half=False, verbose=False)[0]
                finally:
                    hook.remove()
                slots, rejected = boxes_to_slots(output.boxes.xyxy.cpu().numpy(),
                    output.boxes.conf.cpu().numpy(), output.boxes.cls.cpu().numpy(),
                    tuple(record['image_wh']), settings['confidence_floor'])
                seal(root/'detector'/name/(key(sid)+'.json'), dict(slots=slots, rejected=rejected,
                    image_sha256=record['image_sha256'], model_input_shapes=shapes, ground_truth_used=False))
            memory[name] = dict(peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
            del detector; detector = None; gc.collect(); torch.cuda.empty_cache()
            print('DETECTION COMPLETE', name, flush=True)
        stage = 'parseq'; torch.cuda.reset_peak_memory_stats()
        reader = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['parseq_source']),
            checkpoint=Path(c['parseq_checkpoint']), runtime_source_archive=Path(c['parseq_archive']),
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE, device=c['device']))
        for name in candidates:
            raw[name] = {}
            for padding in c['paddings']:
                raw[name][padding] = {}
                for sid in ids:
                    record = records[sid]
                    slots = read(root/'detector'/name/(key(sid)+'.json'))['slots']
                    with Image.open(record['image_path']) as im:
                        rgb = np.ascontiguousarray(im.convert('RGB'))
                    items = []
                    for slot in slots:
                        box = padded_box(slot['bbox'], record['image_wh'], padding)
                        x0, y0, x1, y1 = box
                        items.append((slot, box, np.ascontiguousarray(rgb[y0:y1, x0:x1])))
                    rows = []
                    for offset in range(0, len(items), c['batch_size']):
                        group = items[offset:offset+c['batch_size']]
                        outputs = reader.recognize_candidate_batch([v[2] for v in group])
                        if len(outputs) != len(group):
                            raise RuntimeError('Incomplete actual PARSeq validation response')
                        for (slot, box, crop), output in zip(group, outputs):
                            value = output.to_dict()
                            rows.append(dict(value, bbox=box, detector_score=slot['score'],
                                confidence=reading_confidence(value)['confidence_balanced'],
                                candidate_id=f"{sid}:{slot['source_index']}", crop_rgb_sha256=array_sha(crop)))
                    raw[name][padding][sid] = rows
                    seal(root/'raw'/name/str(padding)/(key(sid)+'.json'), dict(rows=rows,
                        image_sha256=record['image_sha256'], ground_truth_used=False))
            print('PARSEQ COMPLETE', name, flush=True)
        memory['parseq'] = dict(peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
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
    trials = []
    for name, candidate in candidates.items():
        for padding, seq, joint, lexical in itertools.product(c['paddings'],
                c['sequence_thresholds'], c['joint_thresholds'], c['lexical_policies']):
            policy = dict(sequence_threshold=seq, joint_threshold=joint,
                lexical=lexical, same_text_nms_iou=c['same_text_nms_iou'], require_alphanumeric=True)
            values = []; changes = 0
            for sid in ids:
                rows, edits = emit(raw[name][padding][sid], lexicon, policy); changes += edits
                gt, valid = annotation(records[sid])
                flags = [r['ignored'] for r in predictions(rows, records[sid], valid)]
                values.append(counts(rows, gt, flags))
            trials.append(dict(candidate=name, checkpoint=candidate['path'], checkpoint_sha256=candidate['sha256'],
                epoch=candidate['epoch'], arm=candidate['arm'], padding=padding, policy=policy,
                counts=micro(values), changed_words=changes, deepsolo_images=0,
                total_images=len(ids), deepsolo_percentage=0.))
    winner = max(enumerate(trials), key=lambda pair: (pair[1]['counts']['f1'],
        -pair[1]['epoch'], -pair[1]['changed_words'], -pair[1]['padding'], -pair[0]))[1]
    seal(root/'trials.json', trials); seal(root/'memory.json', dict(stages=memory, out_of_memory=False))
    seal(root/'frozen.json', dict(status='FROZEN', winner=winner, test_used_for_selection=False,
        validation_ids=ids, note='Validation expert choice only; freeze the downstream gate before complete TEST pipeline evaluation'))
    print(__import__('json').dumps(dict(status='FROZEN', winner=winner)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.config)
