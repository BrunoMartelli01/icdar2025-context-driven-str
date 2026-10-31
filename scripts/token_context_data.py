"""Checksum-bound full-TRAIN inputs and frozen YOLO/PARSeq/DeepSolo branches."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.run_yolo_f1_quality_study import read, sha, seal
from scripts.crop_clip_runtime import array_sha, _cuda_peak


def key(sid):
    return sid.replace(':', '__')


def rgb(sample):
    if sha(sample['image_path']) != sample['image_sha256']:
        raise ValueError('Image file checksum changed: '+sample['sample_id'])
    with Image.open(sample['image_path']) as im:
        result = np.asarray(im.convert('RGB'))
    if list(result.shape[:2][::-1]) != sample['image_wh'] or array_sha(result) != sample['rgb_sha256']:
        raise ValueError('Stored raster RGB identity/orientation changed')
    return result


def contract(root, sample):
    return {'plan_sha256': sha(root/'resolved_configuration.json'), 'sample_id': sample['sample_id'],
            'image_sha256': sample['image_sha256']}


def shard(root, split, stage, sample):
    path = root/split/stage/(key(sample['sample_id'])+'.json')
    if not path.exists():
        return path, None
    value = read(path)
    if value['contract'] != contract(root, sample):
        raise ValueError('Immutable stage cache identity changed: '+str(path))
    return path, value


def stage_complete(root, split, stage, samples, metadata):
    files = {key(s['sample_id'])+'.json': sha(root/split/stage/(key(s['sample_id'])+'.json')) for s in samples}
    seal(root/split/(stage+'_receipt.json'), {'plan_sha256': sha(root/'resolved_configuration.json'),
        'images': len(samples), 'files_sha256': files, **metadata})


def cached_stage(root, split, stage, samples):
    p = root/split/(stage+'_receipt.json')
    if not p.exists():
        return False
    v = read(p)
    if v['plan_sha256'] != sha(root/'resolved_configuration.json') or v['images'] != len(samples):
        raise ValueError('Stage receipt population changed')
    if any(sha(root/split/stage/file) != digest for file, digest in v['files_sha256'].items()):
        raise ValueError('Completed raw stage changed')
    return True


def detect_and_read(plan, root, split):
    from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_ARCHIVE_SOURCE
    from recognition.yolo_word_refinement import joint_score, nms
    from recognition.group_word_selection import normalize, reading_confidence
    from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots
    from scripts.contextual_lm_verifier_runtime import state_digest
    from ultralytics import YOLO
    ss = plan['samples'][split]; cfg = plan['config']; paths = plan['paths']; stage = 'light'
    if cached_stage(root, split, stage, ss):
        return
    start = time.monotonic(); detector = recognizer = None; reused = generated = 0; shapes = []
    torch.cuda.reset_peak_memory_stats(cfg['execution']['device'])
    try:
        for i, sample in enumerate(ss, 1):
            path, old = shard(root, split, stage, sample)
            if old is not None:
                continue
            sid = sample['sample_id']; source = sample.get('cached_light')
            if source:
                value = read(source)
                if sha(source) != sample['cached_light_sha256'] or value['image_sha256'] != sample['image_sha256'] or value['ground_truth_used']:
                    raise ValueError('Audited light source changed')
                rows = value['rows']; receipt = {'reused_source': source, 'source_sha256': sha(source)}; reused += 1
            else:
                image = rgb(sample)
                if detector is None:
                    detector = YOLO(paths['yolo_checkpoint'])
                    detector.model.register_forward_pre_hook(lambda _, args: shapes.append(list(args[0].shape)))
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths['parseq_source']),
                        checkpoint=Path(paths['parseq_checkpoint']), runtime_source_archive=Path(paths['parseq_archive']),
                        runtime_source_type=PARSEQ_ARCHIVE_SOURCE, device='cpu'))
                    recognizer.model.requires_grad_(False).eval(); before = state_digest(recognizer.model)
                inference = cfg['light_inference']; shapes.clear(); tick = time.monotonic()
                pred = detector.predict(np.ascontiguousarray(image[:, :, ::-1]), imgsz=inference['image_size'],
                    conf=inference['confidence_floor'], iou=inference['nms_iou'], max_det=inference['max_detections'],
                    device=cfg['execution']['device'], half=False, verbose=False)[0]
                torch.cuda.synchronize()
                slots, rejected = boxes_to_slots(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy(),
                    pred.boxes.cls.cpu().numpy(), tuple(sample['image_wh']), inference['confidence_floor'])
                detector_seconds = time.monotonic()-tick; raw = []
                for offset in range(0, len(slots), cfg['execution']['parseq_batch_size']):
                    group = slots[offset:offset+cfg['execution']['parseq_batch_size']]
                    crops = [np.ascontiguousarray(image[r['bbox'][1]:r['bbox'][3], r['bbox'][0]:r['bbox'][2]]) for r in group]
                    values = recognizer.recognize_candidate_batch(crops)
                    if len(values) != len(group):
                        raise RuntimeError('Incomplete genuine PARSeq response')
                    for r, crop, value in zip(group, crops, values):
                        cid = f'{sid}:yolo:{r["source_index"]}'
                        raw.append({'candidate_id': cid, 'instance_id': cid, 'bbox': r['bbox'], 'detector_score': r['score'],
                            'crop_rgb_sha256': array_sha(crop), 'crop_shape': list(crop.shape), **value.to_dict()})
                admitted = [r for r in raw if normalize(r['raw_text']) and joint_score(r) >= cfg['selection']['joint_threshold']]
                rows = [{**r, 'confidence': reading_confidence(r)['confidence_balanced']}
                        for r in nms(admitted, cfg['selection']['nms_iou'], same_text=True, joint=True)]
                receipt = {'new_inference': True, 'detector_seconds': detector_seconds, 'yolo_input_shapes': shapes.copy(),
                    'rejected_geometry': rejected, 'raw_candidates': len(raw), 'parseq_device': 'cpu',
                    'parseq_input_shape': [cfg['execution']['parseq_batch_size'], 3, 32, 128]}; generated += 1
                seal(root/split/'light_candidates'/(key(sid)+'.json'), {'contract': contract(root, sample),
                    'raw_candidates': raw, 'ground_truth_used': False})
            seal(path, {'contract': contract(root, sample), 'rows': rows, 'ground_truth_used': False, 'receipt': receipt})
            if i % cfg['execution']['log_every'] == 0 or i == len(ss):
                print(f'[light {split}] {i}/{len(ss)}', flush=True)
        model_info = {}
        if recognizer is not None:
            after = state_digest(recognizer.model)
            if after != before:
                raise RuntimeError('Frozen PARSeq state changed')
            model_info = {'parseq_state_before': before, 'parseq_state_after': after, 'optimizer_steps': 0}
        stage_complete(root, split, stage, ss, {'seconds': time.monotonic()-start, 'reused': reused, 'generated': generated,
            'model_set': ['YOLO640', 'qualified PARSeq'], 'precision': 'float32', 'yolo_batch_size': 1,
            'parseq_batch_size': cfg['execution']['parseq_batch_size'], 'cuda_peak': _cuda_peak(cfg['execution']['device']), **model_info})
    finally:
        if recognizer is not None:
            recognizer.release()
        detector = recognizer = None; gc.collect(); torch.cuda.empty_cache()


def fallback(plan, root, split):
    from deepsolo.adapter import DeepSoloSubprocessAdapter
    from scripts.run_deepsolo_textocr_comparison import validate_response
    cfg, ss = plan['config'], plan['samples'][split]
    if cached_stage(root, split, 'deepsolo', ss):
        return
    start = time.monotonic(); runtime = plan['deepsolo_runtime']; paths = plan['paths']
    seal(root/'runtime.json', runtime)
    adapter = DeepSoloSubprocessAdapter(Path(paths['deepsolo_python']), root/'runtime.json',
        host_source_root=Path(runtime['source_root']), repository_root=Path(__file__).resolve().parents[1])
    reused = generated = 0
    for i, sample in enumerate(ss, 1):
        path, old = shard(root, split, 'deepsolo', sample)
        if old is not None:
            continue
        source = sample.get('cached_deepsolo')
        if source:
            if sha(source) != sample['cached_deepsolo_sha256']:
                raise ValueError('Audited fallback source changed')
            value = read(source)
            if value['contract']['image_sha256'] != sample['image_sha256']:
                raise ValueError('Fallback source image differs')
            response = value['response']; reused += 1
        else:
            response = adapter.infer(image_path=Path(sample['image_path']), candidate_id=cfg['fallback']['candidate_id'],
                inference_options={'device': cfg['execution']['device'], 'score_threshold': cfg['fallback']['score_threshold'],
                                   'lexicon_type': None}, reduction_policy='reading_order_space_join').to_dict()
            generated += 1
        validate_response(response, cfg['fallback']['candidate_id'], cfg['assets']['deepsolo_checkpoint_sha256'],
            plan['deepsolo_config_sha256'], cfg['fallback']['score_threshold'])
        seal(path, {'contract': contract(root, sample), 'response': response,
            'ground_truth_used_in_inference': False, 'reused_source_sha256': sha(source) if source else None})
        if i % cfg['execution']['log_every'] == 0 or i == len(ss):
            print(f'[deepsolo {split}] {i}/{len(ss)} new={generated} reused={reused}', flush=True)
    stage_complete(root, split, 'deepsolo', ss, {'seconds': time.monotonic()-start, 'reused': reused, 'generated': generated,
        'precision': 'float32', 'batch_size': 1, 'memory_measurements': 'per-image isolated-worker response.memory',
        'native_preprocessing': 'qualified ViTAEv2-S shortest1024/max1892/divisor32'})


def annotations(sample):
    from recognition.scene_text_datasets import parse_totaltext_annotation
    from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
    if sha(sample['annotation_path']) != sample['annotation_sha256']:
        raise ValueError('GT annotation checksum changed')
    instances = parse_totaltext_annotation(Path(sample['annotation_path']))
    instances, _ = clip_instances_to_pixel_extent(instances, tuple(sample['image_wh']))
    result = []
    for instance in instances:
        if not instance.ignore:
            x1, y1, x2, y2 = instance.bbox_xyxy
            result.append({'bbox': [x1, y1, x2+1, y2+1], 'text': instance.transcription})
    return result


def evaluate_branches(plan, root, split):
    from recognition.clip_router_evaluation import native_mask, deepsolo_predictions
    from recognition.paper_image_router_evaluation import word_metrics
    records = {}; ss = plan['samples'][split]
    source = read(Path(plan['paths']['coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']
    for sample in ss:
        sid = sample['sample_id']; gt = annotations(sample)
        if sha(sample['validity_path']) != sample['validity_sha256']:
            raise ValueError('Evaluation-only validity mask changed')
        with Image.open(sample['validity_path']) as im:
            validity = native_mask(im.convert('L'), plan['config']['metric']['native_ignore_size'])
        light = read(root/split/'light'/(key(sid)+'.json'))['rows']; predictions = []
        for r in light:
            a, b, c, d = r['bbox']; mask = Image.new('L', tuple(sample['image_wh']), 0)
            ImageDraw.Draw(mask).rectangle([a, b, c-1, d-1], fill=255)
            footprint = native_mask(mask, validity.shape[0])
            predictions.append({'text': r['raw_text'], 'bbox': r['bbox'], 'ignored': bool(footprint.any() and not np.any(footprint & validity))})
        heavy = read(root/split/'deepsolo'/(key(sid)+'.json'))['response']
        ds = deepsolo_predictions(heavy, sample['image_wh'], sample['orientation'], validity, plan['config']['fallback']['score_threshold'])
        value = {'light': word_metrics(predictions, gt), 'deepsolo': word_metrics(ds, gt)}
        if sid in source.get(split, {}) and value != source[split][sid]:
            raise ValueError('Cached original branch counts fail exact paired replay: '+sid)
        records[sid] = value
    seal(root/split/'branch_metrics.json', {'plan_sha256': sha(root/'resolved_configuration.json'), 'branches': records,
        'metric': plan['config']['metric'], 'ground_truth_use': 'supervision/evaluation only'})
    return records


def arrays(root, split, sid):
    directory = root/split/'features'; meta = read(directory/(key(sid)+'.json')); path = directory/(key(sid)+'.npz')
    if meta['arrays_sha256'] != sha(path):
        raise ValueError('Feature tensor archive changed')
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}, meta


def write_arrays(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(data) or any(not np.array_equal(old[k], v) for k, v in data.items()):
                raise ValueError('Immutable tensor archive changed')
        return
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **data)
    temp.replace(path)
