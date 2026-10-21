"""Stream v1 epoch485 control padding crops directly to qualified PARSeq; save JSON only.

Original TRAIN-IoU selection is frozen; rerun segmentation on CPU for matched conditions. This
is approximate post-hoc TEST exploration, not a paper Table5 reproduction.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import resource
import time
import traceback

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_cached_totaltext_padding import (
    crop_hash, correct_word_ids, evaluation_rows, proposals_for_padding)
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt, _sample_key
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)


def seal(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'existing numeric artifact differs: {path}')
    else:
        write(path, value)


def read_crop(image, proposal):
    x1, y1, x2, y2 = proposal['padded_bbox_original']
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError('invalid or empty remapped crop')
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    if crop_hash(crop) != proposal['crop_array_sha256']:
        raise ValueError('in-memory RGB crop checksum mismatch')
    return crop


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan.get('reproduction_mode') != 'approximate' or plan.get('variant') != 'separated-v1-485-cpu-control-v1':
        raise ValueError('explicit approximate v1 epoch485 control streaming-padding config required')
    pads = plan['padding_native_pixels']
    if (not pads or len(set(pads)) != len(pads) or any(type(p) is not int or p < 0 for p in pads)
            or plan['baseline_padding_native_pixels'] not in pads):
        raise ValueError('invalid explicit padding grid')
    runtime = plan['runtime']
    if (runtime['device'] != 'cpu' or runtime['precision'] != 'float32'
            or runtime['segmentation_batch_size'] != 1 or runtime['parseq_batch_size'] < 1):
        raise ValueError('explicit CPU FP32 memory-bounded runtime required')
    return plan


def run(args):
    started = time.monotonic()
    plan = load_plan(args.config)
    output = args.output_dir.resolve()
    root, data_root = args.run_directory.resolve(), args.data_root.resolve()
    if output.is_relative_to(root) or output.is_relative_to(data_root):
        raise ValueError('output must be outside the original source run and dataset')
    if file_sha256(args.train_selection) != plan['train_selection_results_sha256']:
        raise ValueError('frozen TRAIN checkpoint selection changed')
    selection = json.loads(args.train_selection.read_text())
    if file_sha256(Path(selection['source_selection_path'])) != selection['source_selection_sha256']:
        raise ValueError('original TRAIN Word-F1 evidence changed')
    if file_sha256(Path(selection['source_run_manifest_path'])) != selection['source_run_manifest_sha256']:
        raise ValueError('complete source training manifest changed')
    chosen = [r for r in selection['confirm'] if r['epoch'] == plan['checkpoint_epoch']]
    if len(chosen) != 1 or chosen[0]['epoch'] != plan['checkpoint_epoch']:
        raise ValueError('checkpoint must match the declared original-TRAIN-IoU control')
    chosen = chosen[0]
    if chosen['checkpoint_sha256'] != plan['checkpoint_sha256']:
        raise ValueError('chosen checkpoint SHA256 differs from TRAIN selection')
    config_path = root / 'resolved_config.yaml'
    if file_sha256(config_path) != plan['segmentation_config_sha256']:
        raise ValueError('segmentation configuration changed')
    config = load_segmentation_config(config_path)
    if config.reproduction_mode != 'approximate' or config.transforms.normalization != 'imagenet':
        raise ValueError('qualified approximate ImageNet-normalized source required')
    records = [json.loads(row) for row in (root / 'epochs.jsonl').read_text().splitlines()]
    source_record = [r for r in records if r['epoch'] == chosen['epoch']]
    if len(source_record) != 1 or source_record[0]['checkpoint_sha256'] != chosen['checkpoint_sha256']:
        raise ValueError('checkpoint source provenance differs')
    checkpoint = root / source_record[0]['checkpoint']
    if checkpoint.is_symlink() or checkpoint.parent != root / 'checkpoints':
        raise ValueError('unsafe source checkpoint')
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if (manifest.dataset != 'total_text' or manifest.split_role != 'test'
            or len(manifest.samples) != plan['expected_images']
            or file_sha256(args.test_manifest) != plan['test_manifest_sha256']):
        raise ValueError('qualified complete official TEST manifest required')
    samples = list(manifest.samples)
    if args.limit_images is not None:
        if not 1 <= args.limit_images <= len(samples):
            raise ValueError('invalid explicit smoke-test limit')
        samples = samples[:args.limit_images]
    for sample in samples:
        path = data_root / sample.image_path
        if not path.is_file():
            raise FileNotFoundError(f'TEST data root does not resolve {sample.image_path}; set --data-root to the qualified TEST dataset root')
    if (file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError('qualified pinned PARSeq artifacts required')
    runtime = plan['runtime']
    torch.set_num_threads(runtime['segmentation_threads'])
    torch.set_num_interop_threads(1)
    random.seed(runtime['seed'])
    np.random.seed(runtime['seed'])
    torch.manual_seed(runtime['seed'])
    torch.use_deterministic_algorithms(True)
    code = [Path(__file__), Path('scripts/evaluate_cached_totaltext_padding.py'),
            Path('scripts/evaluate_totaltext_v1_findcontours_parseq.py'),
            Path('localize/repository_contours.py'), Path('localize/native_repository_contours.py'),
            Path('recognition/parseq.py'), Path('recognition/scene_text_v1_evaluation.py'),
            Path('recognition/scene_text_datasets.py'), Path('segmentation/evaluate.py'),
            Path('segmentation/data.py'), Path('models/unet_adv.py')]
    protocol = {
        'schema_version': 1, 'reproduction_mode': 'approximate',
        'result_class': plan['result_class'] if args.limit_images is None else 'smoke-test; approximate reproduction',
        'plan': plan, 'config_sha256': file_sha256(args.config), 'source_run': str(root),
        'train_selection_results_sha256': file_sha256(args.train_selection),
        'source_epochs_sha256': file_sha256(root / 'epochs.jsonl'),
        'source_config_sha256': file_sha256(config_path), 'source_resolved_config': OmegaConf.to_container(config, resolve=True),
        'checkpoint_path': str(checkpoint), 'checkpoint_sha256': chosen['checkpoint_sha256'],
        'checkpoint_epoch': chosen['epoch'], 'training_complete': selection['training_complete'],
        'test_manifest_sha256': file_sha256(args.test_manifest), 'data_root': str(data_root),
        'sample_ids': [s.sample_id for s in samples],
        'parseq_source': str(args.parseq_source.resolve()),
        'parseq_source_archive_sha256': PARSEQ_RUNTIME_ARCHIVE_SHA256,
        'parseq_checkpoint_sha256': PARSEQ_CHECKPOINT_SHA256,
        'parseq_decode_ar': True, 'parseq_refine_iters': 1,
        'code_sha256': {str(p.resolve()): file_sha256(p) for p in code},
        'runtime': {**runtime, 'deterministic_algorithms': True, 'torch_interop_threads': 1,
                    'segmentation_input_shape': [1, 3, 256, 256], 'native_mask_shape': [128, 128],
                    'parseq_input_shape': [runtime['parseq_batch_size'], 3, 32, 128],
                    'peak_cuda_allocated_bytes': None, 'peak_cuda_reserved_bytes': None, 'cuda_used': False,
                    'platform': platform.platform(), 'pytorch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__},
        'outputs': 'small numeric proposal/OCR/evaluation JSON and pixel hashes; no crops/masks/weights saved',
        'ground_truth_use': 'all localization proposals and OCR frozen before loading word annotations; evaluation only',
        'selection': 'original TRAIN-IoU winner485; full CPU segmentation control; exact-pixel CPU-PARSeq cache reuse; no TEST calibration',
        'matching': 'maximum one-to-one IoU>=0.5; casefold exact text; care words; punctuation retained; empty predictions included',
        'table_5_claim': False,
    }
    seal(output / 'protocol.json', protocol)
    protocol_sha = file_sha256(output / 'protocol.json')
    model = _model_from_config(config).eval()
    metadata = validate_checkpoint(checkpoint, model, chosen['checkpoint_sha256'], config.architecture.identity)
    if metadata['epoch'] != chosen['epoch']:
        raise ValueError('loaded checkpoint epoch differs')
    shards, image_paths = [], {}
    for index, sample in enumerate(samples, 1):
        image_path = data_root / sample.image_path
        if file_sha256(image_path) != sample.image_sha256:
            raise ValueError(f'RGB image checksum mismatch: {sample.sample_id}')
        with Image.open(image_path) as opened:
            rgb = opened.convert('RGB')
        image = np.ascontiguousarray(rgb)
        with torch.inference_mode():
            logits = model(_preprocess(rgb).unsqueeze(0))
        if tuple(logits.shape) != (1, 1, 128, 128) or not torch.isfinite(logits).all():
            raise RuntimeError('invalid real segmentation logits')
        native = (logits.sigmoid()[0, 0] >= plan['probability_threshold']).numpy().astype(np.uint8)
        variants = proposals_for_padding(native, image.shape[:2], plan)
        for variant in variants.values():
            for proposal in variant['proposals']:
                x1, y1, x2, y2 = proposal['padded_bbox_original']
                proposal['crop_array_sha256'] = crop_hash(image[y1:y2, x1:x2])
                proposal['crop_hw'] = [y2-y1, x2-x1]
        shard = {'sample_id': sample.sample_id, 'protocol_sha256': protocol_sha,
                 'image_sha256': sample.image_sha256, 'image_hw': list(image.shape[:2]),
                 'native_mask_array_sha256': sha256(native.tobytes()).hexdigest(), 'variants': variants}
        seal(output / 'proposals' / f'{_sample_key(sample.sample_id)}.json', shard)
        shards.append(shard)
        image_paths[sample.sample_id] = image_path
        if index % 25 == 0 or index == len(samples):
            print(f'[v1 epoch485 control segmentation] images={index}/{len(samples)} elapsed_s={time.monotonic()-started:.1f}', flush=True)
    del model, logits, rgb, image, native
    gc.collect()
    unique = {}
    for shard in shards:
        for variant in shard['variants'].values():
            for proposal in variant['proposals']:
                unique.setdefault(proposal['crop_array_sha256'], (shard['sample_id'], proposal))
    ocr_path, ocr = output / 'ocr.jsonl', {}
    reuse = Path(plan['reuse_ocr_run'])
    if (file_sha256(reuse/'protocol.json') != plan['reuse_ocr_protocol_sha256']
            or file_sha256(reuse/'results.json') != plan['reuse_ocr_result_sha256']):
        raise ValueError('CPU OCR source result/protocol changed')
    old_protocol = json.loads((reuse/'protocol.json').read_text())
    old_result = json.loads((reuse/'results.json').read_text())
    if (old_protocol['parseq_checkpoint_sha256'] != PARSEQ_CHECKPOINT_SHA256
            or old_protocol['parseq_source_archive_sha256'] != PARSEQ_RUNTIME_ARCHIVE_SHA256
            or not old_protocol['parseq_decode_ar'] or old_protocol['parseq_refine_iters'] != 1
            or old_protocol['plan']['runtime'] != {'device':'cpu','precision':'float32','batch_size':16,
                'torch_threads':2,'torch_interop_threads':1,'seed':202503}):
        raise ValueError('CPU OCR recognition recipe differs')
    frozen_readings = {}
    for sample in samples:
        old_shard = json.loads((reuse/'proposals'/(_sample_key(sample.sample_id)+'.json')).read_text())
        for pad,variant in old_shard['variants'].items():
            old_rows = next(r for r in old_result['outputs'][pad]['padded_crop_box'] if r['sample_id']==sample.sample_id)
            if len(variant['proposals']) != len(old_rows['predictions']):
                raise ValueError('Old OCR proposal/output population differs')
            for proposal,prediction in zip(variant['proposals'],old_rows['predictions']):
                op = reuse/'ocr'/(proposal['crop_array_sha256']+'.json')
                reading = json.loads(op.read_text())
                if (reading['protocol_sha256'] != plan['reuse_ocr_protocol_sha256']
                        or reading['crop_array_sha256'] != proposal['crop_array_sha256']
                        or reading['raw_text'] != prediction['text']
                        or reading['sequence_confidence_native'] != prediction['confidence']):
                    raise ValueError('Old CPU OCR differs from the frozen result')
                frozen_readings[proposal['crop_array_sha256']] = (op, reading)
    if ocr_path.exists():
        for line in ocr_path.read_text().splitlines():
            record = json.loads(line)
            digest = record['crop_array_sha256']
            if record['protocol_sha256'] != protocol_sha or digest not in unique:
                raise ValueError('cached OCR provenance differs')
            if digest in ocr and ocr[digest] != record:
                raise ValueError('conflicting duplicate OCR result')
            ocr[digest] = record
    reused_count = 0
    last_reused_id, reuse_rgb = None, None
    for digest,(sample_id,proposal) in unique.items():
        if digest in ocr or digest not in frozen_readings:
            continue
        if last_reused_id != sample_id:
            with Image.open(image_paths[sample_id]) as opened:
                reuse_rgb = np.ascontiguousarray(opened.convert('RGB'))
            last_reused_id = sample_id
        read_crop(reuse_rgb,proposal)
        op,old = frozen_readings[digest]
        record = {k:v for k,v in old.items() if k not in {'protocol_sha256','crop_file_sha256'}}
        record.update(protocol_sha256=protocol_sha,reuse={'path':str(op),
            'sha256':file_sha256(op),'original_protocol_sha256':old['protocol_sha256'],
            'method':'identical RGB pixel hash and pinned CPU FP32 PARSeq recipe'})
        with ocr_path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        ocr[digest] = record
        reused_count += 1
    print(f'[v1 epoch485 control reuse] CPU OCR crops={reused_count}',flush=True)
    pending = [(digest, sample_id, proposal) for digest, (sample_id, proposal) in unique.items() if digest not in ocr]
    print(f'[v1 epoch485 control PARSeq CPU] unique_crops={len(unique)} pending={len(pending)}; no raster files', flush=True)
    torch.set_num_threads(runtime['parseq_threads'])
    if pending:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
            device='cpu', runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive.resolve()))
        try:
            previous_id, cached_image = None, None
            batch_size = runtime['parseq_batch_size']
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                images = []
                for digest, sample_id, proposal in batch:
                    if sample_id != previous_id:
                        if file_sha256(image_paths[sample_id]) != next(s['image_sha256'] for s in shards if s['sample_id'] == sample_id):
                            raise ValueError('source RGB changed between segmentation and OCR')
                        with Image.open(image_paths[sample_id]) as opened:
                            cached_image = np.ascontiguousarray(opened.convert('RGB'))
                        previous_id = sample_id
                    images.append(read_crop(cached_image, proposal))
                decoded = recognizer.recognize_candidate_batch(images)
                if len(decoded) != len(batch):
                    raise RuntimeError('PARSeq output cardinality differs from real crop batch')
                with ocr_path.open('a', encoding='utf-8') as stream:
                    for (digest, sample_id, proposal), value in zip(batch, decoded):
                        record = {'protocol_sha256': protocol_sha, 'crop_array_sha256': digest, **value.to_dict()}
                        stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                        ocr[digest] = record
                    stream.flush()
                if start % (batch_size * 8) == 0 or start + len(batch) == len(pending):
                    print(f'[v1 epoch485 control PARSeq CPU] completed={start+len(batch)}/{len(pending)} elapsed_s={time.monotonic()-started:.1f}', flush=True)
        finally:
            recognizer.release()
    gt, annotations = {}, []
    for sample, shard in zip(samples, shards):
        path = image_paths[sample.sample_id]
        annotation = path.parent.parent / 'annotations' / f"{sample.sample_id.partition(':')[2]}.txt"
        h, w = shard['image_hw']
        gt[sample.sample_id] = _gt(annotation, (w, h))
        annotations.append({'sample_id': sample.sample_id, 'path': str(annotation), 'sha256': file_sha256(annotation)})
    care_count = sum(len(words) for words in gt.values())
    if args.limit_images is None and care_count != plan['expected_care_words']:
        raise ValueError('complete official TEST care-word denominator differs')
    baseline = plan['baseline_padding_native_pixels']
    base_ids = {s['sample_id']: {p['contour_index'] for p in s['variants'][str(baseline)]['proposals']} for s in shards}
    summary, rows_by_pad, ids_by_pad = {}, {}, {}
    for pad in plan['padding_native_pixels']:
        summary[str(pad)], rows_by_pad[str(pad)], ids_by_pad[str(pad)] = {}, {}, {}
        for name, padded in [('unpadded_detection_box', False), ('padded_crop_box', True)]:
            rows = evaluation_rows(shards, gt, ocr, pad, padded=padded)
            summary[str(pad)][name] = evaluate_images(rows, text_available=True)
            rows_by_pad[str(pad)][name] = rows
            ids_by_pad[str(pad)][name] = correct_word_ids(rows)
        summary[str(pad)]['selection_changed_images_vs_padding6'] = [
            s['sample_id'] for s in shards if {p['contour_index'] for p in s['variants'][str(pad)]['proposals']} != base_ids[s['sample_id']]]
    for pad in plan['padding_native_pixels']:
        for name in ['unpadded_detection_box', 'padded_crop_box']:
            ids, base = ids_by_pad[str(pad)][name], ids_by_pad[str(baseline)][name]
            summary[str(pad)][name]['vs_padding6'] = {'net_correct_words': len(ids)-len(base),
                'gained_correct_gt_ids': sorted(ids-base), 'lost_correct_gt_ids': sorted(base-ids)}
    result = {'schema_version': 1, 'reproduction_mode': 'approximate', 'result_class': protocol['result_class'],
              'protocol_sha256': protocol_sha, 'checkpoint_epoch': chosen['epoch'], 'checkpoint_sha256': chosen['checkpoint_sha256'],
              'images': len(samples), 'care_words': care_count, 'unique_rgb_crops': len(unique), 'reused_cpu_ocr': sum('reuse' in v for v in ocr.values()),
              'elapsed_seconds': time.monotonic()-started, 'summary': summary, 'outputs': rows_by_pad,
              'annotation_provenance': annotations,
              'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'saved_raster_or_weight_files': 0,
              'note': 'Full CPU segmentation control on original-TRAIN-IoU-selected v1 epoch485; compare with CPU epoch447 at identical geometry/crop/OCR settings. Reuse verified CPU OCR only when crop pixels and recognition recipe match.'}
    write(output / 'results.json', result)
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size > plan['output_budget_bytes']:
        raise RuntimeError('small numeric output budget exceeded')
    print(json.dumps({'completed': True, 'epoch': chosen['epoch'], 'images': len(samples), 'care_words': care_count,
        'numeric_output_bytes': size, 'correct_words_unpadded': {p: v['unpadded_detection_box']['recognition']['exact_casefold_spatial_one_to_one'] for p, v in summary.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'run-directory', 'train-selection', 'test-manifest', 'data-root',
                 'parseq-source', 'parseq-source-archive', 'parseq-checkpoint', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--limit-images', type=int)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        write(arguments.output_dir / 'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__,
            'error': str(error), 'traceback': traceback.format_exc()})
        raise
