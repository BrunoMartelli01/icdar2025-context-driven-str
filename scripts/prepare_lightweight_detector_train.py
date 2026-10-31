"""TRAIN-only external lightweight detector/PARSeq source qualification."""

import argparse
import importlib.metadata
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch

from localize.external_text_detectors import DetectorRuntime, crop_region
from recognition.group_word_selection import reading_confidence
from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from scripts.crop_clip_runtime import array_sha
from scripts.icdar_budget_data import read, sha, key
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def run(path, smoke):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for name, digest in c['pinned_inputs'].items():
        if sha(name) != digest:
            raise ValueError('Changed detector preparation input: '+name)
    plan = read(c['original_plan']); records = plan['records']['icdar13']['train']
    if smoke:
        records = records[:c['smoke_images']]
    root = Path(c['output_root'])/('smoke' if smoke else 'train')
    root.mkdir(parents=True, exist_ok=False); source = {}
    for package in ('scripts', 'recognition', 'localize', 'segmentation', 'models',
                    'pipeline', 'caption', 'scoring', 'research'):
        for file in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/file.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, dest)
            source[str(file.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    torch.set_num_threads(c['cpu_threads']); torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit real CUDA detector qualification required')
    free, total = torch.cuda.mem_get_info()
    if free < c['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free CUDA memory')
    cfg = c['detector_config']; name = c['detector_name']
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction; external detector extension', config=c,
        config_sha256=sha(path), source_sha256=source, smoke=smoke,
        result_class='smoke-test result' if smoke else 'TRAIN inference preparation',
        records=records, test_used=False, ground_truth_used=False,
        hardware=dict(gpu=torch.cuda.get_device_name(c['device']), precision='float32',
            initial_free_bytes=free, total_bytes=total, device=c['device'],
            detector_batch_size=1, parseq_batch_size=c['batch_size'], sequential_loading=True),
        packages={p: importlib.metadata.version(p) for p in
                  ('torch', 'torchvision', 'numpy', cfg['package'])}))
    detector = reader = None; stage = 'detector'; started = time.monotonic(); det_info = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        detector = DetectorRuntime(name, cfg, Path(c['detector_checkpoint']), c['device'])
        for number, record in enumerate(records):
            if sha(record['image_path']) != record['image_sha256']:
                raise ValueError('Changed detector RGB')
            with Image.open(record['image_path']) as im:
                rgb = np.ascontiguousarray(im.convert('RGB'))
            regions = detector.infer(rgb)
            seal(root/'detector'/(key(record['sample_id'])+'.json'), dict(regions=regions,
                image_sha256=record['image_sha256'], ground_truth_used=False,
                input_shape=detector.last_input_shape,
                allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()))
            if number % 50 == 0:
                print('DETECTION', name, number+1, '/', len(records), flush=True)
        det_info = dict(parameters=detector.parameters,
            parameters_before_reparameterize=detector.parameters_before_reparameterize,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            last_input_shape=detector.last_input_shape)
        detector.release(); detector = None; stage = 'parseq'; torch.cuda.reset_peak_memory_stats()
        reader = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['parseq_source']),
            checkpoint=Path(c['parseq_checkpoint']), runtime_source_archive=Path(c['parseq_archive']),
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE, device=c['device']))
        count = rejected = 0; manifests = {view: {} for view in c['output_views']}
        for number, record in enumerate(records):
            with Image.open(record['image_path']) as im:
                rgb = np.ascontiguousarray(im.convert('RGB'))
            regions = read(root/'detector'/(key(record['sample_id'])+'.json'))['regions']
            rejected += sum(not r['valid'] for r in regions)
            items = [(region, view, crop_region(rgb, region, view))
                for region in regions if region['valid'] for view in c['crop_views']]
            rows = []
            for offset in range(0, len(items), c['batch_size']):
                group = items[offset:offset+c['batch_size']]
                outputs = reader.recognize_candidate_batch([g[2] for g in group])
                if len(outputs) != len(group):
                    raise RuntimeError('Incomplete detector PARSeq readings')
                for (region, view, crop), output in zip(group, outputs):
                    raw = output.to_dict()
                    rows.append(dict(raw, confidence=reading_confidence(raw)['confidence_balanced'],
                        detector_score=region['score'], bbox=region['bbox'], view=view,
                        source_index=region['source_index'], candidate_id=f"{record['sample_id']}:{name}:{region['source_index']}:{view}",
                        crop_rgb_sha256=array_sha(crop), crop_shape=list(crop.shape)))
                count += len(group)
            by_region = {}
            for row in rows:
                by_region.setdefault(row['source_index'], []).append(row)
            views = {view: [r for r in rows if r['view'] == view] for view in c['crop_views']}
            views['best_ocr'] = [max(group, key=lambda r: (r['confidence'], r['view'] == 'axis_aligned'))
                                 for group in by_region.values()]
            for view in c['output_views']:
                target = root/'observations'/view/(key(record['sample_id'])+'.json')
                seal(target, dict(image_sha256=record['image_sha256'], rows=views[view],
                    ground_truth_used=False, detector_source_sha256=sha(root/'detector'/(key(record['sample_id'])+'.json')),
                    rejected_detector_regions=sum(not r['valid'] for r in regions)))
                manifests[view][str(target)] = sha(target)
            if number % 50 == 0:
                print('PARSEQ', name, number+1, '/', len(records), 'crops', count, flush=True)
        seal(root/'completion.json', dict(status='COMPLETE', mode='approximate',
            claim_label='approximate reproduction; external detector extension', smoke=smoke,
            images=len(records), actual_parseq_crop_inferences=count,
            actual_new_deepsolo_inferences=0, rejected_detector_regions=rejected,
            detector=det_info, parseq=dict(peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), max_input_shape=[c['batch_size'], 3, 32, 128]),
            seconds=time.monotonic()-started, out_of_memory=False, test_used=False,
            output_sha256=manifests))
    except BaseException as exc:
        seal(root/'failure.json', dict(stage=stage, error=repr(exc), traceback=traceback.format_exc(),
            out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))
        raise
    finally:
        if detector is not None:
            detector.release()
        if reader is not None:
            reader.release()
        torch.cuda.empty_cache()
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); args = p.parse_args(); run(args.config, args.smoke)
