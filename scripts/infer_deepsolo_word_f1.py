"""Apply a sealed APPROXIMATE DeepSolo word-F1 policy to new images, without GT.

Run with the existing legacy DeepSolo Python. The original fallback adapter and
STRICT namespace are unchanged; this explicit experiment entry point is separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from scripts.run_deepsolo_word_f1_study import configure_model_resize, read, save, sha
from recognition.deepsolo_word_f1 import apply_policy, fuse_views, normalize_word, recognition_confidence


def validate_selected_config(cfg):
    policy = cfg['selected_policy']
    if (cfg['schema_version'] != 1 or cfg['reproduction_mode'] != 'approximate' or cfg['claim_label'] != 'approximate reproduction'
            or cfg['table_5_claim'] is not False or cfg['precision'] != 'float32'
            or cfg['batch_size'] != 1 or cfg['device'] != 'cuda:0'
            or policy['decoder'] != 'greedy' or policy['same_text_nms_iou'] is not None
            or not cfg['inference_floor'] <= policy['score_threshold'] <= 1
            or not 0 <= policy['recognition_threshold'] <= 1
            or policy['resolution'] not in (*cfg['resolutions'], 'fused')):
        raise ValueError('Expected a sealed APPROXIMATE greedy word-F1 policy')


def raw_predictions(detections, image_wh, orientation):
    from localize.independent_word_slots import (
        EmptyDetectorBoundaryError, boundary_bbox, detector_boundary_in_original,
    )
    result = []
    for d in detections:
        points = detector_boundary_in_original(d['geometry']['boundary_points'], tuple(image_wh), orientation)
        row = dict(instance_id=d['source_index'], text=d['text'], score=d['score'],
            boundary_original=points, ignored=False, invalid_envelope=None,
            recognition_confidence=recognition_confidence(d['raw']['recognition_class_probabilities']))
        try:
            row['bbox'] = boundary_bbox(points, tuple(image_wh))
        except EmptyDetectorBoundaryError as error:
            row.update(bbox=None, invalid_envelope=dict(unclipped_bbox=error.unclipped_bbox,
                clipped_bbox=error.clipped_bbox, treatment='retain_as_unmatchable_FP'))
        result.append(row)
    return result


def run(args, cfg):
    from deepsolo.candidates import resolve_runtime_candidate, verify_host_source_identity
    runtime = verify_host_source_identity(args.runtime, args.source_code_root)
    resolved = resolve_runtime_candidate(runtime, cfg['candidate'])
    if (resolved['checkpoint_sha256'] != cfg['checkpoint_sha256']
            or resolved['config_sha256'] != cfg['upstream_config_sha256']
            or resolved['base_config_sha256'] != cfg['upstream_base_config_sha256']):
        raise ValueError('Checkpoint/config do not match the evaluated selected policy')
    sys.path.insert(0, str(args.source_code_root))
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from adet.config import get_cfg
    from deepsolo.worker import _load_model_and_validate, _predict, _serialize_instances
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed']); torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.enabled = True; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    base = get_cfg(); base.merge_from_file(resolved['config_path'])
    base.MODEL.WEIGHTS = resolved['checkpoint_path']; base.MODEL.DEVICE = cfg['device']
    base.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg['inference_floor']; base.freeze()
    model, load = _load_model_and_validate(base, resolved['checkpoint_path'])
    code_paths = [Path(__file__), Path(__file__).parents[1]/'recognition/deepsolo_word_f1.py',
                  Path(__file__).parents[1]/'scripts/run_deepsolo_word_f1_study.py']
    save(args.output_root/'resolved_configuration.json', dict(config=cfg, runtime=runtime,
        source_identity=resolved, code_sha256={str(p):sha(p) for p in code_paths}))
    policy = cfg['selected_policy']
    views = list(cfg['resolutions']) if policy['resolution'] == 'fused' else [policy['resolution']]
    report = dict(status='SUCCESS', reproduction_mode='approximate', claim_label=cfg['claim_label'],
        result_class='smoke-test result', official_evaluation=False, selected_policy=policy,
        config_sha256=sha(args.config), source_identity=resolved, load_validation=load,
        gpu_name=torch.cuda.get_device_name(0), precision='float32', batch_size=1,
        device=cfg['device'], seed=cfg['seed'], torch_version=torch.__version__,
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=False, images=[])
    for image_path in args.image:
        image_path = image_path.resolve()
        with Image.open(image_path) as im:
            wh, orientation = list(im.size), int(im.getexif().get(274, 1))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode image: '+str(image_path))
        row = dict(image_path=str(image_path), image_sha256=sha(image_path), image_wh=wh, orientation=orientation,
                   raw_views={}, predictions={})
        for view in views:
            inference = base.clone(); inference.defrost()
            inference.INPUT.MIN_SIZE_TEST, inference.INPUT.MAX_SIZE_TEST = cfg['resolutions'][view]
            inference.freeze(); postprocess_resize = configure_model_resize(model, inference)
            shapes = []
            hook = model.register_forward_pre_hook(lambda module, inp: shapes.append(list(inp[0][0]['image'].shape)))
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0); tick = time.perf_counter()
            try:
                predicted = _predict(model, inference, image, True); torch.cuda.synchronize(0)
                detections = _serialize_instances(predicted['instances'], inference)
            finally:
                hook.remove()
            row['raw_views'][view] = dict(detections=detections, model_input_chw=shapes[0],
                postprocess_resize=postprocess_resize, inference_seconds=time.perf_counter()-tick,
                memory=dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
            row['predictions'][view] = raw_predictions(detections, wh, orientation)
        selected = (fuse_views(row['predictions']['native'], row['predictions']['large'], policy)
                    if policy['resolution'] == 'fused' else apply_policy(row['predictions'][policy['resolution']], policy))
        row['selected_predictions'] = selected
        row['words'] = [normalize_word(p['text']) for p in selected]
        report['images'].append(row)
        print(json.dumps(dict(image=str(image_path), words=row['words'])), flush=True)
    save(args.output_root/'results.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    for name in ('config', 'runtime', 'source-code-root', 'output-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path, action='append')
    args = parser.parse_args(); cfg = read(args.config); validate_selected_config(cfg)
    repo = str(Path(__file__).resolve().parents[1])
    if 'approximate' not in args.config.parts or os.path.commonpath([str(args.output_root.resolve()), repo]) == repo:
        raise ValueError('Explicit APPROXIMATE config and external outputs required')
    try:
        run(args, cfg)
    except Exception as error:
        save(args.output_root/('failure_'+str(time.time_ns())+'.json'), dict(status='FAILED',
            message=str(error), traceback=traceback.format_exc(), oom='out of memory' in str(error).lower()))
        raise


if __name__ == '__main__':
    main()
