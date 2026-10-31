"""APPROXIMATE low-floor/resolution study, using separate existing Python runtimes.

Run export in legacy DeepSolo Python; evaluate in main Python. Source datasets,
checkpoint/runtime, native preprocessing and denominators are verified explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True)+'\n'
    if path.exists():
        if path.read_text() != data:
            raise ValueError('Refuse to overwrite different evidence: '+str(path))
        return
    temporary = path.with_suffix('.partial')
    temporary.write_text(data)
    temporary.replace(path)


def validate_config(cfg):
    if (cfg['reproduction_mode'] != 'approximate' or cfg['claim_label'] != 'approximate reproduction'
            or cfg['table_5_claim'] is not False or cfg['device'] != 'cuda:0'
            or cfg['precision'] != 'float32' or cfg['batch_size'] != 1
            or not 0 <= cfg['inference_floor'] <= min(cfg['score_thresholds'])
            or any(not 0 <= q <= 1 for q in cfg['recognition_thresholds'])
            or any(not cfg['inference_floor'] <= s <= 1 for s in cfg['score_thresholds'])
            or any(len(v) != 2 or any(type(n) is not int or n < 1 for n in v)
                   for v in cfg['resolutions'].values())):
        raise ValueError('Invalid explicit APPROXIMATE experiment configuration')


def configure_model_resize(model, inference):
    """Synchronize ViTAE's cached inverse resize with actual input augmentation.

    Frozen TransformerPureDetector stores these at construction, rather than
    reading the supplied inference cfg in forward(). Both must change together.
    """
    if not hasattr(model, 'min_size_test') or not hasattr(model, 'max_size_test'):
        raise ValueError('Expected frozen ViTAE postprocessing resize attributes')
    model.min_size_test = inference.INPUT.MIN_SIZE_TEST
    model.max_size_test = inference.INPUT.MAX_SIZE_TEST
    return [int(model.min_size_test), int(model.max_size_test)]


def export(args, cfg, plan):
    from deepsolo.candidates import resolve_runtime_candidate, verify_host_source_identity
    runtime = verify_host_source_identity(args.runtime, args.source_code_root)
    resolved = resolve_runtime_candidate(runtime, cfg['candidate'])
    if resolved['checkpoint_sha256'] != cfg['checkpoint_sha256']:
        raise ValueError('Unexpected checkpoint')
    protocol = dict(config=cfg, source_plan_sha256=sha(args.source_plan), runtime=runtime,
        data_root=str(args.data_root.resolve()), config_sha256=sha(args.config),
        code_sha256={str(p):sha(p) for p in [Path(__file__), *sorted((Path(__file__).parents[1]/'deepsolo').glob('*.py'))]},
        images=plan['images'], claim_label=cfg['claim_label'], reproduction_mode='approximate',
        result_class='approximate reproduction', official_evaluation=False)
    save(args.output_root/'protocol.json', protocol)
    # Fail before model execution if any requested image is absent or changed.
    for split in args.splits:
        for record in plan['images'][split]:
            image = args.data_root/'datasets/total_text'/split/'images'/(record['sample_id'].split(':')[1]+'.jpg')
            if sha(image) != record['image_sha256']:
                raise ValueError('Image checksum mismatch: '+str(image))
    sys.path.insert(0, str(args.source_code_root))
    import cv2
    import numpy as np
    import torch
    from adet.config import get_cfg
    from deepsolo.worker import _load_model_and_validate, _predict, _serialize_instances
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed']); torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    base = get_cfg(); base.merge_from_file(resolved['config_path'])
    base.MODEL.WEIGHTS = resolved['checkpoint_path']; base.MODEL.DEVICE = cfg['device']
    base.MODEL.TRANSFORMER.INFERENCE_TH_TEST = cfg['inference_floor']
    base.freeze()
    model, load = _load_model_and_validate(base, resolved['checkpoint_path'])
    context = dict(checkpoint_sha256=resolved['checkpoint_sha256'], config_sha256=resolved['config_sha256'],
        load_validation=load, source_identity=resolved, precision='float32', batch_size=1,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0), cudnn_enabled=True, cudnn_benchmark=False,
        cudnn_deterministic=False, seed=cfg['seed'], protocol_sha256=sha(args.output_root/'protocol.json'))
    save(args.output_root/'runtime.json', context)
    for resolution, sizes in cfg['resolutions'].items():
        inference = base.clone(); inference.defrost()
        inference.INPUT.MIN_SIZE_TEST, inference.INPUT.MAX_SIZE_TEST = sizes
        inference.freeze()
        postprocess_resize = configure_model_resize(model, inference)
        for split in args.splits:
            for i, record in enumerate(plan['images'][split], 1):
                path = args.output_root/resolution/split/(record['sample_id'].replace(':', '__')+'.json')
                contract = dict(protocol_sha256=context['protocol_sha256'], image_sha256=record['image_sha256'],
                                sample_id=record['sample_id'], resolution=resolution)
                if path.exists():
                    old = read(path)
                    payload = {k:v for k,v in old.items() if k != 'payload_sha256'}
                    if (old['contract'] != contract or old['payload_sha256'] != hashlib.sha256(
                            json.dumps(payload, sort_keys=True).encode()).hexdigest()):
                        raise ValueError('Changed batch shard: '+str(path))
                    continue
                image_path = args.data_root/'datasets/total_text'/split/'images'/(record['sample_id'].split(':')[1]+'.jpg')
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Cannot decode input RGB')
                shape = []

                def capture_shape(module, inputs):
                    shape.append(list(inputs[0][0]['image'].shape))

                hook = model.register_forward_pre_hook(capture_shape)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
                tick = time.perf_counter()
                try:
                    prediction = _predict(model, inference, image, True)
                    torch.cuda.synchronize(0)
                    detections = _serialize_instances(prediction['instances'], inference)
                finally:
                    hook.remove()
                shard = dict(schema_version=1, status='SUCCESS', claim_label=cfg['claim_label'], contract=contract,
                    response=dict(success=True, detections=detections), model_input_chw=shape[0],
                    postprocess_resize=postprocess_resize,
                    memory=dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0)),
                    inference_seconds=time.perf_counter()-tick, inference_floor=cfg['inference_floor'])
                shard['payload_sha256'] = hashlib.sha256(json.dumps(shard, sort_keys=True).encode()).hexdigest()
                save(path, shard)
                print('[{} {}] {}/{} {}'.format(resolution, split, i, len(plan['images'][split]), record['sample_id']), flush=True)
    save(args.output_root/('export_'+'_'.join(args.splits)+'.json'),
         dict(status='SUCCESS', splits=args.splits, resolutions=list(cfg['resolutions']), protocol_sha256=context['protocol_sha256']))


def evaluate(args, cfg, plan):
    from itertools import product
    import numpy as np
    from PIL import Image
    from recognition.clip_router_evaluation import deepsolo_predictions, native_mask
    from recognition.deepsolo_word_f1 import aggregate, apply_policy, error_inventory, recognition_confidence, word_counts
    protocol = read(args.output_root/'protocol.json')
    if protocol['config'] != cfg or protocol['images'] != plan['images']:
        raise ValueError('Export protocol mismatch')
    rows = {}
    inputs = {}
    for resolution in cfg['resolutions']:
        rows[resolution] = {}
        for split in ('train', 'test'):
            result = []
            for record in plan['images'][split]:
                path = args.output_root/resolution/split/(record['sample_id'].replace(':', '__')+'.json')
                raw = read(path)
                payload = {k:v for k,v in raw.items() if k != 'payload_sha256'}
                if (raw['status'] != 'SUCCESS' or raw['contract'] != dict(
                        protocol_sha256=sha(args.output_root/'protocol.json'), image_sha256=record['image_sha256'],
                        sample_id=record['sample_id'], resolution=resolution)
                        or raw['payload_sha256'] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()):
                    raise ValueError('Changed or failed shard')
                inputs[str(path)] = sha(path)
                validity_path = args.data_root/'datasets/total_text'/split/'validity'/(record['sample_id'].split(':')[1]+'.png')
                image_path = args.data_root/'datasets/total_text'/split/'images'/(record['sample_id'].split(':')[1]+'.jpg')
                with Image.open(image_path) as im:
                    if list(im.size) != record['image_wh'] or int(im.getexif().get(274, 1)) != record['orientation']:
                        raise ValueError('Image frame changed')
                inputs[str(validity_path)] = sha(validity_path)
                with Image.open(validity_path) as im:
                    valid = native_mask(im.convert('L'), 256)
                if valid.dtype != np.bool_:
                    raise ValueError('Invalid validity mask')
                predictions = deepsolo_predictions(raw['response'], record['image_wh'], record['orientation'], valid, cfg['inference_floor'])
                ds = {d['source_index']:d for d in raw['response']['detections']}
                for p in predictions:
                    p['recognition_confidence'] = recognition_confidence(ds[p['instance_id']]['raw']['recognition_class_probabilities'])
                result.append(dict(sample_id=record['sample_id'], ground_truth=record['ground_truth'], predictions=predictions))
            rows[resolution][split] = result

    def metrics(resolution, split, policy):
        return aggregate(word_counts(apply_policy(r['predictions'], policy), r['ground_truth']) for r in rows[resolution][split])

    grid = []
    for resolution, score, rec in product(cfg['resolutions'], cfg['score_thresholds'], cfg['recognition_thresholds']):
        policy = dict(decoder='greedy', score_threshold=score, recognition_threshold=rec, same_text_nms_iou=None)
        grid.append(dict(resolution=resolution, policy=policy, train=metrics(resolution, 'train', policy)))
    selected = max(grid, key=lambda r:r['train']['f1'])
    # This sealed choice is written before any TEST metric is calculated.
    save(args.output_root/'selection.json', dict(selected=selected, train_grid=grid, test_fitting=False,
        input_sha256=inputs, code_sha256={str(p):sha(p) for p in (Path(__file__), Path(__file__).parents[1]/'recognition/deepsolo_word_f1.py')}))
    result = dict(status='SUCCESS', claim_label=cfg['claim_label'], reproduction_mode='approximate',
        result_class='approximate reproduction', official_evaluation=False, test_previously_explored=True,
        selected=selected, splits={})
    for split in ('train', 'test'):
        policy = selected['policy']; resolution = selected['resolution']
        per_image = []
        for r in rows[resolution][split]:
            predictions = apply_policy(r['predictions'], policy)
            per_image.append(dict(sample_id=r['sample_id'], ground_truth=r['ground_truth'], predictions=predictions,
                metrics=word_counts(predictions, r['ground_truth']),
                errors=error_inventory(predictions, r['ground_truth'], cfg['diagnostic_bbox_iou'])))
        result['splits'][split] = dict(metrics=metrics(resolution, split, policy), per_image=per_image)
    save(args.output_root/'results.json', result)
    print(json.dumps(dict(selected=selected, splits={s:r['metrics'] for s,r in result['splits'].items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--stage', required=True, choices=['export', 'evaluate'])
    for key in ('config', 'source-plan', 'runtime', 'source-code-root', 'data-root', 'output-root'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    args = parser.parse_args(); cfg = read(args.config); validate_config(cfg)
    repo = str(Path(__file__).resolve().parents[1])
    if 'approximate' not in args.config.parts or os.path.commonpath([str(args.output_root.resolve()), repo]) == repo:
        raise ValueError('Use APPROXIMATE namespace and external outputs')
    if sha(args.source_plan) != cfg['source_plan_sha256']:
        raise ValueError('Source plan changed')
    plan = read(args.source_plan)
    try:
        (export if args.stage == 'export' else evaluate)(args, cfg, plan)
    except Exception as error:
        failure = dict(status='FAILED', type=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc(), oom='out of memory' in str(error).lower())
        save(args.output_root/('failure_'+str(time.time_ns())+'.json'), failure)
        raise


if __name__ == '__main__':
    main()
