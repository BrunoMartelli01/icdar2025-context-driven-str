"""Paired APPROXIMATE DeepSolo evaluation; never retune a router or replace old caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import traceback
from types import SimpleNamespace

from deepsolo.adapter import DeepSoloProcessError, DeepSoloSubprocessAdapter
from deepsolo.candidates import resolve_runtime_candidate, verify_host_source_identity
from recognition.clip_router_evaluation import deepsolo_predictions, evaluate_predictions
from recognition.paper_image_router_evaluation import word_metrics
from recognition.yolo_router_comparison import aggregate
from scripts import run_clip_router_e2e as source
from scripts.crop_clip_runtime import sha
from scripts.run_clip_high_recall_admission import payload_digest, seal

REPO = Path(__file__).resolve().parents[1]


def model_input_shape(image_wh, orientation, model_name):
    """CHW after native ResizeShortestEdge and, for ViTAE, bottom/right padding."""
    w, h = image_wh
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    short = 1024 if model_name == 'vitae_textocr' else 1000
    scale = short/min(h, w)
    new_h, new_w = h*scale, w*scale
    if max(new_h, new_w) > 1892:
        scale = 1892/max(new_h, new_w)
        new_h, new_w = new_h*scale, new_w*scale
    new_h, new_w = int(new_h+.5), int(new_w+.5)
    if model_name == 'vitae_textocr':
        new_h, new_w = ((new_h+31)//32)*32, ((new_w+31)//32)*32
    return [3, new_h, new_w]


def validate_response(response, candidate, checkpoint_sha256, config_sha256, threshold):
    a = response['artifact']
    load = a['load_validation']
    if (response['success'] is not True or a['candidate']['artifact_id'] != candidate
            or a['checkpoint_sha256'] != checkpoint_sha256
            or a['config_sha256'] != config_sha256
            or not load['strict_compatibility_check']
            or any(load[k] for k in ('missing_keys', 'unexpected_keys', 'incorrect_shapes'))
            or a['resolved_inference_threshold'] != threshold
            or a['inference_options'] != {'device': 'cuda:0', 'score_threshold': threshold,
                                           'lexicon_type': None}
            or response['placement']['device'] != 'cuda:0'):
        raise ValueError('Wrong candidate, incompatible checkpoint or changed inference protocol')
    return response


def read_shard(path, contract):
    record = json.loads(path.read_text())
    if record['contract'] != contract or record['payload_sha256'] != payload_digest(record):
        raise ValueError('Changed cached DeepSolo response: ' + str(path))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config', type=Path,
                        default=REPO/'configs/approximate/deepsolo_textocr_comparison_v1.json')
    parser.add_argument('--model', required=True, choices=['r50', 'r50_textocr', 'vitae_textocr'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--replay-baseline', action='store_true', help='Audit all cached baseline TEST outputs on CPU')
    args = parser.parse_args()
    cfg = source.read_approximate(args.config)
    if (cfg['reproduction_mode'] != args.mode or cfg['schema_version'] != 1
            or cfg['claim_label'] != 'approximate reproduction' or cfg['table_5_claim'] is not False
            or cfg['device'] != 'cuda:0' or not 0 <= cfg['score_threshold'] <= 1):
        parser.error('Unsupported scientific configuration')
    output = args.output_root.resolve()
    if output.is_relative_to(REPO) or len(set(args.splits)) != len(args.splits):
        parser.error('Use an external output directory and unique splits')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.replay_baseline and (args.model != 'r50' or args.splits != ['test']):
        parser.error('--replay-baseline requires --model r50 --splits test')
    paths = source.read_approximate(REPO/cfg['runtime_paths'])
    original_runtime = source.resolve(paths['deepsolo_runtime_manifest'])
    runtime = verify_host_source_identity(original_runtime, source.resolve(paths['deepsolo_source_root']))
    candidate = cfg['models'][args.model]['candidate']
    if args.model != 'r50':
        if args.checkpoint is None or not re.fullmatch(r'[0-9a-f]{64}', args.checkpoint_sha256 or ''):
            parser.error('New candidates require --checkpoint and explicit --checkpoint-sha256; obtain the official weights first')
        if not args.checkpoint.is_file() or sha(args.checkpoint) != args.checkpoint_sha256:
            parser.error('Checkpoint missing or SHA256 mismatch')
        if args.checkpoint_sha256 in {r['checkpoint_sha256'] for r in runtime['candidates'].values()}:
            parser.error('Cannot relabel an existing baseline checkpoint as a new candidate')
        runtime['candidates'][candidate] = {'checkpoint_path': str(args.checkpoint.resolve()),
                                           'checkpoint_sha256': args.checkpoint_sha256}
    elif args.checkpoint is not None or args.checkpoint_sha256 is not None:
        parser.error('Baseline uses its existing qualified checkpoint unchanged')
    resolved = resolve_runtime_candidate(runtime, candidate)
    random.seed(20261016)
    values = source.qualify(SimpleNamespace(mode=args.mode, config=REPO/cfg['source_config'],
        paths=REPO/cfg['source_paths'], output_root=None))
    old_cfg, _, _, _, _, _, _, fallback, _, _, _, splits = values
    comparison_paths = source.read_approximate(REPO/cfg['comparison_paths'])
    source.checked(source.resolve(comparison_paths['source_root'])/'resolved_configuration.json',
                   cfg['source_plan_sha256'])
    baseline_path = source.resolve(comparison_paths['completed_branches'])
    baseline = source.checked(baseline_path, cfg['baseline_branches_sha256'])
    populations = {split: splits[split][:args.limit] if args.limit else splits[split] for split in args.splits}
    baseline_rows = {split: {r['sample_id']: r for r in baseline['splits'][split]} for split in args.splits}
    for split, samples in populations.items():
        for sample in samples:
            if baseline_rows[split][sample['observed']['sample_id']]['ground_truth'] != sample['ground_truth']:
                raise ValueError('Paired baseline ground truth differs')
    plan = {'reproduction_mode': args.mode, 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'config': cfg, 'model': args.model,
        'source_plan_sha256': cfg['source_plan_sha256'],
        'original_runtime_sha256': sha(original_runtime), 'runtime': runtime,
        'baseline_branches_path': str(baseline_path), 'baseline_branches_sha256': sha(baseline_path),
        'candidate_resolution': resolved, 'splits': args.splits, 'limit': args.limit,
        'result_class': 'partial-dataset result' if args.limit else 'approximate reproduction',
        'replay_baseline': args.replay_baseline,
        'images': {split: [{'sample_id': r['observed']['sample_id'],
            'image_sha256': r['observed']['image_sha256'], 'image_wh': r['image_wh'],
            'orientation': r['orientation'], 'ground_truth': r['ground_truth']}
            for r in rows] for split, rows in populations.items()},
        'code_sha256': {p: sha(REPO/p) for p in ('scripts/run_deepsolo_textocr_comparison.py',
            *[str(p.relative_to(REPO)) for p in sorted((REPO/'deepsolo').glob('*.py'))],
            'recognition/clip_router_evaluation.py', 'recognition/paper_image_router_evaluation.py',
            'recognition/yolo_router_comparison.py')}}
    import hashlib
    root = output/hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    if args.check:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'images': {s: len(r) for s,r in populations.items()},
                          'output': str(root), 'checkpoint_inference': 'NOT_RUN'}), flush=True)
        return
    seal(root/'resolved_configuration.json', plan)
    seal(root/'runtime.json', runtime)
    for name, digest in plan['code_sha256'].items():
        data = (REPO/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Source changed while taking snapshot')
        target = root/'code_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != digest:
            raise ValueError('Source snapshot changed')
        if not target.exists():
            target.write_bytes(data)
    adapter = DeepSoloSubprocessAdapter(Path(paths['deepsolo_python']).absolute(), root/'runtime.json',
        host_source_root=source.resolve(paths['deepsolo_source_root']), repository_root=REPO)
    result = {'claim_label': 'approximate reproduction', 'result_class': plan['result_class'],
        'official_evaluation': False, 'metric': cfg['metric'], 'splits': {}, 'model': args.model}
    try:
        for split, samples in populations.items():
            rows = []
            for i, sample in enumerate(samples, 1):
                sid = sample['observed']['sample_id']
                contract = {'plan_sha256': sha(root/'resolved_configuration.json'),
                            'sample_id': sid, 'image_sha256': sample['observed']['image_sha256']}
                path = root/split/(source.key(sid)+'.json')
                if path.exists():
                    response = read_shard(path, contract)['response']
                elif args.replay_baseline:
                    cached = json.loads(sample['ds_cache_path'].read_text())
                    actual = cached['response']['artifact']['resolved_inference_threshold']
                    response = source.validate_deepsolo(cached['response'], fallback, actual)
                else:
                    response = adapter.infer(image_path=Path(sample['observed']['image_path']),
                        candidate_id=candidate, inference_options={'device': cfg['device'],
                        'score_threshold': cfg['score_threshold'], 'lexicon_type': None},
                        reduction_policy='reading_order_space_join').to_dict()
                if not args.replay_baseline:
                    validate_response(response, candidate, resolved['checkpoint_sha256'],
                                      resolved['config_sha256'], cfg['score_threshold'])
                    if not path.exists():
                        source.save_shard(path, contract, {'response': response})
                predictions = deepsolo_predictions(response, sample['image_wh'], sample['orientation'],
                                                   sample['validity'], cfg['score_threshold'])
                rows.append({'sample_id': sid, 'word': word_metrics(predictions, sample['ground_truth']),
                    'geometric': evaluate_predictions(predictions, sample['ground_truth'], iou_threshold=old_cfg['matching_iou']),
                    'predictions': predictions, 'memory': response['memory'],
                    'model_input_chw': model_input_shape(sample['image_wh'], sample['orientation'], args.model),
                    'precision': 'float32', 'batch_size': 1,
                    'latency': response['latency'], 'load_validation': response['artifact']['load_validation'],
                    'raw_response_path': str(sample['ds_cache_path'] if args.replay_baseline else path),
                    'raw_response_sha256': sha(sample['ds_cache_path'] if args.replay_baseline else path)})
                print(f'[{args.model} {split}] {i}/{len(samples)} {sid}', flush=True)
            word = aggregate(r['word'] for r in rows)
            old_word = aggregate(baseline_rows[split][r['sample_id']]['deepsolo'] for r in rows)
            if args.replay_baseline and word != old_word:
                raise ValueError('Replayed baseline differs from its frozen word counts')
            result['splits'][split] = {'word': word, 'baseline_word': old_word,
                'delta_vs_baseline': {k: word[k]-old_word[k] for k in
                    ('tp', 'fp', 'fn', 'detected_words', 'f1', 'accuracy_tn0')},
                'geometric': aggregate(r['geometric'] for r in rows), 'per_image': rows}
        result['status'] = 'SUCCESS'
        seal(root/'results.json', result)
        print(json.dumps({'output': str(root), 'splits': {s:r['word'] for s,r in result['splits'].items()}}), flush=True)
    except Exception as error:
        failure = {'status': 'FAILED', 'sample_id': locals().get('sid'), 'split': locals().get('split'),
                   'error': str(error), 'traceback': traceback.format_exc(), 'partial_metrics_are_not_final': True}
        if isinstance(error, DeepSoloProcessError) and error.response is not None:
            failure['response'] = error.response.to_dict()
        seal(root/'failure.json', failure)
        raise


if __name__ == '__main__':
    main()
