"""TRAIN-selected DeepSolo resolution/fusion policies under two explicit metrics."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

from scripts.run_deepsolo_word_f1_study import read, save, sha, validate_config
from recognition.deepsolo_word_f1 import (
    aggregate, apply_policy, error_inventory, fuse_views, normalize_word,
    recognition_confidence, word_counts,
)


def predictions(row, policy):
    if policy['resolution'] == 'fused':
        return fuse_views(row['native'], row['large'], policy)
    return apply_policy(row[policy['resolution']], policy)


def counts(preds, gt, metric, iou):
    lexical = word_counts(preds, gt)
    if metric == 'word':
        return lexical
    from recognition.clip_router_evaluation import evaluate_predictions
    result = evaluate_predictions([dict(p, text=normalize_word(p['text'])) for p in preds],
        [dict(g, text=normalize_word(g['text'])) for g in gt], iou_threshold=iou)
    return dict(lexical, **{k:result[k] for k in ('tp', 'fp', 'fn')})


def score(rows, policy, metric, iou):
    return aggregate(counts(predictions(r, policy), r['ground_truth'], metric, iou) for r in rows)


def select_policies(train_rows, policies, iou):
    # Signature accepts TRAIN rows only. No TEST object enters this function.
    grid = [dict(policy=p, word=score(train_rows, p, 'word', iou),
                 localized=score(train_rows, p, 'localized', iou)) for p in policies]
    return {m:max(grid, key=lambda r:r[m]['f1'])['policy'] for m in ('word', 'localized')}, grid


def paired_bootstrap(base, changed, repetitions, seed):
    import numpy as np
    if len(base) != len(changed) or not base:
        raise ValueError('Paired image populations required')
    a = np.array([[r[k] for k in ('tp', 'fp', 'fn')] for r in base], dtype=np.float64)
    b = np.array([[r[k] for k in ('tp', 'fp', 'fn')] for r in changed], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(base), len(base))
        x, y = a[idx].sum(0), b[idx].sum(0)
        deltas.append(2*y[0]/(2*y[0]+y[1]+y[2])-2*x[0]/(2*x[0]+x[1]+x[2]))
    return dict(repetitions=repetitions, seed=seed, unit='paired image',
                delta_f1_95_percentile_interval=np.quantile(deltas, [.025, .975]).tolist(),
                limitation='Conditional on these fixed fitted policies; repeated TEST exploration and upstream training exposure remain.')


def load_rows(source, source_plan, data_root, output):
    from PIL import Image
    from recognition.clip_router_evaluation import deepsolo_predictions, native_mask
    from segmentation.data import load_segmentation_manifest
    protocol = read(source/'protocol.json'); plan = read(source_plan)
    if protocol['images'] != plan['images']:
        raise ValueError('Changed paired population')
    inputs = {str(source/'protocol.json'):sha(source/'protocol.json'), str(source_plan):sha(source_plan)}
    rows = {}
    for split in ('train', 'test'):
        manifest_path = data_root/'manifests'/('total_text_'+split+'.yaml')
        manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
        if manifest.split_role != split:
            raise ValueError('Wrong dataset role')
        samples = {r.sample_id:r for r in manifest.samples}
        inputs[str(manifest_path)] = sha(manifest_path)
        rows[split] = []
        for record in plan['images'][split]:
            sid = record['sample_id']; sample = samples[sid]
            image_path = data_root/sample.image_path
            if sample.image_sha256 != record['image_sha256'] or sha(image_path) != record['image_sha256']:
                raise ValueError('Input image checksum changed')
            inputs[str(image_path)] = record['image_sha256']
            validity_path = data_root/sample.validity_annotation_path
            if sha(validity_path) != sample.validity_annotation_sha256:
                raise ValueError('Validity checksum changed')
            inputs[str(validity_path)] = sample.validity_annotation_sha256
            with Image.open(validity_path) as im:
                valid = native_mask(im.convert('L'), 256)
            row = dict(record)
            for resolution in protocol['config']['resolutions']:
                path = source/resolution/split/(sid.replace(':', '__')+'.json')
                raw = read(path); payload = {k:v for k,v in raw.items() if k != 'payload_sha256'}
                if (raw['status'] != 'SUCCESS' or not raw['response']['success']
                        or raw.get('postprocess_resize') != protocol['config']['resolutions'][resolution]
                        or raw['contract'] != dict(protocol_sha256=sha(source/'protocol.json'),
                            sample_id=sid, image_sha256=record['image_sha256'], resolution=resolution)
                        or raw['payload_sha256'] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()):
                    raise ValueError('Raw shard contract/hash failure: '+str(path))
                inputs[str(path)] = sha(path)
                preds = deepsolo_predictions(raw['response'], record['image_wh'], record['orientation'],
                                             valid, protocol['config']['inference_floor'])
                ds = {d['source_index']:d for d in raw['response']['detections']}
                for p in preds:
                    p['recognition_confidence'] = recognition_confidence(ds[p['instance_id']]['raw']['recognition_class_probabilities'])
                row[resolution] = preds
            rows[split].append(row)
        print('Verified '+split+': '+str(len(rows[split]))+' images', flush=True)
    save(output/'processed_predictions.json', dict(splits=rows, input_sha256=inputs))
    return rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    for name in ('config', 'source-root', 'source-plan', 'data-root', 'output-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(); cfg = read(args.config); validate_config(cfg)
    if 'approximate' not in args.config.parts or args.output_root.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Explicit APPROXIMATE namespace and external outputs required')
    if sha(args.source_root/'protocol.json') != cfg['export_protocol_sha256'] or sha(args.source_plan) != cfg['source_plan_sha256']:
        raise ValueError('Source export or population changed')
    rows, inputs = load_rows(args.source_root, args.source_plan, args.data_root, args.output_root)
    policies = []
    for res, s, q in product(('native', 'large', 'fused'), cfg['score_thresholds'], cfg['recognition_thresholds']):
        for iou, power in (product(cfg['fusion_ious'], cfg['recognition_rank_powers']) if res == 'fused' else [(None, 0)]):
            policies.append(dict(resolution=res, decoder='greedy', score_threshold=s,
                recognition_threshold=q, same_text_nms_iou=None, fusion_iou=iou, recognition_rank_power=power))
    selected, grid = select_policies(rows['train'], policies, cfg['matching_iou'])
    plan = dict(config=cfg, selected=selected, train_grid=grid, selection_split='train', test_fitting=False,
        input_sha256=inputs, claim_label=cfg['claim_label'], reproduction_mode=args.mode,
        source_sha256={str(p):sha(p) for p in (Path(__file__), Path(__file__).parents[1]/'recognition/deepsolo_word_f1.py')})
    save(args.output_root/'selection.json', plan)
    baseline = dict(resolution='native', decoder='greedy', score_threshold=read(args.source_plan)['config']['score_threshold'],
                    recognition_threshold=0., same_text_nms_iou=None)
    original = read(args.source_plan.with_name('results.json'))
    source_gt = {s:{r['sample_id']:r['ground_truth'] for r in read(args.source_plan)['images'][s]} for s in rows}
    # Native .05 extraction, filtered back to .4, must exactly reproduce old
    # counts under both metrics before an improvement is reported.
    replay = {}
    for split in rows:
        replay[split] = {}
        for metric in ('word', 'localized'):
            old = aggregate(counts(r['predictions'], source_gt[split][r['sample_id']], metric, cfg['matching_iou'])
                            for r in original['splits'][split]['per_image'])
            new = score(rows[split], baseline, metric, cfg['matching_iou'])
            if old != new:
                raise ValueError('Native baseline replay changed: '+split+'/'+metric)
            replay[split][metric] = new
    result = dict(status='SUCCESS', claim_label=cfg['claim_label'], reproduction_mode=args.mode,
        result_class='approximate reproduction', official_evaluation=False, test_previously_explored=True,
        selected=selected, baseline_replay='PASS', protocols={}, baseline=replay)
    for metric, policy in selected.items():
        result['protocols'][metric] = {}
        for split, samples in rows.items():
            per_image = []
            for r in samples:
                p = predictions(r, policy); b = predictions(r, baseline)
                per_image.append(dict(sample_id=r['sample_id'], ground_truth=r['ground_truth'], predictions=p,
                    baseline=counts(b, r['ground_truth'], metric, cfg['matching_iou']),
                    selected=counts(p, r['ground_truth'], metric, cfg['matching_iou']),
                    errors=error_inventory(p, r['ground_truth'], cfg['matching_iou'])))
            chosen = aggregate(r['selected'] for r in per_image)
            base = replay[split][metric]
            result['protocols'][metric][split] = dict(baseline=base, selected=chosen,
                delta_f1=chosen['f1']-base['f1'], per_image=per_image)
            if split == 'test':
                result['protocols'][metric][split]['uncertainty'] = paired_bootstrap(
                    [r['baseline'] for r in per_image], [r['selected'] for r in per_image],
                    cfg['bootstrap_repetitions'], cfg['seed'])
    save(args.output_root/'results.json', result)
    print(json.dumps(dict(selected=selected, protocols={m:{s:{k:v for k,v in r.items() if k != 'per_image'}
        for s,r in splits.items()} for m,splits in result['protocols'].items()})), flush=True)


if __name__ == '__main__':
    main()
