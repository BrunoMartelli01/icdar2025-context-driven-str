"""Explicit Qwen caption ablation with the frozen common router comparison."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np

from scripts.run_paper_caption_comparison import (
    cache_path, contract, digest, image_rgb, load_cache, read, report, score, seal,
    sha, verify,
)


def validate_config(c):
    if c['mode'] != 'approximate' or c['claim_label'] != 'approximate reproduction':
        raise ValueError('Explicit APPROXIMATE caption ablation required')
    if c['caption_model_is_paper_blip2'] is not False:
        raise ValueError('Qwen must be explicitly labelled as a separate caption ablation')
    if c['router'] != dict(alpha=.6, beta=.4, tau=.8, primary_gate='paper_selected', lexical_mode='raw'):
        raise ValueError('Frozen paper scoring parameters required')
    if c['caption_batch_size'] not in (1, 2, 3):
        raise ValueError('Unqualified native caption batch size')
    if c['caption_profile']['caption_model'] != 'qwen':
        raise ValueError('Explicit Qwen profile required')


def prepare(c, config_path, limit):
    import torch
    validate_config(c)
    if limit is not None and limit < 1:
        raise ValueError('Smoke limit must be positive')
    for p, h in c['pinned_inputs'].items(): verify(p, h)
    baseline = read(Path(c['baseline_root']) / 'resolved_configuration.json')
    for p, h in baseline['source_sha256'].items(): verify(p, h)
    for p, h in baseline['config']['pinned_inputs'].items(): verify(p, h)
    qualification = read(c['qualification_receipt'])
    if qualification['status'] != 'QUALIFIED_WITH_LIMITATIONS':
        raise ValueError('Complete native and visual caption qualification required')
    if qualification['profile'] != c['caption_profile']:
        raise ValueError('Changed caption profile after qualification')
    for p, h in qualification['evidence_sha256'].items(): verify(p, h)
    cases = copy.deepcopy(baseline['cases'])
    if limit:
        for name, case in cases.items():
            case['partitions'] = {r: ids[:limit] for r, ids in case['partitions'].items()}
            ids = {s for v in case['partitions'].values() for s in v}
            n = len(case['partitions']['test'])
            original_n = len(baseline['cases'][name]['partitions']['test'])
            case['fallback_images'] = round(n * case['fallback_images'] / original_n)
            case['rows'] = {s: v for s, v in case['rows'].items() if s in ids}
            case['tie_keys'] = {s: v for s, v in case['tie_keys'].items() if s in ids}
    ids = {s for case in cases.values() for s in case['rows']}
    assets = {k: copy.deepcopy(baseline['assets'][k]) for k in ('parseq', 'mpnet')}
    qwen = copy.deepcopy(c['qwen_asset'])
    snapshot = Path(c['hf_cache']) / ('models--' + qwen['model_id'].replace('/', '--')) / 'snapshots' / qwen['revision']
    for f, h in qwen['weight_sha256'].items(): verify(snapshot / f, h)
    qwen['snapshot'] = str(snapshot)
    qwen['configuration_file_sha256'] = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.glob('*.json'))}
    assets['qwen'] = qwen
    from importlib.metadata import version
    return dict(config=c, config_sha256=sha(config_path), mode='approximate',
        claim_label='approximate reproduction', result_class='smoke-test result' if limit else 'approximate reproduction',
        caption_model_is_paper_blip2=False, assets=assets, cases=cases,
        sample_limit=limit,
        samples={s: baseline['samples'][s] for s in sorted(ids)},
        source_sha256={p: sha(p) for p in c['scientific_source_files']},
        runtime=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            packages={p: version(p) for p in ('transformers', 'sentence-transformers', 'numpy', 'Pillow', 'rapidfuzz')},
            deterministic_algorithms=True, tf32=False),
        test_used_for_selection=False, historical_test_exposure=True)


def reuse_t1(root, plan):
    baseline_root = Path(plan['config']['baseline_root'])
    baseline = read(baseline_root / 'resolved_configuration.json')
    for sid, sample in plan['samples'].items():
        original = load_cache(baseline_root, baseline, 't1', sample)
        if original is None: raise FileNotFoundError('Missing frozen native T1: ' + sid)
        existing = load_cache(root, plan, 't1', sample)
        if existing is not None:
            if existing != original: raise ValueError('Reused T1 differs from its frozen source: ' + sid)
            continue
        source = cache_path(baseline_root, 't1', sid)
        seal(cache_path(root, 't1', sid), dict(contract=contract(plan, 't1', sample),
            payload=original, payload_sha256=digest(original),
            reuse_provenance=dict(source=str(source), source_sha256=sha(source),
                neural_inference_rerun=False, scope='Unchanged native PARSeq T1; original runtime and peak memory retained in payload')))


def generate(root, plan):
    import torch
    from caption.qwen_batch_scene import LocalQwenBatchScene
    c = plan['config']; p = c['caption_profile']; model = None; began = time.monotonic()
    try:
        samples = sorted(plan['samples'].items()); size = c['caption_batch_size']
        model = LocalQwenBatchScene(Path(plan['assets']['qwen']['snapshot']), c['device'], p['min_pixels'], p['max_pixels'],
            p['placement'], p.get('gpu_weight_budget'), p.get('cpu_weight_budget'))
        for offset in range(0, len(samples), size):
            batch = samples[offset:offset + size]
            existing = [load_cache(root, plan, 'captions', sample) for _, sample in batch]
            if all(v is not None for v in existing): continue
            if any(v is not None for v in existing):
                raise ValueError('Incomplete native batch; preserve outputs and restart in a fresh root')
            rgbs = [image_rgb(sample) for _, sample in batch]
            results = model.generate(rgbs, p, c['caption_target'])
            if len(results) != len(batch): raise ValueError('Native caption cardinality mismatch')
            for (sid, sample), rgb, result in zip(batch, rgbs, results):
                native = result['answers'][0]['result']
                payload = dict(result=result, source_shape=list(rgb.shape), batch_size=len(batch), precision='bfloat16',
                    seconds=native['generation_time_seconds'], shared_batch_ids=[s for s, _ in batch],
                    timing_scope='Shared native batch, generation only; not isolated end-to-end latency',
                    peak_allocated_bytes=native['peak_allocated_bytes'], peak_reserved_bytes=native['peak_reserved_bytes'])
                seal(cache_path(root, 'captions', sid), dict(contract=contract(plan, 'captions', sample),
                    payload=payload, payload_sha256=digest(payload)))
            print(json.dumps(dict(stage='qwen_captions', done=offset + len(batch), total=len(samples),
                elapsed_seconds=time.monotonic() - began, peak_allocated_bytes=payload['peak_allocated_bytes'],
                peak_reserved_bytes=payload['peak_reserved_bytes'])), flush=True)
    finally:
        if model is not None: model.release()
        gc.collect(); torch.cuda.empty_cache()


def ablation_report(root, plan):
    report(root, plan)
    raw = root / 'metrics.json'
    (root / 'results.json').rename(raw)
    results = read(raw)
    results.update(caption_model='Qwen2-VL-7B caption ablation', caption_model_is_paper_blip2=False,
        caption_profile=plan['config']['caption_profile'], original_metrics_sha256=sha(raw),
        caption_quality_scope='Development-selected profile with separate CAL visual/native qualification; author BLIP2 unknown')
    seal(root / 'results.json', results)
    p = root / 'comparison.md'
    p.write_text(p.read_text().replace('# Paper caption router comparison — approximate reproduction',
        '# Detailed Qwen caption router ablation — approximate reproduction\n\n'
        'Qwen2-VL-7B replaces BLIP-2 explicitly in this ablation. The common 2026-10-29 experts and routing budgets are frozen.'))


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    a = argparse.ArgumentParser(); a.add_argument('--config', type=Path, required=True)
    a.add_argument('--output-root', type=Path, required=True); a.add_argument('--limit', type=int)
    a.add_argument('--stage', choices=['all', 'prepare', 'captions', 'score', 'report'], default='all')
    args = a.parse_args(); root = args.output_root; root.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        c = read(args.config); validate_config(c)
        torch.set_num_threads(c['cpu_threads']); random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
        torch.cuda.manual_seed_all(c['seed']); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        path = root / 'resolved_configuration.json'
        if path.exists():
            plan = read(path); verify(args.config, plan['config_sha256'])
            if plan['sample_limit'] != args.limit:
                raise ValueError('Changed smoke/full population; use a fresh output root')
            for p, h in plan['source_sha256'].items(): verify(p, h)
            for p, h in c['pinned_inputs'].items(): verify(p, h)
        else:
            plan = prepare(c, args.config, args.limit); seal(path, plan)
            for p in c['scientific_source_files']:
                target = root / 'code_snapshot' / p; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(p).read_bytes())
        reuse_t1(root, plan)
        if args.stage in ('all', 'captions'): generate(root, plan)
        if args.stage in ('all', 'score'): score(root, plan)
        if args.stage in ('all', 'report'): ablation_report(root, plan)
    except Exception as e:
        seal(root / ('failure_' + str(time.time_ns()) + '.json'), dict(error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
