"""BLIP-2/MPNet/fuzzy vs qualified trained routers, three complete benchmarks.

Only APPROXIMATE mode is supported. Reuses frozen expert outputs; never loads
DeepSolo or fits/selects a model on TEST. Intermediate stages are restartable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import random
import time
import traceback

import numpy as np
from PIL import Image

from recognition.paper_caption_comparison import (accepted_images, evaluate_routing,
    score_transcript, whole_image_counts)
from recognition.common_budget_pipeline import counts, micro
from scoring.paper_image_router import PaperRouterConfig


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seal(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def verify(path, expected):
    if sha(path) != expected: raise ValueError(f'Changed source: {path}')


def prepare(args):
    import torch
    from scripts.check_table5_assets import check_blip2, check_mpnet, check_parseq
    from types import SimpleNamespace
    c = read(args.config)
    if c['mode'] != 'approximate' or c['claim_label'] != 'approximate reproduction':
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    if c['router'] != dict(alpha=.6, beta=.4, tau=.8, primary_gate='paper_selected', lexical_mode='raw'):
        raise ValueError('Paper scoring parameters must be explicit')
    for p, h in c['pinned_inputs'].items(): verify(p, h)
    hf_args = SimpleNamespace(hf_cache=Path(c['hf_cache']))
    assets = dict(blip2=check_blip2(hf_args), mpnet=check_mpnet(hf_args),
                  parseq=check_parseq(SimpleNamespace(parseq_source=Path(c['parseq_source']),
                      parseq_checkpoint=Path(c['parseq_checkpoint']),
                      parseq_source_archive=Path(c['parseq_source_archive']),
                      parseq_source_type='github_commit_archive')))
    for asset in assets.values():
        if 'snapshot' in asset:
            snapshot = Path(asset['snapshot'])
            asset['configuration_file_sha256'] = {str(p.relative_to(snapshot)): sha(p)
                for p in sorted(snapshot.rglob('*')) if p.is_file() and p.suffix in ('.json', '.model')}
    records = {}
    for p in c['record_sources']:
        doc = read(p)
        groups = doc['records']
        if isinstance(next(iter(groups.values())), list):
            groups = {'totaltext': groups}
        for dataset, roles in groups.items():
            for rows in roles.values():
                for row in rows: records[row['sample_id']] = row
    cases = {}; samples = {}
    for run in c['trained_runs']:
        root = Path(run); old_cases = read(root / 'cases.json')
        for name, case in old_cases.items():
            if case['expert'] not in ('unet', 'yolo'): continue
            parts = {r: case['partitions'][r] for r in ('selection', 'calibration', 'test')}
            if args.limit: parts = {r: ids[:args.limit] for r, ids in parts.items()}
            ids = [s for v in parts.values() for s in v]
            observations = read(root / 'data' / (name + '__observations.json'))
            outputs = read(root / 'models' / name / 'observable_outputs.json')['corrected_rows']
            cases[name] = dict(source_run=run, partitions=parts,
                fallback_images=round(len(parts['test']) * case['fallback_count'] / len(case['partitions']['test'])),
                rows={s: outputs[s] for s in ids}, tie_keys={s: observations[s]['tie_key'] for s in ids})
            for sid in ids:
                r = records[sid]
                if r['image_sha256'] != observations[sid]['source_image_sha256']:
                    raise ValueError('Expert/image identity mismatch: ' + sid)
                sample = dict(sample_id=sid, image_path=r['image_path'], image_sha256=r['image_sha256'],
                              image_wh=r['image_wh'], rgb_sha256=observations[sid]['tie_key'])
                if sid in samples and sample != samples[sid]: raise ValueError('Inconsistent shared RGB')
                samples[sid] = sample
    source_files = c['scientific_source_files']
    from importlib.metadata import version
    runtime = dict(python=platform.python_version(), torch=torch.__version__,
                   packages={p: version(p) for p in ('transformers','sentence-transformers','numpy','Pillow','rapidfuzz')},
                   device=c['device'], gpu=torch.cuda.get_device_name(0) if c['device'].startswith('cuda') else None,
                   gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if c['device'].startswith('cuda') else None)
    return dict(config=c, config_sha256=sha(args.config), mode='approximate',
                claim_label='approximate reproduction', result_class='smoke-test result' if args.limit else 'approximate reproduction',
                assets=assets, runtime=runtime, cases=cases, samples=samples,
                source_sha256={p: sha(p) for p in source_files}, test_used_for_selection=False,
                historical_test_exposure=True)


def image_rgb(sample):
    verify(sample['image_path'], sample['image_sha256'])
    rgb = np.array(Image.open(sample['image_path']).convert('RGB'))
    if (list(rgb.shape[1::-1]) != sample['image_wh'] or
            hashlib.sha256(rgb.tobytes()).hexdigest() != sample['rgb_sha256']):
        raise ValueError('Changed RGB pixels/dimensions: ' + sample['sample_id'])
    return rgb


def cache_path(root, stage, sid):
    return root / 'cache' / stage / (hashlib.sha256(sid.encode()).hexdigest() + '.json')


def contract(plan, stage, sample):
    return dict(plan_config_sha256=plan['config_sha256'], stage=stage, sample=sample,
                source_sha256=plan['source_sha256'], runtime=plan['runtime'], assets=plan['assets'])


def load_cache(root, plan, stage, sample):
    path = cache_path(root, stage, sample['sample_id'])
    if not path.exists(): return None
    d = read(path)
    if d['contract'] != contract(plan, stage, sample) or d['payload_sha256'] != digest(d['payload']):
        raise ValueError('Invalid inference cache: ' + str(path))
    return d['payload']


def generate(root, plan, stage):
    import torch
    from caption.blip2_captioner import BLIP2ApproxConfig
    from caption.paper_style_blip2 import LocalPaperBLIP2, generate_profile
    from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer
    from caption.paper_batch_blip2 import generate_batch
    c = plan['config']; model = None; started = time.monotonic()
    try:
        torch.cuda.init()
        if stage == 'captions' and c.get('caption_batch_size', 1) > 1:
            pending = [(sid, sample) for sid, sample in sorted(plan['samples'].items())
                       if load_cache(root, plan, stage, sample) is None]
            a = plan['assets']['blip2']; batch_size = c['caption_batch_size']
            model = LocalPaperBLIP2(BLIP2ApproxConfig(model_id=a['model_id'], revision=a['revision'],
                device=c['device'], dtype='float16', placement='single_device', quantization='none'),
                a['snapshot'], c['caption_max_memory'])
            done = len(plan['samples']) - len(pending)
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset+batch_size];rgbs = [image_rgb(s) for sid,s in batch]
                began=time.monotonic()
                generated=generate_batch(model,rgbs,c['caption_profile'],c['caption_target'])
                elapsed=time.monotonic()-began
                for (sid,sample),rgb,result in zip(batch,rgbs,generated):
                    payload=dict(result=result,source_shape=list(rgb.shape),batch_size=len(batch),seconds=elapsed,
                        precision='float16',peak_allocated_bytes=torch.cuda.max_memory_allocated(c['device']),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(c['device']),
                        shared_batch_ids=[s for s,_ in batch],timing_scope='shared batch; not per-image latency')
                    seal(cache_path(root,stage,sid),dict(contract=contract(plan,stage,sample),
                        payload=payload,payload_sha256=digest(payload)))
                done+=len(batch)
                print(json.dumps(dict(stage=stage,done=done,total=len(plan['samples']),
                    batch_seconds=elapsed,elapsed_seconds=time.monotonic()-started,
                    peak_allocated_bytes=payload['peak_allocated_bytes'],
                    peak_reserved_bytes=payload['peak_reserved_bytes'])),flush=True)
            return
        for i, (sid, sample) in enumerate(sorted(plan['samples'].items()), 1):
            if load_cache(root, plan, stage, sample) is not None: continue
            rgb = image_rgb(sample)
            torch.cuda.reset_peak_memory_stats(c['device'])
            if model is None:
                if stage == 't1':
                    model = ParseqApproxRecognizer(ParseqApproxConfig(Path(c['parseq_source']),
                        Path(c['parseq_checkpoint']), device=c['device'], runtime_source_type='github_commit_archive',
                        runtime_source_archive=Path(c['parseq_source_archive'])))
                else:
                    a = plan['assets']['blip2']
                    model = LocalPaperBLIP2(BLIP2ApproxConfig(model_id=a['model_id'], revision=a['revision'],
                        device=c['device'], dtype='float16', placement='single_device', quantization='none'),
                        a['snapshot'], c['caption_max_memory'])
            began = time.monotonic()
            result = (model.recognize(rgb, input_role='T1_full_image').to_dict() if stage == 't1'
                      else generate_profile(model, rgb, c['caption_profile'], c['caption_target']))
            torch.cuda.synchronize(c['device'])
            payload = dict(result=result, source_shape=list(rgb.shape), batch_size=1,
                seconds=time.monotonic()-began, precision='float32' if stage == 't1' else 'float16',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(c['device']),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(c['device']))
            seal(cache_path(root, stage, sid), dict(contract=contract(plan, stage, sample),
                                                  payload=payload, payload_sha256=digest(payload)))
            if i % 25 == 0 or i == 1 or i == len(plan['samples']):
                print(json.dumps(dict(stage=stage, done=i, total=len(plan['samples']),
                    elapsed_seconds=time.monotonic()-started, sample_id=sid,
                    text=result.get('raw_text', result.get('raw_caption')))), flush=True)
    finally:
        if model is not None: model.release()
        del model; gc.collect(); torch.cuda.empty_cache()


def score(root, plan):
    import torch
    from sentence_transformers import SentenceTransformer
    from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
    cached = {}
    for sid, sample in plan['samples'].items():
        cached[sid] = {stage: load_cache(root, plan, stage, sample) for stage in ('t1', 'captions')}
        if any(v is None for v in cached[sid].values()): raise FileNotFoundError('Missing T1/caption: ' + sid)
    model = SentenceTransformer(plan['assets']['mpnet']['snapshot'], device='cpu', local_files_only=True,
                                model_kwargs={'torch_dtype': torch.float32})
    semantic = MPNetSemanticSimilarity(MPNetApproxConfig(device='cpu', precision='float32'), model=model)
    config = PaperRouterConfig(**plan['config']['router']); results = {}
    for name, case in plan['cases'].items():
        features = {}
        for sid, rows in case['rows'].items():
            t1 = cached[sid]['t1']['result']['raw_text']; caption = cached[sid]['captions']['result']['raw_caption']
            features[sid] = score_transcript(t1, rows, caption, semantic, model.tokenizer, config)
        results[name] = features
        print(json.dumps(dict(stage='score', case=name, images=len(features))), flush=True)
    seal(root / 'per_image_scores.json', results)


def report(root, plan):
    features = read(root / 'per_image_scores.json'); results = {}; table = []
    for name, case in plan['cases'].items():
        source = Path(case['source_run']); old = read(source / 'results.json')['results'][name]
        test = read(source / 'data' / (name + '__test_labels.json'))
        train = read(source / 'data' / (name + '__train_labels.json'))
        ids = case['partitions']['test']; f = {s: features[name][s] for s in ids}
        def branches(ids, labels):
            light = {}; t1 = {}; fallback = {}
            for s in ids:
                rows = case['rows'][s]; label = labels[s]
                flags = [label['ignored'][r['source_row']] for r in rows]
                light[s] = counts(rows, label['ground_truth'], flags)
                t1[s] = whole_image_counts(features[name][s]['texts']['T1'], label['ground_truth'])
                fallback[s] = label['teacher']
            return light, t1, fallback
        light, t1, fallback = branches(ids, test)
        if plan['result_class'] != 'smoke-test result':
            original = read(source / 'test' / (name + '.json'))
            if light != {s:v['light'] for s,v in original['branches'].items()}:
                raise ValueError('Matched light count replay differs: ' + name)
            trained_ids = original['accepted_ids']
        else: trained_ids = [s for s in old['primary']['accepted_ids'] if s in ids]
        # The already qualified gate keeps T3; its own frozen decisions are replayed.
        trained = dict(combined=micro(light[s] if s in trained_ids else fallback[s] for s in ids),
                       deepsolo_images=len(ids)-len(trained_ids), total_images=len(ids),
                       deepsolo_percentage=100*(len(ids)-len(trained_ids))/len(ids), accepted_ids=trained_ids)
        if plan['result_class'] != 'smoke-test result' and trained['combined'] != old['primary']['combined']:
            raise ValueError('Frozen trained-router metric replay differs: ' + name)
        policies = {'trained_lightweight': trained}
        for gate in ('paper_selected', 'joined_t3'):
            scores = {s:f[s]['scores'][gate] for s in ids}; keys = {s:case['tie_keys'][s] for s in ids}
            rules = {'tau0.8': dict(threshold=plan['config']['router']['tau']),
                     'matched_budget': dict(fallback_images=case['fallback_images'])}
            # Calibration uses no labels: match the frozen fallback fraction on CAL.
            cal = case['partitions']['calibration']; target = 1-case['fallback_images']/len(ids)
            ordered = sorted((features[name][s]['scores'][gate] for s in cal
                              if features[name][s]['scores'][gate] is not None), reverse=True)
            keep = round(len(cal)*target)
            cut = ordered[keep-1] if keep and len(ordered)>=keep else None
            if cut is not None: rules['calibrated_threshold'] = dict(threshold=cut)
            for policy, rule in rules.items():
                accepted = accepted_images(scores, keys, **rule)
                point = evaluate_routing(f, light, t1, fallback, accepted, gate)
                point['rule'] = rule
                policies[gate+'__'+policy] = point
        policies['all_deepsolo'] = evaluate_routing(f, light, t1, fallback, [], 'joined_t3')
        policies['all_light'] = dict(combined=micro(light.values()), deepsolo_images=0,
                                   total_images=len(ids), deepsolo_percentage=0.)
        results[name] = dict(policies=policies, light_branches=light, t1_branches=t1,
                             fallback_branches=fallback, calibration_ids=case['partitions']['calibration'])
        for method, point in policies.items():
            table.append(dict(case=name, method=method, f1_percent=100*point['combined']['f1'],
                tp=point['combined']['tp'], fp=point['combined']['fp'], fn=point['combined']['fn'],
                deepsolo_images=point['deepsolo_images'], total_images=point['total_images'],
                deepsolo_percentage=point['deepsolo_percentage']))
    seal(root / 'results.json', dict(mode='approximate', claim_label='approximate reproduction',
        result_class=plan['result_class'], results=results, table=table, test_used_for_selection=False,
        historical_test_exposure=True, actual_new_deepsolo_inference=0,
        metric='casefold + Unicode isalnum; per-image exact multiset micro word F1; no IoU; frozen care/ignore, all images'))
    import csv
    with (root / 'comparison.csv').open('x') as out:
        writer=csv.DictWriter(out,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Paper caption router comparison — approximate reproduction', '',
           'Same frozen experts and DeepSolo; punctuation-free lexical word micro F1. Logical cached routing.', '',
           '| Dataset/expert | Router | F1 % | DeepSolo images/total (%) |',
           '| --- | --- | ---: | --- |']
    for r in table:
        lines.append(f"| {r['case']} | {r['method']} | {r['f1_percent']:.4f} | {r['deepsolo_images']}/{r['total_images']} ({r['deepsolo_percentage']:.2f}%) |")
    (root / 'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage='complete', table=table)), flush=True)


def main():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--stage',choices=['prepare','t1','captions','score','report','all'],default='all')
    p.add_argument('--limit',type=int);args=p.parse_args();root=args.output_root
    root.mkdir(parents=True,exist_ok=True);path=root/'resolved_configuration.json'
    try:
        if not path.exists():
            plan=prepare(args);seal(path,plan)
        else:
            plan=read(path)
            verify(args.config,plan['config_sha256'])
            for path,h in plan['source_sha256'].items():verify(path,h)
            for path,h in plan['config']['pinned_inputs'].items():verify(path,h)
        import torch
        c=plan['config'];torch.set_num_threads(c['cpu_threads']);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if args.stage in ('all','t1'):generate(root,plan,'t1')
        if args.stage in ('all','captions'):generate(root,plan,'captions')
        if args.stage in ('all','score'):score(root,plan)
        if args.stage in ('all','report'):report(root,plan)
    except Exception as e:
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
