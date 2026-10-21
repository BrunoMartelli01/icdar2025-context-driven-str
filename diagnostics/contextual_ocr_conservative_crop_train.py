"""TRAIN-only image-held-out few-shot OCR correction. No gradient updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.contextual_ocr_correction import verify, word_scores, lexical_counts, runtime_info
from diagnostics.contextual_ocr_plaintext_train import load_pilot as load_source
from diagnostics.lavis_captioner_candidates import digest, sha, write_json, atomic

def validate_config(cfg):
    if (cfg['mode'] != 'APPROXIMATE' or cfg['split'] != 'TRAIN'
            or cfg['sample_count'] != 229 or cfg['folds'] != 5
            or cfg['demo_kinds'] != ['keep', 'correct', 'keep', 'correct']
            or cfg['models'] != ['qwen2.5-1.5b']
            or cfg['variants'] != ['ocr_only', 'ocr_caption']
            or cfg['weight_training'] or cfg['test_loaded']):
        raise ValueError('frozen supervised few-shot TRAIN229 protocol required')

def assign_folds(samples, seed):
    # Identical image content cannot occur on both sides of a fold boundary.
    hashes = sorted({s['image_sha256'] for s in samples},
                    key=lambda h: digest({'seed': seed, 'image': h}))
    group_fold = {h: i % 5 for i, h in enumerate(hashes)}
    return {s['sample_id']: group_fold[s['image_sha256']] for s in samples}

def select_demonstrations(samples, rows, folds, heldout, cfg):
    by_id = {s['sample_id']: s for s in samples}
    pool = []
    # Do not inspect held-out GT even when constructing eligibility criteria.
    training_rows = [r for r in rows if folds[r['sample_id']] != heldout]
    associations = Counter((r['sample_id'], tuple(r['crop_bbox']))
                           for r in training_rows if r['available'])
    for r in training_rows:
        if (not r['available'] or r['best_iou'] < cfg['demo_min_iou']
                or r['significant_gt_words_in_crop'] != 1
                or associations[(r['sample_id'], tuple(r['crop_bbox']))] != 1):
            continue
        sample = by_id[r['sample_id']]
        blocks = [b for b in sample['blocks'] if b['bbox'] == r['crop_bbox']]
        if len(blocks) != 1 or blocks[0]['text'] != r['parseq_text']:
            raise RuntimeError('demo source OCR/geometry mismatch')
        block = blocks[0]
        if not block['text'].strip() or any(c in r['gt_text'] for c in '\r\n'):
            continue
        exact = block['text'].casefold() == r['gt_text'].casefold()
        pool.append({'sample_id': sample['sample_id'], 'image_sha256': sample['image_sha256'],
                     'candidate_id': block['candidate_id'], 'gt_index': r['gt_index'],
                     'source_fold': folds[sample['sample_id']], 'kind': 'keep' if exact else 'correct',
                     'target': block['text'], 'answer': block['text'] if exact else r['gt_text'],
                     'gt_text': r['gt_text'], 'source_iou': r['best_iou'],
                     'others': ' | '.join(b['text'] for b in sample['blocks'] if b['id'] != block['id']),
                     'caption': sample['caption']})
    pool.sort(key=lambda d: digest({'seed': cfg['seed'], 'candidate': d['candidate_id']}))
    selected, used = [], set()
    for kind in cfg['demo_kinds']:
        candidates = [d for d in pool if d['kind'] == kind and d['image_sha256'] not in used]
        if not candidates:
            raise RuntimeError(f'not enough unambiguous {kind} demos outside fold {heldout}')
        selected.append(candidates[0])
        used.add(candidates[0]['image_sha256'])
    return selected

def assert_disjoint(sample, demonstrations, folds):
    for demo in demonstrations:
        if (demo['sample_id'] not in folds or demo['source_fold'] != folds[demo['sample_id']]
                or folds[demo['sample_id']] == folds[sample['sample_id']]
                or demo['image_sha256'] == sample['image_sha256']):
            raise RuntimeError('held-out image/GT leaked into demonstrations')

def prepare(a):
    cfg = json.loads(a.config.read_text())
    validate_config(cfg)
    verify(a.source_root/'results.json', cfg['source_report_sha256'])
    old, samples, source_contract = load_source(a.source_root)
    reference = checked(a.source_root/'results.json', source_contract)
    verify(a.audit, old['audit_sha256'])
    audit = json.loads(a.audit.read_text())
    if audit['protocol']['scope'] != {'split': 'TRAIN', 'images': 229, 'gt_words': 849}:
        raise RuntimeError('official TRAIN229/849 required')
    if len(audit['per_word']) != 849:
        raise RuntimeError('GT count mismatch')
    folds = assign_folds(samples, cfg['seed'])
    demos = {str(f): select_demonstrations(samples, audit['per_word'], folds, f, cfg) for f in range(5)}
    selected = []
    for s in samples:
        assert_disjoint(s, demos[str(folds[s['sample_id']])], folds)
        selected.append({**s, 'fold': folds[s['sample_id']], 'demonstrations': demos[str(folds[s['sample_id']])]})
    dependencies = [__file__, 'diagnostics/contextual_ocr_plaintext_train.py',
                    'diagnostics/contextual_ocr_correction.py', 'diagnostics/caption_routing_train.py',
                    'diagnostics/lavis_captioner_candidates.py', str(a.config),
                    'diagnostics/run_contextual_ocr_fewshot_train_host.sh']
    reference_keys = ['baseline'] + [n+'/'+v for n in cfg['models'] for v in cfg['variants']]
    protocol = {'config': cfg, 'config_sha256': sha(a.config),
                'models': {n: old['models'][n] for n in cfg['models']},
                'source_model_metadata': {n: old['source_model_metadata'][n] for n in cfg['models']},
                'source_contract': old['source_contract'], 'audit_sha256': old['audit_sha256'],
                'zero_shot_report': {'path': str((a.source_root/'results.json').resolve()),
                                    'sha256': cfg['source_report_sha256']},
                'zero_shot_summary': {k: reference['summary'][k] for k in reference_keys},
                'sample_ids': [s['sample_id'] for s in selected], 'samples_sha256': digest(selected),
                'folds': folds, 'demonstrations': demos, 'gt_in_generation': True,
                'evaluation_target_gt_in_prompt': False, 'weight_updates': False,
                'code_sha256': {p: sha(p) for p in dependencies}}
    contract = digest(protocol)
    seal_write(a.output_dir/'protocol.json', {**protocol, 'contract_sha256': contract})
    seal_write(a.output_dir/'inputs.json', {'contract_sha256': contract, 'samples': selected})
    crops = sum(len(s['blocks']) for s in samples)
    print(f'[prepare CPU] 229 TRAIN / {crops} crops / {crops*2} requests; 5 image-held-out folds; 4 demos/fold', flush=True)
    for f in range(5):
        labels = [(d['sample_id'], d['target'], d['answer']) for d in demos[str(f)]]
        print(f'[fold {f}] eval_images={list(folds.values()).count(f)} demos={labels}', flush=True)

def load_pilot(root):
    protocol = checked(root/'protocol.json')
    contract = protocol['contract_sha256']
    if digest({k: v for k, v in protocol.items() if k not in ('payload_sha256', 'contract_sha256')}) != contract:
        raise RuntimeError('protocol identity mismatch')
    validate_config(protocol['config'])
    for p, h in protocol['code_sha256'].items():
        verify(p, h)
    samples = checked(root/'inputs.json', contract)['samples']
    if digest(samples) != protocol['samples_sha256'] or [s['sample_id'] for s in samples] != [str(i) for i in range(100, 329)]:
        raise RuntimeError('full TRAIN population changed')
    if assign_folds(samples, protocol['config']['seed']) != protocol['folds']:
        raise RuntimeError('fold assignment changed')
    for s in samples:
        if s['fold'] != protocol['folds'][s['sample_id']] or s['demonstrations'] != protocol['demonstrations'][str(s['fold'])]:
            raise RuntimeError('demonstration assignment changed')
        assert_disjoint(s, s['demonstrations'], protocol['folds'])
    return protocol, samples, contract

def make_prompt(sample, target, variant, cfg):
    if variant not in cfg["variants"]:
        raise ValueError("unexpected condition")
    pieces = [cfg["instruction"]]
    for i, d in enumerate(sample["demonstrations"], 1):
        if d["source_fold"] == sample["fold"] or d["image_sha256"] == sample["image_sha256"]:
            raise RuntimeError("held-out demo leakage")
        pieces.extend([f"\nExample {i}", "PRIMARY OCR: "+d["target"]])
        if variant == "ocr_caption":
            pieces.append("WEAK SCENE CONTEXT: "+d["caption"])
        pieces.append("OUTPUT: "+d["answer"])
    pieces.extend(["\nNow process one independent crop.", "PRIMARY OCR: "+target["text"]])
    if variant == "ocr_caption":
        pieces.append("WEAK SCENE CONTEXT: "+sample["caption"])
    pieces.append("OUTPUT:")
    return "\n".join(pieces)


def interpret_text(raw, original, truncated=False):
    text = raw.strip()
    reason = None
    if not original.strip():
        return {'status': 'empty_target_preserved', 'text': original, 'error': None}
    if truncated:
        reason = 'token_limit_or_missing_eos'
    elif not text:
        reason = 'empty_output'
    elif '\n' in text or '\r' in text:
        reason = 'multiline_output'
    elif text.startswith('```'):
        reason = 'code_fence'
    if reason:
        return {'status': 'target_abstention', 'text': original, 'error': reason}
    # Do not strip quotes/punctuation, infer correctness, or repair model narration.
    return {'status': 'valid', 'text': text, 'error': None}


def decode_ids(ids, architecture, input_tokens):
    return ids[1:] if architecture == 'seq2seq' else ids[input_tokens:]


def output_path(root, model, variant, sid, bid):
    return root/'generations'/model/variant/f'{int(sid):04d}-{bid:02d}.json'


def request_identity(sample, block, variant, prompt):
    return {'sample_id': sample['sample_id'], 'block_id': block['id'], 'candidate_id': block['candidate_id'],
            'variant': variant, 'prompt_sha256': digest(prompt), 'original_text': block['text']}


def check_request(path, contract, request):
    value = checked(path, contract)
    if value['request'] != request:
        raise RuntimeError('cached target/prompt mismatch')
    return value


def infer(a):
    protocol, samples, contract = load_pilot(a.output_dir)
    cfg = protocol['config']
    spec = protocol['models'][a.model]
    meta_path = a.output_dir/'models'/f'{a.model}.json'
    requests = [(s, b, v, make_prompt(s, b, v, cfg)) for s in samples
                for b in s['blocks'] for v in cfg['variants']]
    pending = []
    for s, b, v, prompt in requests:
        path = output_path(a.output_dir, a.model, v, s['sample_id'], b['id'])
        if path.exists():
            g = check_request(path, contract, request_identity(s, b, v, prompt))
            verify(meta_path, g['model_metadata_sha256'])
        else:
            pending.append((s, b, v, prompt))
    if not pending:
        print(f'[reuse] {a.model}: {len(requests)}/{len(requests)} verified, no model load', flush=True)
        return
    source_meta = protocol['source_model_metadata'][a.model]
    verify(source_meta['path'], source_meta['sha256'])
    old = checked(Path(source_meta['path']), protocol['source_contract'])
    snapshot = a.hf_cache/('models--'+spec['model_id'].replace('/', '--'))/'snapshots'/spec['revision']
    for rel, expected in old['snapshot_files_sha256'].items():
        verify(snapshot/rel, expected)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
    device = torch.device(a.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('PAPER GPU stage only; prepare/evaluate are CPU stages')
    torch.cuda.set_device(device)
    torch.cuda.init()
    if spec['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 required; no precision substitution')
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    model = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        cls = AutoModelForSeq2SeqLM if spec['architecture'] == 'seq2seq' else AutoModelForCausalLM
        model, loading = cls.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=getattr(torch, spec['dtype']), output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError(f'checkpoint loading mismatch: {loading}')
        model.to(device).eval()
        torch.cuda.synchronize(device)
        metadata = {'contract_sha256': contract, 'spec': spec, 'source_metadata_sha256': source_meta['sha256'],
                    'runtime': runtime_info(torch, device), 'parameters': sum(p.numel() for p in model.parameters()),
                    'loading_info': loading, 'batch_size': 1}
        if meta_path.exists():
            previous = checked(meta_path, contract)
            if any(previous[k] != value for k, value in metadata.items()):
                raise RuntimeError('model/runtime changed on resume')
        else:
            seal_write(meta_path, {**metadata, 'load_seconds': time.perf_counter()-start,
                       'load_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'load_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
        eos = model.generation_config.eos_token_id
        generation = GenerationConfig(**cfg['generation'], eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id if spec['architecture'] == 'seq2seq' else None)
        eos_ids = eos if isinstance(eos, list) else [eos]
        for number, (sample, block, variant, prompt) in enumerate(pending, len(requests)-len(pending)+1):
            rendered = prompt if spec['architecture'] == 'seq2seq' else tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            encoded = tokenizer(rendered, return_tensors='pt', truncation=False,
                                add_special_tokens=spec['architecture'] == 'seq2seq')
            input_tokens = encoded['input_ids'].shape[1]
            truncated = input_tokens > cfg['max_input_tokens']
            raw, token_count = '', 0
            if block['text'].strip() and not truncated:
                with torch.inference_mode():
                    generated = model.generate(**{k: v.to(device) for k, v in encoded.items()}, generation_config=generation)
                ids = decode_ids(generated[0].detach().cpu().tolist(), spec['architecture'], input_tokens)
                token_count = len(ids)
                truncated = not ids or ids[-1] not in eos_ids
                raw = tokenizer.decode(ids, skip_special_tokens=True)
            parsed = interpret_text(raw, block['text'], truncated)
            torch.cuda.synchronize(device)
            seal_write(output_path(a.output_dir, a.model, variant, sample['sample_id'], block['id']), {
                'contract_sha256': contract, 'request': request_identity(sample, block, variant, prompt),
                'prompt': prompt, 'raw_output': raw, **parsed, 'model_metadata_sha256': sha(meta_path),
                'token_limit_abstention': truncated, 'input_tokens': input_tokens, 'generated_tokens': token_count,
                'latency_seconds': time.perf_counter()-start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
            print(f'[{a.model}] {number}/{len(requests)} sample={sample["sample_id"]} crop={block["id"]} {variant}: {parsed["status"]} {block["text"]!r} -> {parsed["text"]!r}', flush=True)
    except Exception as exc:
        write_json(a.output_dir/'errors'/f'{a.model}.json', {'type': type(exc).__name__, 'message': str(exc),
                   'oom': isinstance(exc, torch.cuda.OutOfMemoryError)})
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def aggregate(word_rows, lexical, image_seconds, outputs):
    matches = sum(r['matches'] for r in lexical)
    predicted = sum(r['predicted'] for r in lexical)
    return {'gt_words': len(word_rows), 'exact': sum(r['exact'] for r in word_rows),
            'fuzzy_ge_0_8': sum(r['fuzzy'] >= .8 for r in word_rows),
            'recovered': sum(r['corrected_error'] for r in word_rows),
            'damaged': sum(r['damaged_correct'] for r in word_rows),
            'lexical_matches': matches, 'lexical_predicted_tokens': predicted,
            'lexical_precision': matches/predicted if predicted else 0.,
            'lexical_surplus': sum(r['surplus'] for r in lexical),
            'target_status': dict(Counter(r['status'] for r in outputs)),
            'changed_targets': sum(r['text'] != r['request']['original_text'] for r in outputs),
            'median_total_correction_seconds_per_image': statistics.median(image_seconds),
            'total_correction_seconds': sum(image_seconds),
            'peak_allocated_bytes': max((r['peak_allocated_bytes'] for r in outputs), default=None),
            'peak_reserved_bytes': max((r['peak_reserved_bytes'] for r in outputs), default=None)}


def evaluate(a):
    protocol, samples, contract = load_pilot(a.output_dir)
    verify(a.audit, protocol['audit_sha256'])
    audit = json.loads(a.audit.read_text())
    if audit['protocol']['scope'] != {'split': 'TRAIN', 'images': 229, 'gt_words': 849}:
        raise RuntimeError('official TRAIN audit required')
    gt = {s['sample_id']: [r for r in audit['per_word'] if r['sample_id'] == s['sample_id']] for s in samples}
    summaries, details, seals = {}, {}, {}
    conditions = [('baseline', None)]
    if not a.baseline_only:
        conditions += [(name, v) for name in protocol['models'] for v in protocol['config']['variants']]
    for model, variant in conditions:
        word_rows, lexical, times, outputs = [], [], [], []
        key = 'baseline' if variant is None else model+'/'+variant
        for sample in samples:
            current = {b['id']: b['text'] for b in sample['blocks']}
            seconds = 0.
            if variant is not None:
                for block in sample['blocks']:
                    prompt = make_prompt(sample, block, variant, protocol['config'])
                    path = output_path(a.output_dir, model, variant, sample['sample_id'], block['id'])
                    g = check_request(path, contract, request_identity(sample, block, variant, prompt))
                    verify(a.output_dir/'models'/f'{model}.json', g['model_metadata_sha256'])
                    parsed = interpret_text(g['raw_output'], block['text'], g['token_limit_abstention'])
                    if any(g[k] != value for k, value in parsed.items()):
                        raise RuntimeError('interpretation drift')
                    current[block['id']] = g['text']
                    seconds += g['latency_seconds']
                    outputs.append(g)
                    seals[str(path.relative_to(a.output_dir))] = sha(path)
            word_rows.extend(word_scores(sample, gt[sample['sample_id']], current))
            lexical.append(lexical_counts([current[b['id']] for b in sample['blocks']], [r['gt_text'] for r in gt[sample['sample_id']]]))
            times.append(seconds)
        summaries[key] = aggregate(word_rows, lexical, times, outputs)
        details[key] = word_rows
    source_rows = [r for s in samples for r in gt[s['sample_id']]]
    if len(source_rows) != 849:
        raise RuntimeError('expected all 849 TRAIN GT words')
    if summaries['baseline']['exact'] != sum(r['exact'] for r in source_rows):
        raise RuntimeError('subset baseline mismatch')
    zero_ref = protocol['zero_shot_report']
    verify(zero_ref['path'], zero_ref['sha256'])
    old = checked(Path(zero_ref['path']))
    comparisons, per_fold = {}, {}
    for key, summary in summaries.items():
        if key == 'baseline':
            continue
        before = old['per_word'][key]
        after = details[key]
        if [(r['sample_id'], r['gt_index']) for r in before] != [(r['sample_id'], r['gt_index']) for r in after]:
            raise RuntimeError('zero-shot word alignment mismatch')
        comparisons[key] = {'zero_shot_exact': old['summary'][key]['exact'],
                            'few_shot_exact': summary['exact'],
                            'improved_words': sum(not b['exact'] and r['exact'] for b, r in zip(before, after)),
                            'worsened_words': sum(b['exact'] and not r['exact'] for b, r in zip(before, after))}
        per_fold[key] = {}
        for fold in range(5):
            rows = [r for r in after if protocol['folds'][r['sample_id']] == fold]
            per_fold[key][str(fold)] = {'gt_words': len(rows), 'exact': sum(r['exact'] for r in rows),
                                      'recovered': sum(r['corrected_error'] for r in rows),
                                      'damaged': sum(r['damaged_correct'] for r in rows)}
    report = {'contract_sha256': contract, 'classification': 'PRACTICAL_APPROXIMATION',
              'scope': {'split': 'TRAIN', 'images': len(samples), 'gt_words': len(source_rows), 'partial_dataset': False},
              'summary': summaries, 'per_word': details, 'inference_shards_sha256': seals,
              'zero_shot_summary': protocol['zero_shot_summary'], 'vs_zero_shot': comparisons,
              'per_fold': per_fold, 'supervision': 'TRAIN GT demonstrations from other image folds; no weight updates',
              'independent_test_result': False,
              'table5_accuracy': None, 'promotion': 'NOT_EVALUATED: TRAIN diagnostic only, no automatic promotion'}
    path = a.output_dir/('baseline.json' if a.baseline_only else 'results.json')
    seal_write(path, report)
    print(json.dumps(summaries, indent=2), flush=True)
    print('JSON', path, 'SHA256', sha(path), flush=True)
    if not a.baseline_only:
        lines = ['# TRAIN229 image-held-out few-shot correction', '',
                 'PRACTICAL_APPROXIMATION. 5 image-held-out folds, TRAIN GT demonstrations; no weight updates. Not Table5.', '',
                 '| Model/input | Exact | Fuzzy | Recovered | Damaged | Target abstentions | Median seconds/image |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for key, row in summaries.items():
            lines.append(f'| {key} | {row["exact"]}/{row["gt_words"]} | {row["fuzzy_ge_0_8"]} | {row["recovered"]} | {row["damaged"]} | {row["target_status"].get("target_abstention", 0)} | {row["median_total_correction_seconds_per_image"]:.3f} |')
        text = '\n'.join(lines)+'\n\nJSON SHA256: '+sha(path)+'\n'
        md = a.output_dir/'results.md'
        if md.exists() and md.read_text() != text:
            raise RuntimeError('existing Markdown report differs')
        if not md.exists():
            atomic(md, text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest='stage', required=True)
    prep = stages.add_parser('prepare')
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--audit', type=Path, required=True)
    inf = stages.add_parser('infer')
    inf.add_argument('--model', required=True)
    inf.add_argument('--hf-cache', type=Path, required=True)
    inf.add_argument('--device', required=True)
    ev = stages.add_parser('evaluate')
    ev.add_argument('--audit', type=Path, required=True)
    ev.add_argument('--baseline-only', action='store_true')
    for sub in (prep, inf, ev):
        sub.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
