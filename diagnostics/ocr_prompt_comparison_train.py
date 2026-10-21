"""Predeclared TRAIN prompt comparison; no changes to models or crop grouping."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from diagnostics import contextual_ocr_conservative_crop_train as original
from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.contextual_ocr_correction import verify, word_scores, lexical_counts, runtime_info, box_iou
from diagnostics.lavis_captioner_candidates import digest, sha, write_json
from diagnostics.conservative_ocr_edit_gates_train import aggregate

def validate_config(cfg):
    keys = ['P1_SHORT', 'P2_RECONSTRUCT', 'P3_CHARACTER_AUDIT', 'P4_EVIDENCE_PRIORITY']
    if (cfg['mode'] != 'APPROXIMATE' or cfg['split'] != 'TRAIN' or cfg['sample_count'] != 229
            or cfg['models'] != ['qwen2.5-1.5b'] or list(cfg['prompts']) != keys
            or cfg['variants'] != [k+'/'+v for k in keys for v in ['ocr_only', 'ocr_caption']]
            or cfg['training'] or cfg['test_loaded']):
        raise ValueError('frozen TRAIN prompt matrix required')

def make_prompt(sample, block, variant, cfg):
    if variant not in cfg['variants']:
        raise ValueError('unexpected prompt condition')
    name, condition = variant.split('/')
    # Same demonstrations, field labels/order and input data; only instruction changes.
    return original.make_prompt(sample, block, condition,
        {'variants': ['ocr_only', 'ocr_caption'], 'instruction': cfg['prompts'][name]})

def prepare(a):
    cfg = json.loads(a.config.read_text())
    validate_config(cfg)
    verify(a.source_root/'results.json', cfg['source_report_sha256'])
    old, samples, source_contract = original.load_pilot(a.source_root)
    checked(a.source_root/'results.json', source_contract)
    if any(cfg[k] != old['config'][k] for k in ['generation', 'max_input_tokens', 'seed', 'models']):
        raise RuntimeError('non-prompt parameters changed')
    verify(a.audit, old['audit_sha256'])
    audit = json.loads(a.audit.read_text())
    if len(audit['per_word']) != 849:
        raise RuntimeError('expected849 TRAIN words')
    log_labels = {}
    for sample in samples:
        rows = [r for r in audit['per_word'] if r['sample_id'] == sample['sample_id']]
        for block in sample['blocks']:
            overlaps = sorted([(box_iou(block['bbox'], r['gt_bbox']), r['gt_index'], r['gt_text'])
                               for r in rows], key=lambda x: (-x[0], x[1]))
            log_labels[sample['sample_id']+':'+str(block['id'])] = {
                'frozen_assignment_gt': [r['gt_text'] for r in rows if r['available'] and r['crop_bbox'] == block['bbox']],
                'best_overlap_gt': overlaps[0][2] if overlaps and overlaps[0][0] > 0 else None,
                'best_overlap_iou': overlaps[0][0] if overlaps else 0.}
    deps = [__file__, str(a.config), 'diagnostics/contextual_ocr_conservative_crop_train.py',
            'diagnostics/contextual_ocr_correction.py', 'diagnostics/caption_routing_train.py',
            'diagnostics/lavis_captioner_candidates.py', 'diagnostics/conservative_ocr_edit_gates_train.py',
            'diagnostics/run_ocr_prompt_comparison_train_host.sh']
    protocol = {'config': cfg, 'models': old['models'],
                'source_model_metadata': old['source_model_metadata'], 'source_contract': old['source_contract'],
                'source_root': str(a.source_root.resolve()), 'source_experiment_contract': source_contract,
                'source_config': old['config'], 'audit_sha256': old['audit_sha256'],
                'samples_sha256': digest(samples), 'log_labels_sha256': digest(log_labels),
                'code_sha256': {p: sha(p) for p in deps}, 'gt_in_query': False,
                'supervision': 'unchanged other-fold TRAIN GT demonstrations; GT log is not model input'}
    contract = digest(protocol)
    seal_write(a.output_dir/'protocol.json', {**protocol, 'contract_sha256': contract})
    seal_write(a.output_dir/'inputs.json', {'contract_sha256': contract, 'samples': samples})
    seal_write(a.output_dir/'gt-log-only.json', {'contract_sha256': contract, 'labels': log_labels})
    prompts = {v: make_prompt(samples[0], samples[0]['blocks'][0], v, cfg) for v in cfg['variants']}
    seal_write(a.output_dir/'prompt-examples.json', {'contract_sha256': contract, 'prompts': prompts})
    print('[prepare CPU] 229 TRAIN /849 GT /1203 crops; 5 prompts x2 contexts; 2406 cached controls +9624 new requests', flush=True)

def load_pilot(root):
    p = checked(root/'protocol.json')
    contract = p['contract_sha256']
    if digest({k: v for k, v in p.items() if k not in ('payload_sha256', 'contract_sha256')}) != contract:
        raise RuntimeError('protocol identity mismatch')
    validate_config(p['config'])
    for path, expected in p['code_sha256'].items():
        verify(path, expected)
    samples = checked(root/'inputs.json', contract)['samples']
    labels = checked(root/'gt-log-only.json', contract)['labels']
    if digest(samples) != p['samples_sha256'] or digest(labels) != p['log_labels_sha256']:
        raise RuntimeError('frozen input mismatch')
    if [s['sample_id'] for s in samples] != [str(i) for i in range(100,329)]:
        raise RuntimeError('TRAIN population changed')
    for s in samples:
        if any(d['source_fold'] == s['fold'] or d['image_sha256'] == s['image_sha256'] for d in s['demonstrations']):
            raise RuntimeError('demonstration leakage')
    return p, samples, contract

def interpret_text(raw, original, truncated=False):
    text = raw.strip()
    reason = None
    if not original.strip():
        return {'status': 'empty_target_preserved', 'text': original, 'error': None}
    if truncated:
        reason = 'token_limit_or_missing_eos'
    elif not text:
        reason = 'empty_output'
    elif any(c.isspace() for c in text):
        reason = 'whitespace_output'
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
    log_labels = checked(a.output_dir/'gt-log-only.json', contract)['labels']
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
            print('GT (diagnostic geometry only):', log_labels[sample['sample_id']+':'+str(block['id'])], flush=True)
    except Exception as exc:
        write_json(a.output_dir/'errors'/f'{a.model}.json', {'type': type(exc).__name__, 'message': str(exc),
                   'oom': isinstance(exc, torch.cuda.OutOfMemoryError)})
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def evaluate(a):
    p, samples, contract = load_pilot(a.output_dir)
    verify(a.audit, p['audit_sha256'])
    audit = json.loads(a.audit.read_text())
    by_gt = {s['sample_id']: [r for r in audit['per_word'] if r['sample_id'] == s['sample_id']] for s in samples}
    source_root = Path(p['source_root'])
    verify(source_root/'results.json', p['config']['source_report_sha256'])
    source = checked(source_root/'results.json', p['source_experiment_contract'])
    variants = ['V1', 'CONTROL/ocr_only', 'CONTROL/ocr_caption']
    if not a.baseline_only:
        variants += p['config']['variants']
    summaries, details, hashes = {}, {}, {}
    model = p['config']['models'][0]
    for variant in variants:
        rows, lexical, crops, times, outputs = [], [], [], [], []
        for sample in samples:
            current, seconds = {}, 0.
            for block in sample['blocks']:
                before, g = block['text'], None
                if variant == 'V1':
                    after, status = before, 'baseline'
                else:
                    if variant.startswith('CONTROL/'):
                        condition = variant.split('/')[1]
                        path = original.output_path(source_root, model, condition, sample['sample_id'], block['id'])
                        rel = str(path.relative_to(source_root))
                        verify(path, source['inference_shards_sha256'][rel])
                        message = original.make_prompt(sample, block, condition, p['source_config'])
                        g = original.check_request(path, p['source_experiment_contract'],
                                                   original.request_identity(sample, block, condition, message))
                        verify(source_root/'models'/(model+'.json'), g['model_metadata_sha256'])
                        old_parse = original.interpret_text(g['raw_output'], before, g['token_limit_abstention'])
                        if any(g[k] != v for k, v in old_parse.items()):
                            raise RuntimeError('cached control interpretation mismatch')
                    else:
                        path = output_path(a.output_dir, model, variant, sample['sample_id'], block['id'])
                        message = make_prompt(sample, block, variant, p['config'])
                        g = check_request(path, contract, request_identity(sample, block, variant, message))
                        verify(a.output_dir/'models'/(model+'.json'), g['model_metadata_sha256'])
                    parsed = interpret_text(g['raw_output'], before, g['token_limit_abstention'])
                    if not variant.startswith('CONTROL/') and any(g[k] != v for k, v in parsed.items()):
                        raise RuntimeError('parser drift')
                    after, status = parsed['text'], parsed['status']
                    seconds += g['latency_seconds']
                    outputs.append({**g, **parsed})
                    hashes[str(path)] = sha(path)
                current[block['id']] = after
                crops.append({'sample_id': sample['sample_id'], 'candidate_id': block['candidate_id'],
                              'original': before, 'text': after, 'changed': after != before, 'reason': status})
            rows.extend(word_scores(sample, by_gt[sample['sample_id']], current))
            lexical.append(lexical_counts([current[b['id']] for b in sample['blocks']],
                                          [r['gt_text'] for r in by_gt[sample['sample_id']]]))
            times.append(seconds)
        summary = aggregate(rows, lexical, crops)
        if summary['gt_words'] != 849 or summary['exact'] != 502+summary['recovered']-summary['damaged']:
            raise RuntimeError('frozen assignment/denominator mismatch')
        if summary['predicted_tokens'] != 1203:
            raise RuntimeError('single-word token budget changed')
        summary.update({'median_seconds_per_image': __import__('statistics').median(times),
                        'total_seconds': sum(times),
                        'peak_allocated_bytes': max((g['peak_allocated_bytes'] for g in outputs), default=None),
                        'peak_reserved_bytes': max((g['peak_reserved_bytes'] for g in outputs), default=None),
                        'decision': 'GO_TRAIN_DIAGNOSTIC_ONLY' if (
                            summary['recovered'] > summary['damaged'] and summary['exact'] > 502
                            and summary['image_lexical_exact_matches'] >= 503) else 'NO-GO'})
        summaries[variant] = summary
        details[variant] = {'per_word': rows, 'per_crop': crops}
    if (summaries['V1']['exact'], summaries['V1']['image_lexical_exact_matches']) != (502,503):
        raise RuntimeError('V1 control mismatch')
    if (summaries['CONTROL/ocr_caption']['exact'], summaries['CONTROL/ocr_caption']['image_lexical_exact_matches']) != (486,493):
        raise RuntimeError('strict single-word control mismatch')
    report = {'contract_sha256': contract, 'classification': 'PRACTICAL_APPROXIMATION',
              'scope': {'split': 'TRAIN', 'images': 229, 'gt_words': 849, 'crops': 1203},
              'summary': summaries, 'details': details, 'inference_sha256': hashes,
              'source_unfiltered_summary': source['summary'], 'table5_accuracy': None,
              'note': 'per-crop correction only; no multi-fragment reconstruction; TRAIN prompt selection is not independent validation',
              'pipeline_promoted': False}
    path = a.output_dir/('baseline.json' if a.baseline_only else 'results.json')
    seal_write(path, report)
    print(json.dumps(summaries, indent=2), flush=True)
    print('JSON', path, 'SHA256', sha(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--audit', type=Path, required=True)
    inf = subs.add_parser('infer')
    inf.add_argument('--model', required=True)
    inf.add_argument('--hf-cache', type=Path, required=True)
    inf.add_argument('--device', required=True)
    ev = subs.add_parser('evaluate')
    ev.add_argument('--audit', type=Path, required=True)
    ev.add_argument('--baseline-only', action='store_true')
    for sub in (prep, inf, ev):
        sub.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
