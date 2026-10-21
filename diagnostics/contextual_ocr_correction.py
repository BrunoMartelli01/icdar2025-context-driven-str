"""TRAIN-only frozen text correction study. GPU inference is an explicit stage.

prepare/infer use only sealed OCR/caption inputs. Only evaluate reads word GT.
No segmentation, OCR, caption generation, routing, model training or TEST data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics
import time

from diagnostics.caption_routing_train import checked, seal_write, contained
from diagnostics.lavis_captioner_candidates import digest, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if sha(path) != expected:
        raise RuntimeError(f'checksum mismatch: {path}')


def validate_config(cfg):
    if cfg['mode'] != 'APPROXIMATE' or cfg['scope'] != {'split': 'TRAIN', 'images': 229, 'gt_words': 849}:
        raise ValueError('APPROXIMATE TRAIN229/849 only')
    if cfg['variants'] != ['ocr_only', 'ocr_caption', 'caption_only']:
        raise ValueError('unexpected ablation set')
    for spec in cfg['models'].values():
        if not re.fullmatch('[0-9a-f]{40}', spec['revision']):
            raise ValueError('immutable model revision required')
    if cfg['training'] or cfg['test_loaded'] or cfg['fallback_executed']:
        raise ValueError('forbidden stage enabled')


def dependencies():
    return {p: sha(p) for p in [__file__, 'diagnostics/caption_routing_train.py',
                               'diagnostics/lavis_captioner_candidates.py']}


def load_run(root):
    protocol = checked(root/'protocol.json')
    contract = protocol['contract_sha256']
    identity = {k: v for k, v in protocol.items() if k not in ('contract_sha256', 'payload_sha256')}
    if digest(identity) != contract:
        raise RuntimeError('run protocol contract mismatch')
    inputs = checked(root/'inputs.json', contract)
    if inputs['samples_sha256'] != digest(inputs['samples']) or inputs['samples_sha256'] != protocol['samples_sha256']:
        raise RuntimeError('prepared samples mismatch')
    for path, expected in protocol['code_sha256'].items():
        verify(path, expected)
    validate_config(protocol['config'])
    return protocol, inputs['samples'], contract


def prepare(a):
    cfg = read(a.config)
    validate_config(cfg)
    verify(a.source_root/'results.json', cfg['caption_report_sha256'])
    result = checked(a.source_root/'results.json')
    old_protocol = checked(a.source_root/'protocol.json')
    old_contract = result['contract_sha256']
    old_inputs = checked(a.source_root/'inputs.json', old_contract)
    if digest({k: v for k, v in old_protocol.items() if k != 'payload_sha256'}) != old_contract:
        raise RuntimeError('source protocol identity mismatch')
    if digest(old_inputs['samples']) != old_protocol['inputs_sha256'] or result['scope'] != cfg['scope']:
        raise RuntimeError('source inputs/scope mismatch')
    prepared = []
    caption_shas = {}
    for sample in old_inputs['samples']:
        sid = sample['sample_id']
        relative = f'cache/instructblip/{int(sid):04d}.json'
        path = contained(a.source_root, relative)
        verify(path, result['input_shards_sha256'][relative])
        caption = checked(path, old_contract)
        if caption['sample_id'] != sid or caption['image_sha256'] != sample['image_sha256']:
            raise RuntimeError('caption image mismatch')
        model_path = a.source_root/'cache/instructblip/model.json'
        verify(model_path, caption['model_metadata_sha256'])
        caption_shas[sid] = sha(path)
        proposals = {p['candidate_id']: p for p in sample['proposals']}
        ordered = sample['joined']['ordered_blocks']
        if len(proposals) != len(ordered) or {p['candidate_id'] for p in ordered} != set(proposals):
            raise RuntimeError('reading-order crop identity mismatch')
        blocks = []
        for i, block in enumerate(ordered):
            if block != proposals[block['candidate_id']]:
                raise RuntimeError('reading-order bbox/text mismatch')
            blocks.append({'id': i, 'candidate_id': block['candidate_id'],
                           'bbox': block['bbox'], 'text': block['text']})
        prepared.append({'sample_id': sid, 'image_sha256': sample['image_sha256'],
                         'blocks': blocks, 'caption': caption['text'],
                         'caption_generation_seconds': caption.get('latency_seconds')})
    if {s['sample_id'] for s in prepared} != {str(i) for i in range(100, 329)}:
        raise RuntimeError('official TRAIN IDs required')
    if len(prepared) != 229 or sum(len(s['blocks']) for s in prepared) != 1203:
        raise RuntimeError('frozen population mismatch')
    protocol = {'config': cfg, 'config_sha256': sha(a.config), 'code_sha256': dependencies(),
                'source_report_sha256': sha(a.source_root/'results.json'),
                'source_inputs_sha256': sha(a.source_root/'inputs.json'),
                'caption_shards_sha256': caption_shas, 'samples_sha256': digest(prepared),
                'gt_in_generation': False, 'source_localizer': 'PRACTICAL_REMAP_V1'}
    contract = digest(protocol)
    seal_write(a.output_dir/'protocol.json', {**protocol, 'contract_sha256': contract})
    seal_write(a.output_dir/'inputs.json', {'contract_sha256': contract,
               'samples_sha256': digest(prepared), 'samples': prepared})
    print('[prepare CPU] verified229 TRAIN images,1203 V1 crops,229 InstructBLIP captions; no GT in inputs', flush=True)
    print('INPUTS SHA256', sha(a.output_dir/'inputs.json'), flush=True)


def build_prompt(sample, variant, cfg):
    if variant not in cfg['variants']:
        raise ValueError('unknown condition')
    if variant == 'caption_only':
        return cfg['caption_instruction']+'\nDATA:\n'+json.dumps({'caption': sample['caption']}, ensure_ascii=False)
    # Whitelist fields: GT, source candidate IDs, image/sample IDs never enter prompts.
    data = {'blocks': [{k: b[k] for k in ('id', 'bbox', 'text')} for b in sample['blocks']]}
    if variant == 'ocr_caption':
        data['caption'] = sample['caption']
    return cfg['correction_instruction']+'\nDATA:\n'+json.dumps(data, ensure_ascii=False)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def parse_output(raw, sample, variant):
    raw = raw.strip()
    if raw.startswith('```json\n') and raw.endswith('\n```'):
        raw = raw[8:-4]
    obj = json.loads(raw, object_pairs_hook=unique_object)
    if variant == 'caption_only':
        if not isinstance(obj, dict) or set(obj) != {'text'} or not isinstance(obj['text'], str):
            raise ValueError('caption-only output must contain exactly one text string')
        return obj
    if not isinstance(obj, dict) or set(obj) != {'blocks'} or not isinstance(obj['blocks'], list):
        raise ValueError('expected blocks JSON object')
    original = {b['id']: b['text'] for b in sample['blocks']}
    seen = {}
    for item in obj['blocks']:
        if not isinstance(item, dict) or set(item) != {'id', 'text', 'action'}:
            raise ValueError('invalid block schema')
        i, text, action = item['id'], item['text'], item['action']
        if type(i) is not int or i not in original or i in seen or not isinstance(text, str):
            raise ValueError('unknown/duplicate id or non-text')
        if action not in ('keep', 'correct', 'uncertain'):
            raise ValueError('invalid action')
        if action in ('keep', 'uncertain') and text != original[i]:
            raise ValueError('keep/uncertain changed text')
        if action == 'correct' and (not original[i].strip() or not text.strip() or text == original[i]):
            raise ValueError('invalid correction or invented originally empty text')
        seen[i] = item
    if set(seen) != set(original):
        raise ValueError('missing IDs')
    return {'blocks': [seen[b['id']] for b in sample['blocks']]}


def interpret(raw, sample, variant, truncated=False):
    try:
        if truncated:
            raise ValueError('token limit exceeded or generation ended without EOS')
        return {'status': 'valid', 'parsed': parse_output(raw, sample, variant), 'error': None}
    except (ValueError, TypeError) as exc:
        # Explicit predeclared abstention, never a swallowed model/runtime failure.
        fallback = {'text': ''} if variant == 'caption_only' else {
            'blocks': [{'id': b['id'], 'text': b['text'], 'action': 'uncertain'} for b in sample['blocks']]}
        return {'status': 'invalid_output_abstention', 'parsed': fallback, 'error': str(exc)}


def inference_path(root, model, variant, sid):
    return root/'generations'/model/variant/f'{int(sid):04d}.json'


def runtime_info(torch, device):
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': importlib.metadata.version('transformers'),
            'tokenizers': importlib.metadata.version('tokenizers'), 'device': str(device),
            'gpu': torch.cuda.get_device_name(device), 'cuda': torch.version.cuda,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


def infer(a):
    protocol, samples, contract = load_run(a.output_dir)
    cfg = protocol['config']
    if a.model not in cfg['models']:
        raise ValueError('model not in frozen configuration')
    spec = cfg['models'][a.model]
    requests = [(s, v, build_prompt(s, v, cfg)) for s in samples for v in cfg['variants']]
    metadata_path = a.output_dir/'models'/f'{a.model}.json'
    pending = []
    for sample, variant, prompt in requests:
        path = inference_path(a.output_dir, a.model, variant, sample['sample_id'])
        if path.exists():
            saved = checked(path, contract)
            if (saved['prompt_sha256'] != digest(prompt) or saved['sample_id'] != sample['sample_id']
                    or saved['variant'] != variant or saved['model'] != a.model):
                raise RuntimeError('cached prompt identity mismatch')
            verify(metadata_path, saved['model_metadata_sha256'])
        else:
            pending.append((sample, variant, prompt))
    if not pending:
        print(f'[reuse] {a.model}: {len(requests)}/{len(requests)} verified; no model load', flush=True)
        return
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
    device = torch.device(a.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU runner is for PAPER only; use prepare/evaluate for CPU')
    torch.cuda.set_device(device)
    torch.cuda.init()
    if spec['dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('frozen BF16 model requires BF16-capable GPU; no precision substitution')
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    snapshot = Path(snapshot_download(spec['model_id'], revision=spec['revision'], cache_dir=a.hf_cache,
        local_files_only=not a.allow_download,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=2))
    if snapshot.name != spec['revision']:
        raise RuntimeError('snapshot revision mismatch')
    files = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    weights = [p for p in snapshot.rglob('*.safetensors')]
    if not weights:
        raise RuntimeError('no safetensors checkpoint in pinned snapshot')
    for path in weights:
        blob = path.resolve().name
        if re.fullmatch('[0-9a-f]{64}', blob) and sha(path) != blob:
            raise RuntimeError('HF weight blob SHA256 mismatch')
    model = None
    try:
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        cls = AutoModelForSeq2SeqLM if spec['architecture'] == 'seq2seq' else AutoModelForCausalLM
        model, loading = cls.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=getattr(torch, spec['dtype']), output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise RuntimeError(f'checkpoint loading mismatch: {loading}')
        model.to(device).eval()
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter()-start
        metadata = {'contract_sha256': contract, 'spec': spec, 'snapshot_files_sha256': files,
                    'runtime': runtime_info(torch, device), 'parameters': sum(p.numel() for p in model.parameters()),
                    'loading_info': loading, 'batch_size': 1, 'quantization': None}
        if metadata_path.exists():
            old = checked(metadata_path, contract)
            for k, v in metadata.items():
                if old[k] != v:
                    raise RuntimeError(f'model/runtime drift on resume: {k}')
        else:
            seal_write(metadata_path, {**metadata, 'load_seconds': load_seconds,
                       'load_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'load_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
        metadata_sha = sha(metadata_path)
        eos = model.generation_config.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        generation = GenerationConfig(**cfg['generation'], eos_token_id=eos, pad_token_id=pad,
                                      decoder_start_token_id=model.config.decoder_start_token_id
                                      if spec['architecture'] == 'seq2seq' else None)
        for number, (sample, variant, prompt) in enumerate(pending, len(requests)-len(pending)+1):
            rendered = prompt if spec['architecture'] == 'seq2seq' else tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            encoded = tokenizer(rendered, return_tensors='pt', truncation=False,
                                add_special_tokens=spec['architecture'] == 'seq2seq')
            input_tokens = encoded['input_ids'].shape[1]
            raw, generated_tokens, truncated = '', 0, input_tokens > cfg['max_input_tokens']
            if not truncated:
                with torch.inference_mode():
                    generated = model.generate(**{k: v.to(device) for k, v in encoded.items()},
                                               generation_config=generation)
                ids = generated[0].detach().cpu().tolist()
                ids = ids[1:] if spec['architecture'] == 'seq2seq' else ids[input_tokens:]
                generated_tokens = len(ids)
                eos_ids = eos if isinstance(eos, list) else [eos]
                truncated = not ids or ids[-1] not in eos_ids
                raw = tokenizer.decode(ids, skip_special_tokens=True)
            parsed = interpret(raw, sample, variant, truncated)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-start
            seal_write(inference_path(a.output_dir, a.model, variant, sample['sample_id']), {
                'contract_sha256': contract, 'model': a.model, 'variant': variant,
                'sample_id': sample['sample_id'], 'prompt_sha256': digest(prompt), 'prompt': prompt,
                'model_metadata_sha256': metadata_sha, 'raw_output': raw, **parsed,
                'input_tokens': input_tokens, 'generated_tokens': generated_tokens,
                'token_limit_abstention': truncated, 'latency_seconds': seconds,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
            print(f'[{a.model}] {number}/{len(requests)} {variant} sample={sample["sample_id"]} {parsed["status"]}', flush=True)
    except Exception as exc:
        write_json(a.output_dir/'errors'/f'{a.model}.json', {'type': type(exc).__name__, 'message': str(exc),
                   'oom': isinstance(exc, torch.cuda.OutOfMemoryError), 'contract_sha256': contract})
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def box_iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def lexical_counts(predicted, expected):
    pred = Counter(' '.join(predicted).casefold().split())
    gt = Counter(t.casefold() for t in expected)
    matches = sum((pred & gt).values())
    return {'matches': matches, 'predicted': sum(pred.values()), 'gt': sum(gt.values()),
            'surplus': sum((pred-gt).values()), 'missing': sum((gt-pred).values())}


def word_scores(sample, gt_rows, corrected):
    from rapidfuzz.fuzz import ratio
    out = []
    for gt in gt_rows:
        # Keep historical maximum-IoU tie choice fixed through original crop_bbox.
        matches = [b for b in sample['blocks'] if b['bbox'] == gt['crop_bbox']] if gt['available'] else []
        if gt['available'] and len(matches) != 1:
            raise RuntimeError('ambiguous frozen crop assignment')
        block = matches[0] if matches else None
        before = block['text'] if block else None
        if before != gt['parseq_text']:
            raise RuntimeError('baseline OCR differs from audit')
        after = corrected[block['id']] if block else None
        score = ratio(after.casefold(), gt['gt_text'].casefold())/100 if after is not None else 0.
        exact = after is not None and after.casefold() == gt['gt_text'].casefold()
        out.append({'sample_id': sample['sample_id'], 'gt_index': gt['gt_index'], 'gt_text': gt['gt_text'],
                    'candidate_id': block['candidate_id'] if block else None,
                    'before': before, 'after': after, 'exact': exact, 'fuzzy': score,
                    'corrected_error': not gt['exact'] and exact, 'damaged_correct': gt['exact'] and not exact})
    return out


def summarize_evaluation(word_rows, lexical, generations):
    total_gt = sum(x['gt'] for x in lexical)
    matched = sum(x['matches'] for x in lexical)
    predicted = sum(x['predicted'] for x in lexical)
    timing = [g['latency_seconds'] for g in generations]
    return {
        'historical_gt_association': None if word_rows is None else {
            'gt_words': len(word_rows), 'exact': sum(r['exact'] for r in word_rows),
            'available_predictions': sum(r['after'] is not None for r in word_rows),
            'exact_rate': sum(r['exact'] for r in word_rows)/len(word_rows),
            'fuzzy_ge_0_8': sum(r['fuzzy'] >= .8 for r in word_rows),
            'corrected_errors': sum(r['corrected_error'] for r in word_rows),
            'damaged_correct': sum(r['damaged_correct'] for r in word_rows)},
        'lexical_multiset': {'gt_words': total_gt, 'matches': matched, 'predicted_tokens': predicted,
            'recall': matched/total_gt, 'precision': matched/predicted if predicted else 0.,
            'surplus_tokens': sum(x['surplus'] for x in lexical), 'missing_tokens': sum(x['missing'] for x in lexical)},
        'valid_images': sum(g['status'] == 'valid' for g in generations),
        'invalid_output_images': sum(g['status'] != 'valid' for g in generations),
        'token_limit_images': sum(g['token_limit_abstention'] for g in generations),
        'correction_latency_seconds_total': sum(timing),
        'correction_latency_seconds_median': statistics.median(timing) if timing else None,
        'peak_allocated_bytes': max((g['peak_allocated_bytes'] for g in generations), default=None),
        'peak_reserved_bytes': max((g['peak_reserved_bytes'] for g in generations), default=None),
    }


def evaluate(a):
    protocol, samples, contract = load_run(a.output_dir)
    cfg = protocol['config']
    verify(a.audit, cfg['audit_sha256'])
    audit = read(a.audit)
    if audit['protocol']['scope'] != cfg['scope']:
        raise RuntimeError('evaluation scope mismatch')
    gt = {s['sample_id']: [r for r in audit['per_word'] if r['sample_id'] == s['sample_id']] for s in samples}
    baseline_rows, baseline_lexical = [], []
    for sample in samples:
        baseline_rows.extend(word_scores(sample, gt[sample['sample_id']], {b['id']: b['text'] for b in sample['blocks']}))
        baseline_lexical.append(lexical_counts([b['text'] for b in sample['blocks']], [r['gt_text'] for r in gt[sample['sample_id']]]))
    baseline = summarize_evaluation(baseline_rows, baseline_lexical, [])
    historical = baseline['historical_gt_association']
    if (historical['gt_words'], historical['exact'], historical['fuzzy_ge_0_8']) != (849, 502, 578):
        raise RuntimeError('502/849 exact and578 fuzzy baseline not reproduced')
    if a.baseline_only:
        seal_write(a.output_dir/'baseline.json', {'contract_sha256': contract, 'baseline': baseline})
        print('[baseline CPU] 502/849 exact,578/849 fuzzy PASS; no GPU model run', flush=True)
        return
    summaries, details, seals = {}, {}, {}
    for model in cfg['models']:
        for variant in cfg['variants']:
            key = f'{model}/{variant}'
            word_rows, lexical, generations, image_details = [], [], [], []
            for sample in samples:
                path = inference_path(a.output_dir, model, variant, sample['sample_id'])
                g = checked(path, contract)
                verify(a.output_dir/'models'/f'{model}.json', g['model_metadata_sha256'])
                prompt = build_prompt(sample, variant, cfg)
                if (g['prompt_sha256'] != digest(prompt) or g['sample_id'] != sample['sample_id']
                        or g['variant'] != variant or g['model'] != model):
                    raise RuntimeError('inference identity mismatch')
                reparsed = interpret(g['raw_output'], sample, variant, g['token_limit_abstention'])
                if any(g[k] != reparsed[k] for k in reparsed):
                    raise RuntimeError('output interpretation drift')
                generations.append(g)
                seals[str(path.relative_to(a.output_dir))] = sha(path)
                if variant == 'caption_only':
                    texts = [g['parsed']['text']]
                else:
                    corrected = {b['id']: b['text'] for b in g['parsed']['blocks']}
                    word_rows.extend(word_scores(sample, gt[sample['sample_id']], corrected))
                    texts = [corrected[b['id']] for b in sample['blocks']]
                counts = lexical_counts(texts, [r['gt_text'] for r in gt[sample['sample_id']]])
                lexical.append(counts)
                image_details.append({'sample_id': sample['sample_id'], 'status': g['status'], 'lexical': counts})
            summaries[key] = summarize_evaluation(None if variant == 'caption_only' else word_rows, lexical, generations)
            details[key] = {'per_word': word_rows, 'per_image': image_details}
    report = {'contract_sha256': contract, 'classification': cfg['classification'], 'scope': cfg['scope'],
              'table5_accuracy': None, 'baseline': baseline, 'variants': summaries, 'details': details,
              'inference_shards_sha256': seals, 'gt_audit_sha256': sha(a.audit),
              'metric_notes': 'Historical per-GT fixed crop association is not detection precision. Lexical casefold whitespace multiset caps duplicates; caption-only has no spatial score. Surplus tokens are not proof of hallucination. All invalid outputs included as explicit abstentions. No fallback or end-to-end speed measured.',
              'promotion': 'NOT_AUTOMATIC: compare net recovery, damage, surplus and total system costs before any routing change'}
    seal_write(a.output_dir/'results.json', report)
    lines = ['# TRAIN contextual OCR correction', '',
             'PRACTICAL_APPROXIMATION; no Table5 accuracy, no model training, no TEST.', '',
             'Fixed V1 baseline:502/849 exact;578/849 fuzzy. Invalid outputs explicitly preserve OCR,',
             'are counted below, and do not leave the denominator. Caption-only has no spatial score.', '',
             '| Model / input | Exact /849 | Fuzzy /849 | Recovered | Damaged | Invalid /229 | Lexical matches /849 | Median correction seconds |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, val in summaries.items():
        h = val['historical_gt_association']
        nums = [h[k] if h else 'n/a' for k in ('exact', 'fuzzy_ge_0_8', 'corrected_errors', 'damaged_correct')]
        lines.append('| '+key+' | '+' | '.join(map(str, nums))+f' | {val["invalid_output_images"]} | {val["lexical_multiset"]["matches"]} | {val["correction_latency_seconds_median"]:.3f} |')
    lines += ['', 'Net exact change = recovered minus damaged. Lexical matches do not enforce bbox correspondence.',
              'Latency is the correction stage only, not end-to-end. No automatic promotion.', '',
              'JSON SHA256: '+sha(a.output_dir/'results.json'), '']
    markdown = '\n'.join(lines)
    md_path = a.output_dir/'results.md'
    if md_path.exists() and md_path.read_text() != markdown:
        raise RuntimeError('existing Markdown report differs; preserved')
    if not md_path.exists():
        from diagnostics.lavis_captioner_candidates import atomic
        atomic(md_path, markdown)
    print(json.dumps({'baseline': baseline, 'variants': summaries}, indent=2), flush=True)
    print('JSON', a.output_dir/'results.json', 'SHA256', sha(a.output_dir/'results.json'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--source-root', type=Path, required=True)
    inf = sub.add_parser('infer')
    inf.add_argument('--model', required=True)
    inf.add_argument('--hf-cache', type=Path, required=True)
    inf.add_argument('--device', required=True)
    inf.add_argument('--allow-download', action='store_true')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--audit', type=Path, required=True)
    ev.add_argument('--baseline-only', action='store_true')
    for parser in (prep, inf, ev):
        parser.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    {'prepare': prepare, 'infer': infer, 'evaluate': evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
