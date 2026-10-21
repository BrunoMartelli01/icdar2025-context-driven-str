"""Frozen, non-oracle ALL_WORDS geometry/caption ablation. GPU is manual only."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import time

from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.contextual_ocr_correction import verify, lexical_counts, runtime_info
from diagnostics.lavis_captioner_candidates import digest, sha, write_json


def validate_config(c):
    if (c['mode'] != 'APPROXIMATE' or c['split'] != 'TRAIN'
            or (c['images'], c['gt_words'], c['crops']) != (229, 849, 1203)
            or any(c[k] for k in ('training', 'test_loaded', 'oracle_grouping'))
            or c['model_key'] != 'qwen2.5-1.5b'
            or c['representations'] != ['BBOX_ONLY', 'EXPLICIT_GEOMETRY']
            or c['captions'] != ['none', 'instructblip', 'florence2-base-ft', 'blip2-vitl']):
        raise ValueError('Only the predeclared non-oracle TRAIN matrix is supported')


def variants(c):
    return [r+'/'+caption for r in c['representations'] for caption in c['captions']]


def bbox_geometry(blocks, lines, decimals):
    """Continuous observable features. No word decisions, text or GT consulted."""
    ids = [b['candidate_id'] for b in blocks]
    flattened = [i for line in lines for i in line]
    if len(set(ids)) != len(ids) or Counter(flattened) != Counter(ids):
        raise ValueError('Invalid crop IDs or frozen line membership')
    by_line = {cid: n for n, line in enumerate(lines) for cid in line}
    for b in blocks:
        x = b['bbox']
        if len(x) != 4 or not all(math.isfinite(v) for v in x) or x[2] <= x[0] or x[3] <= x[1]:
            raise ValueError('Nonpositive or nonfinite bbox')
    rows = []
    for i, a in enumerate(blocks):
        x = a['bbox']; wa, ha = x[2]-x[0], x[3]-x[1]
        for b in blocks[i+1:]:
            y = b['bbox']; wb, hb = y[2]-y[0], y[3]-y[1]
            ox = max(0, min(x[2], y[2])-max(x[0], y[0]))
            oy = max(0, min(x[3], y[3])-max(x[1], y[1]))
            inter = ox*oy
            numeric = {
                'intersection_fraction_A': inter/(wa*ha),
                'intersection_fraction_B': inter/(wb*hb),
                'horizontal_gap_over_mean_height': max(0, max(x[0], y[0])-min(x[2], y[2]))/((ha+hb)/2),
                'vertical_overlap_over_min_height': oy/min(ha, hb),
                'center_y_distance_over_mean_height': abs((x[1]+x[3]-y[1]-y[3])/2)/((ha+hb)/2),
                'height_ratio_max_over_min': max(ha, hb)/min(ha, hb),
                'x_overlap_over_min_width': ox/min(wa, wb),
            }
            rows.append({'A': a['candidate_id'], 'B': b['candidate_id'],
                         'A_inside_B': inter == wa*ha, 'B_inside_A': inter == wb*hb,
                         'same_frozen_line': by_line[a['candidate_id']] == by_line[b['candidate_id']],
                         'A_strictly_left_of_B': x[2] <= y[0],
                         'B_strictly_left_of_A': y[2] <= x[0],
                         **{k: round(v, decimals) for k, v in numeric.items()}})
    return rows


def make_prompt(sample, variant, cfg):
    if variant not in variants(cfg):
        raise ValueError('Unregistered condition')
    representation, caption = variant.split('/')
    # Explicit whitelist: even if a caller adds GT fields they cannot reach Qwen.
    payload = {'observations_in_frozen_reading_order': [
        {'id': b['candidate_id'], 'bbox': b['bbox'], 'ocr': b['text']} for b in sample['blocks']]}
    if representation == 'EXPLICIT_GEOMETRY':
        payload['geometry'] = sample['geometry']
        payload['geometry_note'] = ('Ratios are measurements, not correctness scores. '
                                    'Frozen line membership is fallible. Overlap is not extra text.')
    if caption != 'none':
        payload['scene_caption_fallible'] = sample['captions'][caption]
    return cfg['instruction']+'\nOBSERVATIONS:\n'+json.dumps(payload, ensure_ascii=False, separators=(',', ':'))+'\nTRANSCRIPTION:'


def interpret_text(raw, baseline, ended=True):
    text = raw.strip()
    reason = None
    if not ended:
        reason = 'token_limit_or_missing_eos'
    elif not text or text.casefold() in ('<abstain>', 'abstain'):
        reason = 'empty_or_abstention'
    elif text.startswith(('```', '{', '[')):
        reason = 'structured_output'
    elif re.match(r'^(?:answer|transcription|reconstructed\s+text|explanation|reasoning)\s*:', text, re.I):
        reason = 'output_label'
    elif re.match(r'^(?:here (?:is|are)|sure[,!]\s|to reconstruct\b|the image (?:shows|features|depicts)\b|based on (?:the|these)\b)', text, re.I):
        reason = 'narration'
    elif re.search(r'(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)', text):
        reason = 'list_format'
    # Spaces and newlines are equally valid. Do not strip punctuation or salvage substrings.
    normalized = ' '.join(text.split())
    return {'status': 'baseline_fallback' if reason else 'valid', 'reason': reason,
            'text': baseline if reason else normalized}


def validate_population(samples):
    if [s['sample_id'] for s in samples] != [str(i) for i in range(100, 329)]:
        raise RuntimeError('Official TRAIN229 IDs/order required; TEST forbidden')
    if sum(len(s['blocks']) for s in samples) != 1203:
        raise RuntimeError('Frozen crop count changed')


def prepare(a):
    cfg = json.loads(a.config.read_text()); validate_config(cfg)
    verify(a.routing_root/'results.json', cfg['routing_report_sha256'])
    routing = checked(a.routing_root/'results.json')
    if routing['scope'] != {'split': 'TRAIN', 'images': 229, 'gt_words': 849}:
        raise RuntimeError('Invalid caption source scope')
    inputs = checked(a.routing_root/'inputs.json', routing['contract_sha256'])
    source_protocol = checked(a.routing_root/'protocol.json')
    if digest({k:v for k,v in source_protocol.items() if k != 'payload_sha256'}) != routing['contract_sha256']:
        raise RuntimeError('Caption source protocol mismatch')
    if digest(inputs['samples']) != source_protocol['inputs_sha256']:
        raise RuntimeError('Caption source inputs mismatch')
    samples, sources = [], {}
    for s in inputs['samples']:
        blocks = s['joined']['ordered_blocks']
        if sorted(blocks, key=lambda b:b['candidate_id']) != sorted(s['proposals'], key=lambda b:b['candidate_id']):
            raise RuntimeError('Reading order lost or changed crops')
        captions = {}
        for name in cfg['captions'][1:]:
            rel = f'cache/{name}/{int(s["sample_id"]):04d}.json'
            f = a.routing_root/rel
            verify(f, routing['input_shards_sha256'][rel])
            row = checked(f, routing['contract_sha256'])
            if row['sample_id'] != s['sample_id'] or row['image_sha256'] != s['image_sha256']:
                raise RuntimeError('Caption/image association mismatch')
            if not row['text'].strip():
                raise RuntimeError('Empty caption')
            verify(a.routing_root/'cache'/name/'model.json', row['model_metadata_sha256'])
            captions[name] = row['text']; sources[str(f.resolve())] = sha(f)
        samples.append({'sample_id': s['sample_id'], 'image_sha256': s['image_sha256'],
                        'blocks': blocks, 'baseline': ' '.join(b['text'] for b in blocks),
                        'captions': captions,
                        'geometry': bbox_geometry(blocks, s['joined']['line_candidate_ids'], cfg['feature_decimals'])})
    validate_population(samples)
    verify(a.model_source/'results.json', cfg['model_source_report_sha256'])
    old = checked(a.model_source/'protocol.json')
    checked(a.model_source/'results.json', old['contract_sha256'])
    spec = old['models'][cfg['model_key']]; meta = old['source_model_metadata'][cfg['model_key']]
    if spec['architecture'] != 'causal' or spec['dtype'] != 'bfloat16':
        raise RuntimeError('Qualified causal BF16 model required')
    verify(meta['path'], meta['sha256']); checked(Path(meta['path']), old['source_contract'])
    # GT is isolated from observable samples/prompts. Used only for logging and evaluation.
    verify(a.audit, cfg['audit_sha256'])
    audit = json.loads(a.audit.read_text()); labels = {}
    for s in samples:
        rows = [w for w in audit['per_word'] if w['sample_id'] == s['sample_id']]
        for w in rows:
            if w['available']:
                matching = [b for b in s['blocks'] if b['bbox'] == w['crop_bbox']]
                if len(matching) != 1 or matching[0]['text'] != w['parseq_text']:
                    raise RuntimeError('Frozen V1 audit mismatch')
        labels[s['sample_id']] = [w['gt_text'] for w in rows]
    if sum(map(len, labels.values())) != 849:
        raise RuntimeError('GT denominator changed')
    baseline = sum(lexical_counts([s['baseline']], labels[s['sample_id']])['matches'] for s in samples)
    if baseline != 503:
        raise RuntimeError('Expected 503 baseline image-lexical matches')
    dependencies = [__file__, str(a.config), 'diagnostics/run_geometry_context_train_host.sh',
                    'diagnostics/caption_routing_train.py', 'diagnostics/contextual_ocr_correction.py',
                    'diagnostics/lavis_captioner_candidates.py']
    p = {'config': cfg, 'model': spec, 'model_metadata': meta,
         'model_source_contract': old['source_contract'], 'samples_sha256': digest(samples),
         'labels_sha256': digest(labels), 'caption_source_sha256': sources,
         'caption_specs': {k: source_protocol['specs'][k] for k in cfg['captions'][1:]},
         'code_sha256': {str(Path(f).resolve()): sha(f) for f in dependencies},
         'gt_in_inference': False, 'demonstrations': [], 'table5_accuracy': None}
    contract = digest(p)
    seal_write(a.output_dir/'protocol.json', {**p, 'contract_sha256': contract})
    seal_write(a.output_dir/'inputs.json', {'contract_sha256': contract, 'samples': samples})
    seal_write(a.output_dir/'evaluation-labels.json', {'contract_sha256': contract, 'labels': labels})
    seal_write(a.output_dir/'prompt-examples.json', {'contract_sha256': contract,
               'prompts': {v: make_prompt(samples[0], v, cfg) for v in variants(cfg)}})
    print('[prepare CPU] TRAIN229 /849 GT /1203 frozen crops; 8 conditions, 1832 requests; 687 cached captions; baseline503', flush=True)


def load_run(root):
    p = checked(root/'protocol.json'); contract = p['contract_sha256']
    if digest({k:v for k,v in p.items() if k not in ('contract_sha256','payload_sha256')}) != contract:
        raise RuntimeError('Protocol drift')
    validate_config(p['config'])
    for f,h in p['code_sha256'].items(): verify(f,h)
    samples = checked(root/'inputs.json', contract)['samples']
    if digest(samples) != p['samples_sha256']: raise RuntimeError('Input drift')
    validate_population(samples)
    return p, samples, contract


def path_for(root, sid, variant):
    return root/'generations'/variant/f'{int(sid):04d}.json'


def inspect_generation(root, sample, variant, cfg, contract):
    g = checked(path_for(root, sample['sample_id'], variant), contract)
    if (g['sample_id'] != sample['sample_id'] or g['variant'] != variant
            or g['prompt_sha256'] != digest(make_prompt(sample, variant, cfg))):
        raise RuntimeError('Cached request mismatch')
    verify(root/'model.json', g['model_metadata_sha256'])
    parsed = interpret_text(g['raw_output'], sample['baseline'], g['ended'])
    if any(g[k] != v for k,v in parsed.items()): raise RuntimeError('Parser drift')
    return g


def preflight(a):
    p,samples,_ = load_run(a.output_dir)
    from transformers import AutoTokenizer
    snapshot = a.hf_cache/('models--'+p['model']['model_id'].replace('/','--'))/'snapshots'/p['model']['revision']
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    largest = 0
    for s in samples:
        for v in variants(p['config']):
            n = len(tokenizer.apply_chat_template([{'role':'user','content':make_prompt(s,v,p['config'])}], add_generation_prompt=True))
            if n > p['config']['max_input_tokens']: raise RuntimeError(f'Input too long: {s["sample_id"]} {v}: {n}')
            largest = max(largest,n)
    print(f'[preflight CPU] 1832 prompts fit; maximum {largest} tokens; no model inference', flush=True)


def infer(a):
    p,samples,contract = load_run(a.output_dir); cfg=p['config']
    pending=[]
    for s in samples:
        for v in variants(cfg):
            if path_for(a.output_dir,s['sample_id'],v).exists(): inspect_generation(a.output_dir,s,v,cfg,contract)
            else: pending.append((s,v))
    if not pending:
        print('[reuse] 1832/1832 verified; no model load', flush=True); return
    print(f'[resume] {1832-len(pending)}/1832 verified; {len(pending)} requests pending', flush=True)
    meta=p['model_metadata']; verify(meta['path'],meta['sha256'])
    old=checked(Path(meta['path']),p['model_source_contract']); spec=p['model']
    snapshot=a.hf_cache/('models--'+spec['model_id'].replace('/','--'))/'snapshots'/spec['revision']
    for rel,h in old['snapshot_files_sha256'].items(): verify(snapshot/rel,h)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    device=torch.device(a.device)
    if device.type != 'cuda' or not torch.cuda.is_available(): raise RuntimeError('Manual PAPER GPU stage only')
    torch.cuda.set_device(device); torch.cuda.init()
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('Qualified BF16 required')
    torch.manual_seed(cfg['seed']); torch.backends.cudnn.benchmark=False
    labels=checked(a.output_dir/'evaluation-labels.json',contract)['labels']
    if digest(labels)!=p['labels_sha256']: raise RuntimeError('GT log drift')
    model=None
    try:
        print('[load] pinned Qwen2.5-1.5B, BF16 batch1; no caption/OCR inference', flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        model,loading=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.bfloat16,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError(f'Checkpoint mismatch: {loading}')
        model.to(device).eval(); torch.cuda.synchronize(device)
        metadata={'contract_sha256':contract,'spec':spec,'runtime':runtime_info(torch,device),
                  'loading_info':loading,'batch_size':1,'quantization':None,
                  'snapshot_files_sha256':old['snapshot_files_sha256'],
                  'parameters':sum(x.numel() for x in model.parameters())}
        mp=a.output_dir/'model.json'
        if mp.exists():
            prev=checked(mp,contract)
            if any(prev[k]!=v for k,v in metadata.items()): raise RuntimeError('Runtime changed on resume')
        else: seal_write(mp,{**metadata,'load_seconds':time.perf_counter()-start,
                            'load_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                            'load_peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
        eos=model.generation_config.eos_token_id; eos_ids=eos if isinstance(eos,list) else [eos]
        gen=GenerationConfig(**cfg['generation'],eos_token_id=eos,pad_token_id=tokenizer.pad_token_id)
        for number,(s,v) in enumerate(pending,1832-len(pending)+1):
            message=make_prompt(s,v,cfg)
            encoded=tokenizer.apply_chat_template([{'role':'user','content':message}],tokenize=True,
                add_generation_prompt=True,return_tensors='pt')
            n=encoded.shape[1]
            if n>cfg['max_input_tokens']: raise RuntimeError('Input overflow; no silent truncation')
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
            with torch.inference_mode():
                ids=model.generate(input_ids=encoded.to(device),attention_mask=torch.ones_like(encoded).to(device),
                    generation_config=gen)[0,n:].cpu().tolist()
            torch.cuda.synchronize(device)
            raw=tokenizer.decode(ids,skip_special_tokens=True); ended=bool(ids) and ids[-1] in eos_ids
            parsed=interpret_text(raw,s['baseline'],ended)
            seal_write(path_for(a.output_dir,s['sample_id'],v),{'contract_sha256':contract,
                'sample_id':s['sample_id'],'variant':v,'prompt_sha256':digest(message),'model_metadata_sha256':sha(mp),
                'raw_output':raw,'ended':ended,**parsed,'input_tokens':n,'output_tokens':len(ids),
                'latency_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
            print(f'[{number}/1832] image={s["sample_id"]} {v} {parsed["status"]}',flush=True)
            print('OCR:',s['baseline'],'\nOUTPUT:',parsed['text'],'\nGT (evaluation only):',labels[s['sample_id']],flush=True)
    except Exception as exc:
        write_json(a.output_dir/'last-error.json',{'type':type(exc).__name__,'message':str(exc),
                   'oom':isinstance(exc,torch.cuda.OutOfMemoryError)})
        raise
    finally:
        if model is not None: del model
        torch.cuda.empty_cache()


def text_metrics(text, gt):
    from scipy.optimize import linear_sum_assignment
    from rapidfuzz.fuzz import ratio
    import numpy as np
    tokens=text.split(); counts=lexical_counts(tokens,gt); fuzzy=0
    if tokens and gt:
        good=np.array([[ratio(x.casefold(),y.casefold()) >= 80 for y in gt] for x in tokens])
        r,c=linear_sum_assignment(-good.astype(int)); fuzzy=int(good[r,c].sum())
    return {**counts,'fuzzy_ge_0_8':fuzzy}


def evaluate(a):
    p,samples,contract=load_run(a.output_dir); cfg=p['config']
    labels=checked(a.output_dir/'evaluation-labels.json',contract)['labels']
    if digest(labels)!=p['labels_sha256'] or sum(map(len,labels.values()))!=849: raise RuntimeError('GT drift')
    summaries,details,hashes={},{},{}
    for v in ['V1']+([] if a.baseline_only else variants(cfg)):
        rows=[]
        for s in samples:
            gt=labels[s['sample_id']]; g=None; text=s['baseline']
            if v!='V1':
                g=inspect_generation(a.output_dir,s,v,cfg,contract);text=g['text']
                f=path_for(a.output_dir,s['sample_id'],v);hashes[str(f.relative_to(a.output_dir))]=sha(f)
            m=text_metrics(text,gt); bm=lexical_counts([s['baseline']],gt)
            before=Counter(s['baseline'].casefold().split()) & Counter(x.casefold() for x in gt)
            after=Counter(text.casefold().split()) & Counter(x.casefold() for x in gt)
            rows.append({'sample_id':s['sample_id'],'gt':gt,'baseline':s['baseline'],'output':text,
                'metrics':m,'baseline_matches':bm['matches'],
                'recovered_lexical_occurrences':sum((after-before).values()),
                'lost_lexical_occurrences':sum((before-after).values()),
                'status':g['status'] if g else 'baseline', 'reason':g['reason'] if g else None,
                'seconds':g['latency_seconds'] if g else 0,
                'peak_allocated_bytes':g['peak_allocated_bytes'] if g else 0,
                'peak_reserved_bytes':g['peak_reserved_bytes'] if g else 0,
                'raw_whitespace_metrics':text_metrics(g['raw_output'],gt) if g else m})
        matches=sum(r['metrics']['matches'] for r in rows);pred=sum(r['metrics']['predicted'] for r in rows)
        fuzzy=sum(r['metrics']['fuzzy_ge_0_8'] for r in rows)
        summaries[v]={'gt_words':849,'image_lexical_exact_matches':matches,'exact_recall':matches/849,
            'exact_precision':matches/pred if pred else 0,'exact_f1':2*matches/(849+pred),
            'fuzzy_matches_ge_0_8':fuzzy,'fuzzy_recall':fuzzy/849,'predicted_tokens':pred,
            'recovered_lexical_occurrences':sum(r['recovered_lexical_occurrences'] for r in rows),
            'lost_lexical_occurrences':sum(r['lost_lexical_occurrences'] for r in rows),
            'improved_images':sum(r['metrics']['matches']>r['baseline_matches'] for r in rows),
            'worsened_images':sum(r['metrics']['matches']<r['baseline_matches'] for r in rows),
            'statuses':dict(Counter(r['status'] for r in rows)),
            'total_llm_seconds':sum(r['seconds'] for r in rows),'median_llm_seconds':statistics.median(r['seconds'] for r in rows),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
            'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows),
            'raw_unfiltered_lexical_matches':sum(r['raw_whitespace_metrics']['matches'] for r in rows)}
        if matches != 503+summaries[v]['recovered_lexical_occurrences']-summaries[v]['lost_lexical_occurrences']:
            raise RuntimeError('Lexical accounting mismatch')
        details[v]=rows
    if summaries['V1']['image_lexical_exact_matches']!=503 or summaries['V1']['predicted_tokens']!=1203:
        raise RuntimeError('V1 control mismatch')
    report={'contract_sha256':contract,'mode':'APPROXIMATE','classification':'PRACTICAL_APPROXIMATION',
        'claim_label':cfg['claim_label'],'scope':{'split':'TRAIN','images':229,'gt_words':849,'crops':1203},
        'summary':summaries,'details':details,'generation_sha256':hashes,'table5_accuracy':None,'pipeline_promoted':False,
        'metric_warning':'Image-level one-to-one casefold whitespace matching, NOT spatial exact accuracy. Fuzzy maximum-cardinality >=.8. No GT grouping. Raw metrics may reward narration; report separately.'}
    path=a.output_dir/('baseline.json' if a.baseline_only else 'results.json');seal_write(path,report)
    print(json.dumps(summaries,indent=2),flush=True);print('JSON',path,'SHA256',sha(path),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);subs=ap.add_subparsers(dest='stage',required=True)
    prep=subs.add_parser('prepare')
    for name in ('config','routing-root','model-source','audit'):prep.add_argument('--'+name,type=Path,required=True)
    pre=subs.add_parser('preflight');pre.add_argument('--hf-cache',type=Path,required=True)
    inf=subs.add_parser('infer');inf.add_argument('--hf-cache',type=Path,required=True);inf.add_argument('--device',required=True)
    ev=subs.add_parser('evaluate');ev.add_argument('--baseline-only',action='store_true')
    for sub in (prep,pre,inf,ev):sub.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();{'prepare':prepare,'preflight':preflight,'infer':infer,'evaluate':evaluate}[a.stage](a)


if __name__=='__main__':main()
