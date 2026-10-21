"""APPROXIMATE contextual image-quality probes; frozen models and TRAIN-only cuts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch

from recognition.contextual_quality_features import explicit_caption_evidence, caption_evidence_features
from recognition.contextual_quality_router import (
    FAMILIES, fit_reference, image_scores, lower_quantile, spelling_margin, word_variants)
from recognition.clip_router_prompt_ablation import transcript, budget_accept, coverage_threshold
from recognition.training_free_consistency_router import calibrate, evaluate
from recognition.yolo_router_comparison import aggregate
from scripts.contextual_lm_verifier_runtime import FrozenCaptionJudge, state_digest
from scripts.crop_clip_runtime import sha, array_sha, _embeddings, _cuda_peak
from scripts.run_clip_high_recall_admission import seal, payload_digest
from scripts import run_yolo_router_comparison as source

REPO = Path(__file__).resolve().parents[1]
CODE = ('scripts/run_yolo_context_quality.py', 'recognition/contextual_quality_router.py',
        'recognition/contextual_quality_features.py', 'scripts/contextual_lm_verifier_runtime.py',
        'recognition/training_free_consistency_router.py', *source.CODE)


def stamp(value, plan_hash):
    value = {**value, 'plan_sha256': plan_hash}
    value['payload_sha256'] = payload_digest(value)
    return value


def receipt(path, plan_hash):
    if not path.exists():
        return None
    return source.checked_shard(path, plan_hash)


def prepare(args):
    for path in (args.config, args.paths):
        if path.resolve().parent != REPO/'configs/approximate':
            raise ValueError('An explicit isolated configs/approximate profile is required')
    cfg, paths = source.read(args.config), source.read(args.paths)
    if (args.mode != 'approximate' or cfg['reproduction_mode'] != 'approximate'
            or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim'] is not False
            or cfg['claim_label'] != 'approximate reproduction' or cfg['families'] != list(FAMILIES)
            or cfg['execution']['precision'] != 'float32' or cfg['quality_f1_threshold'] != .8
            or cfg['caption_negative_controls'] < 1):
        raise ValueError('Unsupported explicit contextual quality protocol')
    roots = {k: Path(paths[k+'_run']).resolve() for k in ('paired', 'consistency', 'coverage')}
    output = (args.output_root or Path(paths['output_root'])).resolve()
    if output.is_relative_to(REPO) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots.values()):
        raise ValueError('Output must be external to the repository and frozen inputs')
    plans = {k: source.checked(p/'resolved_configuration.json', cfg[k+'_plan_sha256']) for k, p in roots.items()}
    cr = source.checked(roots['coverage']/'results.json', cfg['coverage_results_sha256'])
    sr = source.checked(roots['consistency']/'results.json', cfg['consistency_results_sha256'])
    audit_path = roots['consistency']/'independent_audit.json'
    audit = source.read(audit_path)
    if (cr['status'] != 'SUCCESS' or sr['status'] != 'SUCCESS' or audit['status'] != 'PASS'
            or audit['results_sha256'] != cfg['consistency_results_sha256']
            or audit['image_features_verified'] != 425 or audit['optimizer_steps'] != 0):
        raise ValueError('Complete audited frozen branches and consistency features are required')
    for plan in plans.values():
        for name, digest in plan['code_sha256'].items():
            if sha(REPO/name) != digest:
                raise ValueError('Frozen scientific input code changed: '+name)
    software = {k: importlib.metadata.version(k) for k in plans['paired']['software']}
    if software != plans['paired']['software']:
        raise ValueError('Runtime differs from qualified caption/CLIP/MPNet inputs')
    selection = list(plans['consistency']['selection_ids'])
    calibration = list(plans['consistency']['threshold_calibration_ids'])
    reference_ids = list(selection)
    samples = plans['coverage']['samples']
    if args.smoke:
        selection, calibration = selection[:2], calibration[:2]
        samples = {'train': [s for s in samples['train'] if s['sample_id'] in selection+calibration],
                   'test': samples['test'][:2]}
    if set(selection) & set(calibration) or set(selection+calibration) != {s['sample_id'] for s in samples['train']}:
        raise ValueError('TRAIN reference/selection and independent calibration must partition TRAIN')
    paired_samples = {s['sample_id']: s for rows in plans['paired']['samples'].values() for s in rows}
    control_pool = {s: paired_samples[s]['caption']['text'] for s in reference_ids}
    observations, branches, inputs = {}, {}, {str(audit_path): sha(audit_path)}
    for split, rows in samples.items():
        observations[split], branches[split] = {}, {}
        for s in rows:
            sid = s['sample_id']; key = source.key(sid)+'.json'; ps = paired_samples[sid]
            if ps['image_sha256'] != s['image_sha256'] or sha(Path(s['image_path'])) != s['image_sha256']:
                raise ValueError('Original image identity mismatch: '+sid)
            cap = ps['caption']; cache = source.checked(Path(cap['path']), cap['sha256'])
            if (cap['image_sha256'] != s['image_sha256'] or cache['payload_sha256'] != payload_digest(cache)
                    or cache['contract']['image_sha256'] != s['image_sha256']):
                raise ValueError('Caption is not tied to this original image')
            # Original run's sealed caption contract already excludes OCR/GT prompting.
            pf = roots['paired']/split/'features'/key
            cf = roots['coverage']/'640'/split/'features'/key
            sf = roots['consistency']/split/'features'/key
            pair = source.checked_shard(pf, cfg['paired_plan_sha256'])
            cov = source.checked_shard(cf, cfg['coverage_plan_sha256'])
            cons = source.checked_shard(sf, cfg['consistency_plan_sha256'])
            if (pair['caption'] != cap or pair['ground_truth_in_scoring'] is not False
                    or cov['ground_truth_used'] is not False or cons['models_trained'] is not False
                    or cons['original_word_outputs_changed'] is not False
                    or any(v['image_sha256'] != s['image_sha256'] for v in (pair, cov, cons))):
                raise ValueError('Unqualified feature or caption provenance')
            if transcript(cov['rows']) != pair['T3_assembly']['text']:
                raise ValueError('Original640 word transcript differs between controls')
            controls = caption_controls(sid, cap['text'], control_pool, cfg['caption_negative_controls'])
            observations[split][sid] = {'sample': s, 'caption': cap, 'rows': cov['rows'],
                'transcript': pair['T3_assembly']['text'], 'controls': controls,
                'source_scores': cons['scores'], 'paper_C3': pair['paper_decision']['C3']}
            branches[split][sid] = cr['resolutions']['640']['branch_metrics'][split][sid]
            inputs.update({str(p): sha(p) for p in (pf, cf, sf, Path(cap['path']))})
    clip = plans['paired']['clip_identity']; mpnet = plans['paired']['mpnet_identity']
    for name, digest in clip['file_sha256'].items():
        if sha(Path(clip['local_path'])/name) != digest:
            raise ValueError('CLIP artifact changed: '+name)
    for name, digest in mpnet['configuration_file_sha256'].items():
        if sha(Path(mpnet['snapshot'])/name) != digest:
            raise ValueError('MPNet configuration changed: '+name)
    if sha(Path(mpnet['snapshot'])/'model.safetensors') != mpnet['weight_sha256']:
        raise ValueError('MPNet weights changed')
    flan = Path(paths['flan_snapshot'])
    if flan.name != '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2':
        raise ValueError('Use the predeclared immutable FLAN-T5-base revision')
    flan_files = {str(p.relative_to(flan)): sha(p) for p in sorted(flan.rglob('*')) if p.is_file()}
    if not {'model.safetensors', 'config.json', 'tokenizer.json'} <= set(flan_files):
        raise ValueError('Incomplete local FLAN snapshot; no downloads will be attempted')
    if not torch.cuda.is_available() or not cfg['execution']['device'].startswith('cuda'):
        raise RuntimeError('Explicit CUDA experiment requires working CUDA; no CPU substitution')
    plan = {'schema_version': 1, 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': 'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_COMPARISON',
        'config': cfg, 'paths': paths, 'config_sha256': sha(args.config), 'paths_sha256': sha(args.paths),
        'samples': samples, 'selection_ids': selection, 'threshold_calibration_ids': calibration,
        'negative_caption_reference_ids': reference_ids, 'negative_caption_reference_pool': control_pool,
        'source_plans': {k: cfg[k+'_plan_sha256'] for k in roots}, 'input_sha256': inputs,
        'clip_identity': clip, 'mpnet_identity': mpnet, 'flan_files_sha256': flan_files,
        'code_sha256': {name: sha(REPO/name) for name in dict.fromkeys(CODE)},
        'software': software, 'python': sys.version, 'metric': cr['metric'],
        'gpu': torch.cuda.get_device_name(0), 'device_capability': list(torch.cuda.get_device_capability(0)),
        'new_network_training': False, 'test_labels_used_for_selection': False,
        'test_history': 'TRAIN/TEST historically explored; this is exploratory, not untouched validation',
        'deterministic_algorithms': True, 'tf32': False}
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    root = output/('smoke' if args.smoke else 'full')/tag
    return cfg, plan, root, observations, branches


def caption_controls(sid, own_caption, pool, count):
    ids = sorted(pool)
    offset = int(hashlib.sha256(sid.encode()).hexdigest(), 16) % len(ids)
    rows, seen = [], {own_caption.casefold()}
    for i in range(len(ids)):
        other = ids[(offset+i) % len(ids)]; text = pool[other]
        if other == sid or text.casefold() in seen:
            continue
        rows.append({'sample_id': other, 'text': text}); seen.add(text.casefold())
        if len(rows) == count:
            return rows
    raise ValueError('Not enough distinct unrelated TRAIN80 captions for controls')


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def mpnet_probes(cfg, plan, root, split, observed):
    ph = sha(root/'resolved_configuration.json'); target = root/split/'mpnet.json'
    hit = receipt(target, ph)
    if hit is not None:
        return hit
    from sentence_transformers import SentenceTransformer
    started = time.monotonic(); device = cfg['execution']['device']; torch.cuda.reset_peak_memory_stats(device)
    model = SentenceTransformer(plan['mpnet_identity']['snapshot'], device=device,
        model_kwargs={'torch_dtype': torch.float32}, local_files_only=True)
    try:
        model.requires_grad_(False).eval(); model.max_seq_length = 384
        before = state_digest(model)
        texts = list(dict.fromkeys(t for v in observed.values()
            for t in [v['transcript'], v['caption']['text'], *(r['text'] for r in v['controls'])]))
        lengths = {t: len(model.tokenizer(t, truncation=False)['input_ids']) for t in texts}
        valid = [t for t in texts if lengths[t] <= model.max_seq_length]
        embeddings = model.encode(valid, batch_size=cfg['execution']['batch_size'],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = dict(zip(valid, embeddings)); rows = {}
        for sid, v in observed.items():
            needed = [v['transcript'], v['caption']['text'], *(r['text'] for r in v['controls'])]
            overflow = any(lengths[t] > model.max_seq_length for t in needed)
            if overflow:
                rows[sid] = {'status': 'OVERFLOW_EXPLICIT_ABSTENTION', 'contrast': None,
                    'token_lengths': [lengths[t] for t in needed]}
            else:
                cosine = [float(vectors[needed[0]] @ vectors[t]) for t in needed[1:]]
                rows[sid] = {'status': 'SCORED', 'own_caption_cosine': cosine[0],
                    'control_cosines': cosine[1:], 'contrast': cosine[0]-float(np.mean(cosine[1:])),
                    'token_lengths': [lengths[t] for t in needed]}
        after = state_digest(model)
        if before != after:
            raise RuntimeError('MPNet model state changed')
        record = stamp({'rows': rows, 'model_state_sha256_before': before, 'model_state_sha256_after': after,
            'optimizer_steps': 0, 'models_trained': False, 'ground_truth_used': False,
            'seconds': time.monotonic()-started, 'device': device, 'precision': 'float32',
            'batch_size': cfg['execution']['batch_size'], 'max_input_tokens': max(lengths.values()),
            'embedding_dimension': int(embeddings.shape[1]), 'cuda_peak': _cuda_peak(device)}, ph)
        seal(target, record)
    finally:
        del model
        free_cuda()
    print(f'[MPNet {split}] {len(observed)} images', flush=True)
    return record


def clip_probes(cfg, plan, root, split, observed):
    ph = sha(root/'resolved_configuration.json'); target = root/split/'clip.json'
    hit = receipt(target, ph)
    if hit is not None:
        return hit
    from transformers import CLIPModel, CLIPProcessor
    device = cfg['execution']['device']; batch_size = cfg['execution']['batch_size']
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    model, loading = CLIPModel.from_pretrained(plan['clip_identity']['local_path'],
        local_files_only=True, output_loading_info=True, torch_dtype=torch.float32, use_safetensors=False)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('CLIP checkpoint mismatch: '+str(loading))
    model.requires_grad_(False).eval().to(device)
    processor = CLIPProcessor.from_pretrained(plan['clip_identity']['local_path'], local_files_only=True)
    rows = {}; word_count = 0; max_tokens = 0; shapes = []
    try:
        before = state_digest(model)
        for number, (sid, v) in enumerate(observed.items(), 1):
            sample = v['sample']
            if sha(Path(sample['image_path'])) != sample['image_sha256']:
                raise ValueError('Original RGB image changed before CLIP')
            with Image.open(sample['image_path']) as opened:
                rgb = np.asarray(opened.convert('RGB'))
            if list(rgb.shape[:2][::-1]) != sample['image_wh']:
                raise ValueError('Original image orientation differs')
            crops = []; requests = []; mapping = []
            for row in v['rows']:
                x1,y1,x2,y2 = row['bbox']; crop = np.ascontiguousarray(rgb[y1:y2,x1:x2])
                if list(crop.shape) != row['crop_shape'] or array_sha(crop) != row['crop_rgb_sha256']:
                    raise ValueError('Original crop pixels differ: '+row['candidate_id'])
                variants = word_variants(row['raw_text'])
                if len(variants) < 2:
                    raise ValueError('Nonempty routing word has no spelling controls')
                crops.append(Image.fromarray(crop)); shapes.append(list(crop.shape))
                prompts = {}
                for condition, caption in [('plain', None), ('own', v['caption']['text']),
                                            ('wrong', v['controls'][0]['text'])]:
                    prompts[condition] = [(cfg['clip_prompt_plain'] if caption is None else
                        cfg['clip_prompt_context'].replace('{caption}', caption)).format(text=w) for w in variants]
                    requests.extend(prompts[condition])
                mapping.append((row['candidate_id'], variants, prompts))
            unique = list(dict.fromkeys(requests))
            lengths = {t: len(processor.tokenizer(t, truncation=False)['input_ids']) for t in unique}
            max_tokens = max([max_tokens, *lengths.values()]); limit = model.config.text_config.max_position_embeddings
            valid = [t for t in unique if lengths[t] <= limit]
            image_vectors = _embeddings(model, processor, images=crops, texts=None, device=device, batch_size=batch_size)
            text_vectors = _embeddings(model, processor, images=None, texts=valid, device=device, batch_size=batch_size)
            by_text = dict(zip(valid, text_vectors)); words = []; overflow = False
            for i, (cid, variants, prompts) in enumerate(mapping):
                scores = {}; margins = {}
                for condition, texts in prompts.items():
                    scores[condition] = [float(image_vectors[i] @ by_text[t]) if t in by_text else None for t in texts]
                    margins[condition] = spelling_margin(scores[condition]) if None not in scores[condition] else None
                overflow |= any(x is None for x in margins.values())
                words.append({'candidate_id': cid, 'variants': variants, 'prompts': prompts,
                    'token_lengths': {k: [lengths[t] for t in ts] for k,ts in prompts.items()},
                    'cosines': scores, 'margins': margins})
            q = cfg['word_margin_quantile']
            rows[sid] = {'status': 'OVERFLOW_EXPLICIT_ABSTENTION' if overflow else 'SCORED', 'words': words,
                'plain_margin': None if overflow else lower_quantile([w['margins']['plain'] for w in words], q),
                'context_margin': None if overflow else lower_quantile([w['margins']['own'] for w in words], q),
                'context_gain': None if overflow else lower_quantile([w['margins']['own']-w['margins']['wrong'] for w in words], q)}
            word_count += len(words)
            if number % 25 == 0 or number == len(observed):
                print(f'[CLIP {split}] {number}/{len(observed)} images; {word_count} words', flush=True)
        after = state_digest(model)
        if before != after:
            raise RuntimeError('CLIP model state changed')
        record = stamp({'rows': rows, 'model_state_sha256_before': before, 'model_state_sha256_after': after,
            'optimizer_steps': 0, 'models_trained': False, 'ground_truth_used': False, 'loading_info': loading,
            'seconds': time.monotonic()-started, 'device': device, 'precision': 'float32', 'batch_size': batch_size,
            'image_preprocessor_shape': [3,224,224], 'original_crop_shapes': sorted({tuple(s) for s in shapes}),
            'word_crops': word_count, 'maximum_prompt_tokens': max_tokens, 'context_limit': 77,
            'cuda_peak': _cuda_peak(device)}, ph)
        seal(target, record)
    finally:
        del model
        free_cuda()
    return record


def lm_probes(cfg, plan, root, split, observed):
    ph = sha(root/'resolved_configuration.json'); target = root/split/'lm.json'
    hit = receipt(target, ph)
    if hit is not None:
        return hit
    judge = FrozenCaptionJudge(plan['paths']['flan_snapshot'], cfg['execution']['device'])
    try:
        if judge.provenance['snapshot_files_sha256'] != plan['flan_files_sha256']:
            raise ValueError('FLAN snapshot changed since preflight')
        ids = list(observed); transcripts = [observed[s]['transcript'] for s in ids]
        captions = [observed[s]['caption']['text'] for s in ids]
        wrong = [observed[s]['controls'][0]['text'] for s in ids]
        exact = judge.score_batch(captions, transcripts, question_kind='exact_word_support', batch_size=cfg['execution']['lm_batch_size'])
        coherent = judge.score_batch(captions, transcripts, question_kind='scene_coherence', batch_size=cfg['execution']['lm_batch_size'])
        wrong_result = judge.score_batch(wrong, transcripts, question_kind='scene_coherence', batch_size=cfg['execution']['lm_batch_size'])
        record = stamp({'sample_ids': ids, 'exact_support': exact, 'scene_coherence': coherent,
                        'wrong_caption_coherence': wrong_result, 'ground_truth_used': False, 'optimizer_steps': 0}, ph)
        seal(target, record)
    finally:
        del judge
        free_cuda()
    print(f'[FLAN {split}] {len(observed)} images,3 conditions', flush=True)
    return record


def features_for_split(cfg, plan, root, split, observed, reference=None):
    ph = sha(root/'resolved_configuration.json')
    mpnet = mpnet_probes(cfg, plan, root, split, observed)
    clip = clip_probes(cfg, plan, root, split, observed)
    lm = lm_probes(cfg, plan, root, split, observed)
    lmindex = {s:i for i,s in enumerate(lm['sample_ids'])}; raw = {}
    for sid, v in observed.items():
        evidence = explicit_caption_evidence(v['caption']['text'])
        anchors = caption_evidence_features([r['raw_text'] for r in v['rows']], evidence,
            minimum_edit_similarity=cfg['literal_minimum_edit_similarity'])
        i = lmindex[sid]
        raw[sid] = {'eligible': bool(v['rows']) and all(r['raw_text'].strip() for r in v['rows']),
            'sample_id': sid, 'image_sha256': v['sample']['image_sha256'],
            'caption': v['caption'], 'transcript': v['transcript'], 'controls': v['controls'],
            'source_scores': v['source_scores'], 'literal_evidence': anchors,
            'signals': {'literal_recall': anchors['exact_caption_recovery'], 'paper_caption_control': v['paper_C3'],
                'mpnet_contrast': mpnet['rows'][sid]['contrast'],
                'clip_plain_margin': clip['rows'][sid]['plain_margin'],
                'clip_context_margin': clip['rows'][sid]['context_margin'],
                'clip_context_gain': clip['rows'][sid]['context_gain'],
                'lm_literal': lm['exact_support']['scores'][i], 'lm_coherence': lm['scene_coherence']['scores'][i]},
            'wrong_caption_coherence': lm['wrong_caption_coherence']['scores'][i],
            'ground_truth_used': False, 'models_trained': False, 'original_word_outputs_changed': False}
    if reference is None:
        reference = fit_reference(raw, plan['selection_ids'])
        seal(root/'frozen_references.json', stamp(reference, ph))
    result = {}
    for sid, r in raw.items():
        result[sid] = stamp({**r, 'scores': image_scores(r, reference)}, ph)
        seal(root/split/'features'/(source.key(sid)+'.json'), result[sid])
    return result, reference


def freeze(cfg, plan, root, features, branches):
    selection, calibration = plan['selection_ids'], plan['threshold_calibration_ids']
    frozen = {'families': {}, 'selected_families': {}, 'selected_budget_families': {},
              'selection_trials': {}, 'budget_selection_trials': {}}
    for family in FAMILIES:
        scores = {s: features[s]['scores'][family] for s in features}
        frozen['families'][family] = {'targets': {}, 'coverage_cuts': {
            str(b): coverage_threshold(scores, calibration, b) for b in cfg['bypass_fractions']}}
        for target in cfg['target_good_image_fractions']:
            kw = dict(target=target, quality_threshold=cfg['quality_f1_threshold'],
                      minimum_accepted=cfg['minimum_calibration_accepted'])
            trial = calibrate({s:scores[s] for s in selection}, {s:branches[s] for s in selection}, selection, **kw)
            frozen['selection_trials'].setdefault(str(target), []).append({'family':family, **trial})
            frozen['families'][family]['targets'][str(target)] = calibrate(
                {s:scores[s] for s in calibration}, {s:branches[s] for s in calibration}, calibration, **kw)
    for target, trials in frozen['selection_trials'].items():
        best = min(trials, key=lambda t: (-(t['calibration']['accepted_images'] if t['calibration'] else 0),
            -(t['calibration']['good_fraction'] if t['calibration'] else 0), FAMILIES.index(t['family'])))
        frozen['selected_families'][target] = best['family']
    for b in cfg['bypass_fractions']:
        trials = []
        for family in FAMILIES:
            values = {s:features[s]['scores'][family] for s in selection}
            accepted = budget_accept(values, selection, round(len(selection)*b))
            trials.append({'family':family, 'accepted_images':len(accepted),
                'accepted_good_images':evaluate(branches,selection,accepted,.8)['accepted_good_images']})
        best = min(trials, key=lambda t:(-t['accepted_good_images'], FAMILIES.index(t['family'])))
        frozen['budget_selection_trials'][str(b)] = trials
        frozen['selected_budget_families'][str(b)] = best['family']
    path = root/'frozen_thresholds.json'
    if not path.exists() and list((root/'test').rglob('*.json')):
        raise ValueError('Refusing fresh calibration after new TEST inference exists')
    frozen.update(selection_ids=selection, threshold_calibration_ids=calibration,
        test_features_present_at_freeze=0, test_labels_used=False, models_trained=False,
        formal_risk_guarantee=False, primary_family=cfg['primary_family'])
    seal(path, stamp(frozen, sha(root/'resolved_configuration.json')))
    print('[frozen] TRAIN references,all thresholds,family choices saved before new TEST probes',flush=True)
    return frozen


def split_metrics(cfg, features, branches, ids, frozen):
    result = {'images':len(ids), 'families':{}, 'light_only':aggregate(branches[s]['light'] for s in ids),
              'deepsolo_only':aggregate(branches[s]['deepsolo'] for s in ids)}
    for family in FAMILIES:
        values = {s:features[s]['scores'][family] for s in ids}; points = {}
        for target in cfg['target_good_image_fractions']:
            cut = frozen['families'][family]['targets'][str(target)]['threshold']
            accepted = {s for s in ids if cut is not None and values[s] is not None and values[s]>=cut}
            points['quality_'+str(target)] = {'threshold':cut, **evaluate(branches,ids,accepted,.8)}
        for b in cfg['bypass_fractions']:
            count = round(len(ids)*b); accepted = budget_accept(values,ids,count)
            points['budget_'+str(b)] = {'requested_images':count, 'unfilled_budget':count-len(accepted),
                'threshold':None, **evaluate(branches,ids,accepted,.8)}
            cut = frozen['families'][family]['coverage_cuts'][str(b)]
            accepted = {s for s in ids if cut is not None and values[s] is not None and values[s]>=cut}
            points['coverage_cut_'+str(b)] = {'threshold':cut, **evaluate(branches,ids,accepted,.8)}
        result['families'][family] = points
    return result


def report(root, cfg, result):
    lines = ['# Frozen contextual image-quality routing', '',
        'APPROXIMATE exploratory reproduction; no network training or word corrections. Not Table5.', '',
        'TRAIN80 reference/family selection; TRAIN45 threshold calibration; historically explored TEST300.',
        'Good image means local casefold multiset wordF1>=.8. Purity is a fraction of images; combinedF1 is micro over words.', '',
        '| Split | Method | Bypass | Good / accepted | Good precision | Combined wordF1 | TN0 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split in ('train','test'):
        for family in FAMILIES:
            for b in (.2,.3,.5):
                p = result['splits'][split]['families'][family]['budget_'+str(b)]
                precision = p['good_fraction_among_accepted']
                lines.append(f"| {split} | {family} | {p['accepted_images']}/{p['images']} | "
                    f"{p['accepted_good_images']}/{p['accepted_images']} | {100*precision:.2f}% | "
                    f"{100*p['combined']['f1']:.4f}% | {100*p['combined']['accuracy_tn0']:.4f}% |" if precision is not None else
                    f'| {split} | {family} |0|0|undefined|--|--|')
    lines += ['', 'Thresholds and full image IDs/counts are in frozen_thresholds.json and results.json.',
        'No empirical threshold or semantic signal certifies the target. Partial/hallucinated captions and missing detections remain limitations.',
        'GPU probe latency includes added verification only; cached caption/U-Net/OCR/DeepSolo latency is not measured end to end.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def run(args):
    cfg, plan, root, observed, branches = prepare(args)
    if args.preflight:
        print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root),'images':{s:len(r) for s,r in observed.items()}}));return
    seal(root/'resolved_configuration.json', plan)
    for name,digest in plan['code_sha256'].items():
        p = root/'code_snapshot'/name; p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and sha(p)!=digest:raise ValueError('Code snapshot differs')
        if not p.exists():p.write_bytes((REPO/name).read_bytes())
    seed = cfg['execution']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.set_num_threads(cfg['execution']['cpu_threads'])
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    started = time.monotonic()
    try:
        train, ref = features_for_split(cfg,plan,root,'train',observed['train'])
        frozen = freeze(cfg,plan,root,train,branches['train'])
        test, _ = features_for_split(cfg,plan,root,'test',observed['test'],ref)
        result = {'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
            'table_5_claim':False,'result_class':plan['result_class'],'plan_sha256':sha(root/'resolved_configuration.json'),
            'metric':plan['metric'],'new_network_training':False,'optimizer_steps':0,'formal_risk_guarantee':False,
            'thresholds_sha256':sha(root/'frozen_thresholds.json'),'references_sha256':sha(root/'frozen_references.json'),
            'primary_family':cfg['primary_family'],'selected_families':frozen['selected_families'],
            'selected_budget_families':frozen['selected_budget_families'],
            'splits':{name:split_metrics(cfg,features,bb,ids,frozen) for name,features,bb,ids in [
                ('train',train,branches['train'],list(train)),('selection',train,branches['train'],plan['selection_ids']),
                ('calibration',train,branches['train'],plan['threshold_calibration_ids']),
                ('test',test,branches['test'],list(test))]}}
        seal(root/'results.json', result)
        report(root,cfg,result)
        if not (root/'runtime.json').exists():
            seal(root/'runtime.json',{'status':'SUCCESS','seconds':time.monotonic()-started,'optimizer_steps':0,
                'gpu_inventory':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],
                    check=True,capture_output=True,text=True).stdout.strip()})
    except Exception as exc:
        failure = {'status':'FAILED','type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'out_of_memory':isinstance(exc,torch.cuda.OutOfMemoryError),'cuda_peak':_cuda_peak(cfg['execution']['device']),
            'partial_results_are_not_final':True,'optimizer_steps':0}
        from scripts.crop_clip_runtime import write_json
        write_json(root/'failure.json',failure)
        raise
    print(json.dumps({'status':'SUCCESS','root':str(root),'results_sha256':sha(root/'results.json')}),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--output-root',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--smoke',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
