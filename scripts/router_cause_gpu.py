"""Frozen-model crop and missing-word interventions, with raw paired evidence."""
from __future__ import annotations

from pathlib import Path
from collections import Counter
import gc
import hashlib
import time

import numpy as np
import torch

from recognition.router_cause_tests import word_views, pool_views, literal_foils, choose_omissions, omit_features, weighted_auc
from recognition.token_context_router import erased_views, transcript_chunks, likelihood_features
from recognition.token_context_runtime import FrozenDenseCLIP, FrozenTranscriptBLIP
from scripts.crop_clip_runtime import array_sha
from scripts.token_context_data import key, rgb, write_arrays, annotations
from scripts.run_yolo_f1_quality_study import read, sha, seal


def processor_boundary_check(processor, settings):
    marker = np.zeros((120, 480, 3), np.uint8); marker[:, :20, 0] = 255; marker[:, -20:, 2] = 255
    result = {}
    for kind in ['native']+settings['variants']:
        views = word_views(marker, kind, settings)
        pixels = processor(images=views, do_normalize=False, do_rescale=False, return_tensors='np')['pixel_values']
        red = (pixels[:, 0] > 200)&(pixels[:, 1] < 20)&(pixels[:, 2] < 20)
        blue = (pixels[:, 2] > 200)&(pixels[:, 1] < 20)&(pixels[:, 0] < 20)
        result[kind] = {'left_marker_pixels': int(red.sum()), 'right_marker_pixels': int(blue.sum()), 'views': len(views)}
        if kind != 'native' and (not red.any() or not blue.any()):
            raise ValueError('Complete-word preprocessing lost a boundary marker')
    return result


def extract_crops(plan, root, split):
    receipt = root/split/'crop_receipt.json'
    if receipt.exists():
        value = read(receipt)
        for file, digest in value['metadata_sha256'].items():
            if sha(root/split/'crop_variants'/file) != digest:
                raise ValueError('Completed crop stage changed')
        return
    if split == 'test' and not (root/'frozen_choices.json').exists():
        raise ValueError('Freeze all choices before TEST crop interventions')
    source = Path(plan['paths']['source_root']); settings = plan['config']['crop_settings']; device = plan['config']['execution']['device']
    snapshot = plan['base_paths']['clip_snapshot']
    for file, digest in plan['base_config']['assets']['clip_files'].items():
        if sha(Path(snapshot)/file) != digest:
            raise ValueError('Pinned CLIP asset changed')
    model = FrozenDenseCLIP(snapshot, plan['base_config']['clip'], device); replay = None; files = {}; started = time.monotonic()
    try:
        geometry_check = processor_boundary_check(model.processor.image_processor, settings)
        for index, sample in enumerate(plan['samples'][split], 1):
            sid = sample['sample_id']; output = root/split/'crop_variants'/(key(sid)+'.npz'); meta = output.with_suffix('.json')
            rows = read(source/split/'light'/(key(sid)+'.json'))['rows']
            if meta.exists() and replay is not None:
                if sha(output) != read(meta)['arrays_sha256']:
                    raise ValueError('Cached crop arrays changed')
                files[meta.name] = sha(meta); continue
            image = rgb(sample); crops = [np.ascontiguousarray(image[r['bbox'][1]:r['bbox'][3], r['bbox'][0]:r['bbox'][2]]) for r in rows]
            for r, crop in zip(rows, crops):
                if array_sha(crop) != r['crop_rgb_sha256']:
                    raise ValueError('Original word crop bytes changed')
            if replay is None and crops:
                actual = model.images(crops)
                with np.load(source/split/'features'/(key(sid)+'.npz'), allow_pickle=False) as original:
                    difference = float(np.max(np.abs(original['word_image']-actual)))
                    if not np.allclose(original['word_image'], actual, atol=plan['config']['replay_tolerance'], rtol=plan['config']['replay_tolerance']):
                        raise ValueError('Original CLIP word embedding replay differs')
                replay = {'sample_id': sid, 'maximum_absolute_difference': difference, 'status': 'PASS'}
            values = {}; view_counts = {}
            for kind in settings['variants']:
                groups = [word_views(crop, kind, settings) for crop in crops]; flat = [v for group in groups for v in group]
                encoded = model.images(flat); cursor = 0; vectors = []
                for group in groups:
                    vectors.append(pool_views(encoded[cursor:cursor+len(group)])); cursor += len(group)
                values[kind] = np.asarray(vectors, np.float32).reshape(-1, model.dimension); view_counts[kind] = [len(g) for g in groups]
            foils = [literal_foils(r['raw_text'], settings['foil_alphabet']) if r['raw_text'] else dict(prefix='', middle='', suffix='') for r in rows]
            for position in ('prefix', 'middle', 'suffix'):
                values['foil_'+position] = model.text([f[position] for f in foils])
            write_arrays(output, values)
            value = {'plan_sha256': sha(root/'resolved_configuration.json'), 'source_rows_sha256': sha(source/split/'light'/(key(sid)+'.json')),
                'arrays_sha256': sha(output), 'original_image_sha256': sample['image_sha256'], 'ground_truth_used_for_embeddings': False,
                'frozen_choices_sha256': sha(root/'frozen_choices.json') if split == 'test' else None,
                'view_counts': view_counts, 'array_shapes': {k: list(v.shape) for k, v in values.items()}}
            seal(meta, value); files[meta.name] = sha(meta)
            if index % plan['config']['execution']['log_every'] == 0 or index == len(plan['samples'][split]):
                print(f'[crop {split}] {index}/{len(plan["samples"][split])}', flush=True)
        info = model.finish()
        seal(receipt, {'model': info, 'source_embedding_replay': replay, 'boundary_markers': geometry_check, 'metadata_sha256': files,
            'images': len(plan['samples'][split]), 'seconds': time.monotonic()-started})
    except BaseException as error:
        seal(root/split/'crop_failure.json', {'error': str(error), 'sample_id': sid if 'sid' in locals() else None,
            'out_of_memory': isinstance(error, torch.cuda.OutOfMemoryError), 'precision': 'float32',
            'image_batch_size': model.settings['image_batch_size']})
        raise
    finally:
        model.close()


def crop_report(plan, root):
    source = Path(plan['paths']['source_root']); cfg = plan['config']; output = {}
    for split, ss in plan['samples'].items():
        labels = []; groups = []; scores = {kind: [] for kind in ['native']+cfg['crop_settings']['variants']}; foil_wins = {k: {p: [] for p in ('prefix', 'middle', 'suffix')} for k in scores}; aspect = []
        for image_index, sample in enumerate(ss):
            sid = sample['sample_id']; rows = read(source/split/'light'/(key(sid)+'.json'))['rows']; y = read(root/split/'labels'/(key(sid)+'.json'))
            with np.load(source/split/'features'/(key(sid)+'.npz'), allow_pickle=False) as original:
                text = original['word_text']; native = original['word_image']
            with np.load(root/split/'crop_variants'/(key(sid)+'.npz'), allow_pickle=False) as z:
                vectors = {'native': native, **{k: z[k] for k in cfg['crop_settings']['variants']}}
                foil_vectors = {p: z['foil_'+p] for p in ('prefix', 'middle', 'suffix')}
                for j, row in enumerate(rows):
                    if y['primary'][j] < 0:
                        continue
                    labels.append(y['primary'][j]); groups.append(image_index)
                    a, b, c, d = row['bbox']; aspect.append(max(c-a, d-b)/min(c-a, d-b))
                    for kind, vv in vectors.items():
                        correct = float(vv[j]@text[j]); scores[kind].append(correct)
                        if y['primary'][j] == 1 and y['auxiliary'][j] == 1 and len(row['raw_text']) >= cfg['crop_settings']['minimum_foil_word_length']:
                            for position in foil_wins[kind]:
                                diff = correct-float(vv[j]@foil_vectors[position][j]); tolerance = cfg['crop_settings']['foil_tie_tolerance']
                                foil_wins[kind][position].append(.5 if abs(diff) <= tolerance else float(diff > 0))
        labels, groups, aspect = np.array(labels), np.array(groups), np.array(aspect)
        report = {'care_words': len(labels), 'correctness_AUC': {}, 'literal_foil_pair_win_rate': {}, 'paired_auc_intervals': {}}
        for kind, values in scores.items():
            report['correctness_AUC'][kind] = {name: weighted_auc(labels[mask], np.asarray(values)[mask])
                for name, mask in [('all', np.ones(len(labels), bool)), ('wide', aspect > cfg['crop_settings']['wide_aspect_threshold']), ('compact', aspect <= cfg['crop_settings']['wide_aspect_threshold'])]}
            report['literal_foil_pair_win_rate'][kind] = {pos: {'words': len(v), 'win_rate_with_half_ties': float(np.mean(v)) if v else None} for pos, v in foil_wins[kind].items()}
        rng = np.random.default_rng(cfg['seeds'][0]); differences = {k: [] for k in cfg['crop_settings']['variants']}
        for _ in range(cfg['bootstrap_replicates']):
            weights = np.bincount(rng.integers(0, len(ss), len(ss)), minlength=len(ss))[groups]
            base = weighted_auc(labels, scores['native'], weights)
            if base is None:
                continue
            for kind in differences:
                value = weighted_auc(labels, scores[kind], weights)
                if value is not None:
                    differences[kind].append(value-base)
        for kind, values in differences.items():
            report['paired_auc_intervals'][kind] = {'difference': report['correctness_AUC'][kind]['all']-report['correctness_AUC']['native']['all'],
                'lower95': float(np.quantile(values, .025)), 'upper95': float(np.quantile(values, .975)), 'resampling_unit': 'image'}
        output[split] = report
    seal(root/'crop_metrics.json', output); return output


def omission_probes(plan, root):
    if (root/'blip_probe_receipt.json').exists():
        return
    from scripts.token_context_training import load_model, score_model
    source = Path(plan['paths']['source_root']); cfg = plan['config']; source_plan = read(source/'resolved_configuration.json')
    frozen = read(source/'frozen_choices.json'); settings = plan['base_config']['blip']; snapshot = plan['base_paths']['blip_snapshot']
    for file, digest in plan['base_config']['assets']['blip_files'].items():
        if sha(Path(snapshot)/file) != digest:
            raise ValueError('Pinned BLIP asset changed')
    model = FrozenTranscriptBLIP(snapshot, settings, cfg['execution']['device'])
    head, prep = load_model(source_plan, source, frozen['arms']['likelihood']['model_path']); completed = {}; started = time.monotonic()
    try:
        for split, ss in plan['samples'].items():
            candidates = []
            for sample in ss:
                sid = sample['sample_id']; rows = read(source/split/'light'/(key(sid)+'.json'))['rows']; label = read(root/split/'labels'/(key(sid)+'.json'))
                if read(source/split/'features'/(key(sid)+'.json'))['eligible'] and choose_omissions(label, rows) is not None:
                    candidates.append(sample)
            candidates.sort(key=lambda s: hashlib.sha256((str(cfg['seeds'][0])+s['sample_id']).encode()).hexdigest())
            chosen = candidates[:cfg['blip_probe_images'][split]]; completed[split] = []
            for i, sample in enumerate(chosen, 1):
                sid = sample['sample_id']; path = root/split/'blip_probes'/(key(sid)+'.json')
                if path.exists():
                    completed[split].append({'file': str(path.relative_to(root)), 'sha256': sha(path)}); continue
                image = rgb(sample); rows = read(source/split/'light'/(key(sid)+'.json'))['rows']; label = read(root/split/'labels'/(key(sid)+'.json'))
                omissions = choose_omissions(label, rows); cached = read(source/split/'blip'/(key(sid)+'.json'))['raw']
                with np.load(source/split/'features'/(key(sid)+'.npz'), allow_pickle=False) as z:
                    original_features = {k: z[k] for k in z.files}
                original_features['eligible'] = True; q, p, _ = model.visual(image)
                erased, _ = erased_views(image, rows, settings['erasure_rgb'], settings['local_margin_height']); eq, ep, _ = model.visual(erased)
                tokenize = lambda text: model.processor.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
                variants = [('original', [], False, False)]+[(kind+'_'+mask, deleted, mask == 'natural', False) for kind, deleted in omissions.items() for mask in ('fixed', 'natural')]+[('word_order_control', [], False, True)]
                probes = []
                for name, deleted, natural, permuted in variants:
                    features, remaining = omit_features(original_features, rows, deleted, sample['image_wh'])
                    proposed = remaining
                    if permuted:
                        proposed = [dict(r) for r in remaining]; readings = [r['raw_text'] for r in proposed][::-1]
                        for row, text in zip(proposed, readings):
                            row['raw_text'] = text
                    chunks = transcript_chunks(proposed, tokenize, settings['maximum_target_tokens']); texts = [c['text'] for c in chunks]
                    oq, op = eq, ep
                    if natural:
                        hidden, _ = erased_views(image, remaining, settings['erasure_rgb'], settings['local_margin_height']); oq, op, _ = model.visual(hidden)
                    original = model.sequence(q, p, texts); removed = model.sequence(oq, op, texts)
                    global_scores = [{'original': a, 'erased': b, 'visual_gain': a['mean_logprob']-b['mean_logprob'],
                        'eos_visual_gain': a['eos_logprob']-b['eos_logprob'], 'original_logprob_per_token': a['mean_logprob'],
                        'row_indices': c['row_indices']} for a, b, c in zip(original, removed, chunks)]
                    excluded = {rows[j]['candidate_id'] for j in deleted}; words = [w for w in cached['word_scores'] if w['candidate_id'] not in excluded]
                    values = likelihood_features(global_scores, words); features['likelihood'] = np.array(list(values.values()), np.float32)
                    scores, _ = score_model(head, prep, {sid: features}, {}, [sid], plan['base_config'])
                    if name == 'original' and not np.allclose(features['likelihood'], original_features['likelihood'], atol=cfg['replay_tolerance'], rtol=cfg['replay_tolerance']):
                        raise ValueError('Original BLIP likelihood replay differs')
                    # The diagnostic selection excludes surplus duplicate predictions.
                    care = Counter(r['raw_text'].casefold() for j, r in enumerate(rows) if label['primary'][j] >= 0)
                    care_after = Counter(r['raw_text'].casefold() for j, r in enumerate(rows) if j not in deleted and label['primary'][j] >= 0)
                    gt = Counter(g['text'].casefold() for g in annotations(sample))
                    delta_tp = sum(min(n, gt[t]) for t, n in care.items() if t)-sum(min(n, gt[t]) for t, n in care_after.items() if t)
                    if delta_tp != len(deleted):
                        raise ValueError('Omission did not create the declared FN increment')
                    probes.append({'name': name, 'deleted_row_indices': deleted, 'known_extra_FN': delta_tp,
                        'score': scores[sid], 'likelihood_features': values, 'global_scores': global_scores,
                        'same_original_image_rgb_sha256': array_sha(image), 'ground_truth_used_to_construct_intervention': bool(deleted),
                        'erasure_mask': 'remaining boxes' if natural else 'original boxes', 'language_order_only': permuted})
                seal(path, {'source_rows_sha256': sha(source/split/'light'/(key(sid)+'.json')), 'sample_id': sid,
                    'tests': probes, 'selection': 'seeded eligible images with at least two safely removable correct care predictions; privileged diagnostic subset',
                    'original_expert_outputs_unchanged': True, 'image_rgb_sha256': array_sha(image)})
                completed[split].append({'file': str(path.relative_to(root)), 'sha256': sha(path)})
                if i % cfg['execution']['log_every'] == 0 or i == len(chosen):
                    print(f'[BLIP omissions {split}] {i}/{len(chosen)}', flush=True)
        seal(root/'blip_probe_receipt.json', {'model': model.finish(), 'files': completed,
            'precision': 'float32', 'vision_batch_size': 1, 'language_batch_size': settings['language_batch_size'],
            'seconds': time.monotonic()-started, 'result_class': 'PRIVILEGED_PARTIAL_DATASET_DIAGNOSTIC'})
    finally:
        del head; model.close(); gc.collect(); torch.cuda.empty_cache()


def probe_report(plan, root):
    result = {'token_deletion_tests': {}, 'blip_tests': {}}; tests = read(root/'deletion_tests.json')['tests']
    for trial, rows in tests.items():
        values = {}
        for split in ('train', 'test'):
            for kind in ('drop_one', 'drop_half'):
                vv = [r['score_drop'] for r in rows if r['split'] == split and r['kind'] == kind and r['score_drop'] is not None]
                values[split+'__'+kind] = {'images': len(vv), 'mean_score_drop': float(np.mean(vv)) if vv else None,
                    'fraction_reacting_in_expected_direction': float(np.mean(np.array(vv) > plan['config']['score_tie_tolerance'])) if vv else None}
        result['token_deletion_tests'][trial] = values
    receipt = read(root/'blip_probe_receipt.json')
    for split, files in receipt['files'].items():
        groups = {}
        for file in files:
            row = read(root/file['file']); baseline = row['tests'][0]
            for probe in row['tests'][1:]:
                change = baseline['score']-probe['score']
                groups.setdefault(probe['name'], []).append(change)
        result['blip_tests'][split] = {name: {'images': len(v), 'mean_score_drop': float(np.mean(v)),
            'fraction_reacting_in_expected_direction': float(np.mean(np.array(v) > plan['config']['score_tie_tolerance']))} for name, v in groups.items()}
    seal(root/'omission_metrics.json', result); return result
