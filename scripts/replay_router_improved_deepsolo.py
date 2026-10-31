"""Approximate frozen-router replay with independently qualified DeepSolo profiles."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from recognition.clip_router_evaluation import deepsolo_predictions, native_mask
from recognition.deepsolo_word_f1 import apply_policy, fuse_views, recognition_confidence, word_counts
from recognition.f1_quality_study import COUNT_FIELDS, aggregate
from recognition.paper_image_router_evaluation import word_metrics
from recognition.trainable_image_router import oracle
from scripts.run_yolo_f1_quality_study import read, seal, sha
from scripts.token_context_data import annotations, key
from scripts.token_context_training import point, ranked

REPO = Path(__file__).resolve().parents[1]


def metric_counts(predictions, gt, metric):
    if metric not in ('casefold_exact', 'casefold_alnum'):
        raise ValueError('Unknown explicit transcription metric')
    value = (word_metrics if metric == 'casefold_exact' else word_counts)(predictions, gt)
    return {k: value[k] for k in COUNT_FIELDS}


def independent_counts(predictions, gt, metric):
    """Separate multiset implementation; keep symbol-only and invalid denominators."""
    def norm(text):
        text = text.casefold()
        return text if metric == 'casefold_exact' else ''.join(c for c in text if c.isalnum())
    care = [p for p in predictions if not p['ignored']]
    a = Counter(norm(p['text']) for p in care if p.get('invalid_envelope') is None and norm(p['text']))
    b = Counter(norm(g['text']) for g in gt if norm(g['text']))
    tp = sum(min(n, b[k]) for k, n in a.items())
    return dict(tp=tp, fp=len(care)-tp, fn=len(gt)-tp, detected_words=len(care), gt_words=len(gt),
                all_outputs=len(predictions), ignored_outputs=len(predictions)-len(care))


def frozen_predictions(row, policy):
    if policy['resolution'] == 'fused':
        return fuse_views(row['native'], row['large'], policy)
    return apply_policy(row['native'], policy)


def independent_route(branches, ids, accepted):
    if len(accepted) != len(set(accepted)) or not set(accepted) <= set(ids):
        raise ValueError('Duplicate or unknown routing decision')
    accepted = set(accepted)
    counts = {k: 0 for k in COUNT_FIELDS}
    for sid in ids:
        for k, v in branches[sid]['light' if sid in accepted else 'deepsolo'].items():
            counts[k] += v
    den = counts['detected_words']+counts['gt_words']; union = counts['tp']+counts['fp']+counts['fn']
    return dict(counts, images=len(ids), f1=2*counts['tp']/den if den else None,
                accuracy_tn0=counts['tp']/union if union else None)


def conditional_interval(branches, ids, accepted, weights):
    accepted = set(accepted)
    changed = np.array([[branches[s]['light' if s in accepted else 'deepsolo'][k]
                        for k in ('tp', 'detected_words', 'gt_words')] for s in ids], float)
    base = np.array([[branches[s]['deepsolo'][k] for k in ('tp', 'detected_words', 'gt_words')] for s in ids], float)
    a, b = weights@changed, weights@base
    draws = 100*(2*a[:, 0]/(a[:, 1]+a[:, 2])-2*b[:, 0]/(b[:, 1]+b[:, 2]))
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(lower95_pp=float(lo), upper95_pp=float(hi), conditional_on_fixed_routes=True)


def load_sources(cfg, paths):
    pinned = {}
    for role, relative_files in cfg['source_sha256'].items():
        for relative, expected in relative_files.items():
            p = Path(paths[role])/relative
            if sha(p) != expected:
                raise ValueError('Qualified source changed: '+str(p))
            pinned[str(p)] = expected
    token, causes, improved = (Path(paths[n]) for n in ('token_root', 'cause_root', 'improved_root'))
    for root in (token, causes):
        audit = read(root/'independent_audit.json')
        if audit['status'] != 'PASS' or audit['results_sha256'] != sha(root/'results.json'):
            raise ValueError('Unqualified source router run')
    audit = read(improved/'independent_audit.json')
    if (audit['status'] != 'PASS' or audit['result_sha256'] != sha(improved/'results.json')
            or audit['processed_predictions_sha256'] != sha(improved/'processed_predictions.json')
            or audit['selection_sha256'] != sha(improved/'selection.json')):
        raise ValueError('Unqualified improved DeepSolo run')
    if read(improved/'deployment_independent_audit.json')['status'] != 'PASS':
        raise ValueError('Unqualified deployment profiles')
    profiles = read(improved/'deployment_profiles.json')['profiles']
    for value in profiles.values():
        if sha(REPO/value['config_path']) != value['config_sha256']:
            raise ValueError('Selected DeepSolo profile changed')
    policies = {'original': cfg['original_policy'], 'confidence90': read(Path(paths['confidence_root'])/'selection.json')['selected']}
    policies['confidence90'] = dict(policies['confidence90'], resolution='native')
    policies.update({name: value['policy'] for name, value in profiles.items()})
    if set(policies) != set(cfg['profiles']):
        raise ValueError('DeepSolo profile population changed')
    return read(token/'resolved_configuration.json'), policies, pinned


def load_rows(plan, cfg, paths, root, policies, pinned):
    token = Path(paths['token_root']); improved = Path(paths['improved_root'])
    qualified_inputs = read(Path(paths['cause_root'])/'resolved_configuration.json')['input_sha256']
    processed = read(improved/'processed_predictions.json')['splits']; rows = {}; checks = 0
    for split, samples in plan['samples'].items():
        expanded = {r['sample_id']: r for r in processed[split]}; rows[split] = {}
        old_branches = read(token/split/'branch_metrics.json')['branches']
        pinned[str(token/split/'branch_metrics.json')] = sha(token/split/'branch_metrics.json')
        for index, sample in enumerate(samples):
            sid = sample['sample_id']; gt = annotations(sample)
            for p, expected in [(Path(sample['annotation_path']), sample['annotation_sha256']),
                                (Path(sample['validity_path']), sample['validity_sha256'])]:
                if sha(p) != expected:
                    raise ValueError('Dataset evidence changed')
                pinned[str(p)] = expected
            lp = token/split/'light'/(key(sid)+'.json'); raw_light = read(lp)['rows']; pinned[str(lp)] = sha(lp)
            if pinned[str(lp)] != qualified_inputs[str(lp)]:
                raise ValueError('Original router input transcripts changed')
            with Image.open(sample['validity_path']) as im:
                validity = native_mask(im.convert('L'), plan['config']['metric']['native_ignore_size'])
            light = []
            for r in raw_light:
                a, b, c, d = r['bbox']; mask = Image.new('L', tuple(sample['image_wh']), 0)
                ImageDraw.Draw(mask).rectangle([a, b, c-1, d-1], fill=255)
                footprint = native_mask(mask, validity.shape[0])
                light.append(dict(text=r['raw_text'], bbox=r['bbox'], ignored=bool(footprint.any() and not np.any(footprint&validity))))
            if sid in expanded:
                data = expanded[sid]
                if (data['ground_truth'] != gt or data['image_sha256'] != sample['image_sha256']
                        or data['image_wh'] != sample['image_wh'] or data['orientation'] != sample['orientation']):
                    raise ValueError('Improved DeepSolo paired image/GT mismatch: '+sid)
                available = policies
            else:
                p = token/split/'deepsolo'/(key(sid)+'.json'); raw = read(p); pinned[str(p)] = sha(p)
                if (not raw['response']['success'] or raw['contract'] != dict(
                        sample_id=sid, image_sha256=sample['image_sha256'],
                        plan_sha256=sha(token/'resolved_configuration.json'))):
                    raise ValueError('Failed DeepSolo source image')
                preds = deepsolo_predictions(raw['response'], sample['image_wh'], sample['orientation'],
                                            validity, policies['original']['score_threshold'])
                detections = {d['source_index']: d for d in raw['response']['detections']}
                for pred in preds:
                    pred['recognition_confidence'] = recognition_confidence(detections[pred['instance_id']]['raw']['recognition_class_probabilities'])
                data = {'native': preds}
                available = {k: policies[k] for k in ('original', 'confidence90')}
            heavy = {k: frozen_predictions(data, p) for k, p in available.items()}
            for name, preds in [('light', light), ('deepsolo', heavy['original'])]:
                got = word_metrics(preds, gt)
                if got != old_branches[sid][name]:
                    raise ValueError('Original punctuation-retaining branch replay differs: '+sid+'/'+name)
            record = dict(sample_id=sid, rgb_sha256=sample['rgb_sha256'], ground_truth=gt, light=light, profiles=heavy)
            record['counts'] = {metric: {'light': metric_counts(light, gt, metric),
                'profiles': {name: metric_counts(preds, gt, metric) for name, preds in heavy.items()}} for metric in cfg['metrics']}
            for metric, value in record['counts'].items():
                for name, preds in [('light', light), *heavy.items()]:
                    expected = value['light'] if name == 'light' else value['profiles'][name]
                    if independent_counts(preds, gt, metric) != expected:
                        raise ValueError('Independent word-count audit failed')
                    checks += 1
            rows[split][sid] = record
            if (index+1) % 250 == 0:
                print(f'Count replay {split}: {index+1}/{len(samples)}', flush=True)
        seal(root/split/'evaluation_rows.json', {'samples': rows[split]})
    return rows, processed, checks


def load_scores(plan, cfg, paths, pinned):
    token = Path(paths['token_root']); causes = Path(paths['cause_root']); frozen = read(token/'frozen_choices.json')
    cv = read(token/'cv_predictions.json')['full']; scores = {'train': {}, 'test': {}}
    for arm in cfg['original_arms']:
        scores['train']['original/'+arm] = cv[arm][str(frozen['arms'][arm]['epoch'])]
        p = token/'predictions'/('test__'+arm+'__evaluated.json'); scores['test']['original/'+arm] = read(p)['scores']; pinned[str(p)] = sha(p)
    trials = read(causes/'model_results.json')['trials']
    for name, trial in trials.items():
        p = causes/'training_trials'/(name+'.json'); value = read(p); pinned[str(p)] = sha(p)
        scores['train']['trial/'+name] = value['oof_scores']
        scores['test']['trial/'+name] = trial['test']['scores']
    for split, ss in plan['samples'].items():
        for name, values in scores[split].items():
            if set(values) != {s['sample_id'] for s in ss} or any(v is not None and not np.isfinite(v) for v in values.values()):
                raise ValueError('Incomplete frozen routing scores: '+split+'/'+name)
    return scores


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--mode', choices=['approximate'], required=True)
    for name in ('config', 'paths', 'output-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); cfg, paths = read(args.config), read(args.paths)
    if (args.config.resolve().parent != REPO/'configs/approximate' or cfg['reproduction_mode'] != args.mode
            or paths['reproduction_mode'] != args.mode or args.output_root.resolve().is_relative_to(REPO)):
        raise ValueError('Explicit isolated APPROXIMATE configuration and external outputs required')
    torch.set_num_threads(cfg['cpu_threads']); root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    plan, policies, pinned = load_sources(cfg, paths)
    seal(root/'resolved_configuration.json', dict(reproduction_mode=args.mode, claim_label='approximate reproduction',
        config=cfg, paths=paths, policies=policies, config_sha256=sha(args.config), paths_sha256=sha(args.paths),
        code_sha256={n: sha(REPO/n) for n in cfg['code_files']},
        new_model_inference=False, routers_refitted=False, strict_defaults_changed=False))
    scores = load_scores(plan, cfg, paths, pinned)
    rows, processed, checks = load_rows(plan, cfg, paths, root, policies, pinned)
    summaries = {}; replay_checks = 0; endpoint_profiles = read(Path(paths['improved_root'])/'deployment_profiles.json')['profiles']
    for population, split in [('train_full', 'train'), ('train_shared125', 'train'), ('test_full', 'test')]:
        ids = sorted(rows[split]) if population != 'train_shared125' else sorted(r['sample_id'] for r in processed['train'])
        groups = [rows[split][s]['rgb_sha256'] for s in ids]; unique = sorted(set(groups)); mapping = np.array([unique.index(g) for g in groups])
        rng = np.random.default_rng(cfg['seed']); weights = np.array([np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[mapping]
            for _ in range(cfg['bootstrap_replicates'])], float)
        summaries[population] = {}
        available = [name for name in cfg['profiles'] if all(name in rows[split][s]['profiles'] for s in ids)]
        for metric in cfg['metrics']:
            summaries[population][metric] = {}
            for profile in available:
                branches = {s: dict(light=rows[split][s]['counts'][metric]['light'],
                    deepsolo=rows[split][s]['counts'][metric]['profiles'][profile]) for s in ids}
                base = aggregate(branches[s]['deepsolo'] for s in ids); light = aggregate(branches[s]['light'] for s in ids)
                if population in ('test_full', 'train_shared125') and metric == 'casefold_alnum' and profile in endpoint_profiles:
                    original = endpoint_profiles[profile]['splits'][split]['word']['metrics']
                    if any(base[k] != original[k] for k in COUNT_FIELDS):
                        raise ValueError('Improved DeepSolo published aggregate replay differs')
                    replay_checks += 1
                router_results = {}; oracles = {}
                for name, full_scores in scores[split].items():
                    values = {s: full_scores[s] for s in ids}; eligible = [s for s in ids if values[s] is not None]; points = {}
                    for fraction in cfg['budget_fractions']:
                        accepted = ranked(values, ids, fraction); value = point(branches, ids, accepted)
                        independent = independent_route(branches, ids, accepted)
                        if any(value['combined'][k] != v for k, v in independent.items()):
                            raise ValueError('Independent combined-system count audit failed')
                        checks += 1
                        signature = (tuple(eligible), len(accepted))
                        if signature not in oracles:
                            oracles[signature] = oracle(branches, ids, eligible, len(accepted))
                        value.update(delta_F1_vs_all_deepsolo_pp=100*(value['combined']['f1']-base['f1']),
                            oracle=oracles[signature])
                        if population == 'test_full' and fraction in cfg['interval_fractions']:
                            value['paired_interval_vs_all_deepsolo'] = conditional_interval(branches, ids, accepted, weights)
                        points[str(fraction)] = value
                    router_results[name] = points
                summaries[population][metric][profile] = dict(images=len(ids), all_light=light, all_deepsolo=base, routers=router_results)
        print('Completed '+population+': '+str(len(ids))+' images, profiles '+','.join(available), flush=True)
    # Replay the user's ~82.3 endpoint separately from the newer profiles.
    expected90 = read(Path(paths['confidence_root'])/'results.json')['splits']['test']['selected']
    actual90 = summaries['test_full']['casefold_alnum']['confidence90']['all_deepsolo']
    if any(actual90[k] != expected90[k] for k in COUNT_FIELDS):
        raise ValueError('82.3-percent confidence-profile replay differs')
    result = dict(status='SUCCESS', reproduction_mode=args.mode, claim_label='approximate reproduction',
        official_evaluation=False, table_5_claim=False, historically_explored_TEST=True,
        plan_sha256=sha(root/'resolved_configuration.json'), profiles=policies, summaries=summaries,
        inputs_sha256=pinned, routers_refitted=False, no_GT_used_for_deployable_routing=True,
        full_TRAIN_profile_availability={k:k in ('original', 'confidence90') for k in cfg['profiles']},
        interpretation_limit='Frozen router scores target the old fallback and punctuation-retaining metric. New fallback replacement is a paired evaluation, not refitting or new TEST model selection. Native/fusion profiles cover TRAIN125 and all TEST300 only; full TRAIN1253 uses original/confidence90 profiles.')
    seal(root/'results.json', result)
    seal(root/'independent_count_audit.json', dict(status='PASS', independent_count_and_route_checks=checks,
        qualified_improved_profile_endpoint_replays=replay_checks+1, results_sha256=sha(root/'results.json'),
        scope='Separate Counter counts and explicit route aggregation; existing exact rational same-budget oracle with certificate; no independent neural refit.'))
    seal(root/'completion.json', dict(status='SUCCESS', results_sha256=sha(root/'results.json'),
        audit_sha256=sha(root/'independent_count_audit.json')))
    print(root/'results.json', flush=True)


if __name__ == '__main__':
    main()
