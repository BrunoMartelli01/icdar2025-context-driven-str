"""Matched seeds/folds and single-factor context-router interventions."""
from __future__ import annotations

import copy
import gc
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from scripts.run_yolo_f1_quality_study import read, sha, seal
from scripts.token_context_data import key
from scripts.token_context_training import fit_router, load_model, score_model, point, save_model
from scripts.run_router_cause_tests import inputs
from recognition.router_cause_tests import fixed_budget, omit_features, choose_omissions


def model_plan(plan, variant):
    cfg = copy.deepcopy(plan['base_config']); test = plan['config']
    cfg['training']['checkpoints'] = test['checkpoints']; cfg['training']['epochs'] = max(test['checkpoints'])
    cfg['training']['weight_decay'] = variant['weight_decay']
    cfg['training']['auxiliary_word_weight'] = variant['auxiliary_word_weight']
    cfg['architecture'].update(variant['architecture_overrides'])
    cfg['execution'].update(test['execution'])
    return {**plan, 'config': cfg}


def fitting_subset(ids, samples, fraction, seed):
    groups = sorted({samples[s]['rgb_sha256'] for s in ids}); random.Random(seed).shuffle(groups)
    wanted = set(groups[:max(1, round(len(groups)*fraction))])
    return [s for s in ids if samples[s]['rgb_sha256'] in wanted]


def intervention_features(features, variant, crop_values, fit_ids, samples, seed):
    value = {s: dict(v) for s, v in features.items()}; donors = {}
    if variant['crop'] != 'native':
        for sid in value:
            value[sid]['word_image'] = crop_values[sid][variant['crop']]
    if variant['shuffle_context']:
        for sid in sorted(value):
            pool = [s for s in fit_ids if samples[s]['rgb_sha256'] != samples[sid]['rgb_sha256']]
            if not pool:
                raise ValueError('Context shuffle needs a distinct fitting RGB donor')
            number = int(hashlib.sha256((str(seed)+':'+sid).encode()).hexdigest()[:12], 16)
            donor = pool[number % len(pool)]; donors[sid] = donor
            for field in ('patches', 'patch_geometry', 'patch_valid', 'patch_mass', 'queries'):
                value[sid][field] = features[donor][field]
    return value, donors


def read_crops(root, split, ids):
    result = {}
    for sid in ids:
        p = root/split/'crop_variants'/(key(sid)+'.npz'); meta = read(p.with_suffix('.json'))
        if sha(p) != meta['arrays_sha256']:
            raise ValueError('Crop intervention arrays changed')
        with np.load(p, allow_pickle=False) as value:
            result[sid] = {k: value[k] for k in ('pad', 'stretch', 'windows')}
    return result


def trial_labels(labels, variant):
    field = 'primary' if variant['auxiliary_labels'] == 'lexical' else 'auxiliary'
    return {sid: {'word_labels': values[field]} for sid, values in labels.items()}


def source_reuse(plan, root, fit, held, name, fold, seed, feature, cfg):
    """Reuse the already audited identical first-seed baseline; verify replay."""
    source = Path(plan['paths']['source_root']); frozen = read(source/'frozen_choices.json')
    source_plan = read(source/'resolved_configuration.json')
    original = next(t for t in frozen['fold_trace'] if t['population'] == 'full' and t['fold'] == fold)
    if original['training_ids'] != fit or original['held_ids'] != held:
        return None
    checkpoints = {}; source_scores = read(source/'cv_predictions.json')['full']['token_context']
    for epoch in cfg['config']['training']['checkpoints']:
        item = frozen['cv_models']['full'+str(fold)+'__token_context__'+str(epoch)]
        src = source/item['model_path']
        if sha(src) != item['model_sha256']:
            raise ValueError('Qualified baseline model changed')
        saved = torch.load(src, map_location='cpu', weights_only=True)
        if saved['preprocessing']['seed'] != seed:
            return None
        saved['reused_source_model'] = {'path': str(src), 'sha256': sha(src), 'original_plan_sha256': saved['plan_sha256']}
        saved['plan_sha256'] = sha(root/'resolved_configuration.json')
        target = root/'models'/(name+'__epoch'+str(epoch)+'.pt')
        if not target.exists():
            save_model(target, saved)
        model, preprocessing = load_model(cfg, root, str(target.relative_to(root)))
        scores, predicted = score_model(model, preprocessing, feature, {}, held, cfg['config'])
        for sid in held:
            expected = source_scores[str(epoch)][sid]
            if (expected is None) != (scores[sid] is None) or (expected is not None and abs(expected-scores[sid]) > plan['config']['replay_tolerance']):
                raise ValueError('Baseline neural score replay differs')
        checkpoints[str(epoch)] = {'scores': scores, 'count_predictions': predicted,
            'model_path': str(target.relative_to(root)), 'model_sha256': sha(target), 'reused_source': str(src)}
        del model
    return checkpoints


def train(plan, root):
    if (root/'frozen_choices.json').exists():
        return
    if list((root/'test/crop_variants').glob('*.npz')):
        raise ValueError('Fresh TEST crop arrays precede model-choice freeze')
    cfg = plan['config']; features, rows, labels = inputs(plan, root, 'train'); ids = sorted(features)
    samples = {s['sample_id']: s for s in plan['samples']['train']}; crops = read_crops(root, 'train', ids)
    source = Path(plan['paths']['source_root']); raw = read(source/'train/branch_metrics.json')['branches']; branches = {s: raw[s] for s in ids}
    folds = list(GroupKFold(n_splits=cfg['cv_folds']).split(ids, groups=[samples[s]['rgb_sha256'] for s in ids])); trace = []
    for fold, (a, b) in enumerate(folds):
        fit, held = [ids[i] for i in a], [ids[i] for i in b]
        trace.append({'fold': fold, 'fit_ids': fit, 'held_ids': held,
            'fit_rgb_sha256': [samples[s]['rgb_sha256'] for s in fit], 'held_rgb_sha256': [samples[s]['rgb_sha256'] for s in held]})
    choices = {'plan_sha256': sha(root/'resolved_configuration.json'), 'test_used_for_selection': False,
        'test_crop_arrays_present_at_freeze': 0, 'fold_trace': trace, 'trials': {}}
    for variant in cfg['variants']:
        model_cfg = model_plan(plan, variant)
        for seed_index in variant['seed_indices']:
            if seed_index >= len(cfg['seeds']):
                continue
            seed = cfg['seeds'][seed_index]; name = variant['name']+'__seed'+str(seed)
            done = root/'training_trials'/(name+'.json')
            if done.exists():
                choices['trials'][name] = read(done); continue
            oof = {str(e): {} for e in cfg['checkpoints']}; counts = {str(e): {} for e in cfg['checkpoints']}; receipts = []
            fitting = []
            for fold, (a, b) in enumerate(folds):
                candidates, held = [ids[i] for i in a], [ids[i] for i in b]
                fit = fitting_subset(candidates, samples, variant['training_fraction'], seed+fold)
                if len(fit) < 2:
                    raise ValueError('Intervention has fewer than two fitting images')
                observed, donors = intervention_features(features, variant, crops, fit, samples, seed+fold)
                target_branch = dict(branches)
                if variant['shuffle_labels']:
                    order = fit.copy(); random.Random(seed+fold).shuffle(order)
                    if order == fit:
                        order = order[1:]+order[:1]
                    for s, donor in zip(fit, order):
                        target_branch[s] = branches[donor]
                prefix = name+'__fold'+str(fold)
                checkpoints = None
                if variant['name'] == 'baseline' and seed_index == 0 and plan['result_class'] != 'SMOKE_TEST':
                    checkpoints = source_reuse(plan, root, fit, held, prefix, fold, seed+fold, observed, model_cfg)
                if checkpoints is None:
                    checkpoints = fit_router(model_cfg, root, observed, {}, trial_labels(labels, variant), target_branch,
                        fit, 'token_context', prefix, seed+fold, max(cfg['checkpoints']), held)
                fitting.append({'fold': fold, 'fit_ids': fit, 'held_ids': held, 'donor_pool_is_fit_only': True,
                    'donor_map': {s: donors[s] for s in held} if donors else {}, 'shuffled_training_labels': variant['shuffle_labels']})
                for epoch, found in checkpoints.items():
                    oof[epoch].update(found['scores']); counts[epoch].update(found['count_predictions'])
                    receipts.append({k: found[k] for k in ('model_path', 'model_sha256')})
                del observed; gc.collect(); torch.cuda.empty_cache()
            points = {e: point(branches, ids, fixed_budget(scores, ids, cfg['primary_bypass_fraction'])) for e, scores in oof.items()}
            epoch = int(max(points, key=lambda e: (points[e]['combined']['f1'], -int(e))))
            fit = fitting_subset(ids, samples, variant['training_fraction'], seed)
            observed, donors = intervention_features(features, variant, crops, fit, samples, seed)
            target_branch = dict(branches)
            if variant['shuffle_labels']:
                order = fit[1:]+fit[:1]
                for s, donor in zip(fit, order):
                    target_branch[s] = branches[donor]
            final = fit_router(model_cfg, root, observed, {}, trial_labels(labels, variant), target_branch,
                fit, 'token_context', name+'__final', seed, epoch, ids)[str(epoch)]
            receipts.append({k: final[k] for k in ('model_path', 'model_sha256')})
            value = {'variant': variant, 'seed': seed, 'epoch': epoch, 'oof_points': points, 'fold_fitting': fitting,
                'oof_scores': oof[str(epoch)], 'oof_count_predictions': counts[str(epoch)],
                'final': final, 'fitting_ids': fit, 'checkpoint_receipts': receipts}
            seal(done, value); choices['trials'][name] = value
            print(f'[trained] {name} epoch={epoch} OOF_F1={100*points[str(epoch)]["combined"]["f1"]:.4f}', flush=True)
            del observed; gc.collect(); torch.cuda.empty_cache()
    seal(root/'frozen_choices.json', choices)
    print('[frozen] every epoch chosen with TRAIN OOF; TEST crop interventions may start', flush=True)


def evaluate_models(plan, root):
    if (root/'model_results.json').exists():
        return
    cfg = plan['config']; frozen = read(root/'frozen_choices.json'); source = Path(plan['paths']['source_root'])
    data = {split: inputs(plan, root, split) for split in ('train', 'test')}
    all_features = {**data['train'][0], **data['test'][0]}
    samples = {s['sample_id']: s for ss in plan['samples'].values() for s in ss}
    crop = {**read_crops(root, 'train', list(data['train'][0])), **read_crops(root, 'test', list(data['test'][0]))}
    trials = {}; deletion_tests = {}
    for name, choice in frozen['trials'].items():
        variant = choice['variant']; observed, donors = intervention_features(all_features, variant, crop, choice['fitting_ids'], samples, choice['seed'])
        model_cfg = model_plan(plan, variant); model, preprocess = load_model(model_cfg, root, choice['final']['model_path'])
        value = {'variant': variant, 'seed': choice['seed'], 'epoch': choice['epoch'], 'oof_point': choice['oof_points'][str(choice['epoch'])],
            'checkpoint_receipts': choice['checkpoint_receipts']}
        for split in ('train', 'test'):
            ids = sorted(data[split][0]); raw = read(source/split/'branch_metrics.json')['branches']; bb = {s: raw[s] for s in ids}
            scores, counts = score_model(model, preprocess, observed, {}, ids, model_cfg['config'])
            value[split] = {'point': point(bb, ids, fixed_budget(scores, ids, cfg['primary_bypass_fraction'])),
                'scores': scores, 'count_predictions': counts}
        fit_bb = read(source/'train/branch_metrics.json')['branches']
        value['fit_point'] = point(fit_bb, choice['fitting_ids'], fixed_budget(value['train']['scores'], choice['fitting_ids'], cfg['primary_bypass_fraction']))
        if variant['name'] in cfg['deletion_model_variants'] and choice['seed'] == cfg['seeds'][0]:
            probes = []
            for split in ('train', 'test'):
                features, rows, labels = data[split]
                for sid in sorted(features):
                    omissions = choose_omissions({'primary': labels[sid]['primary']}, rows[sid])
                    if omissions is None:
                        continue
                    original = value[split]['scores'][sid]
                    for kind, removed in omissions.items():
                        altered, _ = omit_features(observed[sid], rows[sid], removed, samples[sid]['image_wh'])
                        scores, _ = score_model(model, preprocess, {sid: altered}, {}, [sid], model_cfg['config'])
                        probes.append({'split': split, 'sample_id': sid, 'kind': kind, 'removed_rows': removed,
                            'known_extra_FN': len(removed), 'original_score': original, 'altered_score': scores[sid],
                            'score_drop': original-scores[sid] if original is not None and scores[sid] is not None else None,
                            'uses_GT_to_construct_intervention': True})
            deletion_tests[name] = probes
        trials[name] = value; del model, observed; gc.collect(); torch.cuda.empty_cache()
        print('[evaluated] '+name, flush=True)
    seal(root/'deletion_tests.json', {'tests': deletion_tests, 'actual_OCR_words_changed': False,
        'interventions': 'privileged diagnostic copies; correct-word removal, image unchanged; likelihood not used by these token models'})
    seal(root/'model_results.json', {'plan_sha256': sha(root/'resolved_configuration.json'), 'test_used_for_model_selection': False,
        'trials': trials, 'base_words_unchanged': True})
