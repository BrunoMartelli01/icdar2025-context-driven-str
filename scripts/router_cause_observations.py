"""CPU controls, fixed-budget tests and bounded reporting for cause interventions."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import math
import re

import numpy as np
import torch

from recognition.router_cause_tests import row_supervision, fixed_budget, utility
from recognition.trainable_image_router import oracle
from scripts.token_context_data import annotations, key
from scripts.run_yolo_f1_quality_study import read, sha, seal
from scripts.token_context_training import point


def cluster_rate_interval(values, groups, replicates, seed):
    unique = sorted(set(groups)); sums = []; counts = []
    for g in unique:
        vv = [v for v, group in zip(values, groups) if group == g]
        sums.append(sum(vv)); counts.append(len(vv))
    sums, counts = np.asarray(sums), np.asarray(counts); rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(replicates, len(unique)))
    draws = sums[indices].sum(1)/counts[indices].sum(1)
    return {'rate': float(np.mean(values)), 'lower95': float(np.quantile(draws, .025)),
            'upper95': float(np.quantile(draws, .975)), 'draws': draws.tolist()}


def observations(plan, root):
    if (root/'observations.json').exists():
        return
    source = Path(plan['paths']['source_root']); cfg = plan['config']; all_results = read(source/'results.json')
    branch = {}; metadata = {}; summaries = {}; rate_trials = {}
    reference = all_results['splits']['train']['deepsolo_only']['f1']
    for split, ss in plan['samples'].items():
        source_branch = read(source/split/'branch_metrics.json')['branches']
        branch[split] = {s['sample_id']: source_branch[s['sample_id']] for s in ss}
        eligible = []; gains = []; groups = []; scalars = []; no_center = words = mismatch = tp = 0
        for sample in ss:
            sid = sample['sample_id']; stem = key(sid); rp = source/split/'light'/(stem+'.json'); rows = read(rp)['rows']
            gt = annotations(sample); label = row_supervision(sample, rows, gt, plan['base_config']['metric']['native_ignore_size'], plan['base_config']['dense_training']['auxiliary_iou'])
            if label['metrics'] != source_branch[sid]['light']:
                raise ValueError('Care/ignore/matching replay differs')
            seal(root/split/'labels'/(stem+'.json'), {'source_rows_sha256': sha(rp), 'primary': label['primary'].tolist(),
                'auxiliary': label['auxiliary'].tolist(), 'metrics': label['metrics'],
                'ground_truth_use': 'fitting labels or privileged offline diagnostic only'})
            with np.load(source/split/'features'/(stem+'.npz'), allow_pickle=False) as a:
                xy = a['patch_geometry'][a['patch_valid'], :2]*sample['image_wh']; scalar = a['scalars'].copy()
            for g in gt:
                x, y, xx, yy = g['bbox']; good = (xy[:, 0] >= x)&(xy[:, 0] < xx)&(xy[:, 1] >= y)&(xy[:, 1] < yy)
                no_center += int(not good.any()); words += 1
            mismatch += int(((label['primary'] == 1)&(label['auxiliary'] == 0)).sum()); tp += label['metrics']['tp']
            meta = read(source/split/'features'/(stem+'.json'))
            if meta['eligible']:
                eligible.append(sid); gains.append(utility(source_branch[sid], reference)); groups.append(sample['rgb_sha256']); scalars.append(scalar)
        metadata[split] = {'eligible': eligible, 'gains': gains, 'scalars': np.asarray(scalars)}
        trials = cluster_rate_interval([float(x > 0) for x in gains], groups, cfg['bootstrap_replicates'], cfg['seeds'][0]+(split == 'test'))
        rate_trials[split] = np.array(trials.pop('draws'))
        summaries[split] = {'images': len(ss), 'eligible_images': len(eligible), 'positive_common_reference': trials,
            'care_GT_words': words, 'GT_words_without_patch_center': no_center,
            'TP_with_auxiliary_negative_under_declared_assignment': mismatch, 'primary_TP': tp,
            'prior_reference_controls': {name: {'reference_f1': ref, 'positive': sum(utility(source_branch[s], ref) > 0 for s in eligible),
                'neutral': sum(abs(utility(source_branch[s], ref)) < 1e-12 for s in eligible)}
                for name, ref in [('train', reference), ('test', all_results['splits']['test']['deepsolo_only']['f1']), ('fixed', cfg['fixed_utility_reference'])]}}
    difference = rate_trials['test']-rate_trials['train']
    shift = {'common_reference_f1': reference, 'test_minus_train_positive_rate_pp': 100*(summaries['test']['positive_common_reference']['rate']-summaries['train']['positive_common_reference']['rate']),
        'lower95_pp': float(100*np.quantile(difference, .025)), 'upper95_pp': float(100*np.quantile(difference, .975)),
        'expert_exposure_causal_attribution': 'UNRESOLVED: no expert-training counterfactual or independent unseen-expert dataset', 'matched_observable_strata': {}}
    for features in cfg['matching_scalar_indices']:
        edges = [np.unique(np.quantile(metadata['train']['scalars'][:, i], cfg['matching_quantiles'])) for i in features]
        cells = {}
        for split in ('train', 'test'):
            coords = [tuple(int(np.searchsorted(edge, row[index], side='right')) for edge, index in zip(edges, features)) for row in metadata[split]['scalars']]
            cells[split] = defaultdict(list)
            for cell, gain in zip(coords, metadata[split]['gains']):
                cells[split][cell].append(gain > 0)
        weighted = covered = test_pos = 0.
        for cell, vv in cells['test'].items():
            if cell in cells['train']:
                weighted += len(vv)*np.mean(cells['train'][cell]); covered += len(vv); test_pos += sum(vv)
        shift['matched_observable_strata'][str(features)] = {'test_images_with_train_support': int(covered),
            'reweighted_train_positive_rate': float(weighted/covered), 'test_positive_rate_on_supported_cells': float(test_pos/covered),
            'stratum_edges_train_only': [v.tolist() for v in edges], 'causal_identification': False}
    budgets = {}; cv = read(source/'cv_predictions.json')['full']
    for split in ('train', 'test'):
        ids = sorted(branch[split]); eligible = metadata[split]['eligible']; budgets[split] = {}
        for arm in cfg['diagnostic_arms']:
            epoch = str(read(source/'frozen_choices.json')['arms'][arm]['epoch'])
            prediction = read(source/'predictions'/('test__'+arm+'__evaluated.json')) if split == 'test' else None
            scores = prediction['scores'] if prediction else cv[arm][epoch]
            scores = {s: scores[s] for s in ids}; curves = {}
            for q in cfg['budget_fractions']:
                accepted = fixed_budget(scores, ids, q); p = point(branch[split], ids, accepted)
                p['oracle'] = oracle(branch[split], ids, eligible, len(accepted)); curves[str(q)] = p
            chosen = max(curves, key=lambda q: (curves[q]['combined']['f1'], -float(q)))
            positive = [s for s in ids if scores[s] is not None and scores[s] > 0]
            invariant = []
            for slope in cfg['affine_slopes']:
                for offset in cfg['affine_offsets']:
                    transformed = {s: None if v is None else slope*v+offset for s, v in scores.items()}
                    same = fixed_budget(transformed, ids, cfg['primary_bypass_fraction']) == fixed_budget(scores, ids, cfg['primary_bypass_fraction'])
                    if not same:
                        raise ValueError('Positive affine calibration unexpectedly changed fixed-budget ranking')
                    invariant.append({'slope': slope, 'offset': offset, 'same_selected_images': same})
            item = {'curves': curves, 'zero_utility_gate': point(branch[split], ids, positive), 'affine_invariance': invariant,
                'best_observed_fraction': float(chosen), 'fraction_selection_scope': 'TRAIN OOF' if split == 'train' else 'diagnostic TEST curve only; never used for deployment selection'}
            if prediction:
                ablations = {}
                for which in ('true_deltaTP', 'true_deltaPred', 'both_true'):
                    values = {}
                    for s in ids:
                        if scores[s] is None:
                            values[s] = None; continue
                        dt, dn = prediction['count_predictions'][s]
                        if which in ('true_deltaTP', 'both_true'):
                            dt = branch[split][s]['light']['tp']-branch[split][s]['deepsolo']['tp']
                        if which in ('true_deltaPred', 'both_true'):
                            dn = branch[split][s]['light']['detected_words']-branch[split][s]['deepsolo']['detected_words']
                        values[s] = 2*dt-reference*dn
                    ablations[which] = {**point(branch[split], ids, fixed_budget(values, ids, cfg['primary_bypass_fraction'])), 'uses_ground_truth': True, 'deployable': False}
                item['privileged_count_ablations'] = ablations
            budgets[split][arm] = item
    for arm in cfg['diagnostic_arms']:
        q = str(budgets['train'][arm]['best_observed_fraction'])
        budgets['test'][arm]['train_selected_fraction'] = float(q)
        budgets['test'][arm]['train_selected_point'] = budgets['test'][arm]['curves'][q]
    seal(root/'observations.json', {'plan_sha256': sha(root/'resolved_configuration.json'), 'reproduction_mode': 'approximate',
        'splits': summaries, 'distribution_shift': shift, 'budget_tests': budgets, 'GT_used_for_inference_features': False,
        'calibration_only_fixed_quota_hypothesis': 'FALSIFIED for strictly monotone affine recalibration: selected images and F1 are invariant'})
    print('[CPU] distribution, auxiliary/coverage, budget and privileged-count controls complete', flush=True)


def finish(plan, root):
    from scripts.router_cause_gpu import crop_report, probe_report
    crop = crop_report(plan, root); probes = probe_report(plan, root)
    seal(root/'results.json', {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': plan['result_class'], 'plan_sha256': sha(root/'resolved_configuration.json'),
        'base_word_outputs_changed': False, 'observations': read(root/'observations.json'),
        'matched_training': read(root/'model_results.json'), 'crop_tests': crop, 'omission_tests': probes,
        'causal_claim_limit': 'Expert exposure attribution and any untested architecture/model changes remain unresolved; synthetic omissions are privileged offline controls'})


def audit_outputs(plan, root):
    results = read(root/'results.json'); frozen = read(root/'frozen_choices.json')
    if results['plan_sha256'] != sha(root/'resolved_configuration.json') or frozen['test_used_for_selection']:
        raise ValueError('Final plan/selection identity mismatch')
    for file, digest in plan['code_sha256'].items():
        if sha(root/'code_snapshot'/file) != digest:
            raise ValueError('Frozen test source changed')
    for split, ss in plan['samples'].items():
        receipt = read(root/split/'crop_receipt.json')
        if receipt['model']['model_state_sha256_before'] != receipt['model']['model_state_sha256_after']:
            raise ValueError('CLIP weights changed')
        for s in ss:
            p = root/split/'crop_variants'/(key(s['sample_id'])+'.npz'); meta = read(p.with_suffix('.json'))
            if sha(p) != meta['arrays_sha256'] or meta['ground_truth_used_for_embeddings']:
                raise ValueError('Intervention feature bytes or GT boundary changed')
            if meta['frozen_choices_sha256'] != (sha(root/'frozen_choices.json') if split == 'test' else None):
                raise ValueError('TEST intervention preceded the frozen choices')
    seen = Counter()
    for fold in frozen['fold_trace']:
        if set(fold['fit_ids'])&set(fold['held_ids']) or set(fold['fit_rgb_sha256'])&set(fold['held_rgb_sha256']):
            raise ValueError('Fold or pixel leakage')
        seen.update(fold['held_ids'])
    if set(seen) != {s['sample_id'] for s in plan['samples']['train']} or any(v != 1 for v in seen.values()):
        raise ValueError('OOF population incomplete')
    for name, value in read(root/'model_results.json')['trials'].items():
        choice = frozen['trials'][name]
        if set(choice['oof_scores']) != set(seen):
            raise ValueError('Trial OOF prediction population changed')
        selected_epoch = max(choice['oof_points'], key=lambda e: (choice['oof_points'][e]['combined']['f1'], -int(e)))
        if choice['epoch'] != int(selected_epoch):
            raise ValueError('Checkpoint was not selected by the declared TRAIN OOF rule')
        for fold in choice['fold_fitting']:
            if set(fold['fit_ids'])&set(fold['held_ids']) or any(d not in fold['fit_ids'] for d in fold['donor_map'].values()):
                raise ValueError('Fitting or context-donor leakage')
        for split in ('train', 'test'):
            source = Path(plan['paths']['source_root']); bb = read(source/split/'branch_metrics.json')['branches']
            ids = sorted(s['sample_id'] for s in plan['samples'][split]); pp = value[split]['point']
            accepted = set(pp['accepted_ids']); selected = [bb[s]['light' if s in accepted else 'deepsolo'] for s in ids]
            tp = sum(x['tp'] for x in selected); den = sum(x['detected_words']+x['gt_words'] for x in selected)
            if pp['combined']['tp'] != tp or pp['combined']['f1'] != float(Fraction(2*tp, den)):
                raise ValueError('Independent rational-count replay differs')
        for checkpoint in value['checkpoint_receipts']:
            if sha(root/checkpoint['model_path']) != checkpoint['model_sha256']:
                raise ValueError('Trial model changed')
            saved = torch.load(root/checkpoint['model_path'], map_location='cpu', weights_only=True)
            match = re.search(r'__fold(\d+)__epoch', checkpoint['model_path'])
            if match:
                fold = int(match.group(1)); scope = choice['fold_fitting'][fold]['fit_ids']; seed = choice['seed']+fold
            else:
                scope, seed = choice['fitting_ids'], choice['seed']
            if saved['plan_sha256'] != sha(root/'resolved_configuration.json') or saved['preprocessing']['training_ids'] != scope or saved['preprocessing']['seed'] != seed:
                raise ValueError('Actual checkpoint fitting scope, seed or plan differs')
    seal(root/'independent_audit.json', {'status': 'PASS', 'plan_sha256': sha(root/'resolved_configuration.json'),
        'results_sha256': sha(root/'results.json'), 'scope': 'Snapshot/feature hashes, frozen-model state, OOF population and pixel disjointness, declared model hashes and independent rational-count routing replay; no independent neural refit'})
