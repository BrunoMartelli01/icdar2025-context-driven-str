"""Offline paired analysis of completed approximate router cause tests."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
import torch

from recognition.token_context_router import BenefitRouter, base_features
from scripts.run_yolo_f1_quality_study import read, sha, seal

REPO = Path(__file__).resolve().parents[1]


def count_array(branches, ids, accepted):
    accepted = set(accepted)
    if not accepted <= set(ids):
        raise ValueError('Unknown accepted images')
    return np.array([[branches[s]['light' if s in accepted else 'deepsolo'][k]
        for k in ('tp', 'detected_words', 'gt_words')] for s in ids], np.float64)


def f1(counts):
    return 2*counts[..., 0]/(counts[..., 1]+counts[..., 2])


def paired_interval(left, right, groups, settings):
    """Matched image-cluster bootstrap, with empirical matched-seed resampling."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[1] != len(groups):
        raise ValueError('Matched seed/image/count arrays required')
    unique = sorted(set(groups)); mapping = np.array([unique.index(g) for g in groups])
    rng = np.random.default_rng(settings['seed']); draws = []
    for _ in range(settings['bootstrap_replicates']):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[mapping]
        a = f1((left*weights[None, :, None]).sum(1)); b = f1((right*weights[None, :, None]).sum(1))
        seeds = rng.integers(0, len(a), len(a)) if len(a) > 1 else [0]
        draws.append(float(100*(a[seeds]-b[seeds]).mean()))
    alpha = 1-settings['confidence_level']; point = float(100*(f1(left.sum(1))-f1(right.sum(1))).mean())
    lo, hi = np.quantile(draws, [alpha/2, 1-alpha/2])
    return {'difference_pp': point, 'lower95_pp': float(lo), 'upper95_pp': float(hi),
        'matched_seeds': len(left), 'images': left.shape[1], 'support': 'positive' if lo > 0 else 'negative' if hi < 0 else 'inconclusive'}


def mean_interval(values, settings):
    values = np.array(values, np.float64); rng = np.random.default_rng(settings['seed'])
    if not len(values):
        return None
    draws = values[rng.integers(0, len(values), (settings['bootstrap_replicates'], len(values)))].mean(1)
    alpha = 1-settings['confidence_level']; lo, hi = np.quantile(draws, [alpha/2, 1-alpha/2])
    return {'images': len(values), 'mean': float(values.mean()), 'lower95': float(lo), 'upper95': float(hi)}


def direction_interval(values, tolerance, confidence_level):
    a = np.array(values); n = len(a)
    if not n:
        return None
    p = float(np.mean(a > tolerance)); z = norm.ppf((1+confidence_level)/2)
    center = (p+z*z/(2*n))/(1+z*z/n); radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'images': n, 'positive_direction_fraction': p, 'lower95': float(center-radius), 'upper95': float(center+radius)}


def ordered_likelihood(values, names):
    if len(names) != len(set(names)) or set(values) != set(names):
        raise ValueError('Likelihood feature schema changed')
    return np.array([values[name] for name in names], np.float32)


def isolated_blip_fields(plan, root, settings):
    source = Path(plan['paths']['source_root']); frozen = read(source/'frozen_choices.json'); item = frozen['arms']['likelihood']
    if sha(source/item['model_path']) != item['model_sha256']:
        raise ValueError('Original likelihood head checkpoint changed')
    saved = torch.load(source/item['model_path'], map_location='cpu', weights_only=True); p = saved['preprocessing']
    head = BenefitRouter(p['architecture'], plan['base_config']['architecture']['scalar_dim'], p['extra_dim'], p['arm'])
    head.load_state_dict(saved['state_dict'], strict=True); head.requires_grad_(False).eval()
    mu, sd = np.array(p['mean'], np.float32), np.array(p['scale'], np.float32)
    def score(scalars, likelihood):
        x = (np.concatenate([scalars, likelihood])-mu)/sd; n = len(scalars)
        with torch.inference_mode():
            pred, _ = head({'scalars': torch.tensor(x[:n][None]), 'extra': torch.tensor(x[n:][None])})
        pred = pred[0].numpy()*p['target_scale']+np.array(p['target_mean'])
        return float(2*pred[0]-p['reference_f1']*pred[1])
    cases = []; original_errors = []
    for split, records in read(root/'blip_probe_receipt.json')['files'].items():
        samples = {s['sample_id']: s for s in plan['samples'][split]}
        for record in records:
            value = read(root/record['file']); sid = value['sample_id']; stem = sid.replace(':', '__')
            rows = read(source/split/'light'/(stem+'.json'))['rows']
            with np.load(source/split/'features'/(stem+'.npz'), allow_pickle=False) as original:
                scalars, likelihood = original['scalars'], original['likelihood']
            baseline = score(scalars, likelihood); original_errors.append(abs(baseline-value['tests'][0]['score']))
            if not np.isclose(baseline, value['tests'][0]['score'], atol=settings['cpu_head_replay_tolerance'], rtol=settings['cpu_head_replay_tolerance']):
                raise ValueError('CPU likelihood head replay differs')
            for case in value['tests'][1:]:
                deleted = set(case['deleted_row_indices']); remaining = [r for i, r in enumerate(rows) if i not in deleted]
                ss = base_features(remaining, samples[sid]['image_wh'])
                # JSON sorts feature names; explicitly restore the source extraction order.
                source_names = read(source/split/'blip'/(stem+'.json'))['raw']['feature_names']
                ll = ordered_likelihood(case['likelihood_features'], source_names)
                full = baseline-score(ss, ll); semantic = baseline-score(scalars, ll); scalar = baseline-score(ss, likelihood)
                cases.append({'split': split, 'sample_id': sid, 'condition': case['name'], 'known_extra_FN': case['known_extra_FN'],
                    'full_score_drop': full, 'likelihood_only_drop_fixed_scalars': semantic,
                    'scalar_only_drop_fixed_likelihood': scalar, 'nonlinear_interaction_drop': full-semantic-scalar,
                    'word_order_changed': case['language_order_only'] and [r['original']['text'] for r in case['global_scores']] != [r['original']['text'] for r in value['tests'][0]['global_scores']]})
    summary = {}; grouped = defaultdict(list)
    for case in cases:
        grouped[(case['split'], case['condition'])].append(case)
    for (split, condition), cases_in_group in grouped.items():
        summary[split+'__'+condition] = {}
        for field in ('full_score_drop', 'likelihood_only_drop_fixed_scalars', 'scalar_only_drop_fixed_likelihood', 'nonlinear_interaction_drop'):
            vv = [r[field] for r in cases_in_group]
            summary[split+'__'+condition][field] = {**mean_interval(vv, settings), 'direction': direction_interval(vv, settings['score_tie_tolerance'], settings['confidence_level'])}
    return {'scope': 'Additional predeclared offline sensitivity: likelihood/scalar interventions on a frozen CPU head, no model fitting; nonlinear contributions are not additive causal fractions',
        'maximum_cpu_replay_error': max(original_errors, default=0.), 'summary': summary, 'cases': cases}


def analyze(args):
    if args.mode != 'approximate' or args.config.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit isolated approximate analysis config required')
    settings = read(args.config); root = args.run_root.resolve(); plan = read(root/'resolved_configuration.json')
    result = read(root/'results.json'); audit = read(root/'independent_audit.json')
    if settings['reproduction_mode'] != 'approximate' or audit['status'] != 'PASS' or sha(root/'results.json') != audit['results_sha256']:
        raise ValueError('Completed independently audited approximate test run required')
    for file in ('recognition/token_context_router.py',):
        if sha(REPO/file) != plan['code_sha256'][file]:
            raise ValueError('Frozen head replay implementation changed')
    torch.set_num_threads(settings['cpu_threads'])
    source = Path(plan['paths']['source_root']); models = result['matched_training']['trials']; rows = []; by_variant = defaultdict(list)
    for name, value in models.items():
        variant = value['variant']['name']; seed = value['seed']
        row = {'trial': name, 'variant': variant, 'seed': seed, 'epoch': value['epoch'],
            'fit_F1': value['fit_point']['combined']['f1'], 'OOF_F1': value['oof_point']['combined']['f1'],
            'TEST_F1': value['test']['point']['combined']['f1'],
            'fit_minus_OOF_pp': 100*(value['fit_point']['combined']['f1']-value['oof_point']['combined']['f1'])}
        rows.append(row); by_variant[variant].append(row)
    group_summary = {name: {field: {'mean': float(np.mean([r[field] for r in rr])), 'std_across_seeds': float(np.std([r[field] for r in rr]))}
        for field in ('fit_F1', 'OOF_F1', 'TEST_F1', 'fit_minus_OOF_pp')} for name, rr in by_variant.items()}
    comparisons = {}
    for variant, rr in by_variant.items():
        if variant == 'baseline':
            continue
        reference = 'aux_off' if variant == 'label_shuffle' else 'baseline'; matches = {r['seed']: r for r in by_variant[reference]}
        pairs = [(r, matches[r['seed']]) for r in rr if r['seed'] in matches]; effects = {}
        for split in ('oof', 'test'):
            source_split = 'train' if split == 'oof' else 'test'; samples = {s['sample_id']: s for s in plan['samples'][source_split]}
            ids = sorted(samples); branch = read(source/source_split/'branch_metrics.json')['branches']; left = []; right = []
            for a, b in pairs:
                av, bv = models[a['trial']], models[b['trial']]
                ap = av['oof_point'] if split == 'oof' else av['test']['point']; bp = bv['oof_point'] if split == 'oof' else bv['test']['point']
                left.append(count_array(branch, ids, ap['accepted_ids'])); right.append(count_array(branch, ids, bp['accepted_ids']))
            effects[split] = paired_interval(np.stack(left), np.stack(right), [samples[s]['rgb_sha256'] for s in ids], settings)
        comparisons[variant+'__minus__'+reference] = effects
    field_probes = isolated_blip_fields(plan, root, settings)
    output = {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'result_class': 'EXPLORATORY_CONTROLLED_CAUSE_ANALYSIS', 'run_root': str(root), 'results_sha256': sha(root/'results.json'),
        'audit_sha256': sha(root/'independent_audit.json'), 'analysis_config_sha256': sha(args.config), 'analysis_code_sha256': sha(__file__),
        'conditional_inference': 'Paired cluster and empirical matched-seed bootstrap; fixed selected models/routes, no refitting; no multiplicity correction; historically explored TEST',
        'trial_summary': rows, 'variant_summary': group_summary, 'paired_contrasts': comparisons,
        'BLIP_field_controls': field_probes,
        'interpretation_limits': ['Learning curves hold epochs rather than optimizer steps constant.',
            'Random-label negative-control OOF fits use seeded random permutations; its final refit uses a cyclic permutation. Interpret random-label evidence primarily on OOF.',
            'Expert-training exposure attribution remains unresolved.', 'Synthetic omission probes are a privileged partial-dataset subset.',
            'Crop remedies also change scale/distortion; lost pixels alone do not prove a particular F1 loss.']}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); seal(out/'analysis.json', output)
    lines = ['# Controlled router cause tests: completed analysis', '', 'Approximate reproduction; exact lexical micro word F1, not official polygon spotting or paper Table5 accuracy.', '',
        '| Variant | Seeds | Mean TRAIN fit F1 | Mean TRAIN OOF F1 | Mean TEST F1 | Fit–OOF pp |', '|---|---:|---:|---:|---:|---:|']
    for name, value in group_summary.items():
        lines.append(f"|{name}|{len(by_variant[name])}|{100*value['fit_F1']['mean']:.4f}|{100*value['OOF_F1']['mean']:.4f}|{100*value['TEST_F1']['mean']:.4f}|{value['fit_minus_OOF_pp']['mean']:.4f}|")
    lines += ['', '## Matched contrasts', '', '| Variant − reference | OOF delta pp [95%] | TEST delta pp [95%] |', '|---|---:|---:|']
    for name, values in comparisons.items():
        rendered = [f"{values[s]['difference_pp']:+.4f} [{values[s]['lower95_pp']:+.4f}, {values[s]['upper95_pp']:+.4f}]" for s in ('oof', 'test')]
        lines.append('|'+name+'|'+'|'.join(rendered)+'|')
    lines += ['', 'Intervals are conditional on fitted models, selected checkpoints and fixed routes; multiple explored contrasts are not multiplicity-adjusted. Three-seed variation is descriptive, not an independent replication dataset.', '',
        '## Likelihood versus scalar-count sensitivity', '', '| Split / condition | Full score drop | Likelihood-only drop | Scalar-only drop |', '|---|---:|---:|---:|']
    for name, values in field_probes['summary'].items():
        fields = ['full_score_drop', 'likelihood_only_drop_fixed_scalars', 'scalar_only_drop_fixed_likelihood']
        lines.append('|'+name+'|'+'|'.join(f"{values[f]['mean']:+.5f}" for f in fields)+'|')
    lines += ['', 'Feature interventions are offline sensitivity controls; nonlinear head contributions are not additive fractions of physical causation.', '', '## Remaining limits', '']
    lines += ['- '+text for text in output['interpretation_limits']]
    text = '\n'.join(lines)+'\n'; p = out/'analysis.md'
    if p.exists() and p.read_text() != text:
        raise ValueError('Immutable analysis report changed')
    p.write_text(text)
    print(str(out/'analysis.json'))


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--mode', choices=['approximate'], required=True)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); analyze(parser.parse_args())


if __name__ == '__main__':
    main()
