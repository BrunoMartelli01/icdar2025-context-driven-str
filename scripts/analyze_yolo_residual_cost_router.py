"""Independently audit residual components/counts and export approximate results."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.audit_yolo_f1_quality_study import aggregate, check_point, f1, point


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def support(image_wh, boxes, side):
    w, h = image_wh; square = max(w, h)
    pixels = np.zeros((square, square), np.uint8)
    dx, dy = (square-w)//2, (square-h)//2
    for x1, y1, x2, y2 in boxes:
        pixels[y1+dy:y2+dy, x1+dx:x2+dx] = 1
    return np.asarray(Image.fromarray(pixels).resize((side, side), Image.Resampling.NEAREST)) > 0


def expected_regions(probabilities, wh, boxes, settings):
    p = np.asarray(probabilities); side = p.shape[1]; w, h = wh
    canvas = support(wh, [[0, 0, w, h]], side); covered = support(wh, boxes, side)
    masks = (p >= settings['probability_threshold']) & canvas[None]
    heads = {}
    for k, name in enumerate(('text', 'kernel')):
        values = p[k].astype(float); fg = masks[k]; missing = fg & ~covered
        labels, n = ndimage.label(fg, structure=np.ones((3, 3), np.uint8))
        components = []
        for label, slices in enumerate(ndimage.find_objects(labels), 1):
            if slices is None: continue
            local = labels[slices] == label; pixels = int(local.sum())
            if pixels < settings['minimum_component_area']: continue
            exposed = local & missing[slices]; exposed_pixels = int(exposed.sum())
            fraction = exposed_pixels/pixels
            residual = exposed_pixels >= settings['minimum_component_area'] and fraction >= settings['minimum_uncovered_fraction']
            other = masks[1-k][slices] & ~covered[slices]
            overlap = int((other & exposed).sum())/exposed_pixels if exposed_pixels else 0.
            yy, xx = slices
            components.append({'bbox': [xx.start, yy.start, xx.stop, yy.stop], 'pixels': pixels,
                'mass': float(values[slices][local].sum()), 'uncovered_pixels': exposed_pixels,
                'uncovered_fraction': fraction, 'uncovered_mass': float(values[slices][exposed].sum()),
                'other_head_support': overlap, 'residual': residual,
                'corroborated': residual and overlap >= settings['minimum_other_head_support']})
        components.sort(key=lambda c: (c['bbox'][1], c['bbox'][0], c['bbox'][3], c['bbox'][2]))
        weights = np.where(fg, values, 0.)
        masses = [float(weights[support(wh, [box], side)].sum()) for box in boxes]
        positive = [m for m in masses if m > 0]
        scale = float(np.median(positive)) if positive else float(np.median([c['mass'] for c in components])) if components else 1.
        candidates = [c for c in components if c['residual']]; valid = [c for c in components if c['corroborated']]
        heads[name] = {'components': components, 'foreground_pixels': int(fg.sum()),
            'foreground_mass': float(weights.sum()), 'coverage': float(weights[covered].sum()/weights.sum()) if weights.sum() else 1.,
            'uncovered_pixels': int(missing.sum()), 'word_foreground_masses': masses, 'word_mass_scale': scale,
            'word_mass_scale_source': 'positive_predicted_word_median' if positive else 'predicted_component_median' if components else 'empty_neutral',
            'missing_regions': len(candidates), 'corroborated_missing_regions': len(valid),
            'missing_word_mass_equivalents': sum(c['uncovered_mass'] for c in candidates)/scale,
            'corroborated_word_mass_equivalents': sum(c['uncovered_mass'] for c in valid)/scale}
    return {'map_side': side, 'map_shape': list(p.shape), 'map_dtype': str(p.dtype),
        'map_members_accessed': ['probabilities'], 'ground_truth_used': False, 'heads': heads,
        'canvas_pixels': int(canvas.sum()), 'covered_canvas_pixels': int(covered.sum()),
        'missing_count': max(h['missing_regions'] for h in heads.values()),
        'missing_mass': max(h['missing_word_mass_equivalents'] for h in heads.values()),
        'missing_consensus': sum(h['corroborated_missing_regions'] for h in heads.values())/2}


def close(actual, expected, location=''):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), location
        for key, value in expected.items(): close(actual[key], value, location+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), location
        for index, value in enumerate(expected): close(actual[index], value, location+'/'+str(index))
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (location, actual, expected)
    else:
        assert actual == expected, (location, actual, expected)


def audit(root):
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']; result = read(root/'results.json')
    ph = sha(root/'resolved_configuration.json'); frozen = read(root/'frozen_train_choices.json')
    assert result['plan_sha256'] == ph and result['frozen_choices_sha256'] == sha(root/'frozen_train_choices.json')
    assert result['status'] == 'SUCCESS' and result['optimizer_steps'] == 0
    assert not result['network_training'] and not result['new_model_inference'] and not result['table_5_claim']
    for file, digest in plan['code_sha256'].items(): assert sha(root/'code_snapshot'/file) == digest
    for path, digest in plan['input_sha256'].items(): assert sha(path) == digest
    source = Path(plan['paths']['source_study_root']); pins = cfg['source_study']
    for file, key in [('resolved_configuration.json', 'plan'), ('results.json', 'results'), ('independent_audit.json', 'audit')]:
        assert sha(source/file) == pins[key]
    old = read(source/'resolved_configuration.json'); coverage = Path(plan['source_coverage_root'])
    cpins = old['config']['sources']['coverage']
    assert sha(coverage/'resolved_configuration.json') == cpins['plan']
    assert sha(coverage/'results.json') == cpins['results']
    assert sha(coverage/cpins['audit_file']) == cpins['audit']
    branches = read(coverage/'results.json')['resolutions']['640']['branch_metrics']
    features = {}; components_verified = 0
    for split, samples in plan['samples'].items():
        features[split] = {}
        for sample in samples:
            sid = sample['sample_id']; key = sid.replace(':', '__')+'.json'
            value = read(root/split/'features'/key)
            digest = hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != 'payload_sha256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            assert value['payload_sha256'] == digest and value['plan_sha256'] == ph
            assert not value['ground_truth_used'] and not value['original_word_outputs_changed']
            assert not value['new_model_inference'] and not value['network_training']
            base = read(source/split/'features'/key)['scores']
            cov = read(coverage/'640'/split/'features'/key); rows = cov['rows']
            con = read(Path(old['paths']['sources']['consistency'])/split/'features'/key)
            with np.load(sample['map_path'], allow_pickle=False) as archive:
                probabilities = archive['probabilities']
            rebuilt = expected_regions(probabilities, sample['image_wh'], [r['bbox'] for r in rows], cfg['region_settings'])
            close(value['regions'], rebuilt, sid)
            components_verified += sum(len(h['components']) for h in rebuilt['heads'].values())
            stability = {v['candidate_id']: v['minimum_edit_agreement'] for v in con['crop_stability']['words']}
            ref = old['clip_reference']; words = []
            for r in rows:
                rank = (bisect_left(ref, r['clip_word_advantage'])+bisect_right(ref, r['clip_word_advantage']))/(2*len(ref))
                agreement = stability[r['candidate_id']]
                words.append({'candidate_id': r['candidate_id'], 'native': r['confidence'], 'clip_rank': rank,
                    'crop_agreement': agreement, 'visual': (r['confidence']*rank*agreement)**(1/3)})
            n = len(rows); native = math.fsum(1-w['native'] for w in words); visual = math.fsum(1-w['visual'] for w in words)
            empty = any(not r['raw_text'].strip() for r in rows)
            close(value['word_evidence'], {'words': words, 'predicted_word_count': n, 'has_empty_reading': empty,
                'native_error_proxy': native, 'visual_error_proxy': visual, 'probabilities_calibrated': False, 'ground_truth_used': False})
            c, m, s = (rebuilt[k] for k in ('missing_count', 'missing_mass', 'missing_consensus'))
            score_values = ([-c, -(2*native+c), -(2*visual+c), -(2*visual+m), -(2*visual+s),
                -(2*n*(1-base['all_checks_softcrop'])+s), 2*(n-visual)/(2*n+s), base['base_proxy_f1']/(1+s)]
                if n and not empty else [None]*8)
            expected = {**{f: base[f] for f in cfg['control_families']}, **dict(zip(cfg['new_families'], score_values))}
            close(value['scores'], expected); features[split][sid] = value
    def ranked(ids, source_split, family, fraction):
        return sorted((s for s in ids if features[source_split][s]['scores'][family] is not None),
            key=lambda s: (-features[source_split][s]['scores'][family], s))[:round(len(ids)*fraction)]
    sel, cal = plan['selection_ids'], plan['threshold_calibration_ids']
    assert not set(sel)&set(cal) and set(sel+cal) == set(features['train'])
    assert not set(features['train'])&set(features['test'])
    for fraction in cfg['bypass_fractions']:
        trials = []; fits = frozen['budgets'][str(fraction)]
        for i, family in enumerate(plan['families']):
            a = ranked(sel, 'train', family, fraction); p = point(branches['train'], sel, a, cfg['quality_f1_threshold'])
            trials.append((f1(p['combined']), -i, family))
            a = ranked(cal, 'train', family, fraction)
            cut = min((features['train'][s]['scores'][family] for s in a), default=None)
            assert fits['scalar_cuts'][family] == cut
        assert fits['selected_family'] == max(trials)[2]
    verified = 0
    populations = [('train', 'train', sorted(features['train'])), ('selection', 'train', sel),
        ('calibration', 'train', cal), ('test', 'test', sorted(features['test']))]
    for name, split, ids in populations:
        data = result['splits'][name]; b = branches[split]
        assert data['light_only'] == aggregate(b[s]['light'] for s in ids)
        assert data['deepsolo_only'] == aggregate(b[s]['deepsolo'] for s in ids)
        for fraction, budgets in data['budgets'].items():
            assert budgets['selected_family'] == frozen['budgets'][fraction]['selected_family']
            for family in plan['families']:
                for kind in ('families', 'scalar_thresholds'):
                    p = budgets[kind][family]
                    if kind == 'families':
                        accepted = ranked(ids, split, family, float(fraction))
                        assert p['requested_images'] == round(len(ids)*float(fraction))
                        assert p['unfilled_budget'] == p['requested_images']-len(accepted)
                    else:
                        cut = frozen['budgets'][fraction]['scalar_cuts'][family]
                        assert p['threshold'] == cut
                        accepted = [s for s in ids if cut is not None and features[split][s]['scores'][family] is not None
                            and features[split][s]['scores'][family] >= cut]
                    check_point(p, point(b, ids, accepted, cfg['quality_f1_threshold']))
                    secondary = sum(f1(b[s]['light']) >= Fraction(str(cfg['secondary_quality_f1_threshold'])) for s in accepted)
                    assert p['secondary_good_images'] == secondary; verified += 1
            assert budgets['primary'] == budgets['families'][cfg['primary_family']]
            assert budgets['selected'] == budgets['families'][budgets['selected_family']]
    assert not frozen['test_labels_used'] and frozen['test_features_present_at_freeze'] == 0
    assert (root/'frozen_train_choices.json').stat().st_mtime_ns < min(p.stat().st_mtime_ns for p in (root/'test/features').glob('*.json'))
    if plan['result_class'] != 'SMOKE_TEST':
        assert len(features['train']) == 125 and len(features['test']) == 300
        assert result['splits']['train']['light_only']['gt_words'] == 983
        assert result['splits']['test']['light_only']['gt_words'] == 2204
    audit_result = {'status': 'PASS', 'claim_label': 'approximate reproduction', 'plan_sha256': ph,
        'results_sha256': sha(root/'results.json'), 'audit_code_sha256': sha(__file__),
        'features_verified': sum(len(v) for v in features.values()), 'components_verified': components_verified,
        'routing_points_verified': verified, 'independent_component_engine': 'scipy.ndimage (runner: OpenCV)',
        'only_probability_member_read': True, 'train_only_choices_verified': True,
        'all_original_word_denominators_preserved': True, 'optimizer_steps': 0, 'new_model_inference': False}
    (root/'independent_audit.json').write_text(json.dumps(audit_result, indent=2, sort_keys=True)+'\n')
    diagnostics(root, cfg, result, features, branches)
    export(root, cfg, result); print(json.dumps(audit_result), flush=True)


def diagnostics(root, cfg, result, features, branches):
    """All-control uncertainty and explicitly posthoc explanatory ablations."""
    ids = sorted(features['test']); b = branches['test']
    rng = np.random.default_rng(cfg['seed'])
    weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=cfg['bootstrap_iterations']).astype(float)
    def distribution(accepted):
        accepted = set(accepted)
        rows = np.array([[b[s]['light' if s in accepted else 'deepsolo'][k]
                          for k in ('tp', 'fp', 'fn')] for s in ids])
        c = weights@rows
        return 2*c[:, 0]/(2*c[:, 0]+c[:, 1]+c[:, 2])
    out = {'claim_label': 'approximate reproduction', 'results_sha256': sha(root/'results.json'),
        'analysis_code_sha256': sha(__file__), 'new_model_inference': False, 'network_training': False,
        'gates_fixed': True, 'multiplicity_correction': False, 'iterations': cfg['bootstrap_iterations'],
        'posthoc_ablation_warning': 'Explanatory only; declared after main TEST results. No new deployment selection.',
        'control_comparisons': {}, 'posthoc_ablations': {}}
    for fraction in cfg['bypass_fractions']:
        key = str(fraction); data = result['splits']['test']['budgets'][key]
        out['control_comparisons'][key] = {}
        for name in ('primary', 'selected'):
            out['control_comparisons'][key][name] = {}; candidate = distribution(data[name]['accepted_ids'])
            for control in cfg['control_families']:
                reference = data['families'][control]
                out['control_comparisons'][key][name][control] = {
                    'delta_f1': data[name]['combined']['f1']-reference['combined']['f1'],
                    'ci95': np.quantile(candidate-distribution(reference['accepted_ids']), [.025, .975]).tolist(),
                    'delta_tp': data[name]['combined']['tp']-reference['combined']['tp'],
                    'delta_fp': data[name]['combined']['fp']-reference['combined']['fp']}
    for split, ids in [('train', sorted(features['train'])), ('test', sorted(features['test']))]:
        out['posthoc_ablations'][split] = {}
        for method in ('word_visual_risk_without_regions', 'fewest_words_only'):
            values = {}
            for sid in ids:
                e = features[split][sid]['word_evidence']
                values[sid] = (-(2*e['visual_error_proxy'] if method == 'word_visual_risk_without_regions'
                    else e['predicted_word_count'])) if e['predicted_word_count'] and not e['has_empty_reading'] else None
            ranked = sorted((s for s in ids if values[s] is not None), key=lambda s: (-values[s], s))
            out['posthoc_ablations'][split][method] = {str(f): point(branches[split], ids,
                ranked[:round(len(ids)*f)], cfg['quality_f1_threshold']) for f in cfg['bypass_fractions']}
    (root/'control_comparisons_and_ablations.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')


def export(root, cfg, result):
    rows = []
    for split, data in result['splits'].items():
        for fraction, budget in data['budgets'].items():
            for kind in ('families', 'scalar_thresholds'):
                for family, p in budget[kind].items():
                    rows.append({'split': split, 'budget_fraction': fraction, 'policy_kind': kind, 'family': family,
                        'train_selected': family == budget['selected_family'], 'primary': family == cfg['primary_family'],
                        'accepted': p['accepted_images'], 'good80': p['good_images'], 'good85': p['secondary_good_images'],
                        'good_fraction': p['good_fraction'], 'accepted_word_f1': p['accepted_words']['f1'],
                        'rejected_deepsolo_f1': p['deepsolo_rejected_words']['f1'],
                        **{k: p['combined'][k] for k in ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'f1', 'accuracy_tn0')}})
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fmt = lambda v: 'undefined' if v is None else f'{v*100:.3f}%'
    lines = ['# Residual-region/error-cost router results', '',
        'Approximate reproduction; cached predictions, no training or new model inference.',
        'TEST has been explored; local word-multiset F1 is not the official paper protocol.', '',
        '|Budget|Rule|TRAIN accepted|TRAIN final F1|TEST accepted|TEST good80|Accepted word F1|Rejected DeepSolo F1|TEST final F1|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for fraction in cfg['bypass_fractions']:
        key = str(fraction); test = result['splits']['test']['budgets'][key]; train = result['splits']['train']['budgets'][key]
        for family in dict.fromkeys(['base_proxy_f1', cfg['primary_family'], test['selected_family']]):
            te = test['families'][family]; tr = train['families'][family]
            label = family+(' (TRAIN-selected)' if family == test['selected_family'] else '')
            lines.append(f"|{fmt(fraction)}|{label}|{tr['accepted_images']}|{fmt(tr['combined']['f1'])}|{te['accepted_images']}|{te['good_images']}|{fmt(te['accepted_words']['f1'])}|{fmt(te['deepsolo_rejected_words']['f1'])}|{fmt(te['combined']['f1'])}|")
    lines += ['', '## All declared rules on TEST (descriptive; no TEST-selected deployment winner)', '',
        '|Rule|20% F1|30% F1|50% F1|', '|---|---:|---:|---:|']
    for family in result['splits']['test']['budgets']['0.2']['families']:
        values = [fmt(result['splits']['test']['budgets'][str(f)]['families'][family]['combined']['f1']) for f in cfg['bypass_fractions']]
        lines.append('|'+family+'|'+'|'.join(values)+'|')
    lines += ['', 'All DeepSolo TRAIN: '+fmt(result['splits']['train']['deepsolo_only']['f1'])+
        '; TEST: '+fmt(result['splits']['test']['deepsolo_only']['f1'])+'.',
        'Transferred scalar-cut coverages, all raw counts, and both TRAIN partitions are in comparison.csv/results.json.',
        'Paired intervals fix gates; no multiplicity correction. Residual region counts are uncalibrated proxies.']
    diagnostic = read(root/'control_comparisons_and_ablations.json')
    lines += ['', '## All-control paired uncertainty', '',
        '|Budget|Policy|Reference|Delta F1 (pp)|Paired 95% interval (pp)|', '|---|---|---|---:|---|']
    for budget, policies in diagnostic['control_comparisons'].items():
        for policy, comparisons in policies.items():
            for control, values in comparisons.items():
                lo, hi = values['ci95']
                lines.append(f"|{budget}|{policy}|{control}|{values['delta_f1']*100:+.3f}|[{lo*100:+.3f}, {hi*100:+.3f}]|")
    lines += ['', '## Posthoc explanatory ablations (not new deployment choices)', '',
        '|Method|20% TEST F1|30% TEST F1|50% TEST F1|', '|---|---:|---:|---:|']
    for method, budgets in diagnostic['posthoc_ablations']['test'].items():
        lines.append('|'+method+'|'+'|'.join(fmt(budgets[str(f)]['combined']['f1']) for f in cfg['bypass_fractions'])+'|')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
    for family in ['base_proxy_f1', 'all_checks_softcrop', cfg['primary_family'], 'risk_native_count', 'f1_consensus_proxy']:
        data = [result['splits']['test']['budgets'][str(f)]['families'][family] for f in cfg['bypass_fractions']]
        axes[0].plot([100*f for f in cfg['bypass_fractions']], [100*p['combined']['f1'] for p in data], 'o-', label=family)
        axes[1].plot([100*f for f in cfg['bypass_fractions']], [100*p['good_fraction'] for p in data], 'o-', label=family)
    chosen = [result['splits']['test']['budgets'][str(f)]['selected'] for f in cfg['bypass_fractions']]
    axes[0].plot([100*f for f in cfg['bypass_fractions']], [100*p['combined']['f1'] for p in chosen],
        's-.', color='tab:brown', label='Family selected on TRAIN at each budget')
    axes[1].plot([100*f for f in cfg['bypass_fractions']], [100*p['good_fraction'] for p in chosen],
        's-.', color='tab:brown')
    axes[0].axhline(100*result['splits']['test']['deepsolo_only']['f1'], color='black', linestyle=':', label='All DeepSolo')
    axes[0].axhline(79.8, color='firebrick', linestyle='--', label='Paper nominal; protocol differs')
    axes[0].set_ylabel('Final local word F1 (%)'); axes[1].set_ylabel('Accepted images with light F1 >=80% (%)')
    for ax in axes: ax.set_xlabel('Images retained without DeepSolo (%)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.suptitle('Approximate residual-cost routing: TEST300, no trained selector')
    fig.savefig(root/'comparison.png', dpi=180); fig.savefig(root/'comparison.pdf'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
