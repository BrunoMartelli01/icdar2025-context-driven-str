"""Audit supervised fit scopes/portable predictions and report oracle separation."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

from scripts.audit_yolo_f1_quality_study import aggregate, check_point, f1, point
from recognition.trainable_image_router import feature_dict


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def optimum(branches, ids, eligible, n):
    """Independent fractional replay, initialized at all-DeepSolo F1."""
    ratio = f1(aggregate(branches[s]['deepsolo'] for s in ids))
    for _ in range(100):
        advantages = {s: 2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])-
            ratio*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']) for s in eligible}
        chosen = sorted(eligible, key=lambda s: (-advantages[s], s))[:n]
        counts = aggregate(branches[s]['light' if s in chosen else 'deepsolo'] for s in ids)
        updated = f1(counts)
        if updated == ratio:
            base = aggregate(branches[s]['deepsolo'] for s in ids)
            maximum_residual = 2*base['tp']-ratio*(base['detected_words']+base['gt_words'])+sum(sorted(advantages.values(), reverse=True)[:n])
            assert maximum_residual == 0
            return {'accepted_ids': chosen, 'f1': float(ratio), 'f1_fraction': str(ratio),
                'certificate_max_residual': '0', 'uses_ground_truth': True}
        ratio = updated
    raise ValueError('Independent oracle did not converge')


def independent_fit(x, y, recipe, cfg):
    settings = cfg['model_settings']; mean = np.mean(x, axis=0); scale = np.std(x, axis=0)
    scale[scale == 0] = 1.; kind = recipe['kind']
    if kind == 'ridge': model = Ridge(alpha=recipe['alpha'], solver='svd', fit_intercept=True)
    elif kind == 'oracle_logistic':
        model = LogisticRegression(C=recipe['C'], solver='lbfgs', class_weight='balanced',
            max_iter=settings['logistic_max_iter'], tol=settings['logistic_tolerance'], random_state=settings['seed'])
    else:
        cls = ExtraTreesClassifier if kind == 'oracle_trees' else ExtraTreesRegressor
        model = cls(n_estimators=settings['trees'], max_depth=recipe['max_depth'], min_samples_leaf=recipe['min_samples_leaf'],
            max_features=1., bootstrap=False, random_state=settings['seed'], n_jobs=1)
    model.fit((x-mean)/scale, y)
    return model, mean, scale


def audit(root):
    warnings.simplefilter('error', ConvergenceWarning)
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']; result = read(root/'results.json')
    frozen = read(root/'frozen_choices.json'); cv = read(root/'cv_results.json'); ph = sha(root/'resolved_configuration.json')
    assert result['status'] == 'SUCCESS' and result['plan_sha256'] == ph
    assert result['router_training'] and result['base_models_frozen'] and not result['new_visual_model_inference']
    assert result['frozen_choices_sha256'] == sha(root/'frozen_choices.json')
    assert frozen['cv_results_sha256'] == sha(root/'cv_results.json')
    for file, digest in plan['code_sha256'].items(): assert sha(root/'code_snapshot'/file) == digest
    for path, digest in plan['input_sha256'].items(): assert sha(path) == digest
    for name in ('residual', 'quality'):
        source = Path(plan['paths'][name+'_root']); pins = cfg['sources'][name]
        for filename, pin in [('resolved_configuration.json', 'plan'), ('results.json', 'results'), ('independent_audit.json', 'audit')]:
            assert sha(source/filename) == pins[pin]
    branches = read(Path(plan['source_coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']
    features = {}
    for split, samples in plan['samples'].items():
        features[split] = {}
        for sample in samples:
            sid = sample['sample_id']; filename = sid.replace(':', '__')+'.json'; value = read(root/split/'features'/filename)
            r = read(Path(plan['paths']['residual_root'])/split/'features'/filename)
            q = read(Path(plan['paths']['quality_root'])/split/'features'/filename)
            vector, schema, eligible = feature_dict(r, q, sample['image_wh'], cfg['feature_settings'])
            assert value['values'] == vector and value['eligible'] == eligible and schema == plan['feature_schemas']
            assert value['controls'] == {f: r['scores'][f] for f in cfg['controls']}
            assert value['plan_sha256'] == ph and value['image_sha256'] == sample['image_sha256']
            assert not value['per_image_ground_truth_used'] and not value['actual_deepsolo_outputs_used']
            assert value['frozen_choices_sha256'] == (sha(root/'frozen_choices.json') if split == 'test' else None)
            features[split][sid] = value
    fit_ids, val_ids = plan['fit_ids'], plan['validation_ids']
    assert not set(fit_ids)&set(val_ids) and set(fit_ids+val_ids) == set(features['train'])
    assert not set(features['train'])&set(features['test'])
    def matrix(split, ids, names):
        return np.array([[features[split][s]['values'][name] for name in names] for s in ids])
    fit_cache = {}; refits = 0; fold_predictions_verified = 0
    def replay(ids, recipe, names, fraction, teacher):
        nonlocal refits
        assert set(ids) <= set(fit_ids)
        b = branches['train']; kind = recipe['kind']
        if kind.startswith('oracle'):
            expected = optimum(b, ids, [s for s in ids if features['train'][s]['eligible']], round(len(ids)*fraction))
            assert teacher == expected
            keep = set(expected['accepted_ids']); y = np.array([int(s in keep) for s in ids])
        else:
            assert teacher is None
            y = np.array([[b[s]['light']['tp']-b[s]['deepsolo']['tp'],
                b[s]['light']['detected_words']-b[s]['deepsolo']['detected_words']] for s in ids], float)
        key = (tuple(ids), tuple(names), json.dumps(recipe, sort_keys=True), y.tobytes())
        if key not in fit_cache:
            fit_cache[key] = independent_fit(matrix('train', ids, names), y, recipe, cfg); refits += 1
        return fit_cache[key]
    def values(model, split, ids, names, kind, reference):
        estimator, mean, scale = model; z = (matrix(split, ids, names)-mean)/scale
        pred = estimator.predict_proba(z)[:, 1] if kind.startswith('oracle') else estimator.predict(z)
        scores = pred if kind.startswith('oracle') else 2*pred[:, 0]-reference*pred[:, 1]
        return {s: float(v) if features[split][s]['eligible'] else None for s, v in zip(ids, scores)}
    def compare_scores(actual, expected):
        assert set(actual) == set(expected)
        for s, value in expected.items():
            assert actual[s] is None if value is None else math.isclose(actual[s], value, rel_tol=1e-10, abs_tol=1e-10), s
    def ranked(scores, ids, fraction):
        return sorted((s for s in ids if scores[s] is not None), key=lambda s: (-scores[s], s))[:round(len(ids)*fraction)]
    for budget, families in cv.items():
        fraction = float(budget)
        for family, data in families.items():
            kind, feature_set = family.rsplit('_', 1); names = plan['feature_schemas'][feature_set]
            for trial in data['trials']:
                visited = []
                for fold in trial['fold_trace']:
                    training, held = fold['training_ids'], fold['held_out_ids']
                    assert not set(training)&set(held) and set(training+held) == set(fit_ids)
                    reference = float(f1(aggregate(branches['train'][s]['deepsolo'] for s in training)))
                    assert fold['reference_f1'] == reference
                    model = replay(training, trial['recipe'], names, fraction, fold['teacher_oracle'])
                    expected = values(model, 'train', held, names, kind, reference)
                    compare_scores({s: trial['oof_scores'][s] for s in held}, expected)
                    visited.extend(held); fold_predictions_verified += len(held)
                assert sorted(visited) == sorted(fit_ids)
                accepted = ranked(trial['oof_scores'], fit_ids, fraction)
                check_point(trial['oof_point'], point(branches['train'], fit_ids, accepted, cfg['quality_f1_threshold']))
            def objective(i):
                c = data['trials'][i]['oof_point']['combined']; return f1(c), -i
            assert data['selected_trial'] == max(range(len(data['trials'])), key=objective)
    predictions = {split: read(root/'predictions'/(split+'.json')) for split in result['splits']}
    populations = {'train': ('train', sorted(features['train'])), 'fit': ('train', fit_ids),
        'validation': ('train', val_ids), 'test': ('test', sorted(features['test']))}
    for budget, choices in frozen['budgets'].items():
        val_trials = []; fraction = float(budget)
        for index, family in enumerate(plan['families']):
            path = root/'models'/(budget+'__'+family+'.json')
            assert sha(path) == frozen['models_sha256'][str(path.relative_to(root))]
            saved = read(path); assert saved['training_ids'] == fit_ids
            kind, feature_set = family.rsplit('_', 1); names = plan['feature_schemas'][feature_set]
            assert saved['feature_names'] == names
            trial = cv[budget][family]['trials'][cv[budget][family]['selected_trial']]
            assert saved['recipe'] == trial['recipe']
            model = replay(fit_ids, saved['recipe'], names, fraction, saved['teacher_oracle'])
            _, mean, scale = model
            np.testing.assert_allclose(saved['mean'], mean, rtol=0, atol=1e-12)
            np.testing.assert_allclose(saved['scale'], scale, rtol=0, atol=1e-12)
            reference = float(f1(aggregate(branches['train'][s]['deepsolo'] for s in fit_ids)))
            assert saved['reference_f1'] == reference
            for split, (origin, ids) in populations.items():
                compare_scores(predictions[split][budget][family], values(model, origin, ids, names, kind, reference))
            val_scores = predictions['validation'][budget][family]; accepted = ranked(val_scores, val_ids, fraction)
            cut = min((val_scores[s] for s in accepted), default=None)
            assert choices['scalar_cuts'][family] == cut
            p = point(branches['train'], val_ids, accepted, cfg['quality_f1_threshold'])
            val_trials.append((f1(p['combined']), -index, family))
        assert choices['selected_family'] == max(val_trials)[2]
    points_verified = 0; unconstrained = {}
    for split, (origin, ids) in populations.items():
        data = result['splits'][split]; b = branches[origin]; eligible = [s for s in ids if features[origin][s]['eligible']]
        assert data['light_only'] == aggregate(b[s]['light'] for s in ids)
        assert data['deepsolo_only'] == aggregate(b[s]['deepsolo'] for s in ids)
        for budget, row in data['budgets'].items():
            fraction = float(budget); k = round(len(ids)*fraction)
            ref = optimum(b, ids, eligible, k)
            assert {name: row['oracle'][name] for name in ref} == ref
            if split == 'test': unconstrained[budget] = optimum(b, ids, ids, k)
            references = {k: ref}
            def check(actual, accepted):
                nonlocal points_verified
                check_point(actual, point(b, ids, accepted, cfg['quality_f1_threshold']))
                n = len(accepted)
                if n not in references: references[n] = optimum(b, ids, eligible, n)
                optimum_set = set(references[n]['accepted_ids']); a = set(accepted)
                assert actual['oracle_reference_accepted_images'] == n
                assert actual['oracle_reference_f1'] == references[n]['f1']
                assert abs(actual['oracle_f1_gap']-(references[n]['f1']-actual['combined']['f1'])) < 1e-12
                assert actual['canonical_oracle_overlap'] == len(a&optimum_set)
                assert actual['canonical_symmetric_difference'] == len(a^optimum_set)
                assert abs(actual['canonical_decision_agreement']-(1-len(a^optimum_set)/len(ids))) < 1e-12
                points_verified += 1
            for family in plan['families']:
                scores = predictions[split][budget][family]; accepted = ranked(scores, ids, fraction)
                check(row['families'][family], accepted)
                cut = frozen['budgets'][budget]['scalar_cuts'][family]
                accepted = [s for s in ids if cut is not None and scores[s] is not None and scores[s] >= cut]
                check(row['scalar_thresholds'][family], accepted)
            for family in cfg['controls']:
                scores = {s: features[origin][s]['controls'][family] for s in ids}
                check(row['controls'][family], ranked(scores, ids, fraction))
            check(row['oracle']['point'], ref['accepted_ids'])
            assert row['selected'] == row['families'][frozen['budgets'][budget]['selected_family']]
            assert row['primary'] == row['families'][cfg['primary_family']]
    assert not frozen['test_labels_used'] and frozen['test_predictions_present_at_freeze'] == frozen['test_features_present_at_freeze'] == 0
    # Equal timestamps are valid on a coarsely clocked filesystem. The hash
    # binding above proves that each TEST-vector writer had frozen choices.
    assert (root/'frozen_choices.json').stat().st_mtime_ns <= (root/'predictions/test.json').stat().st_mtime_ns
    assert (root/'frozen_choices.json').stat().st_mtime_ns <= min(p.stat().st_mtime_ns for p in (root/'test/features').glob('*.json'))
    if plan['result_class'] != 'SMOKE_TEST':
        assert len(fit_ids) == 80 and len(val_ids) == 45 and len(features['test']) == 300
        assert result['splits']['train']['light_only']['gt_words'] == 983 and result['splits']['test']['light_only']['gt_words'] == 2204
    audit_result = {'status': 'PASS', 'claim_label': 'approximate reproduction', 'plan_sha256': ph,
        'results_sha256': sha(root/'results.json'), 'audit_code_sha256': sha(__file__),
        'features_verified': sum(len(v) for v in features.values()), 'routing_points_verified': points_verified,
        'held_out_predictions_replayed': fold_predictions_verified, 'independent_fit_calls': refits,
        'fit_and_validation_scopes_verified': True, 'frozen_before_test': True, 'base_models_frozen': True,
        'test_labels_used_for_training_or_selection': False, 'new_visual_model_inference': False}
    (root/'independent_audit.json').write_text(json.dumps(audit_result, indent=2, sort_keys=True)+'\n')
    (root/'unconstrained_oracle_reference.json').write_text(json.dumps(unconstrained, indent=2, sort_keys=True)+'\n')
    export(root, plan, result, unconstrained)
    print(json.dumps(audit_result), flush=True)


def export(root, plan, result, unconstrained):
    cfg = plan['config']; rows = []
    for split, data in result['splits'].items():
        for budget, value in data['budgets'].items():
            for kind in ('families', 'controls', 'scalar_thresholds'):
                for family, p in value[kind].items():
                    rows.append({'split': split, 'budget': budget, 'policy_kind': kind, 'family': family,
                        'validation_selected': family == value['selected_family'], 'accepted': p['accepted_images'],
                        'good80': p['good_images'], 'oracle_gap': p['oracle_f1_gap'], 'oracle_overlap': p['canonical_oracle_overlap'],
                        'decision_agreement': p['canonical_decision_agreement'],
                        **{k: p['combined'][k] for k in ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'f1', 'accuracy_tn0')}})
    with (root/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fmt = lambda v: f'{100*v:.3f}%'
    lines = ['# Supervised oracle-separation router: approximate reproduction', '',
        'Only the tabular router is trained. Visual models/outputs stay frozen. Fit80, VAL45, TEST300 are separate.',
        'TEST has already been explored; metric equivalence with paper recognition accuracy remains unresolved.', '',
        '|Bypass|VAL-selected family|Fit80 F1 (in-sample)|VAL45 F1|TRAIN125 F1|TEST F1|Oracle F1 (same eligibility)|Oracle gap (pp)|Canonical oracle overlap|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for budget, data in result['splits']['test']['budgets'].items():
        p = data['selected']; family = data['selected_family']
        vals = [fmt(result['splits'][s]['budgets'][budget]['selected']['combined']['f1']) for s in ('fit', 'validation', 'train')]
        lines.append('|'+budget+'|'+family+'|'+'|'.join(vals)+f"|{fmt(p['combined']['f1'])}|{fmt(data['oracle']['f1'])}|{p['oracle_f1_gap']*100:.3f}|{p['canonical_oracle_overlap']}/{p['accepted_images']}|")
    lines += ['', '## All models and frozen controls on TEST (descriptive; no TEST selection)', '',
        '|Family|20% F1|30% F1|50% F1|', '|---|---:|---:|---:|']
    for family in plan['families']+cfg['controls']:
        values = []
        for fraction in cfg['bypass_fractions']:
            data = result['splits']['test']['budgets'][str(fraction)]; kind = 'families' if family in plan['families'] else 'controls'
            values.append(fmt(data[kind][family]['combined']['f1']))
        lines.append('|'+family+'|'+'|'.join(values)+'|')
    lines += ['', '## Original unconstrained oracle reference', '',
        'The earlier oracle allowed empty lightweight readings to bypass. The inference-compatible oracle above excludes them.',
        'This small eligibility difference changes the numerical ceiling; neither oracle is a deployable model.', '',
        '|Budget|Original unconstrained oracle|Current eligible oracle|', '|---|---:|---:|']
    for key, ref in unconstrained.items():
        lines.append(f"|{key}|{fmt(ref['f1'])}|{fmt(result['splits']['test']['budgets'][key]['oracle']['f1'])}|")
    lines += ['', 'Canonical membership can be nonunique. Final F1 gap is the primary measure of approaching the optimum.',
        'Fixed rank budgets and scalar-cut coverage are distinct. CSV/JSON retain all counts and accepted IDs.',
        'Paired bootstrap intervals fix gates, use 10,000 image samples, and have no multiplicity correction.',
        'Fit scores are in-sample. OOF hyperparameter traces are in cv_results.json; models are portable JSON.',
        'Perfect separation was not presumed. No TEST winner replaces a VAL-selected policy.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout='constrained'); budgets = cfg['bypass_fractions']
    def series(name, group):
        return [result['splits']['test']['budgets'][str(f)][group][name]['combined']['f1']*100 for f in budgets]
    x = [100*f for f in budgets]
    for family in ('base_proxy_f1', 'coverage_baseline', 'risk_visual_mass'):
        axes[0].plot(x, series(family, 'controls'), 'o-', label=family)
    selected = [result['splits']['test']['budgets'][str(f)]['selected'] for f in budgets]
    opt = [result['splits']['test']['budgets'][str(f)]['oracle']['f1'] for f in budgets]
    axes[0].plot(x, [p['combined']['f1']*100 for p in selected], 's-', label='Supervised, selected on VAL45')
    axes[0].plot(x, [v*100 for v in opt], 'k--', label='GT oracle (not deployable)')
    axes[0].axhline(100*result['splits']['test']['deepsolo_only']['f1'], color='gray', ls=':', label='All DeepSolo')
    axes[1].plot(x, [p['oracle_f1_gap']*100 for p in selected], 's-', label='Supervised oracle gap')
    axes[0].set_ylabel('Final local word F1 (%)'); axes[1].set_ylabel('Distance from eligible oracle (F1 pp)')
    for ax in axes: ax.set_xlabel('Images retained without DeepSolo (%)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.suptitle('Supervised routing: frozen visual branches, fit80 / VAL45 / TEST300')
    fig.savefig(root/'comparison.png', dpi=180); fig.savefig(root/'comparison.pdf'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
