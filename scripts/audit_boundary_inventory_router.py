"""Independent finite-policy search and word-count replay for bounded routing."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_yolo_f1_quality_study import read, sha, seal
from scripts.audit_yolo_f1_quality_study import check_point, point, f1


def audit(root):
    p = read(root/'resolved_configuration.json'); cfg = p['config']; r = read(root/'results.json'); f = read(root/'frozen_choices.json')
    assert r['status'] == 'SUCCESS' and r['plan_sha256'] == f['plan_sha256'] == sha(root/'resolved_configuration.json')
    assert r['frozen_choices_sha256'] == sha(root/'frozen_choices.json') and not f['test_labels_used']
    for path, digest in p['input_sha256'].items():
        assert sha(path) == digest
    for file, digest in p['code_sha256'].items():
        assert sha(root/'code_snapshot'/file) == digest
    source = Path(p['paths']['source_root']); features = {}
    for file, key in [('resolved_configuration.json', 'plan'), ('results.json', 'results'), ('independent_audit.json', 'audit')]:
        assert sha(source/file) == cfg['source_pins'][key]
    sp = read(source/'resolved_configuration.json')
    branches = read(Path(sp['source_coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']
    for split, ss in p['samples'].items():
        features[split] = {s['sample_id']: read(source/split/'features'/(s['sample_id'].replace(':', '__')+'.json')) for s in ss}
    fit, val = p['fit_ids'], p['validation_ids']
    def choose(split, ids, budget, rule, policy):
        v = features[split]; base = cfg['references'][budget]
        order = sorted((s for s in ids if v[s]['eligible']), key=lambda s: (-v[s]['controls'][base], s))
        k = round(len(ids)*float(budget)); w = round(len(ids)*policy['half_width_fraction'])
        a, b = max(0, k-w), min(len(order), k+w)
        if w == 0 or policy['audit_weight'] == 0:
            return order[:k]
        middle = order[a:b]; ranked = sorted(middle, key=lambda s: (-v[s]['rules'][rule], s))
        positions = {s: j for j, s in enumerate(ranked)}; weight = policy['audit_weight']
        tuples = [((1-weight)*j+weight*positions[s], j, s) for j, s in enumerate(middle)]
        keep = order[:a]+[s for _, _, s in sorted(tuples)[:k-a]]
        assert len(set(keep)) == k and set(order[:a]) <= set(keep) and not set(keep)&set(order[b:])
        assert len(set(keep)-set(order[:k])) <= w
        return keep
    def exact(split, ids, accepted):
        return f1(point(branches[split], ids, accepted, cfg['quality_f1_threshold'])['combined'])
    total = 0
    base_policy = {'half_width_fraction': 0., 'audit_weight': 0.}
    for budget, choices in f['budgets'].items():
        val_trials = []
        for i, rule in enumerate(cfg['rules']):
            trials = choices['fit_trials'][rule]['trials']; expected = []
            grid = [(w, weight) for w in cfg['half_width_fractions'] for weight in cfg['audit_weights']]
            assert len(trials) == len(grid)
            for trial, (width, weight) in zip(trials, grid):
                assert trial['policy'] == {'half_width_fraction': width, 'audit_weight': weight}
                keep = choose('train', fit, budget, rule, trial['policy']); score = exact('train', fit, keep)
                assert keep == trial['accepted_ids'] and float(score) == trial['f1']; expected.append(score); total += 1
            best = max(range(len(trials)), key=lambda j: (expected[j], -j))
            assert best == choices['fit_trials'][rule]['selected_trial']
            policy = trials[best]['policy']; assert policy == choices['policies'][rule]
            keep = choose('train', val, budget, rule, policy); score = exact('train', val, keep)
            assert float(score) == choices['validation_f1'][rule]; val_trials.append((score, -i, rule))
        keep = choose('train', val, budget, cfg['rules'][0], base_policy); score = exact('train', val, keep)
        val_trials.append((score, 1, 'baseline')); assert choices['selected_family'] == max(val_trials)[2]
        for pop, split, ids in [('fit', 'train', fit), ('validation', 'train', val), ('train', 'train', sorted(features['train'])),
                                ('test', 'test', sorted(features['test']))]:
            out = r['splits'][pop][budget]
            for rule, saved in out['families'].items():
                policy = base_policy if rule == 'baseline' else choices['policies'][rule]
                keep = choose(split, ids, budget, cfg['rules'][0] if rule == 'baseline' else rule, policy)
                check_point(saved, point(branches[split], ids, keep, cfg['quality_f1_threshold'])); total += 1
            assert out['selected'] == out['families'][choices['selected_family']]
    result = {'status': 'PASS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'results_sha256': sha(root/'results.json'), 'plan_sha256': sha(root/'resolved_configuration.json'),
        'audit_code_sha256': sha(Path(__file__)), 'independent_policy_or_point_checks': total,
        'quota_and_protected_decisions_verified': True, 'test_used_for_policy_selection': False}
    seal(root/'independent_audit.json', result); print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
