"""TRAIN/VAL-only policy selection for a bounded CRAFT correction of prior routing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from recognition.boundary_inventory_router import select
from recognition.trainable_image_router import branch_f1
from recognition.f1_quality_study import evaluate
from scripts.run_yolo_f1_quality_study import checked, read, seal, sha

REPO = Path(__file__).resolve().parents[1]
CODE = ('scripts/run_boundary_inventory_router.py', 'recognition/boundary_inventory_router.py',
        'recognition/trainable_image_router.py', 'recognition/f1_quality_study.py', 'scripts/run_yolo_f1_quality_study.py')


def run(args):
    if any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)) or args.mode != 'approximate':
        raise ValueError('Explicit isolated approximate configuration required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 experiment required')
    source = Path(paths['source_root']); p = checked(source/'resolved_configuration.json', cfg['source_pins']['plan'])
    result = checked(source/'results.json', cfg['source_pins']['results'])
    audit = checked(source/'independent_audit.json', cfg['source_pins']['audit'])
    if audit['status'] != 'PASS' or result['status'] != 'SUCCESS' or audit['results_sha256'] != cfg['source_pins']['results']:
        raise ValueError('Audited complete source required')
    inputs = {}; features = {}
    for split, ss in p['samples'].items():
        features[split] = {}
        for sample in ss:
            path = source/split/'features'/(sample['sample_id'].replace(':', '__')+'.json'); v = read(path)
            if v['plan_sha256'] != cfg['source_pins']['plan'] or v['per_image_ground_truth_used'] or v['actual_deepsolo_outputs_used']:
                raise ValueError('GT-blind source vector required')
            features[split][sample['sample_id']] = v; inputs[str(path)] = sha(path)
    branch_path = Path(p['source_coverage_root'])/'results.json'
    if sha(branch_path) != p['input_sha256'][str(branch_path)]:
        raise ValueError('Branch counts changed')
    inputs[str(branch_path)] = sha(branch_path)
    plan = {'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction', 'table_5_claim': False,
        'result_class': 'APPROXIMATE_EXPLORATORY_BOUNDED_ROUTER', 'config': cfg, 'paths': paths,
        'fit_ids': p['fit_ids'], 'validation_ids': p['validation_ids'], 'samples': p['samples'],
        'input_sha256': inputs, 'code_sha256': {f: sha(REPO/f) for f in CODE},
        'base_word_outputs_changed': False, 'new_neural_inference': False, 'historically_explored_test': True,
        'test_signals_seen_before_this_proposal': True, 'policy_selection_uses_only_train_and_validation': True,
        'deployment': 'batch-level fixed quota only; no online scalar-threshold equivalence claimed'}
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(paths['output_root'])/tag
    if root.resolve().is_relative_to(REPO) or root.resolve().is_relative_to(source.resolve()):
        raise ValueError('Separate external outputs required')
    seal(root/'resolved_configuration.json', plan)
    for file in CODE:
        target = root/'code_snapshot'/file; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != plan['code_sha256'][file]:
            raise ValueError('Code snapshot changed')
        if not target.exists():
            target.write_bytes((REPO/file).read_bytes())
    branches = read(branch_path)['resolutions']['640']['branch_metrics']; fit, val = p['fit_ids'], p['validation_ids']
    frozen = {'plan_sha256': sha(root/'resolved_configuration.json'), 'budgets': {}, 'test_labels_used': False}
    def accepted(split, ids, budget, rule, policy):
        values = features[split]
        base = {s: values[s]['controls'][cfg['references'][budget]] if values[s]['eligible'] else None for s in ids}
        other = {s: values[s]['rules'][rule] for s in ids}
        return select(base, other, ids, float(budget), policy)
    for budget in cfg['references']:
        policies = {}; candidates = []; fit_trials = {}
        for index, rule in enumerate(cfg['rules']):
            trials = []
            for width in cfg['half_width_fractions']:
                for weight in cfg['audit_weights']:
                    policy = {'half_width_fraction': width, 'audit_weight': weight}
                    keep = accepted('train', fit, budget, rule, policy)
                    trials.append({'policy': policy, 'accepted_ids': keep, 'f1': float(branch_f1(branches['train'], fit, keep))})
            winner = max(range(len(trials)), key=lambda i: (branch_f1(branches['train'], fit, trials[i]['accepted_ids']), -i))
            policy = trials[winner]['policy']; policies[rule] = policy; fit_trials[rule] = {'trials': trials, 'selected_trial': winner}
            keep = accepted('train', val, budget, rule, policy)
            candidates.append((branch_f1(branches['train'], val, keep), -index, rule))
        # Retaining the old baseline is an explicit family, including ties.
        base_policy = {'half_width_fraction': 0., 'audit_weight': 0.}
        keep = accepted('train', val, budget, cfg['rules'][0], base_policy)
        candidates.append((branch_f1(branches['train'], val, keep), 1, 'baseline'))
        frozen['budgets'][budget] = {'policies': policies, 'fit_trials': fit_trials, 'selected_family': max(candidates)[2],
            'validation_f1': {rule: float(score) for score, _, rule in candidates}}
    seal(root/'frozen_choices.json', frozen)
    output = {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction', 'table_5_claim': False,
        'plan_sha256': sha(root/'resolved_configuration.json'), 'frozen_choices_sha256': sha(root/'frozen_choices.json'), 'splits': {}}
    for pop, split, ids in [('fit', 'train', fit), ('validation', 'train', val), ('train', 'train', sorted(features['train'])),
                            ('test', 'test', sorted(features['test']))]:
        output['splits'][pop] = {}
        for budget, choices in frozen['budgets'].items():
            families = {}
            for rule, policy in choices['policies'].items():
                keep = accepted(split, ids, budget, rule, policy)
                families[rule] = evaluate(branches[split], ids, keep, cfg['quality_f1_threshold'])
            keep = accepted(split, ids, budget, cfg['rules'][0], base_policy)
            families['baseline'] = evaluate(branches[split], ids, keep, cfg['quality_f1_threshold'])
            selected = choices['selected_family']
            output['splits'][pop][budget] = {'families': families, 'selected_family': selected, 'selected': families[selected]}
    seal(root/'results.json', output)
    print(json.dumps({'status': 'SUCCESS', 'root': str(root), 'results_sha256': sha(root/'results.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--paths', type=Path, required=True)
    run(parser.parse_args())
