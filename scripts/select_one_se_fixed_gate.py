"""Cost-aware one-standard-error selection over independently audited TRAIN trials."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def eligible(trial, floor):
    # Both accepted heuristics consume a fitted word-correctness model. The
    # sequence-only control is excluded from this explicitly trained-router study.
    return (trial['result']['combined']['f1'] >= floor and
            (bool(trial['models']) or trial['heuristic'] in ('expected_errors', 'weakest_word')))


def complexity_key(choice):
    t = choice['winner']
    return (choice['additional_map_models'], choice['extra_reader_crops'], t['scalars'],
            -t['selection_score'], len(t['models']), t['models'], t['reference'] or 0,
            t['heuristic'] or '', t['guard'], choice['target_name'])


def choose(parent, cv, config):
    plan = read(parent/'resolved_configuration.json'); view = plan['crossfit_config']['source_view']
    case = read(parent/'data'/view/'case.json'); source = Path(plan['source_root']); ids = case['fit_ids']+case['validation_ids']
    choices = []
    for detail in read(parent/'targets.json'):
        target = parent/'models'/view/detail['target_name']
        for t in read(target/'trials.json'):
            if eligible(t, cv['minimum_validation_f1']):
                extra = int(case['map_available'] and any('maps' in g for g in t['groups']))*case['map_extra_models']
                choices.append(dict(detail, winner=t, additional_map_models=extra))
    if not choices:
        raise ValueError('No explicitly trained candidate meets the validation floor')
    best = max(choices, key=lambda x: x['winner']['selection_score'])
    branches = read(source/'models'/view/best['target_name']/'train_branches.json')
    kept = set(best['winner']['crossfit']['accepted_ids']+best['winner']['result']['accepted_ids'])
    values = np.asarray([[branches[s]['light' if s in kept else 'deepsolo'][k]
        for k in ('tp', 'fp', 'fn')] for s in ids], dtype=np.int64)
    random = np.random.default_rng(config['seed']); indices = random.integers(len(ids), size=(config['bootstrap_draws'], len(ids)))
    sample = values[indices].sum(axis=1); denominator = 2*sample[:, 0]+sample[:, 1]+sample[:, 2]
    if np.any(denominator == 0):
        raise ValueError('Empty lexical denominator in an image bootstrap sample')
    estimates = 2*sample[:, 0]/denominator
    se = float(np.std(estimates, ddof=1)); floor = best['winner']['selection_score']-se
    candidates = [v for v in choices if v['winner']['selection_score'] >= floor]
    return min(candidates, key=complexity_key), dict(best_selection_score=best['winner']['selection_score'],
        bootstrap_standard_error=se, one_se_floor=floor, candidate_count=len(choices),
        eligible_within_one_se=len(candidates), bootstrap_draws=config['bootstrap_draws'], seed=config['seed'],
        scope='Image bootstrap conditional on fixed crossfit predictions and prior candidate search; not nested generalization uncertainty')


def run(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(config_path)
    if c['mode'] != 'approximate' or c['bootstrap_draws'] < 2:
        raise ValueError('Explicit APPROXIMATE bootstrap configuration required')
    for p, digest in c['pinned_inputs'].items():
        if sha(p) != digest:
            raise ValueError('Changed one-SE selection input')
    parent = Path(c['source_root']); qualification = parent/c['qualification_name']; q = read(qualification)
    if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
        raise ValueError('Independently audited crossfit source required')
    for p, digest in q['artifact_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Changed crossfit qualification artifact')
    parent_plan = read(parent/'resolved_configuration.json'); view = parent_plan['crossfit_config']['source_view']
    winner, uncertainty = choose(parent, parent_plan['crossfit_config'], c)
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    settings = dict(parent_plan['config']); settings['output_root'] = str(root); settings['scientific_choices'] = c['scientific_choices']
    code = {}
    for package in settings['snapshot_packages']:
        for p in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/p.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
            code[str(p.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    plan = dict(parent_plan, config=settings, source_root=str(parent), source_sha256=code,
        one_se_config=c, one_se_config_sha256=sha(config_path),
        parent_configuration_sha256=sha(parent/'resolved_configuration.json'),
        parent_qualification_sha256=sha(qualification),
        test_labels_opened=False, test_used_for_selection=False)
    seal(root/'resolved_configuration.json', plan)
    destination = root/'data'/view; destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent/'data'/view/'case.json', destination/'case.json')
    folder = root/'models'/view; folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent/'models'/view/'word_full.joblib', folder/'word_full.joblib')
    target = folder/winner['target_name']; target.mkdir(parents=True, exist_ok=False)
    source = Path(parent_plan['source_root'])/'models'/view/winner['target_name']
    for name in winner['winner']['models']:
        shutil.copyfile(source/name, target/name)
    seal(target/'winner.json', winner['winner'])
    paths = [root/'resolved_configuration.json', destination/'case.json', folder/'word_full.joblib', target/'winner.json']+[target/name for name in winner['winner']['models']]
    seal(root/'selection_frozen.json', dict(status='FROZEN', winner=winner, uncertainty=uncertainty,
        artifact_sha256={str(p.relative_to(root)): sha(p) for p in paths},
        test_used_for_selection=False, validation_target_met=True))
    print(__import__('json').dumps(dict(status='FROZEN', target=winner['target_name'],
        heuristic=winner['winner']['heuristic'], models=winner['winner']['models'],
        pooled_f1=winner['winner']['selection_score'], validation_f1=winner['winner']['result']['combined']['f1'],
        deepsolo_images=winner['winner']['deepsolo_selection_images'],
        total_images=winner['winner']['total_selection_images'],
        deepsolo_percentage=winner['winner']['deepsolo_selection_percentage'],
        standard_error=uncertainty['bootstrap_standard_error'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.config)
