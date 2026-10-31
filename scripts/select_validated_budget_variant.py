"""Select an approximate budget pipeline using only frozen validation counts."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from recognition.common_budget_pipeline import micro
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def qualification(root):
    path = root / 'deployment_qualification.json'
    value = read(path)
    if value['status'] != 'PASS' or sha(value['audit_path']) != value['audit_sha256']:
        raise ValueError('A valid independently audited candidate is required')
    for file, digest in value['artifact_sha256'].items():
        if sha(file) != digest:
            raise ValueError('Qualified candidate changed: ' + file)
    return value


def candidate(spec):
    root = Path(spec['root'])
    qualification(root)
    frozen = read(root / 'selection_frozen.json')
    if frozen['status'] != 'FROZEN' or frozen['test_used_for_selection']:
        raise ValueError('Candidate must be frozen without TEST inputs')
    for file, digest in frozen['artifact_sha256'].items():
        if sha(root / file) != digest:
            raise ValueError('Candidate frozen artifact changed')
    winner = frozen['winner']
    case_path = root / 'data' / winner['view'] / 'case.json'
    case = read(case_path)
    ids = sorted(case['validation_ids'])
    if spec['kind'] == 'fusion':
        selection = frozen['fusion_selection']
        result = selection['validation']
        branch_path = root / 'training_branches' / (str(selection['index']) + '.json')
    elif spec['kind'] == 'fixed':
        result = winner['winner']['result']
        branch_path = root / 'models' / winner['view'] / winner['target_name'] / 'train_branches.json'
    else:
        raise ValueError('Unknown explicit candidate kind')
    branches = read(branch_path)
    accepted = set(result['accepted_ids'])
    if (len(ids) != result['total_images'] or not accepted <= set(ids) or
            len(ids) - len(accepted) != result['deepsolo_images']):
        raise ValueError('Candidate validation image budget differs')
    values = [branches[s]['light' if s in accepted else 'deepsolo'] for s in ids]
    if micro(values) != result['combined']:
        raise ValueError('Candidate validation count replay differs')
    return dict(label=spec['label'], root=str(root), kind=spec['kind'], ids=ids,
        ground_truth_words=[v['gt_words'] for v in values], values=values,
        f1=result['combined']['f1'], deepsolo_images=result['deepsolo_images'],
        total_images=len(ids), deepsolo_percentage=100 * result['deepsolo_images'] / len(ids),
        additional_map_models=winner['additional_map_models'],
        extra_reader_crops=winner['extra_reader_crops'], case_path=str(case_path),
        branch_path=str(branch_path), qualification_sha256=sha(root / 'deployment_qualification.json'),
        frozen_sha256=sha(root / 'selection_frozen.json'))


def check_shared(records):
    if not records:
        raise ValueError('At least one candidate is required')
    first = records[0]
    if any((r['ids'] != first['ids'] or r['ground_truth_words'] != first['ground_truth_words'] or
            r['deepsolo_images'] != first['deepsolo_images']) for r in records):
        raise ValueError('Candidates must share validation populations and budgets')


def choose(records, draws, seed):
    check_shared(records)
    if draws < 2:
        raise ValueError('At least two bootstrap draws are required')
    best = max(records, key=lambda r: r['f1'])
    values = np.asarray([[v[k] for k in ('tp', 'fp', 'fn')] for v in best['values']], dtype=np.int64)
    indices = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    samples = values[indices].sum(axis=1)
    den = 2 * samples[:, 0] + samples[:, 1] + samples[:, 2]
    if np.any(den == 0):
        raise ValueError('Empty bootstrap lexical denominator')
    se = float(np.std(2 * samples[:, 0] / den, ddof=1))
    floor = best['f1'] - se
    eligible = [(i, r) for i, r in enumerate(records) if r['f1'] >= floor]
    _, selected = min(eligible, key=lambda x: (x[1]['additional_map_models'],
        x[1]['extra_reader_crops'], -x[1]['f1'], x[0]))
    return selected, dict(best_f1=best['f1'], bootstrap_standard_error=se,
        one_se_floor=floor, candidate_count=len(records), eligible_count=len(eligible),
        bootstrap_draws=draws, seed=seed,
        scope='Conditional validation image bootstrap after candidate selection; not a nested confidence interval')


def choose_no_extra_neural(records):
    check_shared(records)
    eligible = [(i, r) for i, r in enumerate(records)
                if r['additional_map_models'] == 0 and r['extra_reader_crops'] == 0]
    if not eligible:
        raise ValueError('No qualified candidate satisfies the no-extra-neural constraint')
    _, selected = min(eligible, key=lambda x: (-x[1]['f1'], x[0]))
    return selected, dict(rule='no_extra_neural', candidate_count=len(records),
        eligible_count=len(eligible), additional_map_models=0, extra_reader_crops=0,
        scope='Highest validation F1 within the declared auxiliary-neural-cost constraint')


def run(config_path):
    if config_path.resolve().parent != REPO / 'configs/approximate':
        raise ValueError('Explicit approximate selector configuration required')
    config = read(config_path)
    if config['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed validation-only selector input: ' + path)
    root = Path(config['output_root'])
    if root.exists():
        raise ValueError('Selector output already exists')
    selections = {}
    for name, specs in config['cases'].items():
        records = [candidate(spec) for spec in specs]
        for r in records:
            for path in (r['case_path'], r['branch_path'], str(Path(r['root']) / 'selection_frozen.json'),
                         str(Path(r['root']) / 'deployment_qualification.json')):
                if config['pinned_inputs'].get(path) != sha(path):
                    raise ValueError('Every selection input must be explicitly pinned')
        rule = config.get('selection_rule', 'one_se')
        if rule == 'one_se':
            picked, uncertainty = choose(records, config['bootstrap_draws'], config['seed'])
        elif rule == 'no_extra_neural':
            picked, uncertainty = choose_no_extra_neural(records)
        else:
            raise ValueError('Unknown explicit variant selection rule')
        compact = [{k: v for k, v in r.items() if k not in ('ids', 'ground_truth_words', 'values')} for r in records]
        selections[name] = dict(selected_label=picked['label'], selected_root=picked['root'],
            validation=dict(f1=picked['f1'], deepsolo_images=picked['deepsolo_images'],
                total_images=picked['total_images'], deepsolo_percentage=picked['deepsolo_percentage']),
            uncertainty=uncertainty, candidates=compact)
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / 'code_snapshot'; (snapshot / 'scripts').mkdir(parents=True)
    shutil.copyfile(__file__, snapshot / 'scripts' / Path(__file__).name)
    seal(root / 'resolved_configuration.json', dict(mode='approximate', claim_label='approximate reproduction',
        config=config, config_sha256=sha(config_path), test_used_for_selection=False,
        historical_test_exposure=True, selector_sha256=sha(__file__)))
    seal(root / 'selection_frozen.json', dict(status='FROZEN', mode='approximate',
        claim_label='approximate reproduction', selections=selections,
        test_used_for_selection=False, historical_test_exposure=True,
        config_sha256=sha(root / 'resolved_configuration.json')))
    for name, value in selections.items():
        print(name, value['selected_label'], value['validation'], value['uncertainty'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
