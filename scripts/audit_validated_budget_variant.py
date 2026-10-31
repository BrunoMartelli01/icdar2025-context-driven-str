"""Independently recount and qualify a validation-only variant selection."""

import argparse
from pathlib import Path

import numpy as np

from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def audit(root):
    plan = read(root / 'resolved_configuration.json')
    frozen = read(root / 'selection_frozen.json')
    config = plan['config']
    if (plan['mode'] != 'approximate' or plan['test_used_for_selection'] or
            frozen['test_used_for_selection'] or frozen['status'] != 'FROZEN' or
            frozen['config_sha256'] != sha(root / 'resolved_configuration.json')):
        raise ValueError('Changed validation-only frozen selector')
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed selector input')
    replayed = 0
    for name, specs in config['cases'].items():
        records = []; common_ids = common_truth = quota = None
        for spec in specs:
            folder = Path(spec['root']); q = read(folder / 'deployment_qualification.json')
            if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
                raise ValueError('Candidate qualification changed')
            for path, digest in q['artifact_sha256'].items():
                if sha(path) != digest:
                    raise ValueError('Candidate qualification artifact changed')
            state = read(folder / 'selection_frozen.json'); winner = state['winner']
            if state['status'] != 'FROZEN' or state['test_used_for_selection']:
                raise ValueError('Candidate is not TRAIN/validation frozen')
            ids = sorted(read(folder / 'data' / winner['view'] / 'case.json')['validation_ids'])
            if spec['kind'] == 'fusion':
                fusion = state['fusion_selection']; result = fusion['validation']
                branch_file = folder / 'training_branches' / (str(fusion['index']) + '.json')
            elif spec['kind'] == 'fixed':
                result = winner['winner']['result']
                branch_file = folder / 'models' / winner['view'] / winner['target_name'] / 'train_branches.json'
            else:
                raise ValueError('Invalid candidate kind')
            branches = read(branch_file); accepted = set(result['accepted_ids'])
            if len(ids) - len(accepted) != result['deepsolo_images'] or not accepted <= set(ids):
                raise ValueError('Changed validation quota')
            values = [branches[s]['light' if s in accepted else 'deepsolo'] for s in ids]
            truth = [v['gt_words'] for v in values]
            if common_ids is None:
                common_ids, common_truth, quota = ids, truth, result['deepsolo_images']
            elif ids != common_ids or truth != common_truth or quota != result['deepsolo_images']:
                raise ValueError('Validation population or quota differs')
            sums = {k: sum(v[k] for v in values) for k in ('tp', 'fp', 'fn', 'detected_words',
                'gt_words', 'all_outputs', 'ignored_outputs')}
            if sums['tp'] + sums['fn'] != sums['gt_words'] or sums['tp'] + sums['fp'] != sums['detected_words']:
                raise ValueError('Invalid lexical count accounting')
            denominator = 2 * sums['tp'] + sums['fp'] + sums['fn']
            f1 = 2 * sums['tp'] / denominator if denominator else 0.
            expected = dict(sums, images=len(ids), f1=f1)
            if expected != result['combined']:
                raise ValueError('Independent validation counts differ')
            records.append(dict(label=spec['label'], root=str(folder), f1=f1,
                maps=winner['additional_map_models'], rereads=winner['extra_reader_crops'], values=values))
            replayed += len(ids)
        saved = frozen['selections'][name]; u = saved['uncertainty']
        rule = config.get('selection_rule', 'one_se')
        if rule == 'one_se':
            best = max(records, key=lambda x: x['f1'])
            matrix = np.array([[x['tp'], x['fp'], x['fn']] for x in best['values']], dtype=np.int64)
            draws = config['bootstrap_draws']; seed = config['seed']
            indices = np.random.default_rng(seed).integers(len(matrix), size=(draws, len(matrix)))
            sums = matrix[indices].sum(axis=1); den = 2 * sums[:, 0] + sums[:, 1] + sums[:, 2]
            if np.any(den == 0):
                raise ValueError('Empty bootstrap lexical denominator')
            se = float(np.std(2 * sums[:, 0] / den, ddof=1)); floor = best['f1'] - se
            eligible = [(i, x) for i, x in enumerate(records) if x['f1'] >= floor]
            selected = min(eligible, key=lambda x: (x[1]['maps'], x[1]['rereads'], -x[1]['f1'], x[0]))[1]
            if se != u['bootstrap_standard_error'] or floor != u['one_se_floor']:
                raise ValueError('Independent conditional bootstrap differs')
        elif rule == 'no_extra_neural':
            eligible = [(i, x) for i, x in enumerate(records) if x['maps'] == 0 and x['rereads'] == 0]
            if not eligible:
                raise ValueError('No candidate satisfies the neural-cost constraint')
            selected = min(eligible, key=lambda x: (-x[1]['f1'], x[0]))[1]
            if u['rule'] != 'no_extra_neural':
                raise ValueError('Changed neural-cost constraint')
        else:
            raise ValueError('Unknown selection rule')
        if (selected['label'] != saved['selected_label'] or selected['root'] != saved['selected_root'] or
                selected['f1'] != saved['validation']['f1'] or len(eligible) != u['eligible_count']):
            raise ValueError('Independent cost-aware selection differs')
        for row, saved_row in zip(records, saved['candidates']):
            if (row['label'] != saved_row['label'] or row['f1'] != saved_row['f1'] or
                    row['maps'] != saved_row['additional_map_models'] or
                    row['rereads'] != saved_row['extra_reader_crops']):
                raise ValueError('Candidate comparison differs')
    report = root / 'independent_audit_v1.json'
    seal(report, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        validation_candidate_images_recounted=replayed, selected_cases=len(config['cases']),
        historical_test_exposure=True, test_used_for_selection=False,
        auditor_sha256=sha(__file__), scope='Independent validation count and frozen selection-rule replay'))
    artifacts = [root / 'resolved_configuration.json', root / 'selection_frozen.json',
        root / 'code_snapshot/scripts/select_validated_budget_variant.py',
        Path(__file__).resolve().parents[1] / 'recognition/validated_budget_variant.py']
    pins = {str(p): sha(p) for p in artifacts}
    pins.update(config['pinned_inputs'])
    seal(root / 'deployment_qualification.json', dict(status='PASS', audit_path=str(report),
        audit_sha256=sha(report), artifact_sha256=pins))
    print('PASS', 'cases', len(config['cases']), 'validation_candidate_images', replayed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    audit(parser.parse_args().run_root)
