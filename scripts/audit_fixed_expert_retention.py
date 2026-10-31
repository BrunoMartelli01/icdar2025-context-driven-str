"""Independently audit fixed-gate training on previously qualified TT/IC15 data."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from recognition.fixed_expert_budget import observed_features, with_word_risk
from recognition.fixed_expert_deployment import FixedExpertPipeline
from scripts.audit_common_budget_pipeline import audit_head
from scripts.audit_fixed_expert_budget import heuristic, overlap, winner_key, global_key, norm
from scripts.audit_lightweight_word_pipeline import recount, total
from scripts.icdar_budget_data import read, sha, key
from scripts.prepare_fixed_expert_retention import observe
from scripts.run_yolo_f1_quality_study import seal


def audit(root, config_path):
    plan = read(root/'resolved_configuration.json'); c = plan['config']
    if sha(config_path) != plan['config_sha256']:
        raise ValueError('Changed explicit retention preparation configuration')
    input_config = read(config_path)
    if plan['mode'] != 'approximate' or plan['test_labels_opened'] or plan['source_kind'] != 'qualified_common_outputs':
        raise ValueError('TRAIN-only qualified common-source audit required')
    source = Path(plan['source_root']); name = plan['source_case']; original = read(source/'cases.json')[name]
    q = read(source/'deployment_qualification.json')
    if q['status'] != 'PASS' or sha(source/q['audit_path']) != q['audit_sha256']:
        raise ValueError('Original common qualification changed')
    for p, digest in c['pinned_inputs'].items():
        if sha(p) != digest:
            raise ValueError('Changed pinned retention input')
    oldobs = read(source/'data'/(name+'__observations.json'))
    oldlabels = read(source/'data'/(name+'__train_labels.json'))
    maps = read(source/'data'/(name+'__maps.json'))
    output = read(source/'models'/name/'observable_outputs.json')
    case = read(root/'data/seen/case.json'); fit = case['fit_ids']; val = case['validation_ids']; ids = fit+val
    allowed_fit = set(original['partitions']['fit'])
    allowed_val = set(original['partitions']['selection']+original['partitions']['calibration'])
    populations_valid = (set(fit) <= allowed_fit and set(val) <= allowed_val if plan['smoke'] else
                         set(fit) == allowed_fit and set(val) == allowed_val)
    if not populations_valid:
        raise ValueError('Fixed retention FIT/validation populations differ')
    if {case['keys'][s] for s in fit} & {case['keys'][s] for s in val}:
        raise ValueError('Retention FIT/validation RGB overlap')
    obs = read(root/'data/seen/observations.json'); labels = read(root/'data/seen/labels.json')
    fallback = read(root/'data/fallbacks.json')['native_benchmark']
    for file, digest in read(root/'data/seen/input_sha256.json').items():
        if sha(file) != digest:
            raise ValueError('Changed native fallback source input')
    native = None
    if original['dataset'] == 'totaltext':
        native_path = Path(input_config['tt_native_root'])/'train/branches.json'
        saved = read(source/'source_input_sha256.json')
        if str(native_path) not in saved or sha(native_path) != saved[str(native_path)]:
            raise ValueError('Changed qualified Total-Text native source')
        native = read(native_path)['records']
    xword = {}; groups = {}; target = {}; care = {}; pins = {}; counts = dict(
        source_observation_checks=0, source_teacher_checks=0, word_refits=0,
        branch_recounts=0, selected_head_refits=0, trial_replays=0, deployment_replays=0)
    for sid in ids:
        a = oldobs[sid]; expected = observe(a['rows'], output['corrected_rows'][sid],
            output['traces'][sid], a['image_wh'], maps[sid]['maps'])
        if obs[sid] != expected or labels[sid]['ground_truth'] != oldlabels[sid]['ground_truth'] or labels[sid]['raw_ignored'] != oldlabels[sid]['ignored']:
            raise ValueError('Qualified source observation or TRAIN label differs')
        rows = (native[sid]['deepsolo'] if native is not None else
            read(Path(input_config['ic15_teacher_root'])/'per_image'/(key(sid)+'.json'))['predictions'])
        expected_rows = [dict(r, raw_text='' if r.get('invalid_envelope') is not None else r['text'],
            word_source='deepsolo') for r in rows]
        if fallback[sid]['rows'] != expected_rows or fallback[sid]['counts'] != oldlabels[sid]['teacher']:
            raise ValueError('Qualified fallback source differs')
        counts['source_observation_checks'] += 1; counts['source_teacher_checks'] += 1
        xword[sid], groups[sid] = observed_features(obs[sid]['raw'], obs[sid]['rows'], obs[sid]['trace'],
            obs[sid]['image_wh'], obs[sid]['map_summary'])
        if sid in fit:
            fixed = {int(r['source_row']): r for r in obs[sid]['rows']}
            texts = {norm(g['text']) for g in labels[sid]['ground_truth'] if norm(g['text'])}
            target[sid] = np.array([int(bool(norm(fixed.get(i, r)['raw_text'])) and
                norm(fixed.get(i, r)['raw_text']) in texts) for i, r in enumerate(obs[sid]['raw'])])
            care[sid] = np.logical_not(labels[sid]['raw_ignored'])
    folder = root/'models/seen'; probability = read(folder/'word_probabilities.json')
    folds = read(folder/'word_folds.json')['folds']; covered = []
    for i, f in enumerate(folds+[dict(fit_ids=fit, heldout_ids=val)]):
        if i < len(folds):
            if set(f['fit_ids']) != set(fit)-set(f['heldout_ids']) or {case['keys'][s] for s in f['fit_ids']} & {case['keys'][s] for s in f['heldout_ids']}:
                raise ValueError('Retention word-fold contamination')
            covered.extend(f['heldout_ids'])
        x = np.concatenate([xword[s][care[s]] for s in f['fit_ids']]); y = np.concatenate([target[s][care[s]] for s in f['fit_ids']]); w = c['word_risk']
        refit = HistGradientBoostingClassifier(max_iter=w['iterations'], max_leaf_nodes=w['leaves'],
            min_samples_leaf=w['minimum_leaf'], l2_regularization=w['l2'], early_stopping=False,
            random_state=c['seed']+(i if i < len(folds) else 0)).fit(x, y)
        file = folder/(f'word_fold_{i}.joblib' if i < len(folds) else 'word_full.joblib')
        if i < len(folds) and sha(file) != f['sha256']:
            raise ValueError('Changed retention word-fold model')
        actual = joblib.load(file)
        for sid in f['heldout_ids']:
            expected = refit.predict_proba(xword[sid])[:, 1] if len(xword[sid]) else np.empty(0)
            got = actual.predict_proba(xword[sid])[:, 1] if len(xword[sid]) else np.empty(0)
            np.testing.assert_allclose(expected, got, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(expected, probability[sid], rtol=1e-12, atol=1e-12)
        counts['word_refits'] += 1
    if sorted(covered) != sorted(fit):
        raise ValueError('Incomplete retention word OOF coverage')
    for sid in ids:
        groups[sid] = with_word_risk(groups[sid], probability[sid], obs[sid]['trace'])
    with np.load(folder/'features.npz') as z:
        if z['ids'].tolist() != ids:
            raise ValueError('Retention feature order changed')
        for name_group in groups[ids[0]]:
            np.testing.assert_allclose(z[name_group], np.stack([groups[s][name_group] for s in ids]), rtol=1e-12, atol=1e-12)
    def matrix(samples, names):
        return np.stack([np.concatenate([groups[s][g] for g in names]) for s in samples])
    scale = {s: len(obs[s]['raw'])+1 for s in ids}; choices = []
    for detail in read(folder/'completed.json')['results']:
        target_dir = folder/detail['target_name']; branches = read(target_dir/'train_branches.json')
        for sid in ids:
            fused = list(fallback[sid]['rows']); policy = detail['complement']
            if policy['confidence'] is not None:
                for row in obs[sid]['rows']:
                    if probability[sid][int(row['source_row'])] >= policy['confidence'] and not any(
                            other['bbox'] is not None and overlap(row['bbox'], other['bbox']) >= policy['maximum_overlap'] for other in fused):
                        fused.append(dict(row, fixed_light_complement=True))
            for branch, rows in [('light', obs[sid]['rows']), ('deepsolo', fused)]:
                ignored = [labels[sid]['raw_ignored'][int(r['source_row'])]
                    if branch == 'light' or r.get('fixed_light_complement') else r['ignored'] for r in rows]
                if recount(rows, labels[sid]['ground_truth'], ignored) != branches[sid][branch]:
                    raise ValueError('Independent retention branch recount differs')
                counts['branch_recounts'] += 1
        chosen = read(target_dir/'winner.json')
        y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
            branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fit])/np.array([scale[s] for s in fit])[:, None]
        for model in chosen['models']:
            bundle = joblib.load(target_dir/model)
            audit_head(bundle, matrix(fit, bundle['groups']), y, np.array([scale[s] for s in fit]), c['seed'])
            counts['selected_head_refits'] += 1
        trials = read(target_dir/'trials.json'); predicted = {}
        for model in {m for t in trials for m in t['models']}:
            bundle = joblib.load(target_dir/model); x = bundle['scaler'].transform(matrix(val, bundle['groups'])); ms = bundle['models']
            predicted[model] = ms[0].predict(x) if len(ms) == 1 else np.stack([m.predict(x) for m in ms], axis=1)
        quota = round(len(val)*case['fallback_images']/case['total_images'])
        for t in trials:
            if t['models']:
                pairs = np.mean([predicted[m] for m in t['models']], axis=0)
                scores = {s: float(2*v[0]-t['reference']*v[1])*scale[s] for s, v in zip(val, pairs)}
            else:
                scores = {s: heuristic(obs[s], probability[s], t['heuristic']) for s in val}
            eligible = [s for s in val if obs[s]['rows']] if t['guard'] else val
            keep = sorted(eligible, key=lambda s: (-scores[s], case['keys'][s], s))[:len(val)-quota]
            aggregate = total(branches[s]['light' if s in keep else 'deepsolo'] for s in val)
            if sorted(keep) != t['result']['accepted_ids'] or aggregate != t['result']['combined']:
                raise ValueError('Retention trial ranking/counts differ')
            counts['trial_replays'] += 1
        best = max(t['result']['combined']['f1'] for t in trials)
        expected = min((t for t in trials if t['result']['combined']['f1'] >= best-c['selection_tolerance']), key=winner_key)
        if chosen != expected or chosen != detail['winner']:
            raise ValueError('Retention target selection differs')
        if detail['primary_eligible']:
            choices.append(detail)
    frozen = read(root/'selection_frozen.json'); best = max(r['winner']['result']['combined']['f1'] for r in choices)
    expected = min((r for r in choices if r['winner']['result']['combined']['f1'] >= best-c['selection_tolerance']), key=global_key)
    if frozen['winner'] != expected or frozen['test_used_for_selection']:
        raise ValueError('Retention global selection differs')
    pipeline = FixedExpertPipeline(root, verify_qualification=False); deployed = {}
    for sid in val:
        o = obs[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
        counts['deployment_replays'] += 1
    decision = pipeline.route(deployed, {s: case['keys'][s] for s in val}, fallback_images=quota)
    if decision['accepted_ids'] != expected['winner']['result']['accepted_ids']:
        raise ValueError('Retention annotation-free deployment differs')
    for file in [root/'selection_frozen.json', root/'resolved_configuration.json', source/'deployment_qualification.json',
                 folder/'word_full.joblib', folder/expected['target_name']/'winner.json'] + [folder/expected['target_name']/m for m in expected['winner']['models']]:
        pins[str(file)] = sha(file)
    for name_module in ('fixed_expert_deployment.py', 'fixed_expert_budget.py', 'common_budget_pipeline.py'):
        file = Path(__file__).resolve().parents[1]/'recognition'/name_module; pins[str(file)] = sha(file)
    report_path = root/'independent_audit_v1.json'
    seal(report_path, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        case=plan['source_case'], counters=counts, validation=expected['winner']['result'],
        source_qualification_sha256=sha(source/'deployment_qualification.json'),
        test_used_for_selection=False, auditor_sha256=sha(__file__)))
    seal(root/'deployment_qualification.json', dict(status='PASS', audit_path=str(report_path),
        audit_sha256=sha(report_path), artifact_sha256=pins))
    print(__import__('json').dumps(dict(status='PASS', case=plan['source_case'], counters=counts)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root, a.config)
