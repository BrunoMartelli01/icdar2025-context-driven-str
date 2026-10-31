"""Refit word models and independently replay risk admission, quota and selection."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from recognition.fixed_expert_budget import observed_features
from recognition.risk_admission_deployment import RiskAdmissionPipeline
from scripts.audit_fixed_expert_budget import norm, overlap
from scripts.audit_lightweight_word_pipeline import recount, total
from scripts.icdar_budget_data import read, sha, key
from scripts.prepare_fixed_expert_retention import observe
from scripts.run_yolo_f1_quality_study import seal


def audit(root):
    plan = read(root/'resolved_configuration.json'); cfg = plan['risk_selection_config']; c = plan['config']
    if plan['mode'] != 'approximate' or not plan['risk_admission'] or plan['test_labels_opened'] or plan['test_used_for_selection']:
        raise ValueError('TRAIN-only approximate risk-admission audit required')
    for path, digest in read(root/'source_input_sha256.json').items():
        if sha(path) != digest:
            raise ValueError('Changed risk-admission training input')
    source = Path(plan['source_root']); original = read(source/'resolved_configuration.json')
    q = read(source/'deployment_qualification.json')
    if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
        raise ValueError('Changed source qualification')
    view = cfg['source_view']; case = read(root/'data'/view/'case.json'); fit = case['fit_ids']; val = case['validation_ids']; ids = fit+val
    obs = read(source/'data'/view/'observations.json'); labels = read(source/'data'/view/'labels.json')
    probabilities = read(source/'models'/view/'word_probabilities.json'); fallbacks = read(source/'data/fallbacks.json')
    if original.get('source_kind') == 'qualified_common_outputs':
        prep = read(cfg['retention_config']); common = Path(original['source_root']); name = original['source_case']
        oldobs = read(common/'data'/(name+'__observations.json')); oldlabels = read(common/'data'/(name+'__train_labels.json'))
        maps = read(common/'data'/(name+'__maps.json')); emitted = read(common/'models'/name/'observable_outputs.json')
        if name.startswith('totaltext__'):
            native_path = Path(prep['tt_native_root'])/'train/branches.json'
            if sha(native_path) != read(common/'source_input_sha256.json')[str(native_path)]:
                raise ValueError('Changed qualified Total-Text fallback source')
            native = read(native_path)['records']
        else:
            native = None
        for sid in ids:
            s = oldobs[sid]
            if obs[sid] != observe(s['rows'], emitted['corrected_rows'][sid], emitted['traces'][sid], s['image_wh'], maps[sid]['maps']):
                raise ValueError('Changed qualified source observations')
            if labels[sid]['ground_truth'] != oldlabels[sid]['ground_truth'] or labels[sid]['raw_ignored'] != oldlabels[sid]['ignored']:
                raise ValueError('Changed source TRAIN annotations')
            base = (native[sid]['deepsolo'] if native is not None else
                read(Path(prep['ic15_teacher_root'])/'per_image'/(key(sid)+'.json'))['predictions'])
            base = [dict(row, raw_text='' if row.get('invalid_envelope') is not None else row['text'], word_source='deepsolo') for row in base]
            if fallbacks['native_benchmark'][sid]['rows'] != base:
                raise ValueError('Changed qualified native fallback rows')
    elif str(source/'data/fallbacks.json') not in q['artifact_sha256']:
        raise ValueError('Source qualification must bind its training fallback rows')
    features = {}; target = {}; care = {}
    for sid in ids:
        o = obs[sid]; features[sid], _ = observed_features(o['raw'], o['rows'], o['trace'], o['image_wh'], None)
        if sid in fit:
            selected = {int(r['source_row']): r for r in o['rows']}
            words = {norm(g['text']) for g in labels[sid]['ground_truth'] if norm(g['text'])}
            target[sid] = np.array([int(bool(norm(selected.get(i, r)['raw_text'])) and
                norm(selected.get(i, r)['raw_text']) in words) for i, r in enumerate(o['raw'])])
            care[sid] = np.logical_not(labels[sid]['raw_ignored'])
    folds = read(source/'models'/view/'word_folds.json')['folds']; covered = []; settings = c['word_risk']
    for i, fold in enumerate(folds+[dict(fit_ids=fit, heldout_ids=val)]):
        if i < len(folds):
            if set(fold['fit_ids']) != set(fit)-set(fold['heldout_ids']) or {case['keys'][s] for s in fold['fit_ids']} & {case['keys'][s] for s in fold['heldout_ids']}:
                raise ValueError('Risk model fold or RGB isolation differs')
            covered.extend(fold['heldout_ids'])
        x = np.concatenate([features[s][care[s]] for s in fold['fit_ids']]); y = np.concatenate([target[s][care[s]] for s in fold['fit_ids']])
        model = HistGradientBoostingClassifier(max_iter=settings['iterations'], max_leaf_nodes=settings['leaves'],
            min_samples_leaf=settings['minimum_leaf'], l2_regularization=settings['l2'], early_stopping=False,
            random_state=c['seed']+(i if i < len(folds) else 0)).fit(x, y)
        for sid in fold['heldout_ids']:
            expected = model.predict_proba(features[sid])[:, 1] if len(features[sid]) else np.empty(0)
            np.testing.assert_allclose(expected, probabilities[sid], rtol=1e-12, atol=1e-12)
    if sorted(covered) != sorted(fit):
        raise ValueError('Incomplete risk-model OOF coverage')
    profiles = {v['target_name']: v for v in read(source/'models'/view/'completed.json')['results'] if v['primary_eligible']}
    quotas = {'fit': round(len(fit)*case['fallback_images']/case['total_images']),
              'validation': round(len(val)*case['fallback_images']/case['total_images'])}
    trials = read(root/'trials.json'); branch_cache = {}; seen = set(); replay_count = 0
    for detail in trials:
        t = detail['winner']; name = detail['target_name']; profile = profiles[name]; threshold = t['admission_threshold']; keyname = t['training_branch_key']
        if keyname not in branch_cache:
            branches = read(root/'training_branches'/(keyname+'.json')); light = {}
            for sid in ids:
                p = probabilities[sid]; words = [dict(r, word_probability=float(p[int(r['source_row'])])) for r in obs[sid]['rows']
                    if p[int(r['source_row'])] >= threshold]; light[sid] = words
                fused = list(fallbacks[profile['fallback_profile']][sid]['rows']); rule = profile['complement']
                if rule['confidence'] is not None:
                    for word in words:
                        if word['word_probability'] >= rule['confidence'] and not any(
                                r['bbox'] is not None and overlap(word['bbox'], r['bbox']) >= rule['maximum_overlap'] for r in fused):
                            fused.append(dict(word, fixed_light_complement=True))
                for branch, rows in [('light', words), ('deepsolo', fused)]:
                    flags = [labels[sid]['raw_ignored'][int(r['source_row'])]
                        if branch == 'light' or r.get('fixed_light_complement') else r['ignored'] for r in rows]
                    if recount(rows, labels[sid]['ground_truth'], flags) != branches[sid][branch]:
                        raise ValueError('Independent risk-admission branch recount differs')
                    replay_count += 1
            branch_cache[keyname] = (branches, light)
        branches, light = branch_cache[keyname]
        def score(sid):
            p = probabilities[sid]; keep = {int(r['source_row']) for r in light[sid]}
            if t['heuristic'] == 'expected_errors':
                return -sum((1-v) if i in keep else v for i, v in enumerate(p))
            if t['heuristic'] == 'weakest_word':
                return min((p[i] for i in keep), default=0.)
            raise ValueError('Unexpected risk heuristic')
        chosen = {}
        for part, population, saved in [('fit', fit, t['crossfit']), ('validation', val, t['result'])]:
            eligible = [s for s in population if light[s]] if t['guard'] else population
            keep = sorted(eligible, key=lambda s: (-score(s), case['keys'][s], s))[:len(population)-quotas[part]]
            count = total(branches[s]['light' if s in keep else 'deepsolo'] for s in population)
            if sorted(keep) != saved['accepted_ids'] or count != saved['combined']:
                raise ValueError('Independent risk-admission route differs')
            chosen[part] = keep
        accepted = set(chosen['fit']+chosen['validation'])
        pooled = total(branches[s]['light' if s in accepted else 'deepsolo'] for s in ids)
        if pooled != t['pooled_counts'] or pooled['f1'] != t['selection_score']:
            raise ValueError('Independent risk-admission pooled count differs')
        seen.add((name, threshold, t['heuristic'], t['guard']))
    if len(seen) != len(trials):
        raise ValueError('Repeated risk-admission trial')
    eligible = [v for v in trials if v['winner']['result']['combined']['f1'] >= cfg['minimum_validation_f1']]
    best = max(eligible, key=lambda v: v['winner']['selection_score']); t = best['winner']
    keep = set(t['crossfit']['accepted_ids']+t['result']['accepted_ids']); branches, _ = branch_cache[t['training_branch_key']]
    rows = np.array([[branches[s]['light' if s in keep else 'deepsolo'][k] for k in ('tp', 'fp', 'fn')] for s in ids], np.int64)
    index = np.random.default_rng(cfg['seed']).integers(len(ids), size=(cfg['bootstrap_draws'], len(ids)))
    draw = rows[index].sum(axis=1); se = float(np.std(2*draw[:, 0]/(2*draw[:, 0]+draw[:, 1]+draw[:, 2]), ddof=1))
    lower = t['selection_score']-se; near = [v for v in eligible if v['winner']['selection_score'] >= lower]
    def independent_key(v):
        t = v['winner']
        quality = (t['removed_words'], -t['selection_score']) if cfg.get('prefer_fewer_filtered_words', False) else (
            -t['selection_score'], t['removed_words'])
        return (v['additional_map_models'], v['extra_reader_crops'], *quality,
                t['admission_threshold'], t['heuristic'], t['guard'], v['target_name'])
    winner = min(near, key=independent_key)
    frozen = read(root/'selection_frozen.json')
    if frozen['winner'] != winner or frozen['test_used_for_selection']:
        raise ValueError('Independent risk-admission selection differs')
    np.testing.assert_allclose([frozen['uncertainty']['bootstrap_standard_error'], frozen['uncertainty']['one_se_floor']],
                               [se, lower], rtol=1e-12, atol=1e-12)
    pipeline = RiskAdmissionPipeline(root, verify_qualification=False); deployed = {}
    for sid in val:
        o = obs[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
        np.testing.assert_allclose(deployed[sid]['probabilities'], probabilities[sid], atol=1e-12, rtol=1e-12)
    decision = pipeline.route(deployed, {s: case['keys'][s] for s in val}, fallback_images=quotas['validation'])
    if decision['accepted_ids'] != winner['winner']['result']['accepted_ids']:
        raise ValueError('Risk-admission annotation-free deployment route differs')
    pins = {str(source/'deployment_qualification.json'): sha(source/'deployment_qualification.json'),
            str(root/'selection_frozen.json'): sha(root/'selection_frozen.json'),
            str(root/'source_input_sha256.json'): sha(root/'source_input_sha256.json')}
    for rel in frozen['artifact_sha256']:
        p = root/rel; pins[str(p)] = sha(p)
    if c.get('emission_override') is not None:
        p = Path(c['emission_override']['policy']); pins[str(p)] = sha(p)
    for name in ('risk_admission_deployment.py', 'budget_deployment_factory.py', 'fixed_expert_budget.py', 'common_budget_pipeline.py'):
        p = Path(__file__).resolve().parents[1]/'recognition'/name; pins[str(p)] = sha(p)
    report = root/'independent_audit_v1.json'
    seal(report, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        word_model_refits=len(folds)+1, branch_recounts=replay_count, trial_replays=len(trials),
        validation_deployment_replays=len(val), validation=winner['winner']['result'],
        crossfit=winner['winner']['crossfit'], pooled=winner['winner']['pooled_counts'],
        selected_threshold=winner['winner']['admission_threshold'], selected_heuristic=winner['winner']['heuristic'],
        test_used_for_selection=False, auditor_sha256=sha(__file__),
        scope='Conditional stacked expert/word-risk folds and bootstrap after candidate search remain limitations.'))
    seal(root/'deployment_qualification.json', dict(status='PASS', audit_path=str(report),
        audit_sha256=sha(report), artifact_sha256=pins))
    print(__import__('json').dumps(dict(status='PASS', word_model_refits=len(folds)+1,
        branch_recounts=replay_count, trial_replays=len(trials), validation_deployment_replays=len(val))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True); a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root)
