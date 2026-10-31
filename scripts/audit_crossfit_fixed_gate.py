"""Independently refit every candidate fold and replay crossfit gate selection."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import grouped_folds, fit_count_head, predict_counts
from recognition.fixed_expert_deployment import FixedExpertPipeline
from scripts.audit_common_budget_pipeline import audit_head
from scripts.audit_fixed_expert_budget import heuristic
from scripts.audit_lightweight_word_pipeline import total
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def audit(root, audit_name='independent_audit_v1.json', qualification_name='deployment_qualification.json'):
    plan = read(root/'resolved_configuration.json'); c = plan['config']; cv = plan['crossfit_config']; view = cv['source_view']
    if plan['mode'] != 'approximate' or plan['test_labels_opened'] or plan['test_used_for_selection']:
        raise ValueError('TRAIN/validation-only approximate crossfit required')
    for path, digest in read(root/'source_input_sha256.json').items():
        if sha(path) != digest:
            raise ValueError('Changed qualified crossfit input')
    source = Path(plan['source_root']); case = read(root/'data'/view/'case.json'); fit = case['fit_ids']; val = case['validation_ids']; ids = fit+val
    obs = read(source/'data'/view/'observations.json'); probs = read(source/'models'/view/'word_probabilities.json')
    with np.load(source/'models'/view/'features.npz') as z:
        if z['ids'].tolist() != ids:
            raise ValueError('Crossfit feature order differs')
        groups = {g: z[g].copy() for g in z.files if g != 'ids'}
    pos = {s: i for i, s in enumerate(ids)}; scale = {s: len(obs[s]['raw'])+1 for s in ids}
    def matrix(samples, names):
        return np.concatenate([groups[g][[pos[s] for s in samples]] for g in names], axis=1)
    folds = grouped_folds(fit, case['keys'], c['folds'], c['seed']); weights = np.array([scale[s] for s in fit])
    quota_fit = round(len(fit)*case['fallback_images']/case['total_images']); quota_val = round(len(val)*case['fallback_images']/case['total_images'])
    counts = dict(full_head_refits=0, crossfit_head_refits=0, trial_replays=0, deployment_replays=0)
    choices = []; target_pins = {}
    for detail in read(root/'targets.json'):
        target = root/'models'/view/detail['target_name']; source_target = source/'models'/view/detail['target_name']
        status = read(target/'completed.json'); branches = read(source_target/'train_branches.json')
        for name in ('completed.json', 'winner.json', 'trials.json', 'head_predictions.npz'):
            target_pins[str(target/name)] = sha(target/name)
        if status['folds'] != folds or status['test_used_for_selection']:
            raise ValueError('Crossfit folds or TEST boundary differ')
        y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
            branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fit])/weights[:, None]
        with np.load(target/'head_predictions.npz') as z:
            if z['fit_ids'].tolist() != fit or z['validation_ids'].tolist() != val:
                raise ValueError('Crossfit saved prediction population differs')
            fp = {}; vp = {}
            for gi, names in enumerate(c['feature_groups']):
                x = matrix(fit, names); xv = matrix(val, names)
                for ri, recipe in enumerate(c['head_recipes']):
                    model = f'head_{gi}_{ri}.joblib'; bundle = joblib.load(source_target/model)
                    if bundle['recipe'] != recipe or bundle['groups'] != names:
                        raise ValueError('Changed full-FIT model recipe')
                    audit_head(bundle, x, y, weights, c['seed']); counts['full_head_refits'] += 1
                    expected_val = predict_counts(bundle, xv)
                    np.testing.assert_allclose(expected_val, z['val__'+model], rtol=1e-10, atol=1e-10)
                    predicted = np.zeros((len(fit), 2))
                    for held in folds:
                        test_idx = [fit.index(s) for s in held]; train_idx = [i for i, s in enumerate(fit) if s not in set(held)]
                        local = fit_count_head(x[train_idx], y[train_idx], weights[train_idx], recipe, c['seed'])
                        predicted[test_idx] = predict_counts(local, x[test_idx]); counts['crossfit_head_refits'] += 1
                    np.testing.assert_allclose(predicted, z['fit__'+model], rtol=1e-10, atol=1e-10)
                    fp[model] = predicted; vp[model] = expected_val
        trials = read(target/'trials.json')
        for t in trials:
            if t['models']:
                f = np.mean([fp[m] for m in t['models']], axis=0)
                v = np.mean([vp[m] for m in t['models']], axis=0)
                fs = {s: float(2*q[0]-t['reference']*q[1])*scale[s] for s, q in zip(fit, f)}
                vs = {s: float(2*q[0]-t['reference']*q[1])*scale[s] for s, q in zip(val, v)}
            else:
                fs = {s: heuristic(obs[s], probs[s], t['heuristic']) for s in fit}
                vs = {s: heuristic(obs[s], probs[s], t['heuristic']) for s in val}
            fi = [s for s in fit if obs[s]['rows']] if t['guard'] else fit
            va = [s for s in val if obs[s]['rows']] if t['guard'] else val
            fk = sorted(fi, key=lambda s: (-fs[s], case['keys'][s], s))[:len(fit)-quota_fit]
            vk = sorted(va, key=lambda s: (-vs[s], case['keys'][s], s))[:len(val)-quota_val]
            for subset, keep, result in [(fit, fk, t['crossfit']), (val, vk, t['result'])]:
                got = total(branches[s]['light' if s in keep else 'deepsolo'] for s in subset)
                if sorted(keep) != result['accepted_ids'] or got != result['combined']:
                    raise ValueError('Crossfit trial ranking or branch total differs')
            pooled = total(branches[s]['light' if s in set(fk+vk) else 'deepsolo'] for s in ids)
            if pooled != t['pooled_counts'] or pooled['f1'] != t['selection_score']:
                raise ValueError('Crossfit pooled trial total differs')
            counts['trial_replays'] += 1
        eligible = [t for t in trials if t['result']['combined']['f1'] >= cv['minimum_validation_f1']]
        best = max(t['selection_score'] for t in eligible)
        expected = min((t for t in eligible if t['selection_score'] >= best-cv['selection_tolerance']),
            key=lambda t: (t['scalars'], -t['selection_score'], len(t['models']), t['models'],
                           t['reference'] or 0, t['heuristic'] or '', t['guard']))
        if expected != status['winner'] or expected != read(target/'winner.json'):
            raise ValueError('Crossfit target winner criterion differs')
        extra_maps = int(case['map_available'] and any('maps' in g for g in expected['groups']))*case['map_extra_models']
        choices.append(dict(detail, winner=expected, additional_map_models=extra_maps))
    frozen = read(root/'selection_frozen.json'); best = max(r['winner']['selection_score'] for r in choices)
    expected = min((r for r in choices if r['winner']['selection_score'] >= best-cv['selection_tolerance']),
        key=lambda r: (r['additional_map_models'], r['extra_reader_crops'], r['winner']['scalars'],
                       -r['winner']['selection_score'], r['target_name']))
    if frozen['winner'] != expected or frozen['choices'] != choices or frozen['test_used_for_selection']:
        raise ValueError('Crossfit global winner criterion differs')
    pipeline = FixedExpertPipeline(root, verify_qualification=False); deployed = {}
    for sid in val:
        o = obs[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
        counts['deployment_replays'] += 1
    decision = pipeline.route(deployed, {s: case['keys'][s] for s in val}, fallback_images=quota_val)
    if decision['accepted_ids'] != expected['winner']['result']['accepted_ids']:
        raise ValueError('Crossfit deployment validation route differs')
    pins = dict(target_pins)
    for p in [root/'selection_frozen.json', root/'resolved_configuration.json', root/'source_input_sha256.json',
              root/'targets.json', root/'data'/view/'case.json', root/'models'/view/'word_full.joblib',
              root/'models'/view/expected['target_name']/'winner.json'] + [root/'models'/view/expected['target_name']/m for m in expected['winner']['models']]:
        pins[str(p)] = sha(p)
    if c.get('emission_override') is not None:
        p = Path(c['emission_override']['policy']); pins[str(p)] = sha(p)
    for name in ('fixed_expert_deployment.py', 'fixed_expert_budget.py', 'common_budget_pipeline.py'):
        p = Path(__file__).resolve().parents[1]/'recognition'/name; pins[str(p)] = sha(p)
    report_path = root/audit_name
    seal(report_path, dict(status='PASS', mode='approximate', claim_label='approximate reproduction', counters=counts,
        validation=expected['winner']['result'], crossfit=expected['winner']['crossfit'],
        pooled=expected['winner']['pooled_counts'], selected=expected['target_name'],
        test_used_for_selection=False, auditor_sha256=sha(__file__),
        scope='All image-head folds refitted and rankings replayed; source expert/word-feature stacking is conditional, not fully nested.'))
    seal(root/qualification_name, dict(status='PASS', audit_path=str(report_path),
        audit_sha256=sha(report_path), artifact_sha256=pins))
    print(__import__('json').dumps(dict(status='PASS', counters=counts, target=expected['target_name'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--audit-name', default='independent_audit_v1.json')
    p.add_argument('--qualification-name', default='deployment_qualification.json'); a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root, a.audit_name, a.qualification_name)
