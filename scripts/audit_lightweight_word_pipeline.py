"""Recount admission/route trials and refit selected lightweight CPU models."""

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lightweight_word_router import image_groups
from recognition.lightweight_word_deployment import LightweightWordPipeline
from localize.independent_word_slots import overlap_iou
from scripts.icdar_budget_data import read, sha
from scripts.run_lightweight_word_selection import derive
from scripts.audit_common_budget_pipeline import audit_head
from scripts.run_yolo_f1_quality_study import seal


def recount(rows, truth, ignored):
    norm = lambda s: ''.join(c for c in s.casefold() if c.isalnum())
    target = Counter(norm(g['text']) for g in truth if norm(g['text']))
    tp = n = 0
    for row, skip in zip(rows, ignored):
        if skip:
            continue
        n += 1; word = norm(row['raw_text'])
        if word and target[word]:
            tp += 1; target[word] -= 1
    return dict(tp=tp, fp=n-tp, fn=len(truth)-tp, detected_words=n,
        gt_words=len(truth), all_outputs=len(rows), ignored_outputs=sum(ignored))


def total(rows):
    rows = list(rows)
    keys = ['tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs']
    v = {k: sum(r[k] for r in rows) for k in keys}
    den = v['detected_words']+v['gt_words']
    return dict(v, images=len(rows), f1=2*v['tp']/den if den else 0.)


def lexicon(plan, data, fitting):
    entries = {s.strip().upper(): s.strip() for s in
        Path(plan['config']['common_vocabulary']).read_text().splitlines()}
    for sid in fitting:
        for g in data[sid]['ground_truth']:
            entries[g['text'].strip().upper()] = g['text'].strip()
    return DeepSoloLexicon(sorted(entries), entries, distance_exclusive=1.5)


def model_fit(derived, ids, recipe, seed):
    x = np.concatenate([derived[s]['features'][np.logical_not(derived[s]['ignored'])] for s in ids])
    y = np.concatenate([derived[s]['labels'][np.logical_not(derived[s]['ignored'])] for s in ids])
    if recipe['kind'] == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(C=recipe['C'],
            max_iter=recipe['iterations'], random_state=seed))
    else:
        model = HistGradientBoostingClassifier(max_iter=recipe['iterations'],
            max_leaf_nodes=recipe['leaves'], min_samples_leaf=recipe['minimum_leaf'],
            l2_regularization=recipe['l2'], early_stopping=False, random_state=seed)
    return model.fit(x, y)


def independent_admit(rows, probabilities, threshold, iou):
    def overlap(a, b):
        inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/union if union else 0.
    chosen = []
    for i in sorted(range(len(rows)), key=lambda j: (-probabilities[j], rows[j]['proposal_id'])):
        if probabilities[i] < threshold or any(overlap(rows[i]['bbox'], r['bbox']) >= iou for r in chosen):
            continue
        chosen.append(rows[i])
    return chosen


def admission_audit(root):
    plan = read(root/'resolved_configuration.json'); data = read(root/'train_data.json')
    c = plan['config']; fit = plan['partitions']['fit']; val = plan['partitions']['validation']
    full_lex = lexicon(plan, data, fit)
    if full_lex.pairs != read(root/'fit_vocabulary.json'):
        raise ValueError('FIT-only vocabulary reconstruction differs')
    refits = trials = replays = 0; pins = {}
    for arm, sources in c['arms'].items():
        folder = root/'models'/arm; selected = read(folder/'admission_frozen.json')
        winner = selected['winner']; outputs = read(folder/'training_outputs.json')
        if selected['test_used_for_selection'] or sha(folder/winner['model']) != selected['model_sha256']:
            raise ValueError('Frozen admission changed')
        full = derive(plan, data, fit+val, sources, full_lex)
        fitted = model_fit(full, fit, winner['recipe'], c['seed'])
        actual = joblib.load(folder/winner['model']); refits += 1
        for sid in fit+val:
            x = full[sid]['features']
            if len(x):
                np.testing.assert_allclose(fitted.predict_proba(x), actual.predict_proba(x), atol=1e-12, rtol=1e-12)
        all_trials = read(folder/'trials.json'); probabilities = {}
        for name in {t['model'] for t in all_trials}:
            model = joblib.load(folder/name)
            probabilities[name] = {s: model.predict_proba(full[s]['features'])[:, 1] if len(full[s]['rows']) else [] for s in val}
        for t in all_trials:
            values = []
            for sid in val:
                d = full[sid]; kept = independent_admit(d['rows'], probabilities[t['model']][sid], t['threshold'], t['duplicate_iou'])
                flags = {r['proposal_id']: v for r, v in zip(d['rows'], d['ignored'])}
                values.append(recount(kept, data[sid]['ground_truth'], [flags[r['proposal_id']] for r in kept]))
            if total(values) != t['counts']:
                raise ValueError('Independent admission trial recount differs')
            trials += 1
        maximum = max(t['counts']['f1'] for t in all_trials)
        expected = min((t for t in all_trials if t['counts']['f1'] >= maximum-c['selection_tolerance']),
            key=lambda t: (t['model_bytes'], -t['counts']['f1'], t['emitted'], t['model'], -t['threshold'], t['duplicate_iou']))
        if expected != winner:
            raise ValueError('Admission winner criterion differs')
        covered = []
        for i, f in enumerate(read(folder/'word_folds.json')['folds']):
            if set(f['fit_ids']) != set(fit)-set(f['heldout_ids']):
                raise ValueError('Word-fold image contamination')
            fitting_rgb = {data[s]['rgb_sha256'] for s in f['fit_ids']}
            if fitting_rgb & {data[s]['rgb_sha256'] for s in f['heldout_ids']}:
                raise ValueError('Word-fold RGB contamination')
            lex = lexicon(plan, data, f['fit_ids'])
            derived = derive(plan, data, f['fit_ids']+f['heldout_ids'], sources, lex)
            model = model_fit(derived, f['fit_ids'], winner['recipe'], c['seed']+i); refits += 1
            if sha(folder/f'fold_{i}.joblib') != f['model_sha256']:
                raise ValueError('Changed word-fold model')
            for sid in f['heldout_ids']:
                d = derived[sid]
                p = model.predict_proba(d['features'])[:, 1] if len(d['rows']) else []
                np.testing.assert_allclose(p, outputs[sid]['probabilities'], atol=1e-12, rtol=1e-12)
                if d['rows'] != outputs[sid]['proposals']:
                    raise ValueError('Fold-safe word proposals differ')
                kept = independent_admit(d['rows'], p, winner['threshold'], winner['duplicate_iou'])
                if [r['proposal_id'] for r in kept] != [r['proposal_id'] for r in outputs[sid]['rows']]:
                    raise ValueError('Word OOF admission differs')
                flags = {r['proposal_id']: v for r, v in zip(d['rows'], d['ignored'])}
                if recount(kept, data[sid]['ground_truth'], [flags[r['proposal_id']] for r in kept]) != outputs[sid]['counts']:
                    raise ValueError('Word OOF counts differ')
                replays += 1
            covered.extend(f['heldout_ids'])
        if sorted(covered) != sorted(fit):
            raise ValueError('Incomplete word OOF coverage')
        for file in [folder/winner['model'], folder/'admission_frozen.json', folder/'training_outputs.json', folder/'word_folds.json']:
            pins[str(file)] = sha(file)
    for path, digest in read(root/'training_input_sha256.json').items():
        if sha(path) != digest:
            raise ValueError('Changed cached neural prediction')
    return dict(status='PASS', independent_word_refits=refits,
        admission_trial_recounts=trials, oof_image_replays=replays, artifact_sha256=pins)


def router_audit(root):
    plan = read(root/'resolved_configuration.json'); c = plan['config']; a = plan['admission_plan']
    fit, val = a['partitions']['fit'], a['partitions']['validation']
    data = read(Path(c['admission_root'])/'train_data.json')
    fallback = read(root/'train_fallback.json'); frozen = read(root/'pipeline_frozen.json')
    if frozen['test_used_for_selection']:
        raise ValueError('TEST-selected pipeline')
    refits = trials = branch_replays = deployment_replays = 0; pins = {}
    for arm in a['config']['arms']:
        folder = root/'models'/arm; chosen = read(folder/'router_frozen.json')['winner']
        branches = read(folder/'train_branches.json')
        output = read(Path(c['admission_root'])/'models'/arm/'training_outputs.json')
        policy = read(folder/'reconciliation_frozen.json')['winner']['policy']
        for sid in fit+val:
            light = output[sid]['rows']; fused = list(fallback[sid]['rows'])
            if policy['confidence'] is not None:
                for row in light:
                    if row['admission_probability'] >= policy['confidence'] and not any(
                            other['bbox'] is not None and overlap_iou(row['bbox'], other['bbox']) >=
                            policy['maximum_overlap'] for other in fused):
                        fused.append(row)
            for name, rows in [('light', light), ('deepsolo', fused)]:
                ignored = [r['ignored'] if r['word_source'] == 'deepsolo' else
                    data[sid]['ignored'][r['word_source']][r['source_index']] for r in rows]
                if recount(rows, data[sid]['ground_truth'], ignored) != branches[sid][name]:
                    raise ValueError('Independent reconciled branch counts differ')
                branch_replays += 1
        with np.load(folder/'image_features.npz') as z:
            ids = z['ids'].tolist(); pos = {s: i for i, s in enumerate(ids)}
            groups = {k: z[k] for k in z.files if k != 'ids'}
        for sid in fit+val:
            out = output[sid]
            observable = image_groups(out['proposals'], out['rows'], out['probabilities'],
                a['records'][sid]['image_wh'], a['config']['token_feature_cutoffs'])
            for group, values in observable.items():
                np.testing.assert_allclose(values, groups[group][pos[sid]], atol=1e-12, rtol=1e-12)
        deployed = {}; pipeline = LightweightWordPipeline(root, arm=arm, verify_qualification=False)
        dictionary = lexicon(a, data, fit)
        for sid in val:
            observed = pipeline.observe({s: data[sid]['sources'][s] for s in pipeline.sources},
                                       a['records'][sid]['image_wh'], dictionary)
            for group, values in observed['groups'].items():
                np.testing.assert_allclose(values, groups[group][pos[sid]], atol=1e-12, rtol=1e-12)
            if [r['proposal_id'] for r in observed['rows']] != [r['proposal_id'] for r in output[sid]['rows']]:
                raise ValueError('Annotation-free word deployment differs')
            deployed[sid] = observed; deployment_replays += 1
        scale = {s: 1+sum(not r['lexical_alternative'] for r in output[s]['proposals']) for s in ids}
        y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
            branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']]
            for s in fit])/np.array([scale[s] for s in fit])[:, None]
        if chosen['model']:
            bundle = joblib.load(folder/chosen['model'])
            x = np.concatenate([groups[g][[pos[s] for s in fit]] for g in bundle['groups']], axis=1)
            audit_head(bundle, x, y, np.array([scale[s] for s in fit]), c['seed']); refits += 1
        predictions = {}; all_trials = read(folder/'trials.json')
        for name in {t['model'] for t in all_trials if t['model']}:
            b = joblib.load(folder/name)
            x = np.concatenate([groups[g][[pos[s] for s in val]] for g in b['groups']], axis=1)
            x = b['scaler'].transform(x); models = b['models']
            predictions[name] = models[0].predict(x) if len(models) == 1 else np.stack([m.predict(x) for m in models], axis=1)
        k = round(len(val)*c['fallback_images']/c['total_images'])
        for t in all_trials:
            if t['model']:
                scores = {s: float(2*p[0]-t['reference']*p[1])*scale[s] for s, p in zip(val, predictions[t['model']])}
            else:
                i = t['heuristic_index']; scores = {s: groups['risk'][pos[s], i]*(-1 if i in (1, 3) else 1) for s in val}
            accepted = sorted(val, key=lambda s: (-scores[s], a['records'][s]['rgb_sha256'], s))[:len(val)-k]
            counts = total(branches[s]['light' if s in accepted else 'deepsolo'] for s in val)
            if counts != t['result']['combined'] or set(accepted) != set(t['result']['accepted_ids']):
                raise ValueError('Independent gate replay differs')
            trials += 1
        best = max(t['result']['combined']['f1'] for t in all_trials)
        expected = min((t for t in all_trials if t['result']['combined']['f1'] >= best-c['selection_tolerance']),
            key=lambda t: (t['model_bytes'], -t['result']['combined']['f1'], str(t.get('model')), t.get('reference', 0), t.get('heuristic_index', -1)))
        if expected != chosen:
            raise ValueError('Gate selection criterion differs')
        decision = pipeline.route(deployed, {s: a['records'][s]['rgb_sha256'] for s in val}, fallback_images=k)
        if set(decision['accepted_ids']) != set(chosen['result']['accepted_ids']):
            raise ValueError('Annotation-free image deployment differs')
        for path in [folder/'router_frozen.json', folder/'reconciliation_frozen.json'] + ([folder/chosen['model']] if chosen['model'] else []):
            pins[str(path)] = sha(path)
    for path in [root/'pipeline_frozen.json', root/'resolved_configuration.json']:
        pins[str(path)] = sha(path)
    best = max(t['winner']['result']['combined']['f1'] for t in frozen['trials'])
    expected = min((t for t in frozen['trials'] if t['winner']['result']['combined']['f1'] >= best-c['selection_tolerance']),
        key=lambda t: (t['source_networks'], t['admission_model_bytes']+t['winner']['model_bytes'],
            -t['winner']['result']['combined']['f1'], t['arm']))
    if expected != frozen['choice']:
        raise ValueError('Global validation/complexity choice differs')
    return dict(status='PASS', selected_image_head_refits=refits,
        router_trial_replays=trials, independent_branch_recounts=branch_replays,
        annotation_free_deployment_replays=deployment_replays, artifact_sha256=pins)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--admission-root', type=Path, required=True)
    parser.add_argument('--router-root', type=Path, required=True)
    parser.add_argument('--report-name', default='independent_audit_v1.json')
    args = parser.parse_args()
    with threadpool_limits(limits=read(args.admission_root/'resolved_configuration.json')['config']['cpu_threads']):
        admission = admission_audit(args.admission_root); router = router_audit(args.router_root)
    pins = dict(admission['artifact_sha256'], **router['artifact_sha256'])
    report = dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        smoke=read(args.admission_root/'resolved_configuration.json')['smoke'],
        admission=admission, router=router, auditor_sha256=sha(__file__),
        deployment_code_sha256=sha(Path(__file__).resolve().parents[1]/'recognition/lightweight_word_deployment.py'),
        actual_new_neural_inferences=0, test_used_for_selection=False)
    path = args.router_root/args.report_name; seal(path, report)
    seal(args.router_root/'deployment_qualification.json', dict(status='PASS',
        audit_path=str(path), audit_sha256=sha(path), artifact_sha256=pins))
    print(json.dumps(dict(status='PASS', admission_refits=admission['independent_word_refits'],
        image_refits=router['selected_image_head_refits'], report=str(path))), flush=True)


if __name__ == '__main__':
    main()
