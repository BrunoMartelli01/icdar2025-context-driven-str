"""Independent TRAIN recounts, model refits and deployment replay before TEST."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from recognition.fixed_expert_budget import observed_features, with_word_risk
from recognition.adapted_word_policy import emit
from recognition.fixed_expert_deployment import FixedExpertPipeline
from scripts.audit_common_budget_pipeline import audit_head
from scripts.audit_lightweight_word_pipeline import recount, total
from scripts.crop_clip_runtime import array_sha
from scripts.icdar_budget_data import read, sha
from scripts.prepare_fixed_expert_budget import vocabulary
from scripts.train_lightweight_word_router import teacher
from scripts.run_yolo_f1_quality_study import seal


def norm(s):
    return ''.join(c for c in s.casefold() if c.isalnum())


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def heuristic(o, p, name):
    keep = {int(r['source_row']) for r in o['rows']}
    if name == 'expected_errors':
        return -sum(1-v if i in keep else v for i, v in enumerate(p))
    if name == 'weakest_word':
        return min((p[i] for i in keep), default=0.)
    if name == 'weakest_sequence':
        return min((r['sequence_confidence_native'] for r in o['rows']), default=0.)
    raise ValueError('Unknown heuristic')


def winner_key(t):
    return (t['scalars'], -t['result']['combined']['f1'], len(t['models']),
            t['models'], t['reference'] or 0, t['heuristic'] or '', t['guard'])


def global_key(r):
    return (r['additional_map_models'], r['extra_reader_crops'],
            not r['expert_image_excluded'], r['winner']['scalars'],
            -r['winner']['result']['combined']['f1'], r['view'], r['target_name'])


def audit(root):
    plan = read(root/'resolved_configuration.json'); c = plan['config']
    if plan['mode'] != 'approximate' or plan['test_labels_opened']:
        raise ValueError('TRAIN-only APPROXIMATE audit required')
    pins = {}; counters = dict(word_model_refits=0, selected_count_head_refits=0,
        branch_recounts=0, trial_replays=0, deployment_replays=0, crop_identity_checks=0,
        fallback_replays=0, expert_exclusion_folds=0)
    def pin(path):
        path = Path(path); pins[str(path)] = sha(path)
    for path, digest in c['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed pinned scientific input: '+path)
        pin(path)
    for rel, digest in plan['source_sha256'].items():
        if sha(root/'code_snapshot'/rel) != digest:
            raise ValueError('Changed immutable training source')
    original = read(c['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['train']}
    fit = plan['partitions']['fit']; val = plan['partitions']['validation']; ids = fit+val
    keys = {s: records[s]['rgb_sha256'] for s in ids}
    if {keys[s] for s in fit} & {keys[s] for s in val}:
        raise ValueError('FIT/validation RGB overlap')
    if 'oof' in c['required_views']:
        manifest = read(c['folds_manifest']); covered = []
        for f in manifest['folds']:
            folder = Path(c['fold_root'])/('fold_'+str(f['fold']))
            m = read(folder/'training/full/dataset_manifest.json')
            if m['test_used'] or set(m['records']) != set(f['fit_ids']+val):
                raise ValueError('Actual detector fitting/validation population differs')
            if set(f['fit_ids']) != set(fit)-set(f['heldout_ids']):
                raise ValueError('Detector OOF fitting identities differ')
            if {keys[s] for s in m['records']} & {keys[s] for s in f['heldout_ids']}:
                raise ValueError('Detector fitting/heldout RGB contamination')
            if set(read(folder/'heldout/index.json')) != set(f['heldout_ids']):
                raise ValueError('Detector heldout export coverage differs')
            for file in ('training/full/dataset_manifest.json', 'training/full/completion.json',
                         'heldout/index.json', 'heldout/completion.json', 'heldout/resolved_configuration.json'):
                pin(folder/file)
            covered.extend(f['heldout_ids']); counters['expert_exclusion_folds'] += 1
        if sorted(covered) != sorted(fit):
            raise ValueError('Detector OOF coverage differs')
    fallback = read(root/'data/fallbacks.json')
    for name, profile in c['fallback_profiles'].items():
        lex = vocabulary(profile['train_lexicon'])
        for sid in ids:
            value = teacher(profile, None, records[sid], lex)
            for row in value['rows']:
                if row.get('invalid_envelope') is not None:
                    row.update(raw_text='', text='')
            saved = fallback[name][sid]
            if any(value[k] != saved[k] for k in ('rows', 'source_path', 'source_sha256')):
                raise ValueError('Fallback TRAIN teacher replay differs')
            counters['fallback_replays'] += 1
    choices = []
    for view in c['required_views']:
        data = root/'data'/view; folder = root/'models'/view
        obs = read(data/'observations.json'); labels = read(data/'labels.json')
        case = read(data/'case.json'); probabilities = read(folder/'word_probabilities.json')
        if sorted(obs) != sorted(ids) or sorted(labels) != sorted(ids):
            raise ValueError('Training data population differs')
        for path, digest in read(data/'input_sha256.json').items():
            if sha(path) != digest:
                raise ValueError('Changed real expert export input')
            pin(path)
            value = read(path)
            if 'raw_rows' not in value:
                continue
            matching = [s for s in ids if records[s]['image_sha256'] == value['image_sha256']]
            if len(matching) != 1 or value['ground_truth_used']:
                raise ValueError('Expert source/image identity differs')
            sid = matching[0]
            emitted = value['rows']
            if c.get('emission_override') is not None:
                override = c['emission_override']
                emitted, _ = emit(value['raw_rows'], vocabulary(override['train_lexicon']),
                                  read(override['policy'])['winner']['policy'])
            if value['raw_rows'] != obs[sid]['raw'] or emitted != obs[sid]['rows']:
                raise ValueError('Source expert word emissions changed')
            if view == 'oof' and sid in fit and not value['image_excluded']:
                raise ValueError('Non-excluded expert in OOF FIT')
            with Image.open(records[sid]['image_path']) as im:
                rgb = np.asarray(im.convert('RGB'))
            for row in value['raw_rows']:
                a, b, d, e = row['bbox']
                if array_sha(np.ascontiguousarray(rgb[b:e, a:d])) != row['crop_rgb_sha256']:
                    raise ValueError('PARSeq crop does not bind to original RGB')
                counters['crop_identity_checks'] += 1
        xword = {}; groups = {}; targets = {}; care = {}
        for sid in ids:
            o = obs[sid]
            xword[sid], groups[sid] = observed_features(o['raw'], o['rows'], o['trace'], o['image_wh'], o['map_summary'])
            groups[sid] = with_word_risk(groups[sid], probabilities[sid], o['trace'])
            if sid in fit:
                emitted = {int(r['source_row']): r for r in o['rows']}
                truth = {norm(g['text']) for g in labels[sid]['ground_truth'] if norm(g['text'])}
                targets[sid] = np.array([int(bool(norm(emitted.get(i, r)['raw_text'])) and
                    norm(emitted.get(i, r)['raw_text']) in truth) for i, r in enumerate(o['raw'])])
                care[sid] = np.logical_not(labels[sid]['raw_ignored'])
        with np.load(folder/'features.npz') as z:
            if z['ids'].tolist() != ids:
                raise ValueError('Feature image order differs')
            for g in groups[ids[0]]:
                np.testing.assert_allclose(z[g], np.stack([groups[s][g] for s in ids]), atol=1e-12, rtol=1e-12)
        folds = read(folder/'word_folds.json')['folds']; covered = []
        for i, f in enumerate(folds+[dict(fit_ids=fit, heldout_ids=val)]):
            if i < len(folds):
                if set(f['fit_ids']) != set(fit)-set(f['heldout_ids']):
                    raise ValueError('Word-risk fold fitting population differs')
                if {keys[s] for s in f['fit_ids']} & {keys[s] for s in f['heldout_ids']}:
                    raise ValueError('Word-risk fitting/heldout RGB contamination')
                covered.extend(f['heldout_ids'])
            path = folder/(f'word_fold_{i}.joblib' if i < len(folds) else 'word_full.joblib')
            if i < len(folds) and sha(path) != f['sha256']:
                raise ValueError('Changed word fold model')
            setting = c['word_risk']; xx = np.concatenate([xword[s][care[s]] for s in f['fit_ids']])
            yy = np.concatenate([targets[s][care[s]] for s in f['fit_ids']])
            model = HistGradientBoostingClassifier(max_iter=setting['iterations'],
                max_leaf_nodes=setting['leaves'], min_samples_leaf=setting['minimum_leaf'],
                l2_regularization=setting['l2'], early_stopping=False,
                random_state=c['seed']+(i if i < len(folds) else 0)).fit(xx, yy)
            actual = joblib.load(path)
            for sid in f['heldout_ids']:
                expected = model.predict_proba(xword[sid])[:, 1] if len(xword[sid]) else np.empty(0)
                got = actual.predict_proba(xword[sid])[:, 1] if len(xword[sid]) else np.empty(0)
                np.testing.assert_allclose(expected, got, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(expected, probabilities[sid], atol=1e-12, rtol=1e-12)
            pin(path); counters['word_model_refits'] += 1
        if sorted(covered) != sorted(fit):
            raise ValueError('Word-risk fold coverage differs')
        def matrix(samples, names):
            return np.stack([np.concatenate([groups[s][g] for g in names]) for s in samples])
        scales = {s: len(obs[s]['raw'])+1 for s in ids}
        completed = read(folder/'completed.json')
        for detail in completed['results']:
            target = folder/detail['target_name']; branches = read(target/'train_branches.json')
            policy = detail['complement']; name = detail['fallback_profile']
            for sid in ids:
                o = obs[sid]; light = o['rows']; fused = list(fallback[name][sid]['rows'])
                if policy['confidence'] is not None:
                    for row in light:
                        if probabilities[sid][int(row['source_row'])] >= policy['confidence'] and not any(
                                other['bbox'] is not None and overlap(row['bbox'], other['bbox']) >= policy['maximum_overlap']
                                for other in fused):
                            fused.append(dict(row, fixed_light_complement=True))
                for branch_name, rows in [('light', light), ('deepsolo', fused)]:
                    ignored = [(labels[sid]['raw_ignored'][int(r['source_row'])]
                        if branch_name == 'light' or r.get('fixed_light_complement') else r['ignored']) for r in rows]
                    if recount(rows, labels[sid]['ground_truth'], ignored) != branches[sid][branch_name]:
                        raise ValueError('Independent fixed-emission branch recount differs')
                    counters['branch_recounts'] += 1
            chosen = read(target/'winner.json'); y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
                branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']]
                for s in fit])/np.array([scales[s] for s in fit])[:, None]
            for model_name in chosen['models']:
                bundle = joblib.load(target/model_name)
                audit_head(bundle, matrix(fit, bundle['groups']), y, np.array([scales[s] for s in fit]), c['seed'])
                counters['selected_count_head_refits'] += 1; pin(target/model_name)
            trials = read(target/'trials.json'); prediction = {}
            for model_name in {m for t in trials for m in t['models']}:
                bundle = joblib.load(target/model_name); x = bundle['scaler'].transform(matrix(val, bundle['groups']))
                models = bundle['models']; prediction[model_name] = models[0].predict(x) if len(models) == 1 else np.stack([m.predict(x) for m in models], axis=1)
            quota = round(len(val)*case['fallback_images']/case['total_images'])
            for t in trials:
                if t['models']:
                    pairs = np.mean([prediction[m] for m in t['models']], axis=0)
                    scores = {s: float(2*v[0]-t['reference']*v[1])*scales[s] for s, v in zip(val, pairs)}
                else:
                    scores = {s: heuristic(obs[s], probabilities[s], t['heuristic']) for s in val}
                eligible = [s for s in val if obs[s]['rows']] if t['guard'] else val
                accepted = sorted(eligible, key=lambda s: (-scores[s], keys[s], s))[:len(val)-quota]
                aggregate = total(branches[s]['light' if s in accepted else 'deepsolo'] for s in val)
                if (len(accepted) != len(val)-quota or sorted(accepted) != t['result']['accepted_ids'] or
                        aggregate != t['result']['combined'] or t['result']['deepsolo_images'] != quota):
                    raise ValueError('Independent trial rank/count/budget replay differs')
                counters['trial_replays'] += 1
            best = max(t['result']['combined']['f1'] for t in trials)
            expected = min((t for t in trials if t['result']['combined']['f1'] >= best-c['selection_tolerance']), key=winner_key)
            if chosen != expected or chosen != detail['winner']:
                raise ValueError('Per-target winner rule differs')
            if detail['primary_eligible']:
                choices.append(detail)
            for file in ('winner.json', 'trials.json', 'train_branches.json'):
                pin(target/file)
        for file in ('case.json', 'observations.json', 'labels.json', 'input_sha256.json'):
            pin(data/file)
        pin(folder/'completed.json'); pin(folder/'word_probabilities.json'); pin(folder/'features.npz')
    frozen = read(root/'selection_frozen.json'); best = max(r['winner']['result']['combined']['f1'] for r in choices)
    expected = min((r for r in choices if r['winner']['result']['combined']['f1'] >= best-c['selection_tolerance']), key=global_key)
    if frozen['choices'] != choices or frozen['winner'] != expected or frozen['test_used_for_selection']:
        raise ValueError('Global validation-only freeze differs')
    pipeline = FixedExpertPipeline(root, verify_qualification=False); view = expected['view']
    obs = read(root/'data'/view/'observations.json'); deployed = {}
    for sid in val:
        o = obs[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'],
            trace=o['trace'], map_summary=o['map_summary']); counters['deployment_replays'] += 1
    quota = round(len(val)*c['fallback_images']/c['total_images'])
    decision = pipeline.route(deployed, {s: keys[s] for s in val}, fallback_images=quota)
    if decision['accepted_ids'] != expected['winner']['result']['accepted_ids']:
        raise ValueError('Annotation-free deployment route differs')
    for file in ('selection_frozen.json', 'resolved_configuration.json', 'data/fallbacks.json', 'data/fallback_input_sha256.json'):
        pin(root/file)
    for module in ('recognition/fixed_expert_deployment.py', 'recognition/fixed_expert_budget.py',
                   'recognition/common_budget_pipeline.py'):
        pin(Path(__file__).resolve().parents[1]/module)
    report = dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        counters=counters, validation=expected['winner']['result'],
        test_used=False, artifact_sha256=pins, auditor_sha256=sha(__file__))
    path = root/'independent_audit_v1.json'; seal(path, report)
    seal(root/'deployment_qualification.json', dict(status='PASS', audit_path=str(path),
        audit_sha256=sha(path), artifact_sha256=pins))
    print(__import__('json').dumps(dict(status='PASS', counters=counters, validation=report['validation'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root)
