"""Freeze admission, conditional fallback complement and a pre-fallback gate."""

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from recognition.clip_router_evaluation import deepsolo_predictions
from recognition.common_budget_pipeline import counts, micro, fit_count_head, route
from recognition.fold_safe_teacher import checked_response, correct_detections
from recognition.lightweight_word_router import image_groups, image_score, exact_plan
from recognition.lightweight_word_selection import reconcile
from recognition.fallback_reread import choose_transcript
from scripts.icdar_budget_data import read, sha, key, annotation
from scripts.run_lightweight_word_selection import dictionary
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def prepare(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed router input: '+file)
    source = Path(c['admission_root']); plan = read(source/'resolved_configuration.json')
    complete = read(source/'admission_complete.json')
    if complete['status'] != 'COMPLETE' or complete['test_labels_opened']:
        raise ValueError('Admission must complete without TEST evaluation')
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    files = {}
    for folder in ('recognition', 'scripts', 'segmentation', 'localize', 'models',
                   'pipeline', 'caption', 'scoring', 'research'):
        for file in (REPO/folder).rglob('*.py'):
            rel = file.relative_to(REPO); dest = root/'code_snapshot'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest); files[str(rel)] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction', config=c, config_sha256=sha(path),
        source_sha256=files, admission_plan=plan,
        admission_complete_sha256=sha(source/'admission_complete.json'),
        smoke=plan['smoke'], historical_test_exposure=True, test_used_for_selection=False))
    return root


def teacher(config, admission_plan, record, lexicon):
    protocol = read(config['train_protocol'])
    profile = read(config['fallback_profile'])
    if profile['checkpoint_sha256'] != protocol['source_identity']['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint/profile mismatch')
    if profile['native_resize'] != protocol['resize']:
        raise ValueError('Teacher resize/profile mismatch')
    path = Path(config['raw_train_root'])/(key(record['sample_id'])+'.json')
    response = checked_response(read(path), protocol['binding'], record['sample_id'],
                                record['image_sha256'])
    detections, _ = correct_detections(response, lexicon, profile['neural'],
                                      profile['unmatched_policy'])
    gt, valid = annotation(record)
    predicted = deepsolo_predictions(dict(detections=detections), record['image_wh'],
                                     record['orientation'], valid, 0.)
    rows = [dict(p, raw_text=p['text'], word_source='deepsolo') for p in predicted]
    if config.get('reread_profile') is not None:
        spec = config['reread_profile']; frozen = read(spec['policy'])
        if not frozen['promotion_eligible'] or frozen['test_used_for_selection']:
            raise ValueError('Reread policy is not validation-qualified')
        index = read(spec['index']); entry = index[record['sample_id']]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed fallback PARSeq rereading')
        reread = read(entry['path'])
        if reread['fallback_sha256'] != sha(path) or reread['ground_truth_used']:
            raise ValueError('Rereading/fallback binding differs')
        raw = {r['source_index']: r for r in reread['rows']}
        for row in rows:
            source = raw[row['instance_id']]
            text, trace = choose_transcript(dict(text=source['ctc_text']),
                source['rereadings'], lexicon, frozen['winner']['policy'])
            row.update(raw_text=text, text=text, reread_trace=trace)
    return dict(rows=rows, counts=counts(rows, gt, [r['ignored'] for r in rows]),
                source_path=str(path), source_sha256=sha(path))


def recount(selected, scene, record):
    # Evaluation only. No validity mask enters image_groups/admit/reconcile.
    flags = []
    for row in selected:
        if row.get('word_source') == 'deepsolo':
            flags.append(row['ignored'])
        else:
            flags.append(scene['ignored'][row['word_source']][row['source_index']])
    return counts(selected, scene['ground_truth'], flags)


def train(plan, root):
    c = plan['config']; a = plan['admission_plan']; source = Path(c['admission_root'])
    data = read(source/'train_data.json'); fit = a['partitions']['fit']; val = a['partitions']['validation']
    lexicon = dictionary(a, data, fit)
    fallback = {sid: teacher(c, a, a['records'][sid], lexicon) for sid in fit+val}
    # Targets on FIT correspond to the same dictionary exclusions as word OOF.
    folds = read(source/'models'/next(iter(a['config']['arms']))/'word_folds.json')['folds']
    for f in folds:
        folded = dictionary(a, data, f['fit_ids'])
        for sid in f['heldout_ids']:
            fallback[sid] = teacher(c, a, a['records'][sid], folded)
    seal(root/'train_fallback.json', fallback)
    trials_all = []; arm_details = {}
    quota = c['fallback_images']/c['total_images']
    for arm, sources in a['config']['arms'].items():
        folder = root/'models'/arm; folder.mkdir(parents=True, exist_ok=False)
        frozen = read(source/'models'/arm/'admission_frozen.json')
        outputs = read(source/'models'/arm/'training_outputs.json')
        # A replacement-only policy is always a candidate.
        policies = [dict(confidence=None, maximum_overlap=1.)] + [
            dict(confidence=p, maximum_overlap=iou)
            for p in c['complement_confidences'] for iou in c['complement_ious']]
        fusion_trials = []
        for policy in policies:
            values = [recount(reconcile(outputs[s]['rows'], fallback[s]['rows'], **policy),
                data[s], a['records'][s]) for s in val]
            result = micro(values)
            fusion_trials.append(dict(policy=policy, counts=result))
        fusion = max(enumerate(fusion_trials), key=lambda pair:
            (pair[1]['counts']['f1'], pair[1]['policy']['confidence'] is None,
             -(pair[0])))[1]
        seal(folder/'reconciliation_frozen.json', dict(winner=fusion,
            trials=fusion_trials, test_used_for_selection=False))
        branches = {}; groups = {}; scale = {}
        for sid in fit+val:
            out = outputs[sid]; record = a['records'][sid]
            light = recount(out['rows'], data[sid], record)
            fused = recount(reconcile(out['rows'], fallback[sid]['rows'],
                                      **fusion['policy']), data[sid], record)
            branches[sid] = dict(light=light, deepsolo=fused)
            groups[sid] = image_groups(out['proposals'], out['rows'],
                out['probabilities'], record['image_wh'], a['config']['token_feature_cutoffs'])
            scale[sid] = 1+sum(not row['lexical_alternative'] for row in out['proposals'])
        seal(folder/'train_branches.json', branches)
        np.savez_compressed(folder/'image_features.npz', ids=np.array(fit+val),
            **{g: np.stack([groups[s][g] for s in fit+val]) for g in groups[fit[0]]})
        y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
            branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']]
            for s in fit])/np.array([scale[s] for s in fit])[:, None]
        weights = np.array([scale[s] for s in fit]); trials = []
        keys = {s: a['records'][s]['rgb_sha256'] for s in val}
        n = round(len(val)*quota); vb = {s: branches[s] for s in val}
        for gi, names in enumerate(c['feature_groups']):
            x = np.stack([np.concatenate([groups[s][g] for g in names]) for s in fit])
            for mi, recipe in enumerate(c['head_recipes']):
                bundle = fit_count_head(x, y, weights, recipe, c['seed']); bundle['groups'] = names
                name = f'router_{gi}_{mi}.joblib'; joblib.dump(bundle, folder/name)
                for reference in c['references']:
                    scores = {s: image_score(groups[s], bundle, reference, scale[s]) for s in val}
                    decision = exact_plan(scores, keys, n); result = route(vb, decision['accepted_ids'])
                    trials.append(dict(model=name, feature_groups=names, recipe=recipe,
                        reference=reference, result=result,
                        model_bytes=(folder/name).stat().st_size, source_networks=len(sources)))
        # Cheap probability-only routes quantify whether fitting adds any value.
        for index in (0, 1, 2, 3):
            scores = {s: (-groups[s]['risk'][index] if index in (1, 3)
                         else groups[s]['risk'][index]) for s in val}
            point = exact_plan(scores, keys, n); result = route(vb, point['accepted_ids'])
            trials.append(dict(model=None, heuristic_index=index, result=result,
                model_bytes=0, source_networks=len(sources)))
        best = max(t['result']['combined']['f1'] for t in trials)
        winner = min((t for t in trials if t['result']['combined']['f1'] >= best-c['selection_tolerance']),
            key=lambda t: (t['model_bytes'], -t['result']['combined']['f1'], str(t.get('model')),
                           t.get('reference', 0), t.get('heuristic_index', -1)))
        seal(folder/'trials.json', trials)
        seal(folder/'router_frozen.json', dict(winner=winner,
            model_sha256=sha(folder/winner['model']) if winner['model'] else None,
            admission_frozen_sha256=sha(source/'models'/arm/'admission_frozen.json'),
            reconciliation_sha256=sha(folder/'reconciliation_frozen.json'), test_used_for_selection=False))
        arm_details[arm] = dict(winner=winner, fusion=fusion,
            source_networks=len(sources), admission_model_bytes=frozen['winner']['model_bytes'])
        trials_all.append(dict(arm=arm, **arm_details[arm]))
        print(json.dumps(dict(stage='gate_frozen', arm=arm,
            f1=winner['result']['combined']['f1'],
            deepsolo_images=n, total_images=len(val), deepsolo_percentage=100*n/len(val),
            router=winner.get('model'), fusion=fusion['policy'])), flush=True)
    best = max(t['winner']['result']['combined']['f1'] for t in trials_all)
    chosen = min((t for t in trials_all if t['winner']['result']['combined']['f1'] >= best-c['selection_tolerance']),
        key=lambda t: (t['source_networks'], t['admission_model_bytes']+t['winner']['model_bytes'],
            -t['winner']['result']['combined']['f1'], t['arm']))
    seal(root/'pipeline_frozen.json', dict(status='FROZEN', choice=chosen,
        trials=trials_all, test_used_for_selection=False, train_complete=True,
        target_validation_met=chosen['winner']['result']['combined']['f1'] >= c['minimum_f1']))
    print(json.dumps(dict(status='TRAIN_COMPLETE', chosen=chosen['arm'],
        validation_f1=chosen['winner']['result']['combined']['f1'],
        deepsolo_images=round(len(val)*quota), total_images=len(val),
        deepsolo_percentage=100*round(len(val)*quota)/len(val))), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'train', 'all'], default='all')
    args = parser.parse_args(); root = args.run_root or prepare(args.config)
    plan = read(root/'resolved_configuration.json')
    for path, digest in plan['source_sha256'].items():
        if sha(REPO/path) != digest:
            raise ValueError('Executed source differs from frozen router plan: '+path)
    if args.stage in ('train', 'all'):
        with threadpool_limits(limits=plan['config']['cpu_threads']):
            train(plan, root)


if __name__ == '__main__':
    main()
