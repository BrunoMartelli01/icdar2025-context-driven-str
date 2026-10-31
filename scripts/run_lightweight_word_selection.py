"""TRAIN-only admission experiment for a lightweight fixed-budget OCR pipeline."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import counts, micro, grouped_folds
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.fold_safe_teacher import canonical_vocabulary
from recognition.group_word_selection import reading_confidence
from recognition.lightweight_word_selection import (
    proposals, feature_matrix, correctness_proxy, fit_admission, admit,
)
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def prepare(config_path, smoke, run_name=None):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    config = read(config_path)
    if config['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed pinned input: '+path)
    root = Path(config['output_root'])/(run_name or ('smoke' if smoke else 'admission_v1'))
    root.mkdir(parents=True, exist_ok=False)
    plan = read(Path(config['original_root'])/'resolved_configuration.json')
    parts = plan['partitions']['icdar13']
    partitions = dict(fit=parts['fit'], validation=sorted(
        parts['selection']+parts['calibration']), test=parts['test'])
    if smoke:
        partitions = {k: v[:(24 if k == 'fit' else 8)] for k, v in partitions.items()}
    records = {r['sample_id']: r for group in plan['records']['icdar13'].values()
               for r in group}
    groups = [{records[s]['rgb_sha256'] for s in ids} for ids in partitions.values()]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('RGB overlap across experiment roles')
    source = {}
    for folder in ('recognition', 'scripts', 'segmentation', 'localize', 'models',
                   'pipeline', 'caption', 'scoring', 'research'):
        for file in (REPO/folder).rglob('*.py'):
            rel = file.relative_to(REPO); dest = root/'code_snapshot'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest); source[str(rel)] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(config=config,
        config_sha256=sha(config_path), mode='approximate',
        claim_label='approximate reproduction', smoke=smoke,
        result_class='smoke-test result' if smoke else 'approximate reproduction',
        partitions=partitions, records=records, source_sha256=source,
        historical_test_exposure=True, test_labels_opened=False,
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     device='cpu', actual_new_neural_inferences=0)))
    return root


def load_source(config, record, source, role):
    spec = config['sources'][source]
    if spec['kind'] == 'word_unet':
        original_role = ('fit' if record['sample_id'] in config['original_fit_ids'] else
            'selection' if record['sample_id'] in config['original_selection_ids'] else
            'calibration') if role != 'test' else 'test'
        path = Path(spec['root'])/original_role/'samples'/(key(record['sample_id'])+'.json')
        value = read(path)
        rows = [dict(r, confidence=reading_confidence(r)['confidence_balanced'])
                for r in value['raw_rows']]
        if value['ground_truth_used_for_prediction']:
            raise ValueError('GT-dependent light predictions')
    else:
        path = Path(spec['root'])/(key(record['sample_id'])+'.json')
        value = read(path); rows = value['rows']
        if value['ground_truth_used']:
            raise ValueError('GT-dependent light predictions')
    if value['image_sha256'] != record['image_sha256']:
        raise ValueError('Changed prediction/RGB binding')
    return rows, str(path), sha(path)


def prepare_training(plan, root):
    config = plan['config']; data = {}; pins = {}
    for role in ('fit', 'validation'):
        for sid in plan['partitions'][role]:
            record = plan['records'][sid]
            truth, valid = annotation(record)
            sources = {}; ignored = {}
            for name in config['sources']:
                rows, path, digest = load_source(config, record, name, role)
                pins[path] = digest; sources[name] = rows
                ignored[name] = [p['ignored'] for p in predictions(rows, record, valid)]
            data[sid] = dict(sources=sources, ignored=ignored, ground_truth=truth,
                image_wh=record['image_wh'], rgb_sha256=record['rgb_sha256'])
    seal(root/'train_data.json', data)
    seal(root/'training_input_sha256.json', pins)


def dictionary(plan, data, fit_ids):
    base = Path(plan['config']['common_vocabulary']).read_text().splitlines()
    targets = {sid: [g['text'] for g in data[sid]['ground_truth']] for sid in fit_ids}
    canonical = canonical_vocabulary(base, fit_ids, targets)
    return DeepSoloLexicon(sorted(canonical), canonical, distance_exclusive=1.5)


def derive(plan, data, ids, sources, lexicon):
    result = {}
    for sid in ids:
        scene = data[sid]
        rows = proposals({s: scene['sources'][s] for s in sources}, lexicon)
        ignored = [scene['ignored'][r['word_source']][r['source_index']] for r in rows]
        result[sid] = dict(rows=rows, ignored=ignored,
            features=feature_matrix(rows, scene['image_wh'], plan['config']['source_names'],
                                   plan['config']['token_feature_cutoffs']),
            labels=correctness_proxy(rows, scene['ground_truth']))
    return result


def train_model(derived, ids, recipe, seed):
    x = np.concatenate([d['features'][np.logical_not(d['ignored'])]
                        for s in ids for d in [derived[s]]])
    y = np.concatenate([d['labels'][np.logical_not(d['ignored'])]
                        for s in ids for d in [derived[s]]])
    return fit_admission(x, y, recipe, seed)


def evaluate_admission(data, derived, ids, probabilities, threshold, iou):
    totals = []; selected = {}; n = 0
    for sid in ids:
        d = derived[sid]
        rows = admit(d['rows'], probabilities[sid], threshold=threshold, duplicate_iou=iou)
        flags = {r['proposal_id']: ignored for r, ignored in zip(d['rows'], d['ignored'])}
        totals.append(counts(rows, data[sid]['ground_truth'],
            [flags[r['proposal_id']] for r in rows]))
        selected[sid] = rows; n += len(rows)
    return micro(totals), selected, n


def train(plan, root):
    config = plan['config']; data = read(root/'train_data.json')
    fit = plan['partitions']['fit']; validation = plan['partitions']['validation']
    keys = {s: data[s]['rgb_sha256'] for s in fit}
    folds = grouped_folds(fit, keys, config['folds'], config['seed'])
    full_dictionary = dictionary(plan, data, fit)
    seal(root/'fit_vocabulary.json', full_dictionary.pairs)
    outcomes = {}
    for arm, sources in config['arms'].items():
        folder = root/'models'/arm; folder.mkdir(parents=True, exist_ok=False)
        full = derive(plan, data, fit+validation, sources, full_dictionary)
        trials = []; probabilities = {}
        for index, recipe in enumerate(config['recipes']):
            model = train_model(full, fit, recipe, config['seed'])
            name = f'admission_{index}.joblib'; joblib.dump(model, folder/name)
            scores = {s: model.predict_proba(full[s]['features'])[:, 1] if
                len(full[s]['rows']) else np.empty(0) for s in validation}
            probabilities[name] = scores
            for threshold in config['thresholds']:
                for iou in config['duplicate_ious']:
                    result, _, n = evaluate_admission(data, full, validation, scores, threshold, iou)
                    trials.append(dict(model=name, recipe=recipe, threshold=threshold,
                        duplicate_iou=iou, counts=result, emitted=n,
                        model_bytes=(folder/name).stat().st_size))
        best = max(v['counts']['f1'] for v in trials)
        winner = min((v for v in trials if v['counts']['f1'] >= best-config['selection_tolerance']),
            key=lambda v: (v['model_bytes'], -v['counts']['f1'], v['emitted'],
                           v['model'], -v['threshold'], v['duplicate_iou']))
        seal(folder/'trials.json', trials)
        seal(folder/'admission_frozen.json', dict(status='FROZEN', winner=winner,
            fit_ids=fit, validation_ids=validation, test_used_for_selection=False,
            model_sha256=sha(folder/winner['model'])))
        # End-to-end word-fold predictions reconstruct the dictionary without
        # heldout labels. Each fold also excludes the heldout images from fitting.
        oof = {}; audits = []
        for index, held in enumerate(folds):
            remaining = sorted(set(fit)-set(held))
            lexicon = dictionary(plan, data, remaining)
            derived = derive(plan, data, remaining+held, sources, lexicon)
            model = train_model(derived, remaining, winner['recipe'], config['seed']+index)
            path = folder/f'fold_{index}.joblib'; joblib.dump(model, path)
            for sid in held:
                d = derived[sid]
                p = model.predict_proba(d['features'])[:, 1] if len(d['rows']) else np.empty(0)
                selected = admit(d['rows'], p, threshold=winner['threshold'],
                    duplicate_iou=winner['duplicate_iou'])
                flags = {r['proposal_id']: v for r, v in zip(d['rows'], d['ignored'])}
                oof[sid] = dict(rows=selected, proposals=d['rows'], probabilities=p.tolist(),
                    counts=counts(selected, data[sid]['ground_truth'],
                        [flags[r['proposal_id']] for r in selected]), fold=index)
            audits.append(dict(fit_ids=remaining, heldout_ids=held, model_sha256=sha(path),
                vocabulary_sha256=__import__('hashlib').sha256(json.dumps(
                    lexicon.pairs, sort_keys=True).encode()).hexdigest()))
        result, selected, _ = evaluate_admission(data, full, validation,
            probabilities[winner['model']], winner['threshold'], winner['duplicate_iou'])
        for sid in validation:
            oof[sid] = dict(rows=selected[sid], proposals=full[sid]['rows'],
                probabilities=probabilities[winner['model']][sid].tolist(), fold=None)
        seal(folder/'training_outputs.json', oof)
        seal(folder/'word_folds.json', dict(folds=audits, upstream_expert_oof=False))
        outcomes[arm] = dict(validation=result, fit_oof=micro(
            oof[s]['counts'] for s in fit), selected_policy=winner,
            deepsolo_images=0, total_validation_images=len(validation),
            deepsolo_percentage=0.)
        print(json.dumps(dict(stage='admission_frozen', arm=arm,
            validation_f1=result['f1'], fit_oof_f1=outcomes[arm]['fit_oof']['f1'],
            model=winner['model'], threshold=winner['threshold'],
            deepsolo_images=0, total_images=len(validation), deepsolo_percentage=0.)), flush=True)
    seal(root/'admission_complete.json', dict(status='COMPLETE', mode='approximate',
        claim_label='approximate reproduction', test_labels_opened=False,
        results=outcomes, actual_new_neural_inferences=0))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--run-name')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=['prepare', 'prepare_training', 'train', 'all'], default='all')
    args = parser.parse_args()
    root = args.run_root or prepare(args.config, args.smoke, args.run_name)
    plan = read(root/'resolved_configuration.json')
    for path, digest in plan['source_sha256'].items():
        if sha(REPO/path) != digest:
            raise ValueError('Executed source differs from frozen plan: '+path)
    with threadpool_limits(limits=plan['config']['cpu_threads']):
        if args.stage in ('prepare_training', 'all'):
            prepare_training(plan, root)
        if args.stage in ('train', 'all'):
            train(plan, root)
    print(json.dumps(dict(status='SUCCESS', root=str(root), stage=args.stage)), flush=True)


if __name__ == '__main__':
    main()
