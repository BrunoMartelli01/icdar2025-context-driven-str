"""Image-group OOF head selection, followed by a full TRAIN fit and fixed-budget test."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from recognition.lexical_budget_router import fit, predict
from recognition.router_fallback_replay import replay
from recognition.budget_target_oracle import analyze_budget
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['train', 'evaluate'], required=True)
    args = parser.parse_args()
    config = read(args.config)
    assert config['mode'] == 'approximate'
    for path, digest in config['pinned_inputs'].items():
        assert sha(path) == digest, path
    root = Path(config['study_root'])
    prior = Path(config['source_router_root'])
    plan = read(config['source_plan'])
    partition = plan['partitions']['icdar13']
    train_ids = partition['fit'] + partition['selection'] + partition['calibration']
    assert len(train_ids) == 229 and not set(train_ids) & set(partition['test'])
    records = {r['sample_id']: r for group in plan['records']['icdar13'].values() for r in group}
    archived = read(prior / 'branches/icdar13__yolo.json')
    branches, tie_keys = archived['branches'], archived['tie_keys']
    with np.load(prior / 'models/icdar13__yolo__features.npz', allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    positions = {sid: i for i, sid in enumerate(data['ids'].tolist())}
    def matrix(ids, groups):
        return np.stack([np.concatenate([data[group][positions[sid]] for group in groups]) for sid in ids])
    def scale(ids):
        return np.asarray([data['scale'][positions[sid]] for sid in ids])
    target = config['target_f1']
    def utility(ids):
        return np.asarray([2 * (branches[sid]['light']['tp'] - branches[sid]['deepsolo']['tp'])
            - target * (branches[sid]['light']['detected_words'] - branches[sid]['deepsolo']['detected_words'])
            for sid in ids]) / scale(ids)
    def chosen(scores, ids, fallback):
        return sorted(ids, key=lambda sid: (-scores[sid], tie_keys[sid], sid))[:len(ids)-fallback]
    if args.stage == 'train':
        cv = GroupKFold(n_splits=config['cv_folds'])
        groups = [records[sid]['rgb_sha256'] for sid in train_ids]
        folds = [(list(train_idx), list(val_idx)) for train_idx, val_idx in cv.split(train_ids, groups=groups)]
        trials = []
        for gi, feature_groups in enumerate(config['feature_groups']):
            for ri, recipe in enumerate(config['recipes']):
                oof = {}
                for fold, (train_idx, held_idx) in enumerate(folds):
                    fit_ids = [train_ids[i] for i in train_idx]
                    held_ids = [train_ids[i] for i in held_idx]
                    assert not set(fit_ids) & set(held_ids)
                    assert not {records[sid]['rgb_sha256'] for sid in fit_ids} & {
                        records[sid]['rgb_sha256'] for sid in held_ids}
                    x = matrix(fit_ids, feature_groups)
                    weight = scale(fit_ids)
                    model = fit(x, utility(fit_ids), weight, recipe, config['seed'])
                    model['groups'] = feature_groups
                    values = predict(model, matrix(held_ids, feature_groups), scale(held_ids))
                    oof.update(zip(held_ids, map(float, values)))
                assert set(oof) == set(train_ids)
                fallback = round(len(train_ids) * config['fallback_images'] / config['test_images'])
                outcome = replay({sid: branches[sid] for sid in train_ids},
                    chosen(oof, train_ids, fallback))
                trials.append(dict(feature_groups=feature_groups, recipe=recipe,
                    oof_f1=outcome['f1'], oof_scores=oof, oof_route=outcome, index=[gi, ri]))
                print(gi, ri, 'OOF F1', outcome['f1'], flush=True)
        winner = max(trials, key=lambda row: (row['oof_f1'], -row['index'][0], -row['index'][1]))
        feature_groups, recipe = winner['feature_groups'], winner['recipe']
        model = fit(matrix(train_ids, feature_groups), utility(train_ids), scale(train_ids),
                    recipe, config['seed'])
        model.update(groups=feature_groups, fit_ids=train_ids, target_f1=target)
        path = root / 'final_model.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        assert not path.exists()
        joblib.dump(model, path)
        seal(root / 'trials.json', dict(trials=trials, folds=[
            dict(train_ids=[train_ids[i] for i in train_idx],
                 heldout_ids=[train_ids[i] for i in held_idx])
            for train_idx, held_idx in folds], source_feature_sha256=sha(
                prior / 'models/icdar13__yolo__features.npz')))
        seal(root / 'frozen.json', dict(status='FROZEN', selected=winner['index'],
            selected_oof_f1=winner['oof_f1'], selected_oof_route=winner['oof_route'],
            final_model_sha256=sha(path), train_ids=train_ids,
            test_used_for_selection=False, source_router_frozen_sha256=sha(prior / 'frozen.json'),
            config_sha256=sha(args.config)))
        print('FROZEN', winner['index'], winner['oof_f1'], flush=True)
        return
    frozen = read(root / 'frozen.json')
    trials = read(root / 'trials.json')
    assert frozen['status'] == 'FROZEN' and not frozen['test_used_for_selection']
    assert trials['source_feature_sha256'] == sha(prior / 'models/icdar13__yolo__features.npz')
    path = root / 'final_model.joblib'
    assert frozen['final_model_sha256'] == sha(path)
    model = joblib.load(path)
    assert model['fit_ids'] == train_ids
    ids = partition['test']
    scores = dict(zip(ids, map(float, predict(model, matrix(ids, model['groups']), scale(ids)))))
    kept = chosen(scores, ids, config['fallback_images'])
    population = {sid: branches[sid] for sid in ids}
    result = replay(population, kept)
    oracle = analyze_budget(population, ids, config['fallback_images'], str(target))
    seal(root / 'test_scores.json', dict(scores=scores, branches=population,
        tie_keys={sid: tie_keys[sid] for sid in ids}))
    seal(root / 'results.json', dict(status='PASS', mode='approximate',
        claim_label='approximate reproduction', primary=result,
        all_light=replay(population, ids), all_deepsolo=replay(population, []),
        oracle=oracle, target_met=result['f1'] >= target,
        actual_new_neural_inferences=0, test_images=len(ids),
        scope='CV model choice on TRAIN229, full TRAIN229 final fit; logical cached exact61 DeepSolo images'))
    print('TEST F1', 100 * result['f1'], 'DeepSolo', result['deepsolo_images'],
          '/', result['total_images'], flush=True)


if __name__ == '__main__':
    main()
