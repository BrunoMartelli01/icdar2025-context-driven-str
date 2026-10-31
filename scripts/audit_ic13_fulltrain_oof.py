"""Check image-group OOF isolation, model choice, and exact TEST route."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from recognition.lexical_budget_router import fit, predict
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


FIELDS = ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs')


def route(branches, kept, result):
    kept = set(kept)
    for field in FIELDS:
        assert sum(branches[sid]['light' if sid in kept else 'deepsolo'][field]
                   for sid in branches) == result[field]
    assert result['deepsolo_images'] == len(branches)-len(kept)
    assert result['total_images'] == len(branches)
    assert result['deepsolo_percent'] == 100*(len(branches)-len(kept))/len(branches)
    assert result['f1'] == 2*result['tp']/(result['detected_words']+result['gt_words'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['study_root'])
    for path, digest in config['pinned_inputs'].items():
        assert sha(path) == digest, path
    source = Path(config['source_router_root'])
    plan = read(config['source_plan'])
    partition = plan['partitions']['icdar13']
    train_ids = partition['fit'] + partition['selection'] + partition['calibration']
    test_ids = partition['test']
    assert len(train_ids) == 229 and len(test_ids) == 233
    assert not set(train_ids) & set(test_ids)
    records = {r['sample_id']: r for group in plan['records']['icdar13'].values() for r in group}
    source_branches = read(source / 'branches/icdar13__yolo.json')
    branches = source_branches['branches']
    tie_keys = source_branches['tie_keys']
    with np.load(source / 'models/icdar13__yolo__features.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    position = {sid: i for i, sid in enumerate(arrays['ids'].tolist())}
    def matrix(ids, feature_groups):
        return np.stack([np.concatenate([arrays[group][position[sid]] for group in feature_groups]) for sid in ids])
    def scales(ids):
        return np.asarray([arrays['scale'][position[sid]] for sid in ids])
    def utility(ids):
        return np.asarray([2*(branches[sid]['light']['tp']-branches[sid]['deepsolo']['tp'])
            -config['target_f1']*(branches[sid]['light']['detected_words']-branches[sid]['deepsolo']['detected_words'])
            for sid in ids])/scales(ids)
    def top(scores, ids, fallback):
        return sorted(ids, key=lambda sid: (-scores[sid], tie_keys[sid], sid))[:len(ids)-fallback]
    saved = read(root / 'trials.json')
    frozen = read(root / 'frozen.json')
    results = read(root / 'results.json')
    assert frozen['config_sha256'] == sha(args.config)
    assert not frozen['test_used_for_selection']
    assert frozen['train_ids'] == train_ids
    assert saved['source_feature_sha256'] == sha(source / 'models/icdar13__yolo__features.npz')
    groups = [records[sid]['rgb_sha256'] for sid in train_ids]
    folds = list(GroupKFold(n_splits=config['cv_folds']).split(train_ids, groups=groups))
    for (train_idx, held_idx), proof in zip(folds, saved['folds']):
        fit_ids = [train_ids[i] for i in train_idx]
        held_ids = [train_ids[i] for i in held_idx]
        assert proof['train_ids'] == fit_ids and proof['heldout_ids'] == held_ids
        assert not {records[sid]['rgb_sha256'] for sid in fit_ids} & {
            records[sid]['rgb_sha256'] for sid in held_ids}
    assert len(saved['trials']) == len(config['recipes'])*len(config['feature_groups'])
    fallback = round(len(train_ids)*config['fallback_images']/config['test_images'])
    for trial in saved['trials']:
        gi, ri = trial['index']
        assert trial['feature_groups'] == config['feature_groups'][gi]
        assert trial['recipe'] == config['recipes'][ri]
        assert set(trial['oof_scores']) == set(train_ids)
        route({sid: branches[sid] for sid in train_ids},
              top(trial['oof_scores'], train_ids, fallback), trial['oof_route'])
        assert trial['oof_route']['f1'] == trial['oof_f1']
    winner = max(saved['trials'], key=lambda row: (row['oof_f1'],
        -row['index'][0], -row['index'][1]))
    assert winner['index'] == frozen['selected']
    assert winner['oof_f1'] == frozen['selected_oof_f1']
    oof = {}
    for train_idx, held_idx in folds:
        fit_ids = [train_ids[i] for i in train_idx]
        held_ids = [train_ids[i] for i in held_idx]
        bundle = fit(matrix(fit_ids, winner['feature_groups']), utility(fit_ids),
            scales(fit_ids), winner['recipe'], config['seed'])
        values = predict(bundle, matrix(held_ids, winner['feature_groups']), scales(held_ids))
        oof.update(zip(held_ids, map(float, values)))
    assert oof == winner['oof_scores']
    expected = fit(matrix(train_ids, winner['feature_groups']), utility(train_ids),
        scales(train_ids), winner['recipe'], config['seed'])
    path = root / 'final_model.joblib'
    assert sha(path) == frozen['final_model_sha256']
    model = joblib.load(path)
    assert model['fit_ids'] == train_ids and model['groups'] == winner['feature_groups']
    assert np.array_equal(expected['model'].predict(expected['scaler'].transform(matrix(test_ids, model['groups']))),
        model['model'].predict(model['scaler'].transform(matrix(test_ids, model['groups']))))
    test_scores = read(root / 'test_scores.json')
    predicted = predict(model, matrix(test_ids, model['groups']), scales(test_ids))
    assert dict(zip(test_ids, map(float, predicted))) == test_scores['scores']
    selected = top(test_scores['scores'], test_ids, config['fallback_images'])
    route(test_scores['branches'], selected, results['primary'])
    assert results['primary']['deepsolo_images'] == 61
    seal(root / 'independent_audit.json', dict(status='PASS',
        train_images=len(train_ids), test_images=len(test_ids), oof_folds=len(folds),
        candidate_routes=len(saved['trials']), test_labels_used_for_fit=0,
        results_sha256=sha(root / 'results.json'), code_sha256=sha(__file__)))
    print('PASS', len(folds), 'OOF folds,', len(saved['trials']), 'trials, 61/233 TEST', flush=True)


if __name__ == '__main__':
    main()
