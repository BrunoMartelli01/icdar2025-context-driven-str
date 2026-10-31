"""Audit image-disjoint OOF word risk and its archived router descriptors."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from recognition.lexical_budget_router import correct_rows
from scripts.icdar_budget_data import annotation, key, read, sha
from scripts.prepare_ic13_word_risk import aggregate_risk, word_features
from scripts.run_lexical_budget_router import lexicon
from scripts.run_yolo_f1_quality_study import seal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['study_root'])
    preparation_config = root / 'resolved_configuration.json'
    evidence = read(root / 'word_training_evidence.json')
    archive = read(root / 'word_risk_features.json')
    assert evidence['config_sha256'] == sha(preparation_config)
    assert evidence['code_sha256'] == sha('scripts/prepare_ic13_word_risk.py')
    plan = read(Path(config['original_router_root']) / 'resolved_configuration.json')
    records = {r['sample_id']: r for group in plan['records']['icdar13'].values() for r in group}
    fit_ids = plan['partitions']['icdar13']['fit']
    assert evidence['fit_ids'] == fit_ids == archive['full_fit_ids']
    assert set(evidence['dataset']) == set(evidence['probabilities']) == set(archive['values']) == set(records)
    policy = dict(enabled=True, confidence_cap=1.01, protect_numeric=False, unmatched_policy='source_reject')
    dictionaries = {role: lexicon(config, 'icdar13', role) for role in ('train', 'test')}
    def normal(text):
        return ''.join(char for char in text.casefold() if char.isalnum())
    for sid, record in records.items():
        raw_path = Path(config['word_risk']['raw_ocr_root']) / (key(sid) + '.json')
        raw = read(raw_path)
        observed = evidence['dataset'][sid]
        assert observed['source_sha256'] == sha(raw_path)
        assert not raw['ground_truth_used'] and raw['image_sha256'] == record['image_sha256']
        role = 'train' if ':train:' in sid else 'test'
        rows, _ = correct_rows(raw['rows'], dictionaries[role], policy)
        assert rows == observed['rows']
        expected_x = [word_features(row, record['image_wh']) for row in rows]
        assert np.allclose(expected_x, observed['x'], atol=0, rtol=0)
        assert len(observed['x']) == len(evidence['probabilities'][sid])
        if sid in fit_ids:
            gt, _ = annotation(record)
            truth = {normal(g['text']) for g in gt if normal(g['text'])}
            labels = [int(normal(row['raw_text']) in truth) for row in rows]
            assert observed['labels'] == labels
        else:
            assert 'labels' not in observed

    settings = config['word_risk']
    def new_model():
        return HistGradientBoostingClassifier(max_iter=settings['iterations'], max_leaf_nodes=settings['leaves'],
            min_samples_leaf=settings['minimum_leaf'], l2_regularization=settings['l2'],
            early_stopping=False, random_state=config['seed'])
    def matrix(ids):
        return np.concatenate([np.asarray(evidence['dataset'][sid]['x'], np.float64).reshape(-1, 18) for sid in ids])
    def labels(ids):
        return np.concatenate([evidence['dataset'][sid]['labels'] for sid in ids])
    def verify_predictions(model, ids):
        for sid in ids:
            rows = evidence['dataset'][sid]['x']
            expected = model.predict_proba(np.asarray(rows).reshape(-1, 18))[:, 1].tolist() if rows else []
            assert np.allclose(expected, evidence['probabilities'][sid], atol=1e-14, rtol=0)

    seen = set()
    folds = list(KFold(n_splits=settings['folds'], shuffle=True, random_state=config['seed']).split(fit_ids))
    assert len(folds) == len(archive['folds'])
    for index, ((train_idx, val_idx), saved) in enumerate(zip(folds, archive['folds'])):
        train_ids = [fit_ids[i] for i in train_idx]
        heldout_ids = [fit_ids[i] for i in val_idx]
        assert saved['train_ids'] == train_ids and saved['heldout_ids'] == heldout_ids
        assert not set(train_ids) & set(heldout_ids)
        assert not seen & set(heldout_ids)
        seen.update(heldout_ids)
        assert sha(saved['path']) == saved['sha256']
        model = new_model().fit(matrix(train_ids), labels(train_ids))
        archived = joblib.load(saved['path'])
        assert np.array_equal(model.predict_proba(matrix(heldout_ids)), archived.predict_proba(matrix(heldout_ids)))
        verify_predictions(model, heldout_ids)
    assert seen == set(fit_ids)
    assert sha(archive['full_model_path']) == archive['full_model_sha256']
    full = new_model().fit(matrix(fit_ids), labels(fit_ids))
    other = list(set(records) - set(fit_ids))
    archived_full = joblib.load(archive['full_model_path'])
    assert np.array_equal(full.predict_proba(matrix(other)), archived_full.predict_proba(matrix(other)))
    verify_predictions(full, other)
    for sid, record in records.items():
        role = 'train' if ':train:' in sid else 'test'
        light = read(Path(config['original_router_root']) / 'experts/yolo' / (key(sid) + '.json'))
        assert not light['ground_truth_used'] and light['image_sha256'] == record['image_sha256']
        selected, _ = correct_rows(light['rows'], dictionaries[role], policy)
        expected = aggregate_risk(evidence['dataset'][sid]['rows'], evidence['probabilities'][sid],
            selected, record['image_wh'], settings['duplicate_iou'])
        assert np.allclose(expected, archive['values'][sid], atol=1e-12, rtol=0)
        assert len(expected) == 12 and np.isfinite(expected).all()
    seal(root / 'word_risk_independent_audit.json', dict(status='PASS', samples=len(records),
        fit_images=len(fit_ids), oof_folds=len(folds), test_labels_used=0,
        archived_feature_sha256=sha(root / 'word_risk_features.json'),
        training_evidence_sha256=sha(root / 'word_training_evidence.json'),
        code_sha256=sha(__file__)))
    print('PASS', len(records), 'images,', len(folds), 'OOF folds', flush=True)


if __name__ == '__main__':
    main()
