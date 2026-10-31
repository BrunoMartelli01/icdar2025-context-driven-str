"""Raw-pixel replay, independent estimator refits and count audit for shadow routing."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from localize.external_text_detectors import crop_region
from recognition.shadow_inventory_router import inventory_features
from scripts.run_shadow_inventory_router import image
from scripts.run_yolo_f1_quality_study import read, seal, sha
from scripts.crop_clip_runtime import array_sha
from scripts.analyze_trainable_image_router import independent_fit, optimum
from scripts.audit_yolo_f1_quality_study import check_point, point, aggregate, f1


def close_scores(actual, expected):
    assert set(actual) == set(expected)
    for sid, value in expected.items():
        assert actual[sid] is None if value is None else math.isclose(actual[sid], value, rel_tol=1e-10, abs_tol=1e-10), sid


def audit(root):
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']; result = read(root/'results.json')
    frozen = read(root/'frozen_choices.json'); cv = read(root/'cv_results.json'); ph = sha(root/'resolved_configuration.json')
    assert result['status'] == 'SUCCESS' and result['plan_sha256'] == frozen['plan_sha256'] == ph
    assert result['frozen_choices_sha256'] == sha(root/'frozen_choices.json')
    assert frozen['cv_results_sha256'] == sha(root/'cv_results.json')
    assert not frozen['test_labels_used'] and frozen['test_predictions_present_at_freeze'] == frozen['test_features_present_at_freeze'] == 0
    for name, digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest
    for path, digest in plan['input_sha256'].items():
        assert sha(path) == digest
    source = Path(plan['paths']['source_root']); coverage = Path(plan['source_coverage_root'])
    branches = read(coverage/'results.json')['resolutions']['640']['branch_metrics']
    features = {}; crop_count = region_count = 0
    for split, samples in plan['samples'].items():
        features[split] = {}
        for sample in samples:
            sid = sample['sample_id']; key = sid.replace(':', '__')+'.json'
            v = read(root/split/'features'/key); dpath, opath = root/split/'detector'/key, root/split/'ocr'/key
            det, ocr = read(dpath), read(opath); old = read(source/split/'features'/key)
            words = read(coverage/'640'/split/'features'/key)['rows']; rgb = image(sample)
            binding = sha(root/'frozen_choices.json') if split == 'test' else None
            for record in (v, det, ocr):
                assert record['plan_sha256'] == ph and record['image_sha256'] == sample['image_sha256']
                assert record['frozen_choices_sha256'] == binding
            assert not det['ground_truth_used'] and not ocr['ground_truth_used']
            assert ocr['detector_sha256'] == sha(dpath)
            assert v['detector_sha256'] == sha(dpath) and v['ocr_sha256'] == sha(opath)
            values, rules, evidence = inventory_features(words, det['regions'], ocr['rows'], cfg['inventory'])
            assert v['values'] == {**{n: old['values'][n] for n in plan['feature_schemas']['core']}, **values}
            assert v['evidence'] == evidence and v['rules'] == rules and v['controls'] == old['controls']
            assert v['eligible'] == old['eligible'] == (bool(words) and all(r['raw_text'].strip() for r in words))
            assert not v['per_image_ground_truth_used'] and not v['actual_deepsolo_outputs_used']
            assert set(det['probe_region_ids']) == set(evidence['probe_region_ids'])
            assert len(det['probe_region_ids']) <= cfg['inventory']['maximum_probes']
            # Independently rasterize original integer boxes to check rectangle
            # union coverage rather than repeating its sweep-line algorithm.
            mask = np.zeros(rgb.shape[:2], bool)
            for r in words:
                a, b, c, d = r['bbox']; mask[b:d, a:c] = True
            valid = [r for r in det['regions'] if r['valid']]
            for r, value in zip(valid, evidence['region_coverage']):
                a, b, c, d = r['bbox']
                assert math.isclose(float(mask[b:d, a:c].mean()), value, abs_tol=1e-12)
            by_index = {r['source_index']: r for r in valid}; region_count += len(valid)
            for row in ocr['rows']:
                crop = crop_region(rgb, by_index[row['source_index']], row['view'])
                assert row['crop_rgb_sha256'] == array_sha(crop) and row['crop_shape'] == list(crop.shape)
                crop_count += 1
            features[split][sid] = v
        for stage in ('detector', 'ocr'):
            receipt = read(root/split/(stage+'_receipt.json'))
            assert receipt['plan_sha256'] == ph and receipt['images'] == len(samples)
            assert receipt['model_state_sha256_before'] == receipt['model_state_sha256_after']
            assert receipt['optimizer_steps'] == 0 and receipt['precision'] == 'float32'
            assert all(receipt['cuda_peak'][k] > 0 for k in ('peak_allocated_bytes', 'peak_reserved_bytes'))
    fit, val = plan['fit_ids'], plan['validation_ids']
    assert not set(fit)&set(val) and set(fit+val) == set(features['train'])
    assert not set(features['train'])&set(features['test'])
    cache = {}; refits = held_count = 0
    def replay(ids, names, recipe):
        nonlocal refits
        assert set(ids) <= set(fit)
        key = (tuple(ids), tuple(names), str(recipe))
        if key not in cache:
            x = np.array([[features['train'][s]['values'][n] for n in names] for s in ids])
            y = np.array([[branches['train'][s]['light']['tp']-branches['train'][s]['deepsolo']['tp'],
                branches['train'][s]['light']['detected_words']-branches['train'][s]['deepsolo']['detected_words']] for s in ids])
            cache[key] = independent_fit(x, y, recipe, cfg); refits += 1
        return cache[key]
    def predictions(model, split, ids):
        kind = model['recipe']['kind']
        if kind in ('rule', 'control'):
            return {s: features[split][s]['rules' if kind == 'rule' else 'controls'][model['recipe']['field']]
                    if features[split][s]['eligible'] else None for s in ids}
        estimator, mean, scale = replay(model['training_ids'], model['feature_names'], model['recipe'])
        x = np.array([[features[split][s]['values'][n] for n in model['feature_names']] for s in ids])
        p = estimator.predict((x-mean)/scale)
        reference = float(f1(aggregate(branches['train'][s]['deepsolo'] for s in model['training_ids'])))
        if 'reference_f1' in model:
            assert model['reference_f1'] == reference
        return {s: float(2*p[i, 0]-reference*p[i, 1]) if features[split][s]['eligible'] else None for i, s in enumerate(ids)}
    def ranked(scores, ids, fraction):
        return sorted((s for s in ids if scores[s] is not None), key=lambda s: (-scores[s], s))[:round(len(ids)*fraction)]
    for budget, families in cv.items():
        for family, data in families.items():
            names = plan['feature_schemas'][family.split('_', 1)[1]]
            for trial in data['trials']:
                expected = {}; visited = []
                for fold in trial['fold_trace']:
                    training, held = fold['training_ids'], fold['held_out_ids']
                    assert not set(training)&set(held) and set(training+held) == set(fit)
                    assert not {features['train'][s]['image_sha256'] for s in training}&{features['train'][s]['image_sha256'] for s in held}
                    m = {'recipe': trial['recipe'], 'training_ids': training, 'feature_names': names}
                    expected.update(predictions(m, 'train', held)); visited += held; held_count += len(held)
                assert sorted(visited) == sorted(fit)
                close_scores(trial['oof_scores'], expected)
                accepted = ranked(expected, fit, float(budget)); assert accepted == trial['accepted_ids']
                assert float(f1(point(branches['train'], fit, accepted, cfg['quality_f1_threshold'])['combined'])) == trial['oof_f1']
            assert data['selected_trial'] == max(range(len(data['trials'])), key=lambda i: (data['trials'][i]['oof_f1'], -i))
    checked_points = 0
    populations = {'fit': ('train', fit), 'validation': ('train', val), 'train': ('train', sorted(features['train'])),
                   'test': ('test', sorted(features['test']))}
    for budget, choices in frozen['budgets'].items():
        validation_trials = []
        for index, family in enumerate(plan['families']):
            path = root/'models'/(budget+'__'+family+'.json'); model = read(path)
            assert frozen['models_sha256'][str(path.relative_to(root))] == sha(path)
            if model['recipe']['kind'] not in ('rule', 'control'):
                data = cv[budget][family]; assert model['recipe'] == data['trials'][data['selected_trial']]['recipe']
                assert model['training_ids'] == fit
            val_scores = predictions(model, 'train', val); accepted = ranked(val_scores, val, float(budget))
            score = f1(point(branches['train'], val, accepted, cfg['quality_f1_threshold'])['combined'])
            assert float(score) == choices['validation_f1'][family]
            assert choices['scalar_cuts'][family] == min((val_scores[s] for s in accepted), default=None)
            validation_trials.append((score, -index, family))
            for pop, (split, ids) in populations.items():
                scores = predictions(model, split, ids)
                close_scores(read(root/'predictions'/(pop+'.json'))[budget][family], scores)
                accepted = ranked(scores, ids, float(budget)); out = result['splits'][pop]['budgets'][budget]
                check_point(out['families'][family], point(branches[split], ids, accepted, cfg['quality_f1_threshold'])); checked_points += 1
                cut = choices['scalar_cuts'][family]
                keep = [s for s in ids if cut is not None and scores[s] is not None and scores[s] >= cut]
                check_point(out['scalar_thresholds'][family], point(branches[split], ids, keep, cfg['quality_f1_threshold'])); checked_points += 1
        assert choices['selected_family'] == max(validation_trials)[2]
        for pop, (split, ids) in populations.items():
            out = result['splits'][pop]['budgets'][budget]
            assert out['selected'] == out['families'][choices['selected_family']]
            expected = optimum(branches[split], ids, [s for s in ids if features[split][s]['eligible']], round(len(ids)*float(budget)))
            assert out['oracle'] == expected
    audit = {'status': 'PASS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'plan_sha256': ph, 'results_sha256': sha(root/'results.json'), 'audit_code_sha256': sha(Path(__file__)),
        'images': sum(len(v) for v in features.values()), 'valid_regions': region_count, 'original_crop_hashes_checked': crop_count,
        'independent_refits': refits, 'held_out_predictions_verified': held_count, 'routing_points_verified': checked_points,
        'pixel_union_coverage_verified': True, 'frozen_neural_states_unchanged': True}
    seal(root/'independent_audit.json', audit)
    print(audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
