"""Independent population, leakage, likelihood and rational-count replay audit."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import math

import numpy as np
import torch

from scripts.audit_yolo_f1_quality_study import aggregate
from scripts.run_yolo_f1_quality_study import read, seal, sha


def counts(branches, ids, accepted):
    selected = set(accepted)
    assert len(selected) == len(accepted) and selected <= set(ids)
    return aggregate(branches[s]['light' if s in selected else 'deepsolo'] for s in ids)


def audit(root):
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']
    frozen = read(root/'frozen_choices.json'); results = read(root/'results.json')
    ph = sha(root/'resolved_configuration.json'); fh = sha(root/'frozen_choices.json')
    assert results['status'] == 'SUCCESS' and results['plan_sha256'] == frozen['plan_sha256'] == ph
    assert results['frozen_choices_sha256'] == fh and not frozen['test_used_for_selection']
    assert frozen['test_features_present_at_freeze'] == 0 and not results['base_word_outputs_changed']
    assert plan['reproduction_mode'] == 'approximate' and not plan['table_5_claim']
    samples = {k: {s['sample_id']: s for s in v} for k, v in plan['samples'].items()}
    assert not set(samples['train'])&set(samples['test'])
    assert not {s['rgb_sha256'] for s in samples['train'].values()}&{s['rgb_sha256'] for s in samples['test'].values()}
    for file, digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/file) == digest
    for file, digest in plan['asset_sha256'].items():
        assert sha(file) == digest
    feature_fields = {'patches', 'patch_geometry', 'patch_valid', 'patch_mass', 'word_text',
                      'word_image', 'queries', 'likelihood', 'scalars', 'word_geometry'}
    branches = {}; eligible = {}; probes = 0
    for split, ss in samples.items():
        branches[split] = read(root/split/'branch_metrics.json')['branches']
        assert set(branches[split]) == set(ss)
        eligible[split] = {}
        for stage in ('clip', 'blip'):
            receipt = read(root/split/(stage+'_receipt.json')); model = receipt['model']
            assert receipt['images'] == len(ss) and receipt['plan_sha256'] == ph
            assert model['model_state_sha256_before'] == model['model_state_sha256_after']
            assert model['native_forward_check']['status'] == 'PASS' and model['optimizer_steps'] == 0
            assert model['precision'] == 'float32'
            for file, digest in receipt['files_sha256'].items():
                assert sha(root/split/stage/file) == digest
        for sid, sample in ss.items():
            stem = sid.replace(':', '__'); meta = read(root/split/'features'/(stem+'.json'))
            assert meta['contract'] == dict(plan_sha256=ph, sample_id=sid, image_sha256=sample['image_sha256'])
            assert not meta['ground_truth_used'] and not meta['actual_fallback_output_used']
            assert meta['frozen_choices_sha256'] == (fh if split == 'test' else None)
            path = root/split/'features'/(stem+'.npz'); assert sha(path) == meta['arrays_sha256']
            with np.load(path, allow_pickle=False) as values:
                assert set(values.files) == feature_fields and all(np.isfinite(values[k]).all() for k in values.files)
                rows_path = root/split/'light'/(stem+'.json'); rows = read(rows_path)['rows']
                assert sha(rows_path) == meta['rows_sha256']
                assert len(rows) == len(values['word_text']) == len(values['word_geometry'])
            eligible[split][sid] = bool(rows) and all(r['raw_text'].strip() for r in rows)
            assert meta['eligible'] == eligible[split][sid]
            for stage in ('clip', 'blip'):
                raw = read(root/split/stage/(stem+'.json'))
                assert sha(root/split/stage/(stem+'.npz')) == raw['arrays_sha256']
                assert raw['rows_sha256'] == sha(rows_path) and not raw['ground_truth_used']
            blip = read(root/split/'blip'/(stem+'.json'))['raw']
            for probe in blip['global_scores']+blip['word_scores']:
                for scope in ('original', 'erased'):
                    score = probe[scope]; lp = score['token_logprobs']
                    assert len(lp) == len(score['target_ids']) and len(lp) > 0
                    assert math.isclose(sum(lp), score['sum_logprob'], rel_tol=1e-5, abs_tol=1e-4)
                    assert math.isclose(sum(lp)/len(lp), score['mean_logprob'], rel_tol=1e-5, abs_tol=1e-5)
                    assert score['eos_logprob'] == lp[-1]
                assert math.isclose(probe['visual_gain'], probe['original']['mean_logprob']-probe['erased']['mean_logprob'])
                probes += 1
    cv = read(root/'cv_predictions.json'); assert sha(root/'cv_predictions.json') == frozen['cv_predictions_sha256']
    full_ids = set(samples['train']); legacy_ids = set(plan['legacy_fit_ids']); selections = []
    for population, members in [('full', full_ids), ('legacy80', legacy_ids)]:
        traces = [t for t in frozen['fold_trace'] if t['population'] == population]
        seen = Counter()
        for trace in traces:
            fit, held = set(trace['training_ids']), set(trace['held_ids'])
            assert fit|held == members and not fit&held
            assert not set(trace['training_rgb_sha256'])&set(trace['held_rgb_sha256'])
            seen.update(held)
            if population == 'full':
                checkpoint = torch.load(root/'models'/(population+str(trace['fold'])+'__support.pt'), weights_only=True, map_location='cpu')
                assert set(checkpoint['training_ids']) == fit
            for arm, epochs in cv[population].items():
                for epoch in epochs:
                    record = frozen['cv_models'][population+str(trace['fold'])+'__'+arm+'__'+epoch]
                    path = root/record['model_path']; assert sha(path) == record['model_sha256']
                    model = torch.load(path, weights_only=True, map_location='cpu')
                    assert model['plan_sha256'] == ph and set(model['preprocessing']['training_ids']) == fit
        assert set(seen) == members and all(n == 1 for n in seen.values())
        for arm_index, (arm, epochs) in enumerate(cv[population].items()):
            oof = {}
            for epoch, scores in epochs.items():
                assert set(scores) == members
                accepted = sorted((s for s in members if scores[s] is not None), key=lambda s: (-scores[s], s))[:round(len(members)*cfg['primary_bypass_fraction'])]
                oof[epoch] = counts(branches['train'], members, accepted)['f1']
            best = max(oof, key=lambda e: (oof[e], -int(e)))
            name = arm if population == 'full' else 'scalar_legacy80'
            choice = frozen['arms'][name]; assert choice['epoch'] == int(best)
            final = root/choice['model_path']; assert sha(final) == choice['model_sha256']
            model = torch.load(final, weights_only=True, map_location='cpu')
            assert set(model['preprocessing']['training_ids']) == members
            if population == 'full':
                selections.append((oof[best], -cfg['arms'].index(arm), arm))
    assert max(selections)[2] == frozen['selected_arm']
    support = torch.load(root/'models/final__support.pt', weights_only=True, map_location='cpu')
    assert set(support['training_ids']) == full_ids
    for split in samples:
        ids = sorted(samples[split]); report = results['splits'][split]
        assert report['images'] == len(ids)
        for arm, points in report['arms'].items():
            prediction = read(root/'predictions'/(split+'__'+arm+'__evaluated.json'))
            assert set(prediction['scores']) == set(ids)
            for q, p in points.items():
                accepted = sorted((s for s in ids if eligible[split][s]), key=lambda s: (-prediction['scores'][s], s))[:round(len(ids)*float(q))]
                assert set(accepted) == set(p['accepted_ids'])
                assert counts(branches[split], ids, accepted) == p['combined']
                assert p['accepted_images']+p['fallback_images'] == len(ids)
                assert math.isclose(p['oracle_regret_pp'], 100*(report['oracles'][q]['f1']-p['combined']['f1']), abs_tol=1e-10)
    result = dict(status='PASS', plan_sha256=ph, results_sha256=sha(root/'results.json'),
                  images={k: len(v) for k, v in samples.items()}, likelihood_contrasts_checked=probes,
                  scope='Source/asset bytes, native equivalence receipts, complete words and populations, no GT/fallback features, grouped fold/model fitting scopes, OOF epoch/arm selection and independent rational-count replay; no independent neural refit')
    seal(root/'independent_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(); print(audit(args.run_root))


if __name__ == '__main__':
    main()
