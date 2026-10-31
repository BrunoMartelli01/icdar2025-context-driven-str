"""Audited inventory-router tables and fixed-decision paired bootstrap intervals."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.run_yolo_f1_quality_study import read, sha, seal


def report(args):
    root = args.root; boundary = args.boundary_root
    p, r = read(root/'resolved_configuration.json'), read(root/'results.json')
    b = read(boundary/'results.json')
    for source in (root, boundary):
        audit = read(source/'independent_audit.json')
        if audit['status'] != 'PASS' or audit['results_sha256'] != sha(source/'results.json'):
            raise ValueError('Current complete audit required')
    branches = read(Path(p['source_coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']['test']
    ids = sorted(s['sample_id'] for s in p['samples']['test']); counts = {}
    for name in ('light', 'deepsolo'):
        counts[name] = np.array([[branches[s][name][k] for k in ('tp', 'fp', 'fn')] for s in ids], dtype=np.int64)
    if args.bootstrap_replicates < 1:
        raise ValueError('Explicit positive bootstrap replicate count required')
    rng = np.random.default_rng(args.seed); indices = rng.integers(0, len(ids), (args.bootstrap_replicates, len(ids)), dtype=np.int32)
    def vector(point):
        keep = set(point['accepted_ids']); return np.array([counts['light' if s in keep else 'deepsolo'][i] for i, s in enumerate(ids)])
    def bootstrap(point):
        total = vector(point)[indices].sum(axis=1)
        return 2*total[:, 0]/(2*total[:, 0]+total[:, 1]+total[:, 2])
    references = {'0.2': 'control_base_proxy_f1', '0.3': 'control_risk_visual_consensus', '0.5': 'control_risk_visual_mass'}
    rows = []; intervals = {}
    for budget, out in r['splits']['test']['budgets'].items():
        ref_name = references[budget]; ref = out['families'][ref_name]; boot_ref = bootstrap(ref); intervals[budget] = {}
        for family, point in out['families'].items():
            delta = bootstrap(point)-boot_ref
            intervals[budget][family] = {'reference_family': ref_name,
                'delta_f1_pp': 100*(point['combined']['f1']-ref['combined']['f1']),
                'paired_ci95_pp': (np.quantile(delta, [.025, .975])*100).tolist()}
            train = r['splits']['train']['budgets'][budget]['families'][family]; c = point['combined']
            rows.append({'budget': budget, 'family': family, 'train_images': r['splits']['train']['images'],
                'test_images': len(ids), 'accepted_test_images': point['accepted_images'],
                'train_f1': train['combined']['f1'], 'test_f1': c['f1'], 'test_accuracy_tn0': c['accuracy_tn0'],
                'tp': c['tp'], 'fp': c['fp'], 'fn': c['fn'], 'predicted_care_words': c['detected_words'],
                'all_outputs': c['all_outputs'], 'ignored_outputs': c['ignored_outputs'],
                'accepted_word_f1': point['accepted_words']['f1'], 'good80_accepted': point['good_images'],
                'selected_by_validation': family == out['selected_family']})
        for family, point in b['splits']['test'][budget]['families'].items():
            delta = bootstrap(point)-boot_ref
            intervals[budget]['boundary_'+family] = {'reference_family': ref_name,
                'delta_f1_pp': 100*(point['combined']['f1']-ref['combined']['f1']),
                'paired_ci95_pp': (np.quantile(delta, [.025, .975])*100).tolist()}
    receipts = {split: {stage: read(root/split/(stage+'_receipt.json')) for stage in ('detector', 'ocr')} for split in ('train', 'test')}
    summary = {'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction', 'table_5_claim': False,
        'source_results_sha256': sha(root/'results.json'), 'boundary_results_sha256': sha(boundary/'results.json'),
        'report_code_sha256': sha(Path(__file__)), 'bootstrap_seed': args.seed, 'bootstrap_replicates': args.bootstrap_replicates,
        'interval_scope': 'paired image bootstrap with routing decisions held fixed; exploratory, no multiplicity correction or refitting',
        'references': references, 'intervals': intervals, 'rows': rows, 'stage_receipts': receipts,
        'limitations': ['historically explored TEST', 'reused TRAIN80/VAL45', 'nonofficial nonspatial casefold multiset F1',
            'paper Table5 accuracy protocol unresolved', 'all CRAFT OCR measured for research; probe rule needs at most six additional crops',
            'bypass fraction does not establish wall-clock savings', 'boundary follow-up designed after unrestricted TEST summaries']}
    seal(root/'analysis_summary.json', summary)
    path = root/'all_router_points.csv'
    if not path.exists():
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Shadow inventory routers: approximate reproduction', '',
        'Frozen YOLO640/PARSeq and ViTAEv2-S/TextOCR fallback outputs. TRAIN125 (fit80 / VAL45), TEST300 / 2,204 care words.', '',
        '| Router | TRAIN F1 at 50% | TEST F1 at 20% | TEST F1 at 30% | TEST F1 at 50% | TEST TN=0 accuracy at 50% |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for family in p['families']:
        values = [r['splits']['test']['budgets'][q]['families'][family]['combined']['f1']*100 for q in references]
        train = r['splits']['train']['budgets']['0.5']['families'][family]['combined']['f1']*100
        acc = r['splits']['test']['budgets']['0.5']['families'][family]['combined']['accuracy_tn0']*100
        lines.append(f'| {family} | {train:.3f}% | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% | {acc:.3f}% |')
    lines += ['', 'VAL-selected families: '+', '.join(q+': '+r['splits']['test']['budgets'][q]['selected_family'] for q in references)+'.', '',
        'Bounded follow-up selects geometry at20%, unchanged baseline at30% and50%.', '',
        'The full-TEST agreement boundary variant reaches78.519% at50%, but VAL does not select it. No reliable new gain.', '',
        'These are local word F1 scores, not the unspecified accuracy protocol of the paper. Half the images bypassing DeepSolo is not a measured speedup.', '']
    (root/'comparison.md').write_text('\n'.join(lines))
    print({'status': 'SUCCESS', 'root': str(root), 'analysis_sha256': sha(root/'analysis_summary.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--boundary-root', type=Path, required=True); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, required=True)
    report(parser.parse_args())
