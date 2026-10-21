"""Paired whole-image router evaluation; GT appears only in these diagnostics."""
from __future__ import annotations

import numpy as np


def aggregate(rows):
    rows = list(rows)
    counts = {k: sum(r[k] for r in rows) for k in
              ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs')}
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    if counts['detected_words'] != tp + fp or counts['gt_words'] != tp + fn:
        raise ValueError('Inconsistent word denominators')
    return {**counts, 'images': len(rows), 'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'accuracy_tn0': tp/(tp+fp+fn) if tp+fp+fn else None,
            'safe_images': sum(r['detected_words'] > 0 and r['fp'] == 0 for r in rows),
            'complete_images': sum(r['detected_words'] > 0 and r['fp'] == r['fn'] == 0 for r in rows)}


def route_point(branches, accepted, sources=None):
    """Replace the entire image; never mix YOLO and fallback words."""
    accepted = set(accepted)
    if not accepted <= branches.keys():
        raise ValueError('Unknown accepted image')
    sources = sources or {}
    selected = {sid: sources.get(sid, 'yolo') if sid in accepted else 'deepsolo' for sid in branches}
    metrics = {sid: branches[sid][source] for sid, source in selected.items()}
    baseline = aggregate(r['yolo'] for r in branches.values())
    combined = aggregate(metrics.values())
    return {'accepted_images': len(accepted), 'fallback_images': len(branches)-len(accepted),
            'accepted_ids': sorted(accepted), 'sources': selected,
            'accepted_metrics': aggregate(metrics[sid] for sid in sorted(accepted)),
            'combined': combined, 'delta_tn0_vs_yolo': combined['accuracy_tn0']-baseline['accuracy_tn0'],
            'delta_f1_vs_yolo': combined['f1']-baseline['f1'],
            'changed_images': sum(source != 'yolo' for source in selected.values()),
            'correct_words_gained_vs_yolo': sum(max(0, metrics[sid]['tp']-branches[sid]['yolo']['tp']) for sid in branches),
            'correct_words_lost_vs_yolo': sum(max(0, branches[sid]['yolo']['tp']-metrics[sid]['tp']) for sid in branches)}


def oracle(branches, budget=None):
    """Exact micro-F1/TN0 upper bound (Dinkelbach), GT-only, never a router.

    Both metrics are monotone in TP/(GT+predictions). For a fixed ratio,
    maximize TP-ratio*predictions independently, or keep exactly budget YOLO
    images by ranking that branch advantage. Equality ties prefer YOLO.
    """
    ids = sorted(branches)
    if not ids or (budget is not None and not 0 <= budget <= len(ids)):
        raise ValueError('Invalid oracle population/budget')
    if any(r['yolo']['gt_words'] != r['deepsolo']['gt_words'] for r in branches.values()):
        raise ValueError('Branches must share every care GT word')
    ratio = 0.
    for _ in range(1000):
        advantages = {sid: (branches[sid]['yolo']['tp']-branches[sid]['deepsolo']['tp']) - ratio*(
            branches[sid]['yolo']['detected_words']-branches[sid]['deepsolo']['detected_words']) for sid in ids}
        accepted = ({sid for sid in ids if advantages[sid] >= 0} if budget is None else
                    set(sorted(ids, key=lambda sid: (-advantages[sid], sid))[:budget]))
        total = aggregate(branches[sid]['yolo' if sid in accepted else 'deepsolo'] for sid in ids)
        new_ratio = total['tp']/(total['gt_words']+total['detected_words'])
        if abs(new_ratio-ratio) < 1e-14:
            return {**route_point(branches, accepted), 'diagnostic_only': True,
                    'uses_ground_truth': True, 'optimal_for': 'micro F1 and TN0', 'budget': budget}
        ratio = new_ratio
    raise RuntimeError('Oracle fractional optimizer did not converge')


def paired_bootstrap(branches, left, right, iterations, seed):
    """Conditional interval: fixed trained gates, paired resampling of images."""
    ids = sorted(branches)
    arrays = []
    for point in (left, right):
        arrays.append(np.array([[branches[sid][point['sources'][sid]][k] for k in ('tp','fp','fn')]
                                for sid in ids], dtype=np.float64))
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(iterations):
        sample = rng.integers(0, len(ids), size=len(ids))
        counts = [a[sample].sum(axis=0) for a in arrays]
        delta.append(counts[0][0]/sum(counts[0])-counts[1][0]/sum(counts[1]))
    return {'metric': 'accuracy_tn0', 'estimate': left['combined']['accuracy_tn0']-right['combined']['accuracy_tn0'],
            'ci95': np.quantile(delta, [.025,.975]).tolist(), 'iterations': iterations,
            'seed': seed, 'unit': 'paired image; gates fixed; no multiplicity correction'}
