"""Isolated APPROXIMATE controls for IC13 vocabulary and out-of-fold routing."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import numpy as np


def normalize(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def crossfit_folds(records, count, seed):
    """RGB-grouped deterministic folds, independent of GT and performance."""
    if count < 2 or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Unique samples and at least two folds required')
    groups = {}
    for r in records:
        groups.setdefault(r['rgb_sha256'], []).append(r['sample_id'])
    if len(groups) < count:
        raise ValueError('Fewer independent RGB groups than folds')
    ordered = sorted(groups, key=lambda rgb: hashlib.sha256(f'{seed}:{rgb}'.encode()).hexdigest())
    folds = [[] for _ in range(count)]
    for i, rgb in enumerate(ordered):
        folds[i % count].extend(sorted(groups[rgb]))
    return [sorted(f) for f in folds]


def vocabulary_rows(rows, lexicon, policy, conservative=None):
    """Prediction-only correction. Keep geometry and original PARSeq evidence."""
    if policy not in ('none', 'generic_keep', 'generic_conservative'):
        raise ValueError('Unknown explicit vocabulary policy')
    if policy == 'generic_conservative' and conservative is None:
        raise ValueError('Explicit conservative vocabulary parameters required')
    result = []
    for row in rows:
        output = dict(row)
        original = row['raw_text']
        eligible = policy != 'none'
        if policy == 'generic_conservative':
            # Fixed protection; never infer these rules from TEST outcomes.
            eligible = (len(normalize(original)) >= conservative['minimum_length']
                and not any(c.isdigit() for c in original)
                and row['sequence_confidence_native'] < conservative['protect_probability'])
        match = lexicon.nearest(original) if eligible else None
        output['uncorrected_text'] = original
        output['vocabulary_match'] = match
        if match is not None:
            output['raw_text'] = match['text']
        result.append(output)
    return result


def independent_counts(predictions, gt):
    care = [r for r in predictions if not r['ignored']]
    pp = Counter(normalize(r['text']) for r in care if normalize(r['text']))
    gg = Counter(normalize(r['text']) for r in gt if normalize(r['text']))
    tp = sum((pp & gg).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(gt)-tp, detected_words=len(care),
                gt_words=len(gt), all_outputs=len(predictions), ignored_outputs=len(predictions)-len(care))


def raw_evidence_features(raw, emitted):
    """Eight observable descriptors before filtering, including rejected words."""
    rejected = [r for r in raw if r['candidate_id'] not in {v['candidate_id'] for v in emitted}]
    ps = [r['sequence_confidence_native'] for r in raw]
    det = [r['detector_score'] for r in rejected]
    return np.asarray([math.log1p(len(raw)), math.log1p(len(rejected)),
        np.mean(ps) if ps else 0., min(ps) if ps else 0.,
        np.mean(det) if det else 0., sum(det),
        sum(len(normalize(r['raw_text'])) for r in rejected), float(not emitted)], np.float64)


def gt_crop_diagnosis(gt, raw, emitted, gt_readings, iou_threshold):
    """Fixed GT identities; privileged crop readings never enter the router."""
    from recognition.scene_text_v1_evaluation import box_iou
    if len(gt) != len(gt_readings):
        raise ValueError('Every care GT requires its own crop reading')
    results = []
    for g, reading in zip(gt, gt_readings):
        viable = [r for r in raw if box_iou(g['bbox'], r['bbox']) >= iou_threshold]
        correct = lambda rows: any(normalize(r['raw_text']) == normalize(g['text']) and bool(normalize(g['text'])) for r in rows)
        emitted_viable = [r for r in emitted if box_iou(g['bbox'], r['bbox']) >= iou_threshold]
        group = ('correct_emitted_crop' if correct(emitted_viable) else
                 'correct_crop_rejected_by_filter' if correct(viable) else
                 'no_viable_crop' if not viable else 'viable_crop_wrong_reading')
        results.append(dict(gt_text=g['text'], gt_bbox=g['bbox'], group=group,
            gt_crop_text=reading['raw_text'], gt_crop_correct=bool(normalize(g['text'])) and normalize(reading['raw_text']) == normalize(g['text'])))
    return results
