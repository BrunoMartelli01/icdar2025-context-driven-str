"""Explicit approximate U-Net crop and TRAIN-threshold routing boundaries."""
from __future__ import annotations

import math
import numpy as np
import torch

from recognition.deepsolo_word_f1 import normalize_word, box_iou
from recognition.group_word_selection import reading_confidence
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import expand_word_kernels, instance_proposals
from segmentation.word_slot_data import canvas_to_original


def unet_proposals(probabilities, image_wh, settings):
    p = np.asarray(probabilities)
    if p.shape != (2, settings['native_size'], settings['native_size']) or not np.isfinite(p).all():
        raise ValueError('Qualified two-channel native U-Net maps required')
    canvas = np.asarray(letterbox_canvas_mask(tuple(image_wh), native_size=settings['native_size']), bool)
    labels, audit = expand_word_kernels((p[0] >= settings['text_threshold'])&canvas,
                                       (p[1] >= settings['kernel_threshold'])&canvas)
    candidates = instance_proposals(labels)[:settings['maximum_ocr_candidates']]
    native = torch.tensor([v['bbox_native'] for v in candidates], dtype=torch.float32).reshape(-1, 4)
    boxes = canvas_to_original(native/settings['native_size'], image_wh).tolist()
    result = []
    for item, box in zip(candidates, boxes):
        a, b, c, d = box
        crop = [max(0, math.floor(a)), max(0, math.floor(b)), min(image_wh[0], max(math.ceil(c), math.floor(a)+1)),
                min(image_wh[1], max(math.ceil(d), math.floor(b)+1))]
        if not 0 <= crop[0] < crop[2] <= image_wh[0] or not 0 <= crop[1] < crop[3] <= image_wh[1]:
            raise ValueError('Invalid U-Net original-frame crop')
        result.append(dict(item, bbox=crop, predicted_bbox_float=box,
            detector_score=float(p[0][labels == item['instance_id']].mean())))
    return result, labels, audit


def selected_rows(candidates, settings):
    result = []
    for row in candidates:
        # The historical selected policy uses the upstream decoded sequence
        # product, including EOS; character-only products are not equivalent.
        product = row['sequence_confidence_native']
        if product >= settings['minimum_transcript_probability']:
            result.append(dict(row, transcript_probability=product,
                               confidence=reading_confidence(row)['confidence_balanced']))
    return result


def weak_word_labels(rows, gt, settings):
    return np.array([float(any(normalize_word(r['raw_text']) and normalize_word(r['raw_text']) == normalize_word(g['text'])
            and box_iou(r['bbox'], g['bbox']) >= settings['auxiliary_iou'] for g in gt))
        if r['raw_text'].strip() else -1. for r in rows], dtype=np.float32)


def fit_thresholds(scores, fractions):
    """Fit scalar cuts on TRAIN OOF values; retain all ties and report actual rates."""
    values = sorted((v for v in scores.values() if v is not None), reverse=True)
    if any(not np.isfinite(v) for v in values):
        raise ValueError('Finite TRAIN routing scores required')
    result = []
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError('Invalid TRAIN acceptance target')
        count = min(len(values), round(len(scores)*fraction))
        result.append(dict(train_target_fraction=fraction, action='reject_all' if count == 0 else 'score_at_least',
                           threshold=None if count == 0 else float(values[count-1])))
    return result


def apply_threshold(scores, policy):
    if policy['action'] == 'reject_all':
        return []
    if policy['action'] != 'score_at_least' or policy['threshold'] is None or not np.isfinite(policy['threshold']):
        raise ValueError('Invalid explicit scalar routing threshold')
    return sorted(s for s, value in scores.items() if value is not None and value >= policy['threshold'])


def fallback_counts(point):
    n, accepted = point['images'], point['accepted_images']
    if n < 1 or not 0 <= accepted <= n or point['fallback_images'] != n-accepted:
        raise ValueError('Inconsistent image-level routing denominator')
    return dict(deepsolo_images=n-accepted, total_evaluated_images=n,
                deepsolo_percentage=100*(n-accepted)/n, accepted_images=accepted,
                accepted_percentage=100*accepted/n, cost_scope='logical routing on cached outputs')
