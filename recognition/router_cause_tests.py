"""Predeclared interventions for approximate context-router failure tests."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from recognition.clip_router_evaluation import native_mask
from recognition.paper_image_router_evaluation import word_metrics
from recognition.token_context_router import base_features


def word_views(rgb, kind, settings):
    """Return complete-square views; native is intentionally the old crop control."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Nonempty uint8 RGB crop required')
    h, w = rgb.shape[:2]
    if kind == 'native':
        return [rgb]
    if kind == 'pad':
        side = max(h, w); value = np.full((side, side, 3), settings['padding_rgb'], np.uint8)
        x, y = (side-w)//2, (side-h)//2; value[y:y+h, x:x+w] = rgb
        return [value]
    if kind == 'stretch':
        size = settings['image_size']
        return [np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BICUBIC))]
    if kind != 'windows' or not 0 < settings['window_stride_fraction'] <= 1:
        raise ValueError('Unknown view kind or uncovered-window stride')
    side, length = min(h, w), max(h, w); step = max(1, math.floor(side*settings['window_stride_fraction']))
    starts = list(range(0, length-side+1, step))
    if starts[-1] != length-side:
        starts.append(length-side)
    return [np.ascontiguousarray(rgb[:, a:a+side] if w >= h else rgb[a:a+side, :]) for a in starts]


def pool_views(embeddings):
    if embeddings.ndim != 2 or not len(embeddings) or not np.isfinite(embeddings).all():
        raise ValueError('Finite nonempty view embeddings required')
    mean = embeddings.mean(0); norm = np.linalg.norm(mean)
    if not norm:
        raise ValueError('Degenerate pooled CLIP embedding')
    return (mean/norm).astype(np.float32)


def literal_foils(text, alphabet):
    if not text or len(set(alphabet)) < 2:
        raise ValueError('Nonempty word and alternative characters required')
    result = {}
    for label, index in [('prefix', 0), ('middle', len(text)//2), ('suffix', len(text)-1)]:
        replacement = next((c for c in alphabet if c != text[index]), None)
        if replacement is None:
            raise ValueError('No distinct foil character')
        result[label] = text[:index]+replacement+text[index+1:]
    return result


def row_supervision(sample, rows, gt, ignore_size, auxiliary_iou):
    """Offline care/ignore, metric-consistent matching and spatial auxiliary labels."""
    w, h = sample['image_wh']
    with Image.open(sample['validity_path']) as im:
        validity = native_mask(im.convert('L'), ignore_size)
    remaining = Counter(g['text'].casefold() for g in gt if g['text'])
    primary, auxiliary, ignored, predictions = [], [], [], []
    for r in rows:
        a, b, c, d = r['bbox']; mask = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mask).rectangle([a, b, c-1, d-1], fill=255)
        footprint = native_mask(mask, ignore_size)
        ignore = bool(footprint.any() and not np.any(footprint&validity)); ignored.append(ignore)
        predictions.append({'text': r['raw_text'], 'ignored': ignore})
        text = r['raw_text'].casefold()
        good = not ignore and bool(text) and remaining[text] > 0
        primary.append(-1. if ignore else float(good))
        if good:
            remaining[text] -= 1
        local = False
        for g in gt:
            aa, bb, cc, dd = g['bbox']; inter = max(0, min(c, cc)-max(a, aa))*max(0, min(d, dd)-max(b, bb))
            union = (c-a)*(d-b)+(cc-aa)*(dd-bb)-inter
            if text == g['text'].casefold() and inter/union >= auxiliary_iou:
                local = True; break
        auxiliary.append(float(local) if r['raw_text'].strip() else -1.)
    metrics = word_metrics(predictions, gt)
    if int(np.asarray(primary).clip(0).sum()) != metrics['tp']:
        raise ValueError('Counter-based word supervision does not replay primary TP')
    return {'primary': np.array(primary, np.float32), 'auxiliary': np.array(auxiliary, np.float32),
            'ignored': np.array(ignored, bool), 'metrics': metrics}


def omit_features(features, rows, removed, image_wh):
    removed = set(removed)
    if not removed <= set(range(len(rows))):
        raise ValueError('Unknown omitted row')
    keep = [i for i in range(len(rows)) if i not in removed]; remaining = [rows[i] for i in keep]
    value = dict(features)
    for name in ('word_text', 'word_image', 'word_geometry'):
        value[name] = features[name][keep]
    value['scalars'] = base_features(remaining, image_wh)
    value['eligible'] = bool(remaining) and all(r['raw_text'].strip() for r in remaining)
    return value, remaining


def choose_omissions(supervision, rows):
    """GT-privileged diagnostic only: deletions must increase FN by known amounts."""
    care = Counter(r['raw_text'].casefold() for r, x in zip(rows, supervision['primary']) if x >= 0)
    matches = Counter(r['raw_text'].casefold() for r, x in zip(rows, supervision['primary']) if x == 1)
    correct = [i for i, x in enumerate(supervision['primary']) if x == 1 and care[rows[i]['raw_text'].casefold()] == matches[rows[i]['raw_text'].casefold()]]
    correct.sort(key=lambda i: (-rows[i]['confidence'], rows[i]['candidate_id']))
    if len(correct) < 2:
        return None
    return {'drop_one': correct[:1], 'drop_half': correct[:max(1, len(correct)//2)]}


def utility(branch, reference_f1):
    return 2*(branch['light']['tp']-branch['deepsolo']['tp'])-reference_f1*(branch['light']['detected_words']-branch['deepsolo']['detected_words'])


def fixed_budget(scores, ids, fraction):
    return sorted((s for s in ids if scores[s] is not None), key=lambda s: (-scores[s], s))[:round(len(ids)*fraction)]


def weighted_auc(labels, scores, weights=None):
    labels = np.asarray(labels); scores = np.asarray(scores)
    if weights is None:
        weights = np.ones(len(labels))
    weights = np.asarray(weights)
    mask = weights > 0
    if len(np.unique(labels[mask])) < 2:
        return None
    return float(roc_auc_score(labels[mask], scores[mask], sample_weight=weights[mask]))


def decisions(effects, lower, upper):
    """Tests yield bounded conclusions, never invent proof from a point estimate."""
    if not all(np.isfinite(x) for x in (effects, lower, upper)) or lower > upper:
        raise ValueError('Invalid paired effect interval')
    return 'positive_effect_supported' if lower > 0 else 'negative_effect_supported' if upper < 0 else 'inconclusive'
