"""Pure APPROXIMATE selective word-repair primitives; inference never takes GT."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import torch
from torch import nn

from recognition.text_reconstruction import box_iou, lexical_metrics, nearby, union_box


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def conflicts(a, b, overlap):
    left, right = set(a.get('parents') or [a['id']]), set(b.get('parents') or [b['id']])
    siblings = (a.get('method') == b.get('method') == 'split'
                and a.get('parents') == b.get('parents'))
    return (bool(left & right) and not siblings) or box_iou(a['bbox'], b['bbox']) >= overlap


def append_compatible(baseline, candidates, scores, threshold, overlap):
    """Every baseline observation remains verbatim, including erroneous ones."""
    kept = list(baseline)
    original_ids = {p['id'] for p in baseline}
    for p in sorted(candidates, key=lambda p: (-scores[p['id']], p['id'])):
        if p['id'] in original_ids or scores[p['id']] < threshold:
            continue
        if not any(conflicts(p, q, overlap) for q in kept):
            kept.append(p)
    if kept[:len(baseline)] != baseline:
        raise RuntimeError('Locked baseline prefix changed')
    return kept


def exact_matching(predictions, words, overlap):
    """Maximum bipartite matching; empty/duplicate outputs remain denominators."""
    edges = [[j for j, w in enumerate(words)
              if p['text'].casefold() == w['text'].casefold()
              and box_iou(p['bbox'], w['bbox']) >= overlap] for p in predictions]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return dict(correct=len(owner), predictions=len(predictions), gt_words=len(words),
                matched_gt_indices=sorted(owner), matched_prediction_indices=sorted(owner.values()))


def assess(predictions, words, overlap):
    spatial = exact_matching(predictions, words, overlap)
    lexical = lexical_metrics([p['text'] for p in predictions], words)
    for values in (spatial, lexical):
        n, g, tp = values['predictions'], values['gt_words'], values['correct']
        values.update(fp=n-tp, fn=g-tp, precision=tp/n if n else 0.,
                      recall=tp/g if g else 0., f1=2*tp/(n+g) if n+g else 0.)
    return dict(spatial=spatial, lexical=lexical)


def recovery_label(before, after, words, overlap):
    """Separate genuine recovery, damage and deletion-only precision effects."""
    left, right = exact_matching(before, words, overlap), exact_matching(after, words, overlap)
    a, b = set(left['matched_gt_indices']), set(right['matched_gt_indices'])
    recovered, damaged = len(b-a), len(a-b)
    return dict(recovered=recovered, damaged=damaged,
                beneficial=int(recovered > 0 and damaged == 0),
                cleanup_only=int(not recovered and not damaged and len(after) < len(before)))


def annotation_text_label(candidate, words, minimum_fraction):
    """FIT-only care-annotation overlap proxy, not physical text-presence GT."""
    box = candidate['bbox']
    area = (box[2]-box[0])*(box[3]-box[1])
    if area <= 0:
        raise ValueError('Degenerate candidate')
    return float(max((intersection(box, w['bbox'])/area for w in words), default=0.) >= minimum_fraction)


def conditional_features(candidate, baseline, base_vector):
    values = [box_iou(candidate['bbox'], p['bbox']) for p in baseline]
    same = [p for p in baseline if p['text'].casefold() == candidate['text'].casefold()]
    return list(base_vector) + [len(baseline)/20, max(values, default=0.),
        max((box_iou(candidate['bbox'], p['bbox']) for p in same), default=0.),
        float(bool(same)), len(candidate['text'])/40,
        float(not candidate['text']), float(candidate['text'].isalpha()),
        float(any(c.isdigit() for c in candidate['text']))]


def triage(candidate, text_score, correct_score, policy):
    if correct_score >= policy['correct_threshold']:
        return 'keep'
    if text_score >= policy['text_threshold']:
        return 'reread'
    return 'reject'


def routed_candidates(candidates, scores, policy, budget):
    uncertain = [p for p in candidates if triage(p, *scores[p['id']], policy) == 'reread']
    uncertain.sort(key=lambda p: (scores[p['id']][1], -scores[p['id']][0], p['id']))
    return uncertain[:budget]


def compact(text):
    return ''.join(text.split())


def boundary_layout(text):
    """Unicode character copying, UTF-8 encoder indices, and original gap flags."""
    chars, gaps = [], []
    pending = False
    for char in text:
        if char.isspace():
            pending = bool(chars)
        else:
            if chars:
                gaps.append(int(pending))
            chars.append(char)
            pending = False
    indices, count = [], 0
    for char in chars:
        count += len(char.encode('utf-8'))
        indices.append(count-1)
    return ''.join(chars), gaps, indices[:-1]


def boundary_targets(source, target):
    a, _, _ = boundary_layout(source)
    b, gaps, _ = boundary_layout(target)
    if a.casefold() != b.casefold() or len(a) != len(b):
        return None
    return gaps


def render_boundaries(source, probabilities, threshold, maximum_words):
    chars, _, _ = boundary_layout(source)
    if len(probabilities) != max(0, len(chars)-1):
        raise ValueError('Boundary probability/Unicode length mismatch')
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid boundary probabilities')
    result = ''.join(c + (' ' if i < len(probabilities) and probabilities[i] >= threshold else '')
                     for i, c in enumerate(chars))
    if len(result.split()) > maximum_words:
        return source
    if compact(result) != compact(source):
        raise RuntimeError('Boundary decoder changed characters')
    return result


def boundary_pairs(samples, words_by_id, options, rng):
    """FIT-only pairs; genuine local neighbors, never global reading-order joins."""
    pairs, seen = [], {}

    def add(source, target, kind):
        labels = boundary_targets(source, target)
        if labels is None or not labels or len(compact(source).encode('utf-8')) > options['max_bytes']:
            return
        key = (source, target)
        if key not in seen:
            seen[key] = len(pairs)
            pairs.append(dict(source=source, target=target, labels=labels, kind=kind))
        elif kind == 'real_ocr':
            # Preserve the real provenance of a pair also generated synthetically.
            pairs[seen[key]]['kind'] = kind

    for sample in samples:
        words = words_by_id[sample['sample_id']]
        for word in words:
            text = word['text']
            add(text, text, 'identity')
            if len(text) > 2 and not any(c.isspace() for c in text):
                cut = rng.randrange(1, len(text))
                add(text[:cut]+' '+text[cut:], text, 'spurious_space')
        for i, left in enumerate(words):
            neighbors = [w for j, w in enumerate(words) if i != j and nearby(left, w, options['neighbors'])]
            neighbors.sort(key=lambda w: (w['bbox'][0]-left['bbox'][2], w['text']))
            for right in neighbors[:options['maximum_neighbors']]:
                add(compact(left['text'])+compact(right['text']), left['text']+' '+right['text'], 'local_missing_space')
        for block in sample['text_blocks']:
            covered = [w for w in words if intersection(block['bbox'], w['bbox']) /
                       max(1, (w['bbox'][2]-w['bbox'][0])*(w['bbox'][3]-w['bbox'][1])) >= options['gt_coverage']]
            covered.sort(key=lambda w: (w['bbox'][0], w['bbox'][1]))
            if covered:
                add(block['text'], ' '.join(w['text'] for w in covered), 'real_ocr')
    rng.shuffle(pairs)
    return pairs[:options['maximum_pairs']]


class BoundaryHead(nn.Module):
    def __init__(self, dimensions, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


def local_variants(parent, pool, active, hw, probabilities, options):
    """Discrete RGB alternatives, bounded and GT-blind; no learned warping."""
    h, w = hw
    box = parent['bbox']
    pad = math.ceil((box[3]-box[1])*options['expansion_fraction'])
    enlarged = [max(0, box[0]-pad), max(0, box[1]-pad), min(w, box[2]+pad), min(h, box[3]+pad)]
    variants = []
    if enlarged != box:
        variants.append(dict(kind='expand', boxes=[enlarged]))
    neighbors = [p for p in pool if p['id'] != parent['id'] and p['id'] not in active
                 and (nearby(parent, p, options['neighbors']) or nearby(p, parent, options['neighbors']))]
    neighbors.sort(key=lambda p: (abs((p['bbox'][0]+p['bbox'][2])-(box[0]+box[2])), p['id']))
    if neighbors:
        variants.append(dict(kind='union', boxes=[union_box([box, neighbors[0]['bbox']])]))
    bw, bh = box[2]-box[0], box[3]-box[1]
    if bw/bh >= options['minimum_split_aspect'] and bw >= options['minimum_split_width']:
        p = np.asarray(probabilities)
        if p.ndim != 2 or not np.isfinite(p).all():
            raise ValueError('Invalid frozen text-probability map')
        side = max(h, w); size = p.shape[0]; ox, oy = (side-w)//2, (side-h)//2
        x0, x1 = max(0, math.floor((box[0]+ox)*size/side)), min(size, math.ceil((box[2]+ox)*size/side))
        y0, y1 = max(0, math.floor((box[1]+oy)*size/side)), min(size, math.ceil((box[3]+oy)*size/side))
        patch = p[y0:y1, x0:x1]
        lo = max(1, math.floor(patch.shape[1]*options['split_window'][0]))
        hi = min(patch.shape[1]-1, math.ceil(patch.shape[1]*options['split_window'][1]))
        if patch.size and hi > lo:
            cut = lo+int(np.argmin(patch.mean(0)[lo:hi]))
            x = int(round((x0+cut)*side/size-ox))
            if box[0] < x < box[2]:
                variants.append(dict(kind='split', boxes=[[box[0], box[1], x, box[3]], [x, box[1], box[2], box[3]]]))
    return variants[:options['maximum_variants']]
