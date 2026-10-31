"""GT-free decisions and explicit byte edits for APPROXIMATE word recovery."""
from __future__ import annotations

import copy
import hashlib
import numpy as np

from localize.word_graph_recovery import conflicts
from recognition.text_reconstruction import box_iou

KEEP, DELETE, STOP, BOS, PAD = 256, 257, 258, 259, 260
VOCABULARY = 261


def edit_script(source: str, target: str):
    """FIT-only minimum insertion/deletion script, preserving exact UTF-8 bytes."""
    a, b = source.encode('utf-8'), target.encode('utf-8')
    n, m = len(a), len(b)
    cost = np.zeros((n + 1, m + 1), np.int32)
    cost[:, 0] = np.arange(n + 1); cost[0] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i-1, j] + 1, cost[i, j-1] + 1,
                             cost[i-1, j-1] if a[i-1] == b[j-1] else n+m+1)
    actions = []; i, j = n, m
    while i or j:
        if i and j and a[i-1] == b[j-1] and cost[i, j] == cost[i-1, j-1]:
            actions.append(KEEP); i -= 1; j -= 1
        elif i and cost[i, j] == cost[i-1, j] + 1:
            actions.append(DELETE); i -= 1
        else:
            actions.append(b[j-1]); j -= 1
    return list(reversed(actions)) + [STOP]


def apply_edits(source: str, actions, maximum_bytes: int):
    """Return a declared invalid-generation outcome, never alternate OCR."""
    original = source.encode('utf-8'); output = bytearray(); pointer = 0
    for action in actions:
        if action == STOP:
            if pointer != len(original): return dict(valid=False, reason='unconsumed_source')
            try: text = output.decode('utf-8')
            except UnicodeDecodeError: return dict(valid=False, reason='invalid_utf8')
            return dict(valid=True, text=text, consumed=pointer)
        if action in (KEEP, DELETE):
            if pointer >= len(original): return dict(valid=False, reason='source_overrun')
            if action == KEEP: output.append(original[pointer])
            pointer += 1
        elif 0 <= action < 256: output.append(action)
        else: return dict(valid=False, reason='unknown_operation')
        if len(output) > maximum_bytes: return dict(valid=False, reason='output_limit')
    return dict(valid=False, reason='missing_stop')


def pointer_sequence(actions):
    pointer = 0; result = []
    for action in actions:
        result.append(pointer)
        pointer += action in (KEEP, DELETE)
    return result


def expanded_box(box, image_hw, fraction):
    h, w = image_hw; a, b, d, e = box
    px, py = (d-a)*fraction, (e-b)*fraction
    return [max(0, int(np.floor(a-px))), max(0, int(np.floor(b-py))),
            min(w, int(np.ceil(d+px))), min(h, int(np.ceil(e+py)))]


def tile_boxes(image_hw, fraction):
    if not .5 <= fraction <= 1: raise ValueError('Tile fraction must cover the scene')
    h, w = image_hw; th, tw = int(np.ceil(h*fraction)), int(np.ceil(w*fraction))
    return sorted({(x, y, x+tw, y+th) for y in (0, h-th) for x in (0, w-tw)})


def map_tile_box(box, tile):
    a, b, d, e = box; x, y, right, bottom = tile
    if not 0 <= a < d <= right-x or not 0 <= b < e <= bottom-y:
        raise ValueError('Local proposal is outside its tile')
    return [a+x, b+y, d+x, e+y]


def geometry_features(box, image_hw, prefix, *, observed=0., kind=0.):
    h, w = image_hw; a, b, d, e = box; bw, bh = d-a, e-b
    if min(bw, bh, h, w) <= 0: raise ValueError('Positive geometry required')
    return np.asarray([a/w, b/h, d/w, e/h, bw/w, bh/h, bw*bh/(w*h),
        np.log(bw/bh), max((box_iou(box, p['bbox']) for p in prefix), default=0.),
        len(prefix)/20., observed, kind], np.float32)


def choose_target(candidate, words, matching_iou):
    """FIT-only care target; best IoU, stable annotation order breaks ties."""
    matches = [(box_iou(candidate['bbox'], w['bbox']), -i, w) for i, w in enumerate(words)]
    return max(matches, key=lambda z: z[:2])[2] if matches and max(z[0] for z in matches) >= matching_iou else None


def operation(current, candidate, *, overlap):
    """Single-word additions/replacements; multi-output destructive merges abstain."""
    if any(p['id'] == candidate['id'] for p in current): return None
    touched = [i for i, p in enumerate(current) if conflicts(candidate, p, overlap)]
    if len(touched) > 1: return None
    if not touched: return list(current) + [copy.deepcopy(candidate)], []
    i = touched[0]
    if current[i]['text'] == candidate['text'] and current[i]['bbox'] == candidate['bbox']: return None
    result = copy.deepcopy(current); result[i] = copy.deepcopy(candidate)
    return result, [current[i]]


def decide(current, candidates, scores, *, threshold, margin, overlap, replacements):
    if threshold == 'KEEP': return copy.deepcopy(current)
    if not 0 <= threshold <= 1 or not 0 <= margin <= 1: raise ValueError('Invalid policy')
    for value in scores.values():
        if not np.isfinite(value) or not 0 <= value <= 1: raise ValueError('Invalid word-validity score')
    result = copy.deepcopy(current)
    for candidate in sorted(candidates, key=lambda p: (-scores[p['id']], p['id'])):
        if scores[candidate['id']] < threshold: continue
        change = operation(result, candidate, overlap=overlap)
        if change is None: continue
        proposed, removed = change
        if removed and (not replacements or scores[candidate['id']] <= max(scores[p['id']] for p in removed)+margin): continue
        result = proposed
    return result


def stable_fraction(identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') / 2**64
