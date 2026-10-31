"""GT-blind boundary alternatives and matched contextual selection features."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np

from recognition.text_reconstruction import split_options, lexical_changes


def image_fold(sample_id, folds, seed):
    if folds < 2:
        raise ValueError('At least two image folds are required')
    return int(hashlib.sha256(f'{seed}:{sample_id}'.encode()).hexdigest(), 16) % folds


def copied_segmentation(original, proposed, maximum_words):
    """Copy original Unicode characters, accepting boundaries without case edits."""
    parts = proposed.split()
    compact = ''.join(parts)
    if not 2 <= len(parts) <= maximum_words:
        return None
    if len(compact) != len(original) or compact.casefold() != original.casefold():
        return None
    output, start = [], 0
    for part in parts:
        output.append(original[start:start + len(part)])
        start += len(part)
    return ' '.join(output)


def candidates(original, neural, prior, config):
    output = [dict(text=original, origins=['keep'])]
    if not original.isalpha() or not config['minimum_characters'] <= len(original) <= config['maximum_characters']:
        return output
    seen = {original: output[0]}
    proposals = [(text, 'byt5') for text in neural]
    policy = dict(maximum_word_characters=config['maximum_characters'],
                  maximum_split_words=config['maximum_words'],
                  minimum_split_characters=config['minimum_piece_characters'],
                  maximum_split_candidates=config['maximum_lexical'])
    proposals.extend((' '.join(parts), 'lexical') for parts in split_options(original, prior, policy))
    for text, origin in proposals:
        value = copied_segmentation(original, text, config['maximum_words'])
        if value is None:
            continue
        if value not in seen:
            row = dict(text=value, origins=[])
            seen[value] = row
            output.append(row)
        if origin not in seen[value]['origins']:
            seen[value]['origins'].append(origin)
    return output


def neighbours(blocks, target, config):
    """Original OCR only; no corrected neighbours or transcript annotations."""
    a = blocks[target]['bbox']
    ah = a[3] - a[1]
    left, right = [], []
    for i, block in enumerate(blocks):
        if i == target or not block['text'].strip():
            continue
        b = block['bbox']; bh = b[3] - b[1]
        h = min(ah, bh)
        if h <= 0 or max(ah, bh) / h > config['maximum_height_ratio']:
            continue
        overlap = max(0, min(a[3], b[3]) - max(a[1], b[1])) / h
        if overlap < config['minimum_vertical_overlap']:
            continue
        if b[2] <= a[0] and a[0] - b[2] <= config['maximum_gap_heights'] * max(ah, bh):
            left.append((a[0] - b[2], b[0], i))
        if b[0] >= a[2] and b[0] - a[2] <= config['maximum_gap_heights'] * max(ah, bh):
            right.append((b[0] - a[2], b[0], i))
    def selected(rows):
        return [i for _, _, i in sorted(sorted(rows)[:config['neighbours_per_side']], key=lambda r: (r[1], r[2]))]
    return selected(left), selected(right)


def lexical_features(block, candidate, prior, external):
    old = block['text']; parts = candidate['text'].split(); lengths = list(map(len, parts))
    known = [float(p.casefold() in external) for p in parts]
    frequency = [math.log1p(prior.counts[p.casefold()]) for p in parts]
    scores = [prior.score(p) for p in parts]
    cuts = np.cumsum(lengths)[:-1]
    casing = [float(old[k-1].islower() and old[k].isupper()) for k in cuts]
    box = block['bbox']
    values = [len(old)/32, len(parts), min(lengths)/len(old), max(lengths)/len(old),
              float(old.isupper()), float(old.islower()), float(old.istitle()),
              float(old.casefold() in external), np.mean(known), min(known),
              math.log1p(prior.counts[old.casefold()]), np.mean(frequency), min(frequency),
              prior.score(old), np.mean(scores), min(scores), np.mean(scores)-prior.score(old),
              float('byt5' in candidate['origins']), float('lexical' in candidate['origins']),
              np.mean(casing), block['confidence'], block['minimum_token_probability'] or 0.,
              (box[2]-box[0])/max(1, box[3]-box[1])/len(old)]
    return [float(v) for v in values]


def language_features(original, proposed):
    """Expose score and length differences: mean PLL alone is not calibrated."""
    if original['status'] != 'scored' or proposed['status'] != 'scored':
        raise ValueError('Every eligible candidate must have a valid language score')
    a, b = original['mean_log_probability'], proposed['mean_log_probability']
    return [a, b, b-a, original['sum_log_probability'], proposed['sum_log_probability'],
            proposed['sum_log_probability']-original['sum_log_probability'],
            original['token_count'], proposed['token_count']]


def projection_profile(rgb):
    """Same-crop contrast/edge evidence; this is not character alignment."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected a nonempty RGB crop')
    gray = rgb.astype(np.float32).mean(2)/255.
    contrast = gray.std(0)
    vertical = np.abs(np.diff(gray, axis=0)).mean(0) if len(gray) > 1 else np.zeros(gray.shape[1])
    horizontal = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1])).mean(0)
    return np.stack([contrast, vertical, horizontal]).tolist()


def visual_features(profile, candidate, relative_radius):
    parts = candidate['text'].split(); total = sum(map(len, parts))
    p = np.asarray(profile, dtype=float)
    radius = max(1, int(round(p.shape[1]*relative_radius)))
    rows = []
    for cut in np.cumsum(list(map(len, parts)))[:-1]:
        x = min(p.shape[1]-1, int(round(cut/total*(p.shape[1]-1))))
        local = p[:, max(0, x-radius):min(p.shape[1], x+radius+1)]
        scale = p.mean(1)+1e-6
        rows.append(np.concatenate([local.min(1)/scale, local.mean(1)/scale]))
    if not rows:
        raise ValueError('Visual features require a split proposal')
    return np.concatenate([np.mean(rows, axis=0), np.min(rows, axis=0)]).tolist()


def edit_label(blocks, index, proposed, words):
    """Training/evaluation only: recovery, damage, neutral are distinct classes."""
    before = [b['text'] for b in blocks]
    after = list(before); after[index] = proposed
    result = lexical_changes(before, after, words)
    result['label'] = 2 if result['damaged'] else 1 if result['recovered'] else 0
    return result


def select(sample, scores, threshold):
    """One choice per original observation; KEEP wins ties and empty inputs stay."""
    if len(scores) != len(sample['blocks']):
        raise ValueError('Score/block cardinality mismatch')
    output, changes = [], []
    for block, proposals, values in zip(sample['blocks'], sample['candidates'], scores):
        if len(values) != len(proposals)-1:
            raise ValueError('Candidate/score cardinality mismatch')
        if any(not math.isfinite(v) for v in values):
            raise ValueError('Nonfinite selector score')
        best = max(range(len(values)), key=lambda j: (values[j], -j), default=None)
        index = best+1 if best is not None and values[best] >= threshold else 0
        chosen = proposals[index]['text']
        if index and copied_segmentation(block['text'], chosen, len(chosen.split())) != chosen:
            raise ValueError('Selector modified observed Unicode characters')
        output.append(chosen)
        if index:
            changes.append(dict(block_id=block['id'], before=block['text'], after=chosen,
                                candidate_index=index, score=values[index-1]))
    return output, changes


def class_utility(probabilities, classes):
    """Positive recovery probability minus damage probability; neutral is zero."""
    index = {int(c): i for i, c in enumerate(classes)}
    if 1 not in index:
        raise ValueError('Selector training has no recovery class')
    return probabilities[:, index[1]] - (probabilities[:, index[2]] if 2 in index else 0.)
