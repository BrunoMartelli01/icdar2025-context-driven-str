"""Experimental image-wide OCR correction; inference functions never read GT.

Image-wide Counterfactual Boundary Selection (ICBS) is an experimental name,
not a novelty or faithful-paper claim. All context comes from original OCR.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

import numpy as np
from rapidfuzz.distance import Levenshtein
import torch
from torch import nn

from recognition.text_reconstruction import tokens, union_box


def eligible(text, maximum):
    return bool(text) and text.isalpha() and 2 <= len(text) <= maximum


def adjacent_spans(blocks, cfg):
    """Original-box horizontal neighbours only; no annotation-derived groups."""
    spans = [(i,) for i in range(len(blocks))]
    for i, a in enumerate(blocks):
        rights = []
        for j, b in enumerate(blocks):
            ah, bh = a['bbox'][3]-a['bbox'][1], b['bbox'][3]-b['bbox'][1]
            h = min(ah, bh)
            if i == j or h <= 0 or a['bbox'][0] >= b['bbox'][0]:
                continue
            overlap = max(0, min(a['bbox'][3], b['bbox'][3])-max(a['bbox'][1], b['bbox'][1]))/h
            gap = (b['bbox'][0]-a['bbox'][2])/h
            if overlap >= cfg['minimum_vertical_overlap'] and 0 <= gap <= cfg['maximum_gap_heights']:
                rights.append((gap, b['bbox'][0], j))
        if rights:
            spans.append((i, min(rights)[2]))
    return spans


def overlap_fraction(box, word):
    a, b = box, word
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return intersection/area if area else 0.


def aligned_target(box, words, cfg):
    """TRAIN-only high-coverage target; partial neighbouring words are ambiguous."""
    covered, partial = [], []
    for w in words:
        fraction = overlap_fraction(box, w['bbox'])
        if fraction >= cfg['minimum_gt_coverage']:
            covered.append(w)
        elif fraction > cfg['partial_overlap_limit']:
            partial.append(w)
    if not covered or partial or len(covered) > cfg['maximum_words']:
        return None
    covered.sort(key=lambda w: (w['bbox'][0], w['bbox'][1]))
    if len(covered) > 1:
        heights = [w['bbox'][3]-w['bbox'][1] for w in covered]
        centers = [(w['bbox'][1]+w['bbox'][3])/2 for w in covered]
        if max(centers)-min(centers) > cfg['maximum_center_heights']*max(1, min(heights)):
            return None
    return ' '.join(w['text'] for w in covered)


def unambiguous_pairs(rows):
    """Do not choose one target for a source with conflicting normalized labels."""
    targets = defaultdict(set)
    for r in rows:
        targets[r['source'].casefold()].add(r['target'].casefold())
    rejected = {s for s, values in targets.items() if len(values) > 1}
    return [r for r in rows if r['source'].casefold() not in rejected], len(rejected)


def confusions(rows):
    counts = Counter()
    for row in rows:
        clean = ''.join(row['target'].casefold().split())
        noisy = ''.join(row['source'].replace('|', '').casefold().split())
        for edit in Levenshtein.editops(clean, noisy):
            left = clean[edit.src_pos] if edit.tag != 'insert' else ''
            right = noisy[edit.dest_pos] if edit.tag != 'delete' else ''
            counts[(edit.tag, left, right)] += 1
    return [dict(operation=k[0], clean=k[1], noisy=k[2], count=v) for k, v in sorted(counts.items())]


def corrupt_text(target, rng, empirical, mode, cfg):
    """Deterministic TRAIN-only synthetic corruption, not the ACL2024 recipe."""
    if rng.random() < cfg['identity_probability']:
        return target, 'identity'
    value = target
    if rng.random() < cfg['boundary_probability']:
        value = value.replace(' ', '')
        if len(value) > cfg['minimum_fragment_characters'] and rng.random() < cfg['spurious_boundary_probability']:
            k = rng.randrange(1, len(value)); value = value[:k]+' | '+value[k:]
        return value, 'boundary'
    chars = list(value)
    choices = [r for r in empirical if r['clean'] in value.casefold() or r['operation'] == 'insert']
    if mode == 'empirical' and choices:
        r = rng.choices(choices, weights=[r['count'] for r in choices], k=1)[0]
        positions = [i for i, ch in enumerate(chars) if ch.casefold() == r['clean']]
        if r['operation'] == 'insert':
            chars.insert(rng.randrange(len(chars)+1), r['noisy'])
        elif positions:
            i = rng.choice(positions)
            if r['operation'] == 'delete': chars.pop(i)
            else: chars[i] = r['noisy'].upper() if chars[i].isupper() else r['noisy']
    else:
        i = rng.randrange(len(chars)); action = rng.choice(('replace', 'delete', 'insert'))
        if action == 'delete': chars.pop(i)
        elif action == 'insert': chars.insert(i, rng.choice(cfg['random_alphabet']))
        else: chars[i] = rng.choice(cfg['random_alphabet'])
    return ''.join(chars), 'character'


def accept_generated(originals, generated, cfg):
    if not generated or not generated.split(): return None
    parts = generated.split()
    if len(parts) > cfg['maximum_words'] or any(not p.isalpha() for p in parts): return None
    original = ''.join(originals).casefold(); proposed = ''.join(parts).casefold()
    if not original or Levenshtein.distance(original, proposed)/max(len(original), len(proposed)) > cfg['maximum_relative_edit']:
        return None
    if all(s.isupper() for s in originals): return ' '.join(parts).upper()
    if all(s.islower() for s in originals): return ' '.join(parts).lower()
    return ' '.join(parts)


def context_windows(blocks, parents, tokenizer, budget, mode, confidence_floor=.01):
    """Every non-parent OCR token appears in a window, without truncation.

Oversized individual observations are split into token-ID chunks for context
only. Ordered mode retains their order relative to the target. Inventory mode
uses the other observations as an unordered evidence inventory (stable order).
"""
    if mode not in ('ordered', 'inventory') or budget < 1: raise ValueError('Invalid context mode/budget')
    excluded = set(parents); items = []
    for i, b in enumerate(blocks):
        if i in excluded or not b['text'].strip(): continue
        ids = tokenizer.encode(b['text'], add_special_tokens=False)
        for start in range(0, len(ids), budget):
            segment = ids[start:start+budget]
            items.append(dict(index=i, ids=segment, text=tokenizer.decode(segment), confidence=float(b['confidence'])))
    chunks, current, length = [], [], 0
    for item in items:
        if current and length+len(item['ids']) > budget:
            chunks.append(current); current=[]; length=0
        current.append(item); length += len(item['ids'])
    if current: chunks.append(current)
    output=[]
    for group in chunks:
        target = min(parents)
        left = [r['text'] for r in group if r['index'] < target] if mode == 'ordered' else []
        right = [r['text'] for r in group if r['index'] > target] if mode == 'ordered' else []
        output.append(dict(left=left, right=right,
            caption=' '.join(r['text'] for r in group) if mode == 'inventory' else None,
            source_indices=[r['index'] for r in group], token_ids=[t for r in group for t in r['ids']],
            weight=max(confidence_floor, float(np.mean([r['confidence'] for r in group])))))
    return output


def pooled_language(reference, alternatives, weights):
    """Counterfactual target PLL differences, not correctness probabilities."""
    if len(reference) != len(alternatives) or len(reference) != len(weights) or not weights:
        raise ValueError('Language-window cardinality mismatch')
    a = np.asarray(reference, dtype=float); b = np.asarray(alternatives, dtype=float); w=np.asarray(weights, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Invalid language evidence')
    delta=b-a
    return [float(a.mean()), float(b.mean()), float(delta.mean()), float(delta.min()), float(delta.max()),
            float(delta.std()), float(np.average(delta, weights=w)), float(np.mean(delta > 0)), len(delta)]


def operation_features(blocks, op, prior, vocabulary):
    parents=[blocks[i] for i in op['parents']]; old=[p['text'] for p in parents]; parts=op['text'].split()
    a=''.join(old).casefold(); b=''.join(parts).casefold(); old_scores=[prior.score(s) for s in old]; new_scores=[prior.score(s) for s in parts]
    box=union_box([p['bbox'] for p in parents]); lengths=list(map(len, parts))
    return [len(a)/32, len(b)/32, len(parents), len(parts), min(lengths)/max(1,len(b)),
        Levenshtein.distance(a,b)/max(1,len(a),len(b)), float(a==b),
        float(all(s.isupper() for s in old)),float(all(s.islower() for s in old)),
        float(any(any(ch.isdigit() for ch in s) for s in old)),
        float(np.mean([p['confidence'] for p in parents])),float(min(p['confidence'] for p in parents)),
        float(min(p['minimum_token_probability'] or 0. for p in parents)),
        float(np.mean(old_scores)),float(np.mean(new_scores)),float(min(new_scores)),
        float(np.mean(new_scores)-np.mean(old_scores)),
        float(np.mean([s.casefold() in vocabulary for s in old])),float(np.mean([s.casefold() in vocabulary for s in parts])),
        float(np.mean([math.log1p(prior.counts[s.casefold()]) for s in parts])),
        (box[2]-box[0])/max(1,box[3]-box[1])/max(1,len(a)),
        float(op['kind']=='boundary'),float(op['kind']=='character'),float(op['kind']=='fragment')]


def apply_choices(blocks, operations, scores, threshold):
    if len(operations) != len(scores) or any(not math.isfinite(float(v)) for v in scores):
        raise ValueError('Invalid operation scores')
    if math.isnan(threshold): raise ValueError('Invalid threshold')
    used=set(); chosen=[]
    for i in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i)):
        op=operations[i]; parents=op['parents']
        if not parents or len(set(parents))!=len(parents) or min(parents)<0 or max(parents)>=len(blocks):
            raise ValueError('Invalid parent indices')
        if scores[i] <= threshold or set(parents)&used: continue
        if not op['text'].split(): raise ValueError('Correction cannot silently delete an observation')
        used.update(parents); chosen.append(i)
    replacements={min(operations[i]['parents']):operations[i]['text'] for i in chosen}
    output=[replacements[i] if i in replacements else b['text'] for i,b in enumerate(blocks) if i in replacements or i not in used]
    return output, chosen


def select_subset(operations, kind):
    if kind == 'expanded': return list(range(len(operations)))
    if kind == 'boundary': return [i for i,o in enumerate(operations) if o['kind']=='boundary']
    if kind == 'character': return [i for i,o in enumerate(operations) if o['kind'] in ('boundary','character')]
    if kind == 'fragment': return [i for i,o in enumerate(operations) if o['kind'] in ('boundary','fragment')]
    raise ValueError('Unknown candidate subset')


class ListwiseRanker(nn.Module):
    """Shared candidate scorer; KEEP has fixed zero logit in each parent group."""
    def __init__(self, dimensions, hidden):
        super().__init__(); self.network=nn.Sequential(nn.Linear(dimensions,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,x): return self.network(x).squeeze(-1)


def listwise_loss(values, labels):
    if len(values) != len(labels): raise ValueError('Listwise label cardinality mismatch')
    logits=torch.cat([values.new_zeros(1), values])
    good=torch.as_tensor([v==1 for v in labels],device=values.device,dtype=torch.bool)
    target=torch.cat([(~good.any()).float().reshape(1),good.float()]); target=target/target.sum()
    return -(target*logits.log_softmax(0)).sum()


def block_features(blocks,index,prior,vocabulary,global_original):
    b=blocks[index]; text=b['text']; box=b['bbox']; others=[p for i,p in enumerate(blocks) if i!=index]
    return [len(text)/32,float(text.isalpha()),float(text.isupper()),float(text.islower()),
        float(any(c.isdigit() for c in text)),float(text.casefold() in vocabulary),prior.score(text),
        math.log1p(prior.counts[text.casefold()]),b['confidence'],b['minimum_token_probability'] or 0.,
        (box[2]-box[0])/max(1,box[3]-box[1]),len(others),
        sum(p['text'].casefold()==text.casefold() for p in others),
        float(np.mean([p['confidence'] for p in others])) if others else 0.,*global_original]


def filter_readings(blocks, scores, threshold, policy, presence=None, presence_threshold=None):
    if len(scores)!=len(blocks) or any(not math.isfinite(float(v)) for v in scores): raise ValueError('Invalid filter scores')
    if policy not in ('exact','presence','duplicate'): raise ValueError('Unknown filter policy')
    if policy=='presence' and (presence is None or len(presence)!=len(blocks) or presence_threshold is None):
        raise ValueError('Presence-aware filtering requires explicit scores and threshold')
    kept=[];removed=[]
    for i,b in enumerate(blocks):
        drop=scores[i] < threshold
        if policy=='presence': drop=drop and presence[i]<presence_threshold
        if policy=='duplicate':
            drop=drop and any(blocks[j]['text'].casefold()==b['text'].casefold() and scores[j]>scores[i] and
                overlap_fraction(b['bbox'],blocks[j]['bbox'])>.5 for j in range(len(blocks)) if j!=i)
        if drop: removed.append(i)
        else: kept.append(b['text'])
    return kept,removed
