"""Text-only APPROXIMATE reconstruction of frozen OCR observations.

Annotations are deliberately absent from candidate generation and selection.
Training labels and evaluation are separate callers of these pure interfaces.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import math
import random
import re

from rapidfuzz.distance import Levenshtein


def insertion_candidate(original, candidate, *, maximum_insertions=None, minimum_length=1):
    if (not re.fullmatch(r"[A-Za-z]+", original or "")
            or not re.fullmatch(r"[A-Za-z]+", candidate or "")
            or len(original) < minimum_length):
        return False
    left, right = original.casefold(), candidate.casefold()
    difference = len(right)-len(left)
    if difference < 1 or maximum_insertions is not None and difference > maximum_insertions:
        return False
    position = 0
    for char in right:
        if position < len(left) and char == left[position]:
            position += 1
    return position == len(left)


def preserves_letters(original_parts, candidate):
    """Allow word-boundary changes/additions while retaining all observed letters."""
    old=' '.join(original_parts)
    if not old.split() or not candidate.split() or any(not s.isalpha() for s in old.split()+candidate.split()):
        return False
    left=''.join(old.casefold().split());right=''.join(candidate.casefold().split())
    iterator=iter(right)
    return all(char in iterator for char in left)


def tokens(texts):
    # Empty readings remain false-positive observations, rather than disappearing.
    return [token for text in texts for token in (text.casefold().split() or [""])]


def lexical_metrics(texts, words):
    predicted, expected = Counter(tokens(texts)), Counter(w["text"].casefold() for w in words)
    matched = predicted & expected
    tp, n, g = sum(matched.values()), sum(predicted.values()), sum(expected.values())
    return dict(correct=tp, predictions=n, gt_words=g,
                precision=tp/n if n else 0., recall=tp/g if g else 0.,
                f1=2*tp/(n+g) if n+g else 0., matched=dict(matched))


def lexical_changes(before, after, words):
    expected = Counter(w["text"].casefold() for w in words)
    left, right = Counter(tokens(before)) & expected, Counter(tokens(after)) & expected
    return {"recovered":sum((right-left).values()), "damaged":sum((left-right).values()),
            "recovered_texts":dict(right-left), "damaged_texts":dict(left-right)}


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/area if area else 0.


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def reading_order(blocks):
    return sorted(blocks, key=lambda b: ((b["bbox"][1]+b["bbox"][3])/2, b["bbox"][0], b["id"]))


class TextPrior:
    """Fit-only word counts and smoothed character trigrams; external word list."""
    def __init__(self, fit_texts, external_words):
        self.counts = Counter(s.casefold() for s in fit_texts)
        self.lexicon = {s.casefold() for s in itertools.chain(fit_texts, external_words)
                        if s and len(s) <= 40 and not any(c.isspace() for c in s)}
        self.ngrams, self.contexts = Counter(), Counter()
        alphabet = set()
        for word, count in self.counts.items():
            chars = "^^"+word+"$"
            alphabet.update(chars)
            for i in range(2,len(chars)):
                self.ngrams[chars[i-2:i+1]] += count
                self.contexts[chars[i-2:i]] += count
        self.alphabet_size = max(1, len(alphabet)+1)

    def score(self, word):
        chars = "^^"+word.casefold()+"$"
        return sum(math.log((self.ngrams[chars[i-2:i+1]]+1)/
                            (self.contexts[chars[i-2:i]]+self.alphabet_size))
                   for i in range(2,len(chars)))/max(1,len(chars)-2)


def split_options(text, prior, policy):
    if not text or len(text) > policy["maximum_word_characters"] or any(c.isspace() for c in text):
        return []
    options = []
    for pieces in range(2,policy["maximum_split_words"]+1):
        for cuts in itertools.combinations(range(1,len(text)),pieces-1):
            bounds = (0,)+cuts+(len(text),)
            parts = [text[a:b] for a,b in zip(bounds,bounds[1:])]
            if all(p.casefold() in prior.lexicon and
                   (len(p) >= policy["minimum_split_characters"] or p.casefold() in {"a","i"}) for p in parts):
                value = sum(prior.score(p)+math.log1p(prior.counts[p.casefold()]) for p in parts)/len(parts)
                options.append((value,parts))
    options.sort(key=lambda row:(-row[0],row[1]))
    return [parts for _,parts in options[:policy["maximum_split_candidates"]]]


def nearby(left, right, policy):
    a,b = left["bbox"],right["bbox"]
    ah,bh = a[3]-a[1],b[3]-b[1]
    if min(ah,bh) <= 0 or (a[0]+a[2]) >= (b[0]+b[2]):
        return False
    overlap = max(0,min(a[3],b[3])-max(a[1],b[1]))/min(ah,bh)
    gap = max(0,b[0]-a[2])/min(ah,bh)
    return (overlap >= policy["minimum_vertical_overlap"] and gap <= policy["maximum_gap_heights"]
            and max(ah,bh)/min(ah,bh) <= policy["maximum_height_ratio"])


def merge_text(parts, *, deduplicate=False):
    output = parts[0]
    for part in parts[1:]:
        overlap = 0
        if deduplicate:
            for length in range(1,min(len(output),len(part))+1):
                if output[-length:] == part[:length]:
                    overlap = length
        output += part[overlap:]
    return output


def generate_operations(blocks, prior, policy):
    """Bounded, horizontal, GT-blind split/merge proposals; KEEP is implicit."""
    if len({b["id"] for b in blocks}) != len(blocks):
        raise ValueError("Duplicate OCR block ID")
    by = {b["id"]:b for b in blocks}
    groups = [(b,) for b in blocks]
    left_to_right = sorted(blocks,key=lambda b:(b["bbox"][0],b["bbox"][1],b["id"]))
    for i,left in enumerate(left_to_right):
        neighbors = [b for b in left_to_right[i+1:] if nearby(left,b,policy)]
        neighbors.sort(key=lambda b:(max(0,b["bbox"][0]-left["bbox"][2]),b["id"]))
        for right in neighbors[:policy["maximum_neighbors"]]:
            groups.append((left,right))
            if policy["maximum_merge_blocks"] >= 3:
                third = [b for b in left_to_right if nearby(right,b,policy)]
                third.sort(key=lambda b:(max(0,b["bbox"][0]-right["bbox"][2]),b["id"]))
                groups.extend((left,right,last) for last in third[:policy["maximum_neighbors"]])
    output, seen = [], set()
    for group in groups:
        parts = [b["text"] for b in group]
        if any(not p or len(p) > policy["maximum_word_characters"] for p in parts):
            continue
        parents = tuple(b["id"] for b in group)
        choices = []
        if len(group) == 1:
            choices += [("split",p) for p in split_options(parts[0],prior,policy)]
        else:
            for dedup in [False,True]:
                merged = merge_text(parts,deduplicate=dedup)
                if dedup and merged == "".join(parts):
                    continue
                op = "overlap" if dedup else "merge"
                choices.append((op,[merged]))
                choices += [(op+"_split",p) for p in split_options(merged,prior,policy)]
        for op,texts in choices:
            key = (parents,tuple(texts))
            if key in seen:
                continue
            seen.add(key)
            output.append(dict(id="operation:"+str(len(output)),parents=list(parents),texts=texts,op=op,
                               source_texts=parts,bbox=union_box([by[p]["bbox"] for p in parents])))
    # Capacity is declared, deterministic, and unrelated to correctness labels.
    return output[:policy["maximum_operations_per_image"]]


def operation_features(operation, blocks, prior):
    by = {b["id"]:b for b in blocks}
    parents = [by[p] for p in operation["parents"]]
    before,after = tokens(operation["source_texts"]),tokens(operation["texts"])
    old_scores,new_scores = [prior.score(s) for s in before],[prior.score(s) for s in after]
    gaps,overlaps = [],[]
    for left,right in zip(parents,parents[1:]):
        a,b=left["bbox"],right["bbox"];h=min(a[3]-a[1],b[3]-b[1])
        gaps.append(max(0,b[0]-a[2])/h)
        overlaps.append(max(0,min(a[3],b[3])-max(a[1],b[1]))/h)
    mean = lambda x:sum(x)/len(x)
    op=operation["op"]
    features = [float(op==name) for name in ["split","merge","overlap","merge_split","overlap_split"]]
    features += [len(parents),len(before),len(after),len(after)-len(before),
                 sum(map(len,before)),sum(map(len,after))-sum(map(len,before)),
                 mean([s in prior.lexicon for s in before]),mean([s in prior.lexicon for s in after]),
                 mean(old_scores),mean(new_scores),mean(new_scores)-mean(old_scores),
                 mean([math.log1p(prior.counts[s]) for s in before]),
                 mean([math.log1p(prior.counts[s]) for s in after]),
                 mean([p["confidence"] for p in parents]),min(p["confidence"] for p in parents),
                 mean(gaps) if gaps else 0.,min(overlaps) if overlaps else 1.,
                 min(map(len,after)),max(map(len,after))]
    if not all(math.isfinite(v) for v in features):
        raise ValueError("Nonfinite reconstruction feature")
    return features


def apply_operations(blocks, operations, scores, threshold):
    if len(operations) != len(scores):
        raise ValueError("Operation/score cardinality mismatch")
    by = {b["id"]:b for b in blocks}
    used,selected = set(),[]
    ranked = sorted(zip(operations,scores),key=lambda x:(-x[1],x[0]["id"]))
    for operation,score in ranked:
        if not math.isfinite(float(score)):
            raise ValueError("Nonfinite operation probability")
        parents=operation["parents"]
        if not parents or len(set(parents)) != len(parents) or any(p not in by for p in parents):
            raise ValueError("Malformed parent IDs")
        if score < threshold or used.intersection(parents):
            continue
        used.update(parents)
        selected.append({**operation,"probability":float(score)})
    selected += [dict(id="keep:"+b["id"],parents=[b["id"]],texts=[b["text"]],source_texts=[b["text"]],
                      bbox=b["bbox"],op="keep") for b in blocks if b["id"] not in used]
    consumed=[p for group in selected for p in group["parents"]]
    if sorted(consumed) != sorted(by):
        raise RuntimeError("Every OCR observation must be consumed exactly once")
    return reading_order(selected)


def supervised_target(box, words, *, minimum_coverage):
    """TRAIN labels only: covered GT words, otherwise best positive IoU word."""
    candidates=[]
    for word in words:
        b=word["bbox"]
        overlap=max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1]))
        area=(b[2]-b[0])*(b[3]-b[1])
        if overlap and area:
            candidates.append((overlap/area,box_iou(box,b),word))
    covered=[r[2] for r in candidates if r[0]>=minimum_coverage]
    if not covered and candidates:
        covered=[max(candidates,key=lambda r:(r[1],r[0]))[2]]
    return " ".join(w["text"] for w in sorted(covered,key=lambda w:((w["bbox"][1]+w["bbox"][3])/2,w["bbox"][0]))) or None


def synthetic_pairs(text, rng):
    """Identity, span loss, fragmentation and overlap; source language preserved."""
    pairs=[(text,text,"identity")]
    if len(text)<2:
        return pairs
    cut=rng.randrange(1,len(text))
    pairs.append((text[:cut]+" | "+text[cut:],text,"fragment"))
    pairs.append((text[:min(len(text),cut+1)]+" | "+text[cut:],text,"overlap"))
    length=rng.randrange(1,max(2,min(len(text),len(text)//2+1)))
    start=rng.randrange(0,len(text)-length+1)
    corrupted=text[:start]+text[start+length:]
    if corrupted:
        pairs.append((corrupted,text,"span_deletion"))
    return pairs


def validate_repair(original_parts, candidate, policy):
    original=" ".join(original_parts)
    if policy.get("protect_non_alphabetic", False) and any(not word.isalpha() for word in original.split()):
        return original,"protected_original_KEEP"
    if not candidate or "\n" in candidate or "\r" in candidate:
        return original,"empty_or_multiline_KEEP"
    if len(candidate.split()) > policy["maximum_output_words"]:
        return original,"word_limit_KEEP"
    if len(candidate.encode("utf-8")) > policy["maximum_output_bytes"]:
        return original,"byte_limit_KEEP"
    if any(c in candidate for c in ["<",">","|"]):
        return original,"reserved_character_KEEP"
    distance=Levenshtein.distance(original.casefold(),candidate.casefold())
    if distance > policy["maximum_edits"] or distance/max(1,len(original),len(candidate)) > policy["maximum_relative_edit"]:
        return original,"edit_limit_KEEP"
    return candidate,"accepted"


def structural_oracle(blocks, operations, words):
    """DIAGNOSTIC ONLY: exact feasible lexical ceiling of this bounded pool.

    Integer constraints consume every input once and cap every GT text count.
    The annotations choose operations here; this function must never select an
    inference output or train/calibrate a policy on evaluation annotations.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    expected=Counter(w['text'].casefold() for w in words)
    if not blocks or not expected:
        return dict(correct=0,selected_operation_ids=[],diagnostic_only=True)
    choices=[dict(id='oracle_keep:'+b['id'],parents=[b['id']],texts=[b['text']]) for b in blocks]+operations
    parent_index={b['id']:i for i,b in enumerate(blocks)}
    strings=list(expected);word_index={s:len(blocks)+i for i,s in enumerate(strings)}
    nops,nwords=len(choices),len(strings)
    rows=[];cols=[];values=[]
    for j,operation in enumerate(choices):
        for p in operation['parents']:
            rows.append(parent_index[p]);cols.append(j);values.append(1.)
        for text,count in Counter(tokens(operation['texts'])).items():
            if text in word_index:
                rows.append(word_index[text]);cols.append(j);values.append(-float(count))
    for j,text in enumerate(strings):
        rows.append(word_index[text]);cols.append(nops+j);values.append(1.)
    matrix=coo_matrix((values,(rows,cols)),shape=(len(blocks)+nwords,nops+nwords)).tocsc()
    constraint=LinearConstraint(matrix,
        np.r_[np.ones(len(blocks)),np.full(nwords,-np.inf)],
        np.r_[np.ones(len(blocks)),np.zeros(nwords)])
    result=milp(np.r_[np.zeros(nops),-np.ones(nwords)],integrality=np.ones(nops+nwords),
                bounds=Bounds(np.zeros(nops+nwords),np.r_[np.ones(nops),[expected[s] for s in strings]]),
                constraints=constraint,options={'mip_rel_gap':0.})
    if not result.success or result.x is None:
        raise RuntimeError('Bounded-pool oracle failed: '+str(result.message))
    chosen=[choices[j] for j in range(nops) if result.x[j]>.5]
    consumed=[p for op in chosen for p in op['parents']]
    if sorted(consumed)!=sorted(parent_index):raise RuntimeError('Oracle violated input exclusivity')
    correct=lexical_metrics([s for op in chosen for s in op['texts']],words)['correct']
    if correct!=round(-result.fun):raise RuntimeError('Oracle objective disagrees with lexical matching')
    return dict(correct=correct,selected_operation_ids=[op['id'] for op in chosen],diagnostic_only=True)
