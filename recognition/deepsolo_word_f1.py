"""APPROXIMATE symbol-free word counts and lexicon-free DeepSolo experiments.

No GT enters decoding or filtering. Each detection stays one word; repeated
words count separately. Invalid envelopes remain unmatchable false positives.
This metric is not the official Total-Text polygon/word-spotting evaluator.
"""
from __future__ import annotations

from collections import Counter
import math


def normalize_word(text):
    if not isinstance(text, str):
        raise ValueError('Word must be a string')
    return ''.join(c for c in text.casefold() if c.isalnum())


def word_counts(predictions, ground_truth):
    care = [p for p in predictions if not p['ignored']]
    targets = [normalize_word(g['text']) for g in ground_truth]
    remaining = Counter(t for t in targets if t)
    tp = 0
    symbol_only_predictions = 0
    for p in care:
        key = normalize_word(p['text'])
        if not key and p['text']:
            symbol_only_predictions += 1
        elif p.get('invalid_envelope') is None and remaining[key]:
            tp += 1
            remaining[key] -= 1
    # Preserve every original care denominator, including symbol-only targets
    # and predictions. Empty normalized strings never match, as in the existing
    # repository alphanumeric comparison. Symbol removal is not GT filtering.
    n_pred = len(care)
    n_gt = len(targets)
    return dict(tp=tp, fp=n_pred-tp, fn=n_gt-tp, detected_words=n_pred,
                gt_words=n_gt, all_outputs=len(predictions),
                ignored_outputs=len(predictions)-len(care),
                symbol_only_gt=sum(not t for t in targets),
                symbol_only_predictions=symbol_only_predictions)


def aggregate(rows):
    rows = list(rows)
    keys = ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs',
            'ignored_outputs', 'symbol_only_gt', 'symbol_only_predictions')
    counts = {k: sum(r[k] for r in rows) for k in keys}
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    if counts['detected_words'] != tp+fp or counts['gt_words'] != tp+fn:
        raise ValueError('Inconsistent word denominators')
    return dict(counts, images=len(rows), precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def _logadd(*values):
    top = max(values)
    return top if top == -math.inf else top + math.log(sum(math.exp(v-top) for v in values))


def beam_decode(probabilities, width):
    """CTC prefix beam search over 36 labels; both upstream special IDs reset.

    The frozen evaluator treats IDs >=36 as nonprinting separators. Merge their
    probability mass into a blank so path marginalization preserves that rule.
    No dictionary, language model, target length prior, or GT is consulted.
    """
    if type(width) is not int or width < 1 or not probabilities:
        raise ValueError('Nonempty probabilities and positive beam width required')
    beam = {'': (0., -math.inf)}
    labels = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for row in probabilities:
        if (len(row) != 38 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row)
                or abs(sum(row)-1) > 1e-4):
            raise ValueError('Expected normalized 38-class DeepSolo probabilities')
        logp = [math.log(p) if p else -math.inf for p in row[:36]]
        blank = math.log(sum(row[36:])) if sum(row[36:]) else -math.inf
        nxt = {}

        def add(prefix, pb=-math.inf, pn=-math.inf):
            a, b = nxt.get(prefix, (-math.inf, -math.inf))
            nxt[prefix] = (_logadd(a, pb), _logadd(b, pn))

        for prefix, (pb, pn) in beam.items():
            total = _logadd(pb, pn)
            add(prefix, pb=total+blank)
            for c, prob in zip(labels, logp):
                if prefix.endswith(c):
                    add(prefix, pn=pn+prob)
                    add(prefix+c, pn=pb+prob)
                else:
                    add(prefix+c, pn=total+prob)
        beam = dict(sorted(nxt.items(), key=lambda kv: (-_logadd(*kv[1]), kv[0]))[:width])
    return min(beam, key=lambda k: (-_logadd(*beam[k]), k))


def recognition_confidence(probabilities):
    """Geometric mean over greedy emitted characters, excluding blank positions."""
    emitted, previous = [], None
    for row in probabilities:
        token = max(range(len(row)), key=row.__getitem__)
        if token < 36 and token != previous:
            emitted.append(row[token])
        previous = token if token < 36 else None
    return math.exp(sum(math.log(p) for p in emitted)/len(emitted)) if emitted else 0.


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    area = lambda q: max(0, q[2]-q[0])*max(0, q[3]-q[1])
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = area(a)+area(b)-inter
    return inter/union if union else 0.


def apply_policy(predictions, policy):
    """Apply a frozen policy without ground truth; ignored flags are evaluation-only."""
    kept = []
    for p in sorted(predictions, key=lambda p: (-p['score'], p['instance_id'])):
        if p['score'] < policy['score_threshold'] or p['recognition_confidence'] < policy['recognition_threshold']:
            continue
        text = p['text'] if policy['decoder'] == 'greedy' else p['beam_text']
        if policy['same_text_nms_iou'] is not None and any(
                normalize_word(k['text']) == normalize_word(text)
                and box_iou(k['bbox'], p['bbox']) > policy['same_text_nms_iou'] for k in kept):
            continue
        kept.append(dict(p, text=text))
    return kept


def fuse_views(native, large, policy):
    """Confidence-ranked cross-scale suppression; never consult care/GT flags.

    Suppress only overlapping predictions from different views. Repeated words
    at different locations and overlapping instances within one view survive.
    """
    threshold = policy['fusion_iou']
    power = policy['recognition_rank_power']
    if not 0 < threshold <= 1 or power not in (0, 1):
        raise ValueError('Invalid cross-scale fusion policy')
    candidates = [dict(p, view=v) for v, preds in (('native', native), ('large', large))
                  for p in apply_policy(preds, policy)]
    candidates.sort(key=lambda p: (-p['score']*p['recognition_confidence']**power,
                                   p['view'] != 'native', p['instance_id']))
    kept = []
    for p in candidates:
        if not any(k['view'] != p['view'] and box_iou(p['bbox'], k['bbox']) > threshold for k in kept):
            kept.append(p)
    return kept


def error_inventory(predictions, ground_truth, iou_threshold):
    """Descriptive localization upper bound and lexical errors; never a selector."""
    preds = [p for p in predictions if not p['ignored'] and normalize_word(p['text'])]
    targets = [g for g in ground_truth if normalize_word(g['text'])]
    # Maximum one-to-one geometry matching, independent of transcription.
    assigned = {}

    def visit(i, seen):
        for j, g in enumerate(targets):
            if j in seen or box_iou(preds[i]['bbox'], g['bbox']) < iou_threshold:
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(preds)):
        visit(i, set())
    wrong = [{'gt': targets[j]['text'], 'predicted': preds[i]['text'],
              'score': preds[i]['score']} for j, i in assigned.items()
             if normalize_word(targets[j]['text']) != normalize_word(preds[i]['text'])]
    return dict(localized_words=len(assigned), unlocalized_words=len(targets)-len(assigned),
                localized_wrong_transcription=len(wrong), wrong_transcriptions=wrong,
                missed=[g for j, g in enumerate(targets) if j not in assigned],
                unmatched_predictions=[p['text'] for i, p in enumerate(preds) if i not in assigned.values()])
