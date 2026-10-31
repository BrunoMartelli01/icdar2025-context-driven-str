"""Symbol-free localized word metrics; empty/duplicate predictions remain counted."""
from collections import Counter

from recognition.unet_minimum_fallback import exact_counts, independent_counts
from recognition.text_reconstruction import box_iou


def normalize(text):
    return ''.join(c for c in str(text).casefold() if c.isalnum())


def counts(predictions, words, overlap):
    pred = [{**p, 'text': normalize(p['text'])} for p in predictions]
    gt = [{**w, 'text': normalize(w['text'])} for w in words]
    result = exact_counts(pred, gt, overlap)
    check = independent_counts(pred, gt, overlap)
    if any(result[k] != check[k] for k in check):
        raise RuntimeError('Independent word matching implementations disagree')
    # Lexical matching is a separate secondary diagnostic, never the primary F1.
    a, b = Counter(p['text'] for p in pred if p['text']), Counter(w['text'] for w in gt if w['text'])
    result['lexical_tp'] = sum((a & b).values())
    geometry = [{**p, 'text': 'word'} for p in pred]
    result['localized'] = exact_counts(geometry, [{**w, 'text': 'word'} for w in gt], overlap)['tp']
    result['unmatched_gt'] = [i for i in range(len(gt)) if i not in result['matched_gt']]
    result['geometry_missed_gt'] = [i for i, w in enumerate(gt)
        if not any(p.get('bbox') is not None and box_iou(p['bbox'], w['bbox']) >= overlap for p in pred)]
    return result


def aggregate(rows):
    rows = list(rows)
    result = {k: sum(r[k] for r in rows) for k in ('tp', 'predictions', 'gt', 'lexical_tp', 'localized')}
    tp, n, g = (result[k] for k in ('tp', 'predictions', 'gt'))
    result.update(images=len(rows), fp=n-tp, fn=g-tp, precision=tp/n if n else 0.,
        recall=tp/g if g else 0., f1=2*tp/(n+g) if n+g else 0.,
        lexical_f1=2*result['lexical_tp']/(n+g) if n+g else 0.,
        deepsolo_images=0, deepsolo_fraction=0.)
    return result
