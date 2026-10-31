"""Observable word admission and conditional fallback reconciliation.

APPROXIMATE research extension. All thresholds are explicit caller inputs.
No annotation, care mask, sample identity or fallback observation enters a
light-branch feature. Fallback words are consumed only after image routing.
"""

from collections import Counter
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from localize.independent_word_slots import overlap_iou
from recognition.common_budget_pipeline import normalize


def proposals(sources, lexicon):
    """Keep the literal reading and its qualified distance<=1 alternative."""
    result = []
    for source, rows in sources.items():
        for index, row in enumerate(rows):
            raw = row['raw_text']
            match = lexicon.nearest(raw) if lexicon is not None else None
            alternatives = [raw]
            if match is not None and normalize(match['text']) != normalize(raw):
                alternatives.append(match['text'])
            for alternative, text in enumerate(alternatives):
                result.append(dict(row, raw_text=text, original_raw_text=raw,
                    word_source=source, source_index=index,
                    proposal_id=f'{source}:{index}:{alternative}',
                    lexical_alternative=alternative,
                    lexical_distance=match['distance'] if match is not None else 2,
                    raw_lexical_exact=match is not None and match['distance'] == 0))
    return result


def feature_matrix(rows, image_wh, source_names, token_cutoffs):
    """Token uncertainty, lexical evidence, geometry and cross-source support."""
    w, h = image_wh
    if w <= 0 or h <= 0 or len(set(source_names)) != len(source_names):
        raise ValueError('Positive image size and distinct explicit sources required')
    if len(token_cutoffs) != 2 or any(not 0 <= v <= 1 for v in token_cutoffs):
        raise ValueError('Two explicit token feature cutoffs required')
    values = []
    for row in rows:
        if row['word_source'] not in source_names:
            raise ValueError('Unknown word source')
        x0, y0, x1, y1 = row['bbox']
        p = np.asarray(row['token_probabilities'], dtype=np.float64)
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError('Invalid token probabilities')
        raw, text = normalize(row['original_raw_text']), normalize(row['raw_text'])
        confidence = row['confidence']
        sequence = row.get('sequence_confidence_native', confidence)
        detector = row['detector_score']
        same, conflict, overlap = [], [], []
        for other in rows:
            if other['word_source'] == row['word_source']:
                continue
            iou = overlap_iou(row['bbox'], other['bbox'])
            overlap.append(iou)
            (same if normalize(other['raw_text']) == text else conflict).append(
                iou * other['confidence'])
        values.append([
            confidence, sequence, detector, math.sqrt(confidence * detector),
            float(p.mean()) if len(p) else 0., float(p.min()) if len(p) else 0.,
            float(np.quantile(p, .1)) if len(p) else 0.,
            float(np.mean(p < token_cutoffs[0])) if len(p) else 0.,
            float(np.mean(p < token_cutoffs[1])) if len(p) else 0.,
            -math.log(max(sequence, np.finfo(float).tiny)),
            math.log1p(len(raw)), math.log1p(len(text)),
            sum(c.isdigit() for c in text) / max(len(text), 1),
            sum(c.isupper() for c in row['original_raw_text']) /
                max(len(row['original_raw_text']), 1),
            float(not text), float(text.isdigit()),
            float(row['lexical_alternative']), float(row['lexical_distance']),
            float(row['raw_lexical_exact']),
            (x1-x0)/w, (y1-y0)/h, (x1-x0)*(y1-y0)/(w*h),
            math.log(max(x1-x0, 1)/max(y1-y0, 1)),
            (x0+x1)/(2*w), (y0+y1)/(2*h),
            max(same, default=0.), max(conflict, default=0.),
            max(overlap, default=0.), math.log1p(len(rows)),
            *[float(row['word_source'] == s) for s in source_names]])
    result = np.asarray(values, np.float64).reshape(-1, 29+len(source_names))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite observable word features')
    return result


def correctness_proxy(rows, truth):
    """Lexical set membership proxy; final metrics still enforce multiplicity."""
    targets = {normalize(g['text']) for g in truth if normalize(g['text'])}
    return np.asarray([bool(normalize(r['raw_text'])) and
        normalize(r['raw_text']) in targets for r in rows], dtype=np.int64)


def fit_admission(x, y, recipe, seed):
    if len(np.unique(y)) != 2:
        raise ValueError('Both word label classes required')
    if recipe['kind'] == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=recipe['C'], max_iter=recipe['iterations'], random_state=seed))
    elif recipe['kind'] == 'hgb':
        model = HistGradientBoostingClassifier(max_iter=recipe['iterations'],
            max_leaf_nodes=recipe['leaves'], min_samples_leaf=recipe['minimum_leaf'],
            l2_regularization=recipe['l2'], early_stopping=False, random_state=seed)
    else:
        raise ValueError('Explicit supported admission recipe required')
    return model.fit(x, y)


def admit(rows, probabilities, *, threshold, duplicate_iou):
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(rows) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid admission probabilities')
    if not 0 <= threshold <= 1 or not 0 < duplicate_iou <= 1:
        raise ValueError('Explicit valid admission and suppression thresholds required')
    selected = []
    for index in sorted(range(len(rows)), key=lambda i: (-p[i], rows[i]['proposal_id'])):
        row = rows[index]
        if p[index] < threshold:
            continue
        if any(overlap_iou(row['bbox'], other['bbox']) >= duplicate_iou
                for other in selected):
            continue
        selected.append(dict(row, admission_probability=float(p[index])))
    return selected


def reconcile(light, fallback, *, confidence, maximum_overlap):
    """Preserve fallback and append confident spatially distinct light words.

    The caller must have already selected this image for a real or cached
    fallback invocation. This function does not support choosing image routes.
    """
    if confidence is None:
        return [dict(row) for row in fallback]
    if not 0 <= confidence <= 1 or not 0 < maximum_overlap <= 1:
        raise ValueError('Explicit valid reconciliation thresholds required')
    result = [dict(row) for row in fallback]
    for row in light:
        if row['admission_probability'] < confidence:
            continue
        if any(other['bbox'] is not None and overlap_iou(row['bbox'], other['bbox'])
                >= maximum_overlap for other in result):
            continue
        result.append(dict(row, fallback_complement=True))
    return result


def risk_summary(rows, selected, probabilities):
    p = np.asarray(probabilities)
    retained = [row['admission_probability'] for row in selected]
    out = []
    for values in (p, np.asarray(retained)):
        out.extend([float(sum(values)), float(sum(1-values)),
            float(np.mean(values)) if len(values) else 0.,
            float(min(values)) if len(values) else 0., math.log1p(len(values))])
    out.extend([float(not selected), float(len(Counter(normalize(r['raw_text'])
        for r in selected))) / max(len(selected), 1)])
    return np.asarray(out, dtype=np.float64)
