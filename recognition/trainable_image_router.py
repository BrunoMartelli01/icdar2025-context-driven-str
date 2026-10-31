"""Approximate supervised tabular routing; frozen visual branches and GT-free inputs."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def feature_dict(residual, contextual, image_wh, settings):
    """No branch outcome, annotation, or ground-truth argument is accepted."""
    evidence = residual['word_evidence']; words = evidence['words']; region = residual['regions']
    n = evidence['predicted_word_count']; q = settings['lower_quantile']
    if not 0 <= q <= 1: raise ValueError('Invalid explicit feature quantile')
    values = {}; core = []
    def add(name, value, primary=True):
        missing = value is None
        if not missing and not math.isfinite(value): raise ValueError('Nonfinite inference feature: '+name)
        values[name] = 0. if missing else float(value)
        values[name+'_missing'] = float(missing)
        if primary: core.extend([name, name+'_missing'])
    add('log_words', math.log1p(n))
    for field in ('native', 'clip_rank', 'crop_agreement', 'visual'):
        v = [w[field] for w in words]
        add(field+'_mean', math.fsum(v)/n if n else None)
        add(field+'_minimum', min(v) if v else None)
        add(field+'_lower_quantile', float(np.quantile(v, q)) if v else None)
    for name in ('native_error_proxy', 'visual_error_proxy'):
        add('log_'+name, math.log1p(evidence[name]))
    for name in ('missing_count', 'missing_mass', 'missing_consensus'):
        add('log_'+name, math.log1p(region[name]))
    for head in ('text', 'kernel'):
        h = region['heads'][head]
        add(head+'_coverage', h['coverage'])
    for name in settings['control_features']:
        add('control_'+name, residual['scores'][name])
    for head in ('text', 'kernel'):
        h = region['heads'][head]
        for name in settings['head_features']:
            add(head+'_log_'+name, math.log1p(h[name]), False)
        add(head+'_log_components', math.log1p(len(h['components'])), False)
    add('log_aspect_ratio', math.log(image_wh[0]/image_wh[1]), False)
    for name in settings['context_features']:
        add('context_'+name, contextual['scores'][name], False)
    return values, {'core': core, 'full': list(values)}, n > 0 and not evidence['has_empty_reading']


def branch_f1(branches, ids, accepted):
    accepted = set(accepted)
    tp = sum(branches[s]['light' if s in accepted else 'deepsolo']['tp'] for s in ids)
    pred = sum(branches[s]['light' if s in accepted else 'deepsolo']['detected_words'] for s in ids)
    gt = sum(branches[s]['light']['gt_words'] for s in ids)
    return Fraction(2*tp, pred+gt) if pred+gt else Fraction(0)


def oracle(branches, ids, eligible, n):
    """Exact fixed-cardinality micro-F1 oracle, including an optimality certificate."""
    if set(branches) != set(ids) or not set(eligible) <= set(ids) or not 0 <= n <= len(eligible):
        raise ValueError('Oracle population or cardinality mismatch')
    ratio = Fraction(0)
    while True:
        gains = {s: 2*(branches[s]['light']['tp']-branches[s]['deepsolo']['tp'])
                 - ratio*(branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words'])
                 for s in eligible}
        accepted = sorted(eligible, key=lambda s: (-gains[s], s))[:n]
        updated = branch_f1(branches, ids, accepted)
        if updated == ratio:
            base_tp = sum(branches[s]['deepsolo']['tp'] for s in ids)
            base_den = sum(branches[s]['deepsolo']['detected_words']+branches[s]['light']['gt_words'] for s in ids)
            certificate = 2*base_tp-ratio*base_den+sum(sorted(gains.values(), reverse=True)[:n])
            if certificate != 0: raise ValueError('Oracle optimality certificate failed')
            return {'accepted_ids': accepted, 'f1': float(ratio), 'f1_fraction': str(ratio),
                    'certificate_max_residual': str(certificate), 'uses_ground_truth': True}
        if updated < ratio: raise ValueError('Fractional oracle objective decreased')
        ratio = updated


def regression_targets(branches, ids):
    if set(branches) != set(ids): raise ValueError('Fit labels must cover exactly the declared TRAIN ids')
    return np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
                      branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']]
                     for s in ids], dtype=float)


def fit_model(x, y, ids, recipe, settings):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(ids) != len(set(ids)) or len(x) != len(ids) or len(y) != len(ids):
        raise ValueError('Fit population mismatch')
    if x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid fit tensors')
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale == 0] = 1.
    z = (x-mean)/scale; kind = recipe['kind']; seed = settings['seed']
    if kind == 'ridge':
        model = Ridge(alpha=recipe['alpha'], solver='svd', fit_intercept=True)
    elif kind == 'oracle_logistic':
        if sorted(set(y)) != [0., 1.]: raise ValueError('Oracle supervision requires both classes')
        model = LogisticRegression(C=recipe['C'], solver='lbfgs', class_weight='balanced',
            max_iter=settings['logistic_max_iter'], tol=settings['logistic_tolerance'], random_state=seed)
    else:
        cls = ExtraTreesClassifier if kind == 'oracle_trees' else ExtraTreesRegressor if kind == 'trees' else None
        if cls is None: raise ValueError('Unknown explicit model recipe')
        model = cls(n_estimators=settings['trees'], max_depth=recipe['max_depth'],
            min_samples_leaf=recipe['min_samples_leaf'], max_features=1., bootstrap=False,
            random_state=seed, n_jobs=1)
    model.fit(z, y)
    saved = {'recipe': recipe, 'training_ids': list(ids), 'mean': mean.tolist(), 'scale': scale.tolist()}
    if kind == 'ridge':
        saved.update(coef=model.coef_.tolist(), intercept=model.intercept_.tolist())
    elif kind == 'oracle_logistic':
        saved.update(coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                     iterations=model.n_iter_.tolist())
    else:
        trees = []
        for tree in model.estimators_:
            t = tree.tree_
            if kind == 'oracle_trees':
                if list(model.classes_) != [0., 1.]: raise ValueError('Unexpected oracle classes')
                v = t.value[:, 0, :]; values = (v[:, 1]/v.sum(axis=1))[:, None]
            else:
                values = t.value[:, :, 0]
            trees.append({'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(),
                'left': t.children_left.tolist(), 'right': t.children_right.tolist(), 'value': values.tolist()})
        saved['trees'] = trees
    # Verify that portable JSON inference reproduces the fitted estimator.
    expected = model.predict_proba(z)[:, 1] if kind.startswith('oracle') else model.predict(z)
    if not np.allclose(predict(saved, x), expected, rtol=1e-12, atol=1e-12):
        raise ValueError('Portable model predictions differ from fitted estimator')
    return saved


def predict(model, x):
    z = (np.asarray(x, float)-np.array(model['mean']))/np.array(model['scale'])
    kind = model['recipe']['kind']
    if kind == 'ridge': return z@np.asarray(model['coef']).T+np.asarray(model['intercept'])
    if kind == 'oracle_logistic': return expit(z@np.array(model['coef'])+model['intercept'])
    z = z.astype(np.float32)  # Match sklearn's documented tree inference dtype.
    predictions = []
    for tree in model['trees']:
        rows = []
        for row in z:
            node = 0
            while tree['left'][node] != -1:
                node = tree['left'][node] if row[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
            rows.append(tree['value'][node])
        predictions.append(rows)
    result = np.mean(predictions, axis=0)
    return result[:, 0] if kind.startswith('oracle') else result


def score(predictions, kind, reference_f1):
    p = np.asarray(predictions)
    result = p if kind.startswith('oracle') else 2*p[:, 0]-reference_f1*p[:, 1]
    if not np.isfinite(result).all(): raise ValueError('Nonfinite routing predictions')
    return result
