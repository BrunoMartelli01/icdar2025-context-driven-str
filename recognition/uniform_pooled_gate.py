"""Small shared image-count router, fitted with equal total domain weights."""

import numpy as np

def domain_weights(domains, scales):
    domains, scales = np.asarray(domains), np.asarray(scales, dtype=float)
    if domains.shape != scales.shape or not len(domains) or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError('Aligned nonempty domains and positive finite image scales required')
    result = np.empty(len(scales))
    unique = np.unique(domains)
    for domain in unique:
        selected = domains == domain
        result[selected] = scales[selected]/scales[selected].sum()*len(scales)/len(unique)
    return result


def macro_lower_gain(base, candidate, seed, draws, alpha):
    if set(base) != set(candidate) or not base:
        raise ValueError('Aligned domain-specific count triples required')
    rng = np.random.default_rng(seed)
    # Each draw averages domain F1 differences, not unequal dataset populations.
    gains = []
    for ds in sorted(base):
        a, b = np.asarray(base[ds]), np.asarray(candidate[ds])
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not len(a):
            raise ValueError('Aligned TP/FP/FN triples required')
        idx = rng.integers(len(a),size=(draws,len(a)))
        aa, bb = a[idx].sum(axis=1), b[idx].sum(axis=1)
        def f1(v):
            den=2*v[:,0]+v[:,1]+v[:,2]
            return np.divide(2*v[:,0],den,out=np.zeros(len(v)),where=den!=0)
        gains.append(f1(bb)-f1(aa))
    return float(np.quantile(np.mean(gains,axis=0),alpha))
