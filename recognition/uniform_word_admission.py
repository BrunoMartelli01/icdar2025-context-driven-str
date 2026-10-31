"""Explicit probability filtering and validation selection for light OCR words."""

import math
import numpy as np


def admit(rows, probabilities, threshold):
    p=np.asarray(probabilities)
    if (p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or
            not math.isfinite(threshold) or not 0<=threshold<=1):
        raise ValueError('Finite aligned word probabilities and explicit threshold required')
    result=[]
    for row in rows:
        i=row['source_row']
        if type(i) is not int or not 0<=i<len(p):raise ValueError('Original raw-word source index required')
        if p[i]>=threshold:result.append(dict(row,word_probability=float(p[i])))
    return result


def select_threshold(trials,tolerance):
    if not trials or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Nonempty validation trials and nonnegative tolerance required')
    if any(not math.isfinite(t['counts']['f1']) for t in trials):raise ValueError('Finite validation F1 required')
    best=max(t['counts']['f1'] for t in trials)
    return min((t for t in trials if t['counts']['f1']>=best-tolerance),key=lambda t:t['threshold'])
