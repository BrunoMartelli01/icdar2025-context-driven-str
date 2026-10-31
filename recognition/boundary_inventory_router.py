"""Budget-preserving, bounded exchanges around a frozen router's acceptance rank."""
from __future__ import annotations

import math


def select(base_scores, audit_scores, ids, fraction, policy):
    if set(base_scores) != set(ids) or set(audit_scores) != set(ids):
        raise ValueError('Both label-free score maps must cover the complete batch')
    if not 0 < fraction < 1 or not 0 <= policy['half_width_fraction'] <= min(fraction, 1-fraction) or not 0 <= policy['audit_weight'] <= 1:
        raise ValueError('Explicit bounded rank policy required')
    eligible = [s for s in ids if base_scores[s] is not None]
    if any(audit_scores[s] is None for s in eligible) or any(not math.isfinite(v) for scores in (base_scores, audit_scores) for v in scores.values() if v is not None):
        raise ValueError('Audit evidence missing/nonfinite for eligible image')
    order = sorted(eligible, key=lambda s: (-base_scores[s], s)); k = round(len(ids)*fraction)
    if k > len(order):
        raise ValueError('Requested quota exceeds the eligible population')
    width = round(len(ids)*policy['half_width_fraction'])
    left, right = max(0, k-width), min(len(order), k+width)
    middle = order[left:right]
    # At weight zero preserve ALL baseline ties/order exactly.
    if not width or policy['audit_weight'] == 0:
        return order[:k]
    other = sorted(middle, key=lambda s: (-audit_scores[s], s))
    position = {s: i for i, s in enumerate(other)}; original = {s: i for i, s in enumerate(middle)}
    weight = policy['audit_weight']
    ranked = sorted(middle, key=lambda s: ((1-weight)*original[s]+weight*position[s], original[s]))
    return order[:left]+ranked[:k-left]
