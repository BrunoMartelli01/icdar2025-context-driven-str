"""Frozen image decisions with a replacement fallback and complete count vectors."""

from recognition.essential_threshold_frontier import COUNT_FIELDS, point


def replace_fallback(branches, replacement):
    if not branches or set(branches) != set(replacement):
        raise ValueError("Replacement must cover exactly the evaluated population")
    result = {}
    for sid, row in branches.items():
        if row["light"]["gt_words"] != replacement[sid]["gt_words"]:
            raise ValueError("Light and replacement care populations differ")
        result[sid] = dict(light=dict(row["light"]), deepsolo=dict(replacement[sid]))
    return result


def replay(branches, accepted):
    accepted = set(accepted)
    if not accepted <= set(branches):
        raise ValueError("Unknown accepted image")
    totals = [sum(row["light" if sid in accepted else "deepsolo"][k]
                  for sid, row in branches.items()) for k in COUNT_FIELDS]
    return dict(point(totals, len(branches), len(accepted), None),
                accepted_ids=sorted(accepted))


def ordered_frontier(branches, scores, tie_keys):
    """Every attainable batch prefix; score ties use the frozen RGB key."""
    if set(branches) != set(scores) or set(scores) != set(tie_keys):
        raise ValueError("Frontier inputs must have identical IDs")
    order = sorted((s for s in scores if scores[s] is not None),
                   key=lambda s: (-scores[s], tie_keys[s], s))
    totals = [sum(row["deepsolo"][k] for row in branches.values()) for k in COUNT_FIELDS]
    result = [dict(point(totals, len(branches), 0, None), boundary_tie_key=None)]
    for n, sid in enumerate(order, 1):
        totals = [v + branches[sid]["light"][k] - branches[sid]["deepsolo"][k]
                  for v, k in zip(totals, COUNT_FIELDS)]
        result.append(dict(point(totals, len(branches), n, scores[sid]),
                           boundary_tie_key=tie_keys[sid]))
    return result
