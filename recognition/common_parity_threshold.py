"""Validation-only F1 parity calibration for a frozen image utility score."""
from __future__ import annotations

import math


FIELDS = ("tp", "fp", "fn", "detected_words", "gt_words", "all_outputs", "ignored_outputs")


def aggregate(rows):
    rows = list(rows)
    for row in rows:
        if any(type(row[k]) is not int or row[k] < 0 for k in FIELDS):
            raise ValueError("Counts must be nonnegative integers")
        if (row["tp"] + row["fp"] != row["detected_words"]
                or row["tp"] + row["fn"] != row["gt_words"]
                or row["detected_words"] + row["ignored_outputs"] != row["all_outputs"]):
            raise ValueError("Inconsistent count denominator")
    total = {k: sum(r[k] for r in rows) for k in FIELDS}
    den = total["detected_words"] + total["gt_words"]
    return dict(total, images=len(rows), f1=2 * total["tp"] / den if den else 0.)


def at_least(candidate, reference):
    """Compare rational micro F1 exactly, including the declared empty=0 rule."""
    a = candidate["detected_words"] + candidate["gt_words"]
    b = reference["detected_words"] + reference["gt_words"]
    return candidate["tp"] * max(b, 1) >= reference["tp"] * max(a, 1)


def accepted_ids(scores, eligible, threshold):
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Nonempty finite scores required")
    eligible = set(eligible)
    if not eligible <= set(scores):
        raise ValueError("Unknown eligible image")
    kind = threshold["kind"]
    if kind == "accept_none":
        return []
    if kind != "score_ge" or not math.isfinite(threshold["value"]):
        raise ValueError("Invalid threshold")
    return sorted(s for s in eligible if scores[s] >= threshold["value"])


def route_observations(observations, threshold, *, guard):
    """Apply a frozen scalar cut to GT-free CommonBudgetPipeline.score outputs."""
    scores = {s: v["score"] for s, v in observations.items()}
    eligible = [s for s, v in observations.items() if not guard or v["eligible"]]
    accepted = accepted_ids(scores, eligible, threshold)
    fallback = sorted(set(scores) - set(accepted))
    n = len(scores)
    return dict(accepted_ids=accepted, fallback_ids=fallback,
                light_rows={s: observations[s]["rows"] for s in accepted},
                deepsolo_images=len(fallback), total_images=n,
                deepsolo_percentage=100 * len(fallback) / n)


def evaluate(branches, accepted):
    accepted = set(accepted)
    if not branches or not accepted <= set(branches):
        raise ValueError("Invalid routing population")
    for row in branches.values():
        if row["light"]["gt_words"] != row["deepsolo"]["gt_words"]:
            raise ValueError("Expert ground-truth denominators differ")
    n = len(branches)
    k = n - len(accepted)
    return dict(combined=aggregate(v["light" if s in accepted else "deepsolo"]
                                  for s, v in branches.items()),
                accepted_ids=sorted(accepted), light_images=len(accepted),
                deepsolo_images=k, total_images=n, deepsolo_percentage=100 * k / n,
                scope="logical cached image routing; zero new DeepSolo inference; no latency measurement")


def calibrate(scores, branches, eligible):
    """Exhaust all distinct score cuts; F1 need not vary monotonically with the cut."""
    if set(scores) != set(branches):
        raise ValueError("Score/label population mismatch")
    eligible = set(eligible)
    accepted_ids(scores, eligible, dict(kind="accept_none"))
    baseline = evaluate(branches, [])
    thresholds = [dict(kind="accept_none")]
    thresholds += [dict(kind="score_ge", value=v)
                   for v in sorted({scores[s] for s in eligible}, reverse=True)]
    trials = []
    for threshold in thresholds:
        point = evaluate(branches, accepted_ids(scores, eligible, threshold))
        trials.append(dict(threshold=threshold, **point,
                           parity=at_least(point["combined"], baseline["combined"])))
    # Each candidate has a unique acceptance count; score ties never get split.
    winner = max((t for t in trials if t["parity"]), key=lambda t: t["light_images"])
    return dict(winner=winner, trials=trials, all_deepsolo=baseline,
                all_light=evaluate(branches, list(branches)))
