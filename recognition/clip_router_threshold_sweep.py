"""Whole-image router threshold curves from frozen, per-engine word metrics."""
from __future__ import annotations

from recognition.clip_image_router import accepted
from recognition.clip_router_evaluation import aggregate_split

COUNT_KEYS = ("tp", "fp", "fn", "tn", "detected_words", "gt_words", "all_outputs",
              "ignored_outputs", "invalid_envelopes")


def validate_population(rows, expected_ids, expected_words):
    if (not rows or len(rows) != len(expected_ids) or len({r["sample_id"] for r in rows}) != len(rows)
            or {r["sample_id"] for r in rows} != set(expected_ids)):
        raise ValueError("Missing, duplicate or unexpected split image")
    if sum(r["parseq"]["gt_words"] for r in rows) != expected_words:
        raise ValueError("Frozen GT denominator changed")
    for row in rows:
        for engine in ("parseq", "deepsolo"):
            m = row[engine]
            if m is None:
                if engine == "parseq":raise ValueError("Missing source PARSeq metrics")
                continue
            if any(type(m[k]) is not int or m[k] < 0 for k in COUNT_KEYS):
                raise ValueError("Invalid engine counts")
            if (m["tn"] != 0 or m["tp"] + m["fp"] != m["detected_words"]
                    or m["tp"] + m["fn"] != m["gt_words"]
                    or m["all_outputs"] != m["detected_words"] + m["ignored_outputs"]):
                raise ValueError("Inconsistent engine counts")
        if row["deepsolo"] is not None and row["deepsolo"]["gt_words"] != row["parseq"]["gt_words"]:
            raise ValueError("Engine GT denominators differ")


def point(rows, threshold, expected_ids, expected_words):
    """A threshold replaces all predictions of each rejected image together.

    Missing DeepSolo outputs prevent a full aggregate; they are never treated
    as empty predictions, silently omitted, or replaced by PARSeq.
    """
    kept = [r for r in rows if accepted(r["score"], threshold)]
    rejected = [r for r in rows if not accepted(r["score"], threshold)]
    missing = [r["sample_id"] for r in rejected if r["deepsolo"] is None]
    words = sum(r["parseq"]["detected_words"] for r in kept)
    tp = sum(r["parseq"]["tp"] for r in kept)
    observed = [{"sample_id":r["sample_id"],"status":"SUCCESS",
                 "action":"accept_parseq" if accepted(r["score"], threshold) else "deepsolo",
                 "metrics":r["parseq"] if accepted(r["score"], threshold) else r["deepsolo"]}
                for r in rows]
    return {"threshold":threshold,"status":"COMPLETE" if not missing else "MISSING_DEEPSOLO_OUTPUTS",
            "images":len(rows),"parseq_images":len(kept),"deepsolo_images":len(rejected),
            "coverage_images":len(kept)/len(rows),"words_retained_parseq":words,
            "correct_words_retained_parseq":tp,
            "precision_retained_parseq":tp/words if words else None,
            "raw_outputs_retained_parseq":sum(r["parseq"]["all_outputs"] for r in kept),
            "words_from_deepsolo":sum(r["deepsolo"]["detected_words"] for r in rejected) if not missing else None,
            "combined_model":aggregate_split(observed,expected_ids,expected_words) if not missing else None,
            "missing_deepsolo_sample_ids":missing,
            "accepted_sample_ids":[r["sample_id"] for r in kept]}


def curves(rows, thresholds, expected_ids, expected_words):
    validate_population(rows, expected_ids, expected_words)
    # Rank references and model scores stay fixed; no score/threshold fitting.
    grid = [point(rows,t,expected_ids,expected_words) for t in thresholds]
    breakpoints = sorted(set(thresholds) | {r["score"] for r in rows if r["score"] is not None})
    full = [point(rows,t,expected_ids,expected_words) for t in breakpoints]
    return {"grid":grid,"full_curve":full,"ground_truth_in_routing":False,
            "threshold_selected_on_test":False,"curve_is_exploratory":True}
