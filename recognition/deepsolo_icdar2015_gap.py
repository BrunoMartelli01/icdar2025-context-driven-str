"""Validation-only DeepSolo filtering and paired, symbol-free ICDAR15 diagnostics."""

from fractions import Fraction

import numpy as np

from recognition.deepsolo_word_f1 import apply_policy
from recognition.icdar2015_word_f1 import sample_counts
from recognition.unet_corrected_token_interfaces import lexical_counts, summarize


def score_scene(predictions, ground_truth, policy):
    """Retain invalid envelopes and empty text as unmatchable care predictions."""
    selected = apply_policy(predictions, policy)
    lexical = lexical_counts(
        [{"raw_text": p["text"] if p.get("invalid_envelope") is None else ""}
         for p in selected], ground_truth, [p["ignored"] for p in selected])
    return lexical, selected


def select_policy(scenes, score_thresholds, recognition_thresholds):
    """Exact micro-F1 optimum on supplied calibration data; deterministic ties."""
    if not scenes:
        raise ValueError("Nonempty calibration scenes required")
    grid = []
    for score in score_thresholds:
        for recognition in recognition_thresholds:
            policy = dict(resolution="native", decoder="greedy", score_threshold=score,
                          recognition_threshold=recognition, same_text_nms_iou=None)
            rows = [score_scene(p, g, policy)[0] for p, g in scenes.values()]
            metrics = summarize(rows)
            grid.append(dict(policy=policy, metrics=metrics))
    if not grid:
        raise ValueError("Explicit nonempty threshold grid required")

    def objective(row):
        m, p = row["metrics"], row["policy"]
        denominator = m["detected_words"] + m["gt_words"]
        return (Fraction(2*m["tp"], denominator) if denominator else Fraction(0),
                m["tp"], -p["score_threshold"], -p["recognition_threshold"])

    return max(grid, key=objective), grid


def evaluate_scenes(scenes, policy, *, iou_threshold, height_breaks):
    """Unchanged lexical F1; geometry is used only to diagnose missed small words."""
    if len(height_breaks) != 2 or not 0 < height_breaks[0] < height_breaks[1]:
        raise ValueError("Two explicit increasing word-height boundaries required")
    records, geometric = {}, []
    height = {name: dict(gt=0, localized=0, recognized=0)
              for name in ("small", "medium", "large")}
    for sid, (predictions, gt) in scenes.items():
        counts, selected = score_scene(predictions, gt, policy)
        # Invalid boxes stay in the denominator using a zero-area envelope.
        care = [dict(text=p["text"] if p.get("invalid_envelope") is None else "",
                     bbox=p["bbox"] if p["bbox"] is not None else [0, 0, 0, 0])
                for p in selected if not p["ignored"]]
        spatial = sample_counts(care, gt, iou_threshold=iou_threshold)
        geometric.append(spatial)
        localized = set(spatial["matched_gt_indices"]["geometry"])
        recognized = set(spatial["matched_gt_indices"]["alphanumeric_casefold"])
        for i, word in enumerate(gt):
            h = word["bbox"][3] - word["bbox"][1]
            bucket = height["small" if h <= height_breaks[0] else
                            "medium" if h <= height_breaks[1] else "large"]
            bucket["gt"] += 1
            bucket["localized"] += i in localized
            bucket["recognized"] += i in recognized
        records[sid] = dict(counts=counts, selected_predictions=selected,
                            ground_truth=gt, spatial=spatial)
    summary = summarize(r["counts"] for r in records.values())
    summary.update(deepsolo_images=len(scenes), total_images=len(scenes),
                   deepsolo_percent=100.0, inference_scope="Full-image DeepSolo; no router")
    spatial_counts = {key: sum(row[key] for row in geometric) for key in
                      ("predictions", "gt_words", "geometry_tp", "alphanumeric_casefold_tp")}
    return dict(metrics=summary, spatial_diagnostic_counts=spatial_counts,
                diagnostic_iou_threshold=iou_threshold, diagnostic_height_breaks=height_breaks,
                original_word_height=height, per_image=records)


def paired_interval(reference, candidate, *, repetitions, seed):
    """Paired image bootstrap; conditional on frozen models and chosen policies."""
    if set(reference) != set(candidate) or not reference or repetitions < 1:
        raise ValueError("Identical nonempty paired image populations required")
    ids = sorted(reference)
    arrays = [np.array([[rows[s]["counts"][key] for key in
                        ("tp", "detected_words", "gt_words")] for s in ids], dtype=float)
              for rows in (reference, candidate)]
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(ids), size=len(ids))
        values = [a[indices].sum(axis=0) for a in arrays]
        gains.append(2*values[1][0]/(values[1][1]+values[1][2]) -
                     2*values[0][0]/(values[0][1]+values[0][2]))
    return dict(repetitions=repetitions, seed=seed,
                gain95=np.quantile(gains, [0.025, 0.975]).tolist(),
                scope="Conditional paired image bootstrap; excludes fitting/selection uncertainty")
