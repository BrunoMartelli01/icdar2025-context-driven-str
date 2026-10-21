"""TRAIN-only temperature and routing calibration for APPROXIMATE AdaSpot."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment


def prediction_correctness(sample, *, reading_key):
    """One-to-one correctness labels, preserving duplicates as negative cases."""
    from recognition.scene_text_v1_evaluation import box_iou
    predictions, gt = sample["predictions"], sample["ground_truth"]
    labels = np.zeros(len(predictions), dtype=np.int32)
    if predictions and gt:
        edges = np.asarray([[bool(p[reading_key]["text"]) and p[reading_key]["text"].casefold() == g["text"].casefold()
            and box_iou(p["bbox"], g["bbox"]) >= .5 for g in gt] for p in predictions], dtype=np.int32)
        left, right = linear_sum_assignment(-edges)
        labels[left[edges[left, right] > 0]] = 1
    return labels.tolist()


def fit_temperature(logits, labels, *, log_bounds):
    logits, labels = np.asarray(logits, float), np.asarray(labels, float)
    if (logits.ndim != 1 or logits.shape != labels.shape or not len(logits)
            or not np.isfinite(logits).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Nonempty finite gate logits and binary calibration labels required")
    def nll(log_temperature):
        z = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, z) - labels * z))
    optimized = minimize_scalar(nll, bounds=tuple(log_bounds), method="bounded", options={"xatol": 1e-8})
    if not optimized.success:
        raise RuntimeError("Gate temperature optimization failed: " + optimized.message)
    # T=1 remains eligible, preventing optimization from worsening calibration.
    value = float(optimized.x) if nll(optimized.x) < nll(0.) else 0.
    return {"temperature": float(np.exp(value)), "nll_before": nll(0.), "nll_after": nll(value),
        "samples": len(labels), "positives": int(labels.sum()),
        "single_class": len(np.unique(labels)) < 2}


def calibration_metrics(logits, labels, *, temperature, bins):
    z, y = np.asarray(logits, float), np.asarray(labels, int)
    if z.shape != y.shape or temperature <= 0 or bins < 1 or not np.isfinite(z).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid gate metric inputs")
    p = expit(z / temperature)
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    auc = ((rankdata(p)[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
           if positive and negative else None)
    ece = 0.
    for i in range(bins):
        selected = (p >= i / bins) & ((p < (i + 1) / bins) if i + 1 < bins else (p <= 1))
        if selected.any():
            ece += selected.sum() / max(1, len(y)) * abs(p[selected].mean() - y[selected].mean())
    return {"samples": len(y), "positives": positive, "negatives": negative,
        "auroc": float(auc) if auc is not None else None, "ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "undefined_auroc_reason": "single_class_or_empty" if auc is None else None}


def route_cached(sample, *, temperature, reject_threshold, keep_threshold, deep_threshold):
    if not 0 <= reject_threshold <= keep_threshold <= 1 or not 0 <= deep_threshold <= 1 or temperature <= 0:
        raise ValueError("Invalid calibrated routing thresholds")
    predictions, routes = [], []
    for prediction in sample["predictions"]:
        p = float(expit(prediction["gate_logit"] / temperature))
        route = "keep" if p >= keep_threshold else "reject" if p < reject_threshold else "escalate"
        reading = prediction["deep"] if route == "escalate" else prediction["light"]
        emitted = route != "reject" and bool(reading["text"]) and (route != "escalate" or reading["minimum_probability"] >= deep_threshold)
        # Actual rejection removes predictions from precision's denominator.
        # Recall retains every original careGT; all routing decisions stay saved.
        if emitted:
            predictions.append({**prediction, "text": reading["text"]})
        routes.append({"instance_id": prediction["instance_id"], "route": route,
                       "emitted": emitted, "text": reading["text"] if emitted else "", "gate_probability": p})
    return {**sample, "predictions": predictions}, routes
