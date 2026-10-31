"""Auditable bbox word F1 for approximate ICDAR2015 U-Net/PARSeq evaluation."""

from recognition.deepsolo_word_f1 import normalize_word
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching


def sample_counts(predictions, ground_truth, *, iou_threshold):
    """Keep every prediction and care GT; empty normalized strings cannot match.

    Geometry and exact casefold controls use the identical boxes/populations.
    The primary text rule is Unicode casefold followed by isalnum filtering.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("Explicit matching IoU threshold must be in (0,1]")
    geometry, exact, normalized = [], [], []
    for pi, prediction in enumerate(predictions):
        left = normalize_word(prediction["text"])
        for gi, word in enumerate(ground_truth):
            if box_iou(prediction["bbox"], word["bbox"]) < iou_threshold:
                continue
            geometry.append((pi, gi))
            if prediction["text"] and prediction["text"].casefold() == word["text"].casefold():
                exact.append((pi, gi))
            if left and left == normalize_word(word["text"]):
                normalized.append((pi, gi))
    n_pred, n_gt = len(predictions), len(ground_truth)
    matched = {name: sorted(maximum_matching(n_pred, n_gt, edges)) for name, edges in (
        ("geometry", geometry), ("casefold_exact", exact), ("alphanumeric_casefold", normalized))}
    return {"predictions": n_pred, "gt_words": n_gt,
        "symbol_only_gt": sum(not normalize_word(g["text"]) for g in ground_truth),
        "empty_normalized_predictions": sum(not normalize_word(p["text"]) for p in predictions),
        "matched_gt_indices": matched, **{name + "_tp": len(ids) for name, ids in matched.items()}}


def aggregate_counts(counts):
    """Micro P/R/F1 over all images, with no failed/empty samples omitted."""
    totals = {key: sum(row[key] for row in counts) for key in (
        "predictions", "gt_words", "symbol_only_gt", "empty_normalized_predictions",
        "geometry_tp", "casefold_exact_tp", "alphanumeric_casefold_tp")}
    result = {"images": len(counts), **totals}
    for name in ("geometry", "casefold_exact", "alphanumeric_casefold"):
        tp, np, ng = totals[name + "_tp"], totals["predictions"], totals["gt_words"]
        result[name] = {"tp": tp, "fp": np - tp, "fn": ng - tp,
            "precision": tp / np if np else None, "recall": tp / ng if ng else None,
            "f1": 2 * tp / (np + ng) if np + ng else None}
    return result
