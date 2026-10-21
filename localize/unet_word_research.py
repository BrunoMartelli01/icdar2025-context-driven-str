"""GT-blind mask stability, learned grouping and glyph/word collaboration."""

from collections import deque

import cv2
import numpy as np

from localize.word_multitask import word_instances
from segmentation.totaltext_word_kernel import instance_proposals


def embedding_instances(probabilities, embeddings, canvas, config):
    """PAN-inspired local growth, with explicit fallback for unassigned pixels.

    Embeddings are unbounded similarity vectors, never sigmoid probabilities.
    Growth cannot bridge missing foreground. No owner/GT map is accepted.
    """
    p, e, canvas = np.asarray(probabilities), np.asarray(embeddings), np.asarray(canvas)
    if p.ndim != 3 or p.shape[0] != 2 or e.ndim != 3 or e.shape[1:] != p.shape[1:] or canvas.shape != p.shape[1:] or canvas.dtype != bool:
        raise ValueError("Embedding localization needs aligned word/kernel/embedding/canvas maps")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isfinite(e).all():
        raise ValueError("Nonfinite or invalid probabilities/embeddings")
    distance = config["embedding_distance"]
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Embedding distance must be positive")
    foreground = (p[0] >= config["text_threshold"]) & canvas
    kernel = (p[1] >= config["kernel_threshold"]) & foreground
    count, seeds = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    labels = seeds.astype(np.int32)
    centers = {i: e[:, seeds == i].mean(axis=1) for i in range(1, count)}
    queue = deque(map(tuple, np.argwhere(seeds > 0)))
    height, width = foreground.shape
    while queue:
        y, x = queue.popleft()
        instance = int(labels[y, x])
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and foreground[ny, nx] and labels[ny, nx] == 0
                    and np.linalg.norm(e[:, ny, nx] - centers[instance]) < distance):
                labels[ny, nx] = instance
                queue.append((ny, nx))
    missing = foreground & (labels == 0)
    fallback_count, fallback = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
    labels[missing] = fallback[missing] + count - 1
    return labels, {"seed_components": count - 1, "unassigned_foreground_pixels": int(missing.sum()),
                    "fallback_components": fallback_count - 1}


def glyph_word_instances(labels, glyph_probability, canvas, *, threshold, minimum_overlap):
    """Attach entire glyph fragments to an existing word ID; never union word IDs.

    Ambiguous/unassociated glyphs are retained as diagnostics, not invented words.
    This extension can add glyph support outside a word mask, preserving the base.
    """
    labels, probability, canvas = np.asarray(labels), np.asarray(glyph_probability), np.asarray(canvas)
    if (labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0) or probability.shape != labels.shape
            or canvas.shape != labels.shape or canvas.dtype != bool or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1)) or not 0 < threshold < 1 or not 0.5 < minimum_overlap <= 1):
        raise ValueError("Invalid glyph/word maps or association settings")
    result = labels.astype(np.int32, copy=True)
    count, glyphs = cv2.connectedComponents(((probability >= threshold) & canvas).astype(np.uint8), connectivity=8)
    rows = []
    for glyph in range(1, count):
        support = glyphs == glyph
        ids, counts = np.unique(labels[support], return_counts=True)
        positive = ids > 0
        ids, counts = ids[positive], counts[positive]
        if not len(ids):
            rows.append({"glyph": glyph, "reason": "no_word_support"})
            continue
        best = int(np.argmax(counts))
        fraction = float(counts[best] / support.sum())
        # A glyph bridge across two independent words never merges them.
        if len(ids) != 1 or fraction < minimum_overlap:
            rows.append({"glyph": glyph, "reason": "ambiguous_or_weak_support", "word_ids": ids.tolist(), "overlap": fraction})
            continue
        result[support & (labels == 0)] = int(ids[best])
        rows.append({"glyph": glyph, "word_id": int(ids[best]), "overlap": fraction, "reason": "associated"})
    return result, {"glyph_components": count - 1, "associations": rows,
                    "added_pixels": int(np.count_nonzero((result > 0) & (labels == 0)))}


def research_instances(evidence, canvas, config, index=0):
    p = evidence["words"][index].detach().sigmoid().cpu().numpy()
    post = config["postprocessing"]
    if evidence["threshold"] is not None:
        low, high = config["adaptive"]["threshold_range"]
        t = low + (high - low) * evidence["threshold"][index, 0].detach().sigmoid().cpu().numpy()
        # Shift the actual adaptive decision onto the existing localizer threshold.
        adaptive_p = np.clip(p[0] - t + post["text_threshold"], 0, 1)
        labels, audit, _ = word_instances(np.stack((adaptive_p, p[1])), canvas, post)
        return labels, {**audit, "threshold_min": float(t.min()), "threshold_max": float(t.max())}
    if evidence["embedding"] is not None:
        return embedding_instances(p, evidence["embedding"][index].detach().cpu().numpy(), canvas, post)
    labels, audit, _ = word_instances(p, canvas, post)
    if evidence["glyph"] is not None:
        combined, association = glyph_word_instances(labels,
            evidence["glyph"][index, 0].detach().sigmoid().cpu().numpy(), canvas, **config["glyph_postprocessing"])
        return combined, {**audit, **association}
    return labels, audit


def mask_stability(probabilities, canvas, postprocessing, thresholds):
    """Rank uncertainty from saved maps, without GT and without changing masks."""
    if not thresholds or any(not np.isfinite(t) or not 0 < t < 1 for t in thresholds):
        raise ValueError("Declare valid stability thresholds")
    baseline, _, _ = word_instances(probabilities, canvas, postprocessing)
    variants = [word_instances(probabilities, canvas, {**postprocessing, "text_threshold": t})[0] for t in thresholds]
    rows = []
    for proposal in instance_proposals(baseline):
        support = baseline == proposal["instance_id"]
        agreements, counts = [], []
        for variant in variants:
            ids = np.unique(variant[support]); ids = ids[ids > 0]
            counts.append(len(ids))
            agreements.append(max((np.count_nonzero(support & (variant == i)) /
                np.count_nonzero(support | (variant == i)) for i in ids), default=0.0))
        rows.append({**proposal, "stability_iou": float(min(agreements)),
                     "risk": float(1 - min(agreements)), "overlapping_instances_by_threshold": counts})
    return baseline, rows


def limit_instances(labels, maximum, scores=None):
    """One declared budget; compare area versus learned ranking on identical IDs."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Positive integer instance budget required")
    proposals = instance_proposals(labels)
    if scores is not None:
        if set(scores) != {p["instance_id"] for p in proposals} or any(not np.isfinite(v) for v in scores.values()):
            raise ValueError("Quality scores must cover every predicted instance")
        proposals.sort(key=lambda p: (-scores[p["instance_id"]], -p["foreground_area"], p["instance_id"]))
    selected = [p["instance_id"] for p in proposals[:maximum]]
    return np.where(np.isin(labels, selected), labels, 0).astype(np.int32)
