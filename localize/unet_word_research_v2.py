"""GT-blind soft embedding growth and pixel links; no silent pixel rejection."""

import heapq
import cv2
import numpy as np
from scipy.special import expit

from localize.unet_word_research import glyph_word_instances
from localize.word_multitask import word_instances


NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
OPPOSITE = (7, 6, 5, 4, 3, 2, 1, 0)


def soft_embedding_instances(probabilities, embeddings, canvas, post, options):
    p, e, canvas = np.asarray(probabilities), np.asarray(embeddings), np.asarray(canvas)
    if (p.ndim != 3 or p.shape[0] != 2 or e.ndim != 3 or e.shape[1:] != p.shape[1:]
            or canvas.shape != p.shape[1:] or canvas.dtype != bool or not np.isfinite(p).all()
            or not np.isfinite(e).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Invalid aligned embedding evidence")
    weight, scale = options["weight"], options["distance_scale"]
    if not np.isfinite([weight, scale]).all() or weight < 0 or scale <= 0:
        raise ValueError("Invalid soft embedding cost")
    if weight == 0:
        labels, audit, _ = word_instances(p, canvas, post)
        return labels, {**audit, "soft_embedding_weight": weight}
    foreground = (p[0] >= post["text_threshold"]) & canvas
    kernel = (p[1] >= post["kernel_threshold"]) & foreground
    count, seeds = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=8)
    centers = {i: e[:, seeds == i].mean(axis=1) for i in range(1, count)}
    weights = np.asarray(post["watershed_weights"][:2], np.float64)
    confidence = np.einsum("c,cyx->yx", weights / weights.sum(), p)
    labels = np.zeros(foreground.shape, np.int32)
    queue = [(float(1 - confidence[y, x]), 0, int(seeds[y, x]), int(y), int(x))
             for y, x in np.argwhere(seeds > 0)]
    heapq.heapify(queue)
    h, w = labels.shape
    while queue:
        cost, distance, label, y, x = heapq.heappop(queue)
        if labels[y, x]:
            continue
        labels[y, x] = label
        for dy, dx in NEIGHBORS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w and foreground[ny, nx] and labels[ny, nx] == 0):
                continue
            if seeds[ny, nx] and seeds[ny, nx] != label:
                continue
            difference = float(np.linalg.norm(e[:, ny, nx] - centers[label]))
            penalty = weight * difference / (scale + difference)
            step = float(1 - confidence[ny, nx]) + penalty
            heapq.heappush(queue, (max(cost, step), distance + 1, label, ny, nx))
    missing = foreground & (labels == 0)
    fallback_count, fallback = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
    labels[missing] = fallback[missing] + count - 1
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Soft embedding growth changed foreground")
    return labels, {"seed_components": count - 1, "unseeded_fallback_components": fallback_count - 1,
                    "unassigned_before_fallback_pixels": int(missing.sum()), "soft_embedding_weight": weight}


def link_instances(probabilities, links, canvas, post, options):
    p, links, canvas = np.asarray(probabilities), np.asarray(links), np.asarray(canvas)
    threshold = options["threshold"]
    if (p.ndim != 3 or p.shape[0] != 2 or links.shape != (8, *p.shape[1:])
            or canvas.shape != p.shape[1:] or canvas.dtype != bool or not 0 < threshold < 1
            or not np.isfinite(p).all() or not np.isfinite(links).all()
            or np.any((p < 0) | (p > 1)) or np.any((links < 0) | (links > 1))):
        raise ValueError("Invalid pixel-link maps")
    foreground = (p[0] >= post["text_threshold"]) & canvas
    h, w = foreground.shape
    labels = np.zeros((h, w), np.int32)
    next_id = 0
    for y, x in np.argwhere(foreground):
        if labels[y, x]:
            continue
        next_id += 1; labels[y, x] = next_id; queue = [(int(y), int(x))]
        while queue:
            yy, xx = queue.pop()
            for direction, (dy, dx) in enumerate(NEIGHBORS):
                ny, nx = yy + dy, xx + dx
                if not (0 <= ny < h and 0 <= nx < w and foreground[ny, nx] and labels[ny, nx] == 0):
                    continue
                # Require reciprocal evidence: a single noisy direction cannot join words.
                if min(links[direction, yy, xx], links[OPPOSITE[direction], ny, nx]) >= threshold:
                    labels[ny, nx] = next_id; queue.append((ny, nx))
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Pixel links changed foreground support")
    return labels, {"link_components": next_id, "foreground_pixels": int(foreground.sum())}


def research_v2_instances(evidence, canvas, config, index=0):
    p = evidence["words"][index].detach().sigmoid().cpu().numpy()
    post = config["postprocessing"]
    if evidence["links"] is not None:
        return link_instances(p, evidence["links"][index].detach().sigmoid().cpu().numpy(), canvas, post, config["links"])
    if evidence["threshold"] is not None:
        low, high = config["adaptive"]["threshold_range"]
        threshold = low + (high - low) * expit(evidence["threshold"][index, 0].detach().cpu().numpy())
        p = p.copy(); p[0] = np.clip(p[0] - threshold + post["text_threshold"], 0, 1)
    if evidence["db_probability"] is not None:
        low, high = config["db"]["threshold_range"]
        threshold = low + (high - low) * expit(evidence["db_threshold"][index, 0].detach().cpu().numpy())
        probability = evidence["db_probability"][index, 0].detach().sigmoid().cpu().numpy()
        p = p.copy(); p[1] = np.clip(probability - threshold + post["kernel_threshold"], 0, 1)
    labels, audit, _ = word_instances(p, canvas, post)
    if evidence["glyph"] is not None:
        labels, association = glyph_word_instances(labels,
            evidence["glyph"][index, 0].detach().sigmoid().cpu().numpy(), canvas, **config["glyph_postprocessing"])
        audit.update(association)
    return labels, audit
