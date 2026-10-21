"""Prompt-specific masks and sampled, class-balanced maximin connectivity loss.

The maximin graph is an explicit local adaptation of MALIS, not its original
all-pairs squared Rand objective. GT is consumed only in these training helpers.
"""

import numpy as np
import torch
from torch.nn import functional as F


# One logit per undirected 8-neighbor edge, stored at its upper/left endpoint.
EDGE_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def graph_edges(shape, valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != tuple(shape) or len(shape) != 2:
        raise ValueError("Invalid graph validity geometry")
    h, w = shape
    nodes = np.arange(h * w).reshape(h, w)
    source, destination, positions = [], [], []
    for channel, (dy, dx) in enumerate(EDGE_OFFSETS):
        yy, xx = np.mgrid[0:h-dy, max(0, -dx):min(w, w-dx)]
        active = valid[yy, xx] & valid[yy+dy, xx+dx]
        yy, xx = yy[active], xx[active]
        source.extend(nodes[yy, xx].tolist())
        destination.extend(nodes[yy+dy, xx+dx].tolist())
        positions.extend((channel * h * w + nodes[yy, xx]).tolist())
    return tuple(np.asarray(x, dtype=np.int64) for x in (source, destination, positions))


def prompt_supervision(owners, valid, prompt, window, *, minimum_majority, minimum_visible):
    """Identify a word from the *predicted kernel*, never choose a window from GT.

    Empty/ambiguous/ignore-only prompts and truncated words have zero supervision.
    Neighboring care words remain valid negatives for the selected word mask.
    """
    owners, valid, prompt = np.asarray(owners), np.asarray(valid, bool), np.asarray(prompt, bool)
    if owners.shape != valid.shape or owners.shape != prompt.shape:
        raise ValueError("Prompt/owner geometry mismatch")
    if not .5 <= minimum_majority <= 1 or not 0 < minimum_visible <= 1:
        raise ValueError("Invalid prompt supervision fractions")
    x1, y1, x2, y2 = window
    if not 0 <= x1 < x2 <= owners.shape[1] or not 0 <= y1 < y2 <= owners.shape[0]:
        raise ValueError("Prompt window outside owner map")
    values = owners[prompt & valid]
    positive = values[values > 0]
    reason, owner = None, 0
    if not len(positive):
        reason = "no_care_owner_under_predicted_kernel"
    else:
        ids, counts = np.unique(positive, return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] / len(values) < minimum_majority or np.count_nonzero(counts == counts[best]) > 1:
            reason = "ambiguous_predicted_kernel"
        else:
            owner = int(ids[best])
    local_valid = valid[y1:y2, x1:x2].copy()
    target = (owners[y1:y2, x1:x2] == owner) & local_valid if owner else np.zeros_like(local_valid)
    if owner and target.sum() / max(1, int(((owners == owner) & valid).sum())) < minimum_visible:
        reason = "selected_word_truncated_by_predicted_window"
    if reason:
        local_valid[:] = False
    return target, local_valid, {"owner_id": owner, "unsupervised_reason": reason}


def mask_objective(logits, target, validity):
    if logits.shape != target.shape or validity.shape != logits.shape:
        raise ValueError("Mask loss shapes differ")
    weight = validity.to(logits.dtype)
    if not weight.any():
        return logits.sum() * 0
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum()
    probability = logits.sigmoid() * weight
    truth = target * weight
    dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    return bce + dice


def balanced_edge_loss(logits, owners, validity):
    if logits.shape != (4, *owners.shape):
        raise ValueError("Affinity output/owner shapes differ")
    u, v, positions = graph_edges(owners.shape, validity)
    flat = np.asarray(owners).ravel()
    # Background/background edges do not dominate the connectivity task.
    keep = (flat[u] > 0) | (flat[v] > 0)
    positions = positions[keep]
    target = (flat[u[keep]] == flat[v[keep]]) & (flat[u[keep]] > 0)
    z = logits.flatten()[torch.as_tensor(positions, device=logits.device)]
    y = torch.as_tensor(target, device=logits.device, dtype=logits.dtype)
    values = []
    for chosen in (y > 0, y == 0):
        if chosen.any():
            values.append(F.binary_cross_entropy_with_logits(z[chosen], y[chosen]))
    return torch.stack(values).mean() if values else logits.sum() * 0


def maximin_weights(logits, owners, validity, *, anchors_per_word):
    """Kruskal maximum spanning forest; count anchor pairs joined by each edge.

    Components may traverse valid background, which carries no endpoint label.
    At most K row-major-spaced anchors per owner have total weight one per word.
    Stable channel/y/x tie order. Edge selection is detached; loss gradients flow
    through the selected current logits, including negative edges inside words.
    """
    if type(anchors_per_word) is not int or anchors_per_word < 2:
        raise ValueError("At least two anchors per word required")
    owners = np.asarray(owners)
    if logits.shape != (4, *owners.shape) or np.any(owners < 0) or not torch.isfinite(logits).all():
        raise ValueError("Invalid maximin graph")
    u, v, positions = graph_edges(owners.shape, validity)
    size = owners.size
    parent = list(range(size))
    ranks = [1] * size
    hist = [{} for _ in range(size)]
    for oid in np.unique(owners[np.asarray(validity, bool) & (owners > 0)]):
        pixels = np.flatnonzero((owners == oid) & np.asarray(validity, bool))
        anchors = pixels[np.linspace(0, len(pixels)-1, min(anchors_per_word, len(pixels)), dtype=int)]
        for pixel in anchors:
            hist[int(pixel)][int(oid)] = 1. / len(anchors)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    values = logits.detach().cpu().numpy().reshape(-1)[positions]
    order = np.argsort(-values, kind="stable")
    positive, negative = np.zeros(len(positions)), np.zeros(len(positions))
    for edge in order:
        a, b = root(int(u[edge])), root(int(v[edge]))
        if a == b:
            continue
        ha, hb = hist[a], hist[b]
        same = sum(count * hb.get(oid, 0.) for oid, count in ha.items())
        positive[edge] = same
        negative[edge] = max(0., sum(ha.values()) * sum(hb.values()) - same)
        if ranks[a] < ranks[b]:
            a, b, ha, hb = b, a, hb, ha
        parent[b] = a
        ranks[a] += ranks[b]
        for oid, count in hb.items():
            ha[oid] = ha.get(oid, 0.) + count
        hist[b] = {}
    selected = (positive > 0) | (negative > 0)
    return positions[selected], positive[selected], negative[selected]


def maximin_loss(logits, owners, validity, *, anchors_per_word):
    positions, positive, negative = maximin_weights(logits, owners, validity,
                                                    anchors_per_word=anchors_per_word)
    z = logits.flatten()[torch.as_tensor(positions, device=logits.device)]
    values = []
    for weights, sign in ((positive, -1), (negative, 1)):
        w = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
        if w.sum() > 0:
            values.append((F.softplus(sign * z) * w).sum() / w.sum())
    return torch.stack(values).mean() if values else logits.sum() * 0
