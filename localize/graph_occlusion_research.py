"""Exploratory component graphs and clean-to-occluded instance distillation.

Inference functions never take annotations. This is an APPROXIMATE adaptation,
not the contour-point GCN or the exact training recipe of Das et al.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from localize.crop_recovery import geodesic_partition, tight_box


def fragment_labels(probabilities, instances, *, threshold, minimum_seed_area,
                    maximum_pieces, maximum_nodes):
    """Partition predicted instances with confident predicted kernel seeds.

    Unseeded components retain one atom. The bounded largest-support rule is
    GT-blind; dropped pixels and split counts are explicitly returned.
    """
    if probabilities.shape != (2, *instances.shape) or not .5 < threshold < 1:
        raise ValueError("Aligned text/kernel maps and explicit seed threshold required")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid probabilities")
    if minimum_seed_area < 1 or maximum_pieces < 2 or maximum_nodes < 1:
        raise ValueError("Invalid fragment capacities")
    fragments = []; split = 0
    for identity in sorted(np.unique(instances[instances > 0])):
        support = instances == identity
        x0, y0, x1, y1 = tight_box(support)
        patch = support[y0:y1, x0:x1]
        count, cores = cv2.connectedComponents(
            (patch & (probabilities[1, y0:y1, x0:x1] >= threshold)).astype(np.uint8), connectivity=8)
        sizes = np.bincount(cores.ravel(), minlength=count)
        ids = [j for j in range(1, count) if sizes[j] >= minimum_seed_area]
        if 2 <= len(ids) <= maximum_pieces:
            seeds = np.zeros_like(cores)
            for j, key in enumerate(ids, 1): seeds[cores == key] = j
            parts = geodesic_partition(probabilities[0, y0:y1, x0:x1], patch, seeds)
            # A disconnected remainder without a confident seed is retained.
            missing = patch & (parts == 0)
            n, rest = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
            parts[missing] = rest[missing] + len(ids)
            for j in range(1, len(ids) + n):
                mask = np.zeros_like(instances, dtype=bool)
                mask[y0:y1, x0:x1] = parts == j
                if mask.any(): fragments.append(mask)
            split += 1
        else:
            fragments.append(support)
    fragments.sort(key=lambda m: (-int(m.sum()), tight_box(m)))
    labels = np.zeros_like(instances, dtype=np.int32)
    for j, mask in enumerate(fragments[:maximum_nodes], 1): labels[mask] = j
    return labels, dict(initial_instances=int(instances.max()), split_parents=split,
                        atoms_before_cap=len(fragments), atoms=min(len(fragments), maximum_nodes),
                        dropped_pixels=sum(int(m.sum()) for m in fragments[maximum_nodes:]))


def neighbor_graph(boxes, *, neighbors, maximum_distance):
    """Symmetric nearest-neighbor graph with height-normalized distances."""
    n = len(boxes)
    adjacency = np.eye(n, dtype=np.float32)
    if n == 0: return adjacency, np.empty((0, 2), dtype=np.int64)
    b = np.asarray(boxes, np.float64)
    centers = (b[:, :2] + b[:, 2:]) / 2
    heights = np.maximum(1, b[:, 3] - b[:, 1])
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    distance /= np.sqrt(heights[:, None] * heights[None])
    for i in range(n):
        chosen = sorted((distance[i, j], j) for j in range(n)
                        if j != i and distance[i, j] <= maximum_distance)[:neighbors]
        for _, j in chosen: adjacency[i, j] = adjacency[j, i] = 1
    edges = np.argwhere(np.triu(adjacency, 1) > 0).astype(np.int64)
    degree = adjacency.sum(1)
    adjacency /= np.sqrt(degree[:, None] * degree[None])
    return adjacency, edges


class FragmentGraph(nn.Module):
    """Equal-parameter graph/self-only controls; symmetric learned edge scores."""
    def __init__(self, dimensions, hidden, layers, message_passing):
        super().__init__()
        self.message_passing = message_passing
        self.project = nn.Linear(dimensions, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden * 2, hidden) for _ in range(layers))
        self.node = nn.Linear(hidden, 1)
        self.edge = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features, adjacency, edges):
        if adjacency.shape != (len(features), len(features)) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Malformed graph")
        h = F.relu(self.project(features))
        for layer in self.layers:
            context = adjacency @ h if self.message_passing else h
            h = h + F.relu(layer(torch.cat((h, context), -1)))
        a, b = h[edges[:, 0]], h[edges[:, 1]]
        return self.node(h).squeeze(-1), self.edge(torch.cat((a + b, (a - b).abs()), -1)).squeeze(-1)


def balanced_binary_loss(logits, targets, valid):
    terms = []
    for positive in (False, True):
        mask = valid & ((targets > .5) == positive)
        if mask.any(): terms.append(F.binary_cross_entropy_with_logits(logits[mask], targets[mask]))
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def graph_supervision(labels, owners, validity, *, purity):
    """Training-only majority-owner assignment; ambiguous/ignored atoms masked."""
    assignments = []; eligible = []
    for j in range(1, int(labels.max()) + 1):
        mask = labels == j
        valid = validity[mask].astype(bool)
        values, counts = np.unique(owners[mask][valid], return_counts=True)
        best = int(values[counts.argmax()]) if len(values) else 0
        fraction = float(counts.max() / mask.sum()) if len(counts) else 0.
        assignments.append(best)
        eligible.append(bool(valid.mean() >= purity and fraction >= purity))
    return np.asarray(assignments, np.int64), np.asarray(eligible, bool)


def graph_groups(boxes, node_scores, edges, edge_scores, *, node_threshold, edge_threshold):
    """GT-blind union-find decoding. Word/line granularity comes from fit labels."""
    parent = list(range(len(boxes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    active = np.asarray(node_scores) >= node_threshold
    for (a, b), score in zip(edges, edge_scores):
        if score >= edge_threshold and active[a] and active[b]:
            a, b = find(int(a)), find(int(b)); parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(boxes)):
        if active[i]: groups.setdefault(find(i), []).append(i)
    return [members for _, members in sorted(groups.items())]


def occlude_instances(images, owners, generator, *, probability, width_fraction, maximum_words):
    """Synthetic partial word occlusion; retain clean word targets explicitly.

    Owners select augmentation regions during training only. RGB and native
    rasters must share the exact square letterbox; nearest mapping is integer.
    """
    if images.ndim != 4 or owners.ndim != 3 or images.shape[0] != owners.shape[0]:
        raise ValueError("Unaligned batched image/owner rasters")
    if not 0 <= probability <= 1 or not 0 < width_fraction < 1:
        raise ValueError("Invalid synthetic occlusion")
    result = images.clone(); covered = torch.zeros_like(owners, dtype=torch.bool)
    owner_cpu = owners.detach().cpu()
    for i, owner in enumerate(owner_cpu):
        ids = sorted(int(v) for v in torch.unique(owner) if v > 0)
        if ids:
            order = torch.randperm(len(ids), generator=generator).tolist()[:maximum_words]
            for j in order:
                if float(torch.rand((), generator=generator)) >= probability: continue
                yy, xx = torch.where(owner == ids[j]); x0, x1 = int(xx.min()), int(xx.max()) + 1
                y0, y1 = int(yy.min()), int(yy.max()) + 1
                width = max(1, round((x1 - x0) * width_fraction))
                left = x0 + int(torch.randint(max(1, x1 - x0 - width + 1), (), generator=generator))
                covered[i, y0:y1, left:left + width] = True
                # Exclusive edge mapping, valid for the fixed 512->256 protocol.
                height, native_width = owner.shape
                a, b = left * images.shape[-1] // native_width, (left + width) * images.shape[-1] // native_width
                c, d = y0 * images.shape[-2] // height, y1 * images.shape[-2] // height
                result[i, :, c:d, a:b] = 0
    return result, covered


def teacher_reliability(teacher, targets, validity, confidence):
    """Only confident teacher pixels agreeing with TRAIN GT may supervise."""
    if (teacher.shape != targets.shape or validity.shape != teacher[:, :1].shape
            or not .5 < confidence < 1):
        raise ValueError("Teacher/GT coordinates differ")
    return (validity.bool().expand_as(teacher) & ((teacher >= .5) == (targets >= .5))
            & ((teacher >= confidence) | (teacher <= 1 - confidence)))


def pixel_distillation(logits, teacher, reliable):
    return balanced_probability_error(logits.sigmoid(), teacher.detach(), reliable)


def balanced_probability_error(student, teacher, reliable):
    terms = []
    for positive in (False, True):
        mask = reliable & ((teacher >= .5) == positive)
        if mask.any(): terms.append((student - teacher).square()[mask].mean())
    return torch.stack(terms).mean() if terms else student.sum() * 0


def instance_structure_distillation(logits, teacher, reliable, owners, separation):
    """Word-wise horizontal/vertical occupancy profiles and separation bands.

    This measures global spatial support per annotated word, NOT learned graph
    links, amodal ground truth, MALIS, or a proven connectivity guarantee.
    All teacher targets are detached and GT-gated before profile aggregation.
    """
    prediction = logits.sigmoid(); teacher = teacher.detach(); terms = []
    for i in range(len(owners)):
        for identity in torch.unique(owners[i]):
            if identity <= 0: continue
            mask = (owners[i] == identity)[None] & reliable[i]
            for axis in (-1, -2):
                count = mask.sum(axis); good = count > 0
                if good.any():
                    a = (prediction[i] * mask).sum(axis) / count.clamp_min(1)
                    b = (teacher[i] * mask).sum(axis) / count.clamp_min(1)
                    terms.append((a - b).square()[good].mean())
    profiles = torch.stack(terms).mean() if terms else logits.sum() * 0
    band = reliable & separation.bool().expand_as(reliable) & (teacher < .5)
    barrier = (prediction - teacher).square()[band].mean() if band.any() else logits.sum() * 0
    return profiles + barrier
