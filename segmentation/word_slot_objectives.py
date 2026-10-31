"""Explicit, independently switchable objectives for the APPROXIMATE slot study."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from models.word_slot_unet import xyxy_to_cxcywh


def pairwise_iou(a, b):
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    union = area_a[:, None] + area_b[None] - intersection
    return intersection / union.clamp_min(1e-7), union


def generalized_iou(a, b):
    iou, union = pairwise_iou(a, b)
    enclosing = (torch.maximum(a[:, None, 2:], b[None, :, 2:]) -
                 torch.minimum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-7)


def matching(boxes, logits, masks, target):
    n = len(target["boxes"])
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return empty, empty
    probability = masks.sigmoid().flatten(1)
    truth = target["masks"].flatten(1)
    dice = (2 * probability @ truth.T + 1) / (probability.sum(1)[:, None] + truth.sum(1)[None] + 1)
    cost = (5 * torch.cdist(boxes, target["boxes"], p=1) -
            2 * generalized_iou(boxes, target["boxes"]) - dice - logits.sigmoid()[:, None])
    left, right = linear_sum_assignment(cost.detach().cpu().numpy())
    return (torch.as_tensor(left, device=boxes.device), torch.as_tensor(right, device=boxes.device))


def instance_loss(boxes, logits, masks, target, *, no_object_weight, mask_weight,
                  word_weights=None, fixed_match=None, mal=False):
    left, right = matching(boxes, logits, masks, target) if fixed_match is None else fixed_match
    labels = torch.zeros_like(logits)
    labels[left] = 1
    weights = torch.full_like(logits, no_object_weight)
    weights[left] = 1 if word_weights is None else word_weights[right]
    if mal and len(left):
        # DEIM-inspired quality target; ordinary Hungarian assignment is retained.
        # This is NOT the original DEIM data/architecture recipe.
        labels[left] = pairwise_iou(boxes[left], target["boxes"][right])[0].diagonal().detach()
    classification = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()
    zero = boxes.sum() * 0 + masks.sum() * 0
    if not len(left):
        return classification, (left, right), {"classification": classification.detach(), "box": zero.detach(), "mask": zero.detach()}
    wp = torch.ones_like(left, dtype=boxes.dtype) if word_weights is None else word_weights[right]
    # Background weights are never multiplied by specialist advantage weights.
    l1 = (boxes[left] - target["boxes"][right]).abs().sum(-1)
    giou = 1 - generalized_iou(boxes[left], target["boxes"][right]).diagonal()
    box_loss = ((5 * l1 + 2 * giou) * wp).mean()
    probability, truth = masks[left].sigmoid(), target["masks"][right]
    valid = target["validity"][None]
    bce = (F.binary_cross_entropy_with_logits(masks[left], truth, reduction="none") * valid).sum((-2, -1)) / valid.sum().clamp_min(1)
    dice = 1 - (2 * (probability * truth * valid).sum((-2, -1)) + 1) / ((probability * valid).sum((-2, -1)) + (truth * valid).sum((-2, -1)) + 1)
    mask_loss = ((bce + dice) * wp).mean()
    loss = classification + box_loss + mask_weight * mask_loss
    return loss, (left, right), {"classification": classification.detach(), "box": box_loss.detach(), "mask": mask_loss.detach()}


def make_denoising(target, *, groups, noise, background_queries, generator):
    boxes = xyxy_to_cxcywh(target["boxes"])
    refs, ids, destinations = [], [], []
    for group in range(groups):
        perturb = (torch.rand(boxes.shape, generator=generator) * 2 - 1).to(boxes.device)
        scale = torch.cat((boxes[:, 2:], boxes[:, 2:]), -1)
        refs.append((boxes + noise * perturb * scale).clamp(.01, .99))
        ids.extend([group] * len(boxes)); destinations.extend(range(len(boxes)))
    if background_queries:
        # Background is supervised as background, never assigned a fabricated word.
        bg = torch.rand((background_queries, 4), generator=generator).to(boxes.device)
        bg[:, 2:] = bg[:, 2:] * .15 + .025
        if len(boxes):
            # Mark overlapping candidates as unsupervised instead of false negatives.
            from models.word_slot_unet import cxcywh_to_xyxy
            overlaps = pairwise_iou(cxcywh_to_xyxy(bg).clamp(0, 1), target["boxes"])[0]
            bg = bg[overlaps.max(1).values < .05]
        refs.append(bg); ids.extend([groups] * len(bg)); destinations.extend([-1] * len(bg))
    return {"references": torch.cat(refs) if refs else boxes[:0], "groups": ids,
            "destinations": torch.tensor(destinations, device=boxes.device, dtype=torch.long)}


def denoising_loss(output, target, dn, **kwargs):
    start = output["ordinary_count"]
    destinations = dn["destinations"]
    left = torch.where(destinations >= 0)[0]
    return instance_loss(output["all_boxes"][start:], output["all_presence"][start:],
                         F.interpolate(output["all_masks"][None, start:], target["masks"].shape[-2:], mode="nearest")[0],
                         target, fixed_match=(left, destinations[left]), **kwargs)[0]


def strip_regularization(grid, minimum_step):
    dx = grid[:, :, 1:, 0] - grid[:, :, :-1, 0]
    dy = grid[:, 1:, :, 1] - grid[:, :-1, :, 1]
    folding = F.relu(minimum_step - dx).mean() + F.relu(minimum_step - dy).mean()
    bending = (grid[:, :, 2:] - 2 * grid[:, :, 1:-1] + grid[:, :, :-2]).square().mean()
    outside = F.relu(grid.abs() - 1).square().mean()
    return folding + bending + outside


def pcgrad_two(loss_a, loss_b, parameters):
    """Two-task PCGrad, symmetric projection against ORIGINAL task gradients.

    Parameters belonging to only one task keep that task's gradient. Gradients
    are summed (the control also sums its task objectives). Returns a measured
    cosine and conflict flag; no projection is performed for nonnegative dots.
    """
    parameters = list(parameters)
    ga = torch.autograd.grad(loss_a, parameters, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, parameters, allow_unused=True)
    shared = [(a, b) for a, b in zip(ga, gb) if a is not None and b is not None]
    if not shared:
        for p, a, b in zip(parameters, ga, gb):
            p.grad = a if b is None else b if a is None else a + b
        return {"cosine": None, "conflict": False}
    dot = sum((a * b).sum() for a, b in shared)
    na, nb = sum(a.square().sum() for a, _ in shared), sum(b.square().sum() for _, b in shared)
    conflict = bool(dot < 0)
    for p, a, b in zip(parameters, ga, gb):
        if a is None or b is None:
            p.grad = a if b is None else b
        else:
            p.grad = a + b if not conflict else a - dot / nb.clamp_min(1e-12) * b + b - dot / na.clamp_min(1e-12) * a
    return {"cosine": float(dot / (na * nb).sqrt().clamp_min(1e-12)), "conflict": conflict}


def character_contrastive(student, teacher, labels, temperature):
    """Aligned positives; another occurrence of the same character is not negative."""
    if len(student) == 0:
        return student.sum() * 0
    student, teacher = F.normalize(student, dim=-1), F.normalize(teacher.detach(), dim=-1)
    logits = student @ teacher.T / temperature
    same = labels[:, None] == labels[None]
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, -torch.inf)
    return F.cross_entropy(logits, torch.arange(len(student), device=student.device))
