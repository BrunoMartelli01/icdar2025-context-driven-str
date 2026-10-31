"""Explicit APPROXIMATE teacher/student losses and set selection primitives."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from localize.occlusion_four_tests import iou


@torch.no_grad()
def ema_update(teacher, student, decay):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    target, source = teacher.state_dict(), student.state_dict()
    if target.keys() != source.keys():
        raise ValueError("EMA state keys differ")
    for key, value in target.items():
        other = source[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("EMA state shape/dtype differs: " + key)
        if value.is_floating_point():
            value.lerp_(other, 1 - decay)
        else:
            value.copy_(other)


def confidence_consistency(logits, targets, valid, confidence):
    if logits.shape != targets.shape or valid.shape != logits[:, :1].shape:
        raise ValueError("Consistency maps are not aligned")
    if not .5 < confidence < 1 or not torch.isfinite(targets).all():
        raise ValueError("Invalid teacher confidence/targets")
    targets = targets.detach()
    mask = valid.expand_as(targets).bool() & ((targets >= confidence) | (targets <= 1-confidence))
    error = (logits.sigmoid() - targets).square()
    # Balance foreground/background so empty background cannot dominate.
    terms = []
    for positive in (True, False):
        selected = mask & ((targets >= .5) == positive)
        if selected.any():
            terms.append(error[selected].mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def token_distillation(student_logits, teacher_logits, target, pad_id, temperature):
    if student_logits.shape != teacher_logits.shape or target.shape != student_logits.shape[:2]:
        raise ValueError("Token distributions are not position-aligned")
    if temperature <= 0:
        raise ValueError("Positive distillation temperature required")
    valid = target != pad_id
    loss = F.kl_div(F.log_softmax(student_logits/temperature, -1),
                    F.softmax(teacher_logits.detach()/temperature, -1), reduction="none").sum(-1)
    return loss[valid].mean()*temperature**2 if valid.any() else student_logits.sum()*0


def occlude_words(images, generator, probability, width_fraction):
    """Partial vertical RGB occlusion in normalized PARSeq crops; labels unchanged."""
    if images.ndim != 4 or not 0 <= probability <= 1 or not 0 < width_fraction < 1:
        raise ValueError("Invalid crop occlusion contract")
    result = images.clone()
    width = images.shape[-1]
    covered = max(1, round(width*width_fraction))
    for i in range(len(images)):
        if torch.rand((), generator=generator).item() < probability:
            x = int(torch.randint(width-covered+1, (), generator=generator))
            result[i, :, :, x:x+covered] = 0
    return result


class CandidateScorer(nn.Module):
    def __init__(self, dimensions, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def duplicate_pairs(pool, assignments, threshold):
    """Training-only same-GT competing readings; GT never enters inference."""
    pairs = []
    for i, a in enumerate(pool):
        if assignments[i] < 0:
            continue
        for j in range(i):
            if assignments[j] == assignments[i] and iou(a["bbox"], pool[j]["bbox"]) >= threshold:
                pairs.append((j, i))
    return pairs


def set_loss(logits, targets, assignments, duplicate_weight, coverage_weight):
    """BCE plus one-GT coverage and duplicate occupancy; per-image objective."""
    if logits.shape != targets.shape or len(assignments) != len(logits):
        raise ValueError("Candidate supervision shape mismatch")
    if not len(logits):
        return logits.sum()*0
    probability = logits.sigmoid()
    terms = []
    for identity in sorted(set(assignments)-{-1}):
        ids = [i for i, value in enumerate(assignments) if value == identity and targets[i] > 0]
        if not ids:
            continue
        p = probability[ids]
        coverage = -torch.log((1-torch.prod(1-p)).clamp_min(1e-6))
        duplicates = (p.sum()-1).clamp_min(0).square()
        terms.append(coverage_weight*coverage + duplicate_weight*duplicates)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss + (torch.stack(terms).mean() if terms else logits.sum()*0)


def grouped_set_loss(logits, targets, groups, group_count, duplicate_weight, coverage_weight):
    """Vectorized whole-fit equivalent, with disjoint positive GT-group IDs."""
    base = F.binary_cross_entropy_with_logits(logits, targets)
    selected = groups >= 0
    if not selected.any():
        return base
    p = logits[selected].sigmoid()
    ids = groups[selected]
    sums = p.new_zeros(group_count).scatter_add(0, ids, p)
    log_absence = p.new_zeros(group_count).scatter_add(0, ids, torch.log1p(-p.clamp_max(1-1e-6)))
    present = p.new_zeros(group_count).scatter_add(0, ids, torch.ones_like(p)) > 0
    coverage = -torch.log((-torch.expm1(log_absence[present])).clamp_min(1e-6)).mean()
    duplicate = (sums[present]-1).clamp_min(0).square().mean()
    return base + coverage_weight*coverage + duplicate_weight*duplicate


def compatible_selection(pool, scores, threshold, overlap):
    """Greedy compatible set; conflict means duplicate boxes or shared parents."""
    accepted = []
    for p in sorted(pool, key=lambda p: (-scores[p["id"]], p["id"])):
        if scores[p["id"]] < threshold:
            continue
        origins = set(p["parents"]) or {p["id"]}
        conflict = False
        for other in accepted:
            other_origins = set(other["parents"]) or {other["id"]}
            # Sibling split pieces may coexist; parent versus children may not.
            family_conflict = bool(origins & other_origins) and (
                p["method"] != "split" or other["method"] != "split" or p["parents"] != other["parents"])
            if family_conflict or iou(p["bbox"], other["bbox"]) >= overlap:
                conflict = True
                break
        if not conflict:
            accepted.append(p)
    return accepted


def teacher_matches(clean, alternate, readings, overlap, minimum_confidence):
    """Accept stable geometry AND nonempty exact text, with confidence in both views."""
    accepted = []
    used = set()
    def confidence(reading):
        values = reading.get('token_probabilities', [])
        return float(np.mean(values)) if values else 0.
    for p in clean:
        a = readings[p["crop_array_sha256"]]
        if not a["raw_text"] or confidence(a) < minimum_confidence:
            continue
        choices = []
        for j, q in enumerate(alternate):
            b = readings[q["crop_array_sha256"]]
            score = iou(p["bbox"], q["bbox"])
            if j not in used and score >= overlap and a["raw_text"].casefold() == b["raw_text"].casefold() and confidence(b) >= minimum_confidence:
                choices.append((score, -j, j))
        if choices:
            chosen = max(choices)[2]
            used.add(chosen)
            accepted.append({**p, "pseudo_text": a["raw_text"], "agreement_bbox": alternate[chosen]["bbox"]})
    return accepted
