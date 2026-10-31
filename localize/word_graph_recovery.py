"""APPROXIMATE word hypotheses, non-destructive admission and crop supervision.

Annotations enter only the explicitly named training-target functions. Inference
preserves original observations, including their ordering, text and geometry.
"""
from __future__ import annotations

import copy
import hashlib
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from localize.occlusion_four_tests import iou
from scripts.run_mask_crop_budget import measure


def conflicts(a, b, overlap):
    """Same sibling/family and IoU rule as the qualified compatible selector."""
    x = set(a.get('parents', [])) or {a['id']}
    y = set(b.get('parents', [])) or {b['id']}
    family = bool(x & y) and (a['method'] != 'split' or b['method'] != 'split' or a.get('parents') != b.get('parents'))
    return family or iou(a['bbox'], b['bbox']) >= overlap


def addition_features(proposals, baseline, features, readings, visual, *, text_bins, overlap):
    """Character n-grams, actual visual evidence and current accepted-set context.

    No lexicon/GT is consulted. Stable hashing does not learn from held-out text.
    Visual rows must follow the caller's already-budgeted candidate ordering.
    """
    if text_bins < 1 or len(visual) != len(proposals):
        raise ValueError('Invalid hypothesis feature contract')
    accepted = {p['id'] for p in baseline}
    values = []
    for p, image in zip(proposals, visual):
        r = readings[p['crop_array_sha256']]
        text = r['raw_text'].casefold()
        grams = np.zeros(text_bins, np.float32)
        marked = '^' + text + '$'
        for n in (1, 2, 3):
            for j in range(len(marked) - n + 1):
                h = int.from_bytes(hashlib.sha256(marked[j:j+n].encode('utf-8')).digest()[:8], 'big')
                grams[h % text_bins] += 1
        grams /= max(1., float(np.linalg.norm(grams)))
        confidence = r.get('token_probabilities', [])
        context = [float(p['id'] in accepted),
                   max((iou(p['bbox'], q['bbox']) for q in baseline), default=0.),
                   float(any(conflicts(p, q, overlap) for q in baseline)),
                   float(any(readings[q['crop_array_sha256']]['raw_text'].casefold() == text for q in baseline)),
                   len(text) / 25., float(np.mean(confidence)) if confidence else 0.,
                   float(np.min(confidence)) if confidence else 0.]
        values.append(list(features[p['id']]) + list(image) + context + grams.tolist())
    return np.asarray(values, np.float32)


def hypothesis_adjacency(proposals, readings, *, neighbors, maximum_distance):
    """Pair relations describe overlap/parent/text competition, never a hard merge."""
    if neighbors < 1 or maximum_distance <= 0:
        raise ValueError('Explicit relation limits required')
    n = len(proposals)
    matrix = np.eye(n, dtype=np.float32)
    for j, p in enumerate(proposals):
        b = p['bbox']; center = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
        choices = []
        for k, q in enumerate(proposals):
            if k == j:
                continue
            d = q['bbox']; other = np.array([(d[0]+d[2])/2, (d[1]+d[3])/2])
            distance = np.linalg.norm(center-other) / math.sqrt(max(1, b[3]-b[1]) * max(1, d[3]-d[1]))
            same = readings[p['crop_array_sha256']]['raw_text'].casefold() == readings[q['crop_array_sha256']]['raw_text'].casefold()
            related = set(p.get('parents', [])) & set(q.get('parents', []))
            if distance <= maximum_distance or iou(b, d) > 0 or related or same:
                choices.append((float(distance), k))
        for _, k in sorted(choices)[:neighbors]:
            matrix[j, k] = matrix[k, j] = 1
    degree = matrix.sum(1)
    return matrix / np.sqrt(degree[:, None]*degree[None])


class HypothesisScorer(nn.Module):
    """Self-only and message-passing controls have identical tensors/capacity."""
    def __init__(self, dimensions, hidden, layers, *, message_passing):
        super().__init__()
        self.message_passing = message_passing
        self.project = nn.Linear(dimensions, hidden)
        self.layers = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(layers))
        self.output = nn.Linear(hidden, 1)

    def forward(self, features, adjacency):
        if adjacency.shape != (len(features), len(features)):
            raise ValueError('Misaligned word graph')
        x = F.relu(self.project(features))
        for layer in self.layers:
            context = adjacency @ x if self.message_passing else x
            x = x + F.relu(layer(torch.cat((x, context), -1)))
        return self.output(x).squeeze(-1)


def preserve_and_add(baseline, candidates, scores, *, threshold, overlap):
    """Lock EVERY baseline output; accepted alternatives cannot rerun global NMS."""
    if not 0 <= threshold <= 1 or not 0 < overlap <= 1:
        raise ValueError('Invalid admission policy')
    if len({p['id'] for p in candidates}) != len(candidates):
        raise ValueError('Candidate IDs must be unique')
    ids = {p['id'] for p in baseline}
    if len(ids) != len(baseline) or not ids <= {p['id'] for p in candidates}:
        raise ValueError('Baseline must be a unique subset of admitted candidates')
    result = copy.deepcopy(baseline)
    for p in sorted(candidates, key=lambda q: (-scores[q['id']], q['id'])):
        score = scores[p['id']]
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid admission score')
        if p['id'] not in ids and score >= threshold and not any(conflicts(p, q, overlap) for q in result):
            result.append(copy.deepcopy(p)); ids.add(p['id'])
    if result[:len(baseline)] != baseline:
        raise RuntimeError('Baseline observations changed')
    return result


def marginal_training_targets(baseline, candidates, words, readings, *, overlap):
    """FIT-only labels: exact maximum matching gain conditional on fixed baseline.

    Each positive can still compete with another positive. Admission resolves
    conflicts; diagnostics retain this conditional-target limitation explicitly.
    """
    old = measure(baseline, words, readings)['correct']
    ids = {p['id'] for p in baseline}
    targets, valid = [], []
    for p in candidates:
        eligible = p['id'] not in ids and not any(conflicts(p, q, overlap) for q in baseline)
        gain = measure(baseline+[p], words, readings)['correct'] - old if eligible else 0
        if gain not in (0, 1):
            raise RuntimeError('Invalid incremental matching gain')
        targets.append(float(gain)); valid.append(eligible)
    return np.asarray(targets, np.float32), np.asarray(valid, bool)


def training_owner_targets(boxes, owners, validity, *, minimum_iou, minimum_valid_fraction):
    """Assign FIT predictions to owners; all four extrema use the FULL owner mask."""
    valid, identities, extents = [], [], []
    for box in np.asarray(boxes, int):
        x0, y0, x1, y1 = box
        patch = owners[y0:y1, x0:x1]; ok = validity[y0:y1, x0:x1].astype(bool)
        ids, counts = np.unique(patch[ok & (patch > 0)], return_counts=True)
        identity = int(ids[counts.argmax()]) if len(ids) else 0
        yy, xx = np.where(owners == identity) if identity else ([], [])
        target = [int(np.min(xx)), int(np.min(yy)), int(np.max(xx))+1, int(np.max(yy))+1] if len(xx) else list(box)
        good = bool(identity and ok.size and ok.mean() >= minimum_valid_fraction and iou(box.tolist(), target) >= minimum_iou)
        valid.append(good); identities.append(identity if good else 0); extents.append(target)
    return np.asarray(extents, np.float32).reshape(-1, 4), np.asarray(identities, np.int64), np.asarray(valid, bool)


def point_extents(points):
    """Differentiable exclusive pixel-cell extents, consistent with crop decoding."""
    return torch.cat((points.amin(1)-.5, points.amax(1)+.5), -1)


def complete_word_loss(points, source_boxes, target_boxes, owner_ids, owners, valid, *, inside_weight, extent_weight, contamination_weight):
    """FIT-only asymmetric clipping/extent/other-word coverage objective.

    Target extrema come from complete owner masks, not resampled contour extrema.
    Other-word exclusion uses masks only while training, never at inference.
    """
    if points.shape[0] != len(source_boxes) or target_boxes.shape != source_boxes.shape or valid.shape != (len(points),):
        raise ValueError('Unaligned completeness targets')
    if min(inside_weight, extent_weight, contamination_weight) < 0:
        raise ValueError('Negative loss weights')
    if not bool(valid.any()):
        return points.sum()*0
    pred = point_extents(points); scale = (source_boxes[:, 2:]-source_boxes[:, :2]).clamp_min(1)
    full_scale = torch.cat((scale, scale), 1)
    error = (pred-target_boxes)/full_scale
    extent = F.smooth_l1_loss(pred[valid]/full_scale[valid], target_boxes[valid]/full_scale[valid])
    clipping = torch.cat((error[:, :2].relu(), (-error[:, 2:]).relu()), 1)[valid].mean()
    terms = []
    # A soft rectangle supports useful gradients at a source word's boundaries.
    for j in torch.where(valid)[0].tolist():
        other = (owners > 0) & (owners != owner_ids[j])
        yy, xx = torch.where(other)
        if len(xx):
            b = pred[j]
            probability = torch.sigmoid(xx+.5-b[0])*torch.sigmoid(b[2]-xx-.5)
            probability = probability*torch.sigmoid(yy+.5-b[1])*torch.sigmoid(b[3]-yy-.5)
            terms.append(probability.sum()/((target_boxes[j, 2:]-target_boxes[j, :2]).prod().clamp_min(1)))
    contamination = torch.stack(terms).mean() if terms else points.sum()*0
    return extent_weight*extent + inside_weight*clipping + contamination_weight*contamination


def geometry_targets(owners):
    """Explicit adaptation: per-owner normalized interior distance and unit vector
    toward its nearest exterior pixel. Background direction is UNSUPERVISED.
    """
    owners = np.asarray(owners)
    if owners.ndim != 2 or np.any(owners < 0):
        raise ValueError('Nonnegative owner raster required')
    target = np.zeros((3, *owners.shape), np.float32)
    for identity in np.unique(owners[owners > 0]):
        mask = np.pad((owners == identity).astype(np.uint8), 1)
        distance, labels = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        yy, xx = np.where(mask == 0)
        table = np.zeros((int(labels.max())+1, 2), np.float32)
        table[labels[yy, xx]] = np.stack((xx, yy), 1)
        y, x = np.where(mask > 0)
        vector = table[labels[y, x]] - np.stack((x, y), 1)
        vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-6)
        target[0, y-1, x-1] = distance[y, x]/max(float(distance.max()), 1e-6)
        target[1:, y-1, x-1] = vector.T
    return target


def dense_geometry_loss(raw, target, foreground, validity, *, magnitude_weight):
    valid = validity.bool().expand_as(raw[:, :1])
    fg = valid & foreground.bool()
    distance = F.smooth_l1_loss(raw[:, :1].sigmoid()[valid], target[:, :1][valid]) if valid.any() else raw.sum()*0
    if fg.any():
        vector = F.normalize(raw[:, 1:], dim=1, eps=1e-6)
        direction = (1-(vector*target[:, 1:]).sum(1, keepdim=True))[fg].mean()
        magnitude = (raw[:, 1:].square().sum(1, keepdim=True).clamp_min(1e-12).sqrt().sub(1).square())[fg].mean()
    else:
        direction = magnitude = raw.sum()*0
    return distance + direction + magnitude_weight*magnitude


def probability_extents(probability, box, *, temperature):
    """Soft extrema within a fixed accepted teacher window; no GT at forward time."""
    x0,y0,x1,y1 = map(int, box)
    patch = probability[y0:y1, x0:x1]
    if not patch.numel() or temperature <= 0:
        raise ValueError('Invalid regional extent window')
    yy, xx = torch.meshgrid(torch.arange(y0,y1,device=patch.device,dtype=patch.dtype)+.5,
                            torch.arange(x0,x1,device=patch.device,dtype=patch.dtype)+.5,indexing='ij')
    weight = patch.clamp_min(1e-6).log().flatten()
    answer=[]
    for coord, sign in ((xx,-1),(yy,-1),(xx,1),(yy,1)):
        v=coord.flatten(); answer.append((torch.softmax(weight+sign*v/temperature,0)*v).sum())
    return torch.stack(answer)


def regional_distillation(logits, teacher, regions, *, temperature, extent_weight):
    """Word-instance Dice + soft extrema; unknown/unaccepted regions add NO loss.

    Regions were verified on FIT clean crops. Teacher targets are detached even
    when a caller accidentally supplies a tensor requiring gradients.
    """
    if logits.shape != teacher.shape or len(regions) != len(logits):
        raise ValueError('Unaligned regional teacher')
    teacher=teacher.detach(); student=logits.sigmoid();terms=[]
    for i, words in enumerate(regions):
        for word in words:
            box=word['bbox_native'];x0,y0,x1,y1=map(int,box)
            a=student[i,:,y0:y1,x0:x1]; b=teacher[i,:,y0:y1,x0:x1]
            dice=1-(2*(a*b).sum()+1)/(a.square().sum()+b.square().sum()+1)
            pa=probability_extents(student[i,0],box,temperature=temperature)
            pb=probability_extents(teacher[i,0],box,temperature=temperature)
            scale=logits.new_tensor([max(1,x1-x0),max(1,y1-y0)]*2)
            terms.append(dice+extent_weight*F.smooth_l1_loss(pa/scale,pb/scale))
    return torch.stack(terms).mean() if terms else logits.sum()*0
