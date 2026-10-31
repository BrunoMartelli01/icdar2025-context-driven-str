"""Dense box targets and word-balanced loss; primary recognition GT is unchanged."""

import math
import numpy as np
import torch
from torch.nn import functional as F

from segmentation.unet_dense_instances import aligned_giou, box_iou
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask


def word_box_targets(boxes, validity, *, core_fraction):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    valid = torch.as_tensor(validity, dtype=torch.float32)
    if valid.ndim != 2 or not 0 < core_fraction <= 1:
        raise ValueError('Two-dimensional validity and positive explicit core fraction required')
    if len(boxes) and (not torch.isfinite(boxes).all() or torch.any(boxes < 0) or
                      torch.any(boxes > 1) or torch.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('Positive original normalized boxes required')
    h, w = valid.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    points = torch.stack(((xx+.5)/w, (yy+.5)/h), -1)
    owners = torch.zeros(h, w, dtype=torch.long)
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    # Smaller boxes win ordinary overlaps. Unique forced centers preserve all
    # words without changing the original regression box or metric denominator.
    for index in sorted(range(len(boxes)), key=lambda i: (
            -float((boxes[i, 2:]-boxes[i, :2]).prod()), i)):
        extent = (boxes[index, 2:]-boxes[index, :2])*core_fraction/2
        core = ((points >= centers[index]-extent) &
                (points <= centers[index]+extent)).all(-1)
        owners[core] = index+1
    cells = (centers*torch.tensor([w, h])).floor().long()
    if len({tuple(cell) for cell in cells.tolist()}) != len(boxes):
        raise ValueError('Word centers collide on the field grid')
    for index, (x, y) in enumerate(cells.tolist()):
        owners[y, x] = index+1
    indices = torch.where(owners.flatten() > 0)[0]
    labels = owners.flatten()[indices]
    assigned = boxes[labels-1]
    anchors = points.reshape(-1, 2)[indices].clone()
    if len(anchors):
        outside = ((anchors <= assigned[:, :2]) | (anchors >= assigned[:, 2:])).any(-1)
        anchors[outside] = (assigned[outside, :2]+assigned[outside, 2:])/2
    xy = torch.stack((indices % w, indices//w), -1)
    offsets = anchors*torch.tensor([w, h])-xy
    distances = torch.cat((anchors-assigned[:, :2], assigned[:, 2:]-anchors), -1)
    assert not len(distances) or torch.all(distances > 0)
    counts = torch.bincount(labels, minlength=len(boxes)+1)[1:]
    assert len(counts) == len(boxes) and torch.all(counts > 0)
    weights = counts[labels-1].float().reciprocal()
    centerness = ((torch.minimum(distances[:, 0], distances[:, 2])/
                  torch.maximum(distances[:, 0], distances[:, 2]))*
                 (torch.minimum(distances[:, 1], distances[:, 3])/
                  torch.maximum(distances[:, 1], distances[:, 3]))).sqrt()
    return dict(owners=owners, validity=valid, indices=indices, offsets=offsets,
        distances=distances, centerness=centerness, boxes=assigned, weights=weights,
        represented_words=torch.tensor(len(boxes), dtype=torch.long))


def word_box_loss(output, targets, settings):
    values, diagnostics = [], dict(classification=0., centerness=0., giou=0., offset=0.)
    for i, target in enumerate(targets):
        device = output['class_logits'].device
        t = {name: value.to(device) for name, value in target.items()}
        logits = output['class_logits'][i, 0]
        positive = t['owners'] > 0
        y = positive.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        probability = logits.sigmoid()
        pt = torch.where(positive, probability, 1-probability)
        alpha = torch.where(positive, settings['focal_alpha'], 1-settings['focal_alpha'])
        focal = alpha*(1-pt).pow(settings['focal_gamma'])*bce
        n = len(t['indices'])
        zero = logits.sum()*0
        positive_loss = zero
        center = geometry = offset = zero
        if n:
            weights = t['weights']
            positive_loss = (focal.flatten()[t['indices']]*weights).sum()/weights.sum()
            center = (F.binary_cross_entropy_with_logits(
                output['center_logits'][i, 0].flatten()[t['indices']], t['centerness'],
                reduction='none')*weights).sum()/weights.sum()
            predicted_offset = output['offset_logits'][i].flatten(1).T[t['indices']].sigmoid()
            offset = (F.smooth_l1_loss(predicted_offset, t['offsets'],
                reduction='none').mean(-1)*weights).sum()/weights.sum()
            distance = F.softplus(output['raw_distances'][i].flatten(1).T[t['indices']])
            h, w = logits.shape
            cells = torch.stack((t['indices'] % w, t['indices']//w), -1)
            anchors = (cells+predicted_offset)/torch.tensor([w, h], device=device)
            boxes = torch.cat((anchors-distance[:, :2], anchors+distance[:, 2:]), -1)
            rw = weights*t['centerness']
            geometry = ((1-aligned_giou(boxes, t['boxes']))*rw).sum()/rw.sum()
        negative_loss = (focal*(~positive)*t['validity']).sum()/max(n, 1)
        terms = dict(classification=positive_loss+negative_loss,
                     centerness=center, giou=geometry, offset=offset)
        values.append(sum(settings[name+'_weight']*value for name, value in terms.items()))
        for name, value in terms.items():
            diagnostics[name] += float(value.detach())/len(targets)
    loss = torch.stack(values).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite word-box field loss')
    return loss, diagnostics


def word_box_proposals(output, image_wh, settings):
    for name in ('class_logits', 'center_logits', 'raw_distances', 'offset_logits'):
        if not torch.isfinite(output[name]).all():
            raise FloatingPointError('Nonfinite field prediction: '+name)
    score = (output['class_logits'][0, 0].detach().cpu().sigmoid()*
             output['center_logits'][0, 0].detach().cpu().sigmoid()).sqrt()
    h, w = score.shape
    if h != w:
        raise ValueError('Explicit square letterbox field required')
    canvas = torch.as_tensor(np.asarray(letterbox_canvas_mask(tuple(image_wh),
                                    native_size=h), dtype=bool))
    score = score*canvas
    maxima = F.max_pool2d(score[None, None], 3, stride=1, padding=1)[0, 0]
    yy, xx = torch.where(canvas & (score == maxima) & (score >= settings['score_threshold']))
    ordered = sorted(zip(yy.tolist(), xx.tolist()), key=lambda xy: (
        -float(score[xy]), xy[0]*w+xy[1]))
    offsets = output['offset_logits'][0].detach().cpu().sigmoid()
    distances = F.softplus(output['raw_distances'][0].detach().cpu()).clamp(
        max=settings['maximum_distance'])
    width, height = image_wh
    side = max(width, height)
    left, top = (side-width)//2, (side-height)//2
    rows = []
    for y, x in ordered:
        anchor = [(x+float(offsets[0, y, x]))/w, (y+float(offsets[1, y, x]))/h]
        l, t, r, b = map(float, distances[:, y, x])
        normalized = [anchor[0]-l, anchor[1]-t, anchor[0]+r, anchor[1]+b]
        padding = settings['padding']
        box = [max(0, math.floor(normalized[0]*side-left)-padding),
               max(0, math.floor(normalized[1]*side-top)-padding),
               min(width, math.ceil(normalized[2]*side-left)+padding),
               min(height, math.ceil(normalized[3]*side-top)+padding)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        if any(box_iou(box, row['bbox']) > settings['nms_iou'] for row in rows):
            continue
        rows.append(dict(candidate_id='field:'+str(y*w+x), instance_id=y*w+x,
            bbox=box, detector_score=float(score[y, x]), field_cell=[x, y],
            normalized_box=normalized, geometry_source='U-Net per-pixel word boundary field'))
        if len(rows) == settings['maximum_candidates']:
            break
    return rows
