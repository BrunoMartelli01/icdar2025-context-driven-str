"""Geometry targets, loss and deterministic proposals for a U-Net instance head."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def center_targets(boxes, validity, *, radius_fraction, minimum_radius):
    """Preserve every word center, including boxes smaller than one map pixel."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    valid = torch.as_tensor(validity, dtype=torch.float32)
    if valid.ndim != 2 or not 0 <= radius_fraction <= 1 or minimum_radius < 0:
        raise ValueError("Explicit geometry target policy required")
    h, w = valid.shape
    if len(boxes) and (torch.any(boxes < 0) or torch.any(boxes > 1) or
                       torch.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError("Positive normalized canvas boxes required")
    scale = torch.tensor([w, h], dtype=torch.float32)
    centers = (boxes[:, :2]+boxes[:, 2:])/2*scale
    cells = centers.floor().long()
    indices = cells[:, 1]*w+cells[:, 0]
    if len(torch.unique(indices)) != len(indices):
        raise ValueError("Two word centers collide on the instance grid")
    heatmap = torch.zeros(h, w)
    for box, cell in zip(boxes, cells):
        radius = max(minimum_radius, int(float(((box[2:]-box[:2])*scale).min())*radius_fraction))
        diameter = 2*radius+1
        coordinates = torch.arange(-radius, radius+1, dtype=torch.float32)
        gaussian = torch.exp(-(coordinates[:, None]**2+coordinates[None, :]**2)/
                             (2*(diameter/6)**2))
        x, y = map(int, cell)
        left, right, top, bottom = min(x, radius), min(w-x-1, radius), min(y, radius), min(h-y-1, radius)
        region = heatmap[y-top:y+bottom+1, x-left:x+right+1]
        region.copy_(torch.maximum(region, gaussian[radius-top:radius+bottom+1,
                                                   radius-left:radius+right+1]))
    return dict(heatmap=heatmap, validity=valid, index=indices,
                offset=centers-cells, log_size=(boxes[:, 2:]-boxes[:, :2]).log(),
                boxes=boxes)


def aligned_giou(predicted, truth):
    intersection = (torch.minimum(predicted[:, 2:], truth[:, 2:])-
                    torch.maximum(predicted[:, :2], truth[:, :2])).clamp_min(0).prod(-1)
    pa = (predicted[:, 2:]-predicted[:, :2]).clamp_min(0).prod(-1)
    ta = (truth[:, 2:]-truth[:, :2]).clamp_min(0).prod(-1)
    union = pa+ta-intersection
    enclosing = (torch.maximum(predicted[:, 2:], truth[:, 2:])-
                 torch.minimum(predicted[:, :2], truth[:, :2])).clamp_min(0).prod(-1)
    return intersection/union.clamp_min(1e-9)-(enclosing-union)/enclosing.clamp_min(1e-9)


def dense_instance_loss(output, targets, settings):
    values = []
    diagnostics = dict(center=0., size=0., offset=0., giou=0.)
    for i, target in enumerate(targets):
        device = output['center'].device
        t = {name: value.to(device) for name, value in target.items()}
        logits = output['center'][i, 0]
        probability = logits.sigmoid().clamp(1e-6, 1-1e-6)
        positive = t['heatmap'].eq(1)
        # Care word centers are positive even when their bbox center lies in
        # an ignored concavity; only negative loss is masked by native validity.
        pos = -(probability.log()*(1-probability)**settings['focal_alpha']*positive).sum()
        neg = -((1-probability).log()*probability**settings['focal_alpha']*
                (1-t['heatmap'])**settings['focal_beta']*(~positive)*t['validity']).sum()
        count = len(t['index'])
        center = (pos+neg)/max(count, 1)
        zero = logits.sum()*0
        size = offset = geometry = zero
        if count:
            shape = output['log_size'][i].flatten(1).T[t['index']]
            predicted_offset = output['offset'][i].flatten(1).T[t['index']].sigmoid()
            size = F.smooth_l1_loss(shape, t['log_size'])
            offset = F.l1_loss(predicted_offset, t['offset'])
            h, w = logits.shape
            xy = torch.stack((t['index'] % w, t['index']//w), -1)
            center_xy = (xy+predicted_offset)/torch.tensor([w, h], device=device)
            extent = shape.clamp(*settings['log_size_bounds']).exp()
            boxes = torch.cat((center_xy-extent/2, center_xy+extent/2), -1)
            geometry = (1-aligned_giou(boxes, t['boxes'])).mean()
        terms = dict(center=center, size=size, offset=offset, giou=geometry)
        values.append(sum(settings[name+'_weight']*value for name, value in terms.items()))
        for name, value in terms.items():
            diagnostics[name] += float(value.detach())/len(targets)
    loss = torch.stack(values).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite dense-instance loss")
    return loss, diagnostics


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x2-x1)*max(0., y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def instance_proposals(output, image_wh, *, peak_threshold, maximum_candidates,
                       nms_iou, padding, log_size_bounds):
    """One-image GT-free map -> original RGB exclusive word boxes."""
    if not 0 <= peak_threshold <= 1 or not 0 < nms_iou <= 1 or padding < 0:
        raise ValueError("Explicit valid proposal policy required")
    center = output['center'][0, 0].detach().cpu().sigmoid()
    if not torch.isfinite(center).all():
        raise FloatingPointError("Nonfinite instance probability")
    maxima = F.max_pool2d(center[None, None], 3, stride=1, padding=1)[0, 0]
    indices = torch.where((center == maxima) & (center >= peak_threshold))
    h, w = center.shape
    ordered = sorted(zip(indices[0].tolist(), indices[1].tolist()),
                     key=lambda xy: (-float(center[xy]), xy[0]*w+xy[1]))
    offsets = output['offset'][0].detach().cpu().sigmoid()
    extents = output['log_size'][0].detach().cpu().clamp(*log_size_bounds).exp()
    width, height = image_wh
    side = max(width, height)
    left, top = (side-width)//2, (side-height)//2
    result = []
    for y, x in ordered:
        cx, cy = (x+float(offsets[0, y, x]))/w, (y+float(offsets[1, y, x]))/h
        bw, bh = map(float, extents[:, y, x])
        normalized = [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]
        box = [max(0, math.floor(normalized[0]*side-left)-padding),
               max(0, math.floor(normalized[1]*side-top)-padding),
               min(width, math.ceil(normalized[2]*side-left)+padding),
               min(height, math.ceil(normalized[3]*side-top)+padding)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        if any(box_iou(box, prior['bbox']) > nms_iou for prior in result):
            continue
        result.append(dict(candidate_id='dense:'+str(y*w+x), bbox=box,
            detector_score=float(center[y, x]), center_cell=[x, y],
            normalized_box=normalized, geometry_source='U-Net dense center/extent decoder'))
        if len(result) == maximum_candidates:
            break
    return result
