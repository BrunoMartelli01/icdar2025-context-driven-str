"""APPROXIMATE word hypotheses with typed relations and visual ownership.

Inference functions deliberately accept no ground truth. Existing observations
are preserved by the independently qualified admission function.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from localize.word_graph_recovery import conflicts


def normalize_edges(matrix, *, self_edges=False):
    matrix = np.asarray(matrix, np.float32).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError('Nonnegative square adjacency required')
    np.fill_diagonal(matrix, 1. if self_edges else 0.)
    degree = matrix.sum(1).clip(1.)
    return matrix / np.sqrt(degree[:, None] * degree[None, :])


def relation_edges(pool, adjacency, *, overlap):
    """Separate incompatible hypotheses from the qualified proximity graph."""
    n = len(pool)
    if adjacency.shape != (n, n):
        raise ValueError('Unaligned graph')
    conflict = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i):
            conflict[i, j] = conflict[j, i] = float(conflicts(pool[i], pool[j], overlap))
    support = (adjacency > 0).astype(np.float32) * (1-conflict)
    collapsed = normalize_edges(np.maximum(support, conflict), self_edges=True)
    return normalize_edges(support, self_edges=True), normalize_edges(conflict), collapsed


def visual_regions(probabilities, canvas, pool, baseline, *, tile_size, text_threshold,
                   kernel_threshold, minimum_pixels, maximum_regions):
    """Disjoint predicted text tiles, plus kernel/mass/accepted-set evidence.

    The tile partition bounds memory and avoids pretending that a connected
    component is necessarily a complete character or a complete word. Incidence
    uses actual predicted foreground pixels, not rectangle intersection area.
    Ownership is soft, observable coverage by accepted hypotheses, never GT.
    """
    probability = np.asarray(probabilities)
    if probability.ndim != 3 or probability.shape[0] != 2 or probability.shape[1:] != canvas.shape:
        raise ValueError('Aligned text/kernel/canvas required')
    if tile_size < 1 or maximum_regions < 1 or minimum_pixels < 1:
        raise ValueError('Invalid region partition')
    h, w = canvas.shape
    foreground = (probability[0] >= text_threshold) & canvas.astype(bool)
    accepted = {p['id'] for p in baseline}
    masks = []
    for p in pool:
        a, b, d, e = map(int, p['bbox_native'])
        mask = np.zeros((h, w), bool)
        mask[max(0,b):min(h,e), max(0,a):min(w,d)] = True
        masks.append(mask)
    candidates = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            yy, xx = np.where(foreground[y:y+tile_size, x:x+tile_size])
            if len(xx) >= minimum_pixels:
                yy, xx = yy+y, xx+x
                candidates.append((float(probability[0, yy, xx].sum()), y, x, yy, xx))
    candidates.sort(key=lambda z: (-z[0], z[1], z[2]))
    candidates = candidates[:maximum_regions]
    incidence = np.zeros((len(pool), len(candidates)), np.float32)
    rows = []
    for k, (mass, y, x, yy, xx) in enumerate(candidates):
        for j, mask in enumerate(masks):
            incidence[j, k] = mask[yy, xx].mean()
        covered = max((incidence[j,k] for j,p in enumerate(pool) if p['id'] in accepted), default=0.)
        rows.append([xx.mean()/w, yy.mean()/h, (xx.max()-xx.min()+1)/w,
                     (yy.max()-yy.min()+1)/h, len(xx)/(tile_size**2),
                     probability[0,yy,xx].mean(), probability[0,yy,xx].std(),
                     probability[1,yy,xx].mean(),
                     (probability[1,yy,xx] >= kernel_threshold).mean(),
                     covered, 1-covered, mass/max(1., float(foreground.sum()))])
    return np.asarray(rows, np.float32).reshape(-1,12), incidence


class EvidenceGraph(nn.Module):
    """Equal-capacity collapsed/typed and connected/disconnected controls."""
    def __init__(self, dimensions, hidden, layers, *, typed, ownership):
        super().__init__()
        self.typed, self.ownership = typed, ownership
        self.project = nn.Linear(dimensions, hidden)
        self.region_project = nn.Linear(12, hidden)
        self.word_layers = nn.ModuleList(nn.Linear(4*hidden, hidden) for _ in range(layers))
        self.region_layers = nn.ModuleList(nn.Linear(2*hidden, hidden) for _ in range(layers))
        self.output = nn.Linear(hidden,1)

    def forward(self, x, support, conflict, collapsed, regions, incidence):
        if incidence.shape != (len(x), len(regions)):
            raise ValueError('Region incidence dimensions differ')
        x, r = F.relu(self.project(x)), F.relu(self.region_project(regions))
        incidence = incidence if self.ownership else torch.ones_like(incidence)
        to_word = incidence / incidence.sum(1,keepdim=True).clamp_min(1.)
        to_region = incidence.T / incidence.T.sum(1,keepdim=True).clamp_min(1.)
        a, b = (support, conflict) if self.typed else (collapsed, collapsed)
        for wl, rl in zip(self.word_layers, self.region_layers):
            new_r = r + F.relu(rl(torch.cat((r,to_region@x),-1)))
            x = x + F.relu(wl(torch.cat((x,a@x,b@x,to_word@r),-1)))
            r = new_r
        return self.output(x).squeeze(-1)


def align_characters(target, donor, *, maximum_distance):
    """Monotone character correspondences in original RGB space.

    Cross-attention centroids are explicitly an alignment proxy, not character
    GT. Spatial checks supplement same-index monotonicity; they cannot establish
    perfect alignment. Different-length donor crops can support a subsequence;
    the target template length stays fixed. No hidden glyphs are inserted.
    """
    if not target['text'] or not donor['text']:
        return None
    a, b = np.asarray(target['centers']), np.asarray(donor['centers'])
    if a.shape != (len(target['text']),2) or b.shape != (len(donor['text']),2):
        raise ValueError('Character/world coordinate mismatch')
    h = max(1., min(target['height'], donor['height']))
    if maximum_distance<=0:raise ValueError('Positive spatial alignment margin required')
    distances = np.linalg.norm(a[:,None]-b[None,:],axis=2)/h
    n,m=distances.shape;score=np.zeros((n+1,m+1));operation=np.zeros((n+1,m+1),np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=[(score[i-1,j],1),(score[i,j-1],2)]
            if distances[i-1,j-1]<maximum_distance:
                options.append((score[i-1,j-1]+1-distances[i-1,j-1]/maximum_distance,3))
            score[i,j],operation[i,j]=max(options,key=lambda z:(z[0],z[1]))
    correspondence=np.full(n,-1,np.int64);i,j=n,m
    while i and j:
        code=operation[i,j]
        if code==3:correspondence[i-1]=j-1;i-=1;j-=1
        elif code==2:j-=1
        else:i-=1
    return correspondence


def consensus_reading(target, donors, *, maximum_distance, confidence_margin):
    """Copy actual donor characters only with a stronger spatially aligned read."""
    text = list(target['text']); confidence = list(target['confidence'])
    if len(confidence) != len(text):
        raise ValueError('Character confidence count differs')
    provenance = [None]*len(text)
    for donor in sorted(donors,key=lambda z:z['id']):
        aligned = align_characters(target, donor, maximum_distance=maximum_distance)
        if aligned is None:
            continue
        for i,j in enumerate(aligned):
            if j>=0 and donor['confidence'][j] > confidence[i]+confidence_margin:
                text[i], confidence[i], provenance[i] = donor['text'][j], donor['confidence'][j], donor['id']
    return ''.join(text), confidence, provenance


def action_crop(rgb, parent, pool, kind, *, fraction, overlap, isolation_periphery):
    """Bounded directional crop or neighbor isolation, using predictions only."""
    if kind not in ('left','right','vertical','isolate') or fraction <= 0 or not 0<overlap<=1 or isolation_periphery<0:
        raise ValueError('Unknown explicit visual action')
    h,w = rgb.shape[:2]; box = list(map(int,parent['padded_bbox_original']))
    a,b,d,e = box; pad=max(1,int(np.ceil((e-b)*fraction)))
    if kind=='left':box[0]=max(0,a-pad)
    elif kind=='right':box[2]=min(w,d+pad)
    elif kind=='vertical':box[1],box[3]=max(0,b-pad),min(h,e+pad)
    x0,y0,x1,y1=box;crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]).copy()
    removed=0
    if kind=='isolate':
        yy,xx=np.mgrid[y0:y1,x0:x1];cx,cy=(a+d)/2,(b+e)/2
        own=((xx-cx)/max(1,d-a))**2+((yy-cy)/max(1,e-b))**2
        mask=np.zeros(own.shape,bool)
        for q in pool:
            if q['id']==parent['id'] or conflicts(parent,q,overlap):continue
            qa,qb,qd,qe=q['bbox'];qx,qy=(qa+qd)/2,(qb+qe)/2
            other=((xx-qx)/max(1,qd-qa))**2+((yy-qy)/max(1,qe-qb))**2
            mask |= (xx>=qa)&(xx<qd)&(yy>=qb)&(yy<qe)&(other<own)&(own>isolation_periphery)
        border=np.concatenate((crop[0],crop[-1],crop[:,0],crop[:,-1]))
        crop[mask]=np.median(border,axis=0).astype(np.uint8);removed=int(mask.sum())
    return crop,box,removed
