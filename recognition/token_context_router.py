"""Approximate benefit routing from frozen visual tokens and dense word support.

The networks predict paired branch-count differences. No inference method
accepts GT, ignore masks, or actual fallback outputs. Dense supervision is an
explicit weak annotation-box proxy, not a physical text-presence certificate.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


def context_views(rgb, grid, fill):
    """Original raster plus an exact, nonoverlapping tile partition."""
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or len(grid) != 2 or min(grid) < 1 or grid[0] > h or grid[1] > w:
        raise ValueError('Original RGB uint8 and positive tile grid required')
    boxes = [[0, 0, w, h]]
    xs = np.linspace(0, w, grid[1]+1, dtype=int); ys = np.linspace(0, h, grid[0]+1, dtype=int)
    boxes += [[int(a), int(b), int(c), int(d)] for b, d in zip(ys, ys[1:]) for a, c in zip(xs, xs[1:])]
    result = []
    for i, box in enumerate(boxes):
        a, b, c, d = box
        crop = rgb[b:d, a:c]; side = max(c-a, d-b)
        square = np.full((side, side, 3), fill, np.uint8)
        left, top = (side-(c-a))//2, (side-(d-b))//2
        square[top:top+d-b, left:left+c-a] = crop
        result.append({'rgb': square, 'bbox': box, 'pad_xy': [left, top], 'side': side,
                       'mass_weight': .5 if i == 0 else .5})
    return result


def patch_geometry(view, image_wh, patch_grid):
    """Map native CLIP square patch centers/envelopes back to stored pixels."""
    w, h = image_wh; a, b, c, d = view['bbox']; left, top = view['pad_xy']; side = view['side']
    coords, valid, masses = [], [], []
    for y in range(patch_grid):
        for x in range(patch_grid):
            x1 = a-left+x*side/patch_grid; y1 = b-top+y*side/patch_grid
            x2 = a-left+(x+1)*side/patch_grid; y2 = b-top+(y+1)*side/patch_grid
            cx, cy = (x1+x2)/2, (y1+y2)/2
            good = a <= cx < c and b <= cy < d
            coords.append([cx/w, cy/h, (x2-x1)/w, (y2-y1)/h])
            valid.append(good)
            overlap = max(0., min(c, x2)-max(a, x1))*max(0., min(d, y2)-max(b, y1))
            masses.append(overlap/(w*h)*view['mass_weight'] if good else 0.)
    return np.asarray(coords, np.float32), np.asarray(valid, bool), np.asarray(masses, np.float32)


def erased_views(rgb, rows, fill, margin_height):
    """Global erased view and one original/erased expanded context per word."""
    erased = rgb.copy(); h, w = rgb.shape[:2]; local = []
    for r in rows:
        a, b, c, d = r['bbox']
        if not 0 <= a < c <= w or not 0 <= b < d <= h:
            raise ValueError('Word envelope outside original raster')
        erased[b:d, a:c] = fill
        margin = math.ceil((d-b)*margin_height)
        box = [max(0, a-margin), max(0, b-margin), min(w, c+margin), min(h, d+margin)]
        x, y, xx, yy = box; crop = rgb[y:yy, x:xx].copy(); hidden = crop.copy()
        hidden[b-y:d-y, a-x:c-x] = fill
        local.append({'bbox': box, 'original': crop, 'erased': hidden, 'candidate_id': r['candidate_id']})
    return erased, local


def transcript_chunks(rows, tokenize, maximum_tokens):
    """All nonempty readings, spatial order, whole-word chunks; never truncate."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['bbox'][1], rows[i]['bbox'][0], i))
    result = []; group = []; ids = []
    for i in order:
        word = rows[i]['raw_text']
        if not word.strip():
            continue
        if len(tokenize(word)) > maximum_tokens:
            raise ValueError('One raw OCR word exceeds the configured transcript token budget')
        proposed = ' '.join(group+[word])
        if group and len(tokenize(proposed)) > maximum_tokens:
            result.append({'text': ' '.join(group), 'row_indices': ids}); group = []; ids = []
        group.append(word); ids.append(i)
    if group:
        result.append({'text': ' '.join(group), 'row_indices': ids})
    return result


def likelihood_features(global_scores, word_scores):
    out = {}
    for scope, values in [('global', global_scores), ('word', word_scores)]:
        for field in ('visual_gain', 'eos_visual_gain', 'original_logprob_per_token'):
            v = [r[field] for r in values]
            out[scope+'_'+field+'_mean'] = float(np.mean(v)) if v else 0.
            out[scope+'_'+field+'_min'] = float(min(v)) if v else 0.
        out[scope+'_available'] = float(bool(values))
    return out


def base_features(rows, image_wh):
    n = len(rows); w, h = image_wh
    native = [r['confidence'] for r in rows]; det = [r['detector_score'] for r in rows]
    area = [(r['bbox'][2]-r['bbox'][0])*(r['bbox'][3]-r['bbox'][1])/(w*h) for r in rows]
    lengths = [len(r['raw_text']) for r in rows]
    values = [math.log1p(n), math.log(w/h), math.log1p(sum(lengths)), float(any(not r['raw_text'].strip() for r in rows))]
    for v in (native, det, area, lengths):
        values += [float(np.mean(v)) if v else 0., float(min(v)) if v else 0., float(max(v)) if v else 0.]
    values += [float(sum(1-v for v in native)), float(sum(area))]
    return np.asarray(values, np.float32)


def word_geometry(rows, image_wh):
    w, h = image_wh; result = []
    for r in rows:
        a, b, c, d = r['bbox']; p = r['token_probabilities']
        result.append([a/w, b/h, c/w, d/h, (c-a)/w, (d-b)/h, r['confidence'], r['detector_score'],
            float(min(p)) if p else 0., float(np.mean(p)) if p else 0., math.log1p(len(r['raw_text'])),
            float(not r['raw_text'].strip())])
    return np.asarray(result, np.float32).reshape(-1, 12)


def unbalanced_transport(cost, row_mass, column_mass, settings):
    """Fixed-iteration log-domain unbalanced Sinkhorn; iteration cap is explicit."""
    cost = np.asarray(cost, np.float64); a = np.asarray(row_mass, np.float64); b = np.asarray(column_mass, np.float64)
    if cost.shape != (len(a), len(b)) or not np.isfinite(cost).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError('Invalid transport cost or masses')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or settings['epsilon'] <= 0 or settings['rho'] <= 0 or settings['iterations'] < 1 or settings['mass_floor'] <= 0:
        raise ValueError('Invalid explicit transport configuration')
    if not len(a) or not len(b) or not a.sum() or not b.sum():
        return np.zeros(cost.shape, np.float64), {'iterations': 0, 'maximum_log_update': 0.}
    logk = -cost/settings['epsilon']; power = settings['rho']/(settings['rho']+settings['epsilon'])
    loga = np.log(np.maximum(a, settings['mass_floor'])); logb = np.log(np.maximum(b, settings['mass_floor']))
    u = np.zeros(len(a)); v = np.zeros(len(b)); delta = 0.
    for _ in range(settings['iterations']):
        old = u.copy()
        u = power*(loga-logsumexp(logk+v[None, :], axis=1))
        v = power*(logb-logsumexp(logk+u[:, None], axis=0))
        delta = float(np.max(np.abs(u-old)))
    t = np.exp(logk+u[:, None]+v[None, :]); t[a == 0, :] = 0; t[:, b == 0] = 0
    if not np.isfinite(t).all():
        raise ValueError('Nonfinite transport plan')
    return t, {'iterations': settings['iterations'], 'maximum_log_update': delta}


class DenseSupport(nn.Module):
    def __init__(self, clip_dim, hidden_dim):
        super().__init__()
        self.visual = nn.Linear(clip_dim, hidden_dim); self.text = nn.Linear(clip_dim, hidden_dim)
        self.textness = nn.Linear(clip_dim, 1); self.log_scale = nn.Parameter(torch.zeros(()))

    def similarity(self, patches, text):
        return F.normalize(self.visual(patches), dim=-1)@F.normalize(self.text(text), dim=-1).T

    def paired_logits(self, pooled, text):
        return (F.normalize(self.visual(pooled), dim=-1)*F.normalize(self.text(text), dim=-1)).sum(-1)*self.log_scale.exp()


def dense_features(arrays, support, settings, device):
    valid = arrays['patch_valid']; xy = arrays['patch_geometry'][valid]; patches = arrays['patches'][valid]
    text = arrays['word_text']; boxes = arrays['word_geometry'][:, :4]; base_mass = arrays['patch_mass'][valid]
    with torch.inference_mode():
        pp = torch.as_tensor(patches, device=device); tt = torch.as_tensor(text, device=device)
        probability = support.textness(pp).sigmoid().flatten().cpu().numpy()
        sim = support.similarity(pp, tt).cpu().numpy() if len(text) else np.empty((len(patches), 0))
    a = probability*base_mass; inside = np.zeros((len(xy), len(boxes)), bool); distance = np.zeros_like(sim)
    for j, (x1, y1, x2, y2) in enumerate(boxes):
        inside[:, j] = (xy[:, 0] >= x1)&(xy[:, 0] <= x2)&(xy[:, 1] >= y1)&(xy[:, 1] <= y2)
        distance[:, j] = np.maximum(np.maximum(x1-xy[:, 0], xy[:, 0]-x2), 0)+np.maximum(np.maximum(y1-xy[:, 1], xy[:, 1]-y2), 0)
    b = np.maximum((a[:, None]*inside).sum(0), settings['mass_floor'])
    cost = 1-sim+settings['spatial_cost']*distance
    t, diagnostic = unbalanced_transport(cost, a, b, settings)
    rsum, csum = t.sum(1), t.sum(0)
    def ratio(x, y):
        return float(x/y) if y else 0.
    missing = np.maximum(a-rsum, 0); unsupported = np.maximum(b-csum, 0)
    outside = ~inside.any(1)
    features = np.array([a.sum(), b.sum(), t.sum(), ratio(missing.sum(), a.sum()),
        ratio(unsupported.sum(), b.sum()), ratio(a[outside].sum(), a.sum()),
        float(np.max(sim)) if sim.size else 0., float(np.mean(sim.max(0))) if sim.size else 0.,
        ratio(float((t*cost).sum()), t.sum()), float(len(boxes))], np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite dense-support evidence')
    return features, {**diagnostic, 'row_mass': float(a.sum()), 'column_mass': float(b.sum()), 'transport_mass': float(t.sum())}


class BenefitRouter(nn.Module):
    def __init__(self, settings, scalar_dim, extra_dim, kind):
        super().__init__(); self.kind = kind; d = settings['hidden_dim']; self.contextual = kind in ('token_context', 'combined')
        self.pooled = kind == 'pooled_context'; self.query_count = settings['query_count']
        if self.contextual:
            self.patch = nn.Linear(settings['clip_dim'], d); self.position = nn.Linear(4, d)
            self.query = nn.Linear(settings['query_dim'], d); self.query_position = nn.Parameter(torch.randn(self.query_count, d)*settings['initialization_std'])
            self.word = nn.Linear(2*settings['clip_dim']+settings['word_geometry_dim'], d)
            self.decision = nn.Parameter(torch.randn(1, d)*settings['initialization_std'])
            self.cross = nn.MultiheadAttention(d, settings['heads'], dropout=0., batch_first=True)
            self.self_attention = nn.MultiheadAttention(d, settings['heads'], dropout=0., batch_first=True)
            self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
            self.word_correctness = nn.Linear(d, 1)
            representation = 2*d
        elif self.pooled:
            representation = 3*settings['clip_dim']+settings['query_dim']
        else:
            representation = 0
        self.head = nn.Sequential(nn.Linear(scalar_dim+extra_dim+representation, settings['mlp_dim']), nn.GELU(),
            nn.Linear(settings['mlp_dim'], settings['mlp_dim']), nn.GELU(), nn.Linear(settings['mlp_dim'], 2))

    def forward(self, batch):
        pieces = [batch['scalars'], batch['extra']]; word_logits = None
        if self.contextual:
            memory = torch.cat([self.patch(batch['patches'])+self.position(batch['patch_geometry']),
                                self.query(batch['queries'])+self.query_position], dim=1)
            memory_mask = torch.cat([batch['patch_padding'], torch.zeros(batch['queries'].shape[:2], dtype=torch.bool, device=memory.device)], dim=1)
            words = self.word(torch.cat([batch['word_text'], batch['word_image'], batch['word_geometry']], dim=-1))
            query = torch.cat([self.decision.expand(len(words), -1, -1), words], dim=1)
            query_mask = torch.cat([torch.zeros((len(words), 1), dtype=torch.bool, device=words.device), batch['word_padding']], dim=1)
            value, _ = self.cross(query, memory, memory, key_padding_mask=memory_mask, need_weights=False)
            value = self.norm1(value+query)
            updated, _ = self.self_attention(value, value, value, key_padding_mask=query_mask, need_weights=False)
            value = self.norm2(value+updated); valid = (~batch['word_padding']).float()
            pooled = (value[:, 1:]*valid[..., None]).sum(1)/valid.sum(1).clamp_min(1)[..., None]
            pieces += [value[:, 0], pooled]; word_logits = self.word_correctness(value[:, 1:]).squeeze(-1)
        elif self.pooled:
            valid = (~batch['patch_padding']).float()
            pm = (batch['patches']*valid[..., None]).sum(1)/valid.sum(1).clamp_min(1)[..., None]
            wv = (~batch['word_padding']).float()
            wm = (torch.cat([batch['word_text'], batch['word_image']], dim=-1)*wv[..., None]).sum(1)/wv.sum(1).clamp_min(1)[..., None]
            pieces += [pm, batch['queries'].mean(1), wm]
        result = self.head(torch.cat(pieces, dim=-1))
        return result, word_logits


def benefit_loss(prediction, target, reference_f1, settings):
    """Count regression plus cost-weighted pair ranking; exact ties incur no cost."""
    regression = F.smooth_l1_loss(prediction, target, beta=settings['huber_beta'])
    predicted = 2*prediction[:, 0]-reference_f1*prediction[:, 1]
    truth = 2*target[:, 0]-reference_f1*target[:, 1]
    difference = truth[:, None]-truth[None, :]; mask = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1)&difference.ne(0)
    if mask.any():
        weights = difference[mask].abs(); weights = weights/weights.mean()
        pair = predicted[:, None]-predicted[None, :]
        ranking = (F.softplus(-difference[mask].sign()*pair[mask])*weights).mean()
    else:
        ranking = prediction.sum()*0
    return regression+settings['ranking_weight']*ranking, {'regression': float(regression.detach()), 'ranking': float(ranking.detach())}
