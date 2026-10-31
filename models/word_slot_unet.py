"""APPROXIMATE word-slot U-Net: instance feedback at every decoder scale.

The qualified pixel U-Net is an immutable feature prior. A second, trainable
coarse-to-fine decoder exchanges information with independent instance slots at
all four resolutions. Its residuals condition the next scale, rather than being
an instance head attached to the last feature map. Cached priors are FP32 and
are valid only while the entire original network, including BN, remains frozen.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet


def inverse_sigmoid(x):
    return torch.logit(x.clamp(1e-5, 1 - 1e-5))


def cxcywh_to_xyxy(box):
    return torch.cat((box[..., :2] - box[..., 2:] / 2,
                      box[..., :2] + box[..., 2:] / 2), dim=-1)


def xyxy_to_cxcywh(box):
    return torch.cat(((box[..., :2] + box[..., 2:]) / 2,
                      box[..., 2:] - box[..., :2]), dim=-1)


def denoising_attention_mask(ordinary, groups, device):
    """Neither normal slots nor other DN groups can receive target information."""
    group_ids = torch.as_tensor(groups, device=device)
    count = ordinary + len(groups)
    mask = torch.zeros(count, count, dtype=torch.bool, device=device)
    mask[:ordinary, ordinary:] = True
    mask[ordinary:, :ordinary] = True
    if len(groups):
        mask[ordinary:, ordinary:] = group_ids[:, None] != group_ids[None, :]
    return mask


class SlotPixelExchange(nn.Module):
    def __init__(self, width, heads, key_size, feedback):
        super().__init__()
        self.width, self.key_size, self.feedback = width, key_size, feedback
        self.key = nn.Conv2d(width, width, 1)
        self.query = nn.Linear(width, width)
        self.position = nn.Linear(2, width, bias=False)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.pixel_out = nn.Conv2d(width, width, 1)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, pixels, queries, references, ordinary, mask):
        size = min(self.key_size, pixels.shape[-1])
        small = F.interpolate(pixels, (size, size), mode="nearest")
        yy, xx = torch.meshgrid(torch.linspace(0, 1, size, device=pixels.device),
                                torch.linspace(0, 1, size, device=pixels.device), indexing="ij")
        positions = torch.stack((xx, yy), -1).reshape(-1, 2)
        key = self.key(small).flatten(2).transpose(1, 2) + self.position(positions)[None]
        q = self.norm1(queries)
        q = queries + self.self_attention(q, q, q, attn_mask=mask, need_weights=False)[0]
        logits = torch.einsum("bqd,bnd->bqn", self.query(q), key) / math.sqrt(self.width)
        distance = ((positions[None, None] - references[:, :, None, :2]) /
                    references[:, :, None, 2:].clamp_min(.025)).square().sum(-1)
        attention = (logits - distance * 2).softmax(-1)
        q = q + torch.einsum("bqn,bnd->bqd", attention, key)
        q = q + self.ff(self.norm2(q))
        # DN slots NEVER contribute to the pixel path: a self-attention mask
        # alone would not prevent GT -> pixels -> ordinary-slot leakage.
        if self.feedback:
            weights = (logits[:, :ordinary] - distance[:, :ordinary] * 2).softmax(1)
            context = torch.einsum("bqn,bqd->bdn", weights, q[:, :ordinary])
            context = context.reshape(len(pixels), self.width, size, size)
            pixels = pixels + self.gain * F.interpolate(self.pixel_out(context),
                                                       pixels.shape[-2:], mode="nearest")
        return pixels, q


class WordSlotUNet(PracticalTotalTextWordKernelAGUNet):
    def __init__(self, *, slots, width, heads, key_size, skip_floor, feedback=True,
                 mask_initialization=False, strip_points=8, character_width=384):
        super().__init__(skip_floor=skip_floor, encoder_pretrained=False)
        if slots < 1 or width % heads or strip_points < 2:
            raise ValueError("Invalid slot architecture")
        self.slots, self.width = slots, width
        self.mask_initialization, self.strip_points = mask_initialization, strip_points
        # Freeze the complete qualified prior, including its normalization state.
        self.base_names = tuple(self.state_dict())
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.prior_projection = nn.ModuleList(nn.Conv2d(ch, width, 1) for ch in (512, 256, 128, 64))
        self.refinement = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                                     nn.GroupNorm(heads, width), nn.GELU()) for _ in range(4))
        self.exchange = nn.ModuleList(SlotPixelExchange(width, heads, key_size, feedback) for _ in range(4))
        self.query_content = nn.Parameter(torch.randn(slots, width) * .02)
        columns = math.ceil(math.sqrt(slots * 2))
        rows = math.ceil(slots / columns)
        centers = [((i % columns + .5) / columns, (i // columns + .5) / rows, .25, .10)
                   for i in range(slots)]
        self.register_buffer("reference_boxes", torch.tensor(centers))
        self.box_embedding = nn.Linear(4, width)
        self.box_head = nn.Linear(width, 4)
        nn.init.zeros_(self.box_head.weight); nn.init.zeros_(self.box_head.bias)
        self.presence = nn.Linear(width, 1)
        nn.init.constant_(self.presence.bias, -2)
        self.mask_query, self.mask_pixels = nn.Linear(width, width), nn.Conv2d(width, width, 1)
        self.strip_head = nn.Linear(width, 4 * strip_points)
        nn.init.zeros_(self.strip_head.weight); nn.init.zeros_(self.strip_head.bias)
        self.character_projection = nn.Conv2d(width, character_width, 1)
        self.glyph_head = nn.Conv2d(width, 1, 1)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for name in ("encoder", "bottleneck", "up3", "up2", "up1", "up0", "head"):
            getattr(self, name).eval()
        return self

    def initialize_prior(self, state):
        if set(state) != set(self.base_names):
            raise ValueError("Every qualified prior tensor, and no other tensor, is required")
        own = self.state_dict()
        for name, value in state.items():
            if own[name].shape != value.shape or own[name].dtype != value.dtype:
                raise ValueError("Incompatible prior tensor: " + name)
            own[name] = value
        self.load_state_dict(own, strict=True)
        return {"copied_tensors": len(state), "new_tensors": sorted(set(own) - set(state)),
                "strict_load": True}

    @torch.no_grad()
    def prior_pyramid(self, image):
        e0, e1, e2, e3, e4 = self._encode(image)
        p3 = self.up3(self.bottleneck(e4), e3)
        p2 = self.up2(p3, e2)
        p1 = self.up1(p2, e1)
        p0 = self.up0(p1, e0)
        return (p3, p2, p1, p0), self.head(p0)

    def decode_pyramid(self, pyramid, *, denoising=None, dense_groups=1, prior_boxes=None, dense_offsets=None):
        if len(pyramid) != 4 or len(pyramid[0]) != 1:
            raise ValueError("The reproducible runner uses one image per microbatch")
        if denoising is not None and not self.training:
            raise ValueError("Target-derived denoising queries are training-only")
        if dense_groups != 1 and not self.training:
            raise ValueError("Dense matching groups are training-only")
        initial = self.reference_boxes.clone()
        if prior_boxes is not None:
            if prior_boxes.ndim!=2 or prior_boxes.shape[1]!=4 or len(prior_boxes)>self.slots:
                raise ValueError('Invalid predicted initialization boxes')
            initial[:len(prior_boxes)] = xyxy_to_cxcywh(prior_boxes).clamp(.005,.995)
        references = initial[None].repeat(1, dense_groups, 1)
        if dense_offsets is not None:
            if not self.training or dense_offsets.shape!=(dense_groups-1,self.slots,4):
                raise ValueError('Dense perturbations are isolated training-only query groups')
            auxiliary=references[:,self.slots:].reshape(1,dense_groups-1,self.slots,4)
            scale=torch.cat((auxiliary[...,2:],auxiliary[...,2:]),-1)
            auxiliary=(auxiliary+dense_offsets[None]*scale).clamp(.005,.995)
            references=torch.cat((references[:,:self.slots],auxiliary.reshape(1,-1,4)),1)
        q = self.query_content[None].repeat(1, dense_groups, 1) + self.box_embedding(references)
        ordinary = q.shape[1]
        groups = []
        if denoising is not None:
            references = torch.cat((references, denoising["references"][None]), 1)
            q = torch.cat((q, self.box_embedding(denoising["references"])[None]), 1)
            groups = denoising["groups"]
        mask = denoising_attention_mask(ordinary, groups, q.device)
        if dense_groups > 1:
            ids = torch.arange(ordinary, device=q.device) // self.slots
            mask[:ordinary, :ordinary] = ids[:, None] != ids[None, :]
        carry = None
        for projection, refine, exchange, prior in zip(self.prior_projection, self.refinement,
                                                      self.exchange, pyramid):
            pixels = projection(prior)
            if carry is not None:
                pixels = pixels + F.interpolate(carry, pixels.shape[-2:], mode="nearest")
            pixels = refine(pixels)
            # Only the first ordinary group enters feedback, so auxiliary
            # dense groups cannot alter the inference group's pixel path.
            pixels, q = exchange(pixels, q, references, self.slots, mask)
            carry = pixels
        mask_logits = torch.einsum("bqd,bdhw->bqhw", self.mask_query(q), self.mask_pixels(pixels)) / math.sqrt(self.width)
        box_delta = self.box_head(q)
        if self.mask_initialization:
            small = F.interpolate(mask_logits, (32, 32), mode="nearest")
            yy, xx = torch.meshgrid(torch.linspace(0, 1, 32, device=q.device),
                                    torch.linspace(0, 1, 32, device=q.device), indexing="ij")
            pos = torch.stack((xx, yy), -1).reshape(-1, 2)
            bias = -2 * ((pos[None, None] - references[:, :, None, :2]) /
                         references[:, :, None, 2:].clamp_min(.025)).square().sum(-1)
            weight = (small.flatten(2) + bias).softmax(-1)
            center = torch.einsum("bqn,nd->bqd", weight, pos)
            variance = torch.einsum("bqn,bqnd->bqd", weight,
                                    (pos[None, None] - center[:, :, None]).square())
            references = torch.cat((center, (variance.clamp_min(1e-5).sqrt() * 3.5).clamp(.025, .95)), -1)
        boxes = cxcywh_to_xyxy((inverse_sigmoid(references) + box_delta).sigmoid()).clamp(0, 1)
        output = {"boxes": boxes[0, :self.slots], "presence": self.presence(q)[0, :self.slots, 0],
                  "masks": mask_logits[0, :self.slots], "queries": q[0, :self.slots], "features": pixels,
                  "strip_offsets": self.strip_head(q)[0, :self.slots].reshape(self.slots, 2, self.strip_points, 2),
                  "all_boxes": boxes[0], "all_presence": self.presence(q)[0, :, 0],
                  "all_masks": mask_logits[0], "ordinary_count": ordinary}
        return output

    def forward(self, image):
        pyramid, _ = self.prior_pyramid(image)
        return self.decode_pyramid(pyramid)


def sampling_grid(boxes, offsets, *, height, width, strength):
    """Monotone RGB strips; zero offsets are exactly the ordinary box sampler.

    Upper/lower boundaries are content conditioned. Horizontal displacements
    are bounded, with folding penalized separately by the training objective.
    Coordinates are normalized edge coordinates in [0,1], then mapped to the
    align_corners=False grid_sample convention.
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("Expected N x 4 normalized xyxy boxes")
    t = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + .5) / width
    s = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + .5) / height
    edges = torch.stack((boxes[:, :2], boxes[:, :2]), 1).unsqueeze(2).expand(-1, 2, width, -1).clone()
    span = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-5)
    edges[..., 0] += t[None, None] * span[:, None, None, 0]
    edges[:, 1, :, 1] += span[:, None, 1]
    if offsets is not None:
        curve = F.interpolate(offsets.permute(0, 1, 3, 2).reshape(len(boxes), 4, -1).to('cpu'),
                              size=width, mode="linear", align_corners=True).to(boxes.device)
        curve = curve.reshape(len(boxes), 2, 2, width).permute(0, 1, 3, 2)
        edges = edges + strength * curve.tanh() * span[:, None, None]
    grid = edges[:, :1] * (1 - s[None, :, None, None]) + edges[:, 1:] * s[None, :, None, None]
    return grid * 2 - 1


def deterministic_sample(image, grid, output_device):
    """CPU bilinear sampling has deterministic grid gradients; no FP16 cache."""
    source = image.to("cpu")
    if source.ndim == 3:
        source = source[None]
    return F.grid_sample(source.expand(len(grid), -1, -1, -1), grid.to("cpu"),
                         mode="bilinear", padding_mode="border", align_corners=False).to(output_device)
