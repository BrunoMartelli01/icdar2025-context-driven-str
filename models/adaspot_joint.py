"""Isolated APPROXIMATE AdaSpot: shared AG-UNet, two real OCR heads and gate.

Original paper3.2-3.3 supplies the backbone, author proposal4.3-4.8 the extension.
No RGB re-encoding: both recognizers consume Mask-Align U-Net feature grids.
"""

import copy
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from models.adaspot_mask_align import mask_align_geometry, sample_mask_aligned
from models.unet_word_research import WordResearchAGUNet
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    verify_parseq_runtime_source)


def qualified_parseq_parts(source, archive, checkpoint):
    """Verify official local runtime before importing; no downloads/installations."""
    from scripts.prune_segmentation_checkpoints import sha256_file
    provenance = verify_parseq_runtime_source(Path(source), source_type=PARSEQ_ARCHIVE_SOURCE,
                                             source_archive=Path(archive))
    if sha256_file(Path(checkpoint)) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("Qualified PARSeq checkpoint SHA256 differs")
    sys.path.insert(0, str(source))
    try:
        from strhub.models.parseq.modules import Decoder, DecoderLayer, TokenEmbedding
        from strhub.data.utils import Tokenizer
    finally:
        sys.path.remove(str(source))
    settings = yaml.safe_load((Path(source) / "configs/model/parseq.yaml").read_text())
    charset = yaml.safe_load((Path(source) / "configs/charset/94_full.yaml").read_text())["model"]["charset_train"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return (Decoder, DecoderLayer, TokenEmbedding), Tokenizer(charset), settings, state, provenance


def strict_partial_initialization(module, source, *, prefix_keys):
    """Explicitly merge the expected warm-start subset, then load strictly."""
    own = module.state_dict()
    expected = {key for key in source if key.startswith(prefix_keys)}
    if not expected or not expected <= own.keys():
        raise ValueError("Expected pretrained OCR keys missing from new head")
    for key in expected:
        if own[key].shape != source[key].shape or own[key].dtype != source[key].dtype:
            raise ValueError("Pretrained OCR tensor incompatible: " + key)
        own[key] = source[key].clone()
    module.load_state_dict(own, strict=True)
    return {"copied_keys": sorted(expected), "new_keys": sorted(set(own) - expected), "strict_load": True}


class JointBackbone(WordResearchAGUNet):
    """Separate research subclass: unfreeze encoder, preserve historical classes."""

    def __init__(self, *, skip_floor):
        super().__init__("control", skip_floor=skip_floor, embedding_dimensions=4, encoder_pretrained=False)
        self.encoder.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.train(mode)
        return self

    def _encode(self, image):
        e = self.encoder
        e0 = e.relu(e.bn1(e.conv1(image)))
        e1 = e.layer1(e.maxpool(e0))
        e2, e3 = e.layer2(e1), None
        e3 = e.layer3(e2)
        return [e0, e1, e2, e3, e.layer4(e3)]


def spatial_encoding(height, width, channels, device, dtype):
    """Deterministic 2D sine/cosine positional encoding for flattened grids."""
    if channels % 4:
        raise ValueError("OCR embedding dimension must be divisible by four")
    frequency = torch.exp(-math.log(10000.) * torch.arange(channels // 4, device=device, dtype=dtype) / (channels // 4))
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.cat([(x[..., None] * frequency).sin(), (x[..., None] * frequency).cos(),
                      (y[..., None] * frequency).sin(), (y[..., None] * frequency).cos()], -1).reshape(1, height * width, channels)


class FeatureOCRHead(nn.Module):
    def __init__(self, parts, tokenizer, *, depth, autoregressive, channels, embed_dim,
                 num_heads, mlp_ratio, dropout, max_label_length):
        super().__init__()
        Decoder, DecoderLayer, TokenEmbedding = parts
        self.tokenizer, self.autoregressive = tokenizer, autoregressive
        self.max_label_length, self.embed_dim = max_label_length, embed_dim
        self.projection = nn.Linear(channels, embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        layer = DecoderLayer(embed_dim, num_heads, embed_dim * mlp_ratio, dropout)
        self.decoder = Decoder(layer, depth, nn.LayerNorm(embed_dim))
        # New deep layers have independent fresh initialization, not shared storage.
        for item in self.decoder.layers[1:]:
            for mod in item.modules():
                if isinstance(mod, (nn.Linear, nn.LayerNorm)):
                    mod.reset_parameters()
                elif isinstance(mod, nn.MultiheadAttention):
                    mod._reset_parameters()
        self.text_embed = TokenEmbedding(len(tokenizer), embed_dim)
        self.pos_queries = nn.Parameter(torch.empty(1, max_label_length + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_queries, std=.02)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, len(tokenizer) - 2)

    def initialize_parseq(self, state):
        return strict_partial_initialization(self, state, prefix_keys=("decoder.layers.0.", "decoder.norm.", "text_embed.", "pos_queries", "head."))

    def memory(self, features):
        if features.ndim != 4:
            raise ValueError("OCR expects NCHW aligned feature grids")
        h, w = features.shape[-2:]
        tokens = self.projection(features.flatten(2).transpose(1, 2))
        return self.memory_norm(tokens + spatial_encoding(h, w, self.embed_dim, tokens.device, tokens.dtype))

    def decode(self, tokens, memory, *, query=None, content_mask=None, query_mask=None, padding=None):
        length = tokens.shape[1]
        content = torch.cat([self.text_embed(tokens[:, :1]),
            self.text_embed(tokens[:, 1:]) + self.pos_queries[:, :length - 1]], 1)
        if query is None:
            query = self.pos_queries[:, :length].expand(len(tokens), -1, -1)
        result = self.decoder(self.dropout(query), self.dropout(content), memory, query_mask, content_mask, padding)
        return self.head(result)

    def light(self, memory):
        """One NAR decoder layer, then exactly one cloze refinement pass."""
        n, steps = len(memory), self.max_label_length + 1
        bos = torch.full((n, 1), self.tokenizer.bos_id, dtype=torch.long, device=memory.device)
        queries = self.pos_queries.expand(n, -1, -1)
        first = self.decode(bos, memory, query=queries)
        context = torch.cat([bos, first[:, :-1].detach().argmax(-1)], 1)
        causal = torch.triu(torch.ones(steps, steps, device=memory.device, dtype=torch.bool), 1)
        # Context position i+1 holds output i: exclude that token from query i.
        cloze = torch.zeros_like(causal)
        cloze[torch.arange(steps - 1), torch.arange(1, steps)] = True
        padding = (context == self.tokenizer.eos_id).int().cumsum(-1) > 0
        refined = self.decode(context, memory, query=queries, content_mask=causal, query_mask=cloze, padding=padding)
        return first, refined

    def teacher(self, memory, targets):
        if not self.autoregressive:
            raise ValueError("Teacher-forced causal path is for the deep head")
        context = targets[:, :-1]
        length = context.shape[1]
        causal = torch.triu(torch.ones(length, length, device=memory.device, dtype=torch.bool), 1)
        return self.decode(context, memory, content_mask=causal, query_mask=causal,
                           padding=context == self.tokenizer.pad_id)

    def greedy(self, memory):
        n, steps = len(memory), self.max_label_length + 1
        context = torch.full((n, 1), self.tokenizer.bos_id, device=memory.device, dtype=torch.long)
        done = torch.zeros(n, device=memory.device, dtype=torch.bool)
        outputs = []
        for i in range(steps):
            length = i + 1
            causal = torch.triu(torch.ones(length, length, device=memory.device, dtype=torch.bool), 1)
            logits = self.decode(context, memory, content_mask=causal, query_mask=causal[-1:],
                query=self.pos_queries[:, i:i+1].expand(n, -1, -1))
            outputs.append(logits)
            token = logits[:, 0].argmax(-1)
            done |= token == self.tokenizer.eos_id
            context = torch.cat([context, token[:, None]], 1)
            if bool(done.all()):
                break
        return torch.cat(outputs, 1)

    def read(self, logits):
        probabilities = logits.detach().softmax(-1)
        texts, _ = self.tokenizer.decode(probabilities)
        values, ids = probabilities.max(-1)
        rows = []
        for text, score, indices in zip(texts, values.cpu().tolist(), ids.cpu().tolist()):
            length = indices.index(self.tokenizer.eos_id) if self.tokenizer.eos_id in indices else len(indices)
            characters = score[:length]
            rows.append({"text": text, "token_ids": indices[:length], "character_probabilities": characters,
                "minimum_probability": min(characters) if characters else 0.,
                "mean_probability": float(np.mean(characters)) if characters else 0.,
                "eos_reached": self.tokenizer.eos_id in indices})
        return rows


class CorrectnessGate(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, cues):
        if cues.ndim != 2 or cues.shape[1] != 7 or not torch.isfinite(cues).all():
            raise ValueError("Seven finite predicted-only gate cues required")
        return self.layers(cues.detach()).squeeze(-1)


class AdaSpotJoint(nn.Module):
    def __init__(self, parts, tokenizer, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.backbone = JointBackbone(skip_floor=config["skip_floor"])
        self.gap = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.gap.weight)
        nn.init.constant_(self.gap.bias, config["initial_gap_bias"])
        options = {key: config[key] for key in ("embed_dim", "num_heads", "mlp_ratio", "dropout", "max_label_length")}
        self.light_head = FeatureOCRHead(parts, tokenizer, depth=1, autoregressive=False, channels=64, **options)
        self.deep_head = FeatureOCRHead(parts, tokenizer, depth=4, autoregressive=True, channels=64, **options)
        self.gate = CorrectnessGate(config["gate_hidden"])

    def encode(self, image):
        evidence = self.backbone.forward_evidence(image)
        evidence["gap"] = self.gap(evidence["features"])
        evidence["coarse_features"] = F.avg_pool2d(evidence["features"], 2)
        return evidence

    def align(self, evidence, masks, *, view, options, deep):
        if not masks:
            shape = (16, 64) if deep else (8, 32)
            return evidence["features"].new_empty((0, 64, *shape)), []
        geometries = [mask_align_geometry(mask, view=view, options=options) for mask in masks]
        grids = np.stack([item.grid((16, 64) if deep else (8, 32)) for item in geometries])
        if not deep:
            grids = (grids + .5) / 2 - .5
        features = evidence["features"] if deep else evidence["coarse_features"]
        sampled = sample_mask_aligned(features, torch.from_numpy(grids).to(features.device),
            deterministic_cpu_backward=self.config["deterministic_cpu_backward"])
        return sampled, geometries

    def cues(self, evidence, masks, geometries, first, refined):
        initial, final = self.light_head.read(first), self.light_head.read(refined)
        probabilities = evidence["words"][0].detach().sigmoid()
        rows = []
        for mask, geometry, a, b in zip(masks, geometries, initial, final):
            support = torch.from_numpy(mask).to(probabilities.device)
            mean_text, mean_kernel = probabilities[:, support].mean(1).cpu().tolist()
            grid = geometry.grid((3, 32))
            center = grid[1]
            vectors = np.diff(center, axis=0)
            angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
            curvature = float(np.abs(np.diff(angles)).sum() / math.pi)
            aspect = float(np.linalg.norm(vectors, axis=1).sum() / max(1e-6, 1.5 * np.linalg.norm(grid[2] - grid[0], axis=1).mean()))
            x, y = a["token_ids"], b["token_ids"]
            disagreement = (abs(len(x) - len(y)) + sum(u != v for u, v in zip(x, y))) / max(1, len(x), len(y))
            rows.append([b["minimum_probability"], b["mean_probability"], disagreement,
                         mean_text, mean_kernel, curvature, math.log1p(len(y) / max(aspect, 1e-6))])
        return probabilities.new_tensor(rows).reshape(-1, 7), final

    @torch.no_grad()
    def infer_instances(self, evidence, masks, *, options, temperature, reject_threshold,
                        keep_threshold, deep_threshold, view="mask"):
        """Run deep decoding ONLY for gate-escalated instances; preserve order."""
        if not 0 <= reject_threshold <= keep_threshold <= 1 or not 0 <= deep_threshold <= 1 or temperature <= 0:
            raise ValueError("Invalid calibrated routing parameters")
        if not masks:
            return []
        aligned, geometry = self.align(evidence, masks, view=view, options=options, deep=False)
        first, refined = self.light_head.light(self.light_head.memory(aligned))
        cues, light = self.cues(evidence, masks, geometry, first, refined)
        probability = (self.gate(cues) / temperature).sigmoid().cpu().tolist()
        escalated = [i for i, p in enumerate(probability) if reject_threshold <= p < keep_threshold]
        deep = {}
        if escalated:
            sampled, _ = self.align(evidence, [masks[i] for i in escalated], view=view, options=options, deep=True)
            deep = dict(zip(escalated, self.deep_head.read(self.deep_head.greedy(self.deep_head.memory(sampled)))))
        result = []
        for i, (p, reading) in enumerate(zip(probability, light)):
            route = "keep" if p >= keep_threshold else "reject" if p < reject_threshold else "escalate"
            chosen = deep[i] if i in deep else reading
            emitted = route != "reject" and bool(chosen["text"]) and (route != "escalate" or chosen["minimum_probability"] >= deep_threshold)
            result.append({"text": chosen["text"] if emitted else "", "emitted": emitted,
                "route": route, "gate_probability": p, "light": reading, "deep": deep.get(i),
                "geometry": geometry[i].audit, "gate_cues": cues[i].cpu().tolist()})
        return result
