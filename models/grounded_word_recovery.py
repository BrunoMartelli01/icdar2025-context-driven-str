"""Small RGB/text verification and variable-length edit models, APPROXIMATE."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from recognition.grounded_word_recovery import KEEP, DELETE, STOP, BOS, PAD, VOCABULARY


class GroundedVerifier(nn.Module):
    def __init__(self, visual_dimensions, hidden, layers, maximum_bytes):
        super().__init__(); self.maximum_bytes = maximum_bytes
        self.image = nn.Linear(visual_dimensions, hidden)
        self.text = nn.Embedding(VOCABULARY, hidden, padding_idx=PAD)
        self.position = nn.Embedding(maximum_bytes+1, hidden)
        block = nn.TransformerDecoderLayer(hidden, 4, 4*hidden, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoder(block, layers)
        self.geometry = nn.Sequential(nn.Linear(2*hidden+12, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.joint = nn.Sequential(nn.Linear(3*hidden+12, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, image, context, text, geometry):
        if image.ndim != 3 or image.shape != context.shape or geometry.shape != (len(image), 12):
            raise ValueError('Unaligned verifier observations')
        a, b = self.image(image), self.image(context)
        memory = torch.cat((a, b), 1)
        query = self.text(text) + self.position(torch.arange(text.shape[1], device=text.device))[None]
        mask = text == PAD
        decoded = self.decoder(query, memory, tgt_key_padding_mask=mask)
        pooled = (decoded * (~mask)[..., None]).sum(1) / (~mask).sum(1, keepdim=True).clamp_min(1)
        visual = torch.cat((a.mean(1), b.mean(1), geometry), -1)
        joint = self.joint(torch.cat((pooled, visual), -1))
        return torch.cat((self.geometry(visual), joint), -1)  # geometry, spelling, joint


class VisualEditTransducer(nn.Module):
    """Autoregressive KEEP/DELETE/INSERT-byte/STOP; explicit source cursor."""
    def __init__(self, visual_dimensions, hidden, layers, maximum_bytes, *, visual):
        super().__init__(); self.visual = visual; self.maximum_bytes = maximum_bytes
        self.embedding = nn.Embedding(VOCABULARY, hidden, padding_idx=PAD)
        self.image = nn.Linear(visual_dimensions, hidden)
        self.position = nn.Embedding(2*maximum_bytes+3, hidden)
        self.cursor = nn.Embedding(maximum_bytes+1, hidden)
        self.modality = nn.Parameter(torch.zeros(2, hidden))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, 4, 4*hidden,
            dropout=0., batch_first=True), layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden, 4, 4*hidden,
            dropout=0., batch_first=True), layers)
        self.output = nn.Linear(hidden, VOCABULARY)
        with torch.no_grad(): self.output.bias[KEEP] = 2.

    def encode(self, source, image):
        text = self.embedding(source)+self.position(torch.arange(source.shape[1], device=source.device))[None]+self.modality[0]
        visual = self.image(image if self.visual else torch.zeros_like(image))+self.modality[1]
        memory = torch.cat((text, visual), 1)
        padding = torch.cat((source == PAD, torch.zeros(image.shape[:2], dtype=torch.bool, device=image.device)), 1)
        return self.encoder(memory, src_key_padding_mask=padding), padding

    def decode(self, memory, padding, actions, cursor):
        if cursor.max() > self.maximum_bytes: raise ValueError('Edit cursor overflow')
        text = self.embedding(actions)+self.position(torch.arange(actions.shape[1], device=actions.device))[None]+self.cursor(cursor)
        mask = torch.triu(torch.ones(actions.shape[1], actions.shape[1], dtype=torch.bool, device=actions.device), 1)
        return self.output(self.decoder(text, memory, tgt_mask=mask, memory_key_padding_mask=padding))

    def forward(self, source, image, actions, cursor):
        return self.decode(*self.encode(source, image), actions, cursor)

    @torch.inference_mode()
    def generate(self, source, image, lengths):
        self.eval(); memory, padding = self.encode(source, image); n = len(source)
        actions = torch.full((n, 1), BOS, dtype=torch.long, device=source.device)
        cursors = torch.zeros((n, 1), dtype=torch.long, device=source.device)
        pointer = torch.zeros(n, dtype=torch.long, device=source.device)
        emitted = torch.zeros_like(pointer); finished = torch.zeros(n, dtype=torch.bool, device=source.device)
        lengths = torch.as_tensor(lengths, device=source.device); generated = []
        for _ in range(2*self.maximum_bytes+1):
            logits = self.decode(memory, padding, actions, cursors)[:, -1].clone()
            logits[:, BOS:] = -torch.inf
            logits[pointer >= lengths, KEEP:STOP] = -torch.inf
            logits[pointer < lengths, STOP] = -torch.inf
            logits[emitted >= self.maximum_bytes, :256] = -torch.inf
            logits[emitted >= self.maximum_bytes, KEEP] = -torch.inf
            logits[finished] = -torch.inf; logits[finished, STOP] = 0.
            step = logits.argmax(-1); generated.append(step)
            output = (step < 256) | (step == KEEP)
            emitted += output & ~finished
            pointer += ((step == KEEP) | (step == DELETE)) & ~finished
            finished |= step == STOP
            actions = torch.cat((actions, step[:, None]), 1)
            cursors = torch.cat((cursors, pointer[:, None]), 1)
            if finished.all(): break
        if not finished.all(): raise RuntimeError('Bounded edit generator failed to terminate')
        rows = torch.stack(generated, 1).cpu().tolist()
        return [row[:row.index(STOP)+1] for row in rows]
