"""Approximate matched encoder ablations; inputs contain predictions only."""
import numpy as np
import torch
from torch import nn


class EvidenceBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(hidden*2, hidden), nn.Dropout(dropout))

    def forward(self, words, memory, word_padding, memory_padding):
        x = self.norms[0](words)
        words = words + self.cross(x, memory, memory, key_padding_mask=memory_padding, need_weights=False)[0]
        x = self.norms[1](words)
        words = words + self.self_attention(x, x, x, key_padding_mask=word_padding, need_weights=False)[0]
        return words + self.ff(self.norms[2](words))


class EncoderFamilyRouter(nn.Module):
    """Common UTF8/geometry head with explicitly optional frozen context inputs."""
    def __init__(self, arm, settings, seed):
        super().__init__()
        if arm not in {'none', 'clip', 'blip', 'both'}:
            raise ValueError('Unknown encoder family')
        self.arm = arm
        c = settings; h = c['hidden']
        # Initialize shared modules first, identically for every arm.
        self.embedding = nn.Embedding(257, c['byte_embedding'], padding_idx=256)
        self.byte_conv = nn.Conv1d(c['byte_embedding'], c['byte_channels'], c['byte_kernel'], padding=c['byte_kernel']//2)
        self.word_projection = nn.Linear(2*c['byte_channels']+c['geometry_dim'], h)
        self.decision = nn.Parameter(torch.randn(1, 1, h)*c['initialization_std'])
        self.null_memory = nn.Parameter(torch.randn(1, 1, h)*c['initialization_std'])
        self.blocks = nn.ModuleList([EvidenceBlock(h, c['heads'], c['dropout']) for _ in range(c['blocks'])])
        self.output = nn.Sequential(nn.LayerNorm(2*h+c['scalar_dim']), nn.Linear(2*h+c['scalar_dim'], c['mlp']),
                                    nn.GELU(), nn.Dropout(c['dropout']), nn.Linear(c['mlp'], c['mlp']),
                                    nn.GELU(), nn.Linear(c['mlp'], 2))
        if arm in {'clip', 'both'}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+1)
                self.clip_patch = nn.Linear(c['clip_dim']+4, h)
                self.clip_word = nn.Linear(2*c['clip_dim'], h)
        if arm in {'blip', 'both'}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+2)
                self.blip_projection = nn.Linear(c['query_dim'], h)
                self.query_position = nn.Parameter(torch.randn(1, c['query_count'], h)*c['initialization_std'])

    def forward(self, batch):
        ids = batch['bytes']; b, w, length = ids.shape
        valid = ids.ne(256).reshape(b*w, length)
        x = self.embedding(ids).reshape(b*w, length, -1).transpose(1, 2)
        x = torch.nn.functional.gelu(self.byte_conv(x))
        mean = (x*valid[:, None]).sum(-1)/valid.sum(-1).clamp_min(1)[:, None]
        maximum = x.masked_fill(~valid[:, None], -1e4).max(-1).values
        maximum = torch.where(valid.any(-1)[:, None], maximum, torch.zeros_like(maximum))
        words = self.word_projection(torch.cat([torch.cat([mean, maximum], -1).reshape(b, w, -1), batch['geometry']], -1))
        memory = [self.null_memory.expand(b, -1, -1)]
        memory_padding = [torch.zeros(b, 1, dtype=torch.bool, device=ids.device)]
        if self.arm in {'clip', 'both'}:
            words = words + self.clip_word(torch.cat([batch['word_text'], batch['word_image']], -1))
            memory.append(self.clip_patch(torch.cat([batch['patches'], batch['patch_geometry']], -1)))
            memory_padding.append(~batch['patch_valid'])
        if self.arm in {'blip', 'both'}:
            memory.append(self.blip_projection(batch['queries'])+self.query_position)
            memory_padding.append(torch.zeros(batch['queries'].shape[:2], dtype=torch.bool, device=ids.device))
        words = torch.cat([self.decision.expand(b, -1, -1), words], 1)
        padding = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=ids.device), batch['word_padding']], 1)
        memory = torch.cat(memory, 1); memory_padding = torch.cat(memory_padding, 1)
        for block in self.blocks:
            words = block(words, memory, padding, memory_padding)
        keep = (~batch['word_padding']).float()
        pooled = (words[:, 1:]*keep[..., None]).sum(1)/keep.sum(1).clamp_min(1)[:, None]
        return self.output(torch.cat([words[:, 0], pooled, batch['scalars']], -1))


def collate_observations(rows, arm, device):
    """Preserve every predicted word and UTF8 byte, with finite empty handling."""
    n = len(rows); words = max(1, max(len(r['texts']) for r in rows))
    length = max(1, max((len(t.encode('utf8')) for r in rows for t in r['texts']), default=1))
    byte_ids = np.full((n, words, length), 256, dtype=np.int64)
    geometry = np.zeros((n, words, 12), np.float32); padding = np.ones((n, words), bool)
    for i, row in enumerate(rows):
        count = len(row['texts']); geometry[i, :count] = row['geometry']; padding[i, :count] = False
        for j, text in enumerate(row['texts']):
            data = text.encode('utf8'); byte_ids[i, j, :len(data)] = list(data)
    result = dict(bytes=byte_ids, geometry=geometry, word_padding=padding,
                  scalars=np.stack([r['scalars'] for r in rows]))
    if arm in {'clip', 'both'}:
        for key in ('word_text', 'word_image'):
            values = np.zeros((n, words, 512), np.float32)
            for i, row in enumerate(rows): values[i, :len(row['texts'])] = row[key]
            result[key] = values
        for key in ('patches', 'patch_geometry', 'patch_valid'):
            result[key] = np.stack([r[key] for r in rows])
    if arm in {'blip', 'both'}:
        result['queries'] = np.stack([r['queries'] for r in rows])
    return {k: torch.as_tensor(v, device=device) for k, v in result.items()}


def count_targets(branches, ids):
    return np.asarray([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
                        branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']]
                       for s in ids], np.float32)
