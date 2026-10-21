"""Isolated frozen-feature Gap heads and matched TRAIN-only pixel strata."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GapContextHead(nn.Module):
    """A 1x1 or 3x3 linear convolution, trained on aligned 3x3 feature patches."""

    def __init__(self, kernel_size, *, initial_bias, channels=64):
        super().__init__()
        if kernel_size not in (1, 3) or channels < 1 or not math.isfinite(initial_bias):
            raise ValueError("Declared 1x1/3x3 kernel, channels and finite bias required")
        self.kernel_size, self.channels = kernel_size, channels
        self.linear = nn.Linear(channels * kernel_size ** 2, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, initial_bias)

    def forward(self, patches):
        if patches.ndim != 2 or patches.shape[1] != self.channels * 9:
            raise ValueError("Expected Nx(C*9) channel-major 3x3 patches")
        return self.linear(patches[:, 4::9] if self.kernel_size == 1 else patches)[:, 0]

    def native_logits(self, features):
        if features.ndim != 3 or features.shape[0] != self.channels:
            raise ValueError("Expected native CxHxW features")
        weight = self.linear.weight.reshape(1, self.channels, self.kernel_size, self.kernel_size)
        return F.conv2d(features[None], weight, self.linear.bias, padding=self.kernel_size // 2)[0, 0]


def sampled_patches(features, positions):
    """Gather zero-padded neighbourhoods without materializing a full unfold."""
    if features.ndim != 3 or features.device.type != "cpu":
        raise ValueError("CPU CxHxW frozen features required")
    positions = torch.as_tensor(positions, dtype=torch.long)
    channels, height, width = features.shape
    if positions.ndim != 1 or torch.any((positions < 0) | (positions >= height * width)):
        raise ValueError("Pixel indices outside native feature extent")
    yy, xx = positions // width, positions % width
    padded = F.pad(features, (1, 1, 1, 1))
    offsets = [padded[:, yy + dy, xx + dx].T for dy in range(3) for dx in range(3)]
    return torch.stack(offsets, dim=-1).reshape(len(positions), channels * 9)


def bridge_strata(target, validity, predicted_text, rng, negative_floor):
    """Hard/easy Gap positives and predicted-text/background negatives.

    Hard positive means a valid target-Gap pixel mistaken for text; it is a
    bridge proxy, not a guarantee that removing the pixel disconnects a word CC.
    GT enters training sampling only; native head inference accepts features only.
    """
    target, validity, predicted_text = [np.asarray(v) for v in (target, validity, predicted_text)]
    if (target.ndim != 2 or target.shape != validity.shape or target.shape != predicted_text.shape
            or any(v.dtype != bool for v in (target, validity, predicted_text))
            or type(negative_floor) is not int or negative_floor < 1):
        raise ValueError("Paired boolean TRAIN masks and positive sampling floor required")
    if np.any(target & ~validity):
        raise ValueError("Gap positive outside valid supervision")
    hard, easy = np.flatnonzero(target & predicted_text), np.flatnonzero(target & ~predicted_text)
    count = max(negative_floor, len(hard) + len(easy))
    groups = [hard, easy]
    for mask in (validity & ~target & predicted_text, validity & ~target & ~predicted_text):
        population = np.flatnonzero(mask)
        groups.append(rng.choice(population, min(count, len(population)), replace=False))
    return groups


def stratum_weights(classes):
    if classes.ndim != 1 or classes.dtype != torch.long or torch.any((classes < 0) | (classes > 3)):
        raise ValueError("Four integer TRAIN strata required")
    counts = torch.bincount(classes, minlength=4)
    if torch.any(counts == 0):
        raise ValueError("Every TRAIN stratum needs samples; no silent reweighting fallback")
    return len(classes) / (4 * counts.float()), counts
