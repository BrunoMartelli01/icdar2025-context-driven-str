"""Read-only dense-arithmetic counting, including functional attention projections."""
from __future__ import annotations

from collections import Counter

import torch
from torch import nn


def multihead_attention_macs(batch: int, queries: int, keys: int, width: int) -> int:
    if min(batch, queries, keys, width) < 1:
        raise ValueError("attention dimensions must be positive")
    # Q/K/V/output projections, plus QK and attention-value matrix products.
    return batch * (2 * (queries + keys) * width * width + 2 * queries * keys * width)


def letterbox_shape(image_wh, size: int, stride: int) -> list[int]:
    width, height = image_wh
    if min(width, height, size, stride) < 1 or size % stride:
        raise ValueError("invalid single-image LetterBox geometry")
    ratio = min(size / height, size / width)
    resized_w, resized_h = round(width * ratio), round(height * ratio)
    return [1, 3, resized_h + (size - resized_h) % stride,
            resized_w + (size - resized_w) % stride]


class DenseMacCounter:
    """Attach non-mutating hooks; skip MHA children to prevent double counting."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.counts = Counter()
        self.events = []

    def _record(self, name, kind, macs, inputs, output):
        if not isinstance(macs, int) or macs < 0:
            raise ValueError("MAC count must be a nonnegative integer")
        self.counts[kind] += macs
        self.events.append({"module": name, "kind": kind, "macs": macs,
            "input_shapes": [list(value.shape) for value in inputs if isinstance(value, torch.Tensor)],
            "output_shape": list(output.shape) if isinstance(output, torch.Tensor) else None})

    def __enter__(self):
        self.counts.clear()
        self.events.clear()
        mha_prefixes = [name + "." if name else "" for name, module in self.model.named_modules()
                        if isinstance(module, nn.MultiheadAttention)]
        for name, module in self.model.named_modules():
            if name and any(not prefix or name.startswith(prefix) for prefix in mha_prefixes):
                continue
            if isinstance(module, nn.Conv2d):
                def hook(layer, inputs, output, key=name):
                    k = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels // layer.groups
                    self._record(key, "convolution", int(output.numel() * k), inputs, output)
            elif isinstance(module, nn.Linear):
                def hook(layer, inputs, output, key=name):
                    self._record(key, "linear", int(output.numel() * layer.in_features), inputs, output)
            elif isinstance(module, nn.MultiheadAttention):
                def hook(layer, inputs, output, key=name):
                    if not layer._qkv_same_embed_dim or any(x.ndim != 3 for x in inputs[:3]):
                        raise ValueError("unsupported MHA projection geometry")
                    q, k, v = inputs[:3]
                    axis = 1 if layer.batch_first else 0
                    batch = q.shape[0 if layer.batch_first else 1]
                    if k.shape[axis] != v.shape[axis] or q.shape[-1] != layer.embed_dim:
                        raise ValueError("MHA dimensions differ from qualified attention")
                    macs = multihead_attention_macs(batch, q.shape[axis], k.shape[axis], layer.embed_dim)
                    self._record(key, "multihead_attention", int(macs), inputs, output[0])
            elif (type(module).__module__ == "timm.models.vision_transformer"
                  and type(module).__name__ == "Attention"):
                def hook(layer, inputs, output, key=name):
                    x = inputs[0]
                    if x.ndim != 3 or not hasattr(layer, "qkv") or not hasattr(layer, "proj"):
                        raise ValueError("unsupported ViT attention implementation")
                    batch, tokens, width = x.shape
                    self._record(key, "vit_attention_products", int(2 * batch * tokens * tokens * width), inputs, output)
            else:
                continue
            self.handles.append(module.register_forward_hook(hook))
        return self

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> dict:
        return {"macs": self.total, "flops": 2 * self.total,
                "macs_by_kind": dict(self.counts), "events": list(self.events)}

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
