"""Explicit probability averaging on the native U-Net half-pixel grid."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def fuse(native, larger, weight):
    for array in (native, larger):
        if (array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != 2 or
            array.shape[1] != array.shape[2] or not np.isfinite(array).all() or
            array.min() < 0 or array.max() > 1):
            raise ValueError('Finite FP32 two-channel square probability maps required')
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Explicit probability weight in[0,1] required')
    aligned = F.interpolate(torch.from_numpy(larger)[None], size=native.shape[1:],
        mode='bilinear', align_corners=False)[0].numpy()
    return ((1-weight)*native+weight*aligned).astype(np.float32)
