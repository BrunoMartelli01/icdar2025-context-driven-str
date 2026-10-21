"""APPROXIMATE AdaSpot Mask-Align: mask-only geometry, bilinear feature sampling.

Geometry is discrete and detached. Gradients flow through sampled features only.
See docs/ADASpot_MASK_ALIGN.md for evidence, coordinate conventions and limits.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
import torch
from torch.nn import functional as F

from segmentation.word_centerline import skeletonize_mask


@dataclass(frozen=True)
class MaskAlignOptions:
    radius_floor: float
    maximum_outside_fraction: float
    reverse_step_tolerance: float

    def __post_init__(self):
        values = (self.radius_floor, self.maximum_outside_fraction, self.reverse_step_tolerance)
        if (not np.isfinite(values).all() or self.radius_floor <= 0
                or not 0 <= self.maximum_outside_fraction <= 1 or self.reverse_step_tolerance < 0):
            raise ValueError("Invalid explicit Mask-Align geometry options")


@dataclass
class MaskAlignGeometry:
    origin: np.ndarray
    axis: np.ndarray
    limits: tuple
    coefficients: np.ndarray
    radius_u: np.ndarray
    radii: np.ndarray
    audit: dict

    def grid(self, output_hw):
        """Sample bin centers; coordinates refer to pixel centers, xy order."""
        height, width = output_hw
        if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 1:
            raise ValueError("Positive integer output height/width required")
        u = self.limits[0] + (np.arange(width) + .5) / width * (self.limits[1] - self.limits[0])
        normal = np.array([-self.axis[1], self.axis[0]])
        value = np.polynomial.polynomial.polyval(u, self.coefficients)
        derivative = np.polynomial.polynomial.polyval(u, np.arange(1, len(self.coefficients)) * self.coefficients[1:])
        centers = self.origin + u[:, None] * self.axis + value[:, None] * normal
        normals = normal - derivative[:, None] * self.axis
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        radius = np.interp(u, self.radius_u, self.radii)
        across = 2 * (np.arange(height) + .5) / height - 1
        grid = centers[None] + across[:, None, None] * radius[None, :, None] * normals[None]
        if not np.isfinite(grid).all():
            raise RuntimeError("Nonfinite Mask-Align grid")
        return grid.astype(np.float32)


def _oriented_axis(points):
    corners = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
    edges = np.roll(corners, -1, axis=0) - corners
    axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))].astype(float)
    if np.linalg.norm(axis) < 1e-8:
        axis = np.array([1., 0.])
    axis /= np.linalg.norm(axis)
    sign_coordinate = 0 if abs(axis[0]) >= abs(axis[1]) else 1
    if axis[sign_coordinate] < 0:
        axis = -axis
    return axis


def _rectangle(points, axis, options, reason, skeleton_pixels=0):
    normal = np.array([-axis[1], axis[0]])
    u, v = points @ axis, points @ normal
    # Cover the complete square pixel support, including singletons/lines.
    du, dv = .5 * np.abs(axis).sum(), .5 * np.abs(normal).sum()
    limits = (float(u.min() - du), float(u.max() + du))
    origin = normal * ((v.min() + v.max()) / 2)
    radius = max(options.radius_floor, (v.max() - v.min()) / 2 + dv)
    return MaskAlignGeometry(origin, axis, limits, np.array([0., 0.]),
        np.array(limits), np.array([radius, radius]),
        {"method": "axis" if reason == "axis_requested" else "rotated_rectangle",
         "fallback_reason": reason, "skeleton_pixels": skeleton_pixels})


def _chain(skeleton):
    nodes = {tuple(p) for p in np.argwhere(skeleton)}
    graph = {}
    for y, x in nodes:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                other = (y + dy, x + dx)
                if (dy == dx == 0) or other not in nodes:
                    continue
                # Remove redundant diagonal shortcuts through an axial neighbor.
                if dy and dx and ((y + dy, x) in nodes or (y, x + dx) in nodes):
                    continue
                neighbors.append(other)
        graph[y, x] = sorted(neighbors)
    if any(len(v) > 2 for v in graph.values()):
        return None, "branched_skeleton"
    endpoints = sorted(k for k, v in graph.items() if len(v) == 1)
    if len(endpoints) != 2:
        return None, "cycle_or_degenerate_skeleton"
    path, previous, current = [], None, endpoints[0]
    while True:
        path.append(current)
        following = [p for p in graph[current] if p != previous]
        if not following:
            break
        previous, current = current, following[0]
        if len(path) > len(nodes):
            raise RuntimeError("Unexpected skeleton traversal cycle")
    if len(path) != len(nodes):
        return None, "disconnected_skeleton"
    return np.asarray(path), None


def mask_align_geometry(mask, *, view, options):
    """Fit one predicted instance; no image, text, owner or annotation argument."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Nonempty boolean HxW instance mask required")
    if view not in ("axis", "rotated", "mask"):
        raise ValueError("Unknown Mask-Align geometry view")
    yy, xx = np.where(mask)
    points = np.column_stack((xx, yy)).astype(float)
    axis = np.array([1., 0.]) if view == "axis" else _oriented_axis(points)
    if view != "mask":
        return _rectangle(points, axis, options, view + "_requested")
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    support = mask[y0:y1, x0:x1]
    count, _ = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if count != 2:
        return _rectangle(points, axis, options, "disconnected_mask")
    skeleton = skeletonize_mask(support)
    path, reason = _chain(skeleton)
    skeleton_pixels = int(skeleton.sum())
    def fallback(why):
        return _rectangle(points, axis, options, why, skeleton_pixels)
    if reason:
        return fallback(reason)
    if len(path) < 4:
        return fallback("short_skeleton")
    # Principal-axis graph cubic: scaled u keeps least squares well conditioned.
    origin = points.mean(0)
    normal = np.array([-axis[1], axis[0]])
    xy = path[:, ::-1] + np.array([x0, y0])
    u, v = (xy - origin) @ axis, (xy - origin) @ normal
    if u[-1] < u[0]:
        u, v, path = u[::-1], v[::-1], path[::-1]
    if np.min(np.diff(u)) < -options.reverse_step_tolerance:
        return fallback("folded_centerline")
    unique, inverse = np.unique(u, return_inverse=True)
    if len(unique) < 4 or np.ptp(u) < 1:
        return fallback("rank_deficient_cubic")
    scale = float(np.ptp(u))
    design = np.polynomial.polynomial.polyvander(u / scale, 3)
    fit, _, rank, _ = np.linalg.lstsq(design, v, rcond=None)
    if rank != 4:
        return fallback("rank_deficient_cubic")
    coefficients = fit / scale ** np.arange(4)
    probe_u = np.linspace(u.min(), u.max(), 64)
    probe_v = np.polynomial.polynomial.polyval(probe_u, coefficients)
    probe = origin + probe_u[:, None] * axis + probe_v[:, None] * normal - [x0, y0]
    inside = map_coordinates(support.astype(float), [probe[:, 1], probe[:, 0]], order=0,
                             mode="constant", cval=0)
    if np.mean(inside < .5) > options.maximum_outside_fraction:
        return fallback("cubic_outside_mask")
    distance = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    radius = distance[path[:, 0], path[:, 1]]
    means = np.bincount(inverse, weights=radius) / np.bincount(inverse)
    projected = (points - origin) @ axis
    extent = .5 * np.abs(axis).sum()
    return MaskAlignGeometry(origin, axis, (float(projected.min() - extent), float(projected.max() + extent)),
        coefficients, unique, np.maximum(means, options.radius_floor),
        {"method": "cubic_centerline", "fallback_reason": None, "skeleton_pixels": skeleton_pixels,
         "fitted_outside_fraction": float(np.mean(inside < .5)),
         "radius_min": float(means.min()), "radius_max": float(means.max())})


def native_grid_to_rgb(grid, image_wh, native_hw):
    """Invert the existing centered square letterbox, preserving half pixels."""
    width, height = image_wh
    nh, nw = native_hw
    if min(width, height, nh, nw) < 1 or nh != nw:
        raise ValueError("Positive RGB dimensions and square native map required")
    side = max(width, height)
    padding = np.array([(side - width) // 2, (side - height) // 2])
    return ((np.asarray(grid) + .5) * (side / nw) - padding - .5).astype(np.float32)


def sample_mask_aligned(features, grids, *, deterministic_cpu_backward):
    """Sample one image's BCHW=1 features into NCHW, differentiable in F.

    The explicit CPU-backward option transfers tensors with autograd intact;
    inference otherwise executes on the input device. No silent fallback.
    """
    if (features.ndim != 4 or features.shape[0] != 1 or grids.ndim != 4 or grids.shape[-1] != 2
            or features.dtype != torch.float32 or grids.dtype != torch.float32
            or features.device != grids.device):
        raise ValueError("Single-image FP32 BCHW features and NxHxWx2 grids on the same device required")
    if not torch.isfinite(grids).all():
        raise ValueError("Nonfinite sampling grid")
    height, width = features.shape[-2:]
    normalized = 2 * (grids + .5) / grids.new_tensor([width, height]) - 1
    device = features.device
    if deterministic_cpu_backward and torch.is_grad_enabled() and features.requires_grad:
        features, normalized = features.cpu(), normalized.cpu()
    output = F.grid_sample(features.expand(len(grids), -1, -1, -1), normalized,
                           mode="bilinear", padding_mode="zeros", align_corners=False)
    return output.to(device)
