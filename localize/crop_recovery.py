"""GT-blind crop hypotheses from probability topology and RGB evidence.

Experimental approximate localization. All coordinates are exclusive xyxy;
native maps describe a square letterbox, never a direct RGB resize.
"""
from __future__ import annotations

from hashlib import sha256
import heapq
import json

import cv2
import numpy as np

from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box


METHODS = ("baseline", "threshold_tree", "erosion_reconstruction", "image_seams",
           "tiled_segmentation", "crop_variants")


def components(mask: np.ndarray) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [labels == index for index in range(1, count)]


def tight_box(mask: np.ndarray) -> list[int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("empty component")
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def geodesic_partition(probability: np.ndarray, support: np.ndarray,
                       seeds: np.ndarray) -> np.ndarray:
    """Deterministic seeded minimax flood, constrained to foreground support.

    A path's cost is its largest (1-probability) value. Length breaks cost ties;
    label ID breaks remaining ties. Every support pixel reachable from a seed
    is assigned, while background can never provide a shortcut.
    """
    if probability.shape != support.shape or seeds.shape != support.shape:
        raise ValueError("partition shapes differ")
    if np.any((seeds > 0) & ~support):
        raise ValueError("seeds outside foreground")
    labels = np.zeros(seeds.shape, np.int32)
    queue = [(float(1 - probability[y, x]), 0, int(seeds[y, x]), int(y), int(x))
             for y, x in zip(*np.nonzero(seeds))]
    heapq.heapify(queue)
    height, width = support.shape
    while queue:
        cost, distance, label, y, x = heapq.heappop(queue)
        if labels[y, x]:
            continue
        labels[y, x] = label
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and support[ny, nx] and not labels[ny, nx]:
                # Seed interiors keep their original identity.
                if seeds[ny, nx] and seeds[ny, nx] != label:
                    continue
                heapq.heappush(queue, (max(cost, float(1 - probability[ny, nx])),
                                     distance + 1, label, ny, nx))
    return labels


def adjacent_groups(regions: list[np.ndarray], maximum: int) -> list[np.ndarray]:
    """Keep each partition and short unions along its principal spatial axis."""
    if not regions:
        return []
    centers = np.array([[np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()] for m in regions])
    if len(regions) == 1:
        return regions
    _, _, axes = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.argsort(centers @ axis, kind="stable")
    ordered = [regions[i] for i in order]
    output = list(ordered)
    for size in range(2, min(maximum, len(ordered)) + 1):
        output.extend(np.logical_or.reduce(ordered[start:start + size])
                      for start in range(len(ordered) - size + 1))
    return output


def vertical_seam(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost top-to-bottom path with horizontal steps of at most one."""
    if cost.ndim != 2 or not cost.size or not np.isfinite(cost).all():
        raise ValueError("invalid seam cost")
    height, width = cost.shape
    previous = cost[0].astype(np.float64)
    parents = np.zeros((height, width), np.int32)
    for y in range(1, height):
        neighbors = np.stack((np.r_[np.inf, previous[:-1]], previous,
                              np.r_[previous[1:], np.inf]))
        step = neighbors.argmin(0)
        parents[y] = np.arange(width) + step - 1
        previous = cost[y] + neighbors[step, np.arange(width)]
    path = np.zeros(height, np.int32)
    path[-1] = int(previous.argmin())
    for y in range(height - 1, 0, -1):
        path[y - 1] = parents[y, path[y]]
    return path


def seam_partitions(probability: np.ndarray, support: np.ndarray, rgb: np.ndarray,
                    config: dict) -> list[tuple[np.ndarray, dict]]:
    """Try several RGB/probability corridors in both axial orientations."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    gradient = np.clip(gradient / max(float(np.quantile(gradient, .95)), 1e-6), 0, 1)
    weight = config["probability_weight"]
    cost = weight * probability + (1 - weight) * gradient
    output = []
    for transpose in (False, True):
        current = cost.T if transpose else cost
        foreground = support.T if transpose else support
        height, width = current.shape
        if min(height, width) < 3:
            continue
        margin = max(1, int(round(width * config["interior_fraction"])))
        if width - margin <= margin:
            continue
        profile = cv2.GaussianBlur(current.mean(0)[None], (3, 1), 0)[0]
        seeds = []
        spacing = max(1, int(round(width * config["minimum_spacing_fraction"])))
        for column in sorted(range(margin, width - margin), key=lambda i: (float(profile[i]), i)):
            if all(abs(column - old) >= spacing for old in seeds):
                seeds.append(column)
            if len(seeds) == config["maximum_seams_per_axis"]:
                break
        cuts = []
        radius = max(1, int(round(width * config["search_band_fraction"])))
        for column in seeds:
            left, right = max(margin, column - radius), min(width - margin, column + radius + 1)
            path = vertical_seam(current[:, left:right]) + left
            removed = np.zeros(foreground.shape, bool)
            # A three-pixel band prevents diagonal 8-connectivity bypass.
            for offset in range(-config["cut_half_width"], config["cut_half_width"] + 1):
                removed[np.arange(height), np.clip(path + offset, 0, width - 1)] = True
            cuts.append(removed)
            origin = {"axis": "horizontal" if transpose else "vertical",
                      "path": path.tolist(), "cost": float(current[np.arange(height), path].mean())}
            regions = components(foreground & ~removed)
            if len(regions) > 1:
                output.extend((m.T if transpose else m, origin) for m in regions)
        if len(cuts) > 1:
            regions = components(foreground & ~np.logical_or.reduce(cuts))
            if len(regions) > 1:
                origin = {"axis": "horizontal" if transpose else "vertical", "all_seams": True}
                output.extend((m.T if transpose else m, origin) for m in
                              adjacent_groups(regions, config["maximum_adjacent_union"]))
    return output


def _quad(mask: np.ndarray, image_wh: tuple[int, int], native_size: int,
          offset: tuple[int, int]) -> list[list[float]]:
    y, x = np.nonzero(mask)
    points = np.column_stack((x + offset[0] + .5, y + offset[1] + .5)).astype(np.float32)
    center = points.mean(0)
    if len(points) > 1:
        _, _, axes = np.linalg.svd(points - center, full_matrices=False)
        direction = axes[0]
    else:
        direction = np.array([1., 0.], np.float32)
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    transverse = np.array([-direction[1], direction[0]])
    u, v = (points - center) @ direction, (points - center) @ transverse
    half_pixel = .5 * (abs(float(direction[0])) + abs(float(direction[1])))
    u1, u2 = float(u.min()) - half_pixel, float(u.max()) + half_pixel
    v1, v2 = float(v.min()) - half_pixel, float(v.max()) + half_pixel
    corners = np.array([center + a * direction + b * transverse
                        for a, b in ((u1, v1), (u2, v1), (u2, v2), (u1, v2))])
    width, height = image_wh
    side = max(width, height)
    corners *= side / native_size
    corners -= [(side - width) // 2, (side - height) // 2]
    return corners.tolist()


def native_rgb(image: np.ndarray, native_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(width, height)
    canvas = np.zeros((side, side, 3), np.uint8)
    y, x = (side - height) // 2, (side - width) // 2
    canvas[y:y + height, x:x + width] = image
    return cv2.resize(canvas, (native_size, native_size), interpolation=cv2.INTER_AREA)


class CandidatePool:
    """Deduplicate identical crop recipes, preserving every method membership."""
    def __init__(self, image_wh: tuple[int, int]):
        self.image_wh = image_wh
        self.rows: dict[str, dict] = {}

    def add(self, bbox: list[int], method: str, origin: dict, *, quad=None,
            neutral_fraction: float = 0., parent_methods=None) -> str | None:
        width, height = self.image_wh
        bbox = [max(0, int(bbox[0])), max(0, int(bbox[1])),
                min(width, int(bbox[2])), min(height, int(bbox[3]))]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None  # Geometric window lies entirely outside the RGB canvas.
        recipe = {"bbox": bbox, "quad": quad, "neutral_fraction": neutral_fraction}
        key = sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        if key not in self.rows:
            self.rows[key] = {"candidate_id": key, **recipe, "methods": [], "origins": [],
                              "parent_methods": []}
        row = self.rows[key]
        if method not in row["methods"]:
            row["methods"].append(method)
        record = {"method": method, **origin}
        if record not in row["origins"]:
            row["origins"].append(record)
        row["parent_methods"] = sorted(set(row["parent_methods"]) | set(parent_methods or ()))
        return key


def adjacent_groups_with_members(regions: list[np.ndarray], maximum: int):
    """The original adjacency order, retaining the seed IDs used by each crop."""
    if not regions:
        return []
    if len(regions) == 1:
        return [(regions[0], (1,))]
    centers = np.array([[np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()] for m in regions])
    _, _, axes = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.argsort(centers @ axis, kind="stable")
    output = [(regions[i], (int(i) + 1,)) for i in order]
    for size in range(2, min(maximum, len(regions)) + 1):
        for start in range(len(regions) - size + 1):
            indices = order[start:start + size]
            output.append((np.logical_or.reduce([regions[i] for i in indices]),
                           tuple(int(i) + 1 for i in indices)))
    return output


def mask_candidates(probability: np.ndarray, image: np.ndarray, config: dict, *,
                    include_auxiliary: bool = True, trace=None) -> CandidatePool:
    if (probability.ndim != 2 or probability.shape[0] != probability.shape[1]
        or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1):
        raise ValueError("expected a finite square probability map in [0,1]")
    size = probability.shape[0]
    image_wh = (image.shape[1], image.shape[0])
    pool = CandidatePool(image_wh)
    native_image = native_rgb(image, size)

    def add_region(region, offset, method, origin, lineage=None):
        box = tight_box(region)
        box = [box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1]]
        mapped = remap_letterbox_native_box(box, image_wh, native_size=size)
        cid = pool.add(mapped, method, {**origin, "native_bbox": box})
        if trace is not None and cid is not None:
            trace(cid, region, offset, (0, 0, *image_wh),
                  {"method": method, "native_size": size, **(lineage or {})})
        if config["variants"]["oriented"]:
            quad = _quad(region, image_wh, size, offset)
            q = np.array(quad)
            footprint = [int(np.floor(q[:, 0].min())), int(np.floor(q[:, 1].min())),
                         int(np.ceil(q[:, 0].max())), int(np.ceil(q[:, 1].max()))]
            pool.add(footprint, "crop_variants", {"oriented_from": method, **origin},
                     quad=quad, parent_methods=[method])

    # Baseline uses the existing RETR_EXTERNAL semantics, including contour order.
    contours, _ = cv2.findContours((probability >= config["baseline_threshold"]).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for index, contour in enumerate(contours):
        x, y, width, height = cv2.boundingRect(contour)
        box = remap_letterbox_native_box([x, y, x + width, y + height], image_wh, native_size=size)
        cid = pool.add(box, "baseline", {"contour_index": index})
        if trace is not None and cid is not None:
            contour_mask = np.zeros(probability.shape, np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=-1)
            trace(cid, contour_mask.astype(bool), (0, 0), (0, 0, *image_wh),
                  {"method": "baseline", "native_size": size, "kind": "component",
                   "threshold": config["baseline_threshold"]})

    for threshold in config["threshold_tree"]["thresholds"]:
        for index, region in enumerate(components(probability >= threshold), 1):
            add_region(region, (0, 0), "threshold_tree", {"threshold": threshold},
                       {"kind": "component", "threshold": threshold, "component_id": index})
    # Reconstruct full support from high-threshold seeds, instead of cropping only cores.
    for low, high in config["threshold_tree"]["seed_pairs"]:
        for support_index, region in enumerate(components(probability >= low), 1):
            x1, y1, x2, y2 = tight_box(region)
            support = region[y1:y2, x1:x2]
            p = probability[y1:y2, x1:x2]
            count, seeds = cv2.connectedComponents(((p >= high) & support).astype(np.uint8), connectivity=8)
            if count <= 2:
                continue
            partition = geodesic_partition(p, support, seeds)
            pieces = [partition == label for label in range(1, count)]
            for piece, seed_ids in adjacent_groups_with_members(
                    pieces, config["threshold_tree"]["maximum_adjacent_union"]):
                add_region(piece, (x1, y1), "threshold_tree", {"seed_low": low, "seed_high": high},
                           {"kind": "seed_piece" if len(seed_ids) == 1 else "seed_union",
                            "seed_low": low, "seed_high": high, "support_index": support_index,
                            "seed_ids": seed_ids})

    if not include_auxiliary:
        return pool

    for region in components(probability >= config["baseline_threshold"]):
        x1, y1, x2, y2 = tight_box(region)
        support = region[y1:y2, x1:x2]
        p = probability[y1:y2, x1:x2]
        distance = cv2.distanceTransform(np.pad(support.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        thickness = 2 * float(np.quantile(distance[support], .75))
        radii = sorted({max(1, round(thickness * fraction))
                        for fraction in config["erosion_reconstruction"]["radius_thickness_fractions"]})
        for radius in radii:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            eroded = cv2.erode(support.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            count, seeds = cv2.connectedComponents(eroded, connectivity=8)
            if count <= 2:
                continue
            partition = geodesic_partition(p, support, seeds)
            for piece in adjacent_groups([partition == label for label in range(1, count)],
                                          config["erosion_reconstruction"]["maximum_adjacent_union"]):
                add_region(piece, (x1, y1), "erosion_reconstruction", {"radius_native": radius})
        if min(support.shape) >= config["image_seams"]["minimum_native_extent"]:
            for piece, origin in seam_partitions(p, support, native_image[y1:y2, x1:x2], config["image_seams"]):
                add_region(piece, (x1, y1), "image_seams", {"roi_native": [x1, y1, x2, y2], **origin})
    return pool


def tile_windows(image_wh: tuple[int, int], grid: int, overlap: float) -> list[list[int]]:
    if grid < 2 or not 0 <= overlap < 1 or min(image_wh) < grid:
        raise ValueError("invalid tiling geometry")
    width, height = image_wh
    patch_w = min(width, int(np.ceil(width / (grid - (grid - 1) * overlap))))
    patch_h = min(height, int(np.ceil(height / (grid - (grid - 1) * overlap))))
    xs = np.rint(np.linspace(0, width - patch_w, grid)).astype(int)
    ys = np.rint(np.linspace(0, height - patch_h, grid)).astype(int)
    return [[int(x), int(y), int(x + patch_w), int(y + patch_h)] for y in ys for x in xs]


def add_tile_candidates(pool: CandidatePool, probability: np.ndarray, window: list[int],
                        thresholds: list[float], tile_id: str, trace=None) -> None:
    x1, y1, x2, y2 = window
    for threshold in thresholds:
        for index, region in enumerate(components(probability >= threshold), 1):
            local = remap_letterbox_native_box(tight_box(region), (x2 - x1, y2 - y1),
                                               native_size=probability.shape[0])
            box = [local[0] + x1, local[1] + y1, local[2] + x1, local[3] + y1]
            cid = pool.add(box, "tiled_segmentation", {"tile_id": tile_id, "window": window, "threshold": threshold})
            if trace is not None and cid is not None:
                trace(cid, region, (0, 0), tuple(window),
                      {"method": "tiled_segmentation", "kind": "component", "view": tile_id,
                       "threshold": threshold, "component_id": index,
                       "native_size": probability.shape[0]})


def add_border_variants(pool: CandidatePool, config: dict) -> None:
    for row in list(pool.rows.values()):
        if row["quad"] is not None or not set(row["methods"]) & set(config["source_methods"]):
            continue
        box = row["bbox"]
        scale = min(box[2] - box[0], box[3] - box[1])
        for fraction in config["border_height_fractions"]:
            delta = int(round(scale * fraction))
            if not delta:
                continue
            for edge in range(4):
                adjusted = box.copy()
                adjusted[edge] += -delta if edge < 2 else delta
                pool.add(adjusted, "crop_variants", {"parent": row["candidate_id"], "edge": edge,
                                                     "height_fraction": fraction}, parent_methods=row["methods"])
            adjusted = [box[0] - delta, box[1] - delta, box[2] + delta, box[3] + delta]
            pool.add(adjusted, "crop_variants", {"parent": row["candidate_id"], "symmetric": fraction},
                     parent_methods=row["methods"])
        for fraction in config["neutral_height_fractions"]:
            pool.add(box, "crop_variants", {"parent": row["candidate_id"], "neutral": fraction},
                     neutral_fraction=fraction, parent_methods=row["methods"])


def render_crop(image: np.ndarray, row: dict) -> np.ndarray:
    x1, y1, x2, y2 = row["bbox"]
    if row["quad"] is None:
        crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    else:
        quad = np.array(row["quad"], np.float32)
        width = max(1, int(np.ceil(np.linalg.norm(quad[1] - quad[0]))))
        height = max(1, int(np.ceil(np.linalg.norm(quad[3] - quad[0]))))
        target = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        transform = cv2.getPerspectiveTransform(quad, target)
        fill = tuple(float(v) for v in np.median(image[y1:y2, x1:x2].reshape(-1, 3), axis=0))
        crop = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    if not crop.size:
        raise ValueError("empty rendered crop")
    if row["neutral_fraction"]:
        padding = max(1, int(round(min(crop.shape[:2]) * row["neutral_fraction"])))
        fill = tuple(float(v) for v in np.median(crop.reshape(-1, 3), axis=0))
        crop = cv2.copyMakeBorder(crop, padding, padding, padding, padding,
                                  cv2.BORDER_CONSTANT, value=fill)
    return np.ascontiguousarray(crop)
