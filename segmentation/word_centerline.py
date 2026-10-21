"""Native word skeleton targets and isolated APPROXIMATE radius supervision.

Targets are derived after paired spatial augmentation from the existing filled
word-owner map, never from disconnected character masks. A skeleton can branch
or contain several pieces; these properties are audited rather than concealed.
"""

from hashlib import sha256

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F

from segmentation.totaltext_word_kernel import WordKernelDataset, _integer_map
from segmentation.word_multitask import masked_bce_dice


def _components(mask):
    return cv2.connectedComponentsWithAlgorithm(mask.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)


def _neighbors(padded):
    return (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
            padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2])


def skeletonize_mask(mask):
    """Deterministic two-subiteration Zhang--Suen thinning, zero-padded.

    Original connected components erased by simultaneous thinning (e.g. small
    compact rasters) keep their row-major farthest-inside pixel. This explicit
    discrete safeguard is counted by centerline_targets; no absent word appears.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Skeletonization requires a 2D boolean mask")
    padded = np.pad(mask, 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = _neighbors(padded)
            count = sum(p)
            transitions = sum(((p[index] == 0) & (p[(index + 1) % 8] != 0)).astype(np.uint8)
                              for index in range(8))
            candidates = (padded[1:-1, 1:-1] != 0) & (count >= 2) & (count <= 6) & (transitions == 1)
            if step == 0:
                candidates &= (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                candidates &= (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            if candidates.any():
                padded[1:-1, 1:-1][candidates] = 0
                changed = True
        if not changed:
            break
    skeleton = padded[1:-1, 1:-1].astype(bool)
    count, labels = _components(mask)
    for component in range(1, count):
        support = labels == component
        if not np.any(skeleton & support):
            inside = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
            skeleton.flat[int(np.argmax(inside))] = True
    return skeleton


def centerline_targets(owners, validity, target_config):
    """Return HxW maps and per-owner audit; never mutate original supervision.

    Skeleton radius is the zero-padded Euclidean distance to background in
    native pixels. A ribbon extends fraction*radius around the skeleton, clipped
    to the same valid owner. The radius at ribbon pixels comes from the nearest
    skeleton pixel. Radius weights sum to one per surviving owner, including
    all of its fragments. Original missing words remain absent for evaluation.
    """
    owners = _integer_map(owners)
    valid = np.asarray(validity)
    fraction, minimum = target_config["ribbon_fraction"], target_config["min_half_width"]
    if (valid.shape != owners.shape or np.any((valid != 0) & (valid != 1))
            or not np.isfinite([fraction, minimum]).all() or not 0 <= fraction < 1 or minimum < 0):
        raise ValueError("Invalid centerline target geometry/configuration")
    if np.any((owners > 0) & (valid == 0)):
        raise ValueError("Word owners overlap invalid pixels")
    line = np.zeros(owners.shape, np.float32)
    radius = np.zeros_like(line)
    weight = np.zeros_like(line)
    full_skeleton = np.zeros_like(line)
    rows = []
    for owner in np.unique(owners[owners > 0]):
        yy, xx = np.where(owners == owner)
        y0, y1, x0, x1 = int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1
        support = owners[y0:y1, x0:x1] == owner
        inside = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
        skeleton = skeletonize_mask(support)
        distance, nearest = distance_transform_edt(~skeleton, return_indices=True)
        nearest_radius = inside[tuple(nearest)]
        ribbon = support & ((distance <= np.maximum(minimum, fraction * nearest_radius)) | skeleton)
        line[y0:y1, x0:x1][ribbon] = 1
        radius[y0:y1, x0:x1][ribbon] = nearest_radius[ribbon]
        weight[y0:y1, x0:x1][ribbon] = 1.0 / int(ribbon.sum())
        full_skeleton[y0:y1, x0:x1][skeleton] = 1
        components, component_labels = _components(support)
        skeleton_count, _ = _components(skeleton)
        # Count compact components represented by one pixel; the target audit
        # names this fact without claiming every singleton is a thinning loss.
        singletons = sum(np.count_nonzero(skeleton & (component_labels == item)) == 1
                         for item in range(1, components))
        degree = sum(_neighbors(np.pad(skeleton, 1).astype(np.uint8)))
        rows.append({"owner_id": int(owner), "word_pixels": int(support.sum()),
                     "word_fragments": components - 1, "skeleton_fragments": skeleton_count - 1,
                     "skeleton_pixels": int(skeleton.sum()), "line_pixels": int(ribbon.sum()),
                     "skeleton_branch_pixels": int((skeleton & (degree > 2)).sum()),
                     "singleton_skeleton_components": int(singletons),
                     "radius_min": float(inside[skeleton].min()), "radius_max": float(inside[skeleton].max())})
    union_components, _ = _components(line > 0)
    audit = {"words": len(rows), "word_fragments": sum(row["word_fragments"] for row in rows),
             "skeleton_fragments": sum(row["skeleton_fragments"] for row in rows),
             "line_union_components": union_components - 1,
             "skeleton_branch_pixels": sum(row["skeleton_branch_pixels"] for row in rows),
             "per_owner": rows}
    return {"line": line, "radius": radius, "radius_weight": weight,
            "line_skeleton": full_skeleton, "audit": audit}


class WordCenterlineDataset(WordKernelDataset):
    """Reuse paired owner augmentation and add native line/radius maps only."""

    def __init__(self, base, *, distance_fraction, target_config):
        super().__init__(base, distance_fraction=distance_fraction)
        self.target_config = dict(target_config)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        targets = centerline_targets(batch["owners_native"].numpy(), batch["validity"][0].numpy(), self.target_config)
        for key in ("line", "radius", "radius_weight", "line_skeleton"):
            batch[key] = torch.from_numpy(targets[key])[None]
        return batch


def decoded_radius(raw, radius_floor):
    if not np.isfinite(radius_floor) or radius_floor <= 0:
        raise ValueError("Finite positive radius floor required")
    return F.softplus(raw) + radius_floor


def centerline_loss(evidence, batch, loss_config):
    valid, line = batch["validity"], batch["line"]
    options = {key: loss_config[key] for key in ("bce_weight", "dice_weight", "smooth")}
    coefficient, beta = loss_config["lambda_radius"], loss_config["radius_smooth_l1_beta"]
    if not np.isfinite([coefficient, beta]).all() or coefficient < 0 or beta <= 0:
        raise ValueError("Invalid radius loss coefficient/beta")
    radius, weight, raw = batch["radius"], batch["radius_weight"], evidence["radius_raw"]
    if (raw.shape != line.shape or radius.shape != line.shape or weight.shape != line.shape
            or not all(torch.isfinite(value).all() for value in (raw, radius, weight))
            or torch.any(weight < 0) or torch.any(radius < 0)
            or torch.any((weight > 0) & ((radius <= 0) | (line == 0) | (valid == 0)))
            or torch.any((line > 0) & (valid == 0))):
        raise ValueError("Invalid radius supervision/shape/validity")
    line_loss = masked_bce_dice(evidence["line"], line, valid, **options)
    prediction = decoded_radius(raw, loss_config["radius_floor"])
    errors = F.smooth_l1_loss(prediction.log(), radius.clamp_min(loss_config["radius_floor"]).log(),
                            reduction="none", beta=beta)
    totals = weight.flatten(1).sum(1)
    selected = totals > 0
    per_image = (errors * weight).flatten(1).sum(1) / totals.clamp_min(1)
    radius_loss = per_image[selected].mean() if selected.any() else raw.sum() * 0
    return line_loss + coefficient * radius_loss, {"line": line_loss, "radius": radius_loss}, bool(valid.sum() > 0)


def train_centerline_epoch(model, batches, optimizer, device, config):
    model.train()
    totals, ids, steps, skipped = {}, [], 0, 0
    for batch in batches:
        local = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        evidence = model.forward_evidence(local["image"])
        loss, terms, supervised = centerline_loss(evidence, local, config["loss"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite centerline objective")
        optimizer.zero_grad(set_to_none=True)
        if supervised:
            loss.backward()
            optimizer.step()
        else:
            skipped += 1
        steps += 1
        ids.extend(batch["sample_id"])
        for name, value in {"loss": loss, **terms}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(batch["image"])
    if not ids:
        raise ValueError("Empty centerline training loader")
    return {**{key: value / len(ids) for key, value in totals.items()}, "samples": len(ids),
            "steps": steps, "skipped_unsupervised_batches": skipped,
            "sample_order_sha256": sha256("".join(sample + "\n" for sample in ids).encode()).hexdigest()}
