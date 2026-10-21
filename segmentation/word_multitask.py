"""Approximate word-instance targets and configurable masked multi-task loss.

Reuses qualified owner-map geometry and distance kernels. Scientific choices
are specified in configs/approximate/word_multitask_v1.json and documented in
docs/WORD_MULTITASK.md; none enters the strict reproduction path.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.word_multitask import VARIANTS
from segmentation.totaltext_word_kernel import (
    WordKernelDataset, _integer_map, make_word_kernels,
)
from segmentation.training import SegmentationTrainingConfig


def gaussian_word_centers(owners, kernels, *, sigma_scale, min_sigma):
    """One orientation-aware Gaussian per surviving owner, clipped to its word.

    Project the raster centroid onto the connected kernel (row-major distance
    tie). Covariance of the full word sets orientation and scale; eigenvalues
    have a min_sigma^2 floor. A curved word's centroid may lie in background,
    hence projection is necessary. Native one-pixel words retain a peak of 1.
    """
    owners, kernels = _integer_map(owners), _integer_map(kernels)
    if owners.shape != kernels.shape or np.any((kernels > 0) & (kernels != owners)):
        raise ValueError("Kernel/owner geometry mismatch")
    if not np.isfinite([sigma_scale, min_sigma]).all() or sigma_scale <= 0 or min_sigma <= 0:
        raise ValueError("Gaussian scale and minimum sigma must be finite and positive")
    heatmap = np.zeros(owners.shape, np.float32)
    rows = []
    for owner in np.unique(owners):
        if owner == 0:
            continue
        yy, xx = np.where(owners == owner)
        ky, kx = np.where(kernels == owner)
        if not len(kx):
            raise ValueError("Word {} has no kernel for its center".format(owner))
        points = np.column_stack((xx, yy)).astype(np.float64)
        centroid = points.mean(axis=0)
        nearest = int(np.argmin((kx - centroid[0]) ** 2 + (ky - centroid[1]) ** 2))
        center = np.array([kx[nearest], ky[nearest]], np.float64)
        deviations = points - centroid
        covariance = deviations.T @ deviations / len(points)
        eigenvalues, axes = np.linalg.eigh(covariance)
        covariance = (axes * np.maximum(eigenvalues * sigma_scale ** 2, min_sigma ** 2)) @ axes.T
        delta = points - center
        squared = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(covariance), delta)
        heatmap[yy, xx] = np.exp(-0.5 * squared).astype(np.float32)
        rows.append({"owner_id": int(owner), "center_xy_native": center.astype(int).tolist(),
                     "covariance_xy_native": covariance.tolist(), "pixels": len(points)})
    return heatmap, rows


def targets_from_polygons(size_wh, instances, *, distance_fraction, sigma_scale, min_sigma):
    """Reuse the repository polygon rasterizer, including overlap/ignore policy.

    This helper is for word-granularity annotations. The matched Total-Text
    runner uses its already-qualified v4 polygon-derived owner artifacts.
    """
    from segmentation.totaltext_word_regions import make_word_regions
    from segmentation.scene_text_preparation import rasterize_instances
    instances = tuple(instances)
    if any(item.granularity != "word" for item in instances if not item.ignore):
        raise ValueError("Multi-task word targets require one polygon per word")
    if any(not item.ignore for item in instances):
        text, validity, _, owner, audit = make_word_regions(size_wh, instances)
        if audit.empty_word_count:
            raise ValueError("Polygon rasterization lost care words; inspect target resolution/overlaps")
        owners = np.asarray(owner, dtype=np.int32)
    else:
        if instances:
            text, validity = rasterize_instances(size_wh, instances)
        else:
            text, validity = Image.new("L", size_wh, 0), Image.new("L", size_wh, 255)
        owners = np.zeros((size_wh[1], size_wh[0]), np.int32)
    kernels, _ = make_word_kernels(owners, distance_fraction=distance_fraction)
    center, rows = gaussian_word_centers(owners, kernels, sigma_scale=sigma_scale, min_sigma=min_sigma)
    return {"mask": (np.asarray(text) > 0).astype(np.float32),
            "validity": (np.asarray(validity) > 0).astype(np.float32),
            "owners_native": owners, "kernel": (kernels > 0).astype(np.float32),
            "center": center, "centers": rows}


class WordMultitaskDataset(WordKernelDataset):
    def __init__(self, base, *, distance_fraction, sigma_scale, min_sigma):
        super().__init__(base, distance_fraction=distance_fraction)
        self.sigma_scale, self.min_sigma = sigma_scale, min_sigma

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        owners = batch["owners_native"].numpy()
        kernels = owners * batch["kernel"][0].numpy().astype(np.int32)
        centers, _ = gaussian_word_centers(owners, kernels, sigma_scale=self.sigma_scale,
                                         min_sigma=self.min_sigma)
        batch["center"] = torch.from_numpy(centers)[None]
        return batch


@dataclass(frozen=True)
class WordMultitaskTrainingConfig(SegmentationTrainingConfig):
    loss: str = "masked_BCE_Dice_and_gaussian_MSE"

    def __post_init__(self):
        if self.loss != "masked_BCE_Dice_and_gaussian_MSE":
            raise ValueError("Invalid multi-task loss contract")
        # Preserve all existing optimizer/mode checks without mislabeling the
        # actual loss in serialized checkpoint metadata.
        values = asdict(self)
        values["loss"] = "masked_BCEWithLogitsLoss"
        SegmentationTrainingConfig(**values)


def _maps(logits, target, valid):
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Loss expects paired Bx1xHxW maps")
    if not all(torch.isfinite(x).all() for x in (logits, target, valid)):
        raise ValueError("Nonfinite loss input")
    if torch.any((valid != 0) & (valid != 1)) or torch.any((target < 0) | (target > 1)):
        raise ValueError("Targets must be in [0,1] and validity binary")


def masked_bce_dice(logits, target, valid, *, bce_weight, dice_weight, smooth, pixel_weight=None):
    """Per-image valid-pixel BCE + soft Dice, averaged over non-ignored images.

    All-invalid images contribute differentiable zero. Empty valid foreground
    uses the smooth Dice formula; no sample is removed from evaluation.
    """
    _maps(logits, target, valid)
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("Text/kernel Dice targets must be binary")
    if not np.isfinite([bce_weight, dice_weight, smooth]).all() or min(bce_weight, dice_weight) < 0 or smooth <= 0:
        raise ValueError("Invalid BCE/Dice weights or smoothing")
    if pixel_weight is None:
        pixel_weight = torch.ones_like(valid)
    if pixel_weight.shape != valid.shape or not torch.isfinite(pixel_weight).all() or torch.any(pixel_weight <= 0):
        raise ValueError("Invalid BCE pixel weights")
    selected = valid.flatten(1).sum(1) > 0
    if not selected.any():
        return logits.sum() * 0
    weight = valid * pixel_weight
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).flatten(1).sum(1)
    bce = bce / weight.flatten(1).sum(1).clamp_min(1)
    probability = logits.sigmoid() * valid
    gt = target * valid
    intersection = (probability * gt).flatten(1).sum(1)
    dice = 1 - (2 * intersection + smooth) / (probability.flatten(1).sum(1) + gt.flatten(1).sum(1) + smooth)
    return (bce_weight * bce + dice_weight * dice)[selected].mean()


def masked_gaussian_mse(logits, target, valid, *, balanced):
    """Soft Gaussian regression with sigmoid; optional equal support/background weight."""
    _maps(logits, target, valid)
    error = (logits.sigmoid() - target).square()
    if not balanced:
        weights = [valid]
    else:
        weights = [valid * (target > 0), valid * (target == 0)]
    counts = torch.stack([w.flatten(1).sum(1) for w in weights])
    terms = torch.stack([(error * w).flatten(1).sum(1) for w in weights]) / counts.clamp_min(1)
    active = counts > 0
    per_image = terms.sum(0) / active.sum(0).clamp_min(1)
    selected = active.any(0)
    return per_image[selected].mean() if selected.any() else logits.sum() * 0


def multitask_loss(logits, batch, *, variant, config):
    if variant not in VARIANTS or logits.ndim != 4 or logits.shape[1] != VARIANTS[variant]:
        raise ValueError("Multi-task output/variant mismatch")
    if config["center_loss"] not in ("mse", "balanced_mse"):
        raise ValueError("Unknown Gaussian center loss")
    weights = [config[key] for key in ("lambda_text", "lambda_kernel", "lambda_center")]
    if not np.isfinite(weights).all() or min(weights) < 0 or weights[0] <= 0:
        raise ValueError("Invalid task weights")
    valid, target = batch["validity"], batch["mask"]
    separator = batch["separation"]
    if separator.shape != valid.shape or torch.any((separator != 0) & (separator != 1)):
        raise ValueError("Invalid separation map")
    if torch.any((separator > 0) & (target > 0) & (valid > 0)):
        raise ValueError("Separator overlaps valid text target")
    options = {k: config[k] for k in ("bce_weight", "dice_weight", "smooth")}
    terms = {"text": masked_bce_dice(logits[:, :1], target, valid, **options,
             pixel_weight=1 + (config["separation_weight"] - 1) * separator)}
    if VARIANTS[variant] >= 2:
        kernel = batch["kernel"]
        if torch.any(kernel > target) or torch.any(kernel > valid):
            raise ValueError("Kernel lies outside valid text")
        terms["kernel"] = masked_bce_dice(logits[:, 1:2], kernel, valid, **options)
    if VARIANTS[variant] == 3:
        if torch.any((batch["center"] > 0) & ((target == 0) | (valid == 0))):
            raise ValueError("Center target outside valid text")
        terms["center"] = masked_gaussian_mse(logits[:, 2:3], batch["center"], valid,
                                             balanced=config["center_loss"] == "balanced_mse")
    total = sum(config["lambda_" + name] * value for name, value in terms.items())
    return total, terms


def train_multitask_epoch(model, batches, optimizer, device, *, variant, loss_config):
    from hashlib import sha256
    model.train()
    totals, count, steps, ids = {}, 0, 0, []
    for batch in batches:
        local = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(local["image"])
        loss, terms = multitask_loss(logits, local, variant=variant, config=loss_config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite multi-task loss")
        loss.backward()
        optimizer.step()
        for key, value in {"loss": loss, **terms}.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(logits)
        count += len(logits)
        steps += 1
        ids.extend(batch["sample_id"])
    if not count:
        raise ValueError("Empty multi-task training loader")
    return {**{key: value / count for key, value in totals.items()}, "samples": count, "steps": steps,
            "sample_order_sha256": sha256("".join(x + "\n" for x in ids).encode()).hexdigest()}
