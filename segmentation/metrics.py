"""Explicit foreground IoU/F1 aggregation modes for segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


EmptyPolicy = Literal["perfect_if_both_empty", "zero_if_both_empty", "exclude_both_empty"]


@dataclass(frozen=True)
class ForegroundMetricResult:
    foreground_iou: float
    foreground_f1: float
    aggregation_mode: str
    empty_foreground_policy: EmptyPolicy
    sample_count: int
    evaluated_sample_count: int
    empty_sample_count: int
    intersection_pixels: int
    predicted_foreground_pixels: int
    target_foreground_pixels: int
    union_pixels: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def binarize_probabilities(probabilities: torch.Tensor, *, threshold: float) -> torch.Tensor:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability threshold must be in [0,1]")
    return probabilities >= threshold


def _as_boolean(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        return tensor
    unique = torch.unique(tensor.detach().cpu())
    if not set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{name} must be binary before metric aggregation")
    return tensor.bool()


def _counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = _as_boolean(prediction, "prediction")
    target = _as_boolean(target, "target")
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched shape")
    if valid_mask is not None:
        valid = _as_boolean(valid_mask, "valid_mask")
        if valid.shape != prediction.shape:
            raise ValueError("valid_mask must match prediction shape")
        prediction = prediction & valid
        target = target & valid
    reduce_dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).sum(dim=reduce_dims, dtype=torch.int64)
    predicted = prediction.sum(dim=reduce_dims, dtype=torch.int64)
    ground_truth = target.sum(dim=reduce_dims, dtype=torch.int64)
    union = predicted + ground_truth - intersection
    return intersection, predicted, ground_truth, union


def foreground_iou_f1_per_image_macro(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    empty_foreground_policy: EmptyPolicy,
    valid_mask: torch.Tensor | None = None,
) -> ForegroundMetricResult:
    """Average per-image ratios with a caller-selected both-empty policy."""

    intersection, predicted, ground_truth, union = _counts(prediction, target, valid_mask)
    empty = union == 0
    iou = intersection.double() / union.clamp_min(1).double()
    denominator = predicted + ground_truth
    f1 = 2.0 * intersection.double() / denominator.clamp_min(1).double()
    if empty_foreground_policy == "perfect_if_both_empty":
        iou[empty] = 1.0
        f1[empty] = 1.0
        include = torch.ones_like(empty)
    elif empty_foreground_policy == "zero_if_both_empty":
        iou[empty] = 0.0
        f1[empty] = 0.0
        include = torch.ones_like(empty)
    elif empty_foreground_policy == "exclude_both_empty":
        include = ~empty
        if not include.any():
            raise ValueError("No non-empty samples remain under exclude_both_empty")
    else:
        raise ValueError(f"Unknown empty foreground policy: {empty_foreground_policy}")
    return ForegroundMetricResult(
        foreground_iou=float(iou[include].mean()),
        foreground_f1=float(f1[include].mean()),
        aggregation_mode="per_image_macro",
        empty_foreground_policy=empty_foreground_policy,
        sample_count=int(intersection.numel()),
        evaluated_sample_count=int(include.sum()),
        empty_sample_count=int(empty.sum()),
        intersection_pixels=int(intersection.sum()),
        predicted_foreground_pixels=int(predicted.sum()),
        target_foreground_pixels=int(ground_truth.sum()),
        union_pixels=int(union.sum()),
    )


def foreground_iou_f1_global_micro(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    empty_foreground_policy: EmptyPolicy,
    valid_mask: torch.Tensor | None = None,
) -> ForegroundMetricResult:
    """Compute ratios once from dataset-global foreground pixel counts."""

    intersection, predicted, ground_truth, union = _counts(prediction, target, valid_mask)
    empty = union == 0
    if empty_foreground_policy == "exclude_both_empty":
        include = ~empty
        if not include.any():
            raise ValueError("No non-empty samples remain under exclude_both_empty")
    elif empty_foreground_policy in ("perfect_if_both_empty", "zero_if_both_empty"):
        include = torch.ones_like(empty)
    else:
        raise ValueError(f"Unknown empty foreground policy: {empty_foreground_policy}")
    total_intersection = int(intersection[include].sum())
    total_predicted = int(predicted[include].sum())
    total_target = int(ground_truth[include].sum())
    total_union = int(union[include].sum())
    if total_union == 0:
        value = 1.0 if empty_foreground_policy == "perfect_if_both_empty" else 0.0
        iou = f1 = value
    else:
        iou = total_intersection / total_union
        f1 = 2 * total_intersection / (total_predicted + total_target)
    return ForegroundMetricResult(
        foreground_iou=iou,
        foreground_f1=f1,
        aggregation_mode="global_micro",
        empty_foreground_policy=empty_foreground_policy,
        sample_count=int(intersection.numel()),
        evaluated_sample_count=int(include.sum()),
        empty_sample_count=int(empty.sum()),
        intersection_pixels=total_intersection,
        predicted_foreground_pixels=total_predicted,
        target_foreground_pixels=total_target,
        union_pixels=total_union,
    )


__all__ = [
    "ForegroundMetricResult",
    "binarize_probabilities",
    "foreground_iou_f1_global_micro",
    "foreground_iou_f1_per_image_macro",
]
