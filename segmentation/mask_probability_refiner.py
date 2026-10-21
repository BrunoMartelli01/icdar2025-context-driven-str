"""Paired full-image supervision and losses for residual probability refinement."""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def paired_transform(tensors, *, flip, quarter_turns):
    """Integer D4 transforms: no interpolation, label mixing or RGB/map drift."""
    return {k: torch.rot90(v.flip(-1) if flip else v, quarter_turns, (-2, -1)).contiguous()
            for k, v in tensors.items()}


def corrupt_probabilities(probabilities, rng):
    """GT-free simulated over/under-segmentation of already predicted maps."""
    radius = int(rng.choice([3, 5]))
    if bool(rng.integers(2)):
        return F.max_pool2d(probabilities[None], radius, stride=1, padding=radius//2)[0]
    return -F.max_pool2d(-probabilities[None], radius, stride=1, padding=radius//2)[0]


class RefinerDataset(Dataset):
    def __init__(self, records, ids, *, seed, epoch, corruption_probability=0.):
        self.records, self.ids = records, ids
        self.seed, self.epoch, self.corruption_probability = seed, epoch, corruption_probability

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        row = paired_transform(self.records[sid], flip=bool(rng.integers(2)), quarter_turns=int(rng.integers(4)))
        row["rgb"] = row["rgb"].float()/127.5-1
        row["target"] = row["target"].float()
        row["corrupted"] = bool(rng.random() < self.corruption_probability)
        if row["corrupted"]:
            row["probabilities"] = corrupt_probabilities(row["probabilities"], rng)
        row["sample_id"] = sid
        return row


def refiner_loss(outputs, base, target, validity, *, error_focus=False, dice_weight=.5,
                 preservation_weight=.05, error_weight=4.):
    if validity.shape != (len(target), 1, *target.shape[-2:]) or base.shape != target.shape:
        raise ValueError("Loss geometry differs")
    valid = validity.expand_as(target).float()
    if not bool(valid.any()):
        raise ValueError("No valid supervision; caller must explicitly count skipped batches")
    weights = torch.ones_like(target)
    if error_focus:
        # GT influences only training weights, never inference selection or inputs.
        boundary = (F.max_pool2d(target, 3, 1, 1)+F.max_pool2d(-target, 3, 1, 1)) > 0
        weights = weights + error_weight * (((base-target).abs() >= .5) | boundary).float()
    losses = []
    for output in outputs:
        bce = F.binary_cross_entropy_with_logits(output["logits"], target, reduction="none")
        class_losses = []
        for foreground in (target, 1-target):
            measure = valid * weights * foreground
            count = measure.sum((-2, -1))
            present = count > 0
            # Missing foreground/background classes do not dilute existing classes.
            class_losses.append(((bce*measure).sum((-2, -1))/count.clamp_min(1), present))
        numerator = sum(values*present for values, present in class_losses)
        denominator = sum(present.float() for _, present in class_losses)
        supervised = denominator > 0
        balanced = (numerator/denominator.clamp_min(1))[supervised].mean()
        pred = output["probabilities"]
        intersection = (pred*target*valid).sum((-2, -1))
        dice = 1-(2*intersection+1)/((pred*valid).sum((-2, -1))+(target*valid).sum((-2, -1))+1)
        preserve = ((pred-base).abs()*valid).sum()/valid.sum()
        losses.append(balanced + dice_weight*dice[supervised].mean() + preservation_weight*preserve)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite refiner objective")
    return loss


def pixel_counts(probabilities, target, validity, thresholds):
    pred = probabilities >= np.asarray(thresholds)[:, None, None]
    target, valid = target.astype(bool), validity.astype(bool)
    return [{"tp": int((p & t & valid).sum()), "fp": int((p & ~t & valid).sum()),
             "fn": int((~p & t & valid).sum())} for p, t in zip(pred, target)]
