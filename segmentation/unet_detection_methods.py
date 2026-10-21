"""Nested word nuclei and masked objectives, reusing existing word targets/loss."""

from dataclasses import asdict, dataclass
from hashlib import sha256

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segmentation.totaltext_word_kernel import WordKernelDataset, make_word_kernels
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.training import SegmentationTrainingConfig
from segmentation.word_multitask import masked_bce_dice


@dataclass(frozen=True)
class DetectionTrainingConfig(SegmentationTrainingConfig):
    loss: str = "experimental_detection_BCE_Dice_nested_kernels_v1"

    def __post_init__(self):
        if self.loss != "experimental_detection_BCE_Dice_nested_kernels_v1":
            raise ValueError("Unexpected detection loss identity")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.batch_size < 1:
            raise ValueError("Invalid detection learning rate/batch")
        values = asdict(self)
        values.update(loss="masked_BCEWithLogitsLoss", learning_rate=1e-4, batch_size=8)
        SegmentationTrainingConfig(**values)


def progressive_targets(owners, fractions):
    """Largest smallest core, then connected expansions containing that core.

    Fractions are distance/max-distance, NOT PSENet polygon shrink ratios.
    Every surviving native word keeps one separable smallest seed. Larger
    kernels are nested even when an irregular word has several distance peaks.
    """
    owners = np.asarray(owners)
    if (len(fractions) != 3 or not all(0 < x < 1 for x in fractions)
            or not all(a > b for a, b in zip(fractions, fractions[1:]))):
        raise ValueError("Three decreasing kernel distance fractions required")
    smallest, rows = make_word_kernels(owners, distance_fraction=fractions[0])
    maps = [smallest > 0]
    previous = smallest.copy()
    for fraction in fractions[1:]:
        enlarged = np.zeros_like(smallest)
        for row in rows:
            owner = row["owner_id"]
            yy, xx = np.where(owners == owner)
            y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
            support = owners[y0:y1, x0:x1] == owner
            distance = cv2.distanceTransform(np.pad(support.astype(np.uint8), 1),
                cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
            squared = np.rint(distance.astype(np.float64) ** 2).astype(np.int32)
            candidate = support & (squared >= fraction ** 2 * squared.max())
            seed = previous[y0:y1, x0:x1] == owner
            candidate |= seed
            _, components = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
            connected_ids = np.unique(components[seed])
            enlarged[y0:y1, x0:x1][np.isin(components, connected_ids[connected_ids > 0])] = owner
        if np.any((previous > 0) & (enlarged != previous)):
            raise RuntimeError("Progressive kernels are not nested")
        previous = enlarged
        maps.append(enlarged > 0)
    return np.stack(maps).astype(np.float32)


class DetectionWordDataset(WordKernelDataset):
    def __init__(self, base, fractions):
        super().__init__(base, distance_fraction=fractions[0])
        self.fractions = fractions

    def __getitem__(self, index):
        from PIL import Image
        batch = super().__getitem__(index)
        batch["kernels"] = torch.from_numpy(progressive_targets(batch["owners_native"].numpy(), self.fractions))
        with Image.open(batch["image_path"]) as image:
            batch["canvas"] = letterbox_canvas_mask(image.size, native_size=batch["mask"].shape[-1])
        return batch


def detection_loss(logits, batch, config):
    if logits.shape[1] not in (2, 4):
        raise ValueError("Expected text/smallest kernel, optionally two larger kernels")
    valid = batch["validity"]
    options = {k: config["loss"][k] for k in ("bce_weight", "dice_weight", "smooth")}
    terms = {"text": masked_bce_dice(logits[:, :1], batch["mask"], valid, **options,
        pixel_weight=1 + (config["loss"]["separation_weight"] - 1) * batch["separation"]),
        "kernel": masked_bce_dice(logits[:, 1:2], batch["kernel"], valid, **options)}
    if logits.shape[1] == 4:
        extra = [masked_bce_dice(logits[:, j + 1:j + 2], batch["kernels"][:, j:j + 1], valid,
                                **options) for j in (1, 2)]
        terms["larger_kernels"] = torch.stack(extra).mean()
        nested = logits[:, [1, 2, 3, 0]].sigmoid()
        violation = (nested[:, :-1] - nested[:, 1:]).relu()
        terms["nesting"] = (violation * valid).sum() / (3 * valid.sum()).clamp_min(1)
    total = sum(config["loss"]["lambda_" + name] * value for name, value in terms.items())
    return total, terms, bool(valid.sum() > 0)


def train_detection_epoch(model, batches, optimizer, device, config):
    model.train()
    totals, ids, steps, skipped = {}, [], 0, 0
    target_counts = {}
    for batch in batches:
        local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, terms, supervised = detection_loss(model(local["image"]), local, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite detection objective")
        if supervised:
            loss.backward(); optimizer.step(); steps += 1
        else:
            skipped += 1
        ids.extend(batch["sample_id"])
        for key in ("native_words_lost", "outside_image_words", "restored_contact_words"):
            if key in batch:
                target_counts[key] = target_counts.get(key, 0) + int(torch.as_tensor(batch[key]).sum())
        for key, value in {"loss": loss, **terms}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * len(batch["image"])
    if not ids:
        raise ValueError("Empty detection training loader")
    return {**{k: v / len(ids) for k, v in totals.items()}, **target_counts, "samples": len(ids), "steps": steps,
            "skipped_unsupervised_batches": skipped,
            "sample_order_sha256": sha256("".join(s + "\n" for s in ids).encode()).hexdigest()}
