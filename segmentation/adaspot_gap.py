"""Orientation-independent inter-word corridors, paired targets and masked loss."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import math

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch

from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import WordKernelDataset, _integer_map
from segmentation.training import SegmentationTrainingConfig
from segmentation.word_multitask import masked_bce_dice


def make_interword_gaps(owners, validity, *, radius):
    """Valid background within radius of at least TWO distinct word owners.

    A circular native-pixel neighborhood handles horizontal, rotated and curved
    words without line labels. A fragmented single owner is never its own neighbor.
    Owners have already undergone the exact paired augmentation and native resize.
    No text/kernel pixel is removed or added by target construction.
    """
    owners = _integer_map(owners)
    valid = np.asarray(validity)
    if (valid.shape != owners.shape or np.any((valid != 0) & (valid != 1))
            or type(radius) is not int or radius < 1):
        raise ValueError("Gap targets require paired binary validity and positive integer radius")
    if np.any((owners > 0) & ~valid.astype(bool)):
        raise ValueError("Word owner lies outside valid supervision")
    offsets = np.arange(-radius, radius + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius ** 2
    counts = np.zeros(owners.shape, np.uint8)
    for owner in np.unique(owners[owners > 0]):
        yy, xx = np.where(owners == owner)
        y1, y2 = max(0, int(yy.min()) - radius), min(len(owners), int(yy.max()) + radius + 1)
        x1, x2 = max(0, int(xx.min()) - radius), min(owners.shape[1], int(xx.max()) + radius + 1)
        near = binary_dilation(owners[y1:y2, x1:x2] == owner, structure=disk)
        region = counts[y1:y2, x1:x2]
        # Saturation avoids overflow for dense scenes; only the >=2 predicate matters.
        region[:] = np.minimum(region + near.astype(np.uint8), 2)
    gap = (counts >= 2) & (owners == 0) & valid.astype(bool)
    return gap, {"owners": int(len(np.unique(owners[owners > 0]))),
                 "gap_pixels": int(gap.sum()), "radius_native_pixels": radius}


class AdaSpotGapDataset(WordKernelDataset):
    def __init__(self, base, *, distance_fraction, gap_radius):
        super().__init__(base, distance_fraction=distance_fraction)
        self.gap_radius = gap_radius

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        gap, _ = make_interword_gaps(batch["owners_native"].numpy(),
                                    batch["validity"][0].numpy(), radius=self.gap_radius)
        # Binary resize can retain text with no nearest owner. Such pixels must
        # remain text; they are explicitly excluded from the gap target.
        gap &= ~batch["mask"][0].numpy().astype(bool)
        batch["gap_target"] = torch.from_numpy(gap.astype(np.float32))[None]
        with Image.open(batch["image_path"]) as image:
            batch["canvas"] = letterbox_canvas_mask(image.size, native_size=batch["mask"].shape[-1])
        sample = self.base.manifest.samples[index]
        with Image.open(self.base.data_root / sample.word_owner_annotation_path) as owner:
            values = np.asarray(owner)
            batch["expected_word_count"] = int(len(np.unique(values[values > 0])))
        return batch


@dataclass(frozen=True)
class AdaSpotGapTrainingConfig(SegmentationTrainingConfig):
    loss: str = "masked_BCE_Dice_text_kernel_gap_v1"

    def __post_init__(self):
        if (self.loss != "masked_BCE_Dice_text_kernel_gap_v1"
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError("Invalid AdaSpot loss, learning rate or batch size")
        values = asdict(self)
        values.update(loss="masked_BCEWithLogitsLoss", learning_rate=1e-4, batch_size=8)
        SegmentationTrainingConfig(**values)


def gap_loss(evidence, batch, config):
    words, valid, target = evidence["words"], batch["validity"], batch["mask"]
    kernel, separator, gap = batch["kernel"], batch["separation"], batch["gap_target"]
    if (words.shape != (len(target), 2, *target.shape[-2:]) or kernel.shape != target.shape
            or gap.shape != target.shape or separator.shape != target.shape):
        raise ValueError("Text/kernel/gap supervision shape mismatch")
    if (torch.any(kernel > target) or torch.any(kernel > valid)
            or torch.any((separator != 0) & (separator != 1))
            or torch.any((separator > 0) & ((target > 0) | (valid == 0)))
            or torch.any((gap > 0) & ((target > 0) | (valid == 0)))):
        raise ValueError("Invalid kernel, separator or gap containment")
    if (any(not math.isfinite(config[k]) or config[k] < 0
            for k in ("lambda_text", "lambda_kernel", "lambda_gap"))
            or min(config["lambda_text"], config["lambda_kernel"]) <= 0
            or not 1 <= config["separation_weight"] <= 20):
        raise ValueError("Invalid task loss weights")
    options = {k: config[k] for k in ("bce_weight", "dice_weight", "smooth")}
    terms = {"text": masked_bce_dice(words[:, :1], target, valid, **options,
             pixel_weight=1 + (config["separation_weight"] - 1) * separator),
             "kernel": masked_bce_dice(words[:, 1:2], kernel, valid, **options)}
    if evidence["gap"] is not None:
        terms["gap"] = masked_bce_dice(evidence["gap"], gap, valid, **options)
    return sum(config["lambda_" + name] * value for name, value in terms.items()), terms


def train_gap_epoch(model, batches, optimizer, device, config, *, progress_every=25):
    model.train()
    totals, count, steps, skipped, ids = {}, 0, 0, 0, []
    for batch_index, batch in enumerate(batches, 1):
        local = {key: value.to(device, non_blocking=device.type == "cuda")
                 if torch.is_tensor(value) else value for key, value in batch.items()}
        if local["image"].device != device:
            raise RuntimeError("Training image did not reach the requested compute device")
        optimizer.zero_grad(set_to_none=True)
        evidence = model.forward_evidence(local["image"])
        if evidence["words"].device != local["image"].device:
            raise RuntimeError("Model output left the requested compute device")
        loss, terms = gap_loss(evidence, local, config["loss"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite AdaSpot loss")
        if local["validity"].any():
            loss.backward()
            optimizer.step()
            steps += 1
        else:
            skipped += 1  # Do not apply Adam weight decay to an entirely ignored batch.
        for key, value in {"loss": loss, **terms}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * len(local["image"])
        count += len(local["image"])
        ids.extend(batch["sample_id"])
        if progress_every and (batch_index == 1 or batch_index % progress_every == 0):
            print("[{}] batch={} steps={} device={} loss={:.5f}".format(
                model.branch, batch_index, steps, evidence["words"].device, float(loss.detach())), flush=True)
    if not count:
        raise ValueError("Empty AdaSpot TRAIN loader")
    return {**{key: value / count for key, value in totals.items()}, "samples": count,
            "steps": steps, "skipped_all_invalid_batches": skipped, "compute_device": str(device),
            "sample_order_sha256": sha256("".join(x + "\n" for x in ids).encode()).hexdigest()}
