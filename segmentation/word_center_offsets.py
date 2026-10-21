"""Word-balanced dense attraction targets and training for the auxiliary U-Net."""

from hashlib import sha256
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from segmentation.totaltext_word_kernel import _integer_map, native_owner_map
from segmentation.training import SegmentationTrainingConfig


@dataclass(frozen=True)
class WordCenterOffsetTrainingConfig(SegmentationTrainingConfig):
    loss: str = "word_balanced_center_offset_SmoothL1"
    offset_beta: float = field(kw_only=True)

    def __post_init__(self):
        if self.loss != "word_balanced_center_offset_SmoothL1" or not np.isfinite(self.offset_beta) or self.offset_beta <= 0:
            raise ValueError("Invalid center-offset training loss contract")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Offset training requires a positive integer batch size")
        if self.checkpoint_selection not in ("last", "validation_offset_loss"):
            raise ValueError("Unsupported offset checkpoint selection")
        values = asdict(self)
        values.pop("offset_beta")
        values["loss"] = "masked_BCEWithLogitsLoss"
        # Explicit batch deviations and offset-specific selection are recorded
        # by the new launcher; retain the parent checks for the other settings.
        values["batch_size"] = 8
        values["checkpoint_selection"] = "last"
        SegmentationTrainingConfig(**values)


def load_center_offset_config(path, *, mode):
    path = Path(path).resolve()
    namespace = Path(__file__).resolve().parents[1] / "configs/approximate"
    if mode != "approximate" or namespace not in path.parents:
        raise ValueError("Center-offset extension requires explicit approximate mode and configs/approximate/")
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != mode
        or config.get("claim_label") != "approximate reproduction"
        or config.get("architecture_identity") != "experimental-word-center-offset-agunet-v1"
        or config.get("target") != "visible_word_centroid_offsets_xy_divided_by_native_wh"):
        raise ValueError("Unsupported center-offset configuration/target contract")
    if (not np.isfinite(config["skip_floor"]) or not 0 <= config["skip_floor"] < 1
        or not np.isfinite(config["loss"]["beta"]) or config["loss"]["beta"] <= 0
        or config["loss"].get("aggregation") != "mean_words_per_image_then_mean_supervised_images"):
        raise ValueError("Invalid center-offset model/loss settings")
    from localize.word_center_offsets import validate_vote_config
    validate_vote_config(config["voting"])
    return config


def make_word_center_offsets(owners, validity):
    """Vectors to the visible valid word's raster centroid, not a Gaussian.

    Build AFTER paired augmentation/letterboxing. Every surviving word pixel
    has a target; no kernel, transcription or test label is needed. A centroid
    may lie outside a concave word: it remains a location cue, not its contour.
    """
    owners = _integer_map(owners)
    validity = np.asarray(validity)
    if (not owners.size or validity.shape != owners.shape
        or np.any((validity != 0) & (validity != 1))):
        raise ValueError("Expected nonempty paired owner and binary validity maps")
    owners = np.where(validity, owners, 0)
    height, width = owners.shape
    offsets = np.zeros((2, height, width), np.float32)
    centers = []
    for owner in np.unique(owners[owners > 0]):
        yy, xx = np.where(owners == owner)
        cx, cy = float(xx.mean()), float(yy.mean())
        offsets[0, yy, xx] = (cx - xx) / width
        offsets[1, yy, xx] = (cy - yy) / height
        centers.append({"owner_id": int(owner), "center_xy_native": [cx, cy], "pixels": len(xx)})
    return offsets, centers


class WordCenterOffsetDataset(Dataset):
    """Add dense vectors to the existing checksum-qualified word-owner dataset.

    Reuse the existing exact flip/rotation/letterbox replay, without constructing
    kernels. Touching word nuclei therefore cannot block these targets.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = dict(self.base[index])
        sample = self.base.manifest.samples[index]
        if sample.word_owner_annotation_path is None:
            raise ValueError("Word-center supervision requires one qualified owner ID per WORD, not per line")
        with Image.open(self.base.data_root / sample.word_owner_annotation_path) as opened:
            source = opened.convert("I")
            expected = len(np.unique(np.asarray(source)[np.asarray(source) > 0]))
            owners = native_owner_map(source, self.base.transform, index,
                                      batch["mask"], batch["validity"])
        offsets, _ = make_word_center_offsets(owners, batch["validity"][0].numpy())
        batch["center_offsets"] = torch.from_numpy(offsets)
        batch["owners_native"] = torch.from_numpy(owners)
        batch["expected_word_count"] = expected
        return batch


def word_center_offset_loss(raw_offsets, targets, owners, validity, *, beta):
    """SmoothL1(tanh(raw), target), equal word means, then supervised image means.

    Background and ignored pixels have zero gradient. Empty/fully ignored
    images have no offset objective; an entirely empty batch returns a
    differentiable zero, with no invented center.
    """
    if (raw_offsets.ndim != 4 or raw_offsets.shape[1] != 2
        or targets.shape != raw_offsets.shape
        or owners.shape != (raw_offsets.shape[0], *raw_offsets.shape[-2:])
        or validity.shape != (raw_offsets.shape[0], 1, *raw_offsets.shape[-2:])):
        raise ValueError("Expected Bx2xHxW offsets, BxHxW owners and Bx1xHxW validity")
    if (not np.isfinite(beta) or beta <= 0 or not raw_offsets.is_floating_point()
        or targets.dtype != raw_offsets.dtype or targets.device != raw_offsets.device
        or owners.device != raw_offsets.device or validity.device != raw_offsets.device
        or owners.dtype not in (torch.int32, torch.int64)
        or torch.any(owners < 0) or torch.any((validity != 0) & (validity != 1))
        or not torch.isfinite(raw_offsets).all() or not torch.isfinite(targets).all()
        or torch.any(targets.abs() >= 1)):
        raise ValueError("Invalid normalized offsets, owners, validity or SmoothL1 beta")
    errors = F.smooth_l1_loss(raw_offsets.tanh(), targets, beta=beta, reduction="none").mean(1)
    images = []
    for index in range(len(raw_offsets)):
        valid = validity[index, 0].bool()
        words = [errors[index][(owners[index] == owner) & valid].mean()
                 for owner in torch.unique(owners[index][valid]) if owner > 0]
        if words:
            images.append(torch.stack(words).mean())
    return torch.stack(images).mean() if images else raw_offsets.sum() * 0


def train_word_center_offset_epoch(model, batches, optimizer, device, *, beta):
    """Train only the auxiliary network; keep the classic shape U-Net separate.

    Include every sample in counts. Batches with no valid word supervision
    explicitly skip optimizer updates (including weight decay).
    """
    model.train()
    total, supervised_images, samples, steps, empty_batches, ids = 0., 0, 0, 0, 0, []
    for batch in batches:
        local = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(local["image"])
        loss = word_center_offset_loss(output, local["center_offsets"], local["owners_native"],
                                       local["validity"], beta=beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite word-center offset loss")
        active = (local["owners_native"] > 0) & local["validity"][:, 0].bool()
        count = int(active.flatten(1).any(1).sum())
        if count:
            loss.backward()
            optimizer.step()
            steps += 1
        else:
            empty_batches += 1
        total += float(loss.detach()) * count
        supervised_images += count
        samples += len(output)
        ids.extend(batch["sample_id"])
    if not samples:
        raise ValueError("Empty word-center training loader")
    return {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "loss": total / supervised_images if supervised_images else 0., "samples": samples,
            "supervised_images": supervised_images, "steps": steps, "empty_batches_without_update": empty_batches,
            "sample_order_sha256": sha256("".join(x + "\n" for x in ids).encode()).hexdigest()}

