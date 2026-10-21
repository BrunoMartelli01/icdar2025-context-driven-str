"""Explicit paired transforms for the APPROXIMATE segmentation branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class SegmentationTransformConfig:
    """Paper sizes plus explicit assumptions for every omitted transform detail."""

    horizontal_flip_probability: float
    rotation_probability: float
    rotation_degrees: tuple[float, float]
    image_interpolation: Literal["bilinear"]
    mask_interpolation: Literal["nearest", "bilinear"]
    image_fill: tuple[int, int, int]
    mask_fill: int
    normalization: Literal["imagenet", "none"]
    resize_geometry: Literal["stretch"]
    operation_order: Literal["flip_rotate_resize"]
    image_size: tuple[int, int] = (256, 256)
    mask_size: tuple[int, int] = (128, 128)
    target_binarization_threshold: float = 0.5
    validity_interpolation: Literal["nearest"] = "nearest"
    classification: str = "ASSUMPTION"

    def __post_init__(self) -> None:
        for name, probability in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("rotation_probability", self.rotation_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        low, high = self.rotation_degrees
        if low > high:
            raise ValueError("rotation_degrees must be ordered (minimum, maximum)")
        practical_highres = self.classification == "PRACTICAL_APPROXIMATION_HIGH_RESOLUTION"
        expected_sizes = ((512, 512), (256, 256)) if practical_highres else ((256, 256), (128, 128))
        if (self.image_size, self.mask_size) != expected_sizes:
            raise ValueError(
                "Transform sizes must match the explicit classification: "
                f"image={expected_sizes[0]}, mask={expected_sizes[1]}"
            )
        if self.image_interpolation != "bilinear" or self.mask_interpolation not in (
            "nearest", "bilinear"
        ):
            raise ValueError("Unsupported image or target resize interpolation")
        if self.validity_interpolation != "nearest":
            raise ValueError("Validity-mask resize interpolation must be explicit nearest")
        if not 0.0 <= self.target_binarization_threshold <= 1.0:
            raise ValueError("target_binarization_threshold must be in [0,1]")
        if self.resize_geometry != "stretch" or self.operation_order != "flip_rotate_resize":
            raise ValueError("Unsupported transform assumption")


class PairedSegmentationTransform:
    """Apply geometrically paired, deterministic-per-sample augmentation."""

    def __init__(self, config: SegmentationTransformConfig, *, base_seed: int) -> None:
        self.config = config
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _generator(self, sample_index: int) -> torch.Generator:
        # Explicit deterministic worker-independent mapping. 1,000,003 is prime.
        return torch.Generator().manual_seed(self.base_seed + self.epoch * 1_000_003 + sample_index)

    def __call__(
        self, image: Image.Image, mask: Image.Image, sample_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.mode != "RGB" or mask.mode != "L":
            raise ValueError("Paired transform requires RGB image and L-mode mask")
        image, mask, _ = self._augment(image, mask, None, sample_index)
        return self._finalize(image, mask)

    def _augment(
        self,
        image: Image.Image,
        mask: Image.Image,
        validity: Optional[Image.Image],
        sample_index: int,
    ) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
        generator = self._generator(sample_index)
        if torch.rand((), generator=generator).item() < self.config.horizontal_flip_probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            if validity is not None:
                validity = TF.hflip(validity)
        if torch.rand((), generator=generator).item() < self.config.rotation_probability:
            low, high = self.config.rotation_degrees
            angle = low + (high - low) * torch.rand((), generator=generator).item()
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=list(self.config.image_fill),
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=self.config.mask_fill,
            )
            if validity is not None:
                validity = TF.rotate(
                    validity,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
        return image, mask, validity

    def _finalize(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(
            image,
            list(self.config.image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            list(self.config.mask_size),
            interpolation=(
                InterpolationMode.BILINEAR
                if self.config.mask_interpolation == "bilinear"
                else InterpolationMode.NEAREST
            ),
        )
        image_tensor = TF.to_tensor(image)
        if self.config.normalization == "imagenet":
            image_tensor = TF.normalize(
                image_tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        mask_tensor = (
            TF.pil_to_tensor(mask).to(torch.float32) / 255.0
            >= self.config.target_binarization_threshold
        ).to(torch.float32)
        return image_tensor, mask_tensor

    def transform_with_validity(
        self,
        image: Image.Image,
        mask: Image.Image,
        validity: Image.Image,
        sample_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.mode != "RGB" or mask.mode != "L" or validity.mode != "L":
            raise ValueError("Validity-aware transform requires RGB/L/L inputs")
        image, mask, validity = self._augment(image, mask, validity, sample_index)
        image_tensor, mask_tensor = self._finalize(image, mask)
        validity = TF.resize(
            validity,
            list(self.config.mask_size),
            interpolation=InterpolationMode.NEAREST,
        )
        validity_tensor = (TF.pil_to_tensor(validity) > 0).to(torch.float32)
        return image_tensor, mask_tensor, validity_tensor

    def transform_with_validity_and_separation(
        self,
        image: Image.Image,
        mask: Image.Image,
        validity: Image.Image,
        separation: Image.Image,
        owners: Image.Image,
        sample_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if separation.mode != "L" or separation.size != mask.size or owners.mode != "I" or owners.size != mask.size:
            raise ValueError("Separation/owner maps must match target geometry")
        image_tensor, mask_tensor, validity_tensor = self.transform_with_validity(
            image, mask, validity, sample_index
        )
        # _augment uses a per-sample deterministic generator: the second call
        # replays the identical flip/rotation for the auxiliary mask.
        dummy = Image.new("RGB", image.size)
        _, separation, _ = self._augment(dummy, separation, None, sample_index)
        _, owners, _ = self._augment(dummy, owners, None, sample_index)
        separation = TF.resize(
            separation, list(self.config.mask_size), interpolation=InterpolationMode.NEAREST
        )
        separation_tensor = (TF.pil_to_tensor(separation) > 0).to(torch.float32)
        separation_tensor *= validity_tensor
        owners = TF.resize(
            owners, list(self.config.mask_size), interpolation=InterpolationMode.NEAREST
        )
        owner_tensor = torch.from_numpy(np.asarray(owners, dtype=np.int32).copy())
        contact = torch.zeros_like(owner_tensor, dtype=torch.bool)
        height, width = owner_tensor.shape
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            y0a, y1a = max(0, -dy), height - max(0, dy)
            x0a, x1a = max(0, -dx), width - max(0, dx)
            y0b, y1b = max(0, dy), height - max(0, -dy)
            x0b, x1b = max(0, dx), width - max(0, -dx)
            left = owner_tensor[y0a:y1a, x0a:x1a]
            right = owner_tensor[y0b:y1b, x0b:x1b]
            different = (left > 0) & (right > 0) & (left != right)
            contact[y0a:y1a, x0a:x1a] |= different
            contact[y0b:y1b, x0b:x1b] |= different
        separation_tensor = torch.maximum(
            separation_tensor, contact.unsqueeze(0).to(torch.float32) * validity_tensor
        )
        mask_tensor *= (owner_tensor > 0).unsqueeze(0).to(torch.float32)
        mask_tensor *= 1.0 - separation_tensor
        return image_tensor, mask_tensor, validity_tensor, separation_tensor

    def metadata(self) -> dict[str, object]:
        return {
            "base_seed": self.base_seed,
            "epoch": self.epoch,
            "paper_specified": {
                "image_size": [256, 256],
                "mask_size": [128, 128],
                "augmentations": ["horizontal_flip", "random_rotation"],
            },
            "approximate_choices": asdict(self.config),
        }


__all__ = ["PairedSegmentationTransform", "SegmentationTransformConfig"]
