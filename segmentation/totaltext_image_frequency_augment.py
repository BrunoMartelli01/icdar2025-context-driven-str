"""Isolated TRAIN-only image augmentations for practical Total-Text ablations.

The binary target, validity, separation, and word-owner maps are never modified
by photometric or Fourier operations. Aspect-preserving letterboxing is replayed
identically for all five spatially paired inputs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from segmentation.transforms import PairedSegmentationTransform


_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def letterbox_geometry(image: Image.Image, *, fill: int | tuple[int, int, int]) -> Image.Image:
    """Center-pad to a square without resampling; return an independent image."""
    width, height = image.size
    if min(width, height) < 1:
        raise ValueError("Letterbox requires positive image dimensions")
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    return TF.pad(image, [left, top, side - width - left, side - height - top], fill=fill)


def letterbox_canvas_mask(image_wh: tuple[int, int], *, native_size: int = 256) -> torch.Tensor:
    """Inference-safe valid canvas: depends only on original RGB dimensions."""
    source = Image.new("L", image_wh, 255)
    padded = letterbox_geometry(source, fill=0)
    resized = TF.resize(padded, [native_size, native_size],
                        interpolation=InterpolationMode.NEAREST)
    return TF.pil_to_tensor(resized)[0] > 0


def remap_letterbox_native_box(
    bbox: tuple[int, int, int, int] | list[int], image_wh: tuple[int, int],
    *, native_size: int = 256,
) -> list[int]:
    """Invert square letterbox and native resize for exclusive xyxy coordinates."""
    width, height = image_wh
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if (min(width, height, native_size) <= 0 or not
        (0 <= x1 < x2 <= native_size and 0 <= y1 < y2 <= native_size)):
        raise ValueError("Invalid native box or source image geometry")
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    mapped = [max(0, x1 * side // native_size - left),
              max(0, y1 * side // native_size - top),
              min(width, (x2 * side + native_size - 1) // native_size - left),
              min(height, (y2 * side + native_size - 1) // native_size - top)]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        raise ValueError("Native contour does not intersect the original RGB canvas")
    return mapped


def mix_low_frequency_amplitude(
    source: torch.Tensor, donor: torch.Tensor, *, fraction: float, strength: float
) -> torch.Tensor:
    """Blend only central Fourier amplitudes; retain the source phase and labels."""
    if source.shape != donor.shape or source.ndim != 3 or source.shape[0] != 3:
        raise ValueError("Fourier mixing requires equal CxHxW RGB tensors")
    if not 0 < fraction <= 0.1 or not 0 <= strength <= 1:
        raise ValueError("Fourier fraction/strength out of the predeclared safe range")
    if strength == 0:
        return source.clone()
    height, width = source.shape[-2:]
    radius = max(1, round(fraction * min(height, width) / 2))
    y0, x0 = height // 2 - radius, width // 2 - radius
    y1, x1 = height // 2 + radius, width // 2 + radius
    source_spectrum = torch.fft.fftshift(torch.fft.fft2(source.float()), dim=(-2, -1))
    donor_spectrum = torch.fft.fftshift(torch.fft.fft2(donor.float()), dim=(-2, -1))
    mixed = source_spectrum.clone()
    source_band = source_spectrum[:, y0:y1, x0:x1]
    donor_amplitude = donor_spectrum[:, y0:y1, x0:x1].abs()
    amplitude = (1 - strength) * source_band.abs() + strength * donor_amplitude
    mixed[:, y0:y1, x0:x1] = amplitude * source_band / source_band.abs().clamp_min(1e-8)
    return torch.fft.ifft2(torch.fft.ifftshift(mixed, dim=(-2, -1))).real.clamp(0, 1)


class PracticalTotalTextImageTransform(PairedSegmentationTransform):
    """One intervention at a time; validation never receives image-only noise."""

    def __init__(
        self, config, *, base_seed: int, mode: str, augment_images: bool,
        parameters: dict | None = None, donor_paths: tuple[Path, ...] = (),
        donor_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__(config, base_seed=base_seed)
        if mode not in {"none", "photometric", "letterbox", "fourier_amplitude"}:
            raise ValueError("Unknown practical image intervention")
        if mode == "fourier_amplitude" and augment_images and (
            len(donor_paths) < 2 or len(donor_paths) != len(donor_indices)
        ):
            raise ValueError("Fourier donor pool must contain TRAIN images only")
        self.mode = mode
        self.augment_images = bool(augment_images)
        self.parameters = parameters or {}
        self.donor_paths = donor_paths
        self.donor_indices = donor_indices

    def _augment(self, image, mask, validity, sample_index):
        image, mask, validity = super()._augment(image, mask, validity, sample_index)
        if self.mode == "letterbox":
            image = letterbox_geometry(image, fill=tuple(self.parameters["canvas_fill_rgb"]))
            mask = letterbox_geometry(mask, fill=0)
            if validity is not None:
                validity = letterbox_geometry(validity, fill=0)
        return image, mask, validity

    def _image_only(self, normalized: torch.Tensor, sample_index: int) -> torch.Tensor:
        if not self.augment_images or self.mode not in {"photometric", "fourier_amplitude"}:
            return normalized
        generator = torch.Generator().manual_seed(
            self.base_seed + self.epoch * 1_000_003 + sample_index + 7_000_001
        )
        if torch.rand((), generator=generator).item() >= float(self.parameters["probability"]):
            return normalized
        image = (normalized * _STD + _MEAN).clamp(0, 1)
        if self.mode == "photometric":
            operations = self.parameters["operations"]
            if operations != ["brightness", "contrast", "saturation", "gamma",
                              "gaussian_blur", "gaussian_noise"]:
                raise ValueError("Unexpected photometric operation set/order")
            operation = operations[int(torch.randint(0, len(operations), (), generator=generator).item())]
            magnitude = float(self.parameters["magnitude"])
            unit = float(torch.rand((), generator=generator).item())
            if operation == "brightness":
                image = TF.adjust_brightness(image, 1 + magnitude * (2 * unit - 1))
            elif operation == "contrast":
                image = TF.adjust_contrast(image, 1 + magnitude * (2 * unit - 1))
            elif operation == "saturation":
                image = TF.adjust_saturation(image, 1 + magnitude * (2 * unit - 1))
            elif operation == "gamma":
                image = TF.adjust_gamma(image, 1 + magnitude * (2 * unit - 1))
            elif operation == "gaussian_blur":
                sigma = float(self.parameters["blur_sigma_min"]) + unit * (
                    float(self.parameters["blur_sigma_max"]) - float(self.parameters["blur_sigma_min"])
                )
                image = TF.gaussian_blur(image, [3, 3], [sigma, sigma])
            else:
                noise = torch.randn(image.shape, generator=generator) * float(
                    self.parameters["noise_std_max"]
                ) * unit
                image = image + noise
        else:
            position = int(torch.randint(0, len(self.donor_paths) - 1, (), generator=generator).item())
            if self.donor_indices[position] == sample_index:
                position = (position + 1) % len(self.donor_paths)
            if self.donor_indices[position] == sample_index:
                raise RuntimeError("Fourier donor equals source sample")
            with Image.open(self.donor_paths[position]) as opened:
                donor = TF.to_tensor(TF.resize(opened.convert("RGB"),
                    list(self.config.image_size), antialias=True))
            image = mix_low_frequency_amplitude(image, donor,
                fraction=float(self.parameters["fraction"]),
                strength=float(self.parameters["strength"]))
        return TF.normalize(image.clamp(0, 1), _MEAN.flatten().tolist(), _STD.flatten().tolist())

    def transform_with_validity_and_separation(
        self, image, mask, validity, separation, owners, sample_index
    ):
        result = super().transform_with_validity_and_separation(
            image, mask, validity, separation, owners, sample_index
        )
        return (self._image_only(result[0], sample_index), *result[1:])


__all__ = ["PracticalTotalTextImageTransform", "letterbox_geometry",
           "letterbox_canvas_mask", "remap_letterbox_native_box",
           "mix_low_frequency_amplitude"]
