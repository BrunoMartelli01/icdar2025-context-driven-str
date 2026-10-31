"""Isolated APPROXIMATE native512 targets and training-only ownership edges."""
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from models.directional_transport_unet import shifted
from segmentation.transforms import SegmentationTransformConfig
from segmentation.totaltext_image_frequency_augment import PracticalTotalTextImageTransform
from segmentation.word_multitask import masked_bce_dice


@dataclass(frozen=True)
class TransportTransformConfig(SegmentationTransformConfig):
    """Validate a new resolution without relaxing the original strict boundary."""

    def __post_init__(self):
        if (self.image_size != (512, 512) or self.mask_size not in ((256, 256), (512, 512))
                or self.classification != 'APPROXIMATE_DIRECTIONAL_TRANSPORT_V1'):
            raise ValueError('Explicit transport512-to-256/512 transform required')
        proxy = asdict(self)
        proxy.update(mask_size=(256, 256), classification='PRACTICAL_APPROXIMATION_HIGH_RESOLUTION')
        SegmentationTransformConfig(**proxy)


def transform(config, *, native_size, train):
    a = config['augmentation']
    paired = TransportTransformConfig(
        horizontal_flip_probability=a['horizontal_flip_probability'] if train else 0.,
        rotation_probability=a['rotation_probability'] if train else 0.,
        rotation_degrees=tuple(a['rotation_degrees']), image_interpolation='bilinear',
        mask_interpolation='bilinear', image_fill=(0, 0, 0), mask_fill=0,
        normalization='imagenet', resize_geometry='stretch', operation_order='flip_rotate_resize',
        image_size=(512, 512), mask_size=(native_size, native_size),
        classification='APPROXIMATE_DIRECTIONAL_TRANSPORT_V1')
    return PracticalTotalTextImageTransform(paired, base_seed=config['seed'],
        mode='letterbox', augment_images=train, parameters={'canvas_fill_rgb': a['canvas_fill_rgb']})


def balanced_edge_loss(logits, owner, validity, dy, dx, border):
    """Positive=same foreground owner; negative=different/background endpoint.

    Background/background and invalid pairs contribute zero. Ownership is only
    accepted here, after the image-only network forward has completed.
    """
    size = logits.shape[-2:]
    owner = F.interpolate(owner[:, None].float(), size, mode='nearest')
    validity = F.interpolate(validity.float(), size, mode='nearest') > 0
    neighbor, _ = shifted(owner, dy, dx)
    neighbor_valid, _ = shifted(validity.float(), dy, dx)
    eligible = validity & (neighbor_valid > 0) & (border > 0) & ((owner > 0) | (neighbor > 0))
    positive = eligible & (owner > 0) & (owner == neighbor)
    negative = eligible & ~positive
    loss = F.binary_cross_entropy_with_logits(logits, positive.float(), reduction='none')
    terms = [loss[mask].mean() for mask in (positive, negative) if bool(mask.any())]
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def objective(evidence, batch, config):
    words = evidence['words']
    if words.shape != (len(batch['mask']), 2, *batch['mask'].shape[-2:]):
        raise ValueError('Transport output/target resolution mismatch')
    options = {k: config[k] for k in ('bce_weight', 'dice_weight', 'smooth')}
    text = masked_bce_dice(words[:, :1], batch['mask'], batch['validity'], **options,
        pixel_weight=1 + (config['separation_weight'] - 1) * batch['separation'])
    kernel = masked_bce_dice(words[:, 1:2], batch['kernel'], batch['validity'], **options)
    edges = [balanced_edge_loss(logits, batch['owners_native'], batch['validity'], dy, dx, border)
             for logits, border, dy, dx in evidence.get('edges', [])]
    edge = torch.stack(edges).mean() if edges else words.sum() * 0
    total = text + kernel + config['ownership_weight'] * edge
    return total, {'text': text, 'kernel': kernel, 'ownership': edge}
