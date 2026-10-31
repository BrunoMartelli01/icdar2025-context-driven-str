"""Explicit word-box and strict-state boundaries for approximate YOLO fitting."""

import numpy as np


def training_roles(partitions, protocol):
    """Resolve either held-out model selection or a fixed-epoch final refit."""
    fit = list(partitions['fit'])
    validation = sorted(partitions['selection'] + partitions['calibration'])
    if len(set(fit + validation)) != len(fit) + len(validation):
        raise ValueError('Detector FIT and validation roles must be disjoint before final refit')
    if protocol == 'heldout_selection':
        train = fit
    elif protocol == 'all_train_fixed_epochs':
        train = sorted(fit + validation)
    else:
        raise ValueError('Explicit supported detector fitting protocol required')
    return dict(train=train, validation=validation)


def normalized_box(box, image_wh):
    w, h = image_wh
    a, b, c, d = box
    if w <= 0 or h <= 0 or not (0 <= a < c <= w and 0 <= b < d <= h):
        raise ValueError('Positive exclusive in-image word box required')
    return [(a+c)/(2*w), (b+d)/(2*h), (c-a)/w, (d-b)/h]


def mask_training_ignore(rgb, validity, fill):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Original RGB uint8 training image required')
    if validity.shape != rgb.shape[:2] or validity.dtype != np.bool_ or not 0 <= fill <= 255:
        raise ValueError('Original-frame boolean validity mask and explicit fill required')
    result = rgb.copy(); result[~validity] = fill
    return result


def strict_initialization(model, state):
    import torch
    current = model.state_dict()
    missing = sorted(set(current)-set(state)); unexpected = sorted(set(state)-set(current))
    wrong = sorted(k for k in set(current)&set(state) if current[k].shape != state[k].shape)
    if missing or unexpected or wrong:
        raise ValueError(f'Incomplete detector initialization: missing={missing}, unexpected={unexpected}, shapes={wrong}')
    model.load_state_dict(state, strict=True)
    if any(not torch.equal(value.cpu(), state[k].cpu()) for k, value in model.state_dict().items()):
        raise ValueError('Detector initialization values differ after strict load')
    return dict(strict=True, tensors=len(current), missing=[], unexpected=[], wrong_shapes=[])
