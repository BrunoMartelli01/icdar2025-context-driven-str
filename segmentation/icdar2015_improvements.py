"""Isolated approximate ICDAR2015 resolution, ownership and word-loss variants."""

from dataclasses import asdict, dataclass
from math import gcd

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional as TF

from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from segmentation.totaltext_word_kernel import native_owner_map
from segmentation.transforms import PairedSegmentationTransform


@dataclass(frozen=True)
class ImprovementTransformConfig:
    """Explicit sizes; never accepted by STRICT or historical transform configs."""

    image_size: tuple[int, int]
    mask_size: tuple[int, int]
    horizontal_flip_probability: float
    rotation_probability: float
    rotation_degrees: tuple[float, float]
    image_fill: tuple[int, int, int] = (0, 0, 0)
    mask_fill: int = 0
    image_interpolation: str = "bilinear"
    mask_interpolation: str = "bilinear"
    validity_interpolation: str = "nearest"
    normalization: str = "imagenet"
    resize_geometry: str = "explicit_aspect_letterbox"
    operation_order: str = "flip_rotate_resize"
    target_binarization_threshold: float = 0.5
    classification: str = "ASSUMPTION"
    reproduction_mode: str = "approximate"

    def __post_init__(self):
        if self.reproduction_mode != "approximate":
            raise ValueError("Improvement transforms are approximate only")
        if (any(d < 32 or d % 32 for d in self.image_size)
                or tuple(d // 2 for d in self.image_size) != self.mask_size):
            raise ValueError("RGB sizes must be multiples of 32 and masks half resolution")
        if not all(0 <= p <= 1 for p in (self.horizontal_flip_probability, self.rotation_probability)):
            raise ValueError("Invalid augmentation probability")
        if self.rotation_degrees[0] > self.rotation_degrees[1]:
            raise ValueError("Invalid rotation interval")
        if (self.image_interpolation, self.mask_interpolation, self.validity_interpolation,
                self.normalization, self.operation_order) != ("bilinear", "bilinear", "nearest", "imagenet", "flip_rotate_resize"):
            raise ValueError("Do not silently alter the qualified interpolation/normalization")


def padding_geometry(image_wh, target_hw):
    """Smallest integer source canvas with the exact target aspect ratio."""
    width, height = image_wh
    th, tw = target_hw
    if min(width, height, th, tw) <= 0:
        raise ValueError("Positive image/target dimensions required")
    divisor = gcd(th, tw)
    rw, rh = tw // divisor, th // divisor
    scale = max((width + rw - 1) // rw, (height + rh - 1) // rh)
    cw, ch = rw * scale, rh * scale
    return cw, ch, (cw - width) // 2, (ch - height) // 2


def pad_to_aspect(image, target_hw, *, fill):
    cw, ch, left, top = padding_geometry(image.size, target_hw)
    return TF.pad(image, [left, top, cw-image.width-left, ch-image.height-top], fill=fill)


class ImprovementTransform(PairedSegmentationTransform):
    def __init__(self, config, *, base_seed, canvas_fill):
        super().__init__(config, base_seed=base_seed)
        self.canvas_fill = tuple(canvas_fill)

    def _augment(self, image, mask, validity, sample_index):
        image, mask, validity = super()._augment(image, mask, validity, sample_index)
        image = pad_to_aspect(image, self.config.image_size, fill=self.canvas_fill)
        mask = pad_to_aspect(mask, self.config.image_size, fill=0)
        if validity is not None:
            validity = pad_to_aspect(validity, self.config.image_size, fill=0)
        return image, mask, validity

    def metadata(self):
        return {**asdict(self.config), "canvas_fill": list(self.canvas_fill),
                "padding": "minimal_integer_source_canvas_exact_target_aspect"}


def make_transform(config, image_hw, *, augment):
    training = config["training"]
    return ImprovementTransform(ImprovementTransformConfig(
        image_size=tuple(image_hw), mask_size=tuple(d//2 for d in image_hw),
        horizontal_flip_probability=training["horizontal_flip_probability"] if augment else 0,
        rotation_probability=training["rotation_probability"] if augment else 0,
        rotation_degrees=tuple(training["rotation_degrees"]),
        target_binarization_threshold=config["probability_threshold"]),
        base_seed=training["seed"], canvas_fill=config["canvas_fill_rgb"])


def valid_canvas(image_wh, image_hw):
    image = pad_to_aspect(Image.new("L", image_wh, 255), image_hw, fill=0)
    return np.asarray(image.resize((image_hw[1]//2, image_hw[0]//2), Image.Resampling.NEAREST)) > 0


def remap_native_box(box, image_wh, image_hw):
    nh, nw = image_hw[0]//2, image_hw[1]//2
    x1, y1, x2, y2 = map(int, box)
    if not (0 <= x1 < x2 <= nw and 0 <= y1 < y2 <= nh):
        raise ValueError("Box outside declared native mask")
    cw, ch, left, top = padding_geometry(image_wh, image_hw)
    w, h = image_wh
    mapped = [max(0, x1*cw//nw-left), max(0, y1*ch//nh-top),
              min(w, (x2*cw+nw-1)//nw-left), min(h, (y2*ch+nh-1)//nh-top)]
    if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
        raise ValueError("Native component does not intersect RGB canvas")
    return mapped


class OwnerDataset(Dataset):
    """Replay exact augmentation for owners; never supply GT to inference."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        sample = self.base.manifest.samples[index]
        with Image.open(self.base.data_root / sample.word_owner_annotation_path) as source:
            owner = native_owner_map(source.convert("I"), self.base.transform, index,
                                     item["mask"], item["validity"])
        item["owners"] = torch.from_numpy(owner)
        return item


def owner_topology(mask, owners, expected_ids):
    """All original owners stay in denominators, even when native pixels vanish."""
    _, labels = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    components, lost, fragmented = {}, 0, 0
    for owner in expected_ids:
        ids = set(int(x) for x in np.unique(labels[owners == owner]) if x)
        lost += not bool(ids)
        fragmented += len(ids) > 1
        for component in ids:
            components.setdefault(component, set()).add(owner)
    fused = [v for v in components.values() if len(v) > 1]
    return {"words": len(expected_ids), "lost_words": int(lost), "fragmented_words": int(fragmented),
        "components": int(labels.max()), "fused_components": len(fused),
        "words_touched_by_fused_components": len(set().union(*fused)) if fused else 0}


def native_target_audit_maps(target, validity, separator, owners, image_hw, *, threshold):
    """Independent PIL/NumPy replay of unaugmented targets, before/after guard."""
    size = (image_hw[1]//2, image_hw[0]//2)
    def resize(image, mode):
        return np.asarray(pad_to_aspect(image, image_hw, fill=0).resize(size, mode)).copy()
    mask = resize(target.convert("L"), Image.Resampling.BILINEAR)/255 >= threshold
    valid = resize(validity.convert("L"), Image.Resampling.NEAREST) > 0
    sep = resize(separator.convert("L"), Image.Resampling.NEAREST) > 0
    owner = resize(owners.convert("I"), Image.Resampling.NEAREST).astype(np.int32)
    raw = mask & valid & (owner > 0)
    contact = np.zeros(owner.shape, dtype=bool)
    h, w = owner.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ya, yb = slice(max(0,-dy),h-max(0,dy)), slice(max(0,dy),h-max(0,-dy))
        xa, xb = slice(max(0,-dx),w-max(0,dx)), slice(max(0,dx),w-max(0,-dx))
        left, right = owner[ya,xa], owner[yb,xb]
        different = (left > 0) & (right > 0) & (left != right)
        contact[ya,xa] |= different
        contact[yb,xb] |= different
    final = raw & ~((sep | contact) & valid)
    return raw, final, owner, valid


def binary_proposals(mask, image_wh, image_hw):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        native = [int(x), int(y), int(x+w), int(y+h)]
        rows.append({"component_index": index, "component_area": int(area),
            "bbox_native": native, "bbox": remap_native_box(native, image_wh, image_hw)})
    rows.sort(key=lambda row: (-row["component_area"], row["component_index"]))
    return rows, labels


def original_geometry(predictions, words, *, iou_threshold):
    edges = [(pi, gi) for pi, p in enumerate(predictions) for gi, g in enumerate(words)
             if box_iou(p["bbox"], g["bbox"]) >= iou_threshold]
    tp = len(maximum_matching(len(predictions), len(words), edges))
    return {"matched_gt": tp, "gt_words": len(words), "proposals": len(predictions)}


def instance_balanced_loss(logits, target, validity, separation, owners, *, separation_weight,
                           dice_weight, dice_margin, epsilon):
    """Equal total positive cost per word, balanced background and local word Dice.

    Other owner pixels are excluded from each Dice ROI. GT is supervision only.
    Ignore/padding has zero gradient. Absent owners are not fabricated.
    """
    if logits.shape != target.shape or target.shape != validity.shape or target.shape != separation.shape:
        raise ValueError("Loss maps must have equal Bx1xHxW shapes")
    if owners.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError("Owner map must have shape BxHxW")
    if torch.any((target > 0) & ((validity == 0) | (separation > 0))):
        raise ValueError("Invalid word foreground in supervision")
    positive_terms, dice_terms = [], []
    probability = logits.sigmoid()
    for bi in range(len(logits)):
        ids = torch.unique(owners[bi])
        for owner in ids[ids > 0]:
            own = (owners[bi] == owner) & target[bi, 0].bool() & validity[bi, 0].bool()
            if not own.any():
                continue
            positive_terms.append(F.softplus(-logits[bi, 0][own]).mean())
            yy, xx = torch.where(own)
            y1, y2 = max(0, int(yy.min())-dice_margin), min(own.shape[0], int(yy.max())+dice_margin+1)
            x1, x2 = max(0, int(xx.min())-dice_margin), min(own.shape[1], int(xx.max())+dice_margin+1)
            region_owner = owners[bi, y1:y2, x1:x2]
            allowed = validity[bi, 0, y1:y2, x1:x2] * ((region_owner == 0) | (region_owner == owner))
            p = probability[bi, 0, y1:y2, x1:x2] * allowed
            g = own[y1:y2, x1:x2].to(p.dtype)
            dice_terms.append(1 - (2*(p*g).sum()+epsilon)/(p.sum()+g.sum()+epsilon))
    zero = logits.sum()*0
    positive = torch.stack(positive_terms).mean() if positive_terms else zero
    negative_weight = validity * (1-target) * (1 + (separation_weight-1)*separation)
    negative = (F.softplus(logits)*negative_weight).sum()/negative_weight.sum().clamp_min(1)
    dice = torch.stack(dice_terms).mean() if dice_terms else zero
    return 0.5*(positive+negative) + dice_weight*dice
