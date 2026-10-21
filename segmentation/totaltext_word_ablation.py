"""Practical losses, paired TRAIN zoom, and inference-only pixel-link grouping."""

import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from segmentation.data import SegmentationDataset
from segmentation.totaltext_image_frequency_augment import PracticalTotalTextImageTransform
from segmentation.training import separation_weighted_bce_with_logits


OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
REVERSE = tuple(OFFSETS.index((-dy, -dx)) for dy, dx in OFFSETS)


def paired_slices(height, width, dy, dx):
    return ((slice(max(0, -dy), height - max(0, dy)), slice(max(0, -dx), width - max(0, dx))),
            (slice(max(0, dy), height - max(0, -dy)), slice(max(0, dx), width - max(0, -dx))))


def link_targets(owners, target, validity):
    """Foreground source to valid neighbor; background/different-owner link is negative."""
    if owners.shape != target.shape or target.shape != validity.shape or owners.ndim != 4 or owners.shape[1] != 1:
        raise ValueError("Link targets expect paired Bx1xHxW maps")
    if torch.any((target > 0) & (validity > 0) & (owners <= 0)):
        raise ValueError("Valid foreground has no word owner")
    h, w = owners.shape[-2:]
    labels = torch.zeros((len(owners), 8, h, w), device=target.device, dtype=target.dtype)
    weights = torch.zeros_like(labels)
    for k, (dy, dx) in enumerate(OFFSETS):
        a, b = paired_slices(h, w, dy, dx)
        oa, ob = owners[:, 0, a[0], a[1]], owners[:, 0, b[0], b[1]]
        fa, fb = target[:, 0, a[0], a[1]], target[:, 0, b[0], b[1]]
        valid = validity[:, 0, a[0], a[1]] * validity[:, 0, b[0], b[1]] * fa
        weights[:, k, a[0], a[1]] = valid
        labels[:, k, a[0], a[1]] = ((oa == ob) & (oa > 0) & (fb > 0)).to(target.dtype)
    return labels, weights


def balanced_link_loss(logits, target, validity):
    if logits.shape != target.shape or target.shape != validity.shape:
        raise ValueError("Link loss shape mismatch")
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    terms = []
    for kind in (target, 1 - target):
        weight = kind * validity
        if weight.sum().item() > 0:
            terms.append((loss * weight).sum() / weight.sum())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def owner_weights(owners, mask, validity):
    labels = np.asarray(owners, dtype=np.int32)
    foreground = (np.asarray(mask) > 0) & (np.asarray(validity) > 0)
    if labels.shape != foreground.shape or np.any(foreground & (labels <= 0)):
        raise ValueError("Foreground/owner alignment changed")
    ids, counts = np.unique(labels[foreground], return_counts=True)
    result = np.zeros_like(labels, dtype=np.float32)
    for owner, count in zip(ids, counts):
        result[(labels == owner) & foreground] = 1.0 / (len(ids) * int(count))
    return result


def word_balanced_loss(logits, target, validity, separation, weights, *, negative_ratio, separation_weight):
    if any(x.shape != logits.shape for x in (target, validity, separation, weights)):
        raise ValueError("Word-balanced loss shape mismatch")
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    values = []
    for i in range(len(logits)):
        positive = (target[i] > 0) & (validity[i] > 0)
        negative = (target[i] == 0) & (validity[i] > 0)
        npos = int(positive.sum().item())
        terms = []
        if npos:
            if not torch.isclose(weights[i].sum(), weights.new_tensor(1.0), atol=1e-5):
                raise ValueError("Per-image word weights must sum to1")
            terms.append((loss[i] * weights[i]).sum())
        if negative.any():
            neg_loss = loss[i][negative]
            neg_weight = 1 + (separation_weight - 1) * separation[i][negative]
            count = min(len(neg_loss), max(1, math.ceil(negative_ratio * npos))) if npos else len(neg_loss)
            indices = torch.topk(neg_loss * neg_weight, count, sorted=False).indices
            terms.append((neg_loss[indices] * neg_weight[indices]).sum() / neg_weight[indices].sum())
        if not terms:
            raise ValueError("Image has no valid training pixels")
        values.append(torch.stack(terms).mean())
    return torch.stack(values).mean()


@lru_cache(maxsize=8)
def _bilinear_matrix(before, after):
    """CPU align_corners=False interpolation weights, including edge clamping."""
    position = ((torch.arange(after, dtype=torch.float64) + 0.5) * before / after - 0.5).clamp(0, before - 1)
    low = position.floor().long()
    high = (low + 1).clamp_max(before - 1)
    fraction = position - low
    matrix = torch.zeros(after, before, dtype=torch.float64)
    rows = torch.arange(after)
    matrix[rows, low] += 1 - fraction
    matrix[rows, high] += fraction
    return matrix.float()


def deterministic_bilinear_resize(logits, size):
    """Separable GEMM avoids nondeterministic CUDA bilinear-backward kernels."""
    h, w = logits.shape[-2:]
    vertical = _bilinear_matrix(h, size[0]).to(device=logits.device, dtype=logits.dtype)
    horizontal = _bilinear_matrix(w, size[1]).to(device=logits.device, dtype=logits.dtype)
    return torch.matmul(torch.matmul(vertical, logits), horizontal.transpose(0, 1))


def ablation_loss(evidence, batch, branch, config):
    target, valid, separator = batch["mask"], batch["validity"], batch["separation"]
    logits = evidence["foreground"]
    if branch == "word_balanced":
        return word_balanced_loss(logits, target, valid, separator, batch["word_weights"],
            negative_ratio=config["hard_negative_ratio"], separation_weight=config["separation_weight"])
    result = separation_weighted_bce_with_logits(logits, target, valid, separator,
                                               separation_weight=config["separation_weight"])
    if branch == "pixel_link":
        labels, weights = link_targets(batch["owners"], target, valid)
        result = result + config["link_loss_weight"] * balanced_link_loss(evidence["links"], labels, weights)
    if branch == "deep_supervision":
        if len(evidence["auxiliary"]) != len(config["auxiliary_weights"]):
            raise ValueError("Auxiliary supervision heads changed")
        for weight, auxiliary in zip(config["auxiliary_weights"], evidence["auxiliary"]):
            upsampled = deterministic_bilinear_resize(auxiliary, target.shape[-2:])
            result = result + weight * separation_weighted_bce_with_logits(
                upsampled, target, valid, separator, separation_weight=config["separation_weight"])
    return result


def pixel_link_labels(foreground, probabilities, threshold):
    """Reciprocal links on an undirected 8-neighbor graph. No GT/owner input."""
    fg = np.asarray(foreground, dtype=bool)
    p = np.asarray(probabilities)
    if fg.ndim != 2 or p.shape != (8, *fg.shape) or not np.isfinite(p).all() or not 0 < threshold < 1:
        raise ValueError("Invalid pixel-link inference arrays/threshold")
    labels = np.zeros(fg.shape, np.int32)
    n = int(fg.sum())
    if not n:
        return labels
    indices = np.full(fg.shape, -1, np.int32)
    indices[fg] = np.arange(n, dtype=np.int32)
    rows, cols = [], []
    h, w = fg.shape
    for k in (4, 5, 6, 7):
        a, b = paired_slices(h, w, *OFFSETS[k])
        edge = fg[a] & fg[b] & (p[k][a] >= threshold) & (p[REVERSE[k]][b] >= threshold)
        rows.append(indices[a][edge])
        cols.append(indices[b][edge])
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row), np.uint8), (row, col)), shape=(n, n)).tocsr()
    _, groups = connected_components(graph, directed=False, return_labels=True)
    labels[fg] = groups.astype(np.int32) + 1
    return labels


def inference_labels(evidence, canvas, config, offset=0):
    fg = (torch.sigmoid(evidence["foreground"][offset, 0]) >= config["probability_threshold"]).cpu().numpy() & canvas
    if evidence["links"] is None:
        return cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)[1]
    probabilities = torch.sigmoid(evidence["links"][offset]).cpu().numpy()
    return pixel_link_labels(fg, probabilities, config["link_threshold"])


def label_boxes(labels):
    """Deterministic y/x ordered original component unions, no filtering/padding."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Labels must be a nonnegative integer HxW map")
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if not len(ids):
        return []
    yy, xx = np.nonzero(labels)
    positions = np.searchsorted(ids, labels[yy, xx])
    xmin, ymin = np.full(len(ids), labels.shape[1]), np.full(len(ids), labels.shape[0])
    xmax, ymax, area = np.zeros(len(ids), int), np.zeros(len(ids), int), np.zeros(len(ids), int)
    np.minimum.at(xmin, positions, xx); np.minimum.at(ymin, positions, yy)
    np.maximum.at(xmax, positions, xx + 1); np.maximum.at(ymax, positions, yy + 1)
    np.add.at(area, positions, 1)
    rows = [{"label": int(ids[i]), "bbox_native": [int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])],
             "foreground_area": int(area[i])} for i in range(len(ids))]
    return sorted(rows, key=lambda r: (r["bbox_native"][1], r["bbox_native"][0], r["label"]))


def choose_word_crop(owners, generator, fractions):
    """TRAIN-only anchor selection; partially cut OTHER words are ignored later."""
    labels = np.asarray(owners)
    ids = np.unique(labels); ids = ids[ids > 0]
    if not len(ids):
        return None
    owner = ids[int(torch.randint(len(ids), (), generator=generator))]
    y, x = np.where(labels == owner)
    h, w = labels.shape
    fraction = fractions[int(torch.randint(len(fractions), (), generator=generator))]
    side = math.ceil(float(fraction) * max(h, w))
    cw, ch = min(w, max(side, int(x.max() - x.min() + 1))), min(h, max(side, int(y.max() - y.min() + 1)))
    low_x, high_x = max(0, int(x.max() + 1) - cw), min(int(x.min()), w - cw)
    low_y, high_y = max(0, int(y.max() + 1) - ch), min(int(y.min()), h - ch)
    left = int(torch.randint(low_x, high_x + 1, (), generator=generator))
    top = int(torch.randint(low_y, high_y + 1, (), generator=generator))
    return (left, top, left + cw, top + ch)


def crop_ignore_mask(owners, bbox):
    labels = np.asarray(owners)
    x1, y1, x2, y2 = bbox
    piece = labels[y1:y2, x1:x2]
    ids, counts = np.unique(piece, return_counts=True)
    all_ids, all_counts = np.unique(labels, return_counts=True)
    totals = dict(zip(all_ids, all_counts))
    partial = [owner for owner, count in zip(ids, counts) if owner > 0 and count != totals[owner]]
    return np.isin(piece, partial)


class WordTrainingTransform(PracticalTotalTextImageTransform):
    def __init__(self, base, *, owner_paths, zoom, zoom_probability, zoom_fractions):
        super().__init__(base.config, base_seed=base.base_seed, mode=base.mode,
                         augment_images=base.augment_images, parameters=base.parameters)
        self.owner_paths = tuple(owner_paths)
        self.zoom = zoom
        self.zoom_probability = zoom_probability
        self.zoom_fractions = tuple(zoom_fractions)
        self._crop_key = None
        self._crop_value = None

    def crop_plan(self, index):
        key = (self.epoch, index)
        if self._crop_key == key:
            return self._crop_value
        bbox, ignored = None, None
        if self.zoom:
            generator = torch.Generator().manual_seed(self.base_seed + self.epoch * 1_000_003 + index + 31_000_003)
            if torch.rand((), generator=generator).item() < self.zoom_probability:
                with Image.open(self.owner_paths[index]) as opened:
                    owners = np.asarray(opened, dtype=np.int32).copy()
                bbox = choose_word_crop(owners, generator, self.zoom_fractions)
                if bbox is not None:
                    ignored = crop_ignore_mask(owners, bbox)
        self._crop_key, self._crop_value = key, (bbox, ignored)
        return self._crop_value

    def _augment(self, image, mask, validity, sample_index):
        bbox, ignored = self.crop_plan(sample_index)
        if bbox is not None:
            image, mask = image.crop(bbox), mask.crop(bbox)
            if validity is not None:
                array = np.asarray(validity.crop(bbox)).copy()
                array[ignored] = 0
                validity = Image.fromarray(array, mode="L")
        return super()._augment(image, mask, validity, sample_index)

    def transform_owners(self, owners, sample_index, target, validity):
        _, owners, _ = self._augment(Image.new("RGB", owners.size), owners, None, sample_index)
        owners = TF.resize(owners, list(self.config.mask_size), interpolation=InterpolationMode.NEAREST)
        labels = np.asarray(owners, dtype=np.int32).copy()
        labels[(target[0].numpy() == 0) | (validity[0].numpy() == 0)] = 0
        return torch.from_numpy(labels).unsqueeze(0)


class OwnedSegmentationDataset(SegmentationDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self.paths[index][4] is None:
            raise ValueError("Word ablations require qualified owner maps")
        with Image.open(self.paths[index][4]) as opened:
            owners = opened.convert("I")
        result["owners"] = self.transform.transform_owners(owners, index, result["mask"], result["validity"])
        weights = owner_weights(result["owners"][0].numpy(), result["mask"][0].numpy(), result["validity"][0].numpy())
        result["word_weights"] = torch.from_numpy(weights).unsqueeze(0)
        return result
