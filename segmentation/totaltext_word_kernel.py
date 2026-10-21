"""Practical word nuclei targets and GT-blind simultaneous region expansion.

This is a two-map adaptation, NOT PSENet or paper-faithful Algorithm 1.
GT owner labels are used only by target construction and validation metrics.
"""

from collections import deque

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional as TF

from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.train_totaltext_v4_highres_ablation import _box_iou


def _integer_map(value):
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError("Expected nonnegative integer label map")
    return array


def make_word_kernels(owners, *, distance_fraction=0.5):
    """Shrink each native word independently; keep its largest connected core.

    Distance is computed against zero-padded borders. Ties use row-major CC
    order. A surviving 1-pixel word keeps its pixel; absent words are NOT made up.
    """
    owners = _integer_map(owners)
    if not 0 < distance_fraction < 1:
        raise ValueError("Kernel distance fraction must be in (0,1)")
    kernels = np.zeros(owners.shape, np.int32)
    rows = []
    for owner in np.unique(owners):
        if owner == 0:
            continue
        ys, xs = np.where(owners == owner)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        # Exact ROI optimization: all pixels outside this owner box are zero.
        # Zero-padding gives the SAME distances as the full native canvas.
        word = owners[y1:y2, x1:x2] == owner
        distance = cv2.distanceTransform(np.pad(word.astype(np.uint8), 1),
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        # Euclidean squared distances on the pixel grid are integers. Comparing
        # these avoids +/-ULP threshold changes when OpenCV uses different ROI
        # sizes/parallel paths (e.g. sqrt(2) versus half of sqrt(8)).
        squared_distance = np.rint(distance.astype(np.float64) ** 2).astype(np.int32)
        core = word & (squared_distance >= distance_fraction ** 2 * int(squared_distance.max()))
        count, components, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
            core.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
        if count <= 1:
            raise RuntimeError("Nonempty word lost its nucleus")
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        chosen = components == best
        kernels[y1:y2, x1:x2][chosen] = int(owner)
        rows.append({"owner_id": int(owner), "foreground_pixels": int(word.sum()),
                     "kernel_pixels": int(chosen.sum()), "core_fragments_before_selection": count - 1})
    # Seeds must really be separable by the inference CC rule, not just owner IDs.
    count, labels = cv2.connectedComponents((kernels > 0).astype(np.uint8), connectivity=8)
    if count - 1 != len(rows):
        raise ValueError("Target nuclei touch across words; unsafe target qualification")
    if np.any((kernels > 0) & (owners != kernels)):
        raise RuntimeError("Nucleus outside its original word")
    return kernels, rows


def native_owner_map(owner_image, transform, sample_index, target, validity):
    """Replay EXACT paired flip/rotation/letterbox; nearest owner IDs, no mixing."""
    dummy = Image.new("RGB", owner_image.size)
    _, owner_image, _ = transform._augment(dummy, owner_image, None, sample_index)
    owner_image = TF.resize(owner_image, list(transform.config.mask_size),
                            interpolation=InterpolationMode.NEAREST)
    owners = np.asarray(owner_image, dtype=np.int32).copy()
    foreground = np.asarray(target).squeeze().astype(bool)
    valid = np.asarray(validity).squeeze().astype(bool)
    if owners.shape != foreground.shape or owners.shape != valid.shape:
        raise ValueError("Owner/target/validity mapping mismatch")
    owners[~foreground | ~valid] = 0
    return owners


class WordKernelDataset(Dataset):
    """Augment the qualified dataset with kernels; never alter its full target."""

    def __init__(self, base, *, distance_fraction=0.5):
        self.base = base
        self.distance_fraction = distance_fraction

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = dict(self.base[index])
        sample = self.base.manifest.samples[index]
        if sample.word_owner_annotation_path is None:
            raise ValueError("Qualified v4 word-owner annotation is required")
        with Image.open(self.base.data_root / sample.word_owner_annotation_path) as opened:
            owners = native_owner_map(opened.convert("I"), self.base.transform, index,
                                      batch["mask"], batch["validity"])
        kernels, _ = make_word_kernels(owners, distance_fraction=self.distance_fraction)
        batch["kernel"] = torch.from_numpy((kernels > 0).astype(np.float32))[None]
        batch["owners_native"] = torch.from_numpy(owners)
        return batch


def expand_word_kernels(foreground, kernels):
    """GT-free 8-neighbour multi-source BFS; equal-hop ties use seed/queue order.

    A full foreground CC without any seed is retained as ONE fallback instance.
    Output is an INSTANCE label map: converting it to bool would undo separation.
    """
    foreground, kernels = np.asarray(foreground), np.asarray(kernels)
    if (foreground.ndim != 2 or kernels.shape != foreground.shape
        or foreground.dtype != np.bool_ or kernels.dtype != np.bool_):
        raise ValueError("Expansion expects paired 2D boolean masks")
    seeds = kernels & foreground
    count, labels = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
    labels = labels.astype(np.int32)
    queue = deque((int(y), int(x)) for y, x in np.argwhere(labels > 0))
    height, width = labels.shape
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        y, x = queue.popleft()
        label = labels[y, x]
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width
                and foreground[ny, nx] and labels[ny, nx] == 0):
                labels[ny, nx] = label
                queue.append((ny, nx))
    missing = foreground & (labels == 0)
    fallback_count, fallback = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
    labels[missing] = fallback[missing] + count - 1
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Expansion changed full foreground support")
    return labels, {"seed_components": count - 1,
                    "unseeded_fallback_components": fallback_count - 1,
                    "kernel_pixels_outside_foreground": int((kernels & ~foreground).sum()),
                    "instances": count + fallback_count - 2}


def instance_proposals(labels):
    """One tight box per instance ID, NEVER contours on the binary union."""
    labels = _integer_map(labels)
    proposals = []
    for label in np.unique(labels):
        if label == 0:
            continue
        ys, xs = np.where(labels == label)
        proposals.append({"instance_id": int(label), "foreground_area": len(xs),
                          "bbox_native": [int(xs.min()), int(ys.min()),
                                          int(xs.max()) + 1, int(ys.max()) + 1]})
    proposals.sort(key=lambda p: (-p["foreground_area"], p["bbox_native"][1],
                                  p["bbox_native"][0], p["instance_id"]))
    return proposals


def instance_geometry(labels, owners, expected_ids):
    """Native TRAIN diagnostic; same any-pixel fusion/split rule as prior runs."""
    labels, owners = _integer_map(labels), _integer_map(owners)
    if labels.shape != owners.shape:
        raise ValueError("Paired geometry shapes differ")
    proposals = instance_proposals(labels)
    gt_boxes, word_components = [], {}
    for owner in expected_ids:
        ys, xs = np.where(owners == owner)
        gt_boxes.append(None if not len(xs) else [int(xs.min()), int(ys.min()),
                                                 int(xs.max()) + 1, int(ys.max()) + 1])
        word_components[owner] = set(int(x) for x in np.unique(labels[owners == owner]) if x)
    fused = [{owner for owner, components in word_components.items()
              if p["instance_id"] in components} for p in proposals]
    fused = [words for words in fused if len(words) >= 2]
    result = {"gt_words": len(expected_ids), "raw_proposals": len(proposals),
              "gt_words_lost_in_native_target": sum(box is None for box in gt_boxes),
              "gt_words_split_across_predicted_components": sum(len(x) >= 2 for x in word_components.values()),
              "predicted_components_touching_multiple_gt_words": len(fused),
              "gt_words_touched_by_fused_components": len(set().union(*fused)) if fused else 0}
    for threshold, key in ((0.25, "25"), (0.5, "50"), (0.75, "75")):
        edges = [(pi, gi) for pi, p in enumerate(proposals) for gi, box in enumerate(gt_boxes)
                 if box is not None and _box_iou(p["bbox_native"], box) >= threshold]
        result[f"matched_gt_iou_ge_{key}"] = len(maximum_matching(len(proposals), len(gt_boxes), edges))
    return result
