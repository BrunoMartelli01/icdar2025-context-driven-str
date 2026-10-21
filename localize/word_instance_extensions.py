"""GT-free prompt selection and graph partitions for local word experiments."""

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from segmentation.word_instance_extensions import graph_edges


def anchored_component(labels, prompt):
    if labels.shape != prompt.shape:
        raise ValueError("Prompt and partition shapes differ")
    ids, counts = np.unique(labels[prompt & (labels > 0)], return_counts=True)
    return labels == ids[int(np.argmax(counts))] if len(ids) else None


def mask_component(probability, prompt, canvas, *, threshold):
    if not 0 < threshold < 1 or probability.shape != prompt.shape or canvas.shape != prompt.shape:
        raise ValueError("Invalid local mask geometry/threshold")
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite local mask")
    _, labels = cv2.connectedComponents(((probability >= threshold) & canvas).astype(np.uint8), connectivity=8)
    return anchored_component(labels, prompt)


def affinity_component(probability, foreground, prompt, *, threshold):
    if probability.shape != (4, *foreground.shape) or not np.isfinite(probability).all() or not 0 < threshold < 1:
        raise ValueError("Invalid local affinity geometry/threshold")
    u, v, indexes = graph_edges(foreground.shape, foreground)
    selected = probability.reshape(-1)[indexes] >= threshold
    # Retain isolated foreground pixels. Background vertices never become proposals.
    graph = coo_matrix((np.ones(int(selected.sum()), np.uint8), (u[selected], v[selected])),
                       shape=(foreground.size, foreground.size)).tocsr()
    _, assignments = connected_components(graph, directed=False)
    labels = (assignments.reshape(foreground.shape) + 1) * foreground
    return anchored_component(labels, prompt)


def support_box(support):
    yy, xx = np.where(support)
    if not len(xx):
        raise ValueError("Empty local support")
    return np.asarray([xx.min(), yy.min(), xx.max()+1, yy.max()+1], dtype=np.float32)


def remap_local_box(box, window, image_wh, *, local_size, native_size):
    """Invert the exact local resize, then square letterbox; outward exclusive box."""
    box = np.asarray(box, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Invalid local box")
    if min(local_size, native_size, *image_wh) <= 0 or np.any(window[2:] <= window[:2]):
        raise ValueError("Invalid local mapping geometry")
    width, height = image_wh
    side = max(width, height)
    offset = np.asarray([(side-width)//2, (side-height)//2] * 2)
    local_scale = np.tile((window[2:]-window[:2]) / local_size, 2)
    native_box = box * local_scale + np.tile(window[:2], 2)
    original = native_box * (side/native_size) - offset
    result = np.r_[np.floor(original[:2]), np.ceil(original[2:])].astype(int)
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height)
    if np.any(result[2:] <= result[:2]):
        raise ValueError("Local prediction projects outside original RGB")
    return result.tolist()
