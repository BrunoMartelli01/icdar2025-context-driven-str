"""Image-only word completeness readout from learned ownership affinities."""
import numpy as np


def neighbor(values, dy, dx):
    h, w = values.shape
    result = np.zeros_like(values)
    y0, y1 = max(0, -dy), min(h, h-dy)
    x0, x1 = max(0, -dx), min(w, w-dx)
    result[y0:y1, x0:x1] = values[y0+dy:y1+dy, x0+dx:x1+dx]
    return result


def closedness(labels, edges, canvas, instance_ids):
    """Internal affinity minus outgoing affinity; no GT or OCR text is accepted.

    Missing coarse support/boundary is explicitly unavailable. Such components
    bypass the optional threshold rather than silently becoming negatives.
    """
    if not edges or labels.ndim != 2 or canvas.shape != edges[0][0].shape:
        raise ValueError('Aligned labels/affinities/RGB canvas required')
    size = edges[0][0].shape
    if labels.shape[0] % size[0] or labels.shape[1] % size[1]:
        raise ValueError('Exact integer label-to-affinity mapping required')
    coarse = labels[::labels.shape[0]//size[0], ::labels.shape[1]//size[1]]
    prepared = []
    for probability, border, dy, dx in edges:
        if probability.shape != size or border.shape != size or not np.isfinite(probability).all():
            raise ValueError('Nonfinite or unaligned ownership affinity')
        if np.any((probability < 0) | (probability > 1)):
            raise ValueError('Ownership probabilities must be in [0,1]')
        eligible = border.astype(bool) & canvas & neighbor(canvas,dy,dx)
        prepared.append((probability, eligible, neighbor(coarse,dy,dx)))
    result = {}
    for identity in instance_ids:
        inside = coarse == identity
        a, b, na, nb = 0., 0., 0, 0
        for probability, valid, other in prepared:
            internal = inside & valid & (other == identity)
            boundary = inside & valid & (other != identity)
            a += float(probability[internal].sum(dtype=np.float64)); na += int(internal.sum())
            b += float(probability[boundary].sum(dtype=np.float64)); nb += int(boundary.sum())
        result[int(identity)] = dict(available=bool(na and nb),
            internal_mean=a/na if na else None, outgoing_mean=b/nb if nb else None,
            internal_edges=na, outgoing_edges=nb,
            score=a/na-b/nb if na and nb else None)
    return result
