"""Explicit experimental Gap partition with optional full Text support recovery."""

import cv2
import numpy as np

from localize.adaspot_gap import gap_instances
from localize.crop_recovery import geodesic_partition


def preserve_support(labels, foreground):
    labels = np.asarray(labels)
    foreground = np.asarray(foreground)
    if (labels.shape != foreground.shape or labels.ndim != 2 or foreground.dtype != bool
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
            or np.any((labels > 0) & ~foreground)):
        raise ValueError("Aligned integer seeds contained in boolean foreground required")
    # Existing minimax flood with constant cost becomes shortest-path propagation;
    # length, then instance ID break ties. Every established ID remains fixed.
    output = geodesic_partition(np.ones(foreground.shape), foreground, labels)
    missing=foreground & (output==0)
    _,components=cv2.connectedComponents(missing.astype(np.uint8),connectivity=8)
    output[missing]=components[missing]+int(output.max())
    if not np.array_equal(output>0,foreground) or not np.array_equal(output[labels>0],labels[labels>0]):
        raise RuntimeError("Text recovery changed existing IDs or failed to recover foreground")
    return output


def decode_gap(words, gap, canvas, postprocessing, threshold, decoder):
    if decoder not in ("cut", "preserve"):
        raise ValueError("Explicit cut/preserve decoder required")
    if threshold is None:
        return gap_instances(words,None,canvas,postprocessing,use_gap=False)[0]
    pp={**postprocessing,"gap_threshold":threshold}
    labels=gap_instances(words,gap,canvas,pp,use_gap=True)[0]
    raw=canvas & (words[0]>=postprocessing["text_threshold"])
    return preserve_support(labels,raw) if decoder=="preserve" else labels
