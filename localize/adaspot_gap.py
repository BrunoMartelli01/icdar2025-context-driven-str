"""GT-blind kernel BFS with an explicit, auditable predicted-gap barrier."""

import numpy as np

from localize.word_multitask import word_instances


def gap_instances(word_probabilities, gap_probability, canvas, config, *, use_gap=True):
    words, canvas = np.asarray(word_probabilities), np.asarray(canvas)
    if (words.ndim != 3 or words.shape[0] != 2 or words.shape[1:] != canvas.shape
            or canvas.dtype != np.bool_ or not np.isfinite(words).all()
            or np.any((words < 0) | (words > 1))):
        raise ValueError("Expected finite text/kernel probabilities and an RGB-only canvas")
    threshold = config["gap_threshold"]
    if not np.isfinite(threshold) or not 0 < threshold < 1 or config["method"] != "kernel_bfs":
        raise ValueError("AdaSpot requires declared gap threshold and kernel BFS")
    blocked = np.zeros(canvas.shape, bool)
    if gap_probability is not None:
        gap = np.asarray(gap_probability)
        if (gap.shape != canvas.shape or not np.isfinite(gap).all()
                or np.any((gap < 0) | (gap > 1))):
            raise ValueError("Invalid paired gap probabilities")
        if use_gap:
            blocked = canvas & (gap > threshold)  # Proposal's strict > boundary.
    elif use_gap:
        raise ValueError("Gap-enabled decoding requires predicted gap probabilities")
    labels, audit, foreground = word_instances(words, canvas & ~blocked, config)
    raw = canvas & (words[0] >= config["text_threshold"])
    audit.update(gap_barrier_enabled=bool(use_gap), gap_blocked_pixels=int((raw & blocked).sum()),
                 gap_blocked_kernel_pixels=int((raw & blocked & (words[1] >= config["kernel_threshold"])).sum()),
                 raw_foreground_pixels=int(raw.sum()), retained_foreground_pixels=int(foreground.sum()))
    if np.any((labels > 0) & (~raw | blocked)):
        raise RuntimeError("Gap decoding created foreground or crossed a barrier")
    return labels, audit, foreground
