"""APPROXIMATE joint supervision: predicted ROIs, matched labels, detached gate."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F


def masked_dice(logits, target, validity, *, smooth):
    if logits.shape != target.shape or target.shape != validity.shape or smooth <= 0:
        raise ValueError("Aligned Dice supervision and positive smoothing required")
    if not torch.isfinite(logits).all() or torch.any((target != 0) & (target != 1)) or torch.any((validity != 0) & (validity != 1)):
        raise ValueError("Finite logits and binary targets/validity required")
    if not validity.any():
        return logits.sum() * 0
    prediction = logits.sigmoid() * validity
    target = target * validity
    return 1 - (2 * (prediction * target).sum() + smooth) / (prediction.sum() + target.sum() + smooth)


def segmentation_objective(evidence, batch, config):
    pairs = [(evidence["words"][:, :1], batch["mask"]),
             (evidence["words"][:, 1:2], batch["kernel"]),
             (evidence["gap"], batch["gap_target"])]
    terms = {name: masked_dice(logits, target, batch["validity"], smooth=config["dice_smooth"])
             for name, (logits, target) in zip(("text", "kernel", "gap"), pairs)}
    return sum(terms.values()), terms


def match_owner_texts(proposals, owners, texts, *, iou_threshold):
    """Training-only one-to-one box matching; no GT feeds predicted geometry."""
    ids = sorted(int(i) for i in np.unique(owners) if i > 0)
    if any(i not in texts for i in ids):
        raise ValueError("Word-owner annotation has no corresponding transcription")
    boxes = []
    for owner in ids:
        y, x = np.where(owners == owner)
        boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
    if not proposals or not boxes:
        return {}, {"native_owners": len(ids), "matched_rois": 0}
    a, b = np.asarray([p["bbox_native"] for p in proposals]), np.asarray(boxes)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    # Maximum count first, IoU tie-break second, without greedy duplicate labels.
    weight = (iou >= iou_threshold).astype(float) * (min(iou.shape) + 1) + iou
    left, right = linear_sum_assignment(-weight)
    matched = {int(i): {"owner_id": ids[j], "text": texts[ids[j]], "iou": float(iou[i, j])}
               for i, j in zip(left, right) if iou[i, j] >= iou_threshold}
    return matched, {"native_owners": len(ids), "matched_rois": len(matched)}


def label_support(text, tokenizer, max_length):
    if not text:
        return "empty_annotation"
    if len(text) > max_length:
        return "overlength_annotation"
    if any(char not in tokenizer._stoi for char in text):
        return "unsupported_character"
    return None


def recognition_objective(model, evidence, masks, matched, *, options, config):
    """Both OCR heads learn; gate targets come from actual refined predictions.

    Unmatched/unsupported ROIs remain gate examples. No annotation is stripped,
    lowercased or shortened for CE. All exclusions are returned for audit.
    """
    if not masks:
        zero = evidence["words"].sum() * 0
        return zero, {"light_ce": zero, "deep_ce": zero, "gate_bce": zero}, {"rois": 0, "ocr_supervised": 0, "gate_positive": 0, "excluded": []}
    light_features, geometries = model.align(evidence, masks, view=config["alignment_view"], options=options, deep=False)
    first, refined = model.light_head.light(model.light_head.memory(light_features))
    cues, readings = model.cues(evidence, masks, geometries, first, refined)
    gate_logits = model.gate(cues)
    targets = gate_logits.new_tensor([int(i in matched and bool(r["text"]) and r["text"].casefold() == matched[i]["text"].casefold())
                                     for i, r in enumerate(readings)])
    gate_bce = F.binary_cross_entropy_with_logits(gate_logits, targets)
    eligible, excluded = [], []
    for i, match in sorted(matched.items()):
        reason = label_support(match["text"], model.light_head.tokenizer, model.light_head.max_label_length)
        if reason:
            excluded.append({"roi_index": i, "text": match["text"], "reason": reason})
        else:
            eligible.append(i)
    light_ce, deep_ce = refined.sum() * 0, refined.sum() * 0
    if eligible:
        labels = [matched[i]["text"] for i in eligible]
        tokens = model.light_head.tokenizer.encode(labels, device=refined.device)
        target = tokens[:, 1:]
        length = target.shape[1]
        def ce(logits):
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=model.light_head.tokenizer.pad_id)
        light_ce = .5 * (ce(first[eligible, :length]) + ce(refined[eligible, :length]))
        deep_features, _ = model.align(evidence, [masks[i] for i in eligible], view=config["alignment_view"], options=options, deep=True)
        deep_ce = ce(model.deep_head.teacher(model.deep_head.memory(deep_features), tokens))
    terms = {"light_ce": light_ce, "deep_ce": deep_ce, "gate_bce": gate_bce}
    total = sum(config["lambda_" + name] * value for name, value in terms.items())
    return total, terms, {"rois": len(masks), "ocr_supervised": len(eligible),
        "gate_positive": int(targets.sum()), "excluded": excluded, "gate_targets": targets.detach().cpu().tolist(),
        "light_readings": readings, "alignment": [g.audit for g in geometries]}
