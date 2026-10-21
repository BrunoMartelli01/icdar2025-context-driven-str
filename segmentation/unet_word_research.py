"""Paired glyph/word supervision and declared experimental auxiliary losses."""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from segmentation.data import SegmentationDataset
from segmentation.totaltext_word_kernel import WordKernelDataset
from segmentation.training import SegmentationTrainingConfig
from segmentation.word_multitask import masked_bce_dice


class GlyphWordDataset(WordKernelDataset):
    """Original glyph pixels + modified word regions, with identical geometry.

    The original binary GT has no character instance IDs. Do not fabricate them
    from connected components, where a character can be disconnected or touching.
    """

    def __init__(self, base, glyph_manifest, glyph_root, *, distance_fraction):
        super().__init__(base, distance_fraction=distance_fraction)
        source = {s.sample_id: s for s in glyph_manifest.samples}
        if len(source) != len(glyph_manifest.samples) or set(source) != {s.sample_id for s in base.manifest.samples}:
            raise ValueError("Glyph and word manifests must contain the same unique TRAIN IDs")
        ordered = []
        for sample in base.manifest.samples:
            glyph = source[sample.sample_id]
            if sample.image_sha256 != glyph.image_sha256:
                raise ValueError("Glyph/word RGB checksum mismatch: " + sample.sample_id)
            ordered.append(glyph)
        paired = replace(glyph_manifest, samples=tuple(ordered))
        self.glyph_base = SegmentationDataset(paired, glyph_root, transform=base.transform)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        glyph = self.glyph_base[index]
        if glyph["sample_id"] != batch["sample_id"] or not torch.equal(glyph["image"], batch["image"]):
            raise ValueError("Original and modified targets did not replay identical RGB augmentation")
        batch["glyph_mask"] = glyph["mask"]
        batch["glyph_validity"] = glyph["validity"] * batch["validity"]
        from PIL import Image
        from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
        with Image.open(batch["image_path"]) as image:
            batch["canvas"] = letterbox_canvas_mask(image.size, native_size=batch["mask"].shape[-1])
        sample = self.base.manifest.samples[index]
        with Image.open(self.base.data_root / sample.word_owner_annotation_path) as owner:
            values = np.asarray(owner)
            batch["expected_word_count"] = int(len(np.unique(values[values > 0])))
        return batch


@dataclass(frozen=True)
class WordResearchTrainingConfig(SegmentationTrainingConfig):
    loss: str = "experimental_word_research_v1"

    def __post_init__(self):
        if self.loss != "experimental_word_research_v1" or not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid experimental loss/learning rate")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Positive batch size required")
        if self.checkpoint_selection not in ("validation_word_f1", "validation_quality_loss"):
            raise ValueError("Unsupported experimental checkpoint criterion")
        values = asdict(self)
        values.update(loss="masked_BCEWithLogitsLoss", learning_rate=1e-4, batch_size=8,
                      checkpoint_selection="validation_word_f1")
        # Check shared invariants, while preserving actual declared experiment
        # LR/batch/criterion in this dataclass and checkpoint metadata.
        SegmentationTrainingConfig(**values)


def instance_embedding_loss(embeddings, owners, kernels, validity, *, pull_margin, push_margin):
    """Equal word means, then image means; pull to own core, push other cores."""
    if (embeddings.ndim != 4 or owners.shape != (len(embeddings), *embeddings.shape[-2:])
            or kernels.shape != (len(embeddings), 1, *embeddings.shape[-2:]) or validity.shape != kernels.shape
            or owners.dtype not in (torch.int32, torch.int64) or torch.any(owners < 0)
            or not torch.isfinite(embeddings).all() or torch.any((validity != 0) & (validity != 1))
            or torch.any((kernels != 0) & (kernels != 1)) or torch.any(kernels > validity)
            or not np.isfinite([pull_margin, push_margin]).all() or not 0 < pull_margin < push_margin):
        raise ValueError("Invalid embedding supervision shapes/maps/margins")
    images = []
    for i in range(len(embeddings)):
        active = (owners[i] > 0) & (validity[i, 0] > 0)
        means, pull = [], []
        for word in torch.unique(owners[i][active]):
            support = active & (owners[i] == word)
            core = support & (kernels[i, 0] > 0)
            if not core.any():
                raise ValueError("Supervised word has no kernel")
            mean = embeddings[i, :, core].mean(dim=1)
            means.append(mean)
            distance = torch.linalg.vector_norm(embeddings[i, :, support] - mean[:, None], dim=0)
            pull.append(torch.log1p(F.relu(distance - pull_margin).square()).mean())
        if not means:
            continue
        mean = torch.stack(means)
        if len(mean) > 1:
            distance = torch.linalg.vector_norm(mean[:, None] - mean[None, :], dim=-1)
            pairs = ~torch.eye(len(mean), dtype=torch.bool, device=mean.device)
            push = torch.log1p(F.relu(push_margin - distance[pairs]).square()).mean()
        else:
            push = embeddings[i].sum() * 0
        images.append(torch.stack(pull).mean() + push)
    return torch.stack(images).mean() if images else embeddings.sum() * 0


def adaptive_threshold_targets(owners, validity, *, border_width, threshold_range):
    """DB-inspired border target; isolated adaptation to full-word probability.

    Low threshold in interiors; higher at owner boundaries. Different owners
    are treated individually. GT is used solely to create TRAIN loss targets.
    """
    low, high = threshold_range
    if (owners.ndim != 3 or validity.shape != (len(owners), 1, *owners.shape[-2:])
            or type(border_width) is not int or border_width < 1 or not 0 < low < high < 1):
        raise ValueError("Invalid adaptive threshold target geometry/settings")
    arrays = owners.detach().cpu().numpy()
    targets = np.full((len(arrays), 1, *arrays.shape[-2:]), low, np.float32)
    weights = np.zeros_like(targets)
    for i, image in enumerate(arrays):
        for word in np.unique(image[image > 0]):
            support = image == word
            boundary = support & ~(cv2.erode(support.astype(np.uint8), np.ones((3, 3), np.uint8),
                                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
            distance = distance_transform_edt(~boundary)
            target = low + (high - low) * np.maximum(0, 1 - distance / border_width)
            targets[i, 0] = np.maximum(targets[i, 0], target)
            weights[i, 0] = np.maximum(weights[i, 0], support | (distance <= border_width))
    return (torch.as_tensor(targets, dtype=validity.dtype, device=validity.device),
            torch.as_tensor(weights, dtype=validity.dtype, device=validity.device) * validity)


def _bbox(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def mask_quality_targets(labels, owners, validity):
    """Post-prediction TRAIN labels: best word bbox IoU, coverage and purity.

    Return valid instance IDs; wholly ignored predictions receive no objective.
    Their inference scores/outputs are still preserved by the evaluation runner.
    """
    from scripts.train_totaltext_v4_highres_ablation import _box_iou
    labels, owners, validity = map(np.asarray, (labels, owners, validity))
    if (labels.shape != owners.shape or validity.shape != labels.shape or labels.ndim != 2
            or labels.dtype.kind not in "iu" or owners.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(owners < 0) or np.any((validity != 0) & (validity != 1))):
        raise ValueError("Invalid mask quality supervision")
    words = {int(i): (owners == i) & (validity > 0) for i in np.unique(owners[owners > 0])}
    words = {i: mask for i, mask in words.items() if mask.any()}
    boxes = {i: _bbox(mask) for i, mask in words.items()}
    ids, targets = [], []
    for instance in np.unique(labels[labels > 0]):
        support = labels == instance
        valid_support = support & (validity > 0)
        if not valid_support.any():
            continue
        box = _bbox(support)
        ious = {i: _box_iou(box, gt) for i, gt in boxes.items()}
        best = max(ious, key=lambda i: (ious[i], -i)) if ious else None
        intersection = int((support & words[best]).sum()) if best is not None else 0
        ids.append(int(instance))
        targets.append([ious[best] if best is not None else 0.,
                        intersection / int(words[best].sum()) if best is not None else 0.,
                        intersection / int(valid_support.sum())])
    return ids, np.asarray(targets, np.float32).reshape(-1, 3)


def research_loss(model, evidence, batch, config, labels=None):
    if model.branch == "quality":
        if labels is None or len(labels) != len(batch["image"]):
            raise ValueError("Quality training requires GT-blind predicted labels")
        terms = []
        for i, predicted in enumerate(labels):
            ids, raw = model.quality_scores(evidence["features"][i], predicted, **config["quality"])
            target_ids, targets = mask_quality_targets(predicted, batch["owners_native"][i].cpu().numpy(),
                                                      batch["validity"][i, 0].cpu().numpy())
            if target_ids:
                positions = [ids.index(instance) for instance in target_ids]
                target = torch.as_tensor(targets, device=raw.device, dtype=raw.dtype)
                terms.append(F.mse_loss(raw[positions].sigmoid(), target))
        result = torch.stack(terms).mean() if terms else sum(p.sum() * 0 for p in model.quality.parameters())
        return result, {"quality": result}, bool(terms)
    options = {k: config["loss"][k] for k in ("bce_weight", "dice_weight", "smooth")}
    valid = batch["validity"]
    terms = {"word": masked_bce_dice(evidence["words"][:, :1], batch["mask"], valid, **options,
             pixel_weight=1 + (config["loss"]["separation_weight"] - 1) * batch["separation"]),
             "kernel": masked_bce_dice(evidence["words"][:, 1:2], batch["kernel"], valid, **options)}
    if evidence["glyph"] is not None:
        terms["glyph"] = masked_bce_dice(evidence["glyph"], batch["glyph_mask"], batch["glyph_validity"], **options)
    if evidence["embedding"] is not None:
        terms["embedding"] = instance_embedding_loss(evidence["embedding"], batch["owners_native"], batch["kernel"], valid,
            **{k: config["embedding"][k] for k in ("pull_margin", "push_margin")})
    if evidence["threshold"] is not None:
        target, weight = adaptive_threshold_targets(batch["owners_native"], valid,
            **{k: config["adaptive"][k] for k in ("border_width", "threshold_range")})
        low, high = config["adaptive"]["threshold_range"]
        threshold = low + (high - low) * evidence["threshold"].sigmoid()
        terms["threshold"] = ((threshold - target).abs() * weight).sum() / weight.sum().clamp_min(1)
        soft_binary = config["adaptive"]["sigmoid_steepness"] * (evidence["words"][:, :1].sigmoid() - threshold)
        terms["binary"] = masked_bce_dice(soft_binary, batch["mask"], valid, **options)
    total = sum(config["loss"]["lambda_" + name] * value for name, value in terms.items())
    return total, terms, bool(valid.sum() > 0)


def train_research_epoch(model, batches, optimizer, device, config):
    from localize.unet_word_research import research_instances
    model.train()
    totals, ids, steps, skipped = {}, [], 0, 0
    for batch in batches:
        local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        evidence = model.forward_evidence(local["image"])
        labels = ([research_instances(evidence, batch["canvas"][i].numpy(), config, i)[0]
                   for i in range(len(batch["image"]))] if model.branch == "quality" else None)
        loss, terms, supervised = research_loss(model, evidence, local, config, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite experimental training objective")
        optimizer.zero_grad(set_to_none=True)
        if supervised:
            loss.backward()
            optimizer.step()
        else:
            skipped += 1
        steps += 1
        ids.extend(batch["sample_id"])
        for name, value in {"loss": loss, **terms}.items():
            totals[name] = totals.get(name, 0.) + float(value.detach()) * len(batch["image"])
    if not ids:
        raise ValueError("Empty training loader")
    return {**{k: v / len(ids) for k, v in totals.items()}, "samples": len(ids), "steps": steps,
            "skipped_unsupervised_batches": skipped,
            "sample_order_sha256": sha256("".join(s + "\n" for s in ids).encode()).hexdigest()}
