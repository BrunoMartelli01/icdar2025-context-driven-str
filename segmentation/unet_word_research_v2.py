"""Declared corrective objectives; original glyph/word datasets stay immutable."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from localize.unet_word_research_v2 import NEIGHBORS
from segmentation.training import SegmentationTrainingConfig
from segmentation.unet_word_research import GlyphWordDataset, adaptive_threshold_targets
from segmentation.word_multitask import masked_bce_dice


@dataclass(frozen=True)
class WordResearchV2TrainingConfig(SegmentationTrainingConfig):
    loss: str = "experimental_word_research_v2"

    def __post_init__(self):
        if (self.loss != "experimental_word_research_v2" or not np.isfinite(self.learning_rate)
                or self.learning_rate <= 0 or type(self.batch_size) is not int or self.batch_size < 1
                or self.checkpoint_selection not in ("validation_word_f1", "validation_glyph_f1")):
            raise ValueError("Invalid corrective training configuration")
        values = asdict(self)
        values.update(loss="masked_BCEWithLogitsLoss", learning_rate=1e-4, batch_size=8,
                      checkpoint_selection="validation_word_f1")
        SegmentationTrainingConfig(**values)


def neighbor_slices(height, width, dy, dx):
    return (slice(max(0, -dy), min(height, height - dy)),
            slice(max(0, -dx), min(width, width - dx)),
            slice(max(0, dy), min(height, height + dy)),
            slice(max(0, dx), min(width, width + dx)))


def link_targets(owners, validity):
    if (owners.ndim != 3 or validity.shape != (len(owners), 1, *owners.shape[-2:])
            or owners.dtype not in (torch.int32, torch.int64) or torch.any(owners < 0)
            or torch.any((validity != 0) & (validity != 1))):
        raise ValueError("Invalid pixel-link supervision")
    h, w = owners.shape[-2:]
    target = validity.new_zeros((len(owners), 8, h, w))
    weight = target.clone()
    inverse_areas = validity[:, 0].new_zeros(owners.shape)
    for i in range(len(owners)):
        for owner in torch.unique(owners[i][(owners[i] > 0) & (validity[i, 0] > 0)]):
            support = (owners[i] == owner) & (validity[i, 0] > 0)
            inverse_areas[i][support] = 1 / support.sum().to(weight.dtype)
    for direction, (dy, dx) in enumerate(NEIGHBORS):
        y, x, ny, nx = neighbor_slices(h, w, dy, dx)
        source, neighbor = owners[:, y, x], owners[:, ny, nx]
        active = (source > 0) & (validity[:, 0, y, x] > 0) & (validity[:, 0, ny, nx] > 0)
        target[:, direction, y, x] = ((source == neighbor) & active).to(target.dtype)
        # Equal word contribution before positive/negative class balancing.
        weight[:, direction, y, x] = active * inverse_areas[:, y, x]
    return target, weight


def balanced_link_loss(logits, targets, weights):
    if (logits.shape != targets.shape or weights.shape != targets.shape or logits.shape[1] != 8
            or torch.any(weights < 0) or not torch.isfinite(weights).all()):
        raise ValueError("Invalid link objective shapes/weights")
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    images = []
    for i in range(len(logits)):
        terms = []
        for positive in (True, False):
            weight = weights[i] * ((targets[i] > 0) == positive)
            if weight.sum() > 0:
                terms.append((raw[i] * weight).sum() / weight.sum())
        if terms:
            images.append(torch.stack(terms).sum())
    return torch.stack(images).mean() if images else logits.sum() * 0


def db_raster_targets(owners, validity, *, shrink_ratio, threshold_range):
    """DB-inspired A(1-r²)/L shrink, with raster EDT instead of polygon clipping.

    A missing native core retains its farthest pixel, explicitly counted. Original
    full-word owners/GT are never overwritten; all words remain in evaluation.
    """
    low, high = threshold_range
    if (owners.ndim != 3 or validity.shape != (len(owners), 1, *owners.shape[-2:])
            or owners.dtype not in (torch.int32, torch.int64) or torch.any(owners < 0)
            or not 0 < shrink_ratio < 1 or not 0 < low < high < 1):
        raise ValueError("Invalid raster DB target configuration")
    array = owners.detach().cpu().numpy()
    core = np.zeros((len(array), 1, *array.shape[-2:]), np.float32)
    threshold = np.full_like(core, low); weight = np.zeros_like(core)
    fallback = 0
    for i, image in enumerate(array):
        for owner in np.unique(image[image > 0]):
            support = image == owner
            padded = np.pad(support, 1)
            perimeter = sum(np.count_nonzero(support & ~padded[1 + dy:1 + dy + image.shape[0],
                1 + dx:1 + dx + image.shape[1]]) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
            offset = support.sum() * (1 - shrink_ratio ** 2) / perimeter
            inside = distance_transform_edt(padded)[1:-1, 1:-1]
            shrunk = support & (inside > offset)
            if not shrunk.any():
                shrunk.flat[int(np.argmax(inside))] = True; fallback += 1
            core[i, 0] = np.maximum(core[i, 0], shrunk)
            boundary = support & ((inside <= 1) | (inside == inside.min()))
            distance = distance_transform_edt(~boundary)
            expanded = support | (distance_transform_edt(~support) <= offset)
            value = low + (high - low) * np.maximum(0, 1 - distance / offset)
            threshold[i, 0] = np.maximum(threshold[i, 0], value)
            weight[i, 0] = np.maximum(weight[i, 0], expanded)
    def tensor(value):
        return torch.as_tensor(value, dtype=validity.dtype, device=validity.device)
    return tensor(core) * validity, tensor(threshold), tensor(weight) * validity, fallback


def glyph_weight(config, epoch):
    warmup, ramp = config["glyph"]["warmup_epochs"], config["glyph"]["ramp_epochs"]
    return config["glyph"]["joint_weight"] * min(1., max(0., (epoch - warmup) / ramp))


def research_v2_loss(model, evidence, batch, config):
    options = {k: config["loss"][k] for k in ("bce_weight", "dice_weight", "smooth")}
    valid = batch["validity"]
    zero = evidence["words"].sum() * 0
    terms = {}
    if model.shared_trainable:
        terms["word"] = masked_bce_dice(evidence["words"][:, :1], batch["mask"], valid, **options,
            pixel_weight=1 + (config["loss"]["separation_weight"] - 1) * batch["separation"])
        terms["kernel"] = masked_bce_dice(evidence["words"][:, 1:2], batch["kernel"], valid, **options)
    total = sum(terms.values(), zero)
    if evidence["glyph"] is not None:
        terms["glyph"] = masked_bce_dice(evidence["glyph"], batch["glyph_mask"], batch["glyph_validity"], **options)
        coefficient = glyph_weight(config, model.research_epoch) if model.shared_trainable else 1.
        total = total + coefficient * terms["glyph"]
    if evidence["links"] is not None:
        target, weights = link_targets(batch["owners_native"], valid)
        terms["links"] = balanced_link_loss(evidence["links"], target, weights)
        total = total + terms["links"]
    if evidence["threshold"] is not None:
        target, weight = adaptive_threshold_targets(batch["owners_native"], valid,
            **{k: config["adaptive"][k] for k in ("border_width", "threshold_range")})
        low, high = config["adaptive"]["threshold_range"]
        threshold = low + (high - low) * evidence["threshold"].sigmoid()
        terms["threshold"] = ((threshold - target).abs() * weight).sum() / weight.sum().clamp_min(1)
        binary = config["adaptive"]["sigmoid_steepness"] * (evidence["words"][:, :1].sigmoid() - threshold)
        terms["binary"] = masked_bce_dice(binary, batch["mask"], valid, **options)
        total = total + terms["threshold"] + terms["binary"]
    if evidence["db_probability"] is not None:
        core, target, weight, fallback = db_raster_targets(batch["owners_native"], valid,
            **{k: config["db"][k] for k in ("shrink_ratio", "threshold_range")})
        low, high = config["db"]["threshold_range"]
        threshold = low + (high - low) * evidence["db_threshold"].sigmoid()
        terms["db_probability"] = masked_bce_dice(evidence["db_probability"], core, valid, **options)
        binary = config["db"]["sigmoid_steepness"] * (evidence["db_probability"].sigmoid() - threshold)
        terms["db_binary"] = masked_bce_dice(binary, core, valid, **options)
        terms["db_threshold"] = ((threshold - target).abs() * weight).sum() / weight.sum().clamp_min(1)
        total = total + terms["db_probability"] + terms["db_binary"] + config["db"]["threshold_weight"] * terms["db_threshold"]
    supervised = bool(batch["glyph_validity"].sum() > 0) if evidence["glyph"] is not None and not model.shared_trainable else bool(valid.sum() > 0)
    if evidence["links"] is not None:
        supervised = bool(weights.sum() > 0)
    return total, terms, supervised


def gradient_diagnostics(model, terms, config):
    """Measure shared-decoder conflicts; loss magnitudes alone do not prove them."""
    named = dict(model.named_parameters())
    if model.shared_trainable and "glyph" in terms and any(k not in named for k in config["glyph"]["gradient_parameters"]):
        raise ValueError("Declared shared gradient parameter is missing")
    parameters = [p for name, p in named.items() if name in config["glyph"]["gradient_parameters"] and p.requires_grad]
    if not parameters or "glyph" not in terms or "word" not in terms:
        return {}
    word = torch.autograd.grad(terms["word"] + terms["kernel"], parameters, retain_graph=True)
    glyph = torch.autograd.grad(terms["glyph"], parameters, retain_graph=True)
    dot = sum((a * b).sum() for a, b in zip(word, glyph))
    wn = sum(a.square().sum() for a in word).sqrt(); gn = sum(a.square().sum() for a in glyph).sqrt()
    return {"word_gradient_norm": float(wn.detach()), "glyph_gradient_norm": float(gn.detach()),
            "gradient_cosine": float((dot / (wn * gn).clamp_min(1e-12)).detach())}


def train_research_v2_epoch(model, batches, optimizer, device, config):
    model.train(); totals, ids, steps, skipped, diagnostics = {}, [], 0, 0, {}
    for batch in batches:
        local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        evidence = model.forward_evidence(local["image"])
        loss, terms, supervised = research_v2_loss(model, evidence, local, config)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite corrective objective")
        optimizer.zero_grad(set_to_none=True)
        if supervised:
            if steps == 0:
                diagnostics = gradient_diagnostics(model, terms, config)
            loss.backward(); optimizer.step()
        else:
            skipped += 1
        steps += 1; ids.extend(batch["sample_id"])
        for name, value in {"loss": loss, **terms}.items():
            totals[name] = totals.get(name, 0.) + float(value.detach()) * len(batch["image"])
    if not ids:
        raise ValueError("Empty training loader")
    return {**{k: v / len(ids) for k, v in totals.items()}, **diagnostics,
            "samples": len(ids), "steps": steps, "skipped_unsupervised_batches": skipped,
            "glyph_coefficient": glyph_weight(config, model.research_epoch) if model.branch == "glyph_staged" else None,
            "sample_order_sha256": sha256("".join(s + "\n" for s in ids).encode()).hexdigest()}


def build_research_v2_optimizer(model, config):
    # Include future trainable parameters from the start so warmup resumes safely.
    shared = [p for name, p in model.named_parameters() if model.original_trainability.get(name, False)]
    private = [p for name, p in model.named_parameters() if name not in model.original_trainability]
    groups = []
    if shared:
        groups.append({"params": shared, "lr": config["learning_rate"], "group_role": "shared"})
    if private:
        groups.append({"params": private, "lr": config["head_learning_rate"], "group_role": "private"})
    return torch.optim.Adam(groups, weight_decay=config["weight_decay"])
