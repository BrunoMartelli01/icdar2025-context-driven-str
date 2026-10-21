"""Isolated recall/resolution experiments; original scientific paths stay frozen."""

from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from segmentation.transforms import SegmentationTransformConfig
from segmentation.training import SegmentationTrainingConfig
from segmentation.totaltext_image_frequency_augment import PracticalTotalTextImageTransform
from segmentation.totaltext_word_kernel import native_owner_map
from segmentation.word_multitask import masked_bce_dice


@dataclass(frozen=True)
class TextRecallTransformConfig(SegmentationTransformConfig):
    """Opt into 768/384 without relaxing the original transform's size checks."""

    def __post_init__(self):
        if (self.classification != "EXPERIMENTAL_TEXT_RECALL_RESOLUTION"
            or (self.image_size, self.mask_size) not in
                (((512, 512), (256, 256)), ((768, 768), (384, 384)))):
            raise ValueError("Recall transform explicitly supports 512/256 or 768/384")
        common = asdict(self)
        common.update(image_size=(512, 512), mask_size=(256, 256),
                      classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION")
        # Validate all inherited interpolation/augmentation policies with the
        # original validator. The size exception belongs only to this subclass.
        SegmentationTransformConfig(**common)


def make_transform(source, *, input_size, train):
    from scripts.train_totaltext_v4_image_frequency_ablation import _transform
    existing = _transform(source, "letterbox", train=train)
    values = asdict(existing.config)
    values.update(image_size=(input_size, input_size), mask_size=(input_size//2, input_size//2),
                  classification="EXPERIMENTAL_TEXT_RECALL_RESOLUTION")
    return PracticalTotalTextImageTransform(TextRecallTransformConfig(**values),
        base_seed=existing.base_seed, mode=existing.mode, augment_images=train,
        parameters=dict(existing.parameters))


class TextRecallDataset(Dataset):
    """Reuse full-resolution v4 targets and paired owner mapping, without kernels."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = dict(self.base[index])
        sample = self.base.manifest.samples[index]
        if sample.word_owner_annotation_path is None:
            raise ValueError("Word-owner annotations are required for coverage auditing")
        with Image.open(self.base.data_root/sample.word_owner_annotation_path) as opened:
            image = opened.convert("I")
            expected = len(np.unique(np.asarray(image)[np.asarray(image)>0]))
            owners = native_owner_map(image, self.base.transform, index, batch["mask"], batch["validity"])
        batch.update(owners_native=torch.from_numpy(owners), expected_words=expected)
        return batch


@dataclass(frozen=True)
class TextRecallTrainingConfig(SegmentationTrainingConfig):
    loss: str = "masked_BCE_Dice"

    def __post_init__(self):
        if self.loss not in ("masked_BCE_Dice", "masked_BCE_Tversky"):
            raise ValueError("Unsupported recall loss")
        if (type(self.batch_size) is not int or self.batch_size < 1
            or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
            or self.checkpoint_selection != "validation_word_coverage_at_precision_floor"):
            raise ValueError("Invalid recall training/batch/selection settings")
        common = asdict(self)
        common.update(loss="masked_BCEWithLogitsLoss", learning_rate=1e-4, batch_size=8,
                      checkpoint_selection="last")
        SegmentationTrainingConfig(**common)


def masked_tversky(logits, target, valid, *, fp_weight, fn_weight, smooth):
    """Per-valid-image soft Tversky; ignore pixels have exactly zero gradient.

    Denominator = TP + fp_weight*FP + fn_weight*FN + smooth.
    For an empty valid target, false foreground is still penalized. An entirely
    ignored batch returns differentiable zero and is skipped by the trainer.
    """
    from segmentation.word_multitask import _maps
    _maps(logits, target, valid)
    if torch.any((target != 0)&(target != 1)):
        raise ValueError("Tversky target must be binary")
    if (not np.isfinite([fp_weight, fn_weight, smooth]).all()
        or min(fp_weight, fn_weight, smooth) <= 0
        or not np.isclose(fp_weight+fn_weight, 1)):
        raise ValueError("Positive FP/FN weights summing to one and positive smooth required")
    selected = valid.flatten(1).sum(1)>0
    if not selected.any():
        return logits.sum()*0
    p = logits.sigmoid()
    tp = (p*target*valid).flatten(1).sum(1)
    fp = (p*(1-target)*valid).flatten(1).sum(1)
    fn = ((1-p)*target*valid).flatten(1).sum(1)
    value = 1-(tp+smooth)/(tp+fp_weight*fp+fn_weight*fn+smooth)
    return value[selected].mean()


def recall_loss(logits, batch, *, loss_name, config):
    target, valid, separator = (batch[k] for k in ("mask", "validity", "separation"))
    if (separator.shape != target.shape or torch.any((separator!=0)&(separator!=1))
        or torch.any((separator>0)&((target>0)|(valid==0)))):
        raise ValueError("Separator must contain only valid background pixels")
    weights = 1+(config["separation_weight"]-1)*separator
    if loss_name == "bce_dice":
        return masked_bce_dice(logits,target,valid,bce_weight=config["bce_weight"],
            dice_weight=config["overlap_weight"],smooth=config["smooth"],pixel_weight=weights)
    if loss_name != "bce_tversky":
        raise ValueError("Unknown recall loss")
    bce = masked_bce_dice(logits,target,valid,bce_weight=config["bce_weight"],
                         dice_weight=0,smooth=config["smooth"],pixel_weight=weights)
    overlap = masked_tversky(logits,target,valid,fp_weight=config["fp_weight"],
                            fn_weight=config["fn_weight"],smooth=config["smooth"])
    return bce+config["overlap_weight"]*overlap


def train_recall_epoch(model, batches, optimizer, device, *, loss_name, loss_config):
    model.train()
    total, samples, steps, empty, ordered = 0., 0, 0, 0, []
    shape = None
    for batch in batches:
        local = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(local["image"])
        if logits.device != local["image"].device:
            raise RuntimeError("Recall logits left the requested device")
        loss = recall_loss(logits,local,loss_name=loss_name,config=loss_config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite recall loss")
        if local["validity"].any():
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite recall gradient")
            optimizer.step()
            steps += 1
        else:
            empty += 1
        count = len(batch["sample_id"])
        total += float(loss.detach())*count
        samples += count
        ordered.extend(batch["sample_id"])
        shape = list(local["image"].shape)
    if not samples:
        raise ValueError("Empty training loader")
    return {"loss":total/samples,"samples":samples,"steps":steps,
            "empty_batches_without_update":empty,"last_input_shape":shape,
            "sample_order_sha256":sha256("".join(x+"\n" for x in ordered).encode()).hexdigest()}


def canonical_probability(native_probability, *, size):
    """Sample sigmoid probabilities onto the fixed metric/offset-network grid."""
    values = np.asarray(native_probability,dtype=np.float32)
    if (values.ndim != 2 or not np.isfinite(values).all()
        or np.any((values<0)|(values>1)) or type(size) is not int or size<1):
        raise ValueError("Expected finite probability map and positive metric size")
    if values.shape == (size,size):
        return values.copy()
    return F.interpolate(torch.from_numpy(values)[None,None],size=(size,size),
                         mode="bilinear",align_corners=False)[0,0].numpy()


def coverage_counts(probability, reference, threshold, *, complete_coverage):
    """Count every expected word, including words absent from the metric raster."""
    target, valid, owners, expected, canvas = (reference[k] for k in
        ("target","validity","owners","expected_owner_ids","canvas"))
    if (probability.shape != target.shape or valid.shape != target.shape
        or owners.shape != target.shape or canvas.shape != target.shape):
        raise ValueError("Canonical coverage geometry differs")
    if not 0<threshold<1 or not 0<complete_coverage<=1:
        raise ValueError("Invalid probability or coverage threshold")
    valid = valid&canvas
    prediction = (probability>=threshold)&canvas
    tp = int((prediction&target&valid).sum())
    fp = int((prediction&~target&valid).sum())
    fn = int((~prediction&target&valid).sum())
    word_rows=[]
    for owner in expected:
        support=(owners==owner)&valid&target
        count=int(support.sum());covered=int((prediction&support).sum())
        word_rows.append({"owner_id":int(owner),"gt_pixels":count,"covered_pixels":covered,
                          "coverage":covered/count if count else 0.})
    union=tp+fp+fn
    return {"tp":tp,"fp":fp,"fn":fn,"gt_words":len(expected),
        "lost_words":sum(r["gt_pixels"]==0 for r in word_rows),
        "complete_words":sum(r["coverage"]>=complete_coverage for r in word_rows),
        "coverage_sum":sum(r["coverage"] for r in word_rows),
        "foreground_iou":tp/union if union else 1.,"words":word_rows}


def aggregate_coverage(rows):
    if not rows:
        raise ValueError("No validation rows")
    totals={k:sum(r[k] for r in rows) for k in
        ("tp","fp","fn","gt_words","lost_words","complete_words","coverage_sum")}
    tp,fp,fn=totals["tp"],totals["fp"],totals["fn"]
    words=totals["gt_words"]
    return {**totals,"images":len(rows),"pixel_precision":tp/(tp+fp) if tp+fp else float(tp+fn==0),
        "pixel_recall":tp/(tp+fn) if tp+fn else 1.,
        "pixel_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        "pixel_f2":5*tp/(5*tp+fp+4*fn) if 5*tp+fp+4*fn else 1.,
        "foreground_micro_iou":tp/(tp+fp+fn) if tp+fp+fn else 1.,
        "foreground_macro_iou":sum(r["foreground_iou"] for r in rows)/len(rows),
        "mean_word_coverage":totals["coverage_sum"]/words if words else 0.,
        "complete_word_rate":totals["complete_words"]/words if words else 0.}


def operating_point(aggregates, *, minimum_precision):
    def key(item):
        threshold,metrics=item
        qualified=metrics["pixel_precision"]>=minimum_precision and metrics["tp"]>0
        if qualified:
            score=(metrics["complete_word_rate"],metrics["mean_word_coverage"],
                   metrics["pixel_f2"],metrics["foreground_micro_iou"],-float(threshold))
        else:
            score=(metrics["pixel_precision"],metrics["complete_word_rate"],
                   metrics["mean_word_coverage"],metrics["pixel_f2"],-float(threshold))
        return (qualified,*score)
    threshold,metrics=max(aggregates.items(),key=key)
    return {"threshold":float(threshold),"precision_requirement_met":bool(key((threshold,metrics))[0]),**metrics}


def selection_key(point, epoch):
    if point["precision_requirement_met"]:
        score=(point["complete_word_rate"],point["mean_word_coverage"],point["pixel_f2"],
               point["foreground_micro_iou"])
    else:
        score=(point["pixel_precision"],point["complete_word_rate"],point["mean_word_coverage"],point["pixel_f2"])
    return (point["precision_requirement_met"],*score,-epoch)
