"""Explicit approximate ICDAR2015 word targets, paired geometry and validation."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.icdar2015_unet import PracticalICDAR2015AGUNet
from segmentation.data import file_sha256
from segmentation.scene_text_preparation import rasterize_instances
from segmentation.totaltext_image_frequency_augment import (
    PracticalTotalTextImageTransform, letterbox_canvas_mask,
)
from segmentation.totaltext_word_regions import make_word_regions, WordRegionAudit
from segmentation.transforms import SegmentationTransformConfig


def load_config(path: Path) -> dict:
    """Never resolve this practical configuration through STRICT/demo defaults."""
    if path.resolve().parent.name != "approximate":
        raise ValueError("ICDAR2015 transfer requires an explicit configs/approximate configuration")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "schema_version": 1, "dataset": "icdar2015", "source_sample_count": 1000,
        "architecture": PracticalICDAR2015AGUNet.PRACTICAL_IDENTITY,
        "image_size": [512, 512], "mask_size": [256, 256],
        "resize_geometry": "letterbox", "probability_threshold": 0.5,
        "encoder_weights": "IMAGENET1K_V2", "freeze_encoder": True,
        "checkpoint_selection": "validation_word_f1_then_fewer_fused_words_then_recall_then_iou_then_earliest",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Incompatible ICDAR2015 transfer configuration: {key}")
    for field in ("source_manifest_sha256", "encoder_checkpoint_sha256"):
        value = config.get(field, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing pinned SHA256: {field}")
    if not 0 < config["validation_count"] < config["source_sample_count"]:
        raise ValueError("A nonempty disjoint TRAIN validation subset is required")
    if config["training"]["epochs"] < 1 or config["training"]["num_workers"] < 0:
        raise ValueError("Invalid epoch/worker count")
    if config["skip_floor"] != 0.25 or config["training"]["separation_weight"] != 3.0:
        raise ValueError("Use the qualified residual attention / separator-loss transfer")
    localizer = config["diagnostic_localization"]
    if any(localizer.get(key) != value for key, value in {
            "connectivity": 8, "min_component_area": 0, "padding": 0, "max_blocks": None}.items()):
        raise ValueError("The diagnostic requires uncapped 8-connected proposals without padding")
    return config


def make_targets(size_wh, instances):
    """Keep quadrilaterals; transfer ownership/separators, never curved fitting.

    Ignore-only images remain valid negative examples. Unrepresentable words
    are recorded by the shared audit, not removed from evaluation denominators.
    """
    if any(not item.ignore for item in instances):
        return make_word_regions(size_wh, instances)
    target, validity = rasterize_instances(size_wh, instances)
    return (target, validity, Image.new("L", size_wh, 0),
            Image.new("I;16", size_wh, 0), WordRegionAudit(0, 0, 0, 0, 0, 1.0, 0))


def transform(config, *, augment):
    training = config["training"]
    return PracticalTotalTextImageTransform(
        SegmentationTransformConfig(
            image_size=tuple(config["image_size"]), mask_size=tuple(config["mask_size"]),
            horizontal_flip_probability=training["horizontal_flip_probability"] if augment else 0,
            rotation_probability=training["rotation_probability"] if augment else 0,
            rotation_degrees=tuple(training["rotation_degrees"]),
            image_interpolation="bilinear", mask_interpolation="bilinear",
            image_fill=(0, 0, 0), mask_fill=0, normalization="imagenet",
            resize_geometry="stretch", operation_order="flip_rotate_resize",
            classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION",
        ), base_seed=training["seed"], mode="letterbox", augment_images=False,
        parameters={"canvas_fill_rgb": config["canvas_fill_rgb"]},
    )


def split_indices(manifest, config):
    if (manifest.dataset != "icdar2015" or manifest.split_role != "train"
            or len(manifest.samples) != config["source_sample_count"]):
        raise ValueError("Only the complete official ICDAR2015 TRAIN population is permitted")
    samples = manifest.samples
    ordered = sorted(range(len(samples)), key=lambda i: (
        sha256(f"{config['training']['seed']}:{samples[i].sample_id}".encode()).digest(),
        samples[i].sample_id,
    ))
    held_out = set(ordered[:config["validation_count"]])
    train = [i for i in range(len(samples)) if i not in held_out]
    validation = [i for i in range(len(samples)) if i in held_out]
    for field in ("sample_id", "image_path", "image_sha256"):
        if {getattr(samples[i], field) for i in train} & {
                getattr(samples[i], field) for i in validation}:
            raise ValueError(f"Training/validation leakage in {field}")
    return train, validation


def validation_references(manifest, root, indices, paired_transform):
    references = {}
    for index in indices:
        sample = manifest.samples[index]
        if sample.word_owner_annotation_path is None:
            raise ValueError(f"Missing word ownership: {sample.sample_id}")
        with Image.open(root / sample.word_owner_annotation_path) as source:
            owners = source.convert("I")
            # Count all original care annotations, including words lost at source.
            expected = tuple(range(1, int(manifest.acquisition["care_words_by_id"][sample.sample_id]) + 1))
            blank = Image.new("RGB", owners.size)
            _, owners, _ = paired_transform._augment(blank, owners, None, index)
            owners = np.asarray(TF.resize(owners, list(paired_transform.config.mask_size),
                                interpolation=InterpolationMode.NEAREST), dtype=np.int32).copy()
        with Image.open(root / sample.image_path) as image:
            canvas = letterbox_canvas_mask(image.size).numpy()
        references[sample.sample_id] = (owners, expected, canvas)
    return references


def selection_key(metrics, epoch):
    return (metrics["proposal_f1_iou_50"], -metrics["gt_words_touched_by_fused_components"],
            metrics["word_recall_iou_50"], metrics["foreground_iou"], -epoch)


def evaluate(model, batches, references, *, device, probability_threshold, output_dir=None):
    """Native bbox IoU proxy, not official polygon Hmean or recognition accuracy.

    All 8-connected proposals count (including ignored-region predictions).
    Only the RGB-derived letterbox canvas clips predictions; GT never shapes
    predictions. Unrepresentable GT words stay in the denominator. Empty pixel
    unions score 1.0; empty proposal/word precision and recall score 0.0.
    """
    # Reuse the qualified Total-Text one-to-one matcher, not its fixed population.
    from scripts.train_totaltext_v4_image_frequency_ablation import proposal_metrics

    if not 0 <= probability_threshold <= 1:
        raise ValueError("Probability threshold must be explicitly supplied in [0,1]")
    totals, rows, seen = {}, [], set()
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch["image"].to(device))
            if tuple(logits.shape[1:]) != (1, 256, 256):
                raise ValueError("ICDAR2015 transfer must produce native single-channel 256 logits")
            probabilities = logits.sigmoid().cpu().numpy()[:, 0]
            for i, sample_id in enumerate(batch["sample_id"]):
                if sample_id in seen:
                    raise ValueError("Repeated validation image")
                seen.add(sample_id)
                owners, expected, canvas = references[sample_id]
                target = batch["mask"][i, 0].numpy().astype(bool)
                valid = batch["validity"][i, 0].numpy().astype(bool)
                prediction = (probabilities[i] >= probability_threshold) & canvas
                owners = owners.copy()
                owners[~target | ~valid] = 0
                metrics = proposal_metrics(prediction, owners, expected_owner_ids=expected)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0) + value
                intersection = int((prediction & target & valid).sum())
                union = int(((prediction | target) & valid).sum())
                row = {"sample_id": sample_id, **metrics, "intersection": intersection,
                       "union": union, "foreground_iou": intersection / union if union else 1.0}
                rows.append(row)
                if output_dir is not None:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = output_dir / (sample_id.replace(":", "_") + ".npz")
                    np.savez_compressed(mask_path,
                        probability=probabilities[i], prediction=prediction,
                        target=target, validity=valid, owners=owners,
                        expected_owner_ids=np.asarray(expected, dtype=np.int32))
                    row["intermediate_file"] = mask_path.name
                    row["intermediate_sha256"] = file_sha256(mask_path)
    if seen != set(references) or not rows:
        raise ValueError("Validation population is incomplete")
    tp, proposals, words = totals["matched_gt_iou_ge_50"], totals["raw_proposals"], totals["gt_words"]
    aggregate = {**totals, "foreground_iou": sum(r["foreground_iou"] for r in rows) / len(rows),
        "proposal_precision_iou_50": tp / proposals if proposals else 0.0,
        "word_recall_iou_50": tp / words if words else 0.0,
        "proposal_f1_iou_50": 2 * tp / (proposals + words) if proposals + words else 0.0}
    return {"aggregate": aggregate, "per_image": rows}
