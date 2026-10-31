"""Shared provenance/validation for the isolated approximate improvement suite."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from models.icdar2015_improvements import ImprovementAGUNet
from recognition.scene_text_datasets import parse_icdar2015_annotation
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.icdar2015_improvements import (
    binary_proposals, instance_balanced_loss, original_geometry, owner_topology, valid_canvas,
)
from segmentation.icdar2015_transfer import load_config as old_config, split_indices
from segmentation.training import SegmentationTrainingConfig, separation_weighted_bce_with_logits


REPO = Path(__file__).resolve().parents[1]


def source_hashes(*entry_points):
    from scripts.train_icdar2015_totaltext_transfer import SOURCES
    names=set(SOURCES) | {"models/icdar2015_improvements.py",
        "segmentation/icdar2015_improvements.py", "segmentation/icdar2015_improvement_runtime.py",
        "segmentation/totaltext_word_kernel.py", *entry_points}
    return {name:file_sha256(REPO/name) for name in sorted(names)}


def load_config(path):
    if path.resolve().parent.name != "approximate":
        raise ValueError("Explicit configs/approximate suite profile required")
    config = json.loads(path.read_text())
    if config["reproduction_mode"] != "approximate" or config["claim_label"] != "approximate reproduction":
        raise ValueError("Only approximate reproduction is supported")
    expected = {"control512", "larger_bce", "larger_instance", "larger_instance_gn"}
    if set(config["branches"]) != expected:
        raise ValueError("Unexpected predeclared branch family")
    if config["training"]["precision"] != "float32" or config["training"]["include_initial_state_as_epoch_zero"] is not True:
        raise ValueError("FP32 and source epoch-zero control required")
    source = REPO / config["source_training_config"]
    if file_sha256(source) != config["source_training_config_sha256"]:
        raise ValueError("Source segmentation training configuration changed")
    return config


def load_training_population(config, root):
    path = root / config["datasets"]["icdar2015"]["manifest"]
    if file_sha256(path) != config["datasets"]["icdar2015"]["manifest_sha256"]:
        raise ValueError("Qualified ICDAR2015 target manifest changed")
    manifest = load_segmentation_manifest(path, require_checksums=True)
    original = old_config(REPO / config["source_training_config"])
    train, val = split_indices(manifest, original)
    if len(train) != 900 or len(val) != config["validation_count"]:
        raise ValueError("Fixed FIT900/validation100 assignment changed")
    return manifest, train, val


def references(manifest, root, indices):
    """Freeze original-coordinate care GT for comparison across output sizes."""
    raw = yaml.safe_load((root / "manifests/icdar2015_train.yaml").read_text())
    expected = {row["id"]: row["source_annotation_sha256"] for row in raw["samples"]}
    result = {}
    for index in indices:
        sample = manifest.samples[index]
        with Image.open(root / sample.image_path) as image:
            w, h = image.size
        path = root / Path(sample.image_path).parent.parent / "annotations" / (sample.sample_id.split(":",1)[1]+".txt")
        if file_sha256(path) != expected[sample.sample_id]:
            raise ValueError("Frozen TRAIN annotation changed")
        instances = parse_icdar2015_annotation(path, image_size_wh=(w,h))
        words = [{"text": item.transcription, "bbox": [item.bbox_xyxy[0], item.bbox_xyxy[1],
                  min(w,item.bbox_xyxy[2]+1), min(h,item.bbox_xyxy[3]+1)]}
                 for item in instances if not item.ignore]
        result[sample.sample_id] = {"image_wh": [w,h], "ground_truth": words,
            "expected_owner_ids": list(range(1,len(words)+1)), "annotation_sha256": expected[sample.sample_id]}
    return result


def model(config, normalization, *, source_checkpoint=None):
    state = None
    if source_checkpoint is not None:
        if file_sha256(source_checkpoint) != config["source_checkpoint_sha256"]:
            raise ValueError("Qualified source checkpoint checksum differs")
        payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        if (payload["epoch"] != config["source_checkpoint_epoch"]
                or payload["architecture"] != "practical-icdar2015-totaltext-transfer-agunet-v1"):
            raise ValueError("Unexpected source checkpoint identity")
        state = payload["model_state_dict"]
    return ImprovementAGUNet(normalization=normalization, max_groups=config["group_norm_max_groups"],
                            skip_floor=config["skip_floor"], source_state=state)


def selection_key(metrics, epoch):
    return (metrics["proposal_f1_iou_50"], -metrics["gt_words_touched_by_fused_components"],
            metrics["word_recall_iou_50"], metrics["foreground_iou"], -epoch)


def evaluate_geometry(model, batches, refs, config, image_hw, *, device, save_dir=None):
    """GT affects only metrics, never predicted canvas/components/crops."""
    totals = {k: 0 for k in ("matched_gt", "gt_words", "proposals", "lost_words",
        "fragmented_words", "fused_components", "words_touched_by_fused_components")}
    rows, ious, seen = [], [], set()
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            probability = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:,0]
            if probability.shape[1:] != tuple(d//2 for d in image_hw) or not np.isfinite(probability).all():
                raise ValueError("Invalid native probability output")
            for offset, sample_id in enumerate(batch["sample_id"]):
                if sample_id in seen:raise ValueError("Repeated validation sample")
                seen.add(sample_id)
                ref = refs[sample_id]
                prediction = (probability[offset] >= config["probability_threshold"]) & valid_canvas(ref["image_wh"],image_hw)
                proposals, labels = binary_proposals(prediction, ref["image_wh"], image_hw)
                geometry = original_geometry(proposals, ref["ground_truth"], iou_threshold=config["matching_iou_threshold"])
                target = batch["mask"][offset,0].numpy().astype(bool)
                valid = batch["validity"][offset,0].numpy().astype(bool)
                owners = batch["owners"][offset].numpy().copy()
                owners[~target | ~valid] = 0
                topology = owner_topology(prediction, owners, ref["expected_owner_ids"])
                for key in geometry:totals[key] += geometry[key]
                for key in ["lost_words","fragmented_words","fused_components","words_touched_by_fused_components"]:
                    totals[key] += topology[key]
                union = int(((prediction | target) & valid).sum())
                intersection = int((prediction & target & valid).sum())
                ious.append(intersection/union if union else 1.)
                row = {"sample_id": sample_id, "proposals": proposals,
                    "foreground_iou": ious[-1], **geometry, "topology": topology}
                if save_dir is not None:
                    save_dir.mkdir(parents=True,exist_ok=True)
                    path = save_dir/(sample_id.replace(":","_")+".npz")
                    with path.with_suffix(".tmp").open("wb") as stream:
                        np.savez_compressed(stream,probability=probability[offset],prediction=prediction)
                    path.with_suffix(".tmp").replace(path)
                    row["mask_file"],row["mask_sha256"] = path.name,file_sha256(path)
                rows.append(row)
    if seen != set(refs):raise ValueError("Incomplete validation population")
    tp, words, props = totals["matched_gt"],totals["gt_words"],totals["proposals"]
    aggregate = {**totals,"foreground_iou":float(np.mean(ious)),
        "proposal_f1_iou_50":2*tp/(words+props) if words+props else 0.,
        "proposal_precision_iou_50":tp/props if props else 0.,"word_recall_iou_50":tp/words if words else 0.,
        "gt_words_touched_by_fused_components":totals["words_touched_by_fused_components"]}
    return {"aggregate":aggregate,"per_image":rows}


@dataclass(frozen=True)
class ImprovementTrainingConfig(SegmentationTrainingConfig):
    """Explicit practical LR/loss extension; historical validator is unchanged."""
    def __post_init__(self):
        if self.learning_rate != 1e-5 or self.loss not in ("masked_BCEWithLogitsLoss","instance_balanced_bce_dice"):
            raise ValueError("Unexpected practical continuation LR/loss")
        paper_fields=asdict(self)
        paper_fields.update(learning_rate=1e-4,loss="masked_BCEWithLogitsLoss")
        SegmentationTrainingConfig(**paper_fields)


def training_config(config, loss):
    t=config["training"]
    return ImprovementTrainingConfig(epochs=t["epochs"],seed=t["seed"],adam_betas=tuple(t["adam_betas"]),
        adam_epsilon=t["adam_epsilon"],learning_rate_schedule=t["learning_rate_schedule"],
        checkpoint_selection="validation_word_f1",deterministic_algorithms=True,optimizer=t["optimizer"],
        learning_rate=t["learning_rate"],weight_decay=t["weight_decay"],batch_size=t["batch_size"],
        loss="masked_BCEWithLogitsLoss" if loss=="separation_bce" else "instance_balanced_bce_dice",
        separation_weight=t["separation_weight"])


def train_epoch(model, batches, optimizer, config, *, loss_name, device, max_steps=None):
    model.train()
    samples, steps, total, ids = 0, 0, 0., []
    for batch in batches:
        if max_steps is not None and steps >= max_steps:break
        images=batch["image"].to(device)
        target,valid,sep=[batch[k].to(device) for k in ("mask","validity","separation")]
        optimizer.zero_grad(set_to_none=True)
        logits=model(images)
        if logits.shape!=target.shape:raise ValueError("Native supervision shape differs")
        t=config["training"]
        if loss_name=="separation_bce":
            loss=separation_weighted_bce_with_logits(logits,target,valid,sep,separation_weight=t["separation_weight"])
        elif loss_name=="instance_balanced_bce_dice":
            loss=instance_balanced_loss(logits,target,valid,sep,batch["owners"].to(device),
                separation_weight=t["separation_weight"],dice_weight=t["instance_dice_weight"],
                dice_margin=t["instance_dice_margin_native_pixels"],epsilon=t["loss_epsilon"])
        else:raise ValueError("Unknown explicitly selected loss")
        if not torch.isfinite(loss):raise FloatingPointError("Non-finite word segmentation loss")
        loss.backward()
        optimizer.step()
        n=len(images);samples+=n;steps+=1;total+=float(loss.detach())*n;ids.extend(batch["sample_id"])
    if not samples:raise ValueError("No training samples")
    return {"loss":total/samples,"samples":samples,"steps":steps,
        "image_shape":list(images.shape),"mask_shape":list(target.shape),
        "sample_order_sha256":sha256("".join(x+"\n" for x in ids).encode()).hexdigest()}
