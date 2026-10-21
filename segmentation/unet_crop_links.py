"""Frozen-prediction crop sampling and word-instance supervision; no GT crop boxes."""

from hashlib import sha256
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional as TF

from segmentation.totaltext_word_kernel import make_word_kernels
from segmentation.unet_word_research_v2 import link_targets, balanced_link_loss
from segmentation.word_multitask import masked_bce_dice


def crop_supervision(owners, validity, box, *, minimum_visible_fraction):
    """Ignore truncated word fragments rather than label them complete words."""
    owners, validity = np.asarray(owners), np.asarray(validity)
    if owners.ndim != 2 or validity.shape != owners.shape or not 0 < minimum_visible_fraction <= 1:
        raise ValueError("Invalid crop supervision")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= owners.shape[1] and 0 <= y1 < y2 <= owners.shape[0]):
        raise ValueError("Training crop outside target")
    cropped = owners[y1:y2, x1:x2].copy()
    valid = validity[y1:y2, x1:x2].astype(bool).copy()
    ignored = []
    for owner in np.unique(cropped[cropped > 0]):
        total = np.count_nonzero((owners == owner) & validity.astype(bool))
        visible = np.count_nonzero((cropped == owner) & valid)
        if not total or visible / total < minimum_visible_fraction:
            valid[cropped == owner] = False
            ignored.append(int(owner))
    cropped[~valid] = 0
    return cropped, valid, ignored


class PredictedCropDataset(Dataset):
    def __init__(self, raw_dataset, records, indices, transform, config):
        self.raw, self.records, self.indices = raw_dataset, records, list(indices)
        self.transform, self.config, self.epoch = transform, config, 0

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.transform.set_epoch(epoch)

    def __getitem__(self, position):
        index = self.indices[position]
        sample = self.raw.manifest.samples[index]
        batch = self.raw[index]
        options = self.records[sample.sample_id]["crops"]
        rng = torch.Generator().manual_seed(self.config["seed"] + self.epoch * 1_000_003 + index)
        chosen = int(torch.randint(len(options), (), generator=rng))
        crop = options[chosen]; box = crop["context_bbox"]
        with Image.open(self.raw.paths[index][4]) as opened:
            owners = np.asarray(opened, dtype=np.int32)
        owners, valid, ignored = crop_supervision(owners, np.asarray(batch["validity"]) > 0, box,
            minimum_visible_fraction=self.config["crop"]["minimum_visible_fraction"])
        owner = Image.fromarray(owners, mode="I")
        mask = Image.fromarray((owners > 0).astype(np.uint8) * 255)
        validity = Image.fromarray(valid.astype(np.uint8) * 255)
        separation = batch["separation"].crop(tuple(box))
        rgb = batch["image"].crop(tuple(box))
        image, mask_t, valid_t, sep = self.transform.transform_with_validity_and_separation(
            rgb, mask, validity, separation, owner, index)
        _, augmented, _ = self.transform._augment(Image.new("RGB", rgb.size), owner, None, index)
        resized = TF.resize(augmented, list(self.transform.config.mask_size), interpolation=InterpolationMode.NEAREST)
        native = np.asarray(resized, np.int32).copy()
        native[(mask_t[0].numpy() == 0) | (valid_t[0].numpy() == 0)] = 0
        kernels, _ = make_word_kernels(native, distance_fraction=self.config["targets"]["distance_fraction"])
        return {"image": image, "mask": mask_t, "validity": valid_t, "separation": sep,
                "owners_native": torch.from_numpy(native),
                "kernel": torch.from_numpy((kernels > 0).astype(np.float32))[None],
                "sample_id": sample.sample_id + ":crop:" + str(chosen),
                "truncated_words": len(ignored), "crop_bbox": torch.tensor(box)}


def crop_loss(evidence, batch, config, epoch):
    options = {k: config["loss"][k] for k in ("bce_weight", "dice_weight", "smooth")}
    valid = batch["validity"]
    terms = {"word": masked_bce_dice(evidence["words"][:, :1], batch["mask"], valid, **options,
        pixel_weight=1 + (config["loss"]["separation_weight"] - 1) * batch["separation"]),
        "kernel": masked_bce_dice(evidence["words"][:, 1:2], batch["kernel"], valid, **options)}
    total = terms["word"] + terms["kernel"]
    coefficient = 0.
    if evidence["links"] is not None:
        target, weight = link_targets(batch["owners_native"], valid)
        terms["links"] = balanced_link_loss(evidence["links"], target, weight)
        coefficient = config["links"]["loss_weight"] * min(1., epoch / config["links"]["ramp_epochs"])
        total = total + coefficient * terms["links"]
    return total, terms, coefficient


def train_crop_epoch(model, batches, optimizer, device, config, epoch):
    model.train(); totals, ids, steps, skipped, ignored = {}, [], 0, 0, 0
    for batch in batches:
        local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, terms, coefficient = crop_loss(model.forward_evidence(local["image"]), local, config, epoch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite learned crop loss")
        if local["validity"].sum() > 0:
            loss.backward(); optimizer.step()
        else:
            skipped += 1
        steps += 1; ids.extend(batch["sample_id"])
        ignored += int(batch["truncated_words"].sum())
        for key, value in {"loss": loss, **terms}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * len(batch["image"])
    if not ids:
        raise ValueError("Empty predicted-crop training loader")
    return {**{k: v / len(ids) for k, v in totals.items()}, "samples": len(ids), "steps": steps,
            "link_coefficient": coefficient, "ignored_truncated_words": ignored,
            "skipped_unsupervised_batches": skipped,
            "sample_order_sha256": sha256("".join(s + "\n" for s in ids).encode()).hexdigest()}
