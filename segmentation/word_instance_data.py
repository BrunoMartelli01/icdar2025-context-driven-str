"""Read-only, hash-bound source maps and separated inference/training inputs."""

from pathlib import Path
import json
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from localize.word_multitask import word_instances, parseq_word_proposals
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_geometry
from segmentation.word_instance_extensions import prompt_supervision


def resize_array(array, size, *, nearest=False):
    tensor = torch.as_tensor(np.asarray(array).copy(), dtype=torch.float32)[None, None]
    mode = "nearest" if nearest else "bilinear"
    return F.interpolate(tensor, (size, size), mode=mode,
                         **({} if nearest else {"align_corners": False}))[0, 0]


def inference_packets(rgb, probabilities, labels, canvas, config):
    """No owners, validity, transcripts or GT boxes accepted by this interface."""
    size = config["local_size"]
    n = labels.shape[0]
    if labels.shape != (n, n) or probabilities.shape != (2, n, n) or canvas.shape != (n, n):
        raise ValueError("Source inference map geometry differs")
    square = letterbox_geometry(rgb, fill=tuple(config["canvas_fill_rgb"]))
    patches = []
    for prop in parseq_word_proposals(labels, rgb.size, padding=0, min_area=0, max_instances=None):
        sid = prop["instance_id"]
        x1, y1, x2, y2 = prop["bbox_native"]
        margin_x = max(config["minimum_context_pixels"], math.ceil((x2-x1)*config["context_fraction"]))
        margin_y = max(config["minimum_context_pixels"], math.ceil((y2-y1)*config["context_fraction"]))
        window = [max(0, x1-margin_x), max(0, y1-margin_y), min(n, x2+margin_x), min(n, y2+margin_y)]
        a, b, c, d = window
        prompt_full = (labels == sid) & (probabilities[1] >= config["kernel_threshold"])
        prompt = resize_array(prompt_full[b:d, a:c], size, nearest=True) > 0
        text = resize_array(probabilities[0, b:d, a:c], size)
        patch = square.transform((size, size), Image.Transform.EXTENT,
                                 tuple(v*square.width/n for v in window), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.asarray(patch).copy()).permute(2, 0, 1).float()/255
        box = torch.tensor([(x1-a)*size/(c-a), (y1-b)*size/(d-b),
                            (x2-a)*size/(c-a), (y2-b)*size/(d-b)], dtype=torch.float32)
        patches.append({"proposal": prop, "window": window, "seed_id": sid,
                        "prompt_full": prompt_full, "prompt": prompt,
                        "canvas": resize_array(canvas[b:d, a:c], size, nearest=True) > 0,
                        "rgb": pixels, "text": text, "box": box,
                        "inputs": torch.cat((pixels*2-1, text[None], prompt.float()[None]))})
    return patches


def training_targets(packet, owners, validity, gt, config):
    target, valid, audit = prompt_supervision(owners, validity, packet["prompt_full"], packet["window"],
        minimum_majority=config["owner_majority"], minimum_visible=config["minimum_visible_fraction"])
    size = config["local_size"]
    x1, y1, x2, y2 = packet["window"]
    oid = audit["owner_id"]
    local_owners = resize_array(owners[y1:y2, x1:x2], size, nearest=True).numpy().astype(np.int32)
    local_valid = resize_array(validity[y1:y2, x1:x2], size, nearest=True).bool()
    text, truth_box = None, None
    if oid and audit["unsupervised_reason"] is None:
        if oid > len(gt):
            raise ValueError("Qualified v4 owner ID no longer matches care annotation order")
        text = gt[oid-1]["text"]
        yy, xx = np.where((owners == oid) & validity)
        truth_box = torch.tensor([(xx.min()-x1)*size/(x2-x1), (yy.min()-y1)*size/(y2-y1),
                                  (xx.max()+1-x1)*size/(x2-x1), (yy.max()+1-y1)*size/(y2-y1)], dtype=torch.float32)
    return {"target": resize_array(target, size, nearest=True),
            "mask_valid": resize_array(valid, size, nearest=True).bool(),
            "owners": local_owners, "validity": local_valid.numpy(), "text": text,
            "truth_box": truth_box, "audit": audit}


class EvidenceSource:
    def __init__(self, evidence_root, word_root, config):
        self.root, self.word_root = Path(evidence_root).resolve(), Path(word_root).resolve()
        self.config = config
        for name, expected in config["source_hashes"].items():
            if sha256_file(self.root/name) != expected:
                raise ValueError("Frozen source evidence changed: " + name)
        progress = json.loads((self.root/"progress.json").read_text())
        if progress.get("status") != "SUCCESS" or progress.get("stage") != "complete":
            raise ValueError("Source suite is incomplete")
        self.source_contract = json.loads((self.root/"run_contract.json").read_text())
        self.maps = {}
        for split, count in (("training", config["training_images"]), ("validation", config["evaluation_images"])):
            receipt = json.loads((self.root/(split+"_map_receipt.json")).read_text())
            if len(receipt["maps"]) != count or receipt["images"] != count:
                raise ValueError("Incomplete source map population")
            self.maps[split] = receipt["maps"]
        if set(self.maps["training"]) & set(self.maps["validation"]):
            raise ValueError("Source training/evaluation overlap")
        manifest = self.word_root/config["train_manifest"]
        if sha256_file(manifest) != config["word_manifest_sha256"]:
            raise ValueError("Word dataset manifest changed")
        self.samples = {s.sample_id: s for s in load_segmentation_manifest(manifest, require_checksums=True).samples}
        if set(self.samples) != set(self.maps["training"]) | set(self.maps["validation"]):
            raise ValueError("Dataset/map populations differ")
        self.post = json.loads((self.root/"run_contract.json").read_text())["resolved_config"]
        # Original exact decoder settings, rather than a current mutable source default.
        old_source = Path(self.source_contract["inputs"]["control_run"])/"run_manifest.json"
        if sha256_file(old_source) != config["control_manifest_sha256"]:
            raise ValueError("Qualified control postprocessing receipt changed")
        self.post = json.loads(old_source.read_text())["run_contract"]["resolved_config"]["postprocessing"]
        self.post_receipt_sha256 = sha256_file(old_source)
        if any(config["crop"][key] != self.post[key] for key in ("text_threshold", "kernel_threshold")):
            raise ValueError("Source Text/Kernel thresholds changed")
        self._packets = {}

    def ids(self, split):
        return sorted(self.maps[split], key=lambda s: int(s.split(":")[1]))

    def image_path(self, sid):
        return self.word_root/self.samples[sid].image_path

    def load(self, sid, *, decode=True):
        split = "training" if sid in self.maps["training"] else "validation"
        path = self.root/(split+"_maps")/(sid.replace(":", "__")+".npz")
        if sha256_file(path) != self.maps[split][sid]:
            raise ValueError("Frozen map changed: " + sid)
        with np.load(path, allow_pickle=False) as saved:
            m = {k: saved[k] for k in saved.files}
        image = self.image_path(sid)
        if sha256_file(image) != self.samples[sid].image_sha256:
            raise ValueError("Original RGB changed: " + sid)
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
        if rgb.size != tuple(m["image_wh"]) or m["probabilities"].shape != (2, *m["canvas"].shape):
            raise ValueError("Source RGB/map geometry differs")
        labels = word_instances(m["probabilities"], m["canvas"].astype(bool), self.post)[0] if decode else None
        return rgb, m, labels

    def packets(self, sid):
        rgb, m, labels = self.load(sid)
        packets = inference_packets(rgb, m["probabilities"], labels, m["canvas"].astype(bool), self.config["crop"])
        return rgb, m, labels, packets

    def gt(self, sid, m):
        rows = _train_gt(self.samples[sid], self.word_root, tuple(int(v) for v in m["image_wh"]))[0]
        return [{"bbox": [int(v) for v in row["bbox"]], "text": row["text"]} for row in rows]
