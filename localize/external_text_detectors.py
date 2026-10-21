"""Detector-only adapters; pretrained models never substitute for PARSeq decoding."""
from __future__ import annotations

import gc
import importlib.metadata
import math
from pathlib import Path

import cv2
import numpy as np
import torch


def region(points, score: float, image_wh: tuple[int, int], source_index: int) -> dict:
    points = np.asarray(points, dtype=np.float32)
    width, height = image_wh
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
        or not np.isfinite(points).all() or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("malformed detector polygon or score")
    degenerate = abs(cv2.contourArea(points)) <= 0
    bbox = [max(0, min(width, math.floor(float(points[:, 0].min())))),
            max(0, min(height, math.floor(float(points[:, 1].min())))),
            max(0, min(width, math.ceil(float(points[:, 0].max())))),
            max(0, min(height, math.ceil(float(points[:, 1].max()))))]
    # Out-of-image regions are explicit rejected records, never exception swallowing.
    valid = not degenerate and bbox[2] > bbox[0] and bbox[3] > bbox[1]
    return {"source_index": source_index, "bbox": bbox, "polygon": points.tolist(),
            "score": float(score), "valid": valid,
            "rejection": None if valid else "degenerate detector polygon" if degenerate else "no in-image area"}


def doctr_regions(predictions: dict, image_wh: tuple[int, int]) -> list[dict]:
    """Decode docTR's normalized axis boxes or quadrilaterals + score row."""
    if set(predictions) != {"words"}:
        raise ValueError(f"unexpected docTR classes: {list(predictions)}")
    values = np.asarray(predictions["words"])
    if values.shape not in ((0, 5), (0, 5, 2)) and not (
        values.ndim == 2 and values.shape[1:] == (5,) or
        values.ndim == 3 and values.shape[1:] == (5, 2)):
        raise ValueError(f"unexpected docTR geometry shape: {values.shape}")
    rows = []
    for index, value in enumerate(values):
        if value.ndim == 1:
            x1, y1, x2, y2, score = value
            points = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
        else:
            # docTR 0.11 DB/FAST append (0, score), not a fifth point or
            # duplicated score. Padding removal leaves this row unchanged.
            points, score = value[:4], value[4, 1]
            if value[4, 0] != 0:
                raise ValueError("docTR score row must be (0, score)")
        rows.append(region(points * np.array(image_wh), float(score), image_wh, index))
    return rows


def crop_region(image: np.ndarray, row: dict, view: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not row["valid"]:
        raise ValueError("invalid RGB image or rejected region")
    x1, y1, x2, y2 = row["bbox"]
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError("crop envelope outside image")
    if view == "axis_aligned":
        return np.ascontiguousarray(image[y1:y2, x1:x2])
    if view != "rectified":
        raise ValueError("unknown crop view")
    # A generic rectangle, not a curve-aware rectifier. Preserve raw predicted geometry.
    box = cv2.boxPoints(cv2.minAreaRect(np.asarray(row["polygon"], np.float32)))
    center = box.mean(axis=0)
    order = np.argsort(np.arctan2(box[:,1] - center[1], box[:,0] - center[0]))
    box = box[order]
    box = np.roll(box, -int(np.argmin(box.sum(axis=1))), axis=0)
    a, b = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])
    if b > a:
        box = np.roll(box, -1, axis=0)
        a, b = b, a
    width, height = max(2, math.ceil(float(a))), max(2, math.ceil(float(b)))
    if width * height > image.shape[0] * image.shape[1] * 4:
        raise ValueError("predicted rectangle is excessively outside image")
    destination = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], np.float32)
    transform = cv2.getPerspectiveTransform(box.astype(np.float32), destination)
    return np.ascontiguousarray(cv2.warpPerspective(image, transform, (width,height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)))


class DetectorRuntime:
    def __init__(self, name: str, config: dict, checkpoint: Path, device: str):
        if importlib.metadata.version(config["package"]) != config["version"]:
            raise RuntimeError(f"{name} requires {config['package']}=={config['version']}")
        self.name, self.config, self.device = name, config, device
        self.last_input_shape = None
        if name == "craft":
            from easyocr.craft import CRAFT
            from easyocr.detection import copyStateDict
            self.model = CRAFT(pretrained=False, freeze=False)
            state = copyStateDict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.model.load_state_dict(state, strict=True)
            self.parameters_before_reparameterize = sum(p.numel() for p in self.model.parameters())
            self.predictor = None
        else:
            from doctr.models import detection_predictor, db_resnet50, fast_tiny
            factories = {"db_resnet50": db_resnet50, "fast_tiny": fast_tiny}
            self.model = factories[name](pretrained=False, pretrained_backbone=False,
                assume_straight_pages=False, bin_thresh=config["bin_thresh"], box_thresh=config["box_thresh"])
            if self.model.cfg["url"] != config["url"]:
                raise RuntimeError("qualified docTR checkpoint URL differs from pinned package")
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            self.model.eval()
            self.parameters_before_reparameterize = sum(p.numel() for p in self.model.parameters())
            if name == "fast_tiny" and config["reparameterize"]:
                from doctr.models.detection.fast import reparameterize
                self.model = reparameterize(self.model)
            self.model.cfg = {**self.model.cfg, "input_shape": (3,config["input_size"],config["input_size"])}
            self.predictor = detection_predictor(arch=self.model, pretrained=False,
                assume_straight_pages=False, preserve_aspect_ratio=True, symmetric_pad=True, batch_size=1)
            self.predictor.to(device).eval()
        self.model.to(device).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.hook = self.model.register_forward_pre_hook(self._input_hook)

    def _input_hook(self, module, inputs):
        self.last_input_shape = list(inputs[0].shape)

    @torch.inference_mode()
    def infer(self, image: np.ndarray) -> list[dict]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("detectors require an original RGB uint8 raster")
        image_wh = (image.shape[1], image.shape[0])
        if self.name != "craft":
            result = self.predictor([image])
            if len(result) != 1:
                raise RuntimeError("docTR returned incomplete detector batch")
            return doctr_regions(result[0], image_wh)
        from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
        from easyocr.craft_utils import getDetBoxes_core, adjustResultCoordinates
        cfg = self.config
        resized, ratio, _ = resize_aspect_ratio(image, cfg["canvas_size"],
            interpolation=cv2.INTER_LINEAR, mag_ratio=cfg["mag_ratio"])
        tensor = torch.from_numpy(normalizeMeanVariance(resized).transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
        output, _ = self.model(tensor)
        maps = output[0].cpu().numpy()
        if maps.ndim != 3 or maps.shape[2] != 2 or not np.isfinite(maps).all():
            raise RuntimeError("CRAFT returned malformed or nonfinite heatmaps")
        text, links = maps[:,:,0], maps[:,:,1]
        boxes, labels, mapper = getDetBoxes_core(text, links, cfg["text_threshold"],
            cfg["link_threshold"], cfg["low_text"])
        # CRAFT regresses Gaussian heatmaps, not calibrated probabilities. Preserve
        # the upstream maximum and bound its admission score to [0,1] explicitly.
        scores = [float(text[labels == index].max()) for index in mapper]
        boxes = adjustResultCoordinates(boxes, 1 / ratio, 1 / ratio)
        if len(boxes) != len(scores):
            raise RuntimeError("CRAFT boxes and component scores disagree")
        return [{**region(box, max(0.,min(1.,score)), image_wh, index), "native_detector_score":score}
                for index,(box,score) in enumerate(zip(boxes,scores))]

    def release(self):
        self.hook.remove()
        self.predictor = None
        self.model = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
