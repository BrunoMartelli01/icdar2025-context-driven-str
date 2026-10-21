"""DeepSolo-environment worker. This module is kept parseable by Python 3.8."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .candidates import (
    CANDIDATES,
    DEEPSOLO_SOURCE_REVISION,
    resolve_runtime_candidate,
    sha256_file,
)
from .contracts import CONTRACT_SCHEMA_VERSION, DeepSoloRequest, DeepSoloResponse
from .reduction import reduce_detections


LABELS_37 = list("abcdefghijklmnopqrstuvwxyz0123456789")
LABELS_96 = list(
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
VITAE_PAD_DIVISOR = 32


def _bootstrap_metadata() -> Dict[str, Optional[str]]:
    return {
        "entrypoint_path": os.environ.get("CONTEXT_STR_M6_WORKER_ENTRYPOINT_PATH"),
        "package_path": os.environ.get("CONTEXT_STR_M6_WORKER_PACKAGE_PATH"),
        "worker_module_path": os.environ.get("CONTEXT_STR_M6_WORKER_MODULE_PATH"),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), delete=False, encoding="utf-8"
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(str(temporary), str(path))


def _decode_ctc(recognition: Any, vocabulary_size: int, custom_dictionary: str) -> str:
    if vocabulary_size == 37:
        labels = LABELS_37
    elif vocabulary_size == 96:
        labels = LABELS_96
    else:
        if not custom_dictionary:
            raise RuntimeError("DeepSolo custom vocabulary has no dictionary path")
        with Path(custom_dictionary).open("rb") as stream:
            labels = pickle.load(stream)
    if len(labels) != vocabulary_size - 1:
        raise RuntimeError("DeepSolo vocabulary size does not match its dictionary")
    output, previous = "", None
    for raw_value in recognition:
        value = int(raw_value)
        if value < vocabulary_size - 1:
            if value != previous:
                label = labels[value]
                output += str(label) if vocabulary_size in (37, 96) else chr(label)
                previous = value
        else:
            previous = None
    # Preserve the frozen evaluator's raw CTC decode. Its visualization path
    # uppercases 37-character output for display only; this contract does not.
    return output


def _as_points(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.size == 0:
        return []
    return array.reshape(-1, 2).astype(float).tolist()


def _boundary_polygon(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.size == 0:
        return []
    if array.ndim == 2 and array.shape[1] % 2 == 0:
        first, second = np.hsplit(array, 2)
        return np.vstack([first, second[::-1]]).reshape(-1, 2).astype(float).tolist()
    return array.reshape(-1, 2).astype(float).tolist()


def _load_model_and_validate(cfg: Any, checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    # Registration imports are required by the frozen upstream source.
    from adet.modeling import swin, vitae_v2  # noqa: F401

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpoint = checkpointer._load_file(checkpoint_path)
    incompatible = checkpointer._load_model(checkpoint)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    incorrect = [
        {"key": item[0], "checkpoint_shape": list(item[1]), "model_shape": list(item[2])}
        for item in incompatible.incorrect_shapes
    ]
    if missing or unexpected or incorrect:
        raise RuntimeError(
            "DeepSolo checkpoint load was incompatible: missing={}, unexpected={}, incorrect_shapes={}".format(
                missing, unexpected, incorrect
            )
        )
    total_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return model, {
        "strict_compatibility_check": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "incorrect_shapes": incorrect,
        "model_class": type(model).__name__,
        "backbone_builder": str(cfg.MODEL.BACKBONE.NAME),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def _apply_frozen_vitae_padding(image: Any, transforms: Any) -> Any:
    """Apply frozen upstream Pad semantics without importing ``adet.data``."""
    height, width = image.shape[:2]
    height_remainder = height % VITAE_PAD_DIVISOR
    width_remainder = width % VITAE_PAD_DIVISOR
    pad_h = 0 if height_remainder == 0 else VITAE_PAD_DIVISOR - height_remainder
    pad_w = 0 if width_remainder == 0 else VITAE_PAD_DIVISOR - width_remainder
    transform = transforms.PadTransform(0, 0, pad_w, pad_h, pad_value=0)
    return transform.apply_image(image)


def _predict(model: Any, cfg: Any, image_bgr: Any, is_vitae: bool) -> Any:
    import numpy as np
    import torch
    import detectron2.data.transforms as transforms

    height, width = image_bgr.shape[:2]
    image = image_bgr
    if cfg.INPUT.FORMAT == "RGB":
        image = image[:, :, ::-1]
    augmentation = transforms.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = augmentation.get_transform(image).apply_image(image)
    if is_vitae:
        image = _apply_frozen_vitae_padding(image, transforms)
    tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    with torch.no_grad():
        return model([{"image": tensor, "height": height, "width": width}])[0]


def _serialize_instances(instances: Any, cfg: Any) -> Any:
    instances = instances.to("cpu")
    required_fields = (
        "scores", "pred_classes", "recs", "rec_scores", "bd", "ctrl_points",
    )
    missing = [field for field in required_fields if not instances.has(field)]
    if missing:
        raise RuntimeError("DeepSolo output is missing expected fields: {}".format(missing))
    scores = instances.scores.tolist()
    predicted_classes = instances.pred_classes.tolist()
    recognitions = instances.recs.numpy()
    recognition_scores = instances.rec_scores.numpy()
    boundaries = instances.bd.numpy()
    control_points = instances.ctrl_points.numpy()
    vocabulary_size = int(cfg.MODEL.TRANSFORMER.VOC_SIZE)
    custom_dictionary = str(cfg.MODEL.TRANSFORMER.CUSTOM_DICT)
    detections = []
    for index, (score, predicted_class, rec, rec_score, boundary, controls) in enumerate(
        zip(
            scores,
            predicted_classes,
            recognitions,
            recognition_scores,
            boundaries,
            control_points,
        )
    ):
        text = _decode_ctc(rec, vocabulary_size, custom_dictionary)
        detections.append({
            "source_index": index,
            "text": text,
            "score": float(score),
            "score_type": "DeepSolo instance score",
            "geometry": {
                "type": "DeepSolo boundary polygon and center-line control points",
                "boundary_points": _boundary_polygon(boundary),
                "control_points": _as_points(controls),
            },
            "raw": {
                "predicted_class": int(predicted_class),
                "recognition_token_ids": [int(value) for value in rec.tolist()],
                "recognition_token_confidences": [
                    float(value) for value in rec_score.max(axis=-1).tolist()
                ],
                "recognition_class_probabilities": rec_score.tolist(),
                "boundary": boundary.tolist(),
                "control_points": controls.tolist(),
            },
        })
    return detections


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def execute(request: DeepSoloRequest, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    import cv2
    import torch

    # The final M6 host qualification uses normal cuDNN execution. Disabling
    # cuDNN was diagnostic history for a VM CPU-feature SIGILL, not a runtime
    # requirement or selectable scientific path.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    runtime = resolve_runtime_candidate(manifest, request.candidate_id)
    if sha256_file(Path(request.image_path)) != request.image_sha256:
        raise RuntimeError("DeepSolo input image SHA256 changed after request creation")
    source_root = Path(runtime["config_path"]).parents[3]
    sys.path.insert(0, str(source_root))
    try:
        from adet.config import get_cfg

        candidate = CANDIDATES[request.candidate_id]
        cfg = get_cfg()
        cfg.merge_from_file(runtime["config_path"])
        cfg.MODEL.WEIGHTS = runtime["checkpoint_path"]
        cfg.MODEL.DEVICE = request.inference_options["device"]
        threshold = request.inference_options.get("score_threshold")
        if threshold is not None:
            cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = float(threshold)
        lexicon_type = request.inference_options.get("lexicon_type")
        if lexicon_type is not None:
            cfg.TEST.LEXICON_TYPE = int(lexicon_type)
        expected_builder = (
            "build_resnet_backbone" if candidate.backbone == "ResNet-50"
            else "build_vitaev2_backbone"
        )
        if cfg.MODEL.BACKBONE.NAME != expected_builder:
            raise RuntimeError("Resolved DeepSolo config has the wrong backbone architecture")
        cfg.freeze()

        cuda_index = None
        if str(cfg.MODEL.DEVICE).startswith("cuda"):
            device = torch.device(cfg.MODEL.DEVICE)
            cuda_index = int(device.index if device.index is not None else torch.cuda.current_device())
            initialization_tensor = torch.empty(1, device=device)
            del initialization_tensor
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cuda_index)

        image = cv2.imread(request.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("OpenCV could not decode the DeepSolo input image")
        total_started = time.perf_counter()
        load_started = time.perf_counter()
        model, load_validation = _load_model_and_validate(cfg, runtime["checkpoint_path"])
        load_seconds = time.perf_counter() - load_started
        inference_started = time.perf_counter()
        predictions = _predict(model, cfg, image, candidate.backbone == "ViTAEv2-S")
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
        inference_seconds = time.perf_counter() - inference_started
        detections = _serialize_instances(predictions["instances"], cfg)
        selected = reduce_detections(detections, request.reduction_policy)
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "request_id": request.request_id,
            "success": True,
            "detections": detections,
            "recognized_strings": [item["text"] for item in detections],
            "selected_output": selected,
            "reduction_policy": request.reduction_policy,
            "artifact": {
                "status": "APPROXIMATE candidate; not paper-selected",
                "candidate": candidate.to_dict(),
                "source_revision": DEEPSOLO_SOURCE_REVISION,
                "checkpoint_sha256": runtime["checkpoint_sha256"],
                "config_sha256": runtime["config_sha256"],
                "load_validation": load_validation,
                "inference_options": dict(request.inference_options),
                "resolved_inference_threshold": float(
                    cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
                ),
                "resolved_lexicon_type": int(cfg.TEST.LEXICON_TYPE),
                "versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "torchvision": _package_version("torchvision"),
                    "detectron2": _package_version("detectron2"),
                    "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                },
                "runtime_qualification": {
                    **dict(runtime.get("runtime_metadata", {})),
                    "container_image": os.environ.get(
                        "CONTEXT_STR_DEEPSOLO_IMAGE_TAG"
                    ) or runtime.get("runtime_metadata", {}).get("container_image"),
                    "container_image_id": os.environ.get(
                        "CONTEXT_STR_DEEPSOLO_IMAGE_ID"
                    ) or runtime.get("runtime_metadata", {}).get("container_image_id"),
                    "cudnn_enabled": bool(torch.backends.cudnn.enabled),
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                    "gpu_name": (
                        torch.cuda.get_device_name(cuda_index)
                        if cuda_index is not None else None
                    ),
                    "worker_bootstrap": _bootstrap_metadata(),
                },
            },
            "latency": {
                "load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "total_seconds": time.perf_counter() - total_started,
            },
            "placement": {
                "process_boundary": "isolated context-str-deepsolo subprocess",
                "device": str(cfg.MODEL.DEVICE),
                "worker_bootstrap": _bootstrap_metadata(),
            },
            "memory": {
                "gpu_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(cuda_index))
                    if cuda_index is not None else None
                ),
                "gpu_peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(cuda_index))
                    if cuda_index is not None else None
                ),
            },
            "error": None,
        }
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)


def failure_response(raw_request: Mapping[str, Any], error: BaseException) -> Dict[str, Any]:
    policy = raw_request.get("reduction_policy", "highest_score")
    if policy not in ("highest_score", "source_order_space_join", "reading_order_space_join"):
        policy = "highest_score"
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "request_id": str(raw_request.get("request_id", "unknown")),
        "success": False,
        "detections": [],
        "recognized_strings": [],
        "selected_output": "",
        "reduction_policy": policy,
        "artifact": {},
        "latency": {},
        "placement": {"worker_bootstrap": _bootstrap_metadata()},
        "memory": {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    }


def run(request_path: Path, response_path: Path, manifest_path: Path) -> int:
    raw_request = {}
    try:
        raw_request = json.loads(request_path.read_text(encoding="utf-8"))
        request = DeepSoloRequest.from_dict(raw_request)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        response = execute(request, manifest)
        DeepSoloResponse.from_dict(response)
        _write_json_atomic(response_path, response)
        return 0
    except Exception as error:
        response = failure_response(raw_request, error)
        DeepSoloResponse.from_dict(response)
        _write_json_atomic(response_path, response)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    return run(args.request, args.response, args.runtime_manifest)
