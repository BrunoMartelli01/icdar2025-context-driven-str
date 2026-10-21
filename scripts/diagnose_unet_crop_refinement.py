"""Paired, GT-free APPROXIMATE U-Net crop resegmentation with saved evidence.

Run with ``python -m scripts.diagnose_unet_crop_refinement --help``.
No training, OCR, captioner or DeepSolo is started. No dependency is installed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from localize.unet_crop_refinement import (
    ContourConfig, CropRefinementConfig, refine_contour_crops,
)
from models.unet_adv import AuthorProtocolAGUNet, PracticalTotalTextHighResAGUNet
from segmentation.data import file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import letterbox_geometry
from segmentation.training import load_checkpoint, seed_everything


def load_config(path: Path) -> tuple[dict, CropRefinementConfig]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("reproduction_mode") != "approximate"
        or path.resolve().parent.name != "approximate"
        or config.get("claim_label") != "approximate reproduction"):
        raise ValueError("Crop refinement requires an explicit configs/approximate profile; STRICT forbidden")
    if config.get("schema_version") != 1 or config.get("experiment_kind") != "unet_crop_resegmentation":
        raise ValueError("Unsupported crop-refinement configuration schema")
    if "approximate reproduction" not in config.get("result_class", ""):
        raise ValueError("Diagnostic result_class must include approximate reproduction")
    choices = config.get("scientific_choices", [])
    if not choices or any(choice.get("classification") not in
                          {"PAPER_SPECIFIED", "REPOSITORY_SPECIFIED", "ASSUMPTION", "HARDWARE_DEVIATION"}
                          or not all(choice.get(key) for key in ("rationale", "expected_effect", "ambiguity"))
                          for choice in choices):
        raise ValueError("Scientific choices require classifications, rationale, effect and ambiguity links")
    spec = config["inference"]
    supported = {"author-clarified-ag-unet-2048x1024-v1": ([256, 256], [128, 128]),
                 "practical-totaltext-v4-highres-agunet-v1": ([512, 512], [256, 256])}
    if (spec.get("architecture") not in supported
        or (spec.get("input_size"), spec.get("native_mask_size")) != supported[spec["architecture"]]
        or spec.get("normalization") != "imagenet"
        or spec.get("precision") != "float32" or spec.get("batch_size") != 1
        or spec.get("binary_backprojection") != "nearest_after_native_threshold"
        or spec.get("resize_geometry") not in {"stretch", "letterbox"}):
        raise ValueError("Unsupported model/input/preprocessing contract; use author256 or practical512 float32")
    for name in ("probability_threshold", "skip_floor"):
        value = spec.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0 <= value <= 1
            or (name == "skip_floor" and value == 1)):
            raise ValueError(f"Invalid {name}: require finite [0,1], skip_floor strictly below 1")
    if spec["architecture"].startswith("author-") and spec["skip_floor"] != 0:
        raise ValueError("Author-protocol model has no practical skip floor")
    fill = spec.get("canvas_fill_rgb")
    if (not isinstance(fill, list) or len(fill) != 3
        or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in fill)):
        raise ValueError("canvas_fill_rgb requires three integers in [0,255]")
    if isinstance(spec.get("seed"), bool) or not isinstance(spec.get("seed"), int) or spec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    raw = dict(config["refinement"])
    raw["initial"] = ContourConfig(**raw["initial"])
    raw["second_pass"] = ContourConfig(**raw["second_pass"])
    return config, CropRefinementConfig(**raw)


def verify_training_provenance(spec: dict, training_path: Path, run_path: Path, checkpoint: Path) -> dict:
    """Reject mismatched artifacts, skip-floor behavior and resize geometry early."""
    training = OmegaConf.to_container(OmegaConf.load(training_path), resolve=True)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (training.get("reproduction_mode") != "approximate" or run.get("status") != "SUCCESS"
        or file_sha256(checkpoint) != run.get("checkpoint", {}).get("sha256")):
        raise ValueError("Require approximate training, SUCCESS run and matching checkpoint SHA256")
    contract = run.get("run_contract")
    if spec["architecture"].startswith("practical-"):
        if not isinstance(contract, dict):
            raise ValueError("Practical training run must supply its versioned run_contract")
        if contract.get("config_sha256") != file_sha256(training_path):
            raise ValueError("Training-config SHA256 does not match run provenance")
        if (contract.get("architecture") != spec["architecture"]
            or contract.get("input_shape") != [3, *spec["input_size"]]
            or contract.get("logits_shape") != [1, *spec["native_mask_size"]]):
            raise ValueError("Model architecture/input/logit shapes differ from training run")
        if contract.get("base_model_source_sha256") is not None:
            from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _base_model_source_matches
            if not _base_model_source_matches(contract["base_model_source_sha256"]):
                raise ValueError("Model source differs from qualified training source")
        if contract.get("transform_source_sha256") is not None:
            transform_path = Path(__file__).resolve().parents[1] / "segmentation/totaltext_image_frequency_augment.py"
            if file_sha256(transform_path) != contract["transform_source_sha256"]:
                raise ValueError("Transform source differs from qualified training source")
        variant = contract.get("variant")
        expected_skip = training.get("variants", {}).get(variant, {}).get(
            "skip_floor", training.get("training", {}).get("skip_floor"))
        if spec["skip_floor"] != expected_skip or spec["skip_floor"] != contract.get("skip_floor"):
            raise ValueError("Practical skip floor differs from training configuration/run")
        intervention = contract.get("intervention", {})
        geometry = "letterbox" if intervention.get("image_intervention") == "letterbox" else "stretch"
        if spec["resize_geometry"] != geometry:
            raise ValueError("Resize geometry differs from training run")
        if geometry == "letterbox":
            parameters = training.get("variants", {}).get(variant, {}).get("parameters", {})
            if (spec["canvas_fill_rgb"] != parameters.get("canvas_fill_rgb")
                or spec["canvas_fill_rgb"] != intervention.get("parameters", {}).get("canvas_fill_rgb")):
                raise ValueError("Letterbox fill differs from training configuration/run")
    else:
        transforms = training.get("transforms", {})
        if (run.get("resolved_config", {}).get("sha256") != file_sha256(training_path)
            or run.get("architecture", {}).get("identity") != spec["architecture"]
            or training.get("architecture", {}).get("identity") != spec["architecture"]):
            raise ValueError("Author run requires its exact saved resolved_config.yaml and architecture")
        if (transforms.get("normalization") != spec["normalization"]
            or transforms.get("resize_geometry") != spec["resize_geometry"]
            or transforms.get("image_size") != spec["input_size"]
            or transforms.get("mask_size") != spec["native_mask_size"]):
            raise ValueError("Author preprocessing differs from training configuration")
    return {"checkpoint": {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint),
                           "epoch": run["checkpoint"].get("epoch")},
            "training_config": {"path": str(training_path.resolve()), "sha256": file_sha256(training_path)},
            "training_run": {"path": str(run_path.resolve()), "sha256": file_sha256(run_path)},
            "run_contract": contract}


class RGBSegmenter:
    """Shared model/preprocessing for global and cropped RGB, binary projection."""

    def __init__(self, model: torch.nn.Module, spec: dict, device: torch.device):
        self.model, self.spec, self.device = model.eval().to(device), spec, device

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> dict:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
            raise ValueError("Expected nonempty uint8 RGB image")
        height, width = rgb.shape[:2]
        image = Image.fromarray(rgb)
        letterbox = self.spec["resize_geometry"] == "letterbox"
        if letterbox:
            image = letterbox_geometry(image, fill=tuple(self.spec["canvas_fill_rgb"]))
        image = TF.resize(image, self.spec["input_size"], interpolation=InterpolationMode.BILINEAR,
                          antialias=True)
        tensor = TF.normalize(TF.to_tensor(image), (0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        expected = (1, 1, *self.spec["native_mask_size"])
        if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
            raise RuntimeError(f"Expected finite logits {expected}, got {tuple(logits.shape)}")
        probability = logits.sigmoid()[0, 0].float().cpu().numpy()
        native = (probability >= self.spec["probability_threshold"]).astype(np.uint8)
        target_hw = (max(height, width),) * 2 if letterbox else (height, width)
        projected = F.interpolate(torch.from_numpy(native).float()[None, None],
                                  size=target_hw, mode="nearest")[0, 0].byte().numpy()
        if letterbox:
            top, left = (target_hw[0] - height) // 2, (target_hw[1] - width) // 2
            projected = projected[top:top + height, left:left + width]
        return {"mask": np.ascontiguousarray(projected), "mask_native": native,
                "probability_native": probability,
                "metadata": {"input_shape": list(tensor.shape), "logits_shape": list(logits.shape),
                             "original_rgb_hw": [height, width], "device": str(self.device),
                             "precision": "float32", "batch_size": 1,
                             "preprocessing": dict(self.spec)}}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def save_evidence(directory: Path, rgb: np.ndarray, baseline: dict, result: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {}

    def save(name: str, value: np.ndarray, *, mask: bool = False) -> None:
        path = directory / name
        if path.suffix == ".npy":
            np.save(path, value, allow_pickle=False)
        else:
            Image.fromarray(value * 255 if mask else value).save(path)
        artifacts[name] = {"sha256": file_sha256(path), "shape": list(value.shape),
                           "dtype": str(value.dtype)}

    save("image.png", rgb)
    save("initial-mask.png", result["initial_mask"], mask=True)
    save("refined-roi-union-mask.png", result["refined_mask"], mask=True)
    for key in ("mask_native", "probability_native"):
        save(f"initial-{key}.npy", baseline[key])
    for artifact in result["crop_artifacts"]:
        prefix = f"seed-{artifact['seed_rank']:02d}"
        for key, value in artifact.items():
            if key == "seed_rank":
                continue
            if key in {"crop_rgb", "before_mask", "seed_support_mask", "after_mask"}:
                save(f"{prefix}-{key}.png", value, mask=key.endswith("mask"))
            else:
                save(f"{prefix}-{key}.npy", value)
    report = result["report"]
    for arm, rows in (("baseline", report["initial_localization"]["retained"]),
                      ("refined", report["refined_proposals"])):
        view = Image.fromarray(rgb).copy()
        draw = ImageDraw.Draw(view)
        for row in rows:
            x1, y1, x2, y2 = row["padded_bbox"]
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="red", width=2)
            draw.text((x1, y1), str(row["rank"]), fill="yellow")
            save(f"{arm}-crop-{row['rank']:02d}.png", rgb[y1:y2, x1:x2])
        save(f"{arm}-boxes.png", np.asarray(view))
    return artifacts


def run(args: argparse.Namespace) -> dict:
    config, refinement = load_config(args.config)
    spec = config["inference"]
    # Validate every artifact before allocating/loading the model or writing output.
    provenance = verify_training_provenance(spec, args.training_config,
                                           args.training_run_manifest, args.checkpoint)
    if not args.images or len(set(path.resolve() for path in args.images)) != len(args.images):
        raise ValueError("Supply a nonempty list of distinct input images")
    for image in args.images:
        if not image.is_file():
            raise FileNotFoundError(image)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu or run on the GPU host")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    protocol = {"config": config, "resolved_refinement": asdict(refinement), "provenance": provenance,
                "config_sha256": file_sha256(args.config), "device": str(device),
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "opencv": cv2.__version__,
                "platform": platform.platform(), "machine": platform.machine(),
                "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "sources": {name: file_sha256(source_root / name) for name in
                            ("scripts/diagnose_unet_crop_refinement.py", "localize/unet_crop_refinement.py",
                             "models/unet_adv.py", "segmentation/training.py",
                             "segmentation/totaltext_image_frequency_augment.py")},
                "images": [{"path": str(path.resolve()), "sha256": file_sha256(path)}
                           for path in args.images],
                "result_class": config["result_class"], "reproduction_mode": "approximate",
                "runtime_scientific_choices": ([{
                    "component": "cpu_execution", "classification": "HARDWARE_DEVIATION",
                    "ambiguity": "UCR-001", "rationale": "Explicit CPU engineering smoke instead of paper GPU execution; actual thread/backend settings recorded in summary.",
                    "expected_effect": "Different numerical backend and latency; no CUDA fit or VRAM qualification.",
                }] if device.type == "cpu" else []),
                "ground_truth_use": "none", "recognition_use": "none",
                "comparison_baseline": "this profile's full-image binary projection + component-area contours"}
    write_json(args.output_dir / "resolved-config.json", protocol)
    records = []
    cuda_index = None
    started = time.perf_counter()
    active_image = None
    stage = "model_loading"
    try:
        seed_everything(spec["seed"], deterministic_algorithms=True)
        backend = {"seed": spec["seed"],
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "cpu_threads": torch.get_num_threads()}
        if device.type == "cuda":
            cuda_index = _prepare_cuda_memory_tracking(device)
        if spec["architecture"].startswith("author-"):
            model = AuthorProtocolAGUNet(encoder_pretrained=False)
        else:
            model = PracticalTotalTextHighResAGUNet(encoder_pretrained=False, skip_floor=spec["skip_floor"])
        load_checkpoint(args.checkpoint, model=model, expected_architecture=spec["architecture"],
                        expected_run_contract=provenance["run_contract"])
        segmenter = RGBSegmenter(model, spec, device)
        for position, image_path in enumerate(args.images, 1):
            active_image = str(image_path.resolve())
            stage = "full_image_segmentation"
            print(f"[crop refinement] {position}/{len(args.images)} {image_path.name}", flush=True)
            with Image.open(image_path) as image:
                rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
            baseline = segmenter(rgb)
            stage = "crop_resegmentation"
            result = refine_contour_crops(rgb, baseline["mask"], segmenter, refinement)
            stage = "saving_evidence"
            directory = args.output_dir / f"image-{position:04d}"
            artifacts = save_evidence(directory, rgb, baseline, result)
            record = {"image": protocol["images"][position - 1], "report": result["report"],
                      "baseline_segmentation_metadata": baseline["metadata"], "artifacts": artifacts}
            write_json(directory / "report.json", record)
            records.append({"directory": directory.name, "image": record["image"],
                            "baseline_blocks": len(result["report"]["initial_localization"]["retained"]),
                            "refined_blocks": len(result["report"]["refined_proposals"]),
                            "second_pass_calls": result["report"]["inference_calls_second_pass"],
                            "report_sha256": file_sha256(directory / "report.json")})
        memory = {"peak_allocated_bytes": None, "peak_reserved_bytes": None,
                  "qualification": "CPU run; CUDA/VRAM not measured"}
        if cuda_index is not None:
            torch.cuda.synchronize(device)
            memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_index),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_index),
                      "qualification": "successful CUDA run; OOM safety not implied"}
        summary = {"status": "SUCCESS", "reproduction_mode": "approximate",
                   "claim_label": "approximate reproduction", "result_class": config["result_class"],
                   "images_requested": len(args.images), "images_completed": len(records),
                   "samples": records, "memory": memory, "backend": backend,
                   "seconds": time.perf_counter() - started,
                   "model_set": [spec["architecture"]], "inference": spec,
                   "word_correctness": "UNRESOLVED; inspect paired evidence, no word accuracy claim"}
        write_json(args.output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {"status": "FAILED", "reproduction_mode": "approximate",
                   "claim_label": "approximate reproduction", "result_class": config["result_class"],
                   "error_type": type(error).__name__, "error": str(error),
                   "out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
                   "images_requested": len(args.images), "images_completed": len(records),
                   "samples": records, "model_set": [spec["architecture"]], "inference": spec,
                   "device": str(device)}
        failure.update(stage=stage, active_image=active_image)
        if cuda_index is not None:
            failure["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_index),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_index)}
        write_json(args.output_dir / "failure.json", failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    summary = run(parser.parse_args())
    print(json.dumps({key: summary[key] for key in ("status", "images_completed", "memory")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
