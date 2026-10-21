"""Frozen single U-Net and Hybrid384 detection on Total-Text TEST (APPROXIMATE)."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from localize.word_crop_refinements import filter_fallbacks, resize_probability
from localize.word_multitask import parseq_word_proposals, word_instances
from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
from models.unet_adv import PracticalTotalTextHighResAGUNet
from models.unet_word_research import WordResearchAGUNet
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.evaluate_text_recall_refinements import read_config as read_source_config
from scripts.evaluate_totaltext_highres_contour_padding import _annotation_path, _ground_truth, _rgb
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask, letterbox_geometry
from segmentation.training import load_checkpoint, runtime_environment, seed_everything


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/unet_word_detection_test_v1.json"


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE configuration required; no namespace fallback")
    cfg = json.loads(path.read_text())
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != mode
            or cfg.get("result_class") != "approximate reproduction"
            or cfg.get("models") != ["text_kernel", "hybrid384"]
            or cfg.get("method") != "seeded_only" or cfg.get("tn") != 0
            or cfg.get("batch_size") != 1 or cfg.get("precision") != "float32"
            or cfg.get("test_images") != 300 or cfg.get("test_care_words") != 2204
            or cfg.get("iou_threshold") != 0.5):
        raise ValueError("Unsupported frozen U-Net TEST protocol")
    source_path = REPO / cfg["source_config"]
    if sha256_file(source_path) != cfg["source_config_sha256"]:
        raise ValueError("Frozen TRAIN localization configuration changed")
    source, preprocessing = read_source_config(source_path, mode)
    return cfg, source, preprocessing


def image_tensor(rgb, size, preprocessing):
    fill = tuple(preprocessing["variants"]["letterbox"]["parameters"]["canvas_fill_rgb"])
    image = letterbox_geometry(Image.fromarray(rgb), fill=fill)
    image = TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(TF.to_tensor(image), (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)).unsqueeze(0)


def localize(probabilities, canvas, image_wh, source, threshold):
    post = {**source["postprocessing"], "text_threshold": threshold}
    labels, audit, _ = word_instances(probabilities, canvas, post)
    labels, filtered = filter_fallbacks(labels, probabilities[1],
        seed_threshold=post["kernel_threshold"], fallback_threshold=post["kernel_threshold"])
    proposals = parseq_word_proposals(labels, image_wh, **source["crop"])
    return labels, proposals, {"partition": audit, "seeded_only": filtered}


def score_detection(labels, proposals, validity, ground_truth, threshold):
    if labels.shape != validity.shape or validity.dtype != np.bool_:
        raise ValueError("Native instance/validity geometry mismatch")
    ignored = [int(i) for i in np.unique(labels[labels > 0])
               if not np.any((labels == i) & validity)]
    care = [p for p in proposals if p["instance_id"] not in ignored]
    edges = [(i, j) for i, p in enumerate(care) for j, g in enumerate(ground_truth)
             if box_iou(p["bbox"], g["bbox"]) >= threshold]
    matched = maximum_matching(len(care), len(ground_truth), edges)
    tp = len(matched)
    return {"tp": tp, "tn": 0, "fp": len(care) - tp, "fn": len(ground_truth) - tp,
            "gt_words": len(ground_truth), "care_proposals": len(care),
            "ignored_prediction_ids": ignored, "matched_gt_indices": sorted(matched)}


def aggregate(rows, expected_images, expected_words=None):
    if len(rows) != expected_images or len({r["sample_id"] for r in rows}) != expected_images:
        raise ValueError("Missing or duplicate TEST samples; no omitted denominator allowed")
    if any(r.get("status") != "success" for r in rows):
        raise ValueError("Failed sample prevents aggregate results")
    totals = {k: sum(r[k] for r in rows) for k in ("tp", "tn", "fp", "fn", "gt_words", "care_proposals")}
    if expected_words is not None and totals["gt_words"] != expected_words:
        raise ValueError("Official TEST care-word denominator changed")
    tp, fp, fn = [totals[k] for k in ("tp", "fp", "fn")]
    return {**totals, "images": len(rows), "failure_count": 0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "word_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy_tn0": tp / (tp + fp + fn) if tp + fp + fn else None}


def qualify(args, cfg, source, preprocessing):
    if sha256_file(args.test_manifest) != cfg["test_manifest_sha256"]:
        raise ValueError("Frozen official TEST manifest checksum changed")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != cfg["test_images"]:
        raise ValueError("Expected official Total-Text TEST-300, not a TRAIN holdout")
    reports = {}
    for kind in ("reference", "text", "kernel"):
        root = getattr(args, kind + "_run").resolve()
        path = root / "run_manifest.json"
        report = json.loads(path.read_text())
        contract = report["run_contract"]
        checkpoint = root / report["checkpoint"]["path"]
        if (report.get("status") != "SUCCESS"
                or report.get("result_class") != "approximate reproduction"
                or report["checkpoint"]["sha256"] != source[kind + "_checkpoint_sha256"]
                or sha256_file(checkpoint) != source[kind + "_checkpoint_sha256"]
                or contract["validation_ids_sha256"] != source["validation_ids_sha256"]
                or contract["resolved_preprocessing"] != preprocessing):
            raise ValueError("Frozen checkpoint, TRAIN split or preprocessing changed: " + kind)
        for name, digest in contract["source_sha256"].items():
            if sha256_file(REPO / name) != digest:
                raise ValueError("Historical scientific source changed: " + name)
        reports[kind] = {"run_root": str(root), "checkpoint": str(checkpoint),
                         "run_manifest_sha256": sha256_file(path), "report": report}
    if (reports["reference"]["report"]["run_contract"]["branch"] != "control"
            or reports["text"]["report"]["run_contract"]["branch"] != "tversky768"
            or reports["kernel"]["report"]["run_contract"]["variant"] != "text_kernel"):
        raise ValueError("Incorrect source model branches")
    for sample in manifest.samples:
        for relative, digest in ((sample.image_path, sample.image_sha256),
                                 (sample.validity_annotation_path, sample.validity_annotation_sha256)):
            if relative is None or digest is None or sha256_file(args.data_root / relative) != digest:
                raise ValueError("Missing or changed TEST RGB/validity: " + sample.sample_id)
        if not _annotation_path(sample, args.data_root).is_file():
            raise FileNotFoundError("Missing TEST annotation: " + sample.sample_id)
    return manifest, reports


def infer_source(kind, item, samples, args, source, preprocessing, output):
    contract = item["report"]["run_contract"]
    settings = contract.get("resolved_config", contract.get("config"))
    kwargs = {"skip_floor": settings["skip_floor"], "encoder_pretrained": False}
    if kind == "reference":
        model = WordResearchAGUNet("control", embedding_dimensions=settings["embedding"]["dimensions"], **kwargs)
    elif kind == "kernel":
        model = PracticalTotalTextWordKernelAGUNet(**kwargs)
    else:
        model = PracticalTotalTextHighResAGUNet(**kwargs)
    payload = load_checkpoint(item["checkpoint"], model=model,
        expected_architecture=model.architecture_identity, expected_run_contract=contract)
    if payload["epoch"] != item["report"]["checkpoint"]["epoch"]:
        raise ValueError("Checkpoint epoch differs from selected run")
    del payload
    device = torch.device(args.device)
    size = contract["config"]["branches"]["tversky768"]["input_size"] if kind == "text" else preprocessing["input_size"][0]
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    started = time.monotonic()
    try:
        model.to(device).eval()
        with torch.inference_mode():
            for i, sample in enumerate(samples, 1):
                rgb, _ = _rgb(sample, args.data_root)
                values = model(image_tensor(rgb, size, preprocessing).to(device)).sigmoid().cpu().numpy()[0]
                channels = 1 if kind == "text" else 2
                if values.shape != (channels, size // 2, size // 2) or not np.isfinite(values).all():
                    raise ValueError("Unexpected model probability shape/values")
                path = output / "maps" / kind / (sample.sample_id.replace(":", "__") + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, probabilities=values)
                if i % 10 == 0 or i == len(samples):
                    print(f"[{kind}] {i}/{len(samples)} TEST images", flush=True)
    except BaseException as error:
        _atomic_json(output / (kind + "_failure.json"), {"status": "FAILED", "type": type(error).__name__,
            "message": str(error), "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        raise
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"source": kind, "model_set": "one U-Net loaded at a time", "device": str(device),
            "precision": "float32", "batch_size": 1, "input_shape": [1, 3, size, size],
            "seconds": time.monotonic() - started, "oom": False,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def run(args):
    cfg, source, preprocessing = read_config(args.config, args.mode)
    if args.limit is not None and not 1 <= args.limit <= cfg["test_images"]:
        raise ValueError("Partial smoke limit must be in 1..300")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(cfg["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, reports = qualify(args, cfg, source, preprocessing)
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "test_images": len(manifest.samples),
                          "models": cfg["models"], "result_class": cfg["result_class"]}, indent=2))
        return
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run on the GPU host or select --device cpu explicitly.")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing results are preserved: " + str(output))
    samples = list(manifest.samples[:args.limit] if args.limit is not None else manifest.samples)
    partial = len(samples) != cfg["test_images"]
    protocol = {"reproduction_mode": args.mode, "result_class": "partial-dataset result" if partial else cfg["result_class"],
        "claim_label": cfg["result_class"], "split": "official_TEST_300", "partial": partial,
        "config": cfg, "config_sha256": sha256_file(args.config), "frozen_localization": source,
        "preprocessing": preprocessing, "source_runs": reports, "data_root": str(args.data_root.resolve()),
        "test_manifest": str(args.test_manifest.resolve()), "test_manifest_sha256": sha256_file(args.test_manifest),
        "sample_ids": [s.sample_id for s in samples], "device": args.device, "cpu_threads": args.cpu_threads,
        "software": runtime_environment(), "tf32": False,
        "evaluation": "global micro maximum one-to-one original RGB bbox IoU>=0.50; geometry only",
        "empty_policy": "empty proposal precision/recall/F1=0; accuracy undefined (null) if TP+FP+FN=0",
        "source_sha256": {name: sha256_file(REPO / name) for name in (
            "scripts/evaluate_unet_word_detection_test.py", "localize/word_crop_refinements.py",
            "localize/word_multitask.py", "recognition/scene_text_v1_evaluation.py",
            "scripts/evaluate_totaltext_highres_contour_padding.py")}}
    output.mkdir(parents=True)
    _atomic_json(output / "resolved_configuration.json", protocol)
    try:
        memory = []
        for kind, item in reports.items():
            memory.append(infer_source(kind, item, samples, args, source, preprocessing, output))
            _atomic_json(output / (kind + "_memory.json"), memory[-1])
        _atomic_json(output / "memory.json", memory)
        totals = {name: [] for name in cfg["models"]}
        for i, sample in enumerate(samples, 1):
            key = sample.sample_id.replace(":", "__")
            rgb, _ = _rgb(sample, args.data_root)
            image_wh = (rgb.shape[1], rgb.shape[0])
            maps = {}
            for kind in reports:
                with np.load(output / "maps" / kind / (key + ".npz"), allow_pickle=False) as saved:
                    maps[kind] = saved["probabilities"]
            inputs = {"text_kernel": (maps["reference"], source["reference_threshold"]),
                      "hybrid384": (np.stack([maps["text"][0],
                          resize_probability(maps["kernel"][1], maps["text"].shape[-1])]), source["text_threshold"])}
            frozen = {}
            for name, (values, threshold) in inputs.items():
                canvas = letterbox_canvas_mask(image_wh, native_size=values.shape[-1]).numpy()
                labels, proposals, audit = localize(values, canvas, image_wh, source, threshold)
                folder = output / name / "samples" / key
                folder.mkdir(parents=True)
                np.savez_compressed(folder / "instances.npz", labels=labels)
                _atomic_json(folder / "predictions.json", {"sample_id": sample.sample_id,
                    "reproduction_mode": args.mode, "result_class": protocol["result_class"],
                    "claim_label": cfg["result_class"],
                    "image_sha256": sample.image_sha256, "native_size": values.shape[-1],
                    "text_threshold": threshold, "proposals": proposals, "audit": audit,
                    "instances_sha256": sha256_file(folder / "instances.npz")})
                frozen[name] = (labels, proposals, folder)
            # Annotation/ignore semantics enter only after both arms are frozen.
            gt, annotation_sha = _ground_truth(sample, args.data_root, image_wh)
            with Image.open(args.data_root / sample.validity_annotation_path) as opened:
                validity = opened.convert("L")
                if validity.size != image_wh or not set(np.unique(np.asarray(validity))).issubset({0, 255}):
                    raise ValueError("Invalid TEST care/ignore mask")
                validity = letterbox_geometry(validity, fill=0)
                for name, (labels, proposals, folder) in frozen.items():
                    resized = TF.resize(validity, list(labels.shape), interpolation=InterpolationMode.NEAREST)
                    care = TF.pil_to_tensor(resized)[0].numpy() > 0
                    row = {"sample_id": sample.sample_id, "status": "success", "ground_truth": gt,
                        "reproduction_mode": args.mode, "result_class": protocol["result_class"],
                        "claim_label": cfg["result_class"],
                        "annotation_sha256": annotation_sha, "validity_sha256": sample.validity_annotation_sha256,
                        **score_detection(labels, proposals, care, gt, cfg["iou_threshold"])}
                    totals[name].append(row)
                    _atomic_json(folder / "evaluation.json", row)
            if i % 10 == 0 or i == len(samples):
                print(f"[detection] {i}/{len(samples)} TEST images", flush=True)
        result = {"status": "SUCCESS", "reproduction_mode": args.mode,
            "result_class": protocol["result_class"], "claim_label": cfg["result_class"],
            "split": protocol["split"], "partial": partial, "tn_assumption": 0,
            "resolved_configuration_sha256": sha256_file(output / "resolved_configuration.json"),
            "models": {name: aggregate(rows, len(samples), None if partial else cfg["test_care_words"])
                       for name, rows in totals.items()}}
        _atomic_json(output / "results.json", result)
        print(json.dumps(result, indent=2), flush=True)
        for name, values in result["models"].items():
            accuracy = values["accuracy_tn0"]
            accuracy_text = "undefined" if accuracy is None else f"{100 * accuracy:.2f}%"
            print(f"{name}: F1={100 * values['word_f1']:.2f}%, accuracy(TN=0)={accuracy_text}, "
                  f"TP={values['tp']}, FP={values['fp']}, FN={values['fn']}", flush=True)
        print("Results: " + str(output / "results.json"), flush=True)
    except BaseException as error:
        _atomic_json(output / "failure.json", {"status": "FAILED", "type": type(error).__name__,
            "message": str(error), "result_class": protocol["result_class"],
            "incomplete": True, "oom": isinstance(error, torch.cuda.OutOfMemoryError)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("reference-run", "text-run", "kernel-run", "data-root", "test-manifest", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Explicit partial-dataset smoke; never a full TEST result")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
