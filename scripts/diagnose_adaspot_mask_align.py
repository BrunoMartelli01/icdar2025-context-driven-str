"""Frozen full-TEST feature Mask-Align qualification and matched RGB proxy."""

import argparse
from collections import Counter
import gc
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_mask_align import (MaskAlignOptions, mask_align_geometry,
    native_grid_to_rgb, sample_mask_aligned)
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import memory
from scripts.select_adaspot_gap_ocr import recognize, score
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import load_segmentation_manifest
from segmentation.training import seed_everything


def state_digest(model):
    digest = sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_config(path, mode):
    c = json.loads(path.read_text())
    if (mode != "approximate" or c["reproduction_mode"] != mode
            or REPO / "configs/approximate" not in path.parents
            or c["selected_source_arm"] != "full_selected"
            or c["light_grid"] != [8, 32] or c["deep_grid"] != [16, 64]
            or c["rgb_proxy_grid"] != [32, 128] or c["image_batch_size"] < 1):
        raise ValueError("Explicit isolated approximate Mask-Align configuration required")
    MaskAlignOptions(**c["geometry"])
    return c


def make_report(result):
    lines = ["# Mask-Align v1", "", "**Approximate reproduction; frozen RGB geometry diagnostic, not trained feature OCR.**", "",
        "Actual feature sampler qualified on all300 TEST images: F64x256x256 → 8x32 and16x64, FP32 CUDA.",
        "Explicit deterministic CPU bilinear backward preserves gradients to CUDA features. Segmentation weights remain frozen.", "",
        "| RGB view | exact words /2204 | precision | recall | Hmean | recovered / damaged vs original |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, values in result["arms"].items():
        a, change = values["ocr"]["aggregate"], values["changes_vs_original"]
        lines.append(f"| {name} | {a['exact_casefold_iou_ge_50']} | {a['precision']:.4%} | {a['recall']:.4%} | {a['hmean']:.4%} | {change['recovered']} / {change['damaged']} |")
    lines.extend(["", "All views retain the same2624 care proposals and original matching boxes; geometry matches remain1612.",
        "Exact Unicode casefold with punctuation retained, one-to-one bboxIoU>=.5; empty/duplicate predictions stay in denominator.",
        "Primary300 includes known TRAIN/TEST identical RGB578 and656; secondary298 excludes exactly these two for all views.",
        "TEST was historically explored. No TEST selection, threshold fitting, segmentation training, light/deep head or gate training.",
        "No Table5, official polygon, strict reproduction, shared-feature OCR accuracy or shared-backbone speed claim.", "",
        "Fallbacks: `" + json.dumps(result["geometry_counts"], sort_keys=True) + "`.", "",
        "Secondary hash-disjoint298 /2187 words:"])
    for name, values in result["hash_disjoint_test_subset"]["arms"].items():
        lines.append(f"- {name}: {values['aggregate']['exact_casefold_iou_ge_50']}/2187.")
    return "\n".join(lines) + "\n"


def run(args):
    config_path, source, out = args.config.resolve(), args.source_run.resolve(), args.output_root.resolve()
    c = validate_config(config_path, args.mode)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA GPU required; no CPU inference fallback")
    p = json.loads((source / "protocol.json").read_text())
    selection = json.loads((source / "selected_model.json").read_text())
    audit = json.loads((source / "independent_audit.json").read_text())
    if audit["status"] != "PASS" or selection["selected_epoch"] != 1:
        raise ValueError("Audited fixed epoch1 full-data source required")
    check_hash(source / "results.json", audit["result_sha256"])
    check_hash(source / "results.json", selection["results"]["sha256"])
    # Check inherited evidence BEFORE adding new paths: never replace a failed hash.
    hashes = dict(p["source_hashes"])
    for path, expected in hashes.items():
        check_hash(Path(path), expected)
    for item in (selection["source_checkpoint"], selection["selected_gap_checkpoint"]):
        check_hash(Path(item["path"]), item["sha256"])
    new_sources = [config_path, Path(__file__), REPO / "models/adaspot_mask_align.py",
        REPO / "segmentation/word_centerline.py", REPO / "tests/models/test_adaspot_mask_align.py",
        REPO / "scripts/audit_adaspot_mask_align.py", REPO / "scripts/audit_adaspot_phase0.py",
        REPO / "scripts/evaluate_totaltext_highres_contour_padding.py", REPO / "segmentation/training.py"]
    new_sources += [source / n for n in ("selected_model.json", "protocol.json", "results.json",
        "independent_audit.json", "test_inputs.json", "test_readings.json", "test_artifact_manifest.json")]
    for path in new_sources:
        hashes[str(path.resolve())] = sha256_file(path)
    if any(out == root or root in out.parents for root in (REPO, source, Path(p["test_root"]), Path(p["word_root"]))):
        raise ValueError("New output must be external to repository and inputs")
    out.mkdir(parents=True, exist_ok=False)
    args._output_created = True
    torch.cuda.init()
    torch.set_num_threads(c["cpu_threads"])
    cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = {"schema_version": 1, "mode": args.mode, "result_class": c["claim_label"],
        "resolved_config": c, "source_run": str(source), "source_hashes": hashes,
        "frozen_selection": selection, "parseq_source": p["parseq_source"],
        "parseq_archive": p["parseq_archive"], "parseq_checkpoint": p["parseq_checkpoint"],
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "dtype": "float32", "platform": platform.platform(),
        "test_historically_explored": True, "test_used_for_selection": False,
        "deterministic_algorithms": True, "tf32": False,
        "gpu_before": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()}
    write_json(out / "protocol.json", protocol)
    manifest = load_segmentation_manifest(Path(p["test_manifest"]), require_checksums=True)
    samples = {s.sample_id: s for s in manifest.samples}
    original = json.loads((source / "test_inputs.json").read_text())[c["selected_source_arm"]]
    metadata = {s["sample_id"]: s for s in json.loads((source / "test_artifact_manifest.json").read_text())}
    if (len(original) != c["expected_images"] or len(samples) != len(original)
            or len({s['sample_id'] for s in original}) != len(original)
            or set(samples) != {s["sample_id"] for s in original}
            or sum(len(s["ground_truth"]) for s in original) != c["expected_words"]
            or sum(len(s["predictions"]) for s in original) != c["expected_proposals"]):
        raise ValueError("Frozen TEST population differs")
    transform = _transform(p["resolved_preprocessing"], "letterbox", train=False)
    model = AdaSpotGapAGUNet(**selection["source_model_kwargs"])
    # Trusted, SHA256-verified completed local checkpoint contains run metadata.
    payload = torch.load(selection["source_checkpoint"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    model.eval().requires_grad_(False).to(device)
    before = state_digest(model)
    torch.cuda.reset_peak_memory_stats(device)
    arms = {"original_axis": original, **{name: [] for name in ("bilinear_axis", "rotated", "mask_align")}}
    jobs, receipts, counts, gradient = {}, [], Counter(), None
    options = MaskAlignOptions(**c["geometry"])
    started = time.monotonic()
    def freeze(rgb):
        digest = crop_array_hash(rgb)
        path = out / "crops" / (digest + ".png")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
        jobs[digest] = path
        return digest
    for index, row in enumerate(original):
        sid, sample = row["sample_id"], samples[row["sample_id"]]
        image_path = Path(p["test_root"]) / sample.image_path
        meta = metadata[sid]
        check_hash(image_path, meta["image_sha256"])
        check_hash(Path(meta["annotation_path"]), meta["annotation_sha256"])
        hashes[str(image_path)] = meta["image_sha256"]
        hashes[meta["annotation_path"]] = meta["annotation_sha256"]
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        folder = source / "test_predictions" / sid.replace(":", "__")
        for name in ("source.npz", "full_selected_labels.npz", "full_selected.npz"):
            path = folder / name
            hashes[str(path)] = check_hash(path, meta["prediction_sha256"][str(path)])
        with np.load(folder / "full_selected_labels.npz", allow_pickle=False) as saved:
            labels = saved["labels"]
        with np.load(folder / "source.npz", allow_pickle=False) as saved:
            expected_probabilities = saved["probabilities"]
        with torch.no_grad():
            evidence = model.forward_evidence(_tensor(rgb, transform).to(device))
        features = evidence["features"].detach()
        probabilities = evidence["words"].sigmoid()[0].cpu().numpy()
        difference = float(np.abs(probabilities - expected_probabilities).max())
        if features.shape != (1, 64, 256, 256) or not np.allclose(probabilities, expected_probabilities, atol=1e-6, rtol=0):
            raise ValueError("Original Text/Kernel feature checkpoint no longer replays frozen evidence")
        if gradient is None and row["predictions"]:
            own = labels == row["predictions"][0]["instance_id"]
            geometry = mask_align_geometry(own, view="mask", options=options)
            grid = torch.from_numpy(geometry.grid(tuple(c["light_grid"]))[None]).to(device)
            differentiable = features.clone().requires_grad_()
            aligned = sample_mask_aligned(differentiable, grid, deterministic_cpu_backward=c["deterministic_cpu_backward"])
            aligned.square().mean().backward()
            grad = differentiable.grad
            if grad is None or not grad.is_cuda or not torch.isfinite(grad).all() or not torch.any(grad != 0):
                raise RuntimeError("Real CUDA feature gradient qualification failed")
            gradient = {"feature_device": str(features.device), "output_device": str(aligned.device),
                "sampling_backward_device": "cpu", "gradient_device": str(grad.device),
                "nonzero_feature_gradients": int((grad != 0).sum()), "gradient_l1": float(grad.abs().sum()),
                "input_shape": list(features.shape), "output_shape": list(aligned.shape)}
            del differentiable, aligned, grad
        outputs = {name: [] for name in ("bilinear_axis", "rotated", "mask_align")}
        saved_light, saved_deep, ids, geometries = [], [], [], []
        rgb_tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1)[None].float().to(device)
        for proposal in row["predictions"]:
            x0, y0, x1, y1 = proposal["bbox"]
            baseline = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            if freeze(baseline) != proposal["crop_rgb_sha256"]:
                raise ValueError("Original RGB crop differs from frozen baseline")
            own = labels == proposal["instance_id"]
            geometry = None
            for arm, view in (("bilinear_axis", "axis"), ("rotated", "rotated"), ("mask_align", "mask")):
                fitted = mask_align_geometry(own, view=view, options=options)
                native_grid = fitted.grid(tuple(c["rgb_proxy_grid"]))
                rgb_grid = native_grid_to_rgb(native_grid, (rgb.shape[1], rgb.shape[0]), labels.shape)
                with torch.no_grad():
                    crop = sample_mask_aligned(rgb_tensor, torch.from_numpy(rgb_grid[None]).to(device), deterministic_cpu_backward=False)
                crop = np.ascontiguousarray(crop[0].permute(1, 2, 0).cpu().numpy().round().clip(0, 255).astype(np.uint8))
                outputs[arm].append({**proposal, "crop_rgb_sha256": freeze(crop), "alignment": fitted.audit})
                if view == "mask":
                    geometry = fitted
            counts[geometry.audit["fallback_reason"] or "cubic_centerline"] += 1
            ids.append(proposal["instance_id"])
            geometries.append({"instance_id": proposal["instance_id"], **geometry.audit})
            for resolution, destination in (("light_grid", saved_light), ("deep_grid", saved_deep)):
                grid = torch.from_numpy(geometry.grid(tuple(c[resolution]))[None]).to(device)
                with torch.no_grad():
                    aligned = sample_mask_aligned(features, grid, deterministic_cpu_backward=False)
                if not torch.isfinite(aligned).all():
                    raise RuntimeError("Nonfinite actual aligned U-Net features")
                destination.append(aligned[0].cpu().numpy())
        for arm in outputs:
            arms[arm].append({**row, "predictions": outputs[arm]})
        target = out / "aligned_features" / (sid.replace(":", "__") + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        def stack(values, shape):
            return np.stack(values) if values else np.empty((0, 64, *shape), np.float32)
        np.savez_compressed(target, instance_ids=np.asarray(ids, np.int32),
            light=stack(saved_light, c["light_grid"]), deep=stack(saved_deep, c["deep_grid"]))
        receipts.append({"sample_id": sid, "image_path": str(image_path), "image_sha256": meta["image_sha256"],
            "feature_path": str(target), "feature_sha256": sha256_file(target), "instance_ids": ids,
            "probability_max_abs_error": difference, "geometries": geometries})
        del evidence, features, rgb_tensor
        if (index + 1) % 25 == 0:
            print(f"[features/geometry] {index+1}/{len(original)} images; {sum(counts.values())} proposals; {time.monotonic()-started:.1f}s", flush=True)
    after = state_digest(model)
    if before != after:
        raise RuntimeError("Frozen source weights or BN buffers changed")
    feature_memory = memory(device, "frozen Text/Kernel AG-UNet + Mask-Align + RGB geometry proxy", [1, 3, 512, 512])
    feature_geometry_seconds = time.monotonic() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    write_json(out / "feature_receipts.json", receipts)
    write_json(out / "frozen_inputs.json", arms)
    write_json(out / "gradient_qualification.json", gradient)
    write_json(out / "protocol_final.json", protocol)
    readings = json.loads((source / "test_readings.json").read_text())
    ocr_memory = recognize(jobs, readings, protocol, c, device, out, "rgb_proxy")
    evaluated = {name: score(rows, readings) for name, rows in arms.items()}
    reference = json.loads((source / "results.json").read_text())["arms"]["full_selected"]["ocr"]
    if evaluated["original_axis"] != reference:
        raise ValueError("Complete per-image baseline and strings must replay unchanged")
    if evaluated["original_axis"]["aggregate"]["exact_casefold_iou_ge_50"] != c["expected_baseline_exact"]:
        raise ValueError("Baseline count differs")
    duplicates = {x["test_id"] for x in p["resolved_config"]["known_official_rgb_overlap"]}
    subset = {name: score([r for r in rows if r["sample_id"] not in duplicates], readings) for name, rows in arms.items()}
    result = {"status": "SUCCESS", "result_class": c["claim_label"], "scope": "feature sampler + frozen RGB geometry proxy; no trained feature OCR",
        "arms": {name: {"ocr": values, "changes_vs_original": paired_changes(evaluated["original_axis"], values)} for name, values in evaluated.items()},
        "changes_mask_vs_bilinear_axis": paired_changes(evaluated["bilinear_axis"], evaluated["mask_align"]),
        "hash_disjoint_test_subset": {"images": len(next(iter(subset.values()))["per_image"]),
            "excluded_exact_duplicate_test_ids": sorted(duplicates), "arms": subset},
        "geometry_counts": dict(counts), "gradient": gradient, "source_state_before": before, "source_state_after": after,
        "feature_memory": feature_memory, "ocr_memory": ocr_memory, "feature_geometry_seconds": feature_geometry_seconds}
    for path, expected in hashes.items():
        check_hash(Path(path), expected)
    write_json(out / "results.json", result)
    (out / "report.md").write_text(make_report(result))
    print(json.dumps({k: v["ocr"]["aggregate"] for k, v in result["arms"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if getattr(args, "_output_created", False):
            write_json(args.output_root / "failure.json", {"status": "FAILED", "error": repr(error),
                "oom": isinstance(error, torch.cuda.OutOfMemoryError), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
