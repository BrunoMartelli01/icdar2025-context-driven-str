"""Frozen word instances -> axis/oriented/Bézier RGB -> qualified PARSeq."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from localize.unet_detection_methods import rectified_word_crop
from models.unet_detection_methods import BRANCHES
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_unet_word_research_v2_ocr import cached_text, cache_record, score_predictions
from scripts.evaluate_word_multitask_crops import ratios
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_detection_methods import (DEFAULT_CONFIG, REPO, bind, read_config,
    verify_completed, verify_exports, SCIENCE_FILES)
from scripts.run_unet_word_research_v2 import backend_state
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.training import seed_everything


def qualify_report(directory, c, smoke):
    if smoke:
        saved = json.loads((directory / "smoke.json").read_text())
        if saved["result_class"] != "smoke-test result" or saved["run_contract"]["execution_kind"] != "smoke":
            raise ValueError("Expected separately labeled smoke")
        report = saved["validation"]
        contract = saved["run_contract"]
        for row in report["per_image"]:
            folder = directory / "validation_exports" / row["sample_id"].replace(":", "__")
            for name, key in (("maps.npz", "maps_sha256"), ("proposals.json", "proposals_sha256"), ("instances.png", "labels_sha256")):
                if sha256_file(folder / name) != row[key]:
                    raise ValueError("Smoke export changed")
    else:
        saved = json.loads((directory / "run_manifest.json").read_text())
        report = json.loads((directory / "validation.json").read_text())
        contract = saved["run_contract"]
        if directory.name == "reference":
            if (saved.get("status") != "SUCCESS" or not saved.get("training_complete")
                    or saved["checkpoint"]["sha256"] != sha256_file(Path(saved["checkpoint"]["path"]))):
                raise ValueError("Frozen reference checkpoint changed")
        else:
            verify_completed(directory, saved, contract, c["epochs"])
        verify_exports(directory, report, contract)
        if report["checkpoint_sha256"] != saved["checkpoint"]["sha256"]:
            raise ValueError("Export checkpoint identity differs")
    if (set(contract["source_sha256"]) != set(SCIENCE_FILES)
            or any(sha256_file(REPO / f) != h for f, h in contract["source_sha256"].items())):
        raise ValueError("Scientific code changed after geometry export")
    return report, contract


def export_review(output, results, frozen, records):
    destination = output / "review"
    destination.mkdir(exist_ok=True)
    # Fixed sample order, not GT-based best-example selection.
    examples = []
    axis = "reference/axis"
    for sample in results[axis]["per_image"][:8]:
        sid = sample["sample_id"]
        source = frozen[axis][sid]
        with Image.open(source["image_path"]) as image:
            canvas = image.convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(canvas)
        sw, sh = source["image_wh"]
        for p in sample["predictions"]:
            x0, y0, x1, y1 = p["bbox"]
            draw.rectangle((x0 * 640 / sw, y0 * 480 / sh, x1 * 640 / sw, y1 * 480 / sh), outline="#00dd66", width=2)
        filename = sid.replace(":", "__") + ".png"
        canvas.save(destination / filename)
        examples.append(f'<figure><img src="{filename}" width="640"><figcaption>{sid}</figcaption></figure>')
    (destination / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Word detection comparison</title>'
        '<p>Fixed-order source word boxes. Complete crop views and raw OCR are in ../ocr_inputs and ../comparison.json.</p>' + "\n".join(examples))


def run(args):
    c, _ = read_config(args.config, args.mode)
    output, root = args.run_root.resolve(), args.word_root.resolve()
    if args.batch_size < 1:
        raise ValueError("Positive OCR batch size required")
    torch.set_num_threads(c["cpu_threads"])
    cv2.setNumThreads(c["opencv_threads"])
    seed_everything(c["seed"], deterministic_algorithms=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use an explicit CPU smoke output")
    if (sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or sha256_file(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq checkpoint/runtime bytes differ")
    manifest_path = root / c["train_manifest"]
    if sha256_file(manifest_path) != c["word_manifest_sha256"]:
        raise ValueError("TotalText TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    _, val_indices, split_sha = _split_indices(manifest, c["seed"], c["validation_count"])
    if split_sha != c["validation_ids_sha256"]:
        raise ValueError("Frozen validation split changed")
    by_id = {s.sample_id: s for s in manifest.samples}
    expected_ids = {manifest.samples[i].sample_id for i in val_indices}
    branches = args.branches
    if len(set(branches)) != len(branches):
        raise ValueError("Duplicate comparison branches")
    if "reference" not in branches:
        raise ValueError("A frozen reference is required for paired comparison")
    cache = output / "parseq_cache"
    cache.mkdir(exist_ok=True)
    execution = {"device": str(device), "batch_size": args.batch_size, "dtype": "float32",
                 "environment": experiment_environment(device), "backend": backend_state(),
                 "runtime_revision": PARSEQ_RUNTIME_REVISION, "decode_ar": True, "refine_iters": 1,
                 "checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256, "archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                 "recognizer_sha256": sha256_file(REPO / "recognition/parseq.py"), "cpu_threads": c["cpu_threads"]}
    bind(cache / "execution.json", execution)
    frozen, unique, source_hashes, geometries, population = {}, {}, {}, {}, None
    # All proposed RGB views are frozen before original annotations are opened.
    for branch in branches:
        directory = output / branch
        report, contract = qualify_report(directory, c, args.smoke)
        resolved = contract["resolved_config"]
        if args.smoke:
            resolved = {**resolved, "batch_size": c["batch_size"], "num_workers": c["num_workers"]}
        if resolved != c or contract["word_root"] != str(root):
            raise ValueError("Geometry configuration/input root differs")
        ids = [r["sample_id"] for r in report["per_image"]]
        if (len(ids) != len(set(ids)) or not set(ids).issubset(expected_ids)
                or not args.smoke and set(ids) != expected_ids):
            raise ValueError("Incomplete/duplicated validation population")
        if population is None:
            population = ids
        if ids != population:
            raise ValueError("Branches must evaluate exactly the same image order")
        source_hashes[branch] = sha256_file(directory / ("smoke.json" if args.smoke else "validation.json"))
        geometries[branch] = {k: report[k] for k in ("native", "original", "pixel")}
        for view in c["crop_views"]:
            frozen[branch + "/" + view] = {}
        for row in report["per_image"]:
            sid = row["sample_id"]
            folder = directory / "validation_exports" / sid.replace(":", "__")
            record = json.loads((folder / "proposals.json").read_text())
            sample = by_id[sid]
            image_path = root / sample.image_path
            if (record["sample_id"] != sid or record["image_sha256"] != sample.image_sha256
                    or sha256_file(image_path) != sample.image_sha256):
                raise ValueError("Frozen RGB/proposal sample identity differs")
            with Image.open(image_path) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            with np.load(folder / "maps.npz", allow_pickle=False) as maps:
                labels = maps["labels"].copy()
            views = {v: [] for v in c["crop_views"]}
            for proposal in record["proposals"]:
                for view in c["crop_views"]:
                    crop, geometry = rectified_word_crop(rgb, labels, proposal, view, c)
                    digest = crop_array_hash(crop)
                    destination = output / "ocr_inputs" / branch / sid.replace(":", "__") / (str(proposal["instance_id"]) + "_" + view + ".png")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        with Image.open(destination) as saved:
                            if crop_array_hash(np.asarray(saved.convert("RGB"))) != digest:
                                raise ValueError("Previously frozen rectified crop changed")
                    else:
                        Image.fromarray(crop).save(destination)
                    unique.setdefault(digest, destination)
                    views[view].append({**proposal, "crop_rgb_sha256": digest, "crop_path": str(destination), "rectification": geometry})
            for view, proposals in views.items():
                frozen[branch + "/" + view][sid] = {"sample_id": sid, "image_path": str(image_path),
                    "image_wh": (rgb.shape[1], rgb.shape[0]), "image_sha256": sample.image_sha256,
                    "proposals": proposals, "ignored_ids": row["ignored_instance_ids"]}
    protocol = {"reproduction_mode": "approximate", "result_class": "smoke-test result" if args.smoke else "approximate reproduction",
                "resolved_config": c, "branches": branches, "sample_ids": population, "cache_execution": execution,
                "source_report_sha256": source_hashes, "source_sha256": {f: sha256_file(REPO / f) for f in SCIENCE_FILES},
                "execution_choices": [{"classification": "HARDWARE_DEVIATION", "component": "recognition_device_and_batch",
                    "rationale": f"Explicit {device} FP32/batch{args.batch_size}; PARSeq loaded after segmentation processes exit",
                    "expected_effect": "CPU result is not a GPU fit/timing qualification; no automatic output-changing fallback"}],
                "crop_geometry": "All views retain original axis-aligned instance bbox for IoU; no GT selects a crop",
                "matching": "original RGB maximum one-to-one bbox IoU>=.5 and exact Unicode casefold; P/S secondary"}
    bind(output / "ocr_protocol.json", protocol)
    records, recognizer = {}, None
    for digest in unique:
        path = cache / (digest + ".json")
        if path.exists():
            records[digest] = json.loads(path.read_text()); cached_text(records[digest], digest)
    try:
        if device.type == "cuda": _prepare_cuda_memory_tracking(device)
        jobs = sorted(unique.items())
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]
            if all(digest in records for digest, _ in batch):
                continue
            # Replay the whole declared batch after interruption to preserve
            # batch composition, including any members already in the cache.
            crops = []
            for digest, path in batch:
                with Image.open(path) as image:
                    crop = np.asarray(image.convert("RGB"))
                if crop_array_hash(crop) != digest:
                    raise ValueError("Frozen recognition pixels changed")
                crops.append(crop)
            if recognizer is None:
                recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
                    checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                    runtime_source_archive=args.parseq_source_archive, device=str(device), decode_ar=True, refine_iters=1))
            predictions = recognizer.recognize_candidate_batch(crops)
            if len(predictions) != len(batch):
                raise RuntimeError("PARSeq output count differs")
            for (digest, _), prediction in zip(batch, predictions):
                record = cache_record(prediction, digest, recognizer.config)
                if digest in records:
                    if cached_text(record, digest) != cached_text(records[digest], digest):
                        raise ValueError("Deterministic replay changed a cached PARSeq reading")
                else:
                    records[digest] = record; bind(cache / (digest + ".json"), record)
            print(f"[PARSeq views] {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)
        peak = {"device": str(device), "model_set": ["qualified PARSeq"], "dtype": "float32", "batch_size": args.batch_size,
                "input_shape_max": [args.batch_size, 3, 32, 128],
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
    except Exception as error:
        _json(output / "ocr_failure.json", {"error_type": type(error).__name__, "message": str(error), "protocol": protocol,
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "memory": {"device": str(device), "model_set": ["qualified PARSeq"], "dtype": "float32", "batch_size": args.batch_size,
                "input_shape_max": [args.batch_size, 3, 32, 128],
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}})
        raise
    finally:
        if recognizer is not None: recognizer.release()
    results = {}
    for arm, samples in frozen.items():
        rows, totals, reasons = [], Counter(), Counter()
        for sid, sample in samples.items():
            gt, annotation_sha = _train_gt(by_id[sid], root, sample["image_wh"])
            predictions = [{**p, "text": cached_text(records[p["crop_rgb_sha256"]], p["crop_rgb_sha256"])}
                           for p in sample["proposals"] if p["instance_id"] not in sample["ignored_ids"]]
            counts, matches = score_predictions(predictions, gt)
            totals.update(counts)
            reasons.update(p["rectification"]["reason"] for p in predictions)
            rows.append({"sample_id": sid, "counts": counts, "ground_truth": gt, "annotation_sha256": annotation_sha,
                         "predictions": predictions, "matched_gt_indices": {k: sorted(v) for k, v in matches.items()}})
        if not args.smoke and totals["gt_words"] != c["expected_validation_words"]:
            raise ValueError("Full OCR care-word denominator changed")
        results[arm] = {"aggregate": ratios(dict(totals)), "per_image": rows, "rectification_reasons": dict(reasons),
                        "segmentation": geometries[arm.split("/")[0]]}
    for arm, result in results.items():
        comparisons = {"reference/axis", arm.split("/")[0] + "/axis"}
        if "control/axis" in results: comparisons.add("control/axis")
        changes = {}
        for baseline in sorted(comparisons):
            ref = {r["sample_id"]: set(r["matched_gt_indices"]["exact_casefold_iou_ge_50"]) for r in results[baseline]["per_image"]}
            gains = losses = 0
            for row in result["per_image"]:
                current = set(row["matched_gt_indices"]["exact_casefold_iou_ge_50"])
                gained, lost = sorted(current - ref[row["sample_id"]]), sorted(ref[row["sample_id"]] - current)
                row.setdefault("paired_changes", {})[baseline] = {"gained_gt_indices": gained, "lost_gt_indices": lost}
                gains += len(gained); losses += len(lost)
            changes[baseline] = {"gained": gains, "lost": losses, "net": gains - losses}
        result["paired_changes"] = changes
    result = {"status": "SUCCESS", "result_class": protocol["result_class"], "reproduction_mode": "approximate",
              "protocol": protocol, "results": results, "memory": peak, "unique_crop_count": len(unique),
              "local_bbox_proxy": True, "table5_claim": False, "target": "80percent localized and exact-casefold read / all care GT"}
    destination = output / "comparison.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["protocol"] != protocol or previous["results"] != results:
            raise ValueError("Previously completed OCR comparison differs")
        result = previous  # Preserve actual first-execution memory measurement.
    else:
        bind(destination, result)
    fields = ["arm", "correct", "gt_words", "outputs", "precision", "recall", "hmean", "gained", "lost",
              "localization_precision", "localization_recall", "localization_hmean", "merge_rate", "split_rate", "pixel_iou", "pixel_f1"]
    table = []
    for arm, part in results.items():
        m = part["aggregate"]; changes = part["paired_changes"]["reference/axis"]
        native = part["segmentation"]["native"]; pixel = part["segmentation"]["pixel"]
        table.append(dict(zip(fields, [arm, m["exact_casefold_iou_ge_50"], m["gt_words"], m["proposals"],
            m["exact_casefold_precision"], m["exact_casefold_recall_all_gt"], m["exact_casefold_f1"], changes["gained"], changes["lost"],
            m["geometry_precision"], m["geometry_recall_all_gt"], m["geometry_f1"], native["merge_rate"], native["split_rate"],
            pixel["foreground_iou_macro"], pixel["foreground_f1_macro"]])))
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    lines = ["# Word detection methods — approximate reproduction", "", "Exact casefold with punctuation retained, original bboxIoU>=.5; all care words; no GT crop selection.", "",
             "| Variant/view | Correct/all GT | Precision | Recall | Hmean | Gained/lost vs reference |", "|---|---:|---:|---:|---:|---|"]
    for row in table:
        lines.append(f"| {row['arm']} | {row['correct']}/{row['gt_words']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['hmean']:.2%} | +{row['gained']}/-{row['lost']} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    export_review(output, results, frozen, records)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["approximate"], required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("word-root", "run-root", "parseq-source", "parseq-source-archive", "parseq-checkpoint"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--branches", nargs="+", choices=BRANCHES, required=True)
    p.add_argument("--smoke", action="store_true")
    value = run(p.parse_args())
    print(json.dumps({"status": value["status"], "result_class": value["result_class"], "arms": len(value["results"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
