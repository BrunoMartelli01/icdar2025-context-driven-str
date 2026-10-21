"""Frozen text-recall maps -> alternative word crops -> real pinned PARSeq."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from localize.text_recall_crops import METHODS, prepare_recipes
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_totaltext_word_kernel_parseq_val import _save_labels
from scripts.evaluate_word_multitask_crops import (
    recognize, ratios, score_predictions, validate_ocr, verify_geometry_exports, write_json,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.word_multitask_evaluation import aggregate_word_metrics, render_word_maps


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/text_recall_crops_v1.json"
BRANCHES = ("control512", "tversky512", "control768", "tversky768")


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE config required; no namespace fallback")
    cfg = json.loads(path.read_text())
    if (cfg["schema_version"] != 1 or cfg["reproduction_mode"] != mode
        or cfg["result_class"] != "approximate reproduction" or tuple(cfg["branches"]) != BRANCHES
        or tuple(cfg["methods"]) != METHODS or cfg["threshold_policies"] != ["selected", "common"]
        or cfg["crop"] != {"padding": 0, "min_area": 0, "max_instances": None}
        or cfg["recognition"] != {"decode_ar": True, "refine_iters": 1, "batch_size": 8,
                                 "casefold": True, "symbol_normalized_secondary_metric": True}):
        raise ValueError("Unsupported frozen text-recall/PARSeq crop protocol")
    # Validate all localization paths without annotation inputs or model loading.
    prepare_recipes(np.zeros((16, 16)), np.zeros((2, 16, 16)), np.ones((16, 16), bool),
                    (16, 16), cfg, threshold=cfg["common_threshold"])
    return cfg


def completed_source(root):
    manifest = json.loads((root / "run_manifest.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    if (manifest["status"] != "SUCCESS" or manifest["result_class"] != "approximate reproduction"
        or manifest["run_contract"] != validation["run_contract"]
        or sha256_file(root / manifest["checkpoint"]["path"]) != manifest["checkpoint"]["sha256"]):
        raise ValueError("Incomplete or changed source checkpoint: " + str(root))
    for name, expected in manifest["run_contract"]["source_sha256"].items():
        if sha256_file(REPO / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    return manifest, validation


def load_sources(args, cfg):
    common = json.loads((args.run_root / "experiment_contract.json").read_text())
    comparison = json.loads((args.run_root / "comparison.json").read_text())
    if (common["reproduction_mode"] != "approximate" or comparison["status"] != "SUCCESS"
        or not comparison["matched_sample_order"]
        or cfg["common_threshold"] not in common["config"]["evaluation"]["thresholds"]):
        raise ValueError("Expected the complete matched text-recall comparison")
    target = Path(common["target_root"])
    dataset = common["config"]["dataset"]
    path = target / dataset["manifest"]
    if sha256_file(path) != dataset["manifest_sha256"]:
        raise ValueError("Qualified dataset manifest changed")
    manifest = load_segmentation_manifest(path, require_checksums=True)
    _, indices, fingerprint = _split_indices(manifest, common["config"]["training"]["seed"], dataset["validation_images"])
    ids = [manifest.samples[i].sample_id for i in indices]
    if (manifest.dataset != "total_text" or manifest.split_role != "train" or len(ids) != 125
        or fingerprint != dataset["validation_ids_sha256"]):
        raise ValueError("Frozen TRAIN holdout differs")
    offset, ov = completed_source(args.offset_run)
    oc = offset["run_contract"]
    if (offset["completed_epochs"] != oc["training_config"]["epochs"]
        or oc["base_sha256"] != common["base_checkpoint_sha256"]
        or oc["resolved_preprocessing"] != common["resolved_preprocessing"]
        or oc["validation_ids_sha256"] != fingerprint
        or offset["validation"] != ov["aggregate"]
        or ids != [x["sample_id"] for x in ov["per_image"]]):
        raise ValueError("Offset source is not aligned with the frozen text-recall maps")
    for row in ov["per_image"]:
        path = args.offset_run / "validation_exports" / row["sample_id"].replace(":", "__") / "offsets.npz"
        if sha256_file(path) != row["export_sha256"]:
            raise ValueError("Frozen offset export changed: " + row["sample_id"])
    sources = {}
    for branch in BRANCHES:
        folder = args.run_root / "branches" / branch
        report, validation = completed_source(folder)
        c = report["run_contract"]
        expected = comparison["branches"][branch]
        if (report["completed_epochs"] != common["config"]["training"]["epochs"]
            or c["branch"] != branch or c["config"] != common["config"]
            or c["source_sha256"] != common["source_sha256"]
            or c["validation_ids_sha256"] != fingerprint
            or report["checkpoint"]["sha256"] != expected["checkpoint_sha256"]
            or validation["checkpoint_sha256"] != expected["checkpoint_sha256"]
            or validation["operating_point"] != expected["operating_point"]
            or validation["operating_point"]["gt_words"] != 983
            or ids != [x["sample_id"] for x in validation["exports"]]):
            raise ValueError("Text branch provenance differs: " + branch)
        for row in validation["exports"]:
            source_folder = folder / "validation_exports" / row["sample_id"].replace(":", "__")
            for filename, expected_hash in row["files"].items():
                if sha256_file(source_folder / filename) != expected_hash:
                    raise ValueError("Frozen text export changed: " + branch + "/" + row["sample_id"])
        sources[branch] = {"root": folder, "report": report, "validation": validation}
    return {"common": common, "comparison": comparison, "branches": sources, "offset": offset,
            "offset_validation": ov, "manifest": manifest, "ids": ids, "target": target}


def geometry_protocol(args, cfg, sources):
    names = ("scripts/evaluate_text_recall_crops.py", "localize/text_recall_crops.py",
        "localize/center_guided_word_crops.py", "localize/word_center_offsets.py",
        "localize/word_multitask.py", "localize/word_guided_contours.py", "localize/crop_recovery.py",
        "localize/repository_contours.py", "recognition/parseq.py", "recognition/scene_text_v1_evaluation.py",
        "scripts/evaluate_word_multitask_crops.py", "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
        "scripts/evaluate_totaltext_highres_contour_padding.py", "scripts/rescore_totaltext_highres_padding_symbols.py",
        "segmentation/totaltext_word_kernel.py", "segmentation/word_multitask_evaluation.py",
        "segmentation/totaltext_image_frequency_augment.py")
    return {"schema_version": 1, "reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "source_seed": sources["common"]["config"]["training"]["seed"],
        "config": cfg, "config_sha256": sha256_file(args.config),
        "source_sha256": {name: sha256_file(REPO / name) for name in names},
        "source_run_root": str(args.run_root.resolve()), "offset_run": str(args.offset_run.resolve()),
        "source_contract_sha256": sha256_file(args.run_root / "experiment_contract.json"),
        "comparison_sha256": sha256_file(args.run_root / "comparison.json"),
        "source_reports": {b: {"run_manifest_sha256": sha256_file(s["root"] / "run_manifest.json"),
            "validation_sha256": sha256_file(s["root"] / "validation.json"),
            "checkpoint_sha256": s["report"]["checkpoint"]["sha256"]} for b, s in sources["branches"].items()},
        "offset_report_sha256": sha256_file(args.offset_run / "run_manifest.json"),
        "offset_validation_sha256": sha256_file(args.offset_run / "validation.json"),
        "canonical_size": 256, "opencv": cv2_version(), "numpy": np.__version__,
        "metric_protocol": "original exclusive bbox one-to-one IoU>=0.50 plus exact casefold text; all983 care words; fully invalid instances ignored only after inference; punctuation retained; empty denominators=0"}


def cv2_version():
    import cv2
    return cv2.__version__


def build_geometry(args, cfg, sources, protocol):
    output = args.output_root.resolve()
    cached = output / "geometry_results.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        if result["status"] != "SUCCESS" or result["protocol"] != protocol:
            raise ValueError("Completed geometry provenance differs")
        verify_geometry_exports(result, output)
        return result
    branches = {b+"__"+policy: {m: {"per_image": [], "native_totals": Counter(), "original_totals": Counter()}
        for m in METHODS} for b in BRANCHES for policy in cfg["threshold_policies"]}
    by_id = {s.sample_id: s for s in sources["manifest"].samples}
    offsets = {r["sample_id"]: r for r in sources["offset_validation"]["per_image"]}
    source_rows = {b: {r["sample_id"]: r for r in s["validation"]["exports"]} for b, s in sources["branches"].items()}
    thresholds = {}; native_replays = 0
    for position, sample_id in enumerate(sources["ids"], 1):
        key = sample_id.replace(":", "__")
        sample = by_id[sample_id]; image_path = sources["target"] / sample.image_path
        if sha256_file(image_path) != sample.image_sha256:
            raise ValueError("Original RGB changed: " + sample_id)
        with Image.open(image_path) as im:
            rgb = np.ascontiguousarray(np.asarray(im.convert("RGB")))
        offset_path = args.offset_run / "validation_exports" / key / "offsets.npz"
        if sha256_file(offset_path) != offsets[sample_id]["export_sha256"]:
            raise ValueError("Frozen offset map changed: " + sample_id)
        with np.load(offset_path, allow_pickle=False) as o:
            vector, offset_owners, offset_validity = [o[k].copy() for k in ("normalized_offsets", "owners", "validity")]
        recipe_cache = {}
        frozen = []
        for branch in BRANCHES:
            source = sources["branches"][branch]
            directory = source["root"] / "validation_exports" / key
            row = source_rows[branch][sample_id]
            for filename, expected in row["files"].items():
                if sha256_file(directory / filename) != expected:
                    raise ValueError("Frozen export changed: " + branch + "/" + sample_id + "/" + filename)
            with np.load(directory / "maps.npz", allow_pickle=False) as z:
                p, canvas, validity, owners, expected_ids = [z[k].copy() for k in
                    ("probabilities", "canvas", "validity", "owners", "expected_owner_ids")]
            if (p.shape != (1, 256, 256) or not np.array_equal(owners, offset_owners)
                or not np.array_equal(validity, offset_validity)
                or not np.array_equal(canvas, letterbox_canvas_mask((rgb.shape[1], rgb.shape[0]), native_size=256).numpy())):
                raise ValueError("Canonical map/vector/canvas alignment differs: " + sample_id)
            for policy in cfg["threshold_policies"]:
                threshold = source["validation"]["operating_point"]["threshold"] if policy == "selected" else cfg["common_threshold"]
                name = branch+"__"+policy; thresholds[name] = threshold
                # Identical maps/thresholds share postprocessing, not just OCR.
                cache_key = (p.tobytes(), float(threshold))
                if cache_key not in recipe_cache:
                    recipe_cache[cache_key] = prepare_recipes(p[0], vector, canvas, (rgb.shape[1], rgb.shape[0]), cfg, threshold=threshold)
                recipes = recipe_cache[cache_key]
                for method, recipe in recipes.items():
                    base = Path(name) / method / "samples" / key
                    label_path = output / base / "instances.png"
                    _save_labels(label_path, recipe["labels"])
                    proposals = []
                    for proposal in recipe["proposals"]:
                        x1, y1, x2, y2 = proposal["bbox"]
                        proposals.append({**proposal, "crop_rgb_sha256": crop_array_hash(rgb[y1:y2, x1:x2])})
                    content = {"sample_id": sample_id, "result_class": "approximate reproduction",
                        "branch": branch, "threshold_policy": policy, "threshold": threshold,
                        "image_path": str(image_path), "image_sha256": sample.image_sha256,
                        "source_maps_sha256": row["files"]["maps.npz"],
                        "offset_maps_sha256": offsets[sample_id]["export_sha256"], "proposals": proposals, "audit": recipe["audit"]}
                    write_json(output / base / "proposals.json", content)
                    files = {str(base / n): sha256_file(output / base / n) for n in ("instances.png", "proposals.json")}
                    if method == "findcontours":
                        visual = output / base / "predictions.png"
                        if not visual.exists(): render_word_maps(Image.fromarray(rgb), p, proposals, visual)
                        files[str(base / "predictions.png")] = sha256_file(visual)
                    frozen.append((name, method, content, recipe["labels"], validity, owners, expected_ids, files))
        # Only now read care GT: all32 recipes for this image have been frozen.
        gt, annotation_sha = _train_gt(sample, sources["target"], (rgb.shape[1], rgb.shape[0]))
        for name, method, content, labels, valid, owners, expected, files in frozen:
            ignored = [int(label) for label in np.unique(labels[labels > 0]) if not np.any((labels == label) & valid)]
            care_labels = labels.copy(); care_labels[np.isin(labels, ignored)] = 0
            native = instance_geometry(care_labels, owners, expected)
            if method == "connected_components" and name.endswith("__selected"):
                branch = name.split("__", 1)[0]
                old = next(r for r in sources["branches"][branch]["validation"]["secondary_component_geometry_per_image"] if r["sample_id"] == sample_id)
                if any(native[k] != old[k] for k in native):
                    raise RuntimeError("Original connected-components diagnostic changed")
                native_replays += 1
            care = [v for v in content["proposals"] if v["instance_id"] not in ignored]
            original = score_predictions(care, gt, text_available=False)
            part = branches[name][method]
            part["per_image"].append({**content, "ignored_prediction_ids": ignored, "ground_truth": gt,
                "annotation_path": str(sources["target"] / "datasets/total_text/train/annotations" / (sample_id.split(":", 1)[1]+".txt")),
                "annotation_sha256": annotation_sha, "native_geometry": native, "original_geometry": original, "exports": files})
            part["native_totals"].update(native); part["original_totals"].update(original)
        if position % 25 == 0: print(f"[frozen crops] {position}/125", flush=True)
    for branch in branches.values():
        for part in branch.values():
            if len(part["per_image"]) != 125 or part["original_totals"]["gt_words"] != 983:
                raise RuntimeError("Complete validation population differs")
            part["native"] = aggregate_word_metrics(dict(part.pop("native_totals")))
            part["original"] = ratios(dict(part.pop("original_totals")))
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "reproduction_mode": "approximate",
        "protocol": protocol, "branches": branches, "thresholds": thresholds,
        "segmentation_inference": "none; all aligned frozen maps reused", "original_native_geometry_replays": native_replays}
    write_json(cached, result)
    return result


def prepare_ocr_provider(output, roots, geometry):
    """Adapt completed, hash-bound caches to the existing recognizer's provider schema."""
    wanted = {p["crop_rgb_sha256"] for b in geometry["branches"].values() for part in b.values()
              for row in part["per_image"] for p in row["proposals"]}
    provider = output / "qualified_ocr_provider"
    origins = []; records = {}; sources = {}
    for root in roots:
        root = root.resolve(); path = root / "recognition_results.json"
        result = json.loads(path.read_text()); protocol = result["protocol"]
        if (result["status"] != "SUCCESS" or result["result_class"] != "approximate reproduction"
            or protocol["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256
            or protocol["parseq_runtime_archive_sha256"] != PARSEQ_RUNTIME_ARCHIVE_SHA256
            or protocol["recognition"]["decode_ar"] is not True or protocol["recognition"]["refine_iters"] != 1):
            raise ValueError("OCR source is incomplete or uses different pinned decoding")
        origins.append({"root": str(root), "results_sha256": sha256_file(path)})
        for key in sorted(wanted & result["ocr_records_sha256"].keys()):
            record_path = root / "ocr" / (key+".json")
            expected = result["ocr_records_sha256"][key]
            if sha256_file(record_path) != expected:
                raise ValueError("Historical OCR record changed: " + key)
            record = json.loads(record_path.read_text()); validate_ocr(record, key)
            if key in records and records[key]["raw_text"] != record["raw_text"]:
                raise ValueError("Conflicting exact-crop historical OCR: " + key)
            if key not in records:
                records[key] = record
                sources[key] = {"path": str(record_path), "sha256": expected}
    meta = {"schema_version": 1, "result_class": "approximate reproduction",
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "device": "verified source caches; CPU replay required", "parseq_batch_size": 8,
        "origins": origins, "records": sources, "qualification": "source SHA256 + pinned decoding + exact RGB; existing16-crop current-device replay still mandatory"}
    write_json(provider / "protocol.json", meta)
    for key, record in records.items():
        destination = provider / "ocr" / (key+".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sources[key]["sha256"]:
                raise ValueError("Qualified cache provider record changed")
        else: shutil.copyfile(sources[key]["path"], destination)
    return provider / "ocr"


def summarize(output, geometry, recognition):
    rows = []
    for branch, methods in recognition["branches"].items():
        for method, value in methods.items():
            a = value["aggregate"]
            rows.append({"branch": branch.split("__")[0], "threshold_policy": branch.split("__")[1],
                "threshold": geometry["thresholds"][branch], "method": method, "images": len(value["per_image"]),
                "gt_words": a["gt_words"], "care_proposals": a["proposals"],
                "localized_words": a["geometry_iou_ge_50"], "localization_hmean": a["geometry_f1"],
                "correct_parseq_words": a["exact_casefold_iou_ge_50"], "precision": a["exact_casefold_precision"],
                "recall": a["exact_casefold_recall_all_gt"], "hmean": a["exact_casefold_f1"],
                "merge_rate": value["native"]["merge_rate"], "split_rate": value["native"]["split_rate"]})
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "reproduction_mode": "approximate",
        "population": "125 frozen Total-Text TRAIN holdout images /983 care words; not official TEST",
        "metric": "original exclusive-box IoU>=0.50 AND exact casefold transcription; global one-to-one matching; punctuation retained",
        "rows": rows, "geometry_results_sha256": sha256_file(output / "geometry_results.json"),
        "recognition_results_sha256": sha256_file(output / "recognition_results.json"),
        "ocr": {k: recognition[k] for k in ("unique_rgb_crops", "new_ocr_records", "reused_ocr_records", "cache_replay_qualification", "memory", "elapsed_seconds")}}
    write_json(output / "summary.json", result)
    lines = ["# Frozen text-recall crops + PARSeq", "", "Approximate reproduction; reused TRAIN holdout125 /983 words. No checkpoint reselection or TEST claim.", "", "| Branch | Policy / threshold | Method | Correct PARSeq / GT | Precision | Recall | Hmean |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["branch"]} | {r["threshold_policy"]} / {r["threshold"]:.2f} | {r["method"]} | {r["correct_parseq_words"]}/{r["gt_words"]} | {r["precision"]:.2%} | {r["recall"]:.2%} | {r["hmean"]:.2%} |')
    path = output / "REPORT.md"; text = "\n".join(lines)+"\n"
    if path.exists() and path.read_text() != text: raise ValueError("Existing human report differs")
    if not path.exists(): path.write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("run-root", "offset-run", "output-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--action", choices=["preflight", "geometry", "recognize", "all"], required=True)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    for name in ("parseq-source", "parseq-source-archive", "parseq-checkpoint"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--reuse-run-root", type=Path, action="append", default=[])
    args = parser.parse_args(); args.config = args.config.resolve()
    args.run_root = args.run_root.resolve(); args.offset_run = args.offset_run.resolve()
    args.output_root = args.output_root.resolve()
    if args.cpu_threads < 1: raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    cfg = read_config(args.config, args.mode); sources = load_sources(args, cfg)
    if args.output_root == REPO or any(args.output_root == p or p in args.output_root.parents
        or args.output_root in p.parents for p in [args.run_root, args.offset_run, sources["target"], *[p.resolve() for p in args.reuse_run_root]]):
        raise ValueError("Output must be isolated from source runs, dataset and OCR caches")
    protocol = geometry_protocol(args, cfg, sources)
    if args.action == "preflight":
        print(json.dumps({"status": "PASS", "result_class": "smoke-test result", "branches": 4, "images": 125, "gt_words": 983})); return 0
    output = args.output_root; output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(output / "geometry_protocol.json", protocol)
        try:
            geometry = build_geometry(args, cfg, sources, protocol)
            if args.action in ("recognize", "all"):
                args.reuse_ocr = [prepare_ocr_provider(output, args.reuse_run_root, geometry)] if args.reuse_run_root else []
                recognition = recognize(args, cfg, geometry)
                result = summarize(output, geometry, recognition)
            else: result = {"status": "SUCCESS", "pipelines": sum(len(b) for b in geometry["branches"].values())}
        except Exception as error:
            write_json(output / "failure.json", {"status": "FAILED", "result_class": "approximate reproduction",
                "action": args.action, "type": type(error).__name__, "message": str(error), "device": "cpu"})
            raise
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
