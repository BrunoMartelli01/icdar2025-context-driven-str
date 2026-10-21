"""Replay qualified crop predictions with seeded links and a GT-free gate."""

import argparse
import fcntl
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import scipy
from PIL import Image
from scipy.special import expit

from localize.unet_crop_links import patches_from_labels
from localize.unet_seeded_selective import refine_from_evidence, seeded_link_instances
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _geometry_match, _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_crop_links import FILES as SOURCE_FILES, bind_contract, read_config as read_source_config
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.word_multitask_evaluation import aggregate_word_metrics


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/unet_seeded_selective_v1.json"
ARMS = {"control": "control", "crop_word_all": "crop_word", "crop_links_raw": "crop_links",
        "crop_links_seeded_all": "crop_links", "crop_word_selective": "crop_word",
        "crop_links_seeded_selective": "crop_links"}
FILES = (*SOURCE_FILES, "scripts/evaluate_unet_crop_links_ocr.py", "scripts/summarize_unet_crop_links.py",
         "localize/unet_seeded_selective.py", "scripts/run_unet_seeded_selective.py",
         "scripts/evaluate_unet_seeded_selective_ocr.py", "scripts/summarize_unet_seeded_selective.py")


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Seeded/selective experiments require explicit APPROXIMATE configuration")
    config = json.loads(path.read_text())
    if (config.get("schema_version") != 1 or config.get("experiment_kind") != "unet_seeded_selective_v1"
            or config.get("reproduction_mode") != mode or config.get("claim_label") != "approximate reproduction"
            or config.get("branches") != list(ARMS)):
        raise ValueError("Unsupported seeded/selective configuration")
    source_path = (REPO / config["source_config"]).resolve()
    if sha256_file(source_path) != config["source_config_sha256"]:
        raise ValueError("Qualified crop source config changed")
    source, _ = read_source_config(source_path, mode)
    for group in ("seeded_links", "gate", "guard"):
        if not np.isfinite(list(config[group].values())).all():
            raise ValueError("Nonfinite experimental thresholds: " + group)
    link, gate, guard = [config[k] for k in ("seeded_links", "gate", "guard")]
    if min(link.values()) < 0 or sum(link.values()) <= 0:
        raise ValueError("Invalid link-growth weights")
    if any(not 0 <= v <= 1 for v in gate.values()): raise ValueError("Invalid gate thresholds")
    if (not 0 < guard["minimum_anchor_coverage"] <= 1 or guard["maximum_support_growth"] < 1
            or type(guard["maximum_children"]) is not int or guard["maximum_children"] < 1
            or not 0 <= guard["maximum_neighbor_coverage"] <= 1):
        raise ValueError("Invalid replacement guard")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("cpu_threads", "ocr_batch_size")):
        raise ValueError("Invalid runtime batch/threads")
    if set(config["source_artifacts"]) != set(ARMS.values()): raise ValueError("Missing source provenance")
    return config, source


def qualify_source(root, config, source, expected_ids):
    contract = json.loads((root / "run_contract.json").read_text())
    if contract["resolved_config"] != source:
        raise ValueError("Completed crop experiment used a different configuration")
    for name, digest in contract["source_sha256"].items():
        if sha256_file(REPO / name) != digest: raise ValueError("Qualified source code changed: " + name)
    prepared_path = root / "predicted_crop_manifest.json"
    prepared_sha = sha256_file(prepared_path)
    receipts, reports = {}, {}
    for arm, expected in config["source_artifacts"].items():
        directory = root / arm
        receipt = json.loads((directory / "run_manifest.json").read_text())
        path = directory / "validation.json"; report = json.loads(path.read_text())
        checkpoint = Path(receipt["checkpoint"]["path"])
        if not checkpoint.is_absolute(): checkpoint = directory / checkpoint
        required_epochs = source["epochs"] if arm.startswith("crop_") else 0
        if (receipt.get("status") != "SUCCESS" or receipt.get("training_complete") is not True
                or receipt.get("completed_epochs") != required_epochs
                or receipt["run_contract"]["resolved_config"] != source
                or receipt["run_contract"].get("prepared_crop_manifest_sha256") != prepared_sha
                or receipt["run_contract"] != report["run_contract"]
                or receipt["checkpoint"]["epoch"] != expected["epoch"]
                or sha256_file(checkpoint) != expected["checkpoint_sha256"]
                or receipt["checkpoint"]["sha256"] != expected["checkpoint_sha256"]
                or report["checkpoint_sha256"] != expected["checkpoint_sha256"]
                or sha256_file(path) != expected["validation_sha256"]
                or len(report["per_image"]) != len(expected_ids)
                or {r["sample_id"] for r in report["per_image"]} != expected_ids
                or report["aggregate"]["gt_words"] != source["expected_validation_words"]):
            raise ValueError("Incomplete or incompatible source artifacts: " + arm)
        receipts[arm] = {"manifest_sha256": sha256_file(directory / "run_manifest.json"),
                         "validation_sha256": sha256_file(path), "checkpoint_sha256": expected["checkpoint_sha256"]}
        reports[arm] = {r["sample_id"]: r for r in report["per_image"]}
    prepared = json.loads(prepared_path.read_text())
    if (prepared.get("status") != "SUCCESS" or prepared.get("geometry_uses_gt") is not False
            or prepared.get("checkpoint_sha256") != source["source_checkpoint_sha256"]
            or len(prepared["per_image"]) != len(prepared["indices"])
            or len({r["sample_id"] for r in prepared["per_image"]}) != len(prepared["per_image"])):
        raise ValueError("Invalid frozen predicted-crop manifest")
    records = {r["sample_id"]: r for r in prepared["per_image"]}
    if not expected_ids <= set(records): raise ValueError("Missing first-pass inputs")
    return reports, records, {"sources": receipts, "predicted_manifest_sha256": sha256_file(prepared_path),
                               "source_contract_sha256": sha256_file(root / "run_contract.json")}


def load_export(root, arm, sample_id, row):
    directory = root / arm / "validation_exports" / sample_id.replace(":", "__")
    path = directory / "proposals.json"; maps = directory / "maps.npz"
    if sha256_file(path) != row["proposals_sha256"] or sha256_file(maps) != row["maps_sha256"]:
        raise ValueError("Source export checksum changed: " + arm + "/" + sample_id)
    record = json.loads(path.read_text())
    if record["sample_id"] != sample_id or record["maps_sha256"] != row["maps_sha256"]:
        raise ValueError("Source export identity changed")
    patches = []
    with np.load(maps, allow_pickle=False) as saved:
        for p in record["proposals"]:
            mask = saved["support_" + str(p["instance_id"])].copy()
            x, y, r, b = p["bbox"]
            if mask.dtype != bool or mask.shape != (b - y, r - x) or int(mask.sum()) != p["foreground_area"]:
                raise ValueError("Source support shape/area changed")
            patches.append({"bbox": p["bbox"], "mask": mask})
    return patches, record, directory


def crop_loader(root, sample_id, record, reports, source, config):
    """Lazy checksum-qualified replay with shared decoding across ablation arms."""
    crops = {c["seed_id"]: c for c in record["crops"]}
    if len(crops) != len(record["crops"]): raise ValueError("Duplicate predicted seed IDs")
    references = {arm: {e["seed_id"]: e for e in reports[arm][sample_id]["audit"]["crop_evidence"]}
                  for arm in ("crop_word", "crop_links")}
    if any(set(refs) != set(crops) for refs in references.values()):
        raise ValueError("Missing or duplicate per-crop evidence")
    cached = {}
    def load(seed_id, decoder):
        arm = "crop_word" if decoder == "word" else "crop_links"
        key = (seed_id, decoder)
        if key in cached: return cached[key]
        ref = references[arm][seed_id]
        path = root / arm / "validation_exports" / sample_id.replace(":", "__") / "crop_evidence" / ref["path"]
        if sha256_file(path) != ref["sha256"]: raise ValueError("Raw per-crop prediction checksum changed")
        with np.load(path, allow_pickle=False) as saved:
            p, canvas = saved["probabilities"], saved["canvas"]
            if not np.array_equal(saved["context_bbox"], crops[seed_id]["context_bbox"]):
                raise ValueError("Predicted context coordinates changed")
            if decoder == "seeded":
                child, audit = seeded_link_instances(p, expit(saved["links_logits"]), canvas,
                                                      source["postprocessing"], config["seeded_links"])
            else:
                child, audit = saved["labels"].copy(), {}
        cached[key] = {"decoded_labels": child, "decoder_audit": audit}
        return cached[key]
    return load


def score_image(patches, batch, owners, gt, thresholds):
    """Exactly UCL-v1's original-coordinate care/ignore and merge/split protocol."""
    valid = np.asarray(batch["validity"]) > 0
    reference = owners.copy(); reference[~valid] = 0
    expected = np.unique(owners[owners > 0])
    if len(gt) != len(expected): raise ValueError("Original annotation/owner denominator differs")
    touched = {int(owner): set() for owner in expected}
    proposals, ignored, fused = [], [], set()
    foreground = np.zeros(valid.shape, bool)
    for i, patch in enumerate(patches, 1):
        x, y, r, b = patch["bbox"]; support = patch["mask"]
        if not (0 <= x < r <= valid.shape[1] and 0 <= y < b <= valid.shape[0]) or support.shape != (b-y, r-x):
            raise ValueError("Proposal outside original RGB")
        proposals.append({"instance_id": i, "bbox": patch["bbox"], "bbox_original": patch["bbox"],
                          "foreground_area": int(support.sum())})
        foreground[y:b, x:r] |= support
        if not np.any(support & valid[y:b, x:r]): ignored.append(i); continue
        words = np.unique(reference[y:b, x:r][support]); words = words[words > 0]
        for word in words: touched[int(word)].add(i)
        if len(words) > 1: fused.add(i)
    care = [p for p in proposals if p["instance_id"] not in ignored]
    metrics = {"gt_words": len(gt), "raw_proposals": len(care), "gt_words_lost_in_native_target": 0,
               "gt_words_split_across_predicted_components": sum(len(v) >= 2 for v in touched.values()),
               "gt_words_touched_by_fused_components": sum(bool(v & fused) for v in touched.values()),
               "predicted_components_touching_multiple_gt_words": len(fused)}
    for t in thresholds: metrics["matched_gt_iou_ge_" + str(round(t * 100))] = len(_geometry_match(care, gt, t))
    target = reference > 0
    tp, fp, fn = [int(np.count_nonzero(m & valid)) for m in
                  (foreground & target, foreground & ~target, ~foreground & target)]
    return metrics, proposals, ignored, reference, valid, (tp/(tp+fp+fn) if tp+fp+fn else 1.), (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def save_image(directory, sample, batch, patches, proposals, reference, valid):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"owners": reference, "validity": valid,
              "boxes": np.asarray([p["bbox"] for p in patches], np.int32).reshape(-1, 4)}
    arrays.update({"support_" + str(i): p["mask"] for i, p in enumerate(patches, 1)})
    maps = directory / "maps.npz"
    if maps.exists():
        with np.load(maps, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError("Existing postprocessing export changed; use a new output root")
    else: np.savez_compressed(maps, **arrays)
    path = directory / "proposals.json"
    _json(path, {"sample_id": sample.sample_id, "image_path": batch["image_path"], "image_sha256": sample.image_sha256,
                 "maps_sha256": sha256_file(maps), "proposals": proposals,
                 "quality_scores": {}, "reproduction_mode": "approximate"})
    return {"maps_sha256": sha256_file(maps), "proposals_sha256": sha256_file(path)}


def run(args):
    config, source = read_config(args.config, args.mode)
    cv2.setNumThreads(config["cpu_threads"])
    root, old, output = args.word_root.resolve(), args.crop_root.resolve(), args.output_root.resolve()
    if output == REPO or any(output == p or p in output.parents or output in p.parents for p in (root, old)):
        raise ValueError("Output must be isolated from source artifacts")
    manifest_path = root / source["train_manifest"]
    if sha256_file(manifest_path) != source["word_manifest_sha256"]: raise ValueError("Qualified word manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train": raise ValueError("Held-out official TRAIN only")
    _, indices, split = _split_indices(manifest, source["seed"], source["validation_count"])
    if split != source["validation_ids_sha256"]: raise ValueError("Frozen validation split changed")
    reports, records, provenance = qualify_source(old, config, source, {manifest.samples[i].sample_id for i in indices})
    if args.limit_images is not None:
        if args.limit_images < 1: raise ValueError("Invalid smoke image count")
        indices = indices[:args.limit_images]
    result_class = "smoke-test result" if args.limit_images is not None else "approximate reproduction"
    contract = {"schema_version": 1, "resolved_config": config, "source_config": source,
                "validation_ids_sha256": split, "indices": indices, "result_class": result_class,
                "runtime_inputs": {"word_root": str(root), "crop_root": str(old)}, "source_provenance": provenance,
                "source_sha256": {n: sha256_file(REPO / n) for n in set(FILES)},
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
                                "opencv_threads": cv2.getNumThreads()},
                "execution": {"model_set": "frozen U-Net map replay only", "device": "cpu", "dtype": "float64 flood costs; frozen float32 maps"}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_contract(output / "run_contract.json", contract)
        raw = SegmentationDataset(manifest, root)
        rows = {a: [] for a in ARMS}; sums = {a: {} for a in ARMS}; pixels = {a: [] for a in ARMS}
        try:
            for n, index in enumerate(indices, 1):
                sample = manifest.samples[index]; record = records[sample.sample_id]
                image_path = root / sample.image_path
                if sha256_file(image_path) != sample.image_sha256 or record["image_sha256"] != sample.image_sha256:
                    raise ValueError("Source RGB checksum changed")
                path = old / record["maps_path"]
                if sha256_file(path) != record["maps_sha256"]: raise ValueError("First-pass maps changed")
                with np.load(path, allow_pickle=False) as f: first = {k: f[k] for k in f.files}
                source_patches = {}
                for arm in ("control", "crop_word", "crop_links"):
                    patches, _, _ = load_export(old, arm, sample.sample_id, reports[arm][sample.sample_id])
                    source_patches[arm] = patches
                loader = crop_loader(old, sample.sample_id, record, reports, source, config)
                # Freeze every branch's geometry before accessing annotations.
                frozen = {}
                for arm in ARMS:
                    if arm in ("control", "crop_word_all", "crop_links_raw"):
                        frozen[arm] = (source_patches[ARMS[arm]], {"execution": "qualified original export"})
                    else:
                        decoder = "word" if arm == "crop_word_selective" else "seeded"
                        frozen[arm] = refine_from_evidence(first, record, tuple(record["image_wh"]), loader,
                            source, config, decoder=decoder, selective=arm.endswith("selective"))
                batch = raw[index]
                if list(batch["image"].size) != record["image_wh"]: raise ValueError("Source RGB shape changed")
                with Image.open(raw.paths[index][4]) as opened: owners = np.asarray(opened, np.int32)
                gt, _ = _train_gt(sample, raw.data_root, batch["image"].size)
                for arm, (patches, audit) in frozen.items():
                    metrics, proposals, ignored, reference, valid, iou, f1 = score_image(
                        patches, batch, owners, gt, source["evaluation_iou_thresholds"])
                    if arm in ("control", "crop_word_all", "crop_links_raw"):
                        expected = reports[ARMS[arm]][sample.sample_id]
                        if any(metrics[k] != expected[k] for k in metrics):
                            raise ValueError("Baseline metric replay differs: " + arm)
                    receipt = save_image(output / arm / "validation_exports" / sample.sample_id.replace(":", "__"),
                                         sample, batch, patches, proposals, reference, valid)
                    rows[arm].append({"sample_id": sample.sample_id, **metrics, "ignored_instance_ids": ignored,
                                      "audit": audit, **receipt})
                    for k, v in metrics.items(): sums[arm][k] = sums[arm].get(k, 0) + v
                    pixels[arm].append((iou, f1))
                print(f"[seeded/selective replay] {n}/{len(indices)} {sample.sample_id}", flush=True)
            for arm in ARMS:
                aggregate = aggregate_word_metrics(sums[arm])
                aggregate.update(foreground_iou=float(np.mean([p[0] for p in pixels[arm]])),
                                 foreground_f1_macro=float(np.mean([p[1] for p in pixels[arm]])))
                if result_class != "smoke-test result" and aggregate["gt_words"] != source["expected_validation_words"]:
                    raise ValueError("Validation denominator changed")
                checkpoint = config["source_artifacts"][ARMS[arm]]
                report = {"aggregate": aggregate, "per_image": rows[arm], "run_contract": contract,
                          "reproduction_mode": "approximate", "result_class": result_class,
                          "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                          "coordinate_system": "original RGB exclusive xyxy", "matching": "UCL-v1 maximum one-to-one bbox IoU .25/.5/.75"}
                _json(output / arm / "validation.json", report)
                _json(output / arm / "run_manifest.json", {"status": "SUCCESS", "execution_kind": "frozen_postprocess",
                    "result_class": result_class, "run_contract": contract, "source_checkpoint": checkpoint,
                    "validation_sha256": sha256_file(output / arm / "validation.json"),
                    "memory": {"model_set": "frozen U-Net map replay only", "device": "cpu", "oom": False,
                               "gpu_peak_allocated_bytes": 0, "gpu_peak_reserved_bytes": 0}})
            _json(output / "geometry_comparison.json", {"status": "SUCCESS", "result_class": result_class,
                  "reproduction_mode": "approximate", "run_contract": contract,
                  "branches": {a: json.loads((output / a / "validation.json").read_text())["aggregate"] for a in ARMS}})
        except Exception as error:
            _json(output / "replay_failure.json", {"status": "FAILED", "error_type": type(error).__name__, "message": str(error)})
            raise
    return contract


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--crop-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--limit-images", type=int, help="Separate smoke output only; final OCR rejects it")
    run(p.parse_args())
