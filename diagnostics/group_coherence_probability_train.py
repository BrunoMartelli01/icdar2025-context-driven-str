"""CPU-only TRAIN geometry evaluation; probabilities must be verified cached data."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from diagnostics.gt_mask_localization_oracle import checked_native, compare_success, save_json, sha, summarize, verify_file
from localize.coherent_probability_groups import CoherenceParams, CorridorParams, group_components
from localize.grouped_word_blocks import GroupedWordBlockParams
from pipeline.frozen_segmentation import backproject_binary_mask
from segmentation.data import load_segmentation_manifest, icdar13_task22_rgb_to_target_and_validity
from segmentation.evaluate import resize_binary_target_to_native_logits


def load_context(config_path, oracle_path):
    config = json.loads(config_path.read_text())
    if config["reproduction_mode"] != "approximate" or config["scope"] != {"split": "TRAIN", "images": 229, "gt_words": 849}:
        raise ValueError("approximate TRAIN229/849 only")
    verify_file(oracle_path, config["oracle_sha256"])
    oracle = json.loads(oracle_path.read_text())
    for path, expected in oracle["protocol"]["code_sha256"].items():
        verify_file(path, expected)
    paths = oracle["protocol"]["source_paths"]
    source_path = Path(paths["source_report"])
    verify_file(source_path, oracle["protocol"]["source_report_sha256"])
    source = json.loads(source_path.read_text())
    verify_file(paths["train_manifest"], oracle["protocol"]["train_manifest_sha256"])
    verify_file(config["evaluation_config"], config["evaluation_config_sha256"])
    manifest = load_segmentation_manifest(paths["train_manifest"], require_checksums=False)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise ValueError("TRAIN population drift")
    return config, oracle, source, manifest, source_path, Path(paths["data_root"])


def cached_native(sample, source, source_path, checkpoint_sha):
    return checked_native(source_path.parent / "masks" / f"{sample.sample_id}.json",
        source["contract_sha256"], sample.sample_id,
        source["protocol"]["dataset_files"][str(sample.image_path)]["sha256"], checkpoint_sha)


def read_probability(cache, sample_id, binary, metadata):
    path = cache / f"{sample_id}.json"
    entry = json.loads(path.read_text())
    if entry["sample_id"] != sample_id or entry["checkpoint_sha256"] != metadata["segmentation_metadata"]["checkpoint"]["sha256"] or entry["image_sha256"] != metadata["image_sha256"]:
        raise RuntimeError("probability cache identity mismatch")
    array_path = cache / f"{sample_id}.npy"
    verify_file(array_path, entry["file_sha256"])
    value = np.load(array_path, allow_pickle=False)
    if value.shape != (128, 128) or value.dtype != np.dtype('float32') or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise RuntimeError("invalid native probability map")
    actual = hashlib.sha256(value.tobytes()).hexdigest()
    if actual != entry["array_sha256"] or actual != metadata["segmentation_metadata"]["native_probability_sha256"]:
        raise RuntimeError("probabilities differ from frozen historical inference")
    if not np.array_equal(value >= .5, binary > 0):
        raise RuntimeError("probabilities do not reproduce native frozen mask")
    return value


def project_support(value, hw):
    return F.interpolate(torch.from_numpy(value.astype(np.float32))[None, None], size=tuple(hw), mode="nearest")[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "oracle", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--probability-cache", type=Path)
    a = parser.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    config, oracle, source, manifest, source_path, data_root = load_context(a.config, a.oracle)
    checkpoint_sha = oracle["protocol"]["checkpoint_sha256"]
    if a.probability_cache is not None:
        # Do not begin a supposedly complete comparison with a partial cache.
        for sample in manifest.samples:
            binary, meta = cached_native(sample, source, source_path, checkpoint_sha)
            read_probability(a.probability_cache, sample.sample_id, binary, meta)
    params = GroupedWordBlockParams(**oracle["protocol"]["parameters"]["v1"])
    coherence = CoherenceParams(**config["group_parameters"])
    corridor = CorridorParams(**config["corridor_parameters"])
    protocol = {**config, "config_sha256": sha(a.config), "oracle_path": str(a.oracle.resolve()),
        "checkpoint_sha256": checkpoint_sha, "device": "cpu", "models_loaded": False,
        "v1_parameters": oracle["protocol"]["parameters"]["v1"],
        "probability_cache": str(a.probability_cache.resolve()) if a.probability_cache else None,
        "predicted_corridor_status": "ready" if a.probability_cache else "BLOCKED_NATIVE_PROBABILITIES_MISSING",
        "gt_corridor_interpretation": "binary target support only, NOT network confidence",
        "metric_policy": oracle["protocol"]["metric_policy"],
        "code_sha256": {p: sha(p) for p in [__file__, "localize/coherent_probability_groups.py"]}}
    a.output_dir.mkdir(parents=True)
    save_json(a.output_dir / "protocol.json", protocol)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    refs = {s["sample_id"]: s for s in oracle["samples"]}
    words = defaultdict(list)
    for w in source["per_word"]: words[w["sample_id"]].append({"bbox": w["bbox"], "text": w["text"]})
    records = []
    probability_files = {}
    for index, sample in enumerate(manifest.samples, 1):
        if index == 1 or index % 10 == 0 or index == 229:
            print(f"[group/coherence CPU] {index}/229 sample={sample.sample_id}", flush=True)
        ref = refs[sample.sample_id]
        for rel in (sample.image_path, sample.mask_path, sample.validity_annotation_path):
            verify_file(data_root / rel, source["protocol"]["dataset_files"][str(rel)]["sha256"])
        pred, meta = cached_native(sample, source, source_path, checkpoint_sha)
        with Image.open(data_root / sample.mask_path) as opened:
            target, _ = icdar13_task22_rgb_to_target_and_validity(opened,
                (data_root / sample.validity_annotation_path).read_text(encoding="utf-8-sig"))
        full = (np.asarray(target) > 0).astype(np.uint8)
        gt128 = resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
        if list(full.shape) != ref["image_hw"] or hashlib.sha256(gt128.tobytes()).hexdigest() != ref["gt_native_sha256"] or hashlib.sha256(full.tobytes()).hexdigest() != ref["gt_fullres_sha256"]:
            raise RuntimeError("qualified GT preprocessing mismatch")
        pred_support = None
        if a.probability_cache:
            probability = read_probability(a.probability_cache, sample.sample_id, pred, meta)
            pred_support = project_support(probability, full.shape)
            probability_files[sample.sample_id] = sha(a.probability_cache / f"{sample.sample_id}.json")
        conditions = {}
        for condition, native in (("PRED128", pred), ("GT128", gt128), ("GT_FULLRES", full)):
            binary = full if condition == "GT_FULLRES" else backproject_binary_mask(native, full.shape)
            if condition == "PRED128" and hashlib.sha256(binary.tobytes()).hexdigest() != meta["segmentation_metadata"]["output_mask_sha256"]:
                raise RuntimeError("prediction differs from frozen baseline")
            methods = dict(ref["conditions"][condition])
            methods["GROUP_COHERENCE"] = group_components(binary, params, mode="GROUP_COHERENCE", coherence=coherence, corridor=corridor)
            support = pred_support if condition == "PRED128" else binary.astype(np.float32)
            if support is not None:
                methods["PROBABILITY_CORRIDOR"] = group_components(binary, params, mode="PROBABILITY_CORRIDOR", coherence=coherence, corridor=corridor, support=support)
            conditions[condition] = methods
        records.append({"sample_id": sample.sample_id, "conditions": conditions})
    methods = list(oracle["summary"]) + ["GROUP_COHERENCE", "PROBABILITY_CORRIDOR"]
    summary, rows = summarize(records, words, methods, ["PRED128", "GT128", "GT_FULLRES"])
    for method in oracle["summary"]:
        if summary[method] != oracle["summary"][method]: raise RuntimeError("historical baseline metrics changed")
    transitions = {}
    for method in ("GROUP_COHERENCE", "PROBABILITY_CORRIDOR"):
        transitions[method] = {}
        for condition in summary[method]:
            transitions[method][condition] = compare_success(
                [r["best_iou"] for r in rows if r["method"] == "PRACTICAL_REMAP_V1" and r["condition"] == condition],
                [r["best_iou"] for r in rows if r["method"] == method and r["condition"] == condition])
    with (a.output_dir / "per-word.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"protocol": protocol, "summary": summary, "vs_v1": transitions, "samples": records,
        "probability_metadata_sha256": probability_files, "csv_sha256": sha(a.output_dir / "per-word.csv"),
        "validation": {"images": len(records), "gt_words": sum(map(len, words.values())),
                       "frozen_controls_exact": True, "gt_native_fullres_hashes_verified": True}}
    if result["validation"]["images"] != 229 or result["validation"]["gt_words"] != 849: raise RuntimeError("population mismatch")
    save_json(a.output_dir / "results.json", result)
    print(json.dumps({m: {c: v["geometry"]["gt_with_proposal_iou_ge_0_50"] for c, v in s.items()} for m, s in summary.items()}, indent=2))
    print("JSON", a.output_dir / "results.json", "SHA256", sha(a.output_dir / "results.json"))


if __name__ == "__main__": main()
