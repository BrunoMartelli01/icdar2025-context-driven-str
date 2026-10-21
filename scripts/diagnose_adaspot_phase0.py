"""Paired frozen-instance diagnostics; explicitly approximate AdaSpot Phase-0 proxies."""

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from localize.adaspot_gap import gap_instances
from localize.unet_detection_methods import rectified_word_crop
from localize.word_multitask import parseq_word_proposals
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from recognition.scene_text_v1_evaluation import box_iou
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_unet_word_research_ocr import score_predictions
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.training import seed_everything


REPO = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/diagnose_adaspot_phase0.py", "recognition/parseq.py",
    "localize/adaspot_gap.py", "localize/word_multitask.py", "localize/unet_detection_methods.py",
    "diagnostics/fixed_group_curve_geometry.py", "segmentation/totaltext_word_kernel.py",
    "localize/crop_recovery.py", "recognition/scene_text_v1_evaluation.py",
    "scripts/evaluate_unet_word_research_ocr.py",
    "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Immutable output changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def check_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 changed: {path}")
    return actual


def mixed_predictions(weak, strong, selected):
    if len(weak) != len(strong):
        raise ValueError("Routing must preserve the frozen proposal population")
    output = []
    for i, (left, right) in enumerate(zip(weak, strong)):
        if (left["bbox"], left["crop_rgb_sha256"]) != (right["bbox"], right["crop_rgb_sha256"]):
            raise ValueError("Routing must use identical pixels and matching geometry")
        output.append(right if i in selected else left)
    return output


def confidence_order(predictions):
    def key(i):
        p = predictions[i]
        score = p["minimum_token_probability"]
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid probability for confidence ranking")
        return (score if p["text"] and score is not None else -1., i)
    return sorted(range(len(predictions)), key=key)


def summarize(samples):
    totals, rows = Counter(), []
    for sample in samples:
        counts, matches = score_predictions(sample["predictions"], sample["ground_truth"])
        totals.update(counts)
        rows.append({**sample, "counts": counts,
            "matched_gt_indices": {k: sorted(v) for k, v in matches.items()}})
    n, gt, tp = totals["proposals"], totals["gt_words"], totals["exact_casefold_iou_ge_50"]
    return {"aggregate": {**totals, "precision": tp / n if n else 0.,
        "recall": tp / gt if gt else 0., "hmean": 2 * tp / (n + gt) if n + gt else 0.}, "per_image": rows}


def paired_changes(left, right):
    if [s["sample_id"] for s in left["per_image"]] != [s["sample_id"] for s in right["per_image"]]:
        raise ValueError("Paired comparison sample order differs")
    recovered, damaged = [], []
    for a, b in zip(left["per_image"], right["per_image"]):
        if a["ground_truth"] != b["ground_truth"]:
            raise ValueError("Paired comparison labels differ")
        old, new = [set(s["matched_gt_indices"]["exact_casefold_iou_ge_50"]) for s in (a, b)]
        recovered.extend([a["sample_id"], i] for i in sorted(new - old))
        damaged.extend([a["sample_id"], i] for i in sorted(old - new))
    return {"recovered": len(recovered), "damaged": len(damaged), "net": len(recovered) - len(damaged),
        "recovered_ids": recovered, "damaged_ids": damaged}


def routing(weak, strong, config):
    flat_w = [p for s in weak["per_image"] for p in s["predictions"]]
    flat_s = [p for s in strong["per_image"] for p in s["predictions"]]
    mixed_predictions(flat_w, flat_s, set())  # Validate paired membership before any ranking.
    order = confidence_order(flat_w)
    def evaluate(selected):
        predictions = mixed_predictions(flat_w, flat_s, selected)
        samples, offset = [], 0
        for sample in weak["per_image"]:
            size = len(sample["predictions"])
            samples.append({"sample_id": sample["sample_id"], "ground_truth": sample["ground_truth"],
                "predictions": predictions[offset:offset + size]})
            offset += size
        return summarize(samples)
    # GT-priority routing is a diagnostic, not an exact combinatorial oracle bound.
    priority, offset = [], 0
    for sample in weak["per_image"]:
        for i, p in enumerate(sample["predictions"]):
            q = flat_s[offset + i]
            def correct(candidate):
                return any(candidate["text"].casefold() == g["text"].casefold()
                    and candidate["text"] and box_iou(candidate["bbox"], g["bbox"]) >= .5
                    for g in sample["ground_truth"])
            priority.append((-(int(bool(correct(q))) - int(bool(correct(p)))), offset + i))
        offset += len(sample["predictions"])
    oracle_order = [i for _, i in sorted(priority)]
    rows = []
    for budget in config["budgets"]:
        k = int(math.floor(budget * len(flat_w)))
        selected = set(order[:k]); result = evaluate(selected)
        random_counts = []
        for repeat in range(config["random_repeats"]):
            rng = np.random.default_rng(config["seed"] + repeat)
            chosen = set(int(i) for i in rng.permutation(len(flat_w))[:k])
            random_counts.append(evaluate(chosen)["aggregate"]["exact_casefold_iou_ge_50"])
        rows.append({"requested_fraction": budget, "escalated": k,
            "fraction": k / len(flat_w) if flat_w else 0., "confidence": result["aggregate"],
            "changes": paired_changes(weak, result), "selected_global_indices": sorted(selected),
            "random_exact_counts": random_counts, "random_mean_exact": float(np.mean(random_counts)),
            "gt_priority_diagnostic": evaluate(set(oracle_order[:k]))["aggregate"]})
    return {"weak": "nar1", "strong": "ar1", "scope": "decoding proxy, not trained shared-feature heads",
        "ranking": "minimum character probability; empty first; frozen index ties", "rows": rows}


def prepare(args, config):
    source, root, output = args.source_run.resolve(), args.word_root.resolve(), args.output_root.resolve()
    if output == REPO or REPO in output.parents or output == source or source in output.parents or root == output or root in output.parents:
        raise ValueError("Output must be isolated outside repository and all inputs")
    source_files = [source / n for n in ("ocr_comparison.json", "ocr_protocol.json",
        "control/run_manifest.json", "control/validation.json", "control/run_contract.json", "control/best.pt")]
    hashes = {str(p): sha256_file(p) for p in source_files}
    source_ocr = json.loads(source_files[0].read_text())
    trained = json.loads((source / "control/run_manifest.json").read_text())
    contract = json.loads((source / "control/run_contract.json").read_text())
    validation = json.loads((source / "control/validation.json").read_text())
    if not trained.get("training_complete") or trained.get("status") != "SUCCESS" or trained["run_contract"] != contract:
        raise ValueError("Completed qualified Text+Kernel control required")
    check_hash(source / "control/best.pt", trained["checkpoint"]["sha256"])
    if validation["checkpoint_sha256"] != trained["checkpoint"]["sha256"]:
        raise ValueError("Frozen maps belong to another checkpoint")
    c = contract["resolved_config"]
    manifest_path = root / c["train_manifest"]
    check_hash(manifest_path, c["word_manifest_sha256"])
    hashes[str(manifest_path)] = c["word_manifest_sha256"]
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Only TRAIN holdout is permitted")
    _, indices, split_sha = _split_indices(manifest, c["seed"], c["validation_count"])
    ids = {manifest.samples[i].sample_id for i in indices}
    samples = {s.sample_id: s for s in manifest.samples}
    old = {r["sample_id"]: r for r in source_ocr["arms"]["control"]["per_image"]}
    rows = validation["per_image"]
    if (split_sha != c["validation_ids_sha256"] or len(rows) != config["expected_images"]
            or len(old) != len(rows) or {r["sample_id"] for r in rows} != ids or set(old) != ids):
        raise ValueError("Incomplete/duplicated/different validation population")
    arms = {name: [] for name in ("pred/axis", "pred/oriented", "pred/bezier", "gt/axis", "oracle_gap/axis")}
    unique, oracle_rows, pixels = {}, [], Counter()
    def freeze(crop):
        digest = crop_array_hash(crop)
        path = output / "crops" / (digest + ".png")
        if path.exists():
            with Image.open(path) as image:
                if crop_array_hash(np.asarray(image.convert("RGB"))) != digest:
                    raise ValueError("Frozen crop changed")
        else:
            path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(crop).save(path)
        unique[digest] = path
        return digest
    for row in rows:
        sid = row["sample_id"]
        folder = source / "control/validation_exports" / sid.replace(":", "__")
        for name, key in (("maps.npz", "maps_sha256"), ("proposals.json", "proposals_sha256")):
            path = folder / name; hashes[str(path)] = check_hash(path, row[key])
        record = json.loads((folder / "proposals.json").read_text())
        sample = samples[sid]; image_path = root / sample.image_path
        hashes[str(image_path)] = check_hash(image_path, sample.image_sha256)
        if record["sample_id"] != sid or record["image_sha256"] != sample.image_sha256:
            raise ValueError("Proposal source identity changed")
        with Image.open(image_path) as image: rgb = np.asarray(image.convert("RGB"))
        with np.load(folder / "maps.npz", allow_pickle=False) as saved:
            maps = {k: saved[k].copy() for k in saved.files}
        probabilities, canvas, labels = maps["probabilities"], maps["canvas"], maps["labels"]
        replay = gap_instances(probabilities, None, canvas, c["postprocessing"], use_gap=False)[0]
        if not np.array_equal(replay, labels):
            raise ValueError("Frozen control decoding no longer reproduces instances")
        views = {v: [] for v in ("axis", "oriented", "bezier")}
        for p in record["proposals"]:
            # No GT enters any predicted-instance crop geometry or view selection.
            for view in views:
                crop, geometry = rectified_word_crop(rgb, labels, p, view, config)
                digest = freeze(crop)
                views[view].append({**p, "crop_rgb_sha256": digest, "rectification": geometry})
        gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
        annotation_path = root / "datasets/total_text/train/annotations" / (sid.split(":", 1)[1] + ".txt")
        hashes[str(annotation_path)] = annotation_sha
        if gt != old[sid]["ground_truth"] or annotation_sha != old[sid]["annotation_sha256"]:
            raise ValueError("Original care annotations changed")
        for view, proposals in views.items():
            care = [p for p in proposals if p["instance_id"] not in row["ignored_instance_ids"]]
            arms["pred/" + view].append({"sample_id": sid, "ground_truth": gt, "predictions": care})
        gt_proposals = []
        for i, g in enumerate(gt):
            x0, y0, x1, y1 = g["bbox"]
            digest = freeze(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            gt_proposals.append({"bbox": g["bbox"], "gt_index": i, "crop_rgb_sha256": digest})
        arms["gt/axis"].append({"sample_id": sid, "ground_truth": gt, "predictions": gt_proposals})
        # Explicit oracle: the target G, NOT a prediction or an inference method.
        oracle_labels, audit, _ = gap_instances(probabilities, maps["gap_target"].astype(float),
            canvas, c["postprocessing"], use_gap=True)
        ignored = [int(i) for i in np.unique(oracle_labels[oracle_labels > 0])
            if not np.any((oracle_labels == i) & maps["validity"])]
        care_labels = oracle_labels.copy(); care_labels[np.isin(care_labels, ignored)] = 0
        geometry = instance_geometry(care_labels, maps["owners"], maps["expected_owner_ids"].tolist())
        oracle_rows.append({"sample_id": sid, "native_geometry": geometry, "audit": audit,
            "source_native_geometry": {k: row[k] for k in geometry}})
        raw = canvas & (probabilities[0] >= c["postprocessing"]["text_threshold"])
        pixels.update({"target_gap_pixels": int(maps["gap_target"].sum()),
            "target_gap_inside_predicted_text": int((maps["gap_target"].astype(bool) & raw).sum())})
        proposals = []
        for p in parseq_word_proposals(oracle_labels, (rgb.shape[1], rgb.shape[0]), padding=0, min_area=0, max_instances=None):
            x0, y0, x1, y1 = p["bbox"]
            digest = freeze(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            if p["instance_id"] not in ignored: proposals.append({**p, "crop_rgb_sha256": digest})
        arms["oracle_gap/axis"].append({"sample_id": sid, "ground_truth": gt, "predictions": proposals})
        label_path = output / "oracle_labels" / (sid.replace(":", "__") + ".png")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if label_path.exists():
            with Image.open(label_path) as image:
                if not np.array_equal(np.asarray(image), oracle_labels): raise ValueError("Oracle labels changed")
        else: Image.fromarray(oracle_labels.astype(np.uint16)).save(label_path)
    if sum(len(s["ground_truth"]) for s in arms["pred/axis"]) != config["expected_words"]:
        raise ValueError("Care-word denominator changed")
    write_json(output / "frozen_inputs.json", arms)
    write_json(output / "oracle_gap.json", {"diagnostic_only": True, "pixels": dict(pixels), "per_image": oracle_rows})
    return arms, unique, hashes, source_ocr


def run(args):
    config_path = args.config.resolve()
    if args.mode != "approximate" or REPO / "configs/approximate" not in config_path.parents:
        raise ValueError("Explicit APPROXIMATE namespace required")
    config = json.loads(config_path.read_text())
    if config["reproduction_mode"] != args.mode or config["claim_label"] != "approximate reproduction":
        raise ValueError("Configuration mode differs")
    if (config["decode_modes"] != {"nar0": [False, 0], "nar1": [False, 1], "ar1": [True, 1]}
            or config["batch_size"] < 1 or config["random_repeats"] < 1
            or any(not 0 <= v <= 1 for v in config["budgets"])):
        raise ValueError("Invalid predeclared decoding/routing protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This complete diagnostic requires GPU; no CPU fallback")
    torch.set_num_threads(config["cpu_threads"]); cv2.setNumThreads(1)
    seed_everything(config["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    check_hash(args.parseq_checkpoint, PARSEQ_CHECKPOINT_SHA256)
    check_hash(args.parseq_archive, PARSEQ_RUNTIME_ARCHIVE_SHA256)
    output = args.output_root.resolve()
    if (output / "results.json").exists():
        raise ValueError("Completed output exists; preserve it and use a new output root")
    arms, unique, hashes, source_ocr = prepare(args, config)
    hashes[str(config_path)] = sha256_file(config_path)
    hashes.update({str(REPO / f): sha256_file(REPO / f) for f in SOURCES})
    hashes[str(args.parseq_checkpoint.resolve())] = PARSEQ_CHECKPOINT_SHA256
    hashes[str(args.parseq_archive.resolve())] = PARSEQ_RUNTIME_ARCHIVE_SHA256
    protocol = {"schema_version": 1, "mode": args.mode, "result_class": "approximate reproduction",
        "resolved_config": config, "source_hashes": hashes, "device": str(device),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": "float32", "model_set": "one qualified PARSeq; frozen segmentation exports",
        "shape": [config["batch_size"], 3, 32, 128], "timing": "synchronized warm batch wall seconds including preprocessing, forward and CPU token decode; no shared-backbone or FLOP claim",
        "selection": "no thresholds fitted; reused TRAIN holdout; GT oracles diagnostic only",
        "matching": "maximum one-to-one bbox IoU>=.5 + exact Unicode casefold; punctuation retained; all care words"}
    write_json(output / "protocol.json", protocol)
    results, memory = {}, {}
    recognizer = None
    try:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_archive, device=str(device)))
        for mode, (ar, refinement) in config["decode_modes"].items():
            # Explicit experimental upstream flags, isolated from production adapter/defaults.
            recognizer.model.model.decode_ar = ar; recognizer.model.model.refine_iters = refinement
            recognizer.config = replace(recognizer.config, decode_ar=ar, refine_iters=refinement)
            views = list(arms)  # Identical crop order/batch composition in every decoding mode.
            digests = sorted({p["crop_rgb_sha256"] for view in views for s in arms[view] for p in s["predictions"]})
            def load(digest):
                with Image.open(unique[digest]) as image: crop = np.asarray(image.convert("RGB"))
                if crop_array_hash(crop) != digest: raise ValueError("Frozen RGB changed before inference")
                return crop
            warm = [load(d) for d in digests[:config["batch_size"]]]
            for _ in range(3): recognizer.recognize_candidate_batch(warm)
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            records, batch_times = {}, []
            for start in range(0, len(digests), config["batch_size"]):
                batch = digests[start:start + config["batch_size"]]
                crops = [load(d) for d in batch]
                torch.cuda.synchronize(device); started = time.perf_counter()
                predictions = recognizer.recognize_candidate_batch(crops)
                torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
                batch_times.append({"size": len(batch), "seconds": elapsed, "digests": batch})
                for digest, prediction in zip(batch, predictions):
                    records[digest] = prediction.to_dict()
                if start % 256 == 0: print(f"[{mode}] {min(start + len(batch), len(digests))}/{len(digests)} crops", flush=True)
            write_json(output / (mode + "_readings.json"), records)
            seconds = sum(b["seconds"] for b in batch_times)
            memory[mode] = {"allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.max_memory_reserved(device), "oom": False,
                "unique_crops": len(digests), "seconds": seconds, "seconds_per_crop": seconds / len(digests),
                "decode_ar": ar, "refine_iters": refinement, "batch_times": batch_times}
            for view in views:
                samples = []
                for s in arms[view]:
                    predictions = [{**p, **records[p["crop_rgb_sha256"]],
                        "text": records[p["crop_rgb_sha256"]]["raw_text"]} for p in s["predictions"]]
                    samples.append({**s, "predictions": predictions})
                results[view + "/" + mode] = summarize(samples)
            print(f"[{mode}] done; seconds/crop={seconds / len(digests):.5f}", flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "error": str(error),
            "traceback": traceback.format_exc(), "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "allocated_bytes": torch.cuda.max_memory_allocated(device), "reserved_bytes": torch.cuda.max_memory_reserved(device)})
        raise
    finally:
        if recognizer is not None: recognizer.release()
    reference = results["pred/axis/ar1"]
    old = source_ocr["arms"]["control"]
    old_reordered = {**old, "per_image": sorted(old["per_image"], key=lambda s:s["sample_id"])}
    new_reordered = {**reference, "per_image": sorted(reference["per_image"], key=lambda s:s["sample_id"])}
    replay_changes = paired_changes(old_reordered, new_reordered)
    if replay_changes["recovered"] or replay_changes["damaged"]:
        raise ValueError("AR1 baseline no longer reproduces frozen per-word successes")
    old_readings = {s["sample_id"]: {p["instance_id"]: p for p in s["predictions"]} for s in old["per_image"]}
    for s in reference["per_image"]:
        for p in s["predictions"]:
            previous = old_readings[s["sample_id"]][p["instance_id"]]
            if p["text"] != previous["text"] or p["crop_rgb_sha256"] != previous["crop_rgb_sha256"]:
                raise ValueError("Baseline raw OCR or pixels changed on replay")
    changes = {name: paired_changes(reference, result) for name, result in results.items()}
    gate = routing(results["pred/axis/nar1"], reference, config)
    gt_reference = results["gt/axis/ar1"]
    failures = []
    for pred, gt in zip(reference["per_image"], gt_reference["per_image"]):
        a, b = [set(s["matched_gt_indices"]["exact_casefold_iou_ge_50"]) for s in (pred, gt)]
        geometry = set(pred["matched_gt_indices"]["geometry_iou_ge_50"])
        for i, word in enumerate(pred["ground_truth"]):
            failures.append({"sample_id": pred["sample_id"], "gt_index": i, "ground_truth": word,
                "baseline_correct": i in a, "baseline_localized": i in geometry, "gt_axis_correct": i in b})
    for path, digest in hashes.items(): check_hash(Path(path), digest)
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "protocol": protocol,
        "arms": results, "changes_vs_pred_axis_ar1": changes, "routing_proxy": gate,
        "per_word_diagnostic": failures, "execution": memory, "baseline_replay": replay_changes}
    write_json(output / "results.json", result)
    lines = ["# AdaSpot Phase 0 — approximate reproduction", "",
        "Exploratory reused TRAIN holdout; bbox proxy. GT/oracle arms are diagnostics only. NAR/AR modes use the SAME PARSeq weights, not AdaSpot heads. RGB rectification is not feature Mask-Align.", "",
        "| Arm | Exact / GT | Localized | Proposals | Hmean | Recovered / damaged vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in results.items():
        a, ch = values["aggregate"], changes[name]
        lines.append(f"| {name} | {a['exact_casefold_iou_ge_50']} / {a['gt_words']} | {a['geometry_iou_ge_50']} | {a['proposals']} | {a['hmean']:.2%} | {ch['recovered']} / {ch['damaged']} |")
    lines += ["", "| Mode | Warm synchronized seconds / unique crop | Peak allocated / reserved GiB |", "| --- | ---: | ---: |"]
    for name, m in memory.items(): lines.append(f"| {name} | {m['seconds_per_crop']:.6f} | {m['allocated_bytes']/2**30:.3f} / {m['reserved_bytes']/2**30:.3f} |")
    lines += ["", "| Escalated | Confidence exact | Random mean exact | GT-priority diagnostic exact |", "| --- | ---: | ---: | ---: |"]
    for row in gate["rows"]:
        lines.append(f"| {row['escalated']} ({row['fraction']:.1%}) | {row['confidence']['exact_casefold_iou_ge_50']} | {row['random_mean_exact']:.2f} | {row['gt_priority_diagnostic']['exact_casefold_iou_ge_50']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k:v["aggregate"] for k,v in results.items()}, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["approximate"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--parseq-source", type=Path, required=True)
    p.add_argument("--parseq-archive", type=Path, required=True)
    p.add_argument("--parseq-checkpoint", type=Path, required=True)
    p.add_argument("--device", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
