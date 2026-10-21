"""Train and test genuine shared-feature AdaSpot; explicit APPROXIMATE mode."""

import argparse
from collections import Counter
import gc
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from localize.adaspot_gap import gap_instances
from localize.word_multitask import parseq_word_proposals
from models.adaspot_joint import AdaSpotJoint, qualified_parseq_parts
from models.adaspot_mask_align import MaskAlignOptions
from recognition.adaspot_gate import fit_temperature, calibration_metrics, prediction_correctness, route_cached
from recognition.scene_text_datasets import parse_totaltext_annotation
from scripts.audit_adaspot_phase0 import count
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, summarize, write_json
from scripts.evaluate_totaltext_highres_contour_padding import _ground_truth, _annotation_path
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import memory
from scripts.train_adaspot_gap_fulltrain import ignored_instances, validate_epoch
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.adaspot_joint import match_owner_texts, recognition_objective, segmentation_objective
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.training import seed_everything


def partition(ids, *, seed, calibration_images):
    if len(set(ids)) != len(ids) or not 2 <= calibration_images < len(ids):
        raise ValueError("Unique TRAIN IDs and nonempty fit/calibration partition required")
    ordered = sorted(ids, key=lambda sid: sha256(f"{seed}:adaspot-joint:{sid}".encode()).hexdigest())
    calibration, fit = ordered[:calibration_images], ordered[calibration_images:]
    split = calibration_images // 2
    return fit, calibration[:split], calibration[split:]


def prepare(args):
    cp = args.config.resolve()
    c = json.loads(cp.read_text())
    if args.mode != "approximate" or c["reproduction_mode"] != args.mode or REPO / "configs/approximate" not in cp.parents:
        raise ValueError("Explicit isolated approximate joint configuration required")
    if c["embed_dim"] != 384 or c["num_heads"] != 12 or c["epochs"] < 1 or c["max_training_rois"] < 1:
        raise ValueError("Qualified PARSeq embedding and positive training budget required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA GPU required; no CPU inference/training fallback")
    torch.cuda.init()
    torch.set_num_threads(c["cpu_threads"])
    cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = args.source_run.resolve()
    old = json.loads((source / "protocol.json").read_text())
    audit = json.loads((source / "independent_audit.json").read_text())
    if audit["status"] != "PASS":
        raise ValueError("Audited full-training source required")
    check_hash(source / "results.json", audit["result_sha256"])
    hashes = dict(old["source_hashes"])
    for path, expected in hashes.items():
        check_hash(Path(path), expected)
    files = [cp, Path(__file__), source / "protocol.json", source / "results.json", source / "independent_audit.json"]
    files += [REPO / name for name in ("models/adaspot_joint.py", "models/adaspot_mask_align.py", "recognition/adaspot_gate.py",
        "segmentation/adaspot_joint.py", "tests/models/test_adaspot_joint_components.py", "scripts/audit_adaspot_joint.py")]
    for path in files:
        hashes[str(path.resolve())] = sha256_file(path)
    train = load_segmentation_manifest(Path(old["train_manifest"]), require_checksums=True)
    test = load_segmentation_manifest(Path(old["test_manifest"]), require_checksums=True)
    if (train.split_role != "train" or test.split_role != "test" or len(train.samples) != c["train_images"]
            or len(test.samples) != c["test_images"]):
        raise ValueError("Complete frozen official TRAIN/TEST manifests required")
    fit, temperature_ids, threshold_ids = partition([s.sample_id for s in train.samples], seed=c["seed"], calibration_images=c["calibration_images"])
    if args.stage == "smoke":
        fit, temperature_ids, threshold_ids = fit[:2], temperature_ids[:2], threshold_ids[:2]
    out = args.output_root.resolve()
    if any(out == root or root in out.parents for root in (REPO, source, Path(old["word_root"]), Path(old["test_root"]))):
        raise ValueError("New external output root required")
    out.mkdir(parents=True, exist_ok=False)
    args._output_created = True
    protocol = {"mode": "approximate", "result_class": "Smoke-test result" if args.stage == "smoke" else c["claim_label"],
        "stage": args.stage, "resolved_config": c, "source_hashes": hashes, "source_run": str(source),
        "source_checkpoint": old["source_checkpoint"], "word_root": old["word_root"], "test_root": old["test_root"],
        "train_manifest": old["train_manifest"], "test_manifest": old["test_manifest"],
        "parseq_source": old["parseq_source"], "parseq_archive": old["parseq_archive"], "parseq_checkpoint": old["parseq_checkpoint"],
        "resolved_preprocessing": old["resolved_preprocessing"], "fit_ids": fit, "temperature_ids": temperature_ids,
        "threshold_ids": threshold_ids, "selected_epoch": 1 if args.stage == "smoke" else c["epochs"],
        "test_used_for_selection": False, "test_historically_explored": True,
        "known_official_rgb_overlap": old["resolved_config"]["known_official_rgb_overlap"],
        "device": str(device), "gpu": torch.cuda.get_device_name(device), "dtype": "float32", "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "deterministic_algorithms": True, "tf32": False,
        "seed": c["seed"], "cublas_workspace_config": ":4096:8",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "opencv-python")}}
    write_json(out / "protocol.json", protocol)
    return c, device, protocol, out, train, test


def training_annotations(manifest, root, hashes):
    rows = {}
    for sample in manifest.samples:
        path = root / "datasets/total_text/train/annotations" / (sample.sample_id.split(":", 1)[1] + ".txt")
        hashes[str(path)] = sha256_file(path)
        care = [item for item in parse_totaltext_annotation(path) if not item.ignore]
        with Image.open(root / sample.image_path) as image:
            wh = image.size
        rows[sample.sample_id] = {"texts": {i: item.transcription for i, item in enumerate(care, 1)}, "image_wh": wh}
    return rows


def predicted_rois(evidence, batch, c, image_wh):
    probabilities = evidence["words"][0].detach().sigmoid().cpu().numpy()
    gap = evidence["gap"][0, 0].detach().sigmoid().cpu().numpy()
    canvas = letterbox_canvas_mask(image_wh, native_size=probabilities.shape[-1]).numpy()
    labels, audit, _ = gap_instances(probabilities, gap, canvas, c["postprocessing"], use_gap=True)
    # Geometry is fixed first. Care/ignore admission then follows prior evaluation.
    invalid = ignored_instances(labels, batch["validity"][0].numpy().astype(bool))
    proposals = [p for p in parseq_word_proposals(labels, image_wh, padding=0, min_area=0, max_instances=None)
                 if p["instance_id"] not in invalid]
    return labels, proposals, {**audit, "ignored_instance_ids": invalid}, probabilities, gap


def representative_parameters(model):
    return {"encoder": model.backbone.encoder.conv1.weight, "decoder": model.backbone.up0.block.conv1.weight
        if hasattr(model.backbone.up0.block, "conv1") else next(model.backbone.up0.block.parameters()),
        "text_kernel": model.backbone.head.weight, "gap": model.gap.weight,
        "light": model.light_head.projection.weight, "deep": model.deep_head.projection.weight,
        "gate": model.gate.layers[0].weight}


def train(model, dataset, annotations, p, c, device, out):
    by_id = {s.sample_id: i for i, s in enumerate(dataset.base.manifest.samples)}
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": c["learning_rate_backbone"]},
        {"params": [v for name, v in model.named_parameters() if not name.startswith("backbone.")], "lr": c["learning_rate_heads"]}], weight_decay=c["weight_decay"])
    options = MaskAlignOptions(**c["geometry"])
    history, steps, smoke = [], 0, None
    epochs = 1 if p["stage"] == "smoke" else c["epochs"]
    torch.cuda.reset_peak_memory_stats(device)
    directory = out / "training"
    directory.mkdir()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(np.random.default_rng(c["seed"] + epoch).permutation(len(p["fit_ids"])))
        totals, seen, audit_rows = Counter(), [], []
        started = time.monotonic()
        for offset in order:
            sid = p["fit_ids"][int(offset)]
            batch = dataset[by_id[sid]]
            local = {key: value[None].to(device) for key, value in batch.items() if torch.is_tensor(value)}
            optimizer.zero_grad(set_to_none=True)
            evidence = model.encode(local["image"])
            labels, proposals, _, _, _ = predicted_rois(evidence, batch, c, annotations[sid]["image_wh"])
            seed = int(sha256(f"{c['seed']}:{epoch}:{sid}".encode()).hexdigest()[:16], 16)
            indices = sorted(np.random.default_rng(seed).choice(len(proposals), min(len(proposals), c["max_training_rois"]), replace=False).tolist())
            selected = [proposals[i] for i in indices]
            matched, matching = match_owner_texts(selected, batch["owners_native"].numpy(), annotations[sid]["texts"], iou_threshold=c["training_match_iou"])
            masks = [labels == item["instance_id"] for item in selected]
            seg, seg_terms = segmentation_objective(evidence, local, c)
            ocr, ocr_terms, details = recognition_objective(model, evidence, masks, matched, options=options, config=c)
            loss = seg + ocr
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite complete joint objective")
            tracked = representative_parameters(model)
            before = {key: value.detach().clone() for key, value in tracked.items()} if smoke is None else None
            loss.backward()
            gradients = {key: float(value.grad.abs().sum()) if value.grad is not None else 0. for key, value in tracked.items()} if before is not None else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            if before is not None and details["ocr_supervised"]:
                changes = {key: float((value.detach() - before[key]).abs().max()) for key, value in tracked.items()}
                # The new Gap starts at zero and may receive first-step updates;
                # every scientifically relevant branch must get real gradients.
                if any(not (np.isfinite(gradients[key]) and gradients[key] > 0 and changes[key] > 0) for key in tracked):
                    raise RuntimeError("Joint CUDA gradient/update qualification failed: " + str((gradients, changes)))
                smoke = {"device": str(device), "sample_id": sid, "gradient_l1": gradients, "parameter_max_abs_change": changes,
                    "sampling_backward_device": "cpu", "image_shape": list(local["image"].shape), "rois": len(masks),
                    "light_grid": [8, 32], "deep_grid": [16, 64], "all_parameters_cuda": all(x.is_cuda for x in model.parameters())}
                write_json(out / "first_joint_update.json", smoke)
            steps += 1
            seen.append(sid)
            for key, value in {"loss": loss, **seg_terms, **ocr_terms}.items():
                totals[key] += float(value.detach())
            totals["rois"] += details["rois"]
            totals["ocr_supervised"] += details["ocr_supervised"]
            totals["gate_positive"] += details["gate_positive"]
            audit_rows.append({"sample_id": sid, "proposal_count": len(proposals), "selected_indices": indices,
                "instance_ids": [item["instance_id"] for item in selected], "matching": matching,
                "rois": details["rois"], "ocr_supervised": details["ocr_supervised"],
                "gate_positive": details["gate_positive"], "excluded": details["excluded"]})
            if len(seen) == 1 or len(seen) % 100 == 0:
                print(f"[joint epoch {epoch}/{epochs}] {len(seen)}/{len(order)} images; loss={float(loss):.4f}; OCR={int(totals['ocr_supervised'])}; {time.monotonic()-started:.1f}s", flush=True)
            del evidence, local, loss, seg, ocr, seg_terms, ocr_terms
        validate_epoch(seen, p["fit_ids"])
        row = {"epoch": epoch, "images": len(seen), "sample_ids": seen, "global_cuda_steps": steps,
            "seconds": time.monotonic() - started, "means": {key: value / len(seen) for key, value in totals.items()},
            "totals": dict(totals), "roi_audit": audit_rows}
        write_json(directory / f"epoch_{epoch:03d}.json", row)
        history.append(row)
        print(f"[joint epoch {epoch}] complete; mean loss={row['means']['loss']:.4f}; seconds={row['seconds']:.1f}", flush=True)
    if smoke is None:
        raise RuntimeError("No supervised OCR image qualified joint gradient updates")
    receipt = memory(device, "shared trainable ResNet50 AG-UNet + M/K/G + NAR1+cloze + AR4 + gate", [1, 3, 512, 512])
    checkpoint = directory / "last.pt"
    torch.save({"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs,
        "steps": steps, "resolved_config": c, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(), "fit_ids": p["fit_ids"]}, checkpoint)
    completion = {"status": "SUCCESS", "completed_epochs": epochs, "global_cuda_steps": steps,
        "fit_images": len(p["fit_ids"]), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "history_paths": [str(directory / f"epoch_{i:03d}.json") for i in range(1, epochs+1)], "memory": receipt,
        "final_epoch_fixed_before_test": True}
    write_json(out / "training_complete.json", completion)
    del optimizer, before, tracked
    gc.collect()
    torch.cuda.empty_cache()
    return completion


@torch.no_grad()
def collect(model, dataset, indices, annotation_rows, p, c, device, out, stage):
    model.eval()
    options = MaskAlignOptions(**c["geometry"])
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    for index in indices:
        batch = dataset[index]
        sid = batch["sample_id"]
        with Image.open(batch["image_path"]) as image:
            image_wh = image.size
        evidence = model.encode(batch["image"][None].to(device))
        labels, proposals, audit, probabilities, gap = predicted_rois(evidence, batch, c, image_wh)
        predictions = []
        size = c["evaluation_roi_batch_size"]
        for start in range(0, len(proposals), size):
            values = proposals[start:start+size]
            masks = [labels == v["instance_id"] for v in values]
            aligned, geometries = model.align(evidence, masks, view=c["alignment_view"], options=options, deep=False)
            first, refined = model.light_head.light(model.light_head.memory(aligned))
            cues, light = model.cues(evidence, masks, geometries, first, refined)
            logits = model.gate(cues).cpu().tolist()
            sampled, _ = model.align(evidence, masks, view=c["alignment_view"], options=options, deep=True)
            deep = model.deep_head.read(model.deep_head.greedy(model.deep_head.memory(sampled)))
            predictions.extend({**v, "light": a, "deep": b, "gate_logit": z, "gate_cues": q,
                "alignment": g.audit} for v, a, b, z, q, g in zip(values, light, deep, logits, cues.cpu().tolist(), geometries))
        sample = dataset.manifest.samples[index]
        if dataset.manifest.split_role == "train":
            gt, annotation_sha = _train_gt(sample, Path(p["word_root"]), image_wh)
        else:
            gt, annotation_sha = _ground_truth(sample, Path(p["test_root"]), image_wh)
        folder = out / (stage + "_predictions")
        folder.mkdir(exist_ok=True)
        path = folder / (sid.replace(":", "__") + ".npz")
        np.savez_compressed(path, probabilities=probabilities, gap=gap, labels=labels)
        row = {"sample_id": sid, "ground_truth": gt, "predictions": predictions, "decoder_audit": audit,
            "image_sha256": sample.image_sha256, "annotation_sha256": annotation_sha, "map_path": str(path), "map_sha256": sha256_file(path)}
        row["light_correctness_labels"] = prediction_correctness(row, reading_key="light")
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"[{stage}] {len(rows)}/{len(indices)} images; all-light/all-deep cached", flush=True)
    write_json(out / (stage + "_cache.json"), rows)
    return rows, memory(device, "shared AdaSpot all-light/all-deep diagnostics; evaluation chunks", [1, 3, 512, 512])


def score_arm(samples, name, choice=None):
    if name == "adaptive":
        routed = [route_cached(sample, **choice) for sample in samples]
        rows, decisions = [r[0] for r in routed], [r[1] for r in routed]
    else:
        rows = [{**sample, "predictions": [{**v, "text": v[name]["text"]} for v in sample["predictions"]]} for sample in samples]
        decisions = []
    return summarize(rows), decisions


def calibrate(rows_temperature, rows_threshold, c):
    z = [v["gate_logit"] for s in rows_temperature for v in s["predictions"]]
    y = [v for s in rows_temperature for v in s["light_correctness_labels"]]
    fitted = fit_temperature(z, y, log_bounds=c["temperature_log_bounds"])
    candidates = []
    for reject in c["routing_thresholds"]:
        for keep in c["routing_thresholds"]:
            if reject > keep:
                continue
            for deep in c["routing_thresholds"]:
                choice = {"temperature": fitted["temperature"], "reject_threshold": reject,
                          "keep_threshold": keep, "deep_threshold": deep}
                tp, proposals, gt, escalated = 0, 0, 0, 0
                for sample in rows_threshold:
                    row, decisions = route_cached(sample, **choice)
                    tp += count(row["predictions"], row["ground_truth"], True)
                    proposals += len(row["predictions"])
                    gt += len(row["ground_truth"])
                    escalated += sum(d["route"] == "escalate" for d in decisions)
                candidates.append({"choice": choice, "exact_words": tp, "emitted_predictions": proposals,
                    "gt_words": gt, "deep_calls": escalated, "hmean": 2 * tp / (proposals + gt) if proposals + gt else 0.})
    best = max(candidates, key=lambda row: (row["hmean"], -row["deep_calls"]))
    z2 = [v["gate_logit"] for s in rows_threshold for v in s["predictions"]]
    y2 = [v for s in rows_threshold for v in s["light_correctness_labels"]]
    return {"temperature_fit": fitted, "selection": best, "candidates": candidates,
        "held_out_threshold_gate_metrics": calibration_metrics(z2, y2, temperature=fitted["temperature"], bins=c["calibration_bins"])}


@torch.no_grad()
def selective_replay(model, dataset, index, cached, p, c, device):
    model.eval()
    batch = dataset[index]
    evidence = model.encode(batch["image"][None].to(device))
    with np.load(cached["map_path"]) as saved:
        labels = saved["labels"]
    proposals = cached["predictions"][:c["evaluation_roi_batch_size"]]
    masks = [labels == v["instance_id"] for v in proposals]
    if not masks:
        raise ValueError("Selective qualification sample has no predicted word")
    receipt = []
    policies = [("all_keep", 0., 0.), ("all_reject", 1., 1.), ("all_escalate", 0., 1.)]
    for name, reject, keep in policies:
        calls = []
        hook = model.deep_head.projection.register_forward_pre_hook(lambda module, values: calls.append(len(values[0])))
        try:
            outputs = model.infer_instances(evidence, masks, options=MaskAlignOptions(**c["geometry"]),
                temperature=1., reject_threshold=reject, keep_threshold=keep, deep_threshold=0., view=c["alignment_view"])
        finally:
            hook.remove()
        expected = sum(v["route"] == "escalate" for v in outputs)
        if sum(calls) != expected:
            raise RuntimeError("Deep head executed outside actual escalation set")
        for actual, frozen in zip(outputs, proposals):
            if actual["light"]["text"] != frozen["light"]["text"]:
                raise RuntimeError("Actual selective light replay differs from cache")
            if actual["deep"] is not None and actual["deep"]["text"] != frozen["deep"]["text"]:
                raise RuntimeError("Actual selective deep replay differs from cache")
        receipt.append({"policy": name, "instances": len(outputs), "deep_projection_batches": calls,
            "deep_calls": expected, "routes": [v["route"] for v in outputs], "raw_strings_replayed": True})
    return {"sample_id": cached["sample_id"], "policies": receipt, "scope": "forced policies qualify execution only; not TEST threshold tuning"}


def run(args):
    c, device, p, out, train_manifest, test_manifest = prepare(args)
    parts, tokenizer, settings, state, provenance = qualified_parseq_parts(p["parseq_source"], p["parseq_archive"], p["parseq_checkpoint"])
    if (settings["embed_dim"], settings["dec_num_heads"], settings["dec_mlp_ratio"]) != (c["embed_dim"], c["num_heads"], c["mlp_ratio"]):
        raise ValueError("OCR dimensions differ from qualified PARSeq settings")
    model = AdaSpotJoint(parts, tokenizer, c)
    source = torch.load(p["source_checkpoint"], map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(source["model_state_dict"], strict=True)
    del source
    initialization = {"light": model.light_head.initialize_parseq(state), "deep": model.deep_head.initialize_parseq(state),
        "parseq_provenance": provenance, "source_strict_load": True, "shared_encoder_trainable": True,
        "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad)}
    del state
    write_json(out / "initialization.json", initialization)
    model.to(device)
    transform = _transform(p["resolved_preprocessing"], "letterbox", train=False)
    base_train = SegmentationDataset(train_manifest, Path(p["word_root"]), transform=transform)
    training = AdaSpotGapDataset(base_train, distance_fraction=c["distance_fraction"], gap_radius=c["gap_radius"])
    annotations = training_annotations(train_manifest, Path(p["word_root"]), p["source_hashes"])
    write_json(out / "annotation_mapping.json", annotations)
    completion = train(model, training, annotations, p, c, device, out)
    # Strict checkpoint round trip before any calibration or TEST predictions.
    checkpoint = torch.load(completion["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    by_id = {s.sample_id: i for i, s in enumerate(train_manifest.samples)}
    temperature_rows, _ = collect(model, base_train, [by_id[s] for s in p["temperature_ids"]], annotations, p, c, device, out, "temperature")
    threshold_rows, calibration_memory = collect(model, base_train, [by_id[s] for s in p["threshold_ids"]], annotations, p, c, device, out, "threshold")
    calibrated = calibrate(temperature_rows, threshold_rows, c)
    write_json(out / "calibration.json", calibrated)
    test_dataset = SegmentationDataset(test_manifest, Path(p["test_root"]), transform=transform)
    indices = list(range(2 if args.stage == "smoke" else len(test_manifest.samples)))
    test_rows, test_memory = collect(model, test_dataset, indices, None, p, c, device, out, "test")
    evaluated, routes = {}, None
    for name in ("light", "deep", "adaptive"):
        scored, decisions = score_arm(test_rows, name, calibrated["selection"]["choice"])
        evaluated[name] = scored
        if name == "adaptive":
            routes = decisions
    if args.stage == "full" and sum(len(s["ground_truth"]) for s in test_rows) != c["test_words"]:
        raise ValueError("Complete TEST care denominator changed")
    replay_index = next(i for i, row in enumerate(test_rows) if row["predictions"])
    replay = selective_replay(model, test_dataset, indices[replay_index], test_rows[replay_index], p, c, device)
    write_json(out / "selective_execution.json", replay)
    duplicates = {r["test_id"] for r in p["known_official_rgb_overlap"]}
    reduced = [s for s in test_rows if s["sample_id"] not in duplicates]
    subset = {name: score_arm(reduced, name, calibrated["selection"]["choice"])[0] for name in evaluated}
    z = [v["gate_logit"] for s in test_rows for v in s["predictions"]]
    y = [v for s in test_rows for v in s["light_correctness_labels"]]
    write_json(out / "protocol_final.json", p)
    for path, expected in p["source_hashes"].items():
        check_hash(Path(path), expected)
    result = {"status": "SUCCESS", "result_class": p["result_class"], "stage": args.stage,
        "training": completion, "arms": evaluated, "routes": routes,
        "calibration_choice": calibrated["selection"]["choice"],
        "gate_test_metrics": calibration_metrics(z, y, temperature=calibrated["temperature_fit"]["temperature"], bins=c["calibration_bins"]),
        "hash_disjoint_test_subset": {"excluded_ids": sorted(duplicates), "images": len(reduced), "arms": subset},
        "deep_calls": sum(d["route"] == "escalate" for row in routes for d in row),
        "total_predicted_care_instances": sum(len(s["predictions"]) for s in test_rows),
        "calibration_memory": calibration_memory, "test_memory": test_memory,
        "limitations": "Two-epoch initial joint qualification; no SynthText/TextOCR scene pretraining, convergence, officialpolygon, strict, untouchedTEST, GFLOPs or superior accuracy claim."}
    write_json(out / "results.json", result)
    lines = ["# AdaSpot shared-feature joint prototype", "", "**" + p["result_class"] + "**", "",
        "One shared trainable ResNet50/AG-UNet, Text/Kernel/Gap, true NAR1+cloze and AR4 feature decoders, detached calibrated MLP gate.",
        "Final epoch fixed before TEST;100TRAIN calibration images excluded from all new joint updates (50temperature/50threshold).", "",
        "| Mode | Exact localized+read words | GT words | Emitted predictions | Hmean |", "|---|---:|---:|---:|---:|"]
    for name, value in evaluated.items():
        a = value["aggregate"]
        lines.append(f"| {name} | {a['exact_casefold_iou_ge_50']} | {a['gt_words']} | {a['proposals']} | {a['hmean']:.4%} |")
    lines += ["", f"Adaptive deep calls: {result['deep_calls']}/{result['total_predicted_care_instances']}. Cached always-deep outputs are diagnostic; selective execution separately verified.",
        "Gate TEST metrics: `" + json.dumps(result["gate_test_metrics"], sort_keys=True) + "`.", "",
        result["limitations"], "Known2RGB TRAIN/TEST duplicates and historically exploredTEST are explicitly retained; secondary298 reported separately.",
        "Full routes, unmodified strings, masks, checkpoints, loss/ROI coverage, initialization keys, configuration and memory/OOM receipts are saved."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: value["aggregate"] for name, value in evaluated.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--stage", required=True, choices=["smoke", "full"])
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
