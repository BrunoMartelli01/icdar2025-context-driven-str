"""Controlled full-image residual refinement battery; APPROXIMATE mode only."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from localize.word_multitask import word_instances, parseq_word_proposals
from models.mask_probability_refiner import MaskProbabilityRefiner
from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _geometry_match, _correct, _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_evidence_suite import summarize
from scripts.run_word_instance_extensions import write_json, split_ids, selection_key, model_state
from segmentation.mask_probability_refiner import RefinerDataset, refiner_loss, pixel_counts
from segmentation.totaltext_image_frequency_augment import letterbox_geometry
from segmentation.totaltext_word_kernel import make_word_kernels
from segmentation.training import seed_everything, runtime_environment
from segmentation.word_instance_data import EvidenceSource


REPO = Path(__file__).resolve().parents[1]
ARM_NAMES = ["shallow_balanced", "dilated_balanced", "unet_balanced", "unet_error_focus",
             "unet_corruption", "unet_iterative"]


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO/"configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE namespace required")
    c = json.loads(path.read_text())
    if (c.get("experiment_kind") != "mask_probability_refiner_v1" or c.get("schema_version") != 1
            or c.get("reproduction_mode") != mode or c.get("claim_label") != "approximate reproduction"
            or list(c.get("arms", {})) != ARM_NAMES):
        raise ValueError("Unsupported refiner experiment configuration")
    for k in ("seed", "epochs", "batch_size", "cpu_threads", "native_size", "training_images",
              "evaluation_images", "calibration_images", "expected_evaluation_words"):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError("Invalid positive integer: "+k)
    if (c["native_size"] % 8 or c["evaluation_iou_threshold"] != .5
            or c["calibration_epochs"] != sorted(set([0, *c["calibration_epochs"], c["epochs"]]))
            or any(type(e) is not int or not 0 <= e <= c["epochs"] for e in c["calibration_epochs"])
            or not 0 < c["calibration_images"] < c["training_images"]):
        raise ValueError("Invalid geometry, split, metric or checkpoint schedule")
    for k in ("learning_rate", "minimum_learning_rate", "weight_decay", "epsilon", "max_logit_delta",
              "dice_weight", "preservation_weight", "error_weight", "gradient_clip_norm", "distance_fraction"):
        if not np.isfinite(c[k]) or c[k] <= 0:
            raise ValueError("Invalid finite positive value: "+k)
    if not 0 < c["epsilon"] < .5 or not 0 < c["distance_fraction"] < 1 or c["minimum_learning_rate"] > c["learning_rate"]:
        raise ValueError("Invalid numerical limits")
    for name, a in c["arms"].items():
        if (a["architecture"] not in ("shallow", "dilated", "unet") or type(a["width"]) is not int
                or a["width"] < 4 or a["width"] % 4 or type(a["steps"]) is not int or a["steps"] < 1
                or type(a["error_focus"]) is not bool or not 0 <= a["corruption_probability"] <= 1):
            raise ValueError("Invalid arm: "+name)
    return c


def status(root, stage, **details):
    write_json(root/"progress.json", {"status": "RUNNING", "stage": stage,
        "updated_utc": datetime.now(timezone.utc).isoformat(), **details})


def build_model(c, arm):
    a = c["arms"][arm]
    return MaskProbabilityRefiner(a["architecture"], width=a["width"], steps=a["steps"],
        text_threshold=c["crop"]["text_threshold"], kernel_threshold=c["crop"]["kernel_threshold"],
        epsilon=c["epsilon"], max_logit_delta=c["max_logit_delta"])


def prepare(source, c, root):
    records, manifest = {}, {}
    for index, sid in enumerate(source.ids("training")+source.ids("validation")):
        rgb, m, _ = source.load(sid, decode=False)
        n = c["native_size"]
        if m["probabilities"].shape != (2, n, n):
            raise ValueError("Explicit native grid differs from source: "+sid)
        p = m["probabilities"].astype(np.float32)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Nonfinite/out-of-range source probabilities")
        canvas, valid = m["canvas"].astype(bool), m["validity"].astype(bool)
        if np.any(valid & ~canvas):
            raise ValueError("Qualified valid target outside canvas")
        kernels, _ = make_word_kernels(m["owners"], distance_fraction=c["distance_fraction"])
        pixels = np.asarray(letterbox_geometry(rgb, fill=(0, 0, 0)).resize((n, n), Image.Resampling.BILINEAR)).copy()
        records[sid] = {"rgb": torch.from_numpy(pixels).permute(2, 0, 1).contiguous(),
            "probabilities": torch.from_numpy(p), "target": torch.from_numpy(np.stack((m["text_target"], kernels > 0))).bool(),
            "validity": torch.from_numpy(valid)[None], "canvas": torch.from_numpy(canvas)[None]}
        gt, annotation_sha = _train_gt(source.samples[sid], source.word_root, rgb.size)
        manifest[sid] = {"image_sha256": source.samples[sid].image_sha256, "annotation_sha256": annotation_sha,
                         "care_words": len(gt), "valid_pixels": int(valid.sum())}
        if index % 100 == 0:
            status(root, "prepare", images=index+1)
            print(f"Prepared {index+1}/1255 images", flush=True)
    write_json(root/"input_manifest.json", manifest)
    return records, manifest


def refine_maps(model, rgb, probabilities, canvas, device):
    """Inference-only boundary: no GT mask, owner, validity, count or annotation."""
    if model is None:
        return probabilities.numpy().copy()
    with torch.inference_mode():
        return model(rgb[None].to(device).float()/127.5-1, probabilities[None].to(device),
                     canvas[None].to(device))[-1]["probabilities"][0].cpu().numpy()


def freeze_predictions(source, records, ids, model, c, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    frozen = []
    for sid in ids:
        r = records[sid]
        p = refine_maps(model, r["rgb"], r["probabilities"], r["canvas"], device)
        # Localization is completed before consulting scoring validity or GT.
        labels, audit, _ = word_instances(p, r["canvas"][0].numpy(), source.post)
        image = source.image_path(sid)
        with Image.open(image) as opened:
            wh = opened.size
        proposals = parseq_word_proposals(labels, wh, padding=0, min_area=0, max_instances=None)
        for prop in proposals:
            prop["ignored"] = not bool(((labels == prop["instance_id"]) & r["validity"][0].numpy()).any())
        path = directory/(sid.replace(":", "__")+".npz")
        np.savez_compressed(path, probabilities=p, labels=labels, canvas=r["canvas"][0].numpy())
        row = {"sample_id": sid, "predictions": proposals, "audit": audit, "maps_sha256": sha256_file(path)}
        write_json(path.with_suffix(".json"), row)
        frozen.append(row)
    return frozen


def recognize_crop(crop, recognizer, root, cache_source, device):
    key = hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
    identity = {"crop_sha256": key, "parseq": recognizer._provenance, "device": str(device), "batch_size": 1}
    path = root/"ocr_cache"/(key+".json")
    old = cache_source/(key+".json")
    if path.exists() or old.exists():
        cached = json.loads((path if path.exists() else old).read_text())
        if cached["identity"] != identity:
            raise ValueError("Existing OCR cache contract differs")
    else:
        cached = {"identity": identity, "prediction": recognizer.recognize(crop, input_role="T3_localized_crop").to_dict()}
    if not path.exists():
        write_json(path, cached)
    return cached["prediction"], key


def evaluate(source, records, ids, model, recognizer, c, device, directory, root, cache_source):
    frozen = freeze_predictions(source, records, ids, model, c, device, directory/"predictions")
    rows = []
    thresholds = [c["crop"]["text_threshold"], c["crop"]["kernel_threshold"]]
    for row in frozen:
        sid = row["sample_id"]
        rgb, m, _ = source.load(sid, decode=False)
        image = np.asarray(rgb)
        for prop in row["predictions"]:
            x1, y1, x2, y2 = prop["bbox"]
            reading, key = recognize_crop(image[y1:y2, x1:x2].copy(), recognizer, root, cache_source, device)
            prop.update(text=reading["raw_text"], crop_sha256=key)
        gt = source.gt(sid, m)
        care = [p for p in row["predictions"] if not p["ignored"]]
        geometry, exact = _geometry_match(care, gt, .5), _correct(care, gt, normalize=False)
        with np.load(directory/"predictions"/(sid.replace(":", "__")+".npz")) as maps:
            pixels = pixel_counts(maps["probabilities"], records[sid]["target"].numpy(), m["validity"], thresholds)
        rows.append({**row, "ground_truth": gt, "proposals": len(care), "gt_words": len(gt),
            "geometry_correct": len(geometry), "ocr_correct": len(exact), "pixel_counts": pixels,
            "matched_geometry_gt": list(map(int, geometry)), "matched_ocr_gt": list(map(int, exact))})
    aggregate = summarize(rows)
    pixel_micro = []
    for channel in range(2):
        counts = {key: sum(r["pixel_counts"][channel][key] for r in rows) for key in ("tp", "fp", "fn")}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        pixel_micro.append({**counts, "iou": tp/(tp+fp+fn) if tp+fp+fn else 1.,
                            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.})
    result = {"aggregate": aggregate, "pixel_micro_text_kernel": pixel_micro, "per_image": rows}
    write_json(directory/"metrics.json", result)
    return result


def memory(device, c):
    return {"gpu": torch.cuda.get_device_name(device), "precision": "float32", "device": str(device),
        "training_batch": c["batch_size"], "input_shapes": {"rgb": [c["batch_size"], 3, c["native_size"], c["native_size"]],
        "probabilities": [c["batch_size"], 2, c["native_size"], c["native_size"]]},
        "model_set": "one trainable small refiner plus frozen PARSeq; original segmentation maps cached on CPU",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "oom": False}


def train_arm(source, records, splits, arm, recognizer, c, device, root, baseline, cache_source, smoke):
    seed_everything(c["seed"], deterministic_algorithms=True)
    directory = root/arm
    directory.mkdir(exist_ok=True)
    model = build_model(c, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    epochs = 1 if smoke else c["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=c["minimum_learning_rate"])
    best, best_epoch, best_state = selection_key(baseline), None, None
    history, start, total_steps = [], 0, 0
    contract_sha = sha256_file(root/"run_contract.json")
    checkpoint = directory/"last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract_sha256"] != contract_sha or saved["arm"] != arm:
            raise ValueError("Resume configuration/source/arm changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, total_steps, history = saved["epoch"], saved["total_steps"], saved["history"]
        best, best_epoch, best_state = tuple(saved["best"]), saved["best_epoch"], saved["best_state"]
    else:
        # Full population identity qualification, including thresholding and watershed.
        for sid in splits["fit"]+splits["calibration"]+splits["evaluation"]:
            r = records[sid]
            p = refine_maps(model.eval(), r["rgb"], r["probabilities"], r["canvas"], device)
            if not np.array_equal(p, r["probabilities"].numpy()):
                raise RuntimeError("Refiner does not preserve source maps exactly at initialization: "+sid)
        write_json(directory/"identity.json", {"status": "PASS", "images": sum(map(len, splits.values())),
            "exact_map_identity": True, "trainable_parameters": sum(p.numel() for p in model.parameters())})
    for epoch in range(start+1, epochs+1):
        began = time.monotonic()
        model.train()
        dataset = RefinerDataset(records, splits["fit"], seed=c["seed"], epoch=epoch,
                                 corruption_probability=c["arms"][arm]["corruption_probability"])
        loader = DataLoader(dataset, batch_size=c["batch_size"], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(c["seed"]+epoch), drop_last=False)
        losses, seen, corrupted, skipped, steps, max_gradient = [], [], 0, 0, 0, 0.
        for batch in loader:
            seen.extend(batch["sample_id"])
            corrupted += int(batch["corrupted"].sum())
            if not bool(batch["validity"].any()):
                skipped += 1
                continue
            rgb, base, target, valid, canvas = [batch[k].to(device) for k in
                ("rgb", "probabilities", "target", "validity", "canvas")]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(rgb, base, canvas)
            loss = refiner_loss(outputs, base, target, valid, error_focus=c["arms"][arm]["error_focus"],
                dice_weight=c["dice_weight"], preservation_weight=c["preservation_weight"], error_weight=c["error_weight"])
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip_norm"], error_if_nonfinite=True)
            max_gradient = max(max_gradient, float(gradient))
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        if sorted(seen) != sorted(splits["fit"]) or len(seen) != len(set(seen)) or not steps:
            raise RuntimeError("Fit image population/update invariant failed")
        total_steps += steps
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_seconds = time.monotonic()-began
        report = None
        if smoke or epoch in c["calibration_epochs"]:
            status(root, "calibration", arm=arm, epoch=epoch, epochs=epochs)
            report = evaluate(source, records, splits["calibration"], model.eval(), recognizer, c, device,
                              directory/("calibration_epoch"+str(epoch)), root, cache_source)
            if selection_key(report) > best:
                best, best_state, best_epoch = selection_key(report), model_state(model), epoch
        history.append({"epoch": epoch, "mean_loss": sum(losses)/len(losses), "optimizer_steps": steps,
            "training_seconds": train_seconds, "learning_rate": lr, "seen_ids": seen, "corrupted_images": corrupted,
            "skipped_all_invalid_batches": skipped, "max_gradient_norm_before_clip": max_gradient,
            "calibration": report["aggregate"] if report else None})
        saved = {"arm": arm, "epoch": epoch, "total_steps": total_steps, "model": model_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
            "best": best, "best_epoch": best_epoch, "best_state": best_state, "contract_sha256": contract_sha}
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(checkpoint)
        if epoch in c["calibration_epochs"] or smoke:
            torch.save({"model": saved["model"], "arm": arm, "epoch": epoch, "contract_sha256": contract_sha},
                       directory/("epoch"+str(epoch)+".pt"))
        write_json(directory/"training.json", {"history": history, "epochs": epoch, "total_steps": total_steps,
            "selected_epoch": best_epoch, "selected_off": best_state is None, "memory": memory(device, c)})
        status(root, "training", arm=arm, epoch=epoch, epochs=epochs, selected_epoch=best_epoch,
               calibration=report["aggregate"] if report else None)
        print(f"[{arm}] {epoch}/{epochs}, loss={history[-1]['mean_loss']:.5f}, "
              f"{steps} updates, train={train_seconds:.1f}s, selected={best_epoch}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        torch.save({"model": best_state, "arm": arm, "epoch": best_epoch, "contract_sha256": contract_sha}, directory/"selected.pt")
    return model.eval() if best_state is not None else None, {
        "enabled": best_state is not None, "epoch": best_epoch, "last_sha256": sha256_file(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()), "total_steps": total_steps}


def validate_population(report, expected_words, expected_images):
    rows = report["per_image"]
    if len(rows) != expected_images or len({r["sample_id"] for r in rows}) != expected_images or report["aggregate"]["gt_words"] != expected_words:
        raise RuntimeError("Scoring population or care-word denominator changed")


def run_locked(args, c, root):
    recognizer, stage = None, "preflight"
    device = torch.device(args.device)
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("This qualified battery requires explicit available CUDA; no fallback")
        torch.set_num_threads(c["cpu_threads"])
        cv2.setNumThreads(1)
        seed_everything(c["seed"], deterministic_algorithms=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True)
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
        write_json(root/"hardware_invocation.json", {"hardware": hardware, "processes": processes})
        source = EvidenceSource(args.evidence_root, args.word_root, c)
        source_config = json.loads((Path(source.source_contract["inputs"]["control_run"])/"run_manifest.json").read_text())["run_contract"]["resolved_config"]
        if c["distance_fraction"] != source_config["targets"]["distance_fraction"]:
            raise ValueError("Kernel target changed from qualified source")
        splits = split_ids(source, c)
        contract = {"resolved_config": c, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "result_class": "Smoke-test result" if args.action == "smoke" else "approximate reproduction",
            "splits": splits, "device": str(device), "environment": runtime_environment(), "smoke_images": args.smoke_images,
            "inputs": {k: str(getattr(args, k).resolve()) for k in ("evidence_root", "word_root", "parseq_source", "parseq_checkpoint", "parseq_archive", "ocr_cache_source")},
            "parseq_checkpoint_sha256": sha256_file(args.parseq_checkpoint), "parseq_archive_sha256": sha256_file(args.parseq_archive),
            "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for folder in ("models", "segmentation", "recognition", "localize", "scripts") for p in (REPO/folder).glob("*.py")}}
        contract_path = root/"run_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Resume contract differs; use new output root")
        write_json(contract_path, contract)
        records, manifest = prepare(source, c, root)
        if sum(manifest[s]["care_words"] for s in splits["evaluation"]) != c["expected_evaluation_words"]:
            raise RuntimeError("Source evaluation words changed")
        write_json(root/"preflight.json", {"status": "PASS", "images": {k: len(v) for k, v in splits.items()}, "all_source_maps_rgb_annotations_verified": True})
        if args.action == "preflight":
            return
        if args.action == "smoke":
            splits = {"fit": splits["fit"][:args.smoke_images], "calibration": splits["calibration"][:1],
                      "evaluation": splits["evaluation"][:1]}
            write_json(root/"smoke_population.json", splits)
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_archive, device=str(device)))
        recognizer.model.eval().requires_grad_(False)
        parseq_before = model_state(recognizer.model)
        stage = "baseline"
        baseline = evaluate(source, records, splits["calibration"], None, recognizer, c, device,
                            root/"baseline/calibration", root, args.ocr_cache_source)
        if args.action != "smoke" and (baseline["aggregate"]["geometry_correct"], baseline["aggregate"]["ocr_correct"], baseline["aggregate"]["proposals"]) != (622, 525, 720):
            raise RuntimeError("Previous calibration reference does not replay exactly")
        results, choices = {}, {}
        # Fit/select EVERY arm before reading any held-out result in this battery.
        for arm in c["arms"]:
            stage = arm
            status(root, "training", arm=arm, epoch=0, epochs=c["epochs"])
            model, choice = train_arm(source, records, splits, arm, recognizer, c, device,
                                      root, baseline, args.ocr_cache_source, args.action == "smoke")
            choices[arm] = choice
            write_json(root/"choices.json", choices)
            del model
        stage = "evaluation"
        status(root, stage)
        results["baseline"] = evaluate(source, records, splits["evaluation"], None, recognizer, c, device,
                                      root/"baseline/evaluation", root, args.ocr_cache_source)
        if args.action != "smoke" and (results["baseline"]["aggregate"]["geometry_correct"], results["baseline"]["aggregate"]["ocr_correct"], results["baseline"]["aggregate"]["proposals"]) != (647, 520, 1037):
            raise RuntimeError("Previous evaluation reference does not replay exactly")
        for arm, choice in choices.items():
            model = None
            if choice["enabled"]:
                model = build_model(c, arm).to(device)
                saved = torch.load(root/arm/"selected.pt", map_location=device, weights_only=False)
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
            results[arm] = evaluate(source, records, splits["evaluation"], model, recognizer, c, device,
                                    root/arm/"evaluation", root, args.ocr_cache_source)
            del model
        base = {r["sample_id"]: set(r["matched_ocr_gt"]) for r in results["baseline"]["per_image"]}
        for arm, result in results.items():
            expected_words = sum(manifest[s]["care_words"] for s in splits["evaluation"])
            validate_population(result, expected_words, len(splits["evaluation"]))
            gained, lost = [], []
            for row in result["per_image"]:
                sid, new = row["sample_id"], set(row["matched_ocr_gt"])
                gained.extend([sid, i] for i in sorted(new-base[sid]))
                lost.extend([sid, i] for i in sorted(base[sid]-new))
            result["paired_vs_baseline"] = {"gained": gained, "lost": lost}
        if any(not torch.equal(v, recognizer.model.state_dict()[k].cpu()) for k, v in parseq_before.items()):
            raise RuntimeError("Frozen PARSeq state changed")
        # Reverify source evidence and raw input identities after all training/scoring.
        for sid, item in manifest.items():
            rgb, _, _ = source.load(sid, decode=False)
            if _train_gt(source.samples[sid], source.word_root, rgb.size)[1] != item["annotation_sha256"]:
                raise RuntimeError("Annotation changed during experiment")
        output = {"status": "SUCCESS", "result_class": contract["result_class"], "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction", "arms": results, "choices": choices,
            "memory": memory(device, c), "metric_protocol": c["metric_protocol"], "frozen_parseq_unchanged": True,
            "contract_sha256": sha256_file(contract_path)}
        write_json(root/"results.json", output)
        lines = ["# Residual mask refinement — approximate reproduction", "", "| Arm | Selected epoch | Localized | Exact OCR | Proposals | OCR Hmean | Gained/lost |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for arm, result in results.items():
            a, paired = result["aggregate"], result["paired_vs_baseline"]
            epoch = choices.get(arm, {}).get("epoch")
            lines.append(f"| {arm} | {epoch if epoch is not None else 'off / source'} | {a['geometry_correct']} | {a['ocr_correct']} | {a['proposals']} | {a['ocr_hmean']:.4%} | {len(paired['gained'])}/{len(paired['lost'])} |")
        lines += ["", "Single seed, reused TRAIN holdout; no independent generalization or paper-result claim.",
                  "Off rows replay the source; discarded trained heads were measured on calibration only."]
        (root/"comparison.md").write_text("\n".join(lines)+"\n")
        write_json(root/"progress.json", {"status": "SUCCESS", "stage": "smoke_complete" if args.action == "smoke" else "complete",
            "updated_utc": datetime.now(timezone.utc).isoformat(), "results_sha256": sha256_file(root/"results.json")})
    except Exception as error:
        write_json(root/"failure.json", {"status": "FAILED", "stage": stage, "error": str(error),
            "type": type(error).__name__, "oom": isinstance(error, torch.OutOfMemoryError), "traceback": traceback.format_exc()})
        write_json(root/"progress.json", {"status": "FAILED", "stage": stage})
        raise
    finally:
        if recognizer is not None:
            recognizer.release()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["approximate"])
    p.add_argument("--config", required=True, type=Path)
    for name in ("evidence-root", "word-root", "parseq-source", "parseq-checkpoint", "parseq-archive", "ocr-cache-source", "output-root"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--device", required=True)
    p.add_argument("--action", required=True, choices=["preflight", "smoke", "all"])
    p.add_argument("--smoke-images", type=int, default=8)
    return p


def run(args):
    c = read_config(args.config, args.mode)
    if not 1 <= args.smoke_images <= c["training_images"]-c["calibration_images"]:
        raise ValueError("Invalid smoke population")
    root = args.output_root.resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError("Generated results must be outside repository")
    for name in ("evidence_root", "word_root", "parseq_source", "ocr_cache_source"):
        source = getattr(args, name).resolve()
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Output/input roots must be isolated")
    root.mkdir(parents=True, exist_ok=True)
    with (root/"runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, c, root)


if __name__ == "__main__":
    run(parser().parse_args())
