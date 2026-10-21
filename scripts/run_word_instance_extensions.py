"""Three isolated local-word ablations, with matched controls and TRAIN calibration."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from localize.word_instance_extensions import mask_component, affinity_component, support_box, remap_local_box
from models.word_instance_extensions import LocalWordModel
from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from recognition.parseq_crop_objective import bounded_boxes, sample_boxes, frozen_parseq_loss
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _geometry_match, _correct
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_evidence_suite import summarize
from segmentation.training import seed_everything, runtime_environment
from segmentation.word_instance_data import EvidenceSource, training_targets
from segmentation.word_instance_extensions import mask_objective, balanced_edge_loss, maximin_loss


REPO = Path(__file__).resolve().parents[1]
ARMS = {"prompt_control": "mask", "prompt_instance": "mask", "affinity_control": "affinity",
        "affinity_maximin": "affinity", "geometry_control": "geometry", "geometry_ocr": "geometry"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO/"configs/approximate" not in path.parents:
        raise ValueError("Local word extensions require an explicit APPROXIMATE config namespace")
    c = json.loads(path.read_text())
    if (c.get("schema_version") != 1 or c.get("experiment_kind") != "word_instance_extensions_v1"
            or c.get("reproduction_mode") != mode or c.get("claim_label") != "approximate reproduction"
            or c.get("arms") != list(ARMS)):
        raise ValueError("Unsupported local word experiment configuration")
    for key in ("seed", "training_images", "evaluation_images", "calibration_images", "epochs", "channels", "cpu_threads"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError("Invalid positive integer: " + key)
    crop = c["crop"]
    if (not 0 < c["calibration_images"] < c["training_images"] or crop["local_size"] < 8
            or type(crop["local_size"]) is not int or crop["minimum_context_pixels"] < 0
            or crop["context_fraction"] < 0 or not .5 <= crop["owner_majority"] <= 1
            or not 0 < crop["minimum_visible_fraction"] <= 1 or not 0 < c["max_box_fraction"] < .5
            or c["learning_rate"] <= 0 or c["weight_decay"] < 0 or c["maximin_weight"] <= 0
            or c["ocr_weight"] <= 0 or c["anchors_per_word"] < 2):
        raise ValueError("Invalid local geometry, losses or training split")
    for v in (crop["mask_threshold"], crop["text_threshold"], crop["kernel_threshold"], crop["affinity_threshold"]):
        if not 0 < v < 1:
            raise ValueError("Invalid fixed inference threshold")
    if not 0 < c["evaluation_iou_threshold"] <= 1 or any(type(e) is not int or not 0 <= e <= c["epochs"] for e in c["calibration_epochs"]):
        raise ValueError("Invalid evaluation IoU or calibration epoch")
    if c["sampling_device"] != "cpu" or c["ocr_input_size"] != [32, 128] or c["calibration_epochs"] != sorted(set([0, *c["calibration_epochs"], c["epochs"]])):
        raise ValueError("Explicit deterministic sampling and fixed calibration schedule required")
    if not np.isfinite([c["learning_rate"], c["weight_decay"], c["ocr_weight"], c["maximin_weight"], c["max_box_fraction"]]).all():
        raise ValueError("Nonfinite scientific configuration")
    return c


def split_ids(source, config):
    ids = source.ids("training")
    order = np.random.default_rng(config["seed"]).permutation(len(ids))
    calibration = sorted(ids[i] for i in order[:config["calibration_images"]])
    fit = sorted(set(ids)-set(calibration))
    return {"fit": fit, "calibration": calibration, "evaluation": source.ids("validation")}


def source_inputs(packet, arm, device):
    inputs = packet["inputs"].clone()
    # Mask control has the identical selected-word targets and capacity, but no prompt.
    # Both affinity arms have identical GT-free RGB/Text inputs and local edge decoder.
    if arm in ("prompt_control", "affinity_control", "affinity_maximin"):
        inputs[4] = 0
    return inputs[None].to(device)


def objective(model, arm, packet, target, recognizer, config, device):
    inputs = source_inputs(packet, arm, device)
    box = packet["box"][None].to(device)
    output = model(inputs, box)
    details = {"supervised": True, "ocr_supervised": False}
    if model.kind == "mask":
        validity = target["mask_valid"][None, None].to(device)
        details["supervised"] = bool(validity.any())
        loss = mask_objective(output, target["target"][None, None].to(device), validity)
    elif model.kind == "affinity":
        details["supervised"] = bool(np.any((target["owners"] > 0) & target["validity"]))
        loss = balanced_edge_loss(output[0], target["owners"], target["validity"])
        if arm == "affinity_maximin":
            structural = maximin_loss(output[0], target["owners"], target["validity"],
                                       anchors_per_word=config["anchors_per_word"])
            details["maximin_loss"] = float(structural.detach())
            loss = loss + config["maximin_weight"] * structural
    else:
        if target["truth_box"] is None:
            return output.sum()*0, {"supervised": False, "ocr_supervised": False}
        corrected = bounded_boxes(box, output, inputs.shape[-2:], max_fraction=config["max_box_fraction"])
        scale = torch.cat((box[:, 2:]-box[:, :2], box[:, 2:]-box[:, :2]), 1)
        loss = F.smooth_l1_loss((corrected-box)/scale, (target["truth_box"][None].to(device)-box)/scale)
        if arm == "geometry_ocr":
            text = target["text"]
            tokenizer = recognizer.model.tokenizer
            if len(text) > recognizer.model.model.max_label_length:
                details["ocr_unsupervised_reason"] = "label_too_long"
            elif any(ch not in tokenizer._stoi for ch in text):
                details["ocr_unsupervised_reason"] = "unsupported_character"
            elif not text:
                details["ocr_unsupervised_reason"] = "empty_annotation"
            else:
                crops = sample_boxes(packet["rgb"][None], corrected.cpu(), config["ocr_input_size"]).to(device)
                ocr = frozen_parseq_loss(recognizer, crops, [text])
                details.update(ocr_supervised=True, ocr_loss=float(ocr.detach()))
                loss = loss + config["ocr_weight"] * ocr
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite local word objective")
    return loss, details


def predict_image(source, sid, model, arm, config, device, evidence_dir):
    rgb, m, labels, packets = source.packets(sid)
    rows, arrays = [], {}
    audit = {"proposals": len(packets), "missing_predicted_kernel": 0, "empty_anchor_fallback": 0}
    with torch.inference_mode():
        for index, packet in enumerate(packets):
            prop = dict(packet["proposal"])
            box = packet["box"][None].to(device)
            if model is not None and packet["prompt"].any():
                inputs = source_inputs(packet, arm, device)
                output = model(inputs, box)
                arrays[str(index)+"_output"] = output[0].cpu().numpy()
                if model.kind == "geometry":
                    # An exactly zero initialized residual must replay the original box.
                    if torch.any(output != 0):
                        corrected = bounded_boxes(box, output, inputs.shape[-2:], max_fraction=config["max_box_fraction"])
                        prop["bbox"] = remap_local_box(corrected[0].cpu().numpy(), packet["window"], rgb.size,
                            local_size=config["crop"]["local_size"], native_size=labels.shape[0])
                else:
                    p = output[0].sigmoid().cpu().numpy()
                    prompt, canvas = packet["prompt"].numpy(), packet["canvas"].numpy()
                    if model.kind == "mask":
                        support = mask_component(p[0], prompt, canvas, threshold=config["crop"]["mask_threshold"])
                    else:
                        foreground = (packet["text"].numpy() >= config["crop"]["text_threshold"]) & canvas
                        support = affinity_component(p, foreground, prompt, threshold=config["crop"]["affinity_threshold"])
                    if support is None:
                        audit["empty_anchor_fallback"] += 1
                    else:
                        arrays[str(index)+"_support"] = support
                        prop["bbox"] = remap_local_box(support_box(support), packet["window"], rgb.size,
                            local_size=config["crop"]["local_size"], native_size=labels.shape[0])
            elif model is not None:
                audit["missing_predicted_kernel"] += 1
            prop.update(instance_id=packet["seed_id"], source_bbox=packet["proposal"]["bbox"], window=packet["window"])
            rows.append(prop)
    # Validity has no effect on model inputs, windows, geometry or fallback decisions.
    for row in rows:
        row["ignored"] = not bool(np.any((labels == row["instance_id"]) & m["validity"].astype(bool)))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    key = sid.replace(":", "__")
    np.savez_compressed(evidence_dir/(key+".npz"), **arrays)
    frozen = {"sample_id": sid, "image_sha256": source.samples[sid].image_sha256,
              "proposals": rows, "audit": audit, "outputs_sha256": sha256_file(evidence_dir/(key+".npz"))}
    write_json(evidence_dir/(key+".json"), frozen)
    return frozen


def evaluate(source, ids, model, arm, recognizer, config, device, directory):
    # Freeze all inference geometries before requesting any scoring annotations.
    frozen = [predict_image(source, sid, model, arm, config, device, directory/"predictions") for sid in ids]
    rows = []
    for record in frozen:
        sid = record["sample_id"]
        rgb, m, _ = source.load(sid)
        predictions = []
        for proposal in record["proposals"]:
            x1, y1, x2, y2 = proposal["bbox"]
            crop = np.asarray(rgb)[y1:y2, x1:x2].copy()
            key = hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest()
            cache = directory.parent.parent/"ocr_cache"/(key+".json")
            identity = {"crop_sha256": key, "parseq": recognizer._provenance, "device": str(device), "batch_size": 1}
            if cache.exists():
                cached = json.loads(cache.read_text())
                if cached["identity"] != identity:
                    raise ValueError("OCR cache provenance differs")
            else:
                reading = recognizer.recognize(crop, input_role="T3_localized_crop")
                cached = {"identity": identity, "prediction": reading.to_dict()}
                write_json(cache, cached)
            predictions.append({**proposal, "text": cached["prediction"]["raw_text"], "crop_sha256": key})
        gt = source.gt(sid, m)
        care = [p for p in predictions if not p["ignored"]]
        geom = _geometry_match(care, gt, config["evaluation_iou_threshold"])
        # Existing exact-OCR helper has a qualified fixed IoU0.5 contract.
        if config["evaluation_iou_threshold"] != .5:
            raise ValueError("Qualified exact OCR helper requires explicit evaluation IoU0.5")
        exact = _correct(care, gt, normalize=False)
        rows.append({**record, "predictions": predictions, "ground_truth": gt, "proposals": len(care),
                     "gt_words": len(gt), "geometry_correct": len(geom), "ocr_correct": len(exact),
                     "matched_geometry_gt": [int(i) for i in geom], "matched_ocr_gt": [int(i) for i in exact]})
    result = {"aggregate": summarize(rows), "per_image": rows}
    write_json(directory/"metrics.json", result)
    return result


def selection_key(report):
    a = report["aggregate"]
    return a["ocr_correct"], a["ocr_hmean"], a["geometry_hmean"]


def model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def memory(device, config, stage):
    cuda = device.type == "cuda"
    return {"device": str(device), "stage": stage, "precision": "float32", "batch_size": 1,
            "input_shape": [1, 5, config["crop"]["local_size"], config["crop"]["local_size"]],
            "model_set": "one local word model plus frozen qualified PARSeq; source segmentation maps on disk",
            "differentiable_sampler_device": "cpu", "oom": False,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}


def train_arm(source, splits, arm, recognizer, config, device, root, baseline, *, smoke=False):
    directory = root/arm
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"], deterministic_algorithms=True)
    model = LocalWordModel(ARMS[arm], config["channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, best, best_state, best_epoch, start = [], selection_key(baseline), None, None, 0
    checkpoint = directory/"last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["contract_sha256"] != sha256_file(root/"run_contract.json") or saved["arm"] != arm:
            raise ValueError("Resume contract differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_state, best_epoch, start = saved["history"], tuple(saved["best"]), saved["best_state"], saved["best_epoch"], saved["epoch"]
    else:
        initial = evaluate(source, splits["calibration"], model.eval(), arm, recognizer, config, device,
                           directory/"calibration_epoch0")
        if selection_key(initial) > best:
            best, best_state, best_epoch = selection_key(initial), model_state(model), 0
    epochs = 1 if smoke else config["epochs"]
    for epoch in range(start+1, epochs+1):
        model.train()
        order = np.random.default_rng(config["seed"]+epoch).permutation(len(splits["fit"]))
        losses, audits, steps = [], [], 0
        for position in order:
            sid = splits["fit"][int(position)]
            _, m, _, packets = source.packets(sid)
            if not packets:
                audits.append({"sample_id": sid, "reason": "no_predicted_proposals"})
                continue
            rng = np.random.default_rng(config["seed"]+epoch*1_000_003+int(position))
            packet = packets[int(rng.integers(len(packets)))]
            gt = source.gt(sid, m)
            target = training_targets(packet, m["owners"], m["validity"].astype(bool), gt, config["crop"])
            optimizer.zero_grad(set_to_none=True)
            loss, detail = objective(model, arm, packet, target, recognizer, config, device)
            if detail["supervised"]:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite local model gradient")
                detail["gradient_l2"] = float(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).sqrt())
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
            audits.append({"sample_id": sid, "seed_id": packet["seed_id"], **target["audit"], **detail})
        if not steps:
            raise ValueError("No supervised optimizer step in this arm/epoch")
        report = None
        if smoke or epoch in config["calibration_epochs"]:
            model.eval()
            report = evaluate(source, splits["calibration"], model, arm, recognizer, config, device,
                              directory/("calibration_epoch"+str(epoch)))
            key = selection_key(report)
            if key > best:
                best, best_state, best_epoch = key, model_state(model), epoch
        history.append({"epoch": epoch, "optimizer_steps": steps, "mean_loss": sum(losses)/len(losses),
                        "training_audit": audits, "calibration": report["aggregate"] if report else None})
        saved = {"contract_sha256": sha256_file(root/"run_contract.json"), "model": model_state(model),
                 "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history,
                 "best": best, "best_state": best_state, "best_epoch": best_epoch, "arm": arm}
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(checkpoint)
        write_json(directory/"training.json", {"result_class": "Smoke-test result" if smoke else "approximate reproduction",
                   "status": "SUCCESS" if epoch == epochs else "RUNNING", "epochs": epoch, "history": history,
                   "selected_epoch": best_epoch, "selected_off": best_state is None})
        print(f"[{arm}] epoch {epoch}/{epochs}, updates {steps}, loss {history[-1]['mean_loss']:.5f}, selected {best_epoch}", flush=True)
    # Off is a legitimate choice. Never silently promote a worse trained model.
    if best_state is None:
        return None, {"enabled": False, "epoch": None, "checkpoint_sha256": sha256_file(checkpoint)}
    model.load_state_dict(best_state, strict=True)
    return model.eval(), {"enabled": True, "epoch": best_epoch, "checkpoint_sha256": sha256_file(checkpoint)}


def run(args):
    config = read_config(args.config, args.mode)
    if args.action == "smoke" and not 1 <= args.smoke_images <= config["training_images"]-config["calibration_images"]:
        raise ValueError("Smoke population must be positive and within the fit partition")
    root = args.output_root.resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError("Generated outputs must be outside repository")
    for input_root in (args.evidence_root.resolve(), args.word_root.resolve(), args.parseq_source.resolve()):
        if root == input_root or input_root in root.parents or root in input_root.parents:
            raise ValueError("Output must be isolated from all scientific inputs")
    root.mkdir(parents=True, exist_ok=True)
    with (root/"runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, config, root)


def run_locked(args, config, root):
    stage, recognizer = "preflight", None
    device = torch.device(args.device)
    try:
        torch.set_num_threads(config["cpu_threads"])
        cv2.setNumThreads(1)
        seed_everything(config["seed"], deterministic_algorithms=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        source = EvidenceSource(args.evidence_root, args.word_root, config)
        if device.type == "cuda":
            hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True, capture_output=True, text=True)
            processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], check=True, capture_output=True, text=True)
            write_json(root/"gpu_before.json", {"hardware": hardware.stdout, "processes": processes.stdout})
        splits = split_ids(source, config)
        contract = {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                    "result_class": "Smoke-test result" if args.action == "smoke" else "approximate reproduction",
                    "resolved_config": config, "device": str(device), "environment": runtime_environment(),
                    "splits": splits, "smoke": args.action == "smoke",
                    "smoke_images": args.smoke_images if args.action == "smoke" else None,
                    "inputs": {k: str(getattr(args, k).resolve()) for k in ("evidence_root", "word_root", "parseq_source", "parseq_checkpoint", "parseq_archive")},
                    "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for folder in
                        ("models", "segmentation", "recognition", "localize", "scripts") for p in (REPO/folder).glob("*.py")},
                    "source_postprocessing_receipt_sha256": source.post_receipt_sha256}
        if (root/"run_contract.json").exists():
            if json.loads((root/"run_contract.json").read_text()) != contract:
                raise ValueError("Existing experiment contract changed; use a new output root")
        else:
            write_json(root/"run_contract.json", contract)
        # Validate every map and RGB hash, including source denominator; no new inference yet.
        for split in ("training", "validation"):
            for sid in source.ids(split):
                source.load(sid, decode=False)
        write_json(root/"preflight.json", {"status": "PASS", "images": {k: len(v) for k, v in splits.items()},
                   "classification": "approximate reproduction", "contract_sha256": sha256_file(root/"run_contract.json")})
        if args.action == "preflight":
            return
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint, device=str(device), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_archive))
        recognizer.model.eval().requires_grad_(False)
        before_parseq = model_state(recognizer.model)
        if args.action == "smoke":
            splits = {"fit": splits["fit"][:args.smoke_images], "calibration": splits["calibration"][:1],
                      "evaluation": splits["fit"][:1]}
            write_json(root/"smoke_population.json", splits)
        stage = "calibration_reference"
        baseline = evaluate(source, splits["calibration"], None, "baseline", recognizer, config, device, root/"baseline/calibration")
        results = {"baseline": evaluate(source, splits["evaluation"], None, "baseline", recognizer, config, device, root/"baseline/evaluation")}
        choices = {}
        for arm in ARMS:
            stage = arm
            write_json(root/"progress.json", {"stage": stage, "status": "RUNNING", "result_class": contract["result_class"]})
            model, choice = train_arm(source, splits, arm, recognizer, config, device, root, baseline, smoke=args.action == "smoke")
            choices[arm] = choice
            results[arm] = evaluate(source, splits["evaluation"], model, arm, recognizer, config, device, root/arm/"evaluation")
            del model
        if any(not torch.equal(value, recognizer.model.state_dict()[key].detach().cpu()) for key, value in before_parseq.items()):
            raise RuntimeError("Frozen PARSeq state changed")
        expected = config["expected_evaluation_words"]
        if args.action != "smoke" and any(a["aggregate"]["gt_words"] != expected for a in results.values()):
            raise ValueError("Evaluation care-word denominator changed")
        base = {r["sample_id"]: set(r["matched_ocr_gt"]) for r in results["baseline"]["per_image"]}
        for part in results.values():
            gained, lost = [], []
            for row in part["per_image"]:
                old, new = base[row["sample_id"]], set(row["matched_ocr_gt"])
                gained.extend([row["sample_id"], i] for i in sorted(new-old))
                lost.extend([row["sample_id"], i] for i in sorted(old-new))
            part["paired_vs_baseline"] = {"gained": gained, "lost": lost}
        measured_memory = memory(device, config, "completed suite")
        write_json(root/("memory_invocation_"+str(os.getpid())+".json"), measured_memory)
        if (root/"results.json").exists():
            previous = json.loads((root/"results.json").read_text())
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                if measured_memory[key] is not None:
                    measured_memory[key] = max(measured_memory[key], previous["memory"][key])
        write_json(root/"results.json", {"status": "SUCCESS", "reproduction_mode": "approximate",
            "result_class": contract["result_class"], "choices": choices, "arms": results,
            "metric_protocol": config["metric_protocol"],
            "frozen_parseq_state_unchanged": True, "memory": measured_memory,
            "contract_sha256": sha256_file(root/"run_contract.json")})
        write_json(root/"progress.json", {"status": "SUCCESS", "stage": "smoke_complete" if args.action == "smoke" else "complete",
                   "result_class": contract["result_class"]})
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
    for name in ("evidence-root", "word-root", "parseq-source", "parseq-checkpoint", "parseq-archive", "output-root"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--device", required=True)
    p.add_argument("--action", required=True, choices=["preflight", "smoke", "all"])
    p.add_argument("--smoke-images", type=int, default=2)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
