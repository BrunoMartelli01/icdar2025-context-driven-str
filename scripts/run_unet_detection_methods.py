"""Resumable APPROXIMATE text pretraining/fusion/progressive-kernel experiments."""

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.unet_detection_methods import detection_instances
from localize.word_multitask import parseq_word_proposals
from models.unet_detection_methods import BRANCHES, DetectionMethodsAGUNet
from scripts.evaluate_word_multitask_crops import ratios, score_predictions
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_word_research_v2 import backend_state, load_source_state, memory
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _append_epoch, _epoch_log, _json, _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.train_totaltext_letterbox_word_f1_separator_head import _validation_reference
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.synthtext_words import SynthTextWordDataset, prepare_synthtext, TARGET_POLICIES
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.train import EpochShuffleSampler
from segmentation.training import load_checkpoint, save_checkpoint, seed_everything
from segmentation.unet_detection_methods import DetectionTrainingConfig, DetectionWordDataset, train_detection_epoch
from segmentation.word_multitask_evaluation import aggregate_word_metrics, render_word_maps
from segmentation.word_multitask_provenance import json_contract


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/unet_detection_methods_v1.json"
SCIENCE_FILES = (
    "models/unet_detection_methods.py", "segmentation/unet_detection_methods.py",
    "segmentation/synthtext_words.py", "localize/unet_detection_methods.py",
    "scripts/run_unet_detection_methods.py", "scripts/evaluate_unet_detection_methods.py",
    "scripts/run_unet_detection_methods_host.sh", "models/unet_adv.py", "models/totaltext_word_kernel.py",
    "segmentation/data.py", "segmentation/transforms.py", "segmentation/training.py",
    "segmentation/totaltext_word_kernel.py", "segmentation/word_multitask.py",
    "segmentation/totaltext_word_regions.py", "segmentation/totaltext_image_frequency_augment.py",
    "segmentation/word_multitask_evaluation.py", "segmentation/evaluate.py", "segmentation/train.py",
    "localize/word_multitask.py", "localize/crop_recovery.py", "diagnostics/fixed_group_curve_geometry.py",
    "recognition/parseq.py", "recognition/scene_text_datasets.py", "recognition/scene_text_v1_evaluation.py",
    "scripts/run_unet_word_research_v2.py", "scripts/run_word_multitask.py",
    "scripts/train_totaltext_v4_highres_ablation.py", "scripts/train_totaltext_v4_image_frequency_ablation.py",
    "scripts/train_totaltext_letterbox_word_f1_separator_head.py",
    "scripts/evaluate_word_multitask_crops.py", "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
    "scripts/evaluate_totaltext_highres_contour_padding.py", "scripts/rescore_totaltext_highres_padding_symbols.py",
)


def bind(path, content):
    path = Path(path)
    content = json_contract(content)
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise ValueError("Existing artifact/contract differs; use a NEW output directory: " + str(path))
    else:
        _json(path, content)


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE configuration required; no namespace fallback")
    c = json.loads(path.read_text())
    if (c.get("schema_version") != 1 or c.get("reproduction_mode") != mode
            or c.get("claim_label") != "approximate reproduction" or c.get("experiment_kind") != "unet_detection_methods_v1"
            or c.get("branches") != list(BRANCHES) or c.get("input_size") != [512, 512]
            or c.get("native_mask_size") != [256, 256] or c.get("validation_count") != 125
            or c.get("expected_validation_words") != 983 or c.get("precision") != "float32"
            or c.get("crop_views") != ["axis", "oriented", "bezier"]
            or c.get("kernel_fractions") != [0.5, 0.3, 0.1] or c.get("fusion_channels") != 32
            or c["postprocessing"]["padding"] != 0 or c["postprocessing"]["min_area"] != 0
            or c["postprocessing"]["max_instances"] is not None or c["rectification"]["pad"] != 0):
        raise ValueError("Unsupported detection experiment contract")
    if (any(type(c[k]) is not int or c[k] < 1 for k in ("epochs", "batch_size", "cpu_threads"))
            or type(c["num_workers"]) is not int or c["num_workers"] < 0
            or any(not np.isfinite(c[k]) or c[k] <= 0 for k in
                   ("learning_rate", "encoder_learning_rate", "head_learning_rate"))
            or any(not np.isfinite(v) or v < 0 for k, v in c["loss"].items())
            or c["synthtext"]["coordinate_origin"] not in (0, 1)
            or c["synthtext"]["checkpoint_selection"] != "last"):
        raise ValueError("Invalid training hyperparameters")
    if (c["loss"]["smooth"] <= 0 or c["loss"]["separation_weight"] < 1
            or c["loss"]["bce_weight"] + c["loss"]["dice_weight"] <= 0
            or c["loss"]["lambda_text"] <= 0 or c["loss"]["lambda_kernel"] <= 0
            or any(not 0 < c["postprocessing"][k] < 1 for k in ("text_threshold", "kernel_threshold", "center_threshold"))
            or any(type(c["synthtext"][k]) is not int or c["synthtext"][k] < 1 for k in ("epochs", "max_images"))
            or any(not np.isfinite(c["synthtext"][k]) or c["synthtext"][k] <= 0 for k in ("learning_rate", "encoder_learning_rate"))
            or not 4 <= c["rectification"]["minimum_occupied_bins"] <= c["rectification"]["envelope_bins"]
            or c["rectification"]["minimum_pixels"] < 2 or c["rectification"]["max_condition_number"] <= 1):
        raise ValueError("Invalid loss, threshold, synthetic phase or rectification settings")
    source_path = REPO / c["source_config"]
    if c["synthtext"].get("target_policy", "legacy_fail_closed") not in TARGET_POLICIES:
        raise ValueError("Unsupported SynthText target policy")
    if sha256_file(source_path) != c["source_config_sha256"]:
        raise ValueError("Qualified source dataset/preprocessing configuration changed")
    return c, json.loads(source_path.read_text())


def training_config(c, synthetic=False):
    phase = c["synthtext"] if synthetic else c
    return DetectionTrainingConfig(epochs=phase["epochs"], seed=c["seed"], adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8, learning_rate_schedule="none", checkpoint_selection="last" if synthetic else "validation_word_f1",
        deterministic_algorithms=True, learning_rate=phase["learning_rate"], weight_decay=c["weight_decay"],
        batch_size=c["batch_size"], separation_weight=c["loss"]["separation_weight"])


def optimizer_for(model, c, synthetic=False):
    phase = c["synthtext"] if synthetic else c
    groups = {"encoder": [], "shared": [], "new": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = "encoder" if name.startswith("encoder.") else "new" if name.startswith(("scale_fusion.", "kernel_scales.")) else "shared"
            groups[key].append(parameter)
    rates = {"encoder": phase["encoder_learning_rate"], "shared": phase["learning_rate"], "new": c["head_learning_rate"]}
    return torch.optim.Adam([{"params": p, "lr": rates[k], "group_name": k} for k, p in groups.items() if p],
                            weight_decay=c["weight_decay"], betas=(0.9, 0.999), eps=1e-8)


def selection_key(metrics, epoch, synthetic=False):
    if synthetic:
        return (epoch,)
    return (metrics["original"]["geometry_f1"], -metrics["native"]["gt_words_touched_by_fused_components"],
            metrics["original"]["geometry_recall_all_gt"], metrics["pixel"]["foreground_iou_macro"], -epoch)


def save_maps(directory, arrays):
    path = directory / "maps.npz"
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError("Existing exported arrays differ: " + str(path))
    else:
        np.savez_compressed(path, **arrays)
    return sha256_file(path)


def evaluate(model, loader, references, original_gt, c, device, export_dir=None):
    model.eval()
    native_totals, original_totals, rows, ious, f1s = {}, {}, [], [], []
    pixel_totals = np.zeros(3, np.int64)
    with torch.inference_mode():
        for batch in loader:
            evidence = model.forward_evidence(batch["image"].to(device))
            probabilities = evidence["logits"].sigmoid().cpu().numpy()
            for i, sid in enumerate(batch["sample_id"]):
                owners, expected, canvas = references[sid]
                labels, audit = detection_instances(probabilities[i], canvas, c)
                target, valid = batch["mask"][i, 0].numpy().astype(bool), batch["validity"][i, 0].numpy().astype(bool)
                owners = owners.copy()
                owners[~target | ~valid] = 0  # Same qualified native target convention as V2.
                ignored = [int(x) for x in np.unique(labels[labels > 0]) if not np.any((labels == x) & valid)]
                care = labels.copy(); care[np.isin(care, ignored)] = 0
                native = instance_geometry(care, owners, expected)
                image_wh, gt, annotation_sha = original_gt[sid]
                proposals = parseq_word_proposals(labels, image_wh, padding=0, min_area=0, max_instances=None)
                original = score_predictions([p for p in proposals if p["instance_id"] not in ignored], gt, text_available=False)
                for totals, part in ((native_totals, native), (original_totals, original)):
                    for key, value in part.items():
                        totals[key] = totals.get(key, 0) + value
                foreground = labels > 0
                tp, fp, fn = [int(np.count_nonzero(mask & valid)) for mask in
                             (foreground & target, foreground & ~target, ~foreground & target)]
                pixel_totals += (tp, fp, fn)
                ious.append(tp / (tp + fp + fn) if tp + fp + fn else 1.)
                f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.)
                row = {"sample_id": sid, "native": native, "original": original,
                       "ignored_instance_ids": ignored, "annotation_sha256": annotation_sha, "audit": audit}
                if export_dir is not None:
                    directory = Path(export_dir) / sid.replace(":", "__")
                    arrays = {"probabilities": probabilities[i], "labels": labels, "owners": owners,
                        "text_target": target, "validity": valid, "canvas": canvas,
                        "kernel_targets": batch["kernels"][i].numpy(), "expected_owner_ids": np.asarray(expected, np.int32)}
                    if evidence["scale_weights"] is not None:
                        arrays["scale_weights"] = evidence["scale_weights"][i].cpu().numpy()
                    row["maps_sha256"] = save_maps(directory, arrays)
                    image_path = Path(batch["image_path"][i])
                    bind(directory / "proposals.json", {"sample_id": sid, "image_path": str(image_path),
                        "image_sha256": sha256_file(image_path), "maps_sha256": row["maps_sha256"],
                        "proposals": proposals, "result_class": "approximate reproduction", "reproduction_mode": "approximate"})
                    row["proposals_sha256"] = sha256_file(directory / "proposals.json")
                    label_path = directory / "instances.png"
                    if labels.max() > 65535:
                        raise ValueError("Instance labels exceed lossless uint16 export")
                    if label_path.exists():
                        with Image.open(label_path) as saved:
                            if not np.array_equal(np.asarray(saved), labels):
                                raise ValueError("Existing instance PNG changed")
                    else:
                        Image.fromarray(labels.astype(np.uint16)).save(label_path)
                    row["labels_sha256"] = sha256_file(label_path)
                    with Image.open(image_path) as opened:
                        render_word_maps(opened.convert("RGB"), probabilities[i, :2], proposals, directory / "overview.png")
                    # Label additional scales accurately instead of calling a kernel "Center".
                    from PIL import ImageDraw
                    extra = Image.new("RGB", (768, 280), "white")
                    draw = ImageDraw.Draw(extra)
                    for j, values in enumerate(batch["kernels"][i].numpy()):
                        panel = Image.fromarray(np.rint(values * 255).astype(np.uint8)).convert("RGB")
                        extra.paste(panel, (256 * j, 24)); draw.text((256 * j + 4, 4), "GT kernel " + str(c["kernel_fractions"][j]), fill="black")
                    extra.save(directory / "kernel_targets.png")
                    if probabilities.shape[1] == 4:
                        predicted = Image.new("RGB", (768, 280), "white")
                        captions = ImageDraw.Draw(predicted)
                        for j, channel in enumerate((1, 2, 3)):
                            panel = Image.fromarray(np.rint(probabilities[i, channel] * 255).astype(np.uint8)).convert("RGB")
                            predicted.paste(panel, (256 * j, 24))
                            captions.text((256 * j + 4, 4), "Predicted kernel " + str(c["kernel_fractions"][j]), fill="black")
                        predicted.save(directory / "kernel_predictions.png")
                rows.append(row)
    if not rows or len(rows) != len({r["sample_id"] for r in rows}):
        raise ValueError("Empty/duplicated validation population")
    tp, fp, fn = (int(x) for x in pixel_totals)
    return {"native": aggregate_word_metrics(native_totals), "original": ratios(original_totals),
            "pixel": {"foreground_iou_macro": float(np.mean(ious)), "foreground_f1_macro": float(np.mean(f1s)),
                      "foreground_iou_micro": tp / (tp + fp + fn) if tp + fp + fn else 1.,
                      "foreground_f1_micro": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.},
            "per_image": rows}


def train_branch(directory, model, optimizer, train_loader, validate, setters, c, contract, device,
                 synthetic=False, stop_after_epoch=None):
    """Reuse checkpoint/RNG, epoch logs and sampling; preserve epoch0 and resume."""
    tc = training_config(c, synthetic)
    last, best, log = directory / "last.pt", directory / "best.pt", directory / "epochs.jsonl"
    history = _epoch_log(log)
    epoch, step, best_epoch, best_metrics = 0, 0, None, None
    def save(path, e, trained, metrics):
        return save_checkpoint(path, model=model, optimizer=optimizer, epoch=e, global_step=step, config=tc,
            resolved_config=json.dumps(contract, sort_keys=True), architecture=model.architecture_identity,
            result_class="approximate reproduction", run_contract=contract, train_metrics=trained, validation_metrics=metrics)
    if best.exists():
        saved = load_checkpoint(best, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc)
        best_epoch, best_metrics = saved["epoch"], saved["validation_metrics"]
    if last.exists():
        saved = load_checkpoint(last, model=model, optimizer=optimizer, expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc, restore_rng=True)
        epoch, step = saved["epoch"], saved["global_step"]
        if epoch and len(history) == epoch - 1:
            row = {"epoch": epoch, "global_step": step, "train": saved["train_metrics"],
                   "validation": saved["validation_metrics"], "last_checkpoint_sha256": sha256_file(last),
                   "recovered_from_checkpoint": True}
            _append_epoch(log, row); history.append(row)
        if len(history) != epoch or history and history[-1]["last_checkpoint_sha256"] != sha256_file(last):
            raise ValueError("Latest checkpoint/epoch log mismatch")
        if best_metrics is None or selection_key(saved["validation_metrics"], epoch, synthetic) > selection_key(best_metrics, best_epoch, synthetic):
            best_epoch, best_metrics = epoch, saved["validation_metrics"]
            save(best, epoch, saved["train_metrics"], best_metrics)
        del saved
    elif history or best.exists():
        raise ValueError("Training files lack a resumable last checkpoint")
    else:
        initial = {"synthetic_last_epoch": 0} if synthetic else validate()
        best_epoch, best_metrics = 0, initial
        save(last, 0, {}, initial); save(best, 0, {}, initial)
        bind(directory / "epoch0.json", initial)
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Noncontiguous training history")
    budget = tc.epochs if stop_after_epoch is None else stop_after_epoch
    if not 1 <= budget <= tc.epochs:
        raise ValueError("Stop epoch outside declared phase budget")
    for e in range(epoch + 1, budget + 1):
        for setter in setters:
            setter.set_epoch(e - 1)
        trained = train_detection_epoch(model, train_loader, optimizer, device, c)
        metrics = {"synthetic_last_epoch": e} if synthetic else validate()
        if not synthetic and metrics["original"]["gt_words"] != c["expected_validation_words"]:
            raise ValueError("Validation denominator changed")
        step += trained["steps"]
        digest = save(last, e, trained, metrics)
        if selection_key(metrics, e, synthetic) > selection_key(best_metrics, best_epoch, synthetic):
            best_epoch, best_metrics = e, metrics; save(best, e, trained, metrics)
        row = {"epoch": e, "global_step": step, "train": trained, "validation": metrics, "last_checkpoint_sha256": digest}
        _append_epoch(log, row); history.append(row)
        score = "last synthetic checkpoint" if synthetic else "wordF1=" + str(round(metrics["original"]["geometry_f1"], 5))
        print(f"[{model.branch}] {e}/{tc.epochs} loss={trained['loss']:.5f} {score} selected={best_epoch}", flush=True)
    complete = len(history) == tc.epochs
    return {"status": "SUCCESS" if complete else "IN_PROGRESS", "training_complete": complete,
            "completed_epochs": len(history), "run_contract": contract, "result_class": "approximate reproduction",
            "checkpoint": {"path": "best.pt", "epoch": best_epoch, "sha256": sha256_file(best)},
            "validation": best_metrics, "memory": memory(device, c), "environment": experiment_environment(device)}


def source_state(args, c, branch):
    state = load_source_state(args.v1_root.resolve() / "control", c)
    source = {"path": str(args.v1_root.resolve() / "control/best.pt"), "sha256": c["source_checkpoint_sha256"]}
    if branch.startswith("pretrained_"):
        pretrain_root = getattr(args, "pretrain_root", None) or args.output_root
        directory = pretrain_root / "synthtext_pretrain"
        receipt = json.loads((directory / "run_manifest.json").read_text())
        path = directory / "best.pt"
        if (receipt.get("status") != "SUCCESS" or not receipt.get("training_complete")
                or receipt["completed_epochs"] != c["synthtext"]["epochs"]
                or receipt["checkpoint"]["sha256"] != sha256_file(path)
                or receipt["checkpoint"]["epoch"] != c["synthtext"]["epochs"]):
            raise ValueError("Pretrained branches require completed, SHA-qualified SynthText pretraining")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("schema_version") != 1 or payload.get("reproduction_mode") != "approximate"
                or payload.get("claim_label") != "approximate reproduction" or payload.get("epoch") != c["synthtext"]["epochs"]
                or payload["run_contract"] != receipt["run_contract"] or payload["architecture"] != "experimental-unet-detection-methods-v1:control"
                or payload["training_config"] != asdict(training_config(c, True))
                or receipt["run_contract"]["resolved_config"] != c
                or receipt["run_contract"]["phase"] != "SynthText_domain_pretrain"
                or receipt["run_contract"]["synthetic_manifest_sha256"] != sha256_file(pretrain_root / "synthtext_manifest.json")
                or set(receipt["run_contract"]["source_sha256"]) != set(SCIENCE_FILES)
                or any(sha256_file(REPO / f) != h for f, h in receipt["run_contract"]["source_sha256"].items())):
            raise ValueError("Synthetic checkpoint provenance/config/code differs")
        state = payload["model_state_dict"]
        source = {"path": str(path.resolve()), "sha256": sha256_file(path), "receipt_sha256": sha256_file(directory / "run_manifest.json")}
    return state, source


def run(args):
    c, original = read_config(args.config, args.mode)
    if args.batch_size is not None or args.num_workers is not None:
        if args.action != "smoke":
            raise ValueError("Batch/worker overrides are restricted to separately labeled smokes")
        if args.batch_size is not None: c["batch_size"] = args.batch_size
        if args.num_workers is not None: c["num_workers"] = args.num_workers
    torch.set_num_threads(c["cpu_threads"])
    cv2.setNumThreads(c["opencv_threads"])
    seed_everything(c["seed"], deterministic_algorithms=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for an explicitly separate CPU smoke")
    root, output = args.word_root.resolve(), args.output_root.resolve()
    manifest_path = root / c["train_manifest"]
    if sha256_file(manifest_path) != c["word_manifest_sha256"]:
        raise ValueError("Qualified TotalText TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    train_ids, val_ids, split_sha = _split_indices(manifest, c["seed"], c["validation_count"])
    if manifest.dataset != "total_text" or manifest.split_role != "train" or split_sha != c["validation_ids_sha256"]:
        raise ValueError("Only the frozen TotalText TRAIN1130/125 split is allowed")
    if args.limit_images is not None and args.action != "smoke":
        raise ValueError("Image limits are smoke-only; partial training is not a full result")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_hashes = {name: sha256_file(REPO / name) for name in SCIENCE_FILES}
        suite = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                 "resolved_config": c, "word_root": str(root), "v1_root": str(args.v1_root.resolve()),
                 "device": str(device), "backend": backend_state(), "software": experiment_environment(device),
                 "source_sha256": source_hashes, "validation_ids_sha256": split_sha,
                 "execution_choices": [{"classification": "HARDWARE_DEVIATION", "component": "execution_device_and_batch",
                     "rationale": f"Explicit {device} FP32/batch{c['batch_size']}; workers{c['num_workers']}; sequential models",
                     "expected_effect": "CPU execution qualifies engineering only; actual GPU memory/outputs require host qualification"}]}
        bind(output / "run_contract.json", suite)
        if args.action == "prepare-synthtext":
            if args.synthtext_root is None:
                raise ValueError("--synthtext-root is required; no synthetic-data fallback/download")
            destination = output / "synthtext_manifest.json"
            request = {"root": str(args.synthtext_root.resolve()), "settings": c["synthtext"],
                       "gt_mat_sha256": sha256_file(args.synthtext_root / "gt.mat")}
            if destination.exists():
                previous = json.loads(destination.read_text())
                if previous["request"] != request or previous["suite_contract"] != suite:
                    raise ValueError("Existing SynthText manifest contract differs")
                return {"status": "PASS", "images": previous["images"], "cached": True,
                        **({"target_summary": previous["target_summary"]} if "target_summary" in previous else {})}
            value = prepare_synthtext(args.synthtext_root, max_images=c["synthtext"]["max_images"], seed=c["seed"],
                coordinate_origin=c["synthtext"]["coordinate_origin"], excluded_image_hashes={s.image_sha256 for s in manifest.samples},
                target_policy=c["synthtext"].get("target_policy", "legacy_fail_closed"))
            value.update(request=request, suite_contract=suite)
            bind(destination, value)
            return {"status": "PASS", "images": value["images"],
                    **({"target_summary": value["target_summary"]} if "target_summary" in value else {})}
        transforms = [_transform(original, "letterbox", train=train) for train in (True, False)]
        datasets = [DetectionWordDataset(SegmentationDataset(manifest, root, transform=t), c["kernel_fractions"]) for t in transforms]
        if args.action == "preflight":
            load_source_state(args.v1_root.resolve() / "control", c)
            rows = []
            for index in val_ids:
                batch = datasets[1][index]
                sample = manifest.samples[index]
                with Image.open(root / sample.word_owner_annotation_path) as owner:
                    expected = len(np.unique(np.asarray(owner)[np.asarray(owner) > 0]))
                native = len(np.unique(batch["owners_native"].numpy()[batch["owners_native"].numpy() > 0]))
                if native != expected:
                    raise ValueError("Held-out native targets lost a word: " + sample.sample_id)
                rows.append({"sample_id": sample.sample_id, "words": expected,
                             "kernel_pixels": [int(k.sum()) for k in batch["kernels"]]})
            if sum(r["words"] for r in rows) != c["expected_validation_words"]:
                raise ValueError("Held-out care-word population changed")
            value = {"status": "PASS", "result_class": "smoke-test result", "suite_contract": suite,
                     "images": len(rows), "words": sum(r["words"] for r in rows), "per_image": rows}
            bind(output / "target_qualification.json", value)
            return value
        synthetic = args.action == "pretrain"
        branch = "control" if synthetic else args.branch
        if branch is None:
            raise ValueError("--branch required for smoke/train/evaluate")
        state, initialization = source_state(args, c, branch)
        references = None
        originals = {}
        if synthetic:
            synth = json.loads((output / "synthtext_manifest.json").read_text())
            if synth["suite_contract"] != suite or synth["images"] != c["synthtext"]["max_images"]:
                raise ValueError("Synthetic manifest incompatible with this suite")
            train_data = SynthTextWordDataset(synth, transforms[0], c["kernel_fractions"],
                target_policy=c["synthtext"].get("target_policy", "legacy_fail_closed"))
            sampler = EpochShuffleSampler(len(train_data), seed=c["seed"])
        else:
            references = _validation_reference(manifest, root, val_ids, transforms[1])
            for index in val_ids:
                sample = manifest.samples[index]
                with Image.open(root / sample.image_path) as image:
                    gt, annotation_sha = _train_gt(sample, root, image.size)
                    originals[sample.sample_id] = (image.size, gt, annotation_sha)
            if sum(len(v[1]) for v in originals.values()) != c["expected_validation_words"]:
                raise ValueError("Original annotation denominator changed")
            if args.action == "smoke":
                limit = args.limit_images or c["batch_size"]
                if limit < 1: raise ValueError("Positive smoke image limit required")
                train_ids, val_ids = train_ids[:limit], val_ids[:limit]
            train_data = Subset(datasets[0], train_ids)
            sampler = EpochShuffleSampler(len(train_ids), seed=c["seed"])
        options = {"num_workers": c["num_workers"], "timeout": 60 if c["num_workers"] else 0,
                   "pin_memory": device.type == "cuda"}
        train_loader = DataLoader(train_data, batch_size=c["batch_size"], sampler=sampler, **options)
        val_loader = None if synthetic else DataLoader(Subset(datasets[1], val_ids), batch_size=c["batch_size"], shuffle=False, **options)
        directory = output / ("synthtext_pretrain" if synthetic else branch)
        directory.mkdir(exist_ok=True)
        if not (directory / "run_contract.json").exists() and any(directory.iterdir()):
            raise ValueError("Nonempty branch lacks its run contract; use a NEW output directory")
        contract = json_contract({**suite, "branch": branch, "phase": "SynthText_domain_pretrain" if synthetic else "TotalText",
                    "initialization": initialization, "training_config": asdict(training_config(c, synthetic)),
                    "synthetic_manifest_sha256": sha256_file(output / "synthtext_manifest.json") if synthetic else None,
                    "execution_kind": "smoke" if args.action == "smoke" else "experiment"})
        bind(directory / "run_contract.json", contract)
        model = optimizer = None
        try:
            if device.type == "cuda": _prepare_cuda_memory_tracking(device)
            model = DetectionMethodsAGUNet(branch, skip_floor=c["skip_floor"], fusion_channels=c["fusion_channels"])
            model.initialize_source(state); del state
            model.to(device)
            params = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            validate = lambda: evaluate(model, val_loader, references, originals, c, device)
            if args.action == "smoke":
                optimizer = optimizer_for(model, c)
                trained = train_detection_epoch(model, [next(iter(train_loader))], optimizer, device, c) if branch != "reference" else {"steps": 0}
                value = {"status": "PASS", "result_class": "smoke-test result", "run_contract": contract,
                         "training": trained,
                         "validation": evaluate(model, val_loader, references, originals, c, device, directory / "validation_exports"),
                         "parameters": params, "memory": memory(device, c)}
                bind(directory / "smoke.json", value)
            elif branch == "reference":
                value = {"status": "SUCCESS", "training_complete": True, "completed_epochs": 0, "result_class": "approximate reproduction",
                         "run_contract": contract, "checkpoint": {**initialization, "epoch": c["source_epoch"]}, "parameters": params}
                bind(directory / "run_manifest.json", value)
                destination = directory / "validation.json"
                if destination.exists():
                    verify_exports(directory, json.loads(destination.read_text()), contract)
                else:
                    v = evaluate(model, val_loader, references, originals, c, device, directory / "validation_exports")
                    v.update(status="SUCCESS", run_contract=contract, checkpoint_sha256=initialization["sha256"], memory=memory(device, c))
                    bind(destination, v)
            elif args.action in ("train", "pretrain"):
                receipt = directory / "run_manifest.json"
                previous = json.loads(receipt.read_text()) if receipt.exists() else None
                if previous and previous["status"] == "SUCCESS":
                    verify_completed(directory, previous, contract, training_config(c, synthetic).epochs)
                    value = previous
                else:
                    optimizer = optimizer_for(model, c, synthetic)
                    value = train_branch(directory, model, optimizer, train_loader, validate,
                        (sampler, transforms[0]), c, contract, device, synthetic, args.stop_after_epoch)
                    value["parameters"] = params
                    if previous:
                        if previous.get("run_contract") != contract:
                            raise ValueError("Previous training receipt contract differs")
                        for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
                            old, current = previous.get("memory", {}).get(key), value["memory"].get(key)
                            if old is not None and current is not None:
                                value["memory"][key] = max(old, current)
                    # IN_PROGRESS -> completed may replace only this task's receipt.
                    _json(receipt, value)
            else:
                receipt = json.loads((directory / "run_manifest.json").read_text())
                verify_completed(directory, receipt, contract, c["epochs"])
                destination = directory / "validation.json"
                if destination.exists():
                    value = json.loads(destination.read_text()); verify_exports(directory, value, contract)
                else:
                    load_checkpoint(directory / "best.pt", model=model, expected_architecture=model.architecture_identity,
                        expected_run_contract=contract, expected_training_config=training_config(c))
                    value = evaluate(model, val_loader, references, originals, c, device, directory / "validation_exports")
                    value.update(status="SUCCESS", run_contract=contract, checkpoint_sha256=receipt["checkpoint"]["sha256"], memory=memory(device, c))
                    bind(destination, value)
            return value
        except Exception as error:
            _json(directory / "failure.json", {"status": "FAILED", "error_type": type(error).__name__, "message": str(error),
                "action": args.action, "run_contract": contract, "memory": memory(device, c, oom=isinstance(error, torch.cuda.OutOfMemoryError))})
            raise
        finally:
            del optimizer, model; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()


def verify_completed(directory, receipt, contract, epochs):
    if (receipt.get("status") != "SUCCESS" or not receipt.get("training_complete") or receipt.get("completed_epochs") != epochs
            or receipt.get("run_contract") != contract or receipt["checkpoint"]["sha256"] != sha256_file(directory / "best.pt")):
        raise ValueError("Completed checkpoint/training provenance differs; partial runs are not results")


def verify_exports(directory, report, contract):
    if (report.get("run_contract") != contract or report.get("status") != "SUCCESS" or len(report["per_image"]) != 125
            or len({r["sample_id"] for r in report["per_image"]}) != 125
            or report["original"]["gt_words"] != contract["resolved_config"]["expected_validation_words"]):
        raise ValueError("Incomplete or incompatible validation export")
    for row in report["per_image"]:
        folder = directory / "validation_exports" / row["sample_id"].replace(":", "__")
        for name, key in (("maps.npz", "maps_sha256"), ("proposals.json", "proposals_sha256"), ("instances.png", "labels_sha256")):
            if sha256_file(folder / name) != row[key]:
                raise ValueError("Exported artifact changed: " + str(folder / name))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["approximate"], required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("word-root", "v1-root", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--synthtext-root", type=Path)
    p.add_argument("--pretrain-root", type=Path, help="Qualified synthetic run root, for separate pretrained smokes")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--action", choices=["preflight", "prepare-synthtext", "pretrain", "smoke", "train", "evaluate"], required=True)
    p.add_argument("--branch", choices=BRANCHES)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--limit-images", type=int)
    p.add_argument("--stop-after-epoch", type=int)
    return p


def main():
    args = parser().parse_args()
    value = run(args)
    print(json.dumps({k: value[k] for k in ("status", "result_class", "completed_epochs", "images", "words", "target_summary") if k in value}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
