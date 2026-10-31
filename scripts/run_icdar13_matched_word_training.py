"""Run frozen, matched ICDAR13 loss controls and selection-only PARSeq policies."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_yolo_f1_quality_study import seal
from scripts.train_icdar13_word_refit import recognizer, read_rows, save_checkpoint
from segmentation.icdar13_word_refit import WordRefitUNet, WordDataset
from segmentation.icdar13_matched_training import matched_word_loss, emit_words, choose_selection
from segmentation.separated_word_gt import audit_word_owners
from recognition.icdar_budget_router import lexical_counts
from recognition.f1_quality_study import aggregate

REPO = Path(__file__).resolve().parents[1]


def checked_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or path.parent != REPO / "configs/approximate":
        raise ValueError("Explicit approximate configuration namespace required")
    value = read(path)
    if value["mode"] != mode:
        raise ValueError("Configuration mode mismatch")
    return value


def prepare(args):
    c = checked_config(args.config, args.mode)
    p = checked_config(args.paths, args.mode)
    if c["branches"] != ["refit_bce", "totaltext_bce_dice"] or c["claim_label"] != "approximate reproduction":
        raise ValueError("Explicit two-arm approximate study required")
    for name in ("parent_plan", "tt_checkpoint", "tt_contract"):
        if sha(p[name]) != p[name + "_sha256"]:
            raise ValueError("Changed pinned input: " + name)
    parent, targets, population = (read(p[name]) for name in ("parent_plan", "targets_manifest", "population_plan"))
    for name in ("parseq_archive", "parseq_checkpoint"):
        if p[name] != parent["paths"][name] or sha(p[name]) != parent["config"]["pinned_inputs"][p[name]]:
            raise ValueError("Qualified PARSeq input changed: " + name)
    if sha(p["targets_manifest"]) != parent["targets_manifest_sha256"] or targets["partitions"] != parent["partitions"]:
        raise ValueError("Qualified target/partition contract changed")
    parts = copy.deepcopy(parent["partitions"])
    for name, expected in c["expected_populations"].items():
        if len(parts[name]) != expected:
            raise ValueError("Unexpected population: " + name)
    if args.smoke:
        parts = {name: values[:(c["training"]["physical_batch_size"] if name == "fit" else 4)] for name, values in parts.items()}
    train = targets["records"]
    test = population["records"]["icdar13"]["test"]
    for r in train:
        for name in ("image", "annotation", "mask", "validity", "separator", "owners"):
            if sha(r[name + "_path"]) != r[name + "_sha256"]:
                raise ValueError("Changed target input: " + r["sample_id"] + ":" + name)
    populations = [{r["rgb_sha256"] for r in train + test if r["sample_id"] in ids} for ids in parts.values()]
    if any(a & b for a, b in itertools.combinations(populations, 2)):
        raise ValueError("RGB overlap between experimental populations")
    root = Path(p["output_root"]) / ("smoke" if args.smoke else "full")
    if (root / "resolved_configuration.json").exists():
        raise ValueError("Run already prepared; use its explicit --run-root")
    root.mkdir(parents=True)
    code = {str(f.relative_to(REPO)): sha(f) for folder in ("models", "segmentation", "scripts", "recognition", "localize", "caption", "scoring", "pipeline", "research") for f in (REPO / folder).rglob("*.py")}
    plan = dict(mode="approximate", claim_label="approximate reproduction", config=c, paths=p,
        partitions=parts, train_records=train, test_records=test, code_sha256=code,
        config_sha256=sha(args.config), paths_sha256=sha(args.paths),
        targets_manifest_sha256=sha(p["targets_manifest"]), population_plan_sha256=sha(p["population_plan"]),
        smoke=args.smoke, result_class="smoke-test result" if args.smoke else "approximate reproduction",
        metric="Per-image lexical multiset micro F1; Unicode casefold+isalnum; native256 ignore; no IoU; explicit no-alphanumeric rejection; all care GT retained",
        hardware=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), precision="float32", segmentation_batch=c["training"]["physical_batch_size"],
            inference_batch=1, parseq_batch=c["parseq_batch_size"], image_hw=c["image_hw"], model_set="one frozen-encoder Text+Kernel U-Net and genuine qualified PARSeq; no DeepSolo"),
        historical_test_exposure=True, test_used_for_selection=False)
    seal(root / "resolved_configuration.json", plan)
    for file in code:
        dest = root / "code_snapshot" / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / file, dest)
        if sha(dest) != code[file]:
            raise ValueError("Source changed during freeze")
    shutil.copytree(REPO / "configs", root / "code_snapshot/configs")
    # Audit the actual configured loader before any optimizer step.
    torch.set_num_threads(c["cpu_threads"])
    audit = []
    for i, r in enumerate(train):
        batch = WordDataset([r], c, augment=False)[0]
        value = audit_word_owners(batch["owners"].numpy(), r["care_words"])
        if value["missing_word_ids"] or value["fragmented_word_ids"] or value["interword_contact_pairs"]:
            raise ValueError("Unsafe word topology at new training resolution: " + r["sample_id"])
        audit.append(dict(sample_id=r["sample_id"], care_words=r["care_words"], topology=value))
    seal(root / "target_audit.json", dict(images=len(audit), care_words=sum(r["care_words"] for r in train), rows=audit))
    return root


def model(plan, checkpoint):
    c = plan["config"]
    net = WordRefitUNet(partial_encoder=False, skip_floor=c["skip_floor"], initial_gap_bias=c["initial_gap_bias"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if Path(checkpoint) == Path(plan["paths"]["tt_checkpoint"]) and state["run_contract"] != read(plan["paths"]["tt_contract"]):
        raise ValueError("Source state contract changed")
    net.load_state_dict(state["model_state_dict"], strict=True)
    return net.to(c["device"])


def records(plan, split):
    lookup = {r["sample_id"]: r for r in plan["train_records"] + plan["test_records"]}
    return [lookup[s] for s in plan["partitions"][split]]


def infer_maps(net, plan, split, directory):
    c = plan["config"]
    rs = records(plan, split)
    dataset = WordDataset(rs, c, augment=False) if split != "test" else None
    # TEST has no fabricated training masks/owners; preprocessing uses RGB only.
    from segmentation.icdar2015_improvements import make_transform
    transform = make_transform(c, c["image_hw"], augment=False)
    net.eval()
    torch.cuda.reset_peak_memory_stats()
    directory.mkdir(parents=True, exist_ok=False)
    for i, r in enumerate(rs):
        if sha(r["image_path"]) != r["image_sha256"]:
            raise ValueError("Inference RGB changed")
        if dataset is not None:
            x = dataset[i]["image"]
        else:
            with Image.open(r["image_path"]) as im:
                rgb = im.convert("RGB")
                x, _ = transform(rgb, Image.new("L", rgb.size, 0), i)
        with torch.inference_mode():
            probability = net(x[None].to(c["device"]))[0].sigmoid().cpu().numpy()
        np.savez_compressed(directory / (key(r["sample_id"]) + ".npz"), probabilities=probability)
    seal(directory / "memory.json", dict(model_set=plan["hardware"]["model_set"], device=c["device"],
        precision="float32", image_shape=[1, 3, *c["image_hw"]], parseq_batch=c["parseq_batch_size"],
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))


def score_maps(plan, split, directory, settings, ocr, output):
    torch.cuda.reset_peak_memory_stats()
    result = []
    for r in records(plan, split):
        mp = directory / (key(r["sample_id"]) + ".npz")
        with np.load(mp) as z:
            probability = z["probabilities"]
        _, raw, expansion = read_rows(probability, r, settings, ocr, plan["config"]["parseq_batch_size"])
        gt, valid = annotation(r)
        rows = emit_words(raw, settings["minimum_transcript_probability"])
        counts = lexical_counts(predictions(rows, r, valid), gt)
        seal(output / "samples" / (key(r["sample_id"]) + ".json"), dict(rows=rows, raw_rows=raw,
            counts=counts, expansion=expansion, image_sha256=r["image_sha256"], maps_sha256=sha(mp),
            ground_truth_used_for_prediction=False))
        result.append(counts)
    value = dict(split=split, settings=settings, counts=aggregate(result), deepsolo_images=0,
        total_images=len(result), deepsolo_percentage=0., actual_deepsolo_calls=0,
        gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved())
    seal(output / "metrics.json", value)
    return value


def train(plan, root, branch, ocr):
    c, t = plan["config"], plan["config"]["training"]
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"]); torch.cuda.manual_seed_all(c["seed"])
    net = model(plan, plan["paths"]["tt_checkpoint"])
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=t["learning_rate"],
        weight_decay=t["weight_decay"], betas=tuple(t["adam_betas"]), eps=t["adam_epsilon"])
    fit = WordDataset(records(plan, "fit"), c, augment=True)
    out = root / branch
    history = []
    # Both new arms include exactly the same source baseline in epoch selection.
    best = read(root / "source/selection/metrics.json")["counts"]
    best_epoch = 0
    save_checkpoint(out / "best.pt", dict(model_state_dict=net.state_dict(), epoch=0, validation=best,
        branch=branch, plan_sha256=sha(root / "resolved_configuration.json")))
    epochs = 1 if plan["smoke"] else t["epochs"]
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        fit.transform.set_epoch(epoch - 1)
        loader = DataLoader(fit, batch_size=t["physical_batch_size"], num_workers=t["num_workers"],
            shuffle=True, generator=torch.Generator().manual_seed(c["seed"] + epoch), pin_memory=True)
        net.train(); seen = []; total = 0.; torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            local = {k: v.to(c["device"]) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = matched_word_loss(net(local["image"]), local, c, branch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch["sample_id"])
            seen.extend(batch["sample_id"])
        if sorted(seen) != sorted(plan["partitions"]["fit"]):
            raise ValueError("Incomplete fitting population")
        row = dict(epoch=epoch, loss=total / len(seen), samples=len(seen),
            sample_order_sha256=hashlib.sha256(("\n".join(seen) + "\n").encode()).hexdigest(),
            gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved())
        if epoch % c["validation_interval"] == 0 or epoch == epochs:
            val = out / "validation" / ("epoch_" + str(epoch).zfill(3))
            infer_maps(net, plan, "selection", val / "maps")
            result = score_maps(plan, "selection", val / "maps", c["inference"], ocr, val)
            row["selection"] = result["counts"]
            if result["counts"]["f1"] > best["f1"]:
                best, best_epoch = result["counts"], epoch
                save_checkpoint(out / "best.pt", dict(model_state_dict=net.state_dict(), epoch=epoch,
                    validation=best, branch=branch, plan_sha256=sha(root / "resolved_configuration.json")))
        history.append(row)
        save_checkpoint(out / "last.pt", dict(model_state_dict=net.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch, history=history, branch=branch, plan_sha256=sha(root / "resolved_configuration.json")))
        (out / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(dict(branch=branch, epoch=epoch, best_epoch=best_epoch, selection_f1=best["f1"], seconds=time.monotonic()-start)), flush=True)
    seal(out / "training.json", dict(status="SUCCESS", epochs=epochs, fit_images=len(fit),
        best_epoch=best_epoch, best_validation=best, checkpoint_sha256=sha(out / "best.pt"),
        peak_allocated=max(r["gpu_allocated"] for r in history), peak_reserved=max(r["gpu_reserved"] for r in history),
        result_class=plan["result_class"]))
    del net, optimizer
    torch.cuda.empty_cache()


def finish(plan, root, ocr):
    c = plan["config"]
    candidates = [("source", plan["paths"]["tt_checkpoint"])] + [(name, root / name / "best.pt") for name in c["branches"]]
    scores = []
    for name, checkpoint in candidates:
        checkpoint_digest = sha(checkpoint)
        net = model(plan, checkpoint)
        directory = root / "policy_selection" / name
        infer_maps(net, plan, "selection", directory / "maps")
        del net; torch.cuda.empty_cache()
        # Decode once per threshold/padding pair, then rescore the fixed confidence cuts.
        for threshold, padding in itertools.product(c["policy_grid"]["text_threshold"], c["policy_grid"]["padding"]):
            settings = dict(c["inference"], text_threshold=threshold, padding=padding, minimum_transcript_probability=0.)
            folder = directory / ("threshold_" + str(threshold) + "_padding_" + str(padding))
            score_maps(plan, "selection", directory / "maps", settings, ocr, folder)
            for cutoff in c["policy_grid"]["minimum_transcript_probability"]:
                counted = []
                for r in records(plan, "selection"):
                    raw = read(folder / "samples" / (key(r["sample_id"]) + ".json"))["raw_rows"]
                    gt, valid = annotation(r)
                    counted.append(lexical_counts(predictions(emit_words(raw, cutoff), r, valid), gt))
                scores.append(dict(candidate=name, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_digest,
                    split="selection", settings=dict(settings, minimum_transcript_probability=cutoff), counts=aggregate(counted)))
        fixed = next(r for r in scores if r["candidate"] == name and r["settings"] == c["inference"])
        expected = read(root / "source/selection/metrics.json")["counts"] if name == "source" else read(root / name / "training.json")["best_validation"]
        if fixed["counts"] != expected:
            raise ValueError("Strictly loaded selected-checkpoint replay changed: " + name)
        print(json.dumps(dict(stage="selection_grid", candidate=name, best_f1=max(r["counts"]["f1"] for r in scores if r["candidate"]==name))), flush=True)
    winner = choose_selection(scores)
    seal(root / "selection_frozen.json", dict(winner=winner, all_candidates=scores, test_used_for_selection=False))
    # The exact winner is frozen before any calibration/test scoring.
    final = []
    for name, checkpoint, settings in [("source_matched_fixed", plan["paths"]["tt_checkpoint"], c["inference"]),
            ("selected", winner["checkpoint"], winner["settings"])]:
        net = model(plan, checkpoint)
        for split in ("calibration", "test"):
            directory = root / "evaluation" / name / split
            infer_maps(net, plan, split, directory / "maps")
            result = score_maps(plan, split, directory / "maps", settings, ocr, directory)
            if split == "test" and not plan["smoke"] and result["counts"]["gt_words"] != c["expected_test_care_words"]:
                raise ValueError("Incomplete TEST care-word denominator")
            final.append(dict(candidate=name, **result))
            print(json.dumps(dict(stage="final", candidate=name, split=split, counts=result["counts"])), flush=True)
        del net; torch.cuda.empty_cache()
    # Independent count replay from emitted per-image rows, separate from stored aggregates.
    for result in final:
        directory = root / "evaluation" / result["candidate"] / result["split"] / "samples"
        counts = []
        for r in records(plan, result["split"]):
            sample = read(directory / (key(r["sample_id"]) + ".json"))
            gt, valid = annotation(r)
            counts.append(lexical_counts(predictions(sample["rows"], r, valid), gt))
        if aggregate(counts) != result["counts"]:
            raise ValueError("Independent emitted-output replay failed")
    seal(root / "results.json", dict(status="COMPLETE", mode="approximate", claim_label="approximate reproduction",
        result_class=plan["result_class"], metric=plan["metric"], winner=winner, results=final,
        actual_deepsolo_calls=0, test_used_for_selection=False, historical_test_exposure=True,
        independent_emitted_output_replay="PASS"))


def execute(root):
    plan = read(root / "resolved_configuration.json")
    c = plan["config"]
    if plan["mode"] != "approximate":
        raise ValueError("Approximate study cannot enter STRICT mode")
    for file, digest in plan["code_sha256"].items():
        if sha(REPO / file) != digest:
            raise ValueError("Frozen source changed: " + file)
    torch.set_num_threads(c["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    if torch.cuda.mem_get_info()[0] < c["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient free GPU memory")
    ocr = None
    try:
        ocr = recognizer(plan)
        net = model(plan, plan["paths"]["tt_checkpoint"])
        infer_maps(net, plan, "selection", root / "source/selection/maps")
        score_maps(plan, "selection", root / "source/selection/maps", c["inference"], ocr, root / "source/selection")
        del net; torch.cuda.empty_cache()
        for branch in c["branches"]:
            train(plan, root, branch, ocr)
        a, b = (read(root / branch / "epochs.json") for branch in c["branches"])
        if [r["sample_order_sha256"] for r in a] != [r["sample_order_sha256"] for r in b]:
            raise ValueError("Unmatched augmentation/sample order across loss controls")
        finish(plan, root, ocr)
    except BaseException as error:
        seal(root / ("failure_" + str(time.time_ns()) + ".json"), dict(error=str(error), traceback=traceback.format_exc(),
            out_of_memory=isinstance(error, torch.cuda.OutOfMemoryError)))
        raise
    finally:
        if ocr is not None:
            ocr.release()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode != "approximate":
        raise ValueError("Explicit APPROXIMATE mode required")
    root = args.run_root.resolve() if args.run_root else prepare(args)
    if REPO != (root / "code_snapshot").resolve():
        import subprocess
        subprocess.run([sys.executable, "-u", "-m", "scripts.run_icdar13_matched_word_training", "--mode", args.mode,
            "--run-root", str(root)], cwd=root / "code_snapshot", env={**os.environ, "PYTHONPATH": str(root / "code_snapshot")}, check=True)
        return
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(root)


if __name__ == "__main__":
    main()
