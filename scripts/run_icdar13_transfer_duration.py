"""Compare fixed Total-Text transfer and IC13/mixed training durations.

All paths are explicit. TEST is evaluated only after both validation selections
are sealed. This is an approximate, previously exposed research comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.icdar_budget_data import read, sha
from scripts.run_icdar13_hypothesis_tests import evaluate_net, lookup
from scripts.run_icdar13_matched_word_training import model, recognizer
from scripts.train_icdar13_word_refit import save_checkpoint
from scripts.run_yolo_f1_quality_study import seal
from segmentation.icdar13_matched_training import matched_word_loss
from segmentation.icdar13_word_refit import WordDataset


def choose_epoch(rows, max_epoch):
    """Choose solely from validation, with deterministic earliest-epoch ties."""
    eligible = [r for r in rows if r["epoch"] <= max_epoch]
    if not eligible or any(r.get("split") != "validation" for r in eligible):
        raise ValueError("Only nonempty validation rows may select checkpoints")
    return max(eligible, key=lambda r: (r["counts"]["f1"], -r["epoch"]))


def prepare(args):
    cfg = read(args.config)
    if cfg["mode"] != "approximate" or cfg["claim_label"] != "approximate reproduction":
        raise ValueError("Explicit approximate configuration required")
    if args.config.resolve().parent.name != "approximate":
        raise ValueError("Configuration must be in approximate namespace")
    old = read(args.source_plan)
    if old["mode"] != "approximate" or old["smoke"]:
        raise ValueError("Qualified full source plan required")
    for path, expected in old["config"]["pinned_inputs"].items():
        if sha(path) != expected:
            raise ValueError("Changed pinned input: " + path)
    for k in ("tt_checkpoint", "tt_contract"):
        if sha(old["light"]["paths"][k]) != old["light"]["paths"][k + "_sha256"]:
            raise ValueError("Changed source state: " + k)
    parent = read(old["light"]["paths"]["parent_plan"])
    for k in ("parseq_archive", "parseq_checkpoint"):
        path = old["light"]["paths"][k]
        if sha(path) != parent["config"]["pinned_inputs"][path]:
            raise ValueError("Changed qualified PARSeq artifact: " + k)
    records = lookup(old)
    parts = old["partitions"]
    if {k: len(v) for k, v in parts.items()} != {"fit": 183, "selection": 23, "calibration": 23, "test": 233}:
        raise ValueError("Unexpected original cohort")
    groups = [{records[s]["rgb_sha256"] for s in ids} for ids in parts.values()]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError("IC13 image leakage")
    if {r["rgb_sha256"] for r in old["tt_records"]} & set.union(*groups):
        raise ValueError("Total-Text donor overlap")
    for r in old["records"]:
        for field in (("image", "annotation", "mask", "validity", "separator", "owners") if r["split"] == "train" else ("image", "annotation")):
            if sha(r[field + "_path"]) != r[field + "_sha256"]:
                raise ValueError("Changed IC13 input " + r["sample_id"] + " " + field)
    for r in old["tt_records"]:
        for field in ("image", "mask", "validity", "separator", "owners"):
            if sha(r[field + "_path"]) != r[field + "_sha256"]:
                raise ValueError("Changed Total-Text donor " + r["sample_id"] + " " + field)
    if args.smoke:
        old["partitions"] = {k: v[:(8 if k == "fit" else 4)] for k, v in parts.items()}
        cfg["epochs"], cfg["validation_epochs"], cfg["test_fixed_epochs"] = 1, [0, 1], [0, 1]
    out = args.output / ("smoke" if args.smoke else "full")
    out.mkdir(parents=True, exist_ok=False)
    recovery = None
    if args.reuse_completed:
        previous = args.reuse_completed.resolve()
        previous_plan = read(previous / "resolved_configuration.json")
        if args.smoke or previous_plan["config"] != cfg or previous_plan["source_plan_sha256"] != sha(args.source_plan):
            raise ValueError("Recovery must preserve the complete original scientific configuration")
        old_history = read(previous / "icdar13_only/epochs.json")
        if [r["epoch"] for r in old_history] != list(range(1, cfg["epochs"] + 1)):
            raise ValueError("Original ICDAR-only arm is not complete")
        for epoch in cfg["validation_epochs"][1:]:
            if not (previous / "icdar13_only" / f"epoch_{epoch:03d}.pt").exists():
                raise ValueError("Original checkpoint missing")
        shutil.copytree(previous / "source", out / "source", copy_function=os.link)
        shutil.copytree(previous / "icdar13_only", out / "icdar13_only", copy_function=os.link)
        recovery = dict(previous_root=str(previous), previous_plan_sha256=sha(previous / "resolved_configuration.json"),
            previous_ic_history_sha256=sha(previous / "icdar13_only/epochs.json"),
            previous_mixed_history_sha256=sha(previous / "icdar13_totaltext_mixed/epochs.json"),
            description="Reuse completed ICDAR-only arm; deterministically replay mixed arm from source and verify original epochs and available model tensors.")
    source = Path(__file__).resolve()
    shutil.copyfile(source, out / "runner_snapshot.py")
    repo = source.parents[1]
    code_paths = ["scripts/run_icdar13_hypothesis_tests.py", "scripts/run_icdar13_matched_word_training.py",
        "scripts/train_icdar13_word_refit.py", "scripts/finish_icdar13_word_refit.py", "scripts/icdar_budget_data.py",
        "segmentation/icdar13_matched_training.py", "segmentation/icdar13_word_refit.py", "segmentation/word_multitask.py",
        "recognition/icdar13_hypothesis_tests.py", "recognition/icdar13_matched_router.py", "recognition/parseq.py"]
    code = {name: sha(repo / name) for name in code_paths}
    for name, digest in code.items():
        target = out / "code_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
        if sha(target) != digest: raise ValueError("Source changed during snapshot: " + name)
    plan = dict(mode="approximate", claim_label="approximate reproduction", result_class="smoke-test result" if args.smoke else "approximate reproduction",
        config=cfg, inherited=old, source_plan=str(args.source_plan.resolve()), source_plan_sha256=sha(args.source_plan),
        config_sha256=sha(args.config), runner_sha256=sha(source), code_sha256=code, paper_sha256=sha(args.paper), paper=str(args.paper.resolve()),
        recovery=recovery, test_used_for_selection=False, hardware=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        precision="float32", segmentation_train_batch=8, inference_batch=1, parseq_batch=old["config"]["parseq_batch_size"], image_hw=old["light"]["config"]["image_hw"]))
    seal(out / "resolved_configuration.json", plan)
    return out


def run(root):
    plan = read(root / "resolved_configuration.json")
    cfg, old = plan["config"], plan["inherited"]
    if sha(root / "runner_snapshot.py") != plan["runner_sha256"] or sha(Path(__file__)) != plan["runner_sha256"]:
        raise ValueError("Runner changed after preparation")
    repo = Path(__file__).resolve().parents[1]
    for name, digest in plan["code_sha256"].items():
        if sha(repo / name) != digest or sha(root / "code_snapshot" / name) != digest:
            raise ValueError("Scientific code changed: " + name)
    if (root / "selection_frozen.json").exists():
        raise ValueError("Run already has frozen selections; preserve existing results")
    torch.set_num_threads(old["config"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.init()
    if torch.cuda.mem_get_info()[0] < cfg["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient free GPU memory; no implicit adaptation")
    import cv2
    cv2.setNumThreads(1)
    ids = old["partitions"]
    validation_ids = sorted(ids["selection"] + ids["calibration"])
    if set(validation_ids) & (set(ids["fit"]) | set(ids["test"])):
        raise ValueError("Validation IDs overlap fit or test")
    train_cfg, train = old["light"]["config"], old["light"]["config"]["training"]
    rec = lookup(old)
    seed = old["config"]["seed"]
    ocr = recognizer(old["light"])
    try:
        if plan.get("recovery"):
            base_val = read(root / "source/validation/metrics.json")["counts"]
        else:
            base = model(old["light"], old["light"]["paths"]["tt_checkpoint"]).eval()
            base_val = evaluate_net(base, ocr, old, validation_ids, root / "source/validation")
            del base
            torch.cuda.empty_cache()
        choices = {}
        for arm in cfg["training_arms"]:
            mixed = arm == "icdar13_totaltext_mixed"
            if arm not in ("icdar13_only", "icdar13_totaltext_mixed"):
                raise ValueError("Unknown training arm")
            dest = root / arm
            if arm == "icdar13_only" and plan.get("recovery"):
                history = read(dest / "epochs.json")
                rows = [dict(epoch=0, split="validation", counts=base_val, checkpoint=old["light"]["paths"]["tt_checkpoint"], checkpoint_sha256=old["light"]["paths"]["tt_checkpoint_sha256"])]
                for epoch in cfg["validation_epochs"][1:]:
                    path = dest / f"epoch_{epoch:03d}.pt"
                    counts = read(dest / "validation" / f"epoch_{epoch:03d}" / "metrics.json")["counts"]
                    if abs(counts["f1"] - history[epoch-1]["validation_f1"]) > 1e-14:
                        raise ValueError("Original validation history changed")
                    rows.append(dict(epoch=epoch, split="validation", counts=counts, checkpoint=str(path), checkpoint_sha256=sha(path)))
                choices[arm] = dict(validation=rows, best30=choose_epoch(rows, 30), best60=choose_epoch(rows, cfg["epochs"]),
                    peak_allocated=max(r["gpu_allocated"] for r in history), peak_reserved=max(r["gpu_reserved"] for r in history))
                print(json.dumps(dict(stage="reuse_completed", arm=arm, epochs=len(history))), flush=True)
                continue
            if dest.exists():
                raise ValueError("Incomplete or existing arm; preserve and inspect: " + str(dest))
            dest.mkdir()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            net = model(old["light"], old["light"]["paths"]["tt_checkpoint"])
            frozen = {n: v.detach().cpu().clone() for n, v in net.state_dict().items() if n.startswith("encoder.")}
            opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=train["learning_rate"],
                weight_decay=train["weight_decay"], betas=tuple(train["adam_betas"]), eps=train["adam_epsilon"])
            fit = WordDataset([rec[s] for s in ids["fit"]], train_cfg, augment=True)
            tt = WordDataset(old["tt_records"], train_cfg, augment=True) if mixed else None
            rows = [dict(epoch=0, split="validation", counts=base_val, checkpoint=old["light"]["paths"]["tt_checkpoint"], checkpoint_sha256=old["light"]["paths"]["tt_checkpoint_sha256"])]
            history = []
            previous_mixed = read(Path(plan["recovery"]["previous_root"]) / "icdar13_totaltext_mixed/epochs.json") if mixed and plan.get("recovery") else []
            for epoch in range(1, cfg["epochs"] + 1):
                fit.transform.set_epoch(epoch - 1)
                loader = DataLoader(fit, batch_size=train["physical_batch_size"], num_workers=train["num_workers"], shuffle=True,
                    generator=torch.Generator().manual_seed(seed + epoch), pin_memory=True)
                if mixed:
                    tt.transform.set_epoch(epoch - 1)
                    donors = iter(DataLoader(tt, batch_size=train["physical_batch_size"], num_workers=train["num_workers"], shuffle=True,
                        generator=torch.Generator().manual_seed(old["config"]["fold_seed"] + epoch), pin_memory=True))
                net.train(); seen = []; used = []; total = 0.; steps = 0
                torch.cuda.reset_peak_memory_stats()
                for batch in loader:
                    local = {k: v.to(train_cfg["device"]) if torch.is_tensor(v) else v for k, v in batch.items()}
                    opt.zero_grad(set_to_none=True)
                    loss = matched_word_loss(net(local["image"]), local, train_cfg, "totaltext_bce_dice")
                    if not torch.isfinite(loss): raise FloatingPointError("Non-finite IC loss")
                    weight = cfg["mixed_totaltext_weight"] if mixed else 0.
                    ((1. - weight) * loss).backward()
                    if mixed:
                        donor = next(donors); size = len(batch["sample_id"]); used.extend(donor["sample_id"][:size])
                        domain = {k: v[:size].to(train_cfg["device"]) if torch.is_tensor(v) else v[:size] for k, v in donor.items()}
                        extra = matched_word_loss(net(domain["image"]), domain, train_cfg, "totaltext_bce_dice")
                        if not torch.isfinite(extra): raise FloatingPointError("Non-finite donor loss")
                        (weight * extra).backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                        raise FloatingPointError("Non-finite gradient")
                    opt.step(); seen.extend(batch["sample_id"]); total += float(loss.detach()) * len(batch["sample_id"]); steps += 1
                if sorted(seen) != sorted(ids["fit"]) or len(used) != (len(seen) if mixed else 0):
                    raise ValueError("Incomplete training population")
                row = dict(epoch=epoch, ic_images=len(seen), tt_images=len(used), steps=steps, loss=total / len(seen),
                    ic_order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
                    tt_order_sha256=hashlib.sha256(('\n'.join(used)+'\n').encode()).hexdigest(),
                    gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved())
                if epoch <= len(previous_mixed):
                    original = previous_mixed[epoch-1]
                    for field in ("ic_images", "tt_images", "steps", "loss", "ic_order_sha256", "tt_order_sha256"):
                        if row[field] != original[field]: raise ValueError("Interrupted mixed replay differs at epoch " + str(epoch) + ": " + field)
                if epoch in cfg["validation_epochs"]:
                    counts = evaluate_net(net, ocr, old, validation_ids, dest / "validation" / f"epoch_{epoch:03d}")
                    checkpoint = dest / f"epoch_{epoch:03d}.pt"
                    save_checkpoint(checkpoint, dict(model_state_dict=net.state_dict(), epoch=epoch, arm=arm, plan_sha256=sha(root / "resolved_configuration.json")))
                    if previous_mixed and epoch <= len(previous_mixed):
                        previous_checkpoint = Path(plan["recovery"]["previous_root"]) / arm / f"epoch_{epoch:03d}.pt"
                        saved = torch.load(previous_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
                        if any(not torch.equal(saved[n], v.detach().cpu()) for n, v in net.state_dict().items()):
                            raise ValueError("Interrupted mixed model tensors differ at epoch " + str(epoch))
                    rows.append(dict(epoch=epoch, split="validation", counts=counts, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
                    row["validation_f1"] = counts["f1"]
                history.append(row)
                (dest / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
                print(json.dumps(dict(stage="training", arm=arm, **row)), flush=True)
            if any(not torch.equal(net.state_dict()[n].cpu(), v) for n, v in frozen.items()):
                raise ValueError("Frozen encoder changed")
            choices[arm] = dict(validation=rows, best30=choose_epoch(rows, min(30, cfg["epochs"])), best60=choose_epoch(rows, cfg["epochs"]),
                peak_allocated=max(r["gpu_allocated"] for r in history), peak_reserved=max(r["gpu_reserved"] for r in history))
            del net, opt
            torch.cuda.empty_cache()
        frozen = dict(choices=choices, validation_ids=validation_ids, test_used_for_selection=False, policy=old["selected"]["settings"], plan_sha256=sha(root / "resolved_configuration.json"))
        seal(root / "selection_frozen.json", frozen)
        # Only after both arms and validation-only choices are frozen.
        test = {}
        for arm, data in choices.items():
            requested = sorted(set(cfg["test_fixed_epochs"] + [data["best30"]["epoch"], data["best60"]["epoch"]]))
            test[arm] = {}
            for epoch in requested:
                item = next(r for r in data["validation"] if r["epoch"] == epoch)
                if sha(item["checkpoint"]) != item["checkpoint_sha256"]: raise ValueError("Checkpoint changed after selection")
                net = model(old["light"], item["checkpoint"]).eval()
                counts = evaluate_net(net, ocr, old, ids["test"], root / arm / "test" / f"epoch_{epoch:03d}")
                if plan["result_class"] != "smoke-test result" and (counts["images"] != 233 or counts["gt_words"] != 1095): raise ValueError("Full test denominator changed")
                test[arm][str(epoch)] = counts
                print(json.dumps(dict(stage="test", arm=arm, epoch=epoch, counts=counts)), flush=True)
                del net; torch.cuda.empty_cache()
        seal(root / "results.json", dict(status="SUCCESS", mode="approximate", claim_label="approximate reproduction", result_class=plan["result_class"],
            selection_sha256=sha(root / "selection_frozen.json"), choices=choices, test=test, deepsolo_images=0,
            deepsolo_total=len(ids["test"]), deepsolo_percentage=0., test_used_for_selection=False))
    finally:
        ocr.release()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source-plan", type=Path, required=True)
    p.add_argument("--paper", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--reuse-completed", type=Path, help="Preserve and reuse a completed ICDAR-only arm; replay the interrupted mixed arm")
    args = p.parse_args()
    root = prepare(args)
    try:
        run(root)
    except BaseException as exc:
        (root / ("failure_" + str(time.time_ns()) + ".json")).write_text(json.dumps(dict(error=repr(exc), traceback=traceback.format_exc(), out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError)), indent=2))
        raise


if __name__ == "__main__":
    main()
