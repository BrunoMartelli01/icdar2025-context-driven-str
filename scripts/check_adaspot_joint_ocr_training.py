"""Fit-only single-word OCR memorization diagnostic after full joint training."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from models.adaspot_mask_align import MaskAlignOptions
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, write_json
from scripts.evaluate_adaspot_joint_selective import load_completed
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import memory
from scripts.run_adaspot_joint import predicted_rois
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.adaspot_joint import match_owner_texts, label_support
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.training import seed_everything


def run(args):
    cp = args.config.resolve()
    d = json.loads(cp.read_text())
    if args.mode != "approximate" or d["reproduction_mode"] != args.mode or REPO / "configs/approximate" not in cp.parents or d["steps"] < 1:
        raise ValueError("Explicit approximate diagnostic config required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA GPU required")
    root, out = args.run_root.resolve(), args.output_root.resolve()
    if out.exists() or out == REPO or REPO in out.parents:
        raise ValueError("New external diagnostic output root required")
    seed_everything(d["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, p = load_completed(root, device)
    torch.set_num_threads(p["resolved_config"]["cpu_threads"])
    c = p["resolved_config"]
    manifest = load_segmentation_manifest(Path(p["train_manifest"]), require_checksums=True)
    base = SegmentationDataset(manifest, Path(p["word_root"]), transform=_transform(p["resolved_preprocessing"], "letterbox", train=False))
    dataset = AdaSpotGapDataset(base, distance_fraction=c["distance_fraction"], gap_radius=c["gap_radius"])
    annotations = json.loads((root / "annotation_mapping.json").read_text())
    by_id = {s.sample_id: i for i, s in enumerate(manifest.samples)}
    selected = None
    with torch.no_grad():
        for sid in p["fit_ids"]:
            batch = dataset[by_id[sid]]
            evidence = model.encode(batch["image"][None].to(device))
            record = annotations[sid]
            labels, proposals, _, _, _ = predicted_rois(evidence, batch, c, tuple(record["image_wh"]))
            matched, _ = match_owner_texts(proposals, batch["owners_native"].numpy(),
                {int(k): v for k, v in record["texts"].items()}, iou_threshold=c["training_match_iou"])
            eligible = [i for i, v in sorted(matched.items()) if label_support(v["text"], model.light_head.tokenizer, c["max_label_length"]) is None]
            if eligible:
                index = eligible[0]
                mask = labels == proposals[index]["instance_id"]
                light, _ = model.align(evidence, [mask], view=c["alignment_view"], options=MaskAlignOptions(**c["geometry"]), deep=False)
                deep, _ = model.align(evidence, [mask], view=c["alignment_view"], options=MaskAlignOptions(**c["geometry"]), deep=True)
                selected = {"sample_id": sid, "proposal": proposals[index], "match": matched[index]}
                break
    if selected is None:
        raise RuntimeError("No supported matched fit word available for sanity check")
    out.mkdir(parents=True)
    args._output_created = True
    np.savez_compressed(out / "actual_feature_grids.npz", light=light.cpu().numpy(), deep=deep.cpu().numpy())
    model.backbone.requires_grad_(False)
    model.gap.requires_grad_(False)
    model.gate.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(model.light_head.parameters()) + list(model.deep_head.parameters()),
        lr=d["learning_rate"], weight_decay=d["weight_decay"])
    tokens = model.light_head.tokenizer.encode([selected["match"]["text"]], device=device)
    target = tokens[:, 1:]
    def ce(logits):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=model.light_head.tokenizer.pad_id)
    @torch.no_grad()
    def inspect(step):
        model.light_head.eval()
        model.deep_head.eval()
        first, refined = model.light_head.light(model.light_head.memory(light))
        deep_memory = model.deep_head.memory(deep)
        reading_light = model.light_head.read(refined)[0]
        reading_deep = model.deep_head.read(model.deep_head.greedy(deep_memory))[0]
        return {"step": step, "light_ce": float(.5 * (ce(first[:, :target.shape[1]]) + ce(refined[:, :target.shape[1]]))),
            "deep_teacher_ce": float(ce(model.deep_head.teacher(deep_memory, tokens))),
            "light_text": reading_light["text"], "deep_greedy_text": reading_deep["text"],
            "light_exact_casefold": reading_light["text"].casefold() == selected["match"]["text"].casefold(),
            "deep_exact_casefold": reading_deep["text"].casefold() == selected["match"]["text"].casefold()}
    history = [inspect(0)]
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, d["steps"] + 1):
        model.light_head.train()
        model.deep_head.train()
        optimizer.zero_grad(set_to_none=True)
        first, refined = model.light_head.light(model.light_head.memory(light.detach()))
        loss_light = .5 * (ce(first[:, :target.shape[1]]) + ce(refined[:, :target.shape[1]]))
        loss_deep = ce(model.deep_head.teacher(model.deep_head.memory(deep.detach()), tokens))
        loss = loss_light + loss_deep
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite fit-word diagnostic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.light_head.parameters()) + list(model.deep_head.parameters()), d["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if step % d["log_every"] == 0 or step == d["steps"]:
            history.append(inspect(step))
            print(json.dumps(history[-1]), flush=True)
    source_hashes = {str(root / name): sha256_file(root / name) for name in ("results.json", "training/last.pt", "independent_audit.json")}
    result = {"status": "SUCCESS", "result_class": "Smoke-test result", "scope": "single fit-word memorization only; no benchmark or generalization claim",
        "resolved_config": d, "selection": selected, "history": history,
        "both_heads_can_memorize_this_word": history[-1]["light_exact_casefold"] and history[-1]["deep_exact_casefold"],
        "source_hashes": source_hashes, "script_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(cp),
        "actual_feature_grids_sha256": sha256_file(out / "actual_feature_grids.npz"),
        "source_checkpoint_not_overwritten": True, "test_used": False,
        "memory": memory(device, "light NAR1+cloze and AR4 OCR only; cached real U-Net features; frozen backbone/gate", [1, 64, 8, 32])}
    write_json(out / "results.json", result)
    print(json.dumps({"both_heads_can_memorize_this_word": result["both_heads_can_memorize_this_word"], "selection": selected}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if getattr(args, "_output_created", False):
            write_json(args.output_root / "failure.json", {"status": "FAILED", "error": repr(error),
                "oom": isinstance(error, torch.cuda.OutOfMemoryError), "traceback": traceback.format_exc()})
        raise
