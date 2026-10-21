"""Re-evaluate the frozen high-res residual baseline on the identical TRAIN holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from models.unet_adv import PracticalTotalTextHighResAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import (
    CONFIG_PATH as BASE_CONFIG_PATH, _load_config as base_config,
    _split_indices, _training_config, _json,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    _config, _transform, _validation_geometry,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.training import load_checkpoint


def run(args) -> dict:
    config = _config()
    old_config = base_config()
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified TRAIN manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Baseline comparison permits TRAIN only")
    _, val_ids, split_sha = _split_indices(manifest, config["training"]["seed"], 125)
    run_path = args.baseline_run.resolve() / "run_manifest.json"
    old_run = json.loads(run_path.read_text())
    contract = old_run.get("run_contract", {})
    if (old_run.get("status") != "SUCCESS" or not old_run.get("training_complete")
        or contract.get("variant") != "highres_separator_residual"
        or contract.get("validation_ids_sha256") != split_sha
        or contract.get("config_sha256") != sha256_file(BASE_CONFIG_PATH)):
        raise ValueError("Frozen baseline is incomplete or uses a different TRAIN split")
    checkpoint = args.baseline_run.resolve() / "best.pt"
    if sha256_file(checkpoint) != old_run["checkpoint"]["sha256"]:
        raise ValueError("Frozen baseline checkpoint SHA256 mismatch")
    model = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False)
    load_checkpoint(checkpoint, model=model,
        expected_architecture=model.architecture_identity, expected_run_contract=contract,
        expected_training_config=_training_config(old_config,
            old_config["variants"]["highres_separator_residual"]))
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    model.to(device).eval()
    transform = _transform(config, "wavelet_haar", train=False)
    dataset = Subset(SegmentationDataset(manifest, root, transform=transform), val_ids)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda")
    geometry = _validation_geometry(model, loader, manifest, root, transform, device=device)
    if (geometry["aggregate"]["gt_words"] != old_run["validation_word_topology"]["gt_words"]
        or geometry["aggregate"]["gt_with_predicted_bbox_iou_ge_0_5"]
            != old_run["validation_word_topology"]["gt_with_predicted_bbox_iou_ge_0_5"]
        or geometry["aggregate"]["gt_words_touched_by_fused_components"]
            != old_run["validation_word_topology"]["gt_words_touched_by_fused_components"]):
        raise RuntimeError("New validation metric path does not reproduce the frozen baseline")
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
        "split": "TRAIN_heldout_125", "baseline_checkpoint_sha256": old_run["checkpoint"]["sha256"],
        "baseline_run_manifest_sha256": sha256_file(run_path),
        "new_ablation_config_sha256": sha256_file(
            Path(__file__).resolve().parents[1] /
            "configs/approximate/totaltext_v4_image_frequency_ablation_v1.json"),
        "validation_ids_sha256": split_sha, "geometry": geometry}
    report["runtime"] = {"device": str(device), "num_workers": args.num_workers,
                         "batch_size": config["training"]["batch_size"]}
    _json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output), "aggregate": geometry["aggregate"]},
                     indent=2, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
