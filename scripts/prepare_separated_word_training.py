"""Bridge new Total-Text word masks to the existing approximate V4 trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from segmentation.config import load_segmentation_config
from segmentation.data import (
    SegmentationDataset, file_sha256, load_segmentation_manifest,
)
from segmentation.train import _transform, _validate_scene_text_training_manifest


PROTOCOL = "practical_totaltext_separated_words_500_v1"
TARGET = "v4_wider_gaps_separated_word_gt_v1"
ARTIFACT = "ART-TOTALTEXT-SEPARATED-WORDS-V1"
VARIANTS = {
    "separated_v1": {"protocol": PROTOCOL, "target": TARGET, "artifact": ARTIFACT,
                     "run_name": "total-text-separated-words-v1-500ep"},
    "compact_v2": {"protocol": "practical_totaltext_tighter_words_500_v2",
                   "target": "compact_word_cores_v2", "artifact": "ART-TOTALTEXT-COMPACT-WORDS-V2",
                   "run_name": "total-text-tighter-words-v2-500ep"},
}


def relative_input(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Input must be inside the explicit --data-root: {path}") from error


def write_frozen(path: Path, value: dict) -> None:
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite a different training artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def training_manifest(gt_root: Path, data_root: Path, *, variant="separated_v1") -> dict:
    identity = VARIANTS[variant]
    source_path = gt_root / "total_text/train/manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("dataset") != "total_text" or source.get("split") != "train"
        or source.get("images") != 1255 or len(source["samples"]) != 1255
        or source.get("claim_label") != "approximate reproduction"
        or source.get("complete_official_split") is not True
        or source.get("missing_words") != 0 or source.get("interword_contact_pairs") != 0):
        raise ValueError("Requires the complete qualified separated Total-Text TRAIN-1255 GT")
    audit_path = gt_root / "independent_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    qualified = [r for r in audit["datasets"] if r["dataset"] == "total_text" and r["split"] == "train"]
    if (len(qualified) != 1 or qualified[0]["manifest_sha256"] != file_sha256(source_path)
        or qualified[0]["images"] != 1255 or qualified[0]["missing_words"] != 0
        or qualified[0]["merged_word_components"] != 0):
        raise ValueError("GT manifest differs from the independently verified export")
    source_config = gt_root / "total_text/train/resolved_config.json"
    if file_sha256(source_config) != source["resolved_config_sha256"]:
        raise ValueError("GT resolved configuration checksum mismatch")
    if variant == "compact_v2" and source.get("target_source") != identity["target"]:
        raise ValueError("Compact-v2 training requires independently qualified compact cores")
    if variant == "separated_v1" and source.get("target_source") not in (None, identity["target"]):
        raise ValueError("Separated-v1 training cannot relabel a compact or different target variant")
    samples = []
    field_map = {"masks": "mask", "validity": "validity_annotation",
                 "separators": "separation_annotation", "owners": "word_owner_annotation"}
    for row in source["samples"]:
        item = {"id": row["id"], "image": relative_input(Path(row["image"]), data_root),
                "image_sha256": row["image_sha256"]}
        for source_field, target_field in field_map.items():
            item[target_field] = relative_input(gt_root / row[source_field], data_root)
            item[target_field + "_sha256"] = row[source_field + "_sha256"]
        samples.append(item)
    if len({r["id"] for r in samples}) != len(samples):
        raise ValueError("Duplicate TRAIN image IDs")
    return {"schema_version": 1, "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction", "dataset": "total_text",
            "split": "official-train-separated-words-v1" if variant == "separated_v1" else "official-train-compact-word-cores-v2", "split_role": "train",
            "artifact_id": identity["artifact"],
            "split_policy": {"algorithm_id": "official-split-no-holdout-v1",
                             "classification": "PAPER_SPECIFIED", "validation_count": 0},
            "acquisition": {"classification": "ASSUMPTION", "variant_class": "PRACTICAL_APPROXIMATION",
                            "source_gt_manifest": str(source_path.resolve()),
                            "source_gt_manifest_sha256": file_sha256(source_path),
                            "independent_audit_sha256": file_sha256(audit_path),
                            "source_gt_config_sha256": file_sha256(source_config)},
            "annotation_policy": {"target_conversion": "binary-mask-with-validity-v1",
                                  "target_source": identity["target"], "target_classification": "ASSUMPTION",
                                  "masked_bce": True, "dont_care_training_behavior": "excluded_by_validity_mask",
                                  "mask_encoding": "lossless_png_values_0_or_255",
                                  "validity_encoding": "lossless_png_values_0_or_255",
                                  "training_ready": True,
                                  "separation_map": "source_gaps_plus_resized_owner_contact_removal"},
            "samples": samples}


def prepare(args) -> dict:
    variant = getattr(args, "variant", "separated_v1")
    identity = VARIANTS[variant]
    if args.config_output.parent.name != "approximate":
        raise ValueError("New training config must be in an explicit approximate namespace")
    base = load_segmentation_config(args.base_config)
    if base.training.protocol != "practical_totaltext_curved_word_boxes_500_v1":
        raise ValueError("Use the qualified frozen V4 training config as the comparison baseline")
    manifest_path = args.output_root / "manifests/total_text_train.yaml"
    write_frozen(manifest_path, training_manifest(args.gt_root, args.data_root, variant=variant))
    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["run"]["name"] = identity["run_name"]
    config["training"]["protocol"] = identity["protocol"]
    config["data"]["train"].update({"artifact_id": identity["artifact"], "target_source": identity["target"],
                                     "manifest_sha256": file_sha256(manifest_path)})
    config["scientific_choices"] = [
        {"component": "separated_word_training", "classification": "ASSUMPTION",
         "rationale": "Train on qualified V4-derived wider-gap full-word masks; WORD-GT-001. Original V4/strict targets are preserved."},
        {"component": "matched_v4_training_protocol", "classification": "REPOSITORY_SPECIFIED",
         "rationale": "Reuse V4 AG-U-Net, frozen ResNet50, FP32/batch8, 500 epochs, transforms, loss and TRAIN-IoU checkpoint selection for a target comparison."},
        {"component": "native128_target_topology", "classification": "ASSUMPTION",
         "rationale": "Existing paired transforms resize owner IDs with nearest and reapply contact separation; very small words can disappear at128. This is not a guarantee of original-resolution word survival."},
    ]
    if variant == "compact_v2":
        compact_config = json.loads((args.gt_root / "total_text/train/resolved_config.json").read_text())
        config["supervision"] = {"classification": "ASSUMPTION", "ambiguity_id": "WORD-GT-002",
                                 "source_config_sha256": file_sha256(args.gt_root / "total_text/train/resolved_config.json"),
                                 "parameters": compact_config["parameters"]}
        config["scientific_choices"][0] = {
            "component": "compact_word_cores", "classification": "ASSUMPTION",
            "rationale": "Smaller connected word interiors; paper padding does not specify GT erosion. WORD-GT-002. TRAIN parameters are explicit, untuned; previous targets and TEST transcripts are preserved."}
    write_frozen(args.config_output, config)
    loaded = load_segmentation_config(args.config_output)
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    _validate_scene_text_training_manifest(manifest, expected_artifact_id=identity["artifact"],
        expected_sample_count=1255, expected_datasets=("total_text",))
    dataset = SegmentationDataset(manifest, args.data_root,
        transform=_transform(loaded.transforms, seed=int(loaded.training.seed), augmentation_enabled=False),
        verify_checksums=True)
    # Decode representative examples, including both protected V4 fragmented cases.
    wanted = {"total_text:11", "total_text:228", "total_text:1111"}
    smoke = []
    for i, sample in enumerate(manifest.samples):
        if sample.sample_id not in wanted:
            continue
        row = dataset[i]
        if tuple(row["image"].shape) != (3, 256, 256) or tuple(row["mask"].shape) != (1, 128, 128):
            raise ValueError("Training image/target tensor shape mismatch")
        if (row["mask"] * row["separation"]).any():
            raise ValueError("Native training target overlaps separator pixels")
        smoke.append(sample.sample_id)
    if set(smoke) != wanted:
        raise ValueError("Missing required full-dataset transform smoke examples")
    test = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if test.dataset != "total_text" or test.split_role != "test" or len(test.samples) != 300:
        raise ValueError("Evaluation requires official Total-Text TEST-300")
    train_ids = {s.sample_id for s in manifest.samples}
    if train_ids & {s.sample_id for s in test.samples}:
        raise ValueError("TRAIN/TEST image IDs overlap")
    report = {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
              "training_images": len(dataset), "test_images": len(test.samples),
              "data_root": str(args.data_root.resolve()), "manifest_root": str(args.output_root.resolve()),
              "config": str(args.config_output.resolve()), "config_sha256": file_sha256(args.config_output),
              "manifest_sha256": file_sha256(manifest_path), "test_manifest_sha256": file_sha256(args.test_manifest),
              "verified_all_training_input_checksums": True, "transform_smoke_ids": smoke,
              "gpu_training_started": False, "epochs": int(loaded.training.epochs)}
    (args.output_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), default="separated_v1")
    for name in ("gt-root", "data-root", "output-root", "base-config", "config-output", "test-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
