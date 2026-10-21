"""Export separated CTW1500 TEST word targets from a pinned source archive."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
import yaml

from scripts.prepare_separated_word_gt import preview, write_json
from segmentation.ctw1500_word_gt import (parse_vimts_word_gt, restore_split_single_token_lines,
                                        remove_redundant_word_annotations)
from recognition.scene_text_datasets import parse_ctw1500_test_annotation
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent, file_sha256
from segmentation.separated_word_gt import separated_word_regions, widen_word_gaps


def prepare(args) -> dict:
    root = args.output_root / "ctw1500/test"
    if root.exists():
        raise ValueError(f"Refusing to overwrite CTW targets: {root}")
    source_path = args.prepared_root / "manifests/ctw1500_test.yaml"
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if source["dataset"] != "ctw1500" or source["split_role"] != "test" or len(source["samples"]) != 500:
        raise ValueError("Expected complete CTW1500 TEST source manifest")
    source_note = json.loads(args.source_provenance.read_text())
    if source_note["archive_sha256"] != file_sha256(args.archive):
        raise ValueError("Word GT archive does not match the pinned source provenance")
    root.mkdir(parents=True)
    config = {"schema_version": 1, "reproduction_mode": "approximate",
              "claim_label": "approximate reproduction", "classification": "ASSUMPTION",
              "result_class": "smoke-test result" if args.limit else "approximate reproduction",
              "dataset": "ctw1500", "split": "test", "expected_images": 500,
              "word_gt_variant": "VimTS word-level TEST ground truth",
              "archive": str(args.archive), "archive_sha256": file_sha256(args.archive),
              "source_provenance": source_note,
              "source_manifest": str(source_path), "source_manifest_sha256": file_sha256(source_path),
              "prepared_root": str(args.prepared_root), "output_root": str(args.output_root),
              "clearance_reference_pixels": args.clearance, "reference_wh": args.reference_size,
              "max_word_area_loss_fraction": args.max_word_area_loss,
              "protected_stroke_gt": "unavailable", "ignore_policy": args.ignore_policy,
              "single_token_policy": args.single_token_policy,
              "redundancy_policy": "exact duplicate removal; strict lexical fragment with full polygon containment",
              "ambiguity_id": "WORD-GT-001", "original_line_gt_overwritten": False,
              "code_sha256": {str(p): file_sha256(p) for p in [Path(__file__),
                  Path("segmentation/ctw1500_word_gt.py"), Path("segmentation/separated_word_gt.py"),
                  Path("segmentation/totaltext_word_regions.py")]}}
    write_json(root / "resolved_config.json", config)
    records = []
    with ZipFile(args.archive) as archive:
        filenames = [n for n in archive.namelist() if not n.endswith("/")]
        by_id = {}
        for name in filenames:
            if Path(name).suffix != ".txt" or not Path(name).stem.isdecimal():
                raise ValueError(f"Unexpected word annotation filename: {name}")
            sample_id = str(int(Path(name).stem))
            if sample_id in by_id:
                raise ValueError(f"Duplicate word-GT sample{sample_id}")
            by_id[sample_id] = name
        expected = {r["id"].split(":")[-1] for r in source["samples"]}
        if set(by_id) != expected or len(by_id) != 500:
            raise ValueError("Word archive IDs differ from the500 downloaded CTW TEST images")
        for offset, row in enumerate(source["samples"][:args.limit] if args.limit else source["samples"], 1):
            sample_id = row["id"].split(":")[-1]
            image_path = args.prepared_root / row["image"]
            try:
                if file_sha256(image_path) != row["image_sha256"]:
                    raise ValueError(f"CTW image checksum changed: {sample_id}")
                with Image.open(image_path) as opened:
                    size = opened.size
                payload = archive.read(by_id[sample_id])
                import hashlib
                instances, clipped = clip_instances_to_pixel_extent(
                    parse_vimts_word_gt(payload.decode("utf-8-sig")), size)
                raw_instance_count=len(instances)
                instances,redundancy_audit=remove_redundant_word_annotations(instances,size)
                original_path = args.prepared_root / "datasets/ctw1500/test/recognition_annotations" / f"{sample_id}.txt"
                merge_audit = None
                if args.single_token_policy == "restore_matching_original_line":
                    instances, merge_audit = restore_split_single_token_lines(instances,
                        parse_ctw1500_test_annotation(original_path,image_size_wh=size))
                mask, valid, sep, owners, audit = separated_word_regions(
                    size, instances, ignore_policy=args.ignore_policy)
                old_owners = np.asarray(owners).copy()
                mask, owners, gap_audit = widen_word_gaps(owners, clearance=args.clearance,
                    reference_wh=tuple(args.reference_size), max_area_loss=args.max_word_area_loss)
                removed = (old_owners > 0) & (np.asarray(owners) == 0)
                new_sep = np.asarray(sep).copy(); new_sep[removed & (np.asarray(valid)>0)] = 255
                sep = Image.fromarray(new_sep)
                record = {"id": row["id"], "image": str(image_path),
                          "image_sha256": row["image_sha256"], "size_wh": list(size),
                          "source_archive_entry": by_id[sample_id],
                          "source_annotation_sha256": hashlib.sha256(payload).hexdigest(),
                          "clipped_source_instances": clipped, "audit": audit, "gap_audit": gap_audit,
                          "single_token_recovery": merge_audit,
                          "raw_vimts_instance_count":raw_instance_count,"redundancy_audit":redundancy_audit,
                          "original_line_annotation": str(original_path),
                          "original_line_annotation_sha256":file_sha256(original_path),
                          "instances": [{"owner_id": i, **asdict(item)} for i, item in
                                        enumerate([x for x in instances if not x.ignore], 1)],
                          "ignored_instances": [asdict(x) for x in instances if x.ignore]}
                annotation_path = root / "source_annotations" / f"{sample_id}.txt"
                annotation_path.parent.mkdir(parents=True, exist_ok=True)
                annotation_path.write_bytes(payload)
                for name, item in [("masks", mask), ("owners", owners), ("validity", valid), ("separators", sep)]:
                    path = root / name / f"{sample_id}.png"; path.parent.mkdir(parents=True, exist_ok=True)
                    item.save(path, format="PNG")
                    with Image.open(path) as saved:
                        if not np.array_equal(np.asarray(saved), np.asarray(item)):
                            raise ValueError(f"CTW PNG roundtrip changed pixels: {path}")
                    record[name] = path.relative_to(args.output_root).as_posix()
                    record[name + "_sha256"] = file_sha256(path)
                write_json(root / "annotations" / f"{sample_id}.json", record)
                records.append(record)
                if offset <= 3:
                    preview(image_path, owners, mask, root / "previews" / f"{sample_id}.png",
                            f"CTW1500 TEST {sample_id}: word GT from VimTS")
            except Exception as error:
                write_json(root / "failure.json", {"sample_id": sample_id, "completed_images": len(records),
                           "error_type": type(error).__name__, "error": str(error)})
                raise
            if offset % 50 == 0 or offset == (args.limit or 500):
                print(f"ctw1500/test: {offset}/{args.limit or 500} images", flush=True)
    report = {"schema_version": 1, "claim_label": "approximate reproduction", "classification": "ASSUMPTION",
              "dataset": "ctw1500", "split": "test", "result_class": config["result_class"],
              "complete_official_split": args.limit is None,
              "word_gt_variant": config["word_gt_variant"],
              "images": len(records), "words": sum(r["audit"]["words"] for r in records),
              "ignored_instances": sum(r["audit"]["ignored_instances"] for r in records),
              "gap_changed_words": sum(r["gap_audit"]["changed_words"] for r in records),
              "gap_pixels_removed": sum(r["gap_audit"]["removed_pixels"] for r in records),
              "fragmented_words": sum(len(r["gap_audit"]["fragmented_word_ids"]) for r in records),
              "merged_original_single_token_lines":sum((r["single_token_recovery"] or {}).get(
                  "merged_original_single_token_lines",0) for r in records),
              "exact_duplicates_removed":sum(r["redundancy_audit"]["exact_duplicates_removed"] for r in records),
              "contained_lexical_fragments_removed":sum(r["redundancy_audit"]["contained_lexical_fragments_removed"] for r in records),
              "equal_text_nested_duplicates_removed":sum(r["redundancy_audit"]["equal_text_nested_duplicates_removed"] for r in records),
              "missing_words": 0, "interword_contact_pairs": 0,
              "resolved_config_sha256": file_sha256(root/"resolved_config.json"), "samples": records}
    write_json(root / "manifest.json", report)
    if args.zip:
        destination = args.output_root / "ctw1500-test-separated-word-gt.zip"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite archive: {destination}")
        with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(args.output_root))
        print(f"ARCHIVE {destination} SHA256 {file_sha256(destination)}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    for name in ["prepared-root", "output-root", "archive", "source-provenance"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--ignore-policy", choices=["care_region_precedence", "ignore_region_precedence"], required=True)
    parser.add_argument("--clearance", type=float, required=True)
    parser.add_argument("--reference-size", nargs=2, type=int, required=True)
    parser.add_argument("--max-word-area-loss", type=float, required=True)
    parser.add_argument("--single-token-policy",choices=["retain_vimts", "restore_matching_original_line"],required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.output_root=args.output_root.resolve(); args.prepared_root=args.prepared_root.resolve()
    if args.prepared_root == args.output_root or args.prepared_root in args.output_root.parents:
        parser.error("Output must be isolated from source data")
    prepare(args)


if __name__ == "__main__":
    main()
