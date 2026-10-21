"""Generate isolated smaller word GT from the complete separated-v1 delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_separated_word_gt import preview, write_json
from segmentation.scene_text_preparation import file_sha256
from segmentation.tighter_word_gt import compact_word_owners


def prepare(args, dataset, split):
    source_path = args.source_root / dataset / split / "manifest.json"
    source = json.loads(source_path.read_text())
    if (source["claim_label"] != "approximate reproduction"
        or not source["complete_official_split"] or source["images"] != len(source["samples"])):
        raise ValueError("Complete qualified source split required")
    qualified = json.loads((args.source_root / "independent_audit.json").read_text())
    audited = [r for r in qualified["datasets"] if r["dataset"] == dataset and r["split"] == split]
    if len(audited) != 1 or audited[0]["manifest_sha256"] != file_sha256(source_path):
        raise ValueError("Source manifest differs from the independently audited delivery")
    root = args.output_root / dataset / split
    if root.exists():
        raise ValueError(f"Refusing to overwrite targets: {root}")
    root.mkdir(parents=True)
    rows = source["samples"][:args.limit] if args.limit else source["samples"]
    parameters = {"distance_fraction": args.distance_fraction, "max_inset_reference": args.max_inset_reference,
                  "reference_wh": tuple(args.reference_size), "crop_padding_reference": args.crop_padding_reference,
                  "min_area_fraction": args.min_area_fraction, "min_span_fraction": args.min_span_fraction,
                  "backoff_steps": args.backoff_steps}
    config = {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
              "classification": "ASSUMPTION", "target_source": "compact_word_cores_v2",
              "source_manifest": str(source_path), "source_manifest_sha256": file_sha256(source_path),
              "parameters": parameters, "device": "cpu", "python": platform.python_version(),
              "result_class": "smoke-test result" if args.limit else "approximate reproduction",
              "stroke_protection": "not applied: this variant intentionally supervises smaller word interiors",
              "code_sha256": {str(p): file_sha256(p) for p in [Path(__file__),
                  Path("segmentation/tighter_word_gt.py"), Path("segmentation/separated_word_gt.py")]},
              "ambiguity_id": "WORD-GT-002"}
    write_json(root / "resolved_config.json", config)
    records = []
    for i, old in enumerate(rows, 1):
        try:
            arrays = {}
            for name in ("owners", "masks", "validity", "separators"):
                path = args.source_root / old[name]
                if file_sha256(path) != old[name + "_sha256"]:
                    raise ValueError(f"Source target checksum mismatch: {path}")
                with Image.open(path) as im:
                    arrays[name] = np.asarray(im).copy()
            owners, sep, audit = compact_word_owners(arrays["owners"], arrays["validity"],
                                                    arrays["separators"], **parameters)
            sample_id = old["id"].split(":")[-1]
            record = {"id": old["id"], "image": old["image"], "image_sha256": old["image_sha256"],
                      "size_wh": old["size_wh"], "instances": old["instances"],
                      "ignored_instances": old["ignored_instances"], "compact_audit": audit,
                      "source_owners": str(args.source_root / old["owners"]),
                      "source_owners_sha256": old["owners_sha256"],
                      "source_record": str(args.source_root / dataset / split / "annotations" / f"{sample_id}.json")}
            record["source_record_sha256"] = file_sha256(Path(record["source_record"]))
            for name, pixels in (("owners", owners.astype(np.uint16)), ("masks", (owners > 0).astype(np.uint8)*255),
                                 ("validity", arrays["validity"]), ("separators", sep)):
                path = root / name / f"{sample_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
                with Image.open(path) as saved:
                    if not np.array_equal(np.asarray(saved), pixels):
                        raise ValueError("PNG roundtrip changed target")
                record[name] = path.relative_to(args.output_root).as_posix()
                record[name + "_sha256"] = file_sha256(path)
            write_json(root / "annotations" / f"{sample_id}.json", record)
            records.append(record)
            if i <= 3 or sample_id in {"147", "228", "520", "1111"}:
                preview(Path(old["image"]), Image.fromarray(owners), Image.fromarray((owners>0).astype(np.uint8)*255),
                        root / "previews" / f"{sample_id}.png", f"{dataset} {split} {sample_id}: compact word cores v2",
                        baseline=Image.fromarray(arrays["masks"]))
                # The source here is separated-v1, which derives from V4 for
                # Total-Text TRAIN; label this comparison without claiming it
                # is the unchanged V4 target.
                preview_path = root / "previews" / f"{sample_id}.png"
                with Image.open(preview_path) as panel:
                    panel = panel.copy()
                draw = ImageDraw.Draw(panel)
                draw.rectangle((420, 20, 839, 40), fill="white")
                draw.text((428, 23), "Previous separated GT v1", fill="black")
                panel.save(preview_path)
        except Exception as error:
            write_json(root / "failure.json", {"sample_id": old["id"], "completed_images": len(records),
                       "error_type": type(error).__name__, "error": str(error)})
            raise
        if i % 100 == 0 or i == len(rows):
            print(f"{dataset}/{split}: {i}/{len(rows)}", flush=True)
    foreground = sum(r["compact_audit"]["foreground_pixels"] for r in records)
    before = sum(r["compact_audit"]["source_foreground_pixels"] for r in records)
    report = {"schema_version": 1, "claim_label": "approximate reproduction", "classification": "ASSUMPTION",
              "target_source": "compact_word_cores_v2", "dataset": dataset, "split": split,
              "result_class": config["result_class"], "complete_official_split": args.limit is None,
              "images": len(records), "words": sum(len(r["instances"]) for r in records),
              "missing_words": 0, "interword_contact_pairs": 0,
              "fragmented_words": sum(len(r["compact_audit"]["fragmented_word_ids"]) for r in records),
              "changed_words": sum(r["compact_audit"]["changed_words"] for r in records),
              "foreground_pixels": foreground, "source_foreground_pixels": before,
              "foreground_reduction_fraction": 1-foreground/before,
              "native_survival_restorations": sum(len(r["compact_audit"]["native_survival_restored_ids"]) for r in records),
              "already_absent_native_words": sum(len(r["compact_audit"]["source_words_absent_at_training_resolution"]) for r in records),
              "new_words_lost_at_training_resolution": 0,
              "resolved_config_sha256": file_sha256(root / "resolved_config.json"), "samples": records}
    write_json(root / "manifest.json", report)
    return {k:v for k,v in report.items() if k != "samples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--distance-fraction", type=float, required=True)
    parser.add_argument("--max-inset-reference", type=float, required=True)
    parser.add_argument("--reference-size", nargs=2, type=int, required=True)
    parser.add_argument("--crop-padding-reference", type=float, required=True)
    parser.add_argument("--min-area-fraction", type=float, required=True)
    parser.add_argument("--min-span-fraction", type=float, required=True)
    parser.add_argument("--backoff-steps", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve(); args.output_root = args.output_root.resolve()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output_root == args.source_root or args.source_root in args.output_root.parents:
        parser.error("New targets must be isolated from the source delivery")
    summaries = []
    for dataset in args.dataset:
        sources = sorted((args.source_root / dataset).glob('*/manifest.json'))
        if not sources:
            raise ValueError(f"No complete dataset delivery: {dataset}")
        for source in sources:
            summaries.append(prepare(args, dataset, source.parent.name))
        if args.zip:
            path = args.output_root / f"{dataset}-compact-word-gt-v2-approximate.zip"
            if path.exists():
                raise ValueError(f"Refusing to overwrite ZIP: {path}")
            with ZipFile(path, "w", ZIP_DEFLATED) as z:
                for item in sorted((args.output_root / dataset).rglob('*')):
                    if item.is_file(): z.write(item, item.relative_to(args.output_root))
            print(f"ARCHIVE {path} SHA256 {file_sha256(path)}", flush=True)
    write_json(args.output_root / "qualification.json", {"claim_label": "approximate reproduction", "datasets": summaries})


if __name__ == "__main__":
    main()
