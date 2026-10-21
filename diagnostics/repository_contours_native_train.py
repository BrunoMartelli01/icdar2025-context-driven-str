"""CPU native128 geometry from verified TRAIN cache; optional PAPER-only OCR.

No segmentation, mask resizing, grouping, model training or TEST loading.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from diagnostics.repository_contours_train import digest, read_sample, unpack_mask, save_shard, read_shard
from localize.native_repository_contours import localize_native_repository_contours
from localize.repository_contours import RepositoryContourParams
from localize.practical_remap_protocol import train_task22_word_boxes, proposal_geometry
from scripts.diagnose_train_gtbox_vs_v1_parseq import _texts, _wordness, _iou, _inter
from scripts.export_train_v1_parseq_inputs import (
    _sha, _contained, _write_atomic, EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TRAIN_MANIFEST_SHA256, EXPECTED_SPLIT,
)
from segmentation.data import load_segmentation_manifest


def write_once(path, value):
    """Never replace a different completed result."""
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise RuntimeError(f"existing report differs: {path}")
        return
    _write_atomic(path, value)


def geometry(a):
    if _sha(a.source_report) != a.source_sha256:
        raise RuntimeError("previous TRAIN report hash mismatch")
    if _sha(a.train_manifest) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("frozen TRAIN manifest mismatch")
    source = json.loads(a.source_report.read_text())
    if source["protocol"]["scope"] != {"split": "TRAIN", "images": 229, "gt_words": 849}:
        raise RuntimeError("expected TRAIN229/849 source")
    source_root = a.source_report.parent
    source_contract = source["contract_sha256"]
    if digest(source["protocol"]) != source_contract:
        raise RuntimeError("source protocol hash mismatch")
    config = json.loads(a.config.read_text())
    if (config.get("variant") != "repository-contours-native128-v1"
            or config.get("reproduction_mode") != "approximate"
            or config.get("parameters") != {"min_area": 150, "pad": 6, "nmax": 10}
            or config.get("resize") != "none"):
        raise RuntimeError("require declared native128 protocol, no parameter changes")
    params = RepositoryContourParams(**config["parameters"])
    manifest = load_segmentation_manifest(a.train_manifest)
    if manifest.split != EXPECTED_SPLIT or manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("requires TRAIN229")
    for relative, reference in source["protocol"]["dataset_files"].items():
        if _sha(_contained(a.data_root, relative)) != reference["sha256"]:
            raise RuntimeError(f"dataset hash mismatch: {relative}")
    samples, geometry_rows, crop_counts, word_rows = [], [], Counter(), []
    for number, sample in enumerate(manifest.samples, 1):
        relative = f"masks/{sample.sample_id}.json"
        path = source_root / relative
        if _sha(path) != source["shards"][relative]:
            raise RuntimeError(f"mask file hash mismatch: {sample.sample_id}")
        record = read_sample(path, source_contract, sample)
        metadata = record["segmentation_metadata"]
        if (metadata["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256
                or metadata["threshold"] != .5 or metadata["threshold_comparison"] != "greater_than_or_equal"):
            raise RuntimeError("native mask provenance mismatch")
        mask = unpack_mask(record)
        if metadata["native_mask_sha256"] != record["array_sha256"]:
            raise RuntimeError("mask not equal to frozen adapter output")
        with Image.open(a.data_root / sample.image_path) as image:
            shape = (image.height, image.width)
        localized = localize_native_repository_contours(mask, shape, params)
        # No GT is loaded until inference-safe proposal generation is complete.
        boxes = train_task22_word_boxes(a.data_root / sample.mask_path, a.data_root / sample.validity_annotation_path)
        texts = _texts(a.data_root / sample.validity_annotation_path)
        if len(boxes) != len(texts):
            raise RuntimeError("GT box/text count mismatch")
        words = [{"bbox": list(b.bbox), "text": t} for b, t in zip(boxes, texts)]
        candidates = localized["retained"]
        for crop in candidates:
            category, best, best_iou = _wordness(crop["padded_bbox_original"], words)
            crop.update(category=category, best_gt_index=best, best_gt_iou=best_iou)
            crop_counts[category] += 1
        geometry_rows.append(([c["padded_bbox_original"] for c in candidates], [w["bbox"] for w in words],
            localized["native_localization"]["candidates_before_cap"], 0, 0))
        for index, word in enumerate(words):
            best = max(candidates, key=lambda c: _iou(c["padded_bbox_original"], word["bbox"]), default=None)
            overlap = best is not None and _inter(best["padded_bbox_original"], word["bbox"]) > 0
            word_rows.append({"sample_id": sample.sample_id, "gt_index": index, **word,
                "matched_crop_rank": best["rank"] if overlap else None,
                "best_iou": _iou(best["padded_bbox_original"], word["bbox"]) if overlap else 0})
        samples.append({"sample_id": sample.sample_id, "image_path": str(sample.image_path),
            "image_sha256": sample.image_sha256, "native_mask_sha256": record["array_sha256"],
            "localization": localized})
        print(f"[native128 CPU] {number}/229 sample={sample.sample_id} crops={len(candidates)}", flush=True)
    if len(word_rows) != 849:
        raise RuntimeError("expected 849 TRAIN GT words")
    protocol = {"classification": "PRACTICAL_APPROXIMATION", "reproduction_mode": "approximate",
        "resolved_config": config, "source_report_sha256": a.source_sha256,
        "source_contract": source_contract, "scope": source["protocol"]["scope"],
        "segmentation_rerun": False, "mask_resize": "none", "gt_usage": "post-hoc only",
        "code_sha256": {str(p): _sha(p) for p in (Path(__file__), Path("localize/native_repository_contours.py"),
            Path("localize/repository_contours.py"), Path("diagnostics/repository_contours_train.py"),
            Path("scripts/diagnose_train_gtbox_vs_v1_parseq.py"), Path("localize/practical_remap_protocol.py"))}}
    report = {"protocol": protocol, "protocol_sha256": digest(protocol),
        "summary": {"geometry": proposal_geometry(geometry_rows).as_dict(),
            "wordness": {k: crop_counts[k] for k in ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")}},
        "previous_original_coordinate_results": source["summary"],
        "samples": samples, "per_word": word_rows, "ocr": None, "table5_accuracy": None}
    output = a.output_dir / "geometry.json"
    write_once(output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"JSON {output} SHA256 {_sha(output)}", flush=True)


def ocr(a):
    from recognition.parseq import (ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_ARCHIVE_SOURCE,
        PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256, verify_parseq_runtime_source)
    if _sha(a.source_report) != a.source_sha256:
        raise RuntimeError("source report mismatch")
    for name in ("parseq_source", "parseq_checkpoint", "parseq_archive"):
        if getattr(a, name) is None:
            raise ValueError(f"missing {name}")
    if _sha(a.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256 or _sha(a.parseq_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("PARSeq artifacts mismatch")
    verify_parseq_runtime_source(a.parseq_source, source_type=PARSEQ_ARCHIVE_SOURCE, source_archive=a.parseq_archive)
    source = json.loads(a.source_report.read_text())
    report = json.loads((a.output_dir / "geometry.json").read_text())
    if report["protocol"]["source_report_sha256"] != a.source_sha256 or digest(report["protocol"]) != report["protocol_sha256"]:
        raise RuntimeError("geometry provenance mismatch")
    previous = {(row["sample_id"], tuple(row["bbox"])): row["text"] for row in source["per_crop"]}
    contract = digest({"geometry_sha256": _sha(a.output_dir / "geometry.json"),
        "parseq_checkpoint": PARSEQ_CHECKPOINT_SHA256, "archive": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "device": a.device, "script_sha256": _sha(Path(__file__))})
    recognizer, predictions = None, {}
    reused = generated = 0
    try:
        for number, sample in enumerate(report["samples"], 1):
            path = a.output_dir / "ocr" / f"{sample['sample_id']}.json"
            if path.exists():
                result = read_shard(path, contract)
            else:
                image_path = _contained(a.data_root, sample["image_path"])
                if _sha(image_path) != sample["image_sha256"]:
                    raise RuntimeError("RGB hash mismatch")
                with Image.open(image_path) as image:
                    rgb = np.array(image.convert("RGB"))
                result = {"sample_id": sample["sample_id"], "predictions": [], "reused": 0, "generated": 0}
                for crop in sample["localization"]["retained"]:
                    box = crop["padded_bbox_original"]
                    key = (sample["sample_id"], tuple(box))
                    if key in previous:
                        text = previous[key]
                        result["reused"] += 1
                    else:
                        if recognizer is None:
                            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                                runtime_source=a.parseq_source, checkpoint=a.parseq_checkpoint, device=a.device,
                                runtime_source_type=PARSEQ_ARCHIVE_SOURCE, runtime_source_archive=a.parseq_archive))
                        x1, y1, x2, y2 = box
                        text = recognizer.recognize(rgb[y1:y2, x1:x2], input_role="T3_localized_crop").raw_text
                        result["generated"] += 1
                    result["predictions"].append({"rank": crop["rank"], "text": text})
                save_shard(path, contract, result)
            for pred in result["predictions"]:
                predictions[(sample["sample_id"], pred["rank"])] = pred["text"]
            reused += result["reused"]
            generated += result["generated"]
            print(f"[native128 OCR] {number}/229 sample={sample['sample_id']}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    counts, rows = Counter(), []
    for word in report["per_word"]:
        rank = word["matched_crop_rank"]
        text = predictions[(word["sample_id"], rank)] if rank is not None else None
        exact = text is not None and text.casefold() == word["text"].casefold()
        fuzzy = ratio(text.casefold(), word["text"].casefold()) / 100 if text is not None else None
        counts.update(available=int(text is not None), exact=int(exact), fuzzy=int(fuzzy is not None and fuzzy >= .8))
        rows.append({**word, "prediction": text, "exact_casefold": exact, "rapidfuzz_casefold": fuzzy})
    result = {"geometry_sha256": _sha(a.output_dir / "geometry.json"), "contract_sha256": contract,
        "classification": "PRACTICAL_APPROXIMATION", "scope": report["protocol"]["scope"],
        "summary": {**counts, "exact_rate": counts["exact"] / 849, "fuzzy_rate": counts["fuzzy"] / 849},
        "new_ocr": generated, "reused_ocr": reused, "per_word": rows, "table5_accuracy": None,
        "ocr_shards": {str(p.relative_to(a.output_dir)): _sha(p) for p in sorted((a.output_dir / "ocr").glob("*.json"))}}
    target = a.output_dir / "ocr-results.json"
    write_once(target, result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"JSON {target} SHA256 {_sha(target)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("geometry", "ocr"), required=True)
    for name in ("source-report", "data-root", "output-dir", "config", "train-manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    for name in ("parseq-source", "parseq-checkpoint", "parseq-archive"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    geometry(a) if a.stage == "geometry" else ocr(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
