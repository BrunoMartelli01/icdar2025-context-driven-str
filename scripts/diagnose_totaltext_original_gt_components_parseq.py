"""Oracle diagnostic: PARSeq on every full-resolution Total-Text GT contour.

This is not paper Algorithm 1 or a deployable word recognizer. GT polygons are
used only to associate already-extracted contours with words for evaluation.
No area filter, top-10 cap, padding, or morphology is applied.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw

from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig,
    ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent


EXPECTED_TEST_MANIFEST_SHA256 = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing JSON differs: {path}")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _array_sha(array: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _owner_map(size_wh: tuple[int, int], care: list) -> np.ndarray:
    """Rasterize oracle polygons; overlap belongs to no unique word."""
    width, height = size_wh
    owner = np.zeros((height, width), dtype=np.int32)
    for index, word in enumerate(care, 1):
        polygon = Image.new("1", size_wh, 0)
        ImageDraw.Draw(polygon).polygon(word.polygon_xy, fill=1)
        inside = np.asarray(polygon, dtype=bool)
        conflict = inside & (owner != 0)
        owner[inside & (owner == 0)] = index
        owner[conflict] = -1
    return owner


def _components(mask: np.ndarray, owner: np.ndarray) -> list[dict]:
    """Find all full-resolution 8-connected components via contour labels."""
    if mask.shape != owner.shape or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("Expected same-size binary uint8 mask and owner map")
    # RETR_EXTERNAL omits foreground islands inside holes (123 components on
    # this TEST split). RETR_LIST plus label dedup keeps every actual CC while
    # ignoring duplicate hole contours of the same label.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    seen_labels = set()
    for index, contour in enumerate(contours):
        px, py = map(int, contour[0, 0])
        label = int(labels[py, px])
        if label <= 0:
            raise ValueError("Contour seed is not foreground")
        if label in seen_labels:
            continue
        seen_labels.add(label)
        x, y, width, height, pixels = map(int, stats[label])
        support = labels[y:y + height, x:x + width] == label
        touched = np.unique(owner[y:y + height, x:x + width][support])
        word_ids = [int(value) for value in touched if value > 0]
        # An overlap pixel is deliberately ambiguous, even when only one
        # non-overlap word label remains in the component.
        assigned = word_ids[0] if len(word_ids) == 1 and -1 not in touched else None
        result.append({"contour_index": index, "bbox": [x, y, x + width, y + height],
                       "foreground_pixels": pixels, "word_id": assigned,
                       "touched_word_ids": word_ids, "polygon_overlap": bool(-1 in touched)})
    if len(seen_labels) != label_count - 1:
        raise ValueError("FindContours did not recover every 8-connected component")
    return result


def _word_order(components: list[dict], orientation: str | None) -> list[int]:
    """Oracle reading order: top-to-bottom only for explicit vertical GT."""
    def key(index: int) -> tuple[float, float, int]:
        x1, y1, x2, y2 = components[index]["bbox"]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        return (y, x, index) if orientation == "v" else (x, y, index)
    return sorted(range(len(components)), key=key)


def _normalized(text: str, mode: str) -> str:
    if mode == "casefold":
        return text.casefold()
    if mode == "punctuation":
        return "".join(char for char in text.casefold()
                       if unicodedata.category(char)[0] not in "PS")
    raise ValueError(mode)


def _inputs(args):
    if file_sha256(args.manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Expected qualified official Total-Text TEST-300 manifest")
    manifest = load_segmentation_manifest(args.manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected Total-Text official TEST-300")
    return manifest


def _source(args, sample):
    paths = [args.data_root / sample.image_path, args.data_root / sample.mask_path,
             args.data_root / sample.validity_annotation_path]
    hashes = [sample.image_sha256, sample.mask_sha256, sample.validity_annotation_sha256]
    for path, expected in zip(paths, hashes):
        if file_sha256(path) != expected:
            raise ValueError(f"Source SHA256 mismatch: {path}")
    with Image.open(paths[0]) as opened:
        rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
    with Image.open(paths[1]) as opened:
        gt = np.asarray(opened.convert("L")) >= 128
    with Image.open(paths[2]) as opened:
        valid = np.asarray(opened.convert("L")) > 0
    if gt.shape != rgb.shape[:2] or valid.shape != gt.shape:
        raise ValueError(f"Source dimensions differ: {sample.sample_id}")
    return rgb, np.ascontiguousarray((gt & valid).astype(np.uint8))


def _annotation(args, sample, size_wh):
    number = sample.sample_id.split(":", 1)[1]
    path = args.data_root / "datasets/total_text/test/annotations" / f"{number}.txt"
    instances, _ = clip_instances_to_pixel_extent(parse_totaltext_annotation(path), size_wh)
    return path, [word for word in instances if not word.ignore]


def prepare(args) -> int:
    manifest = _inputs(args)
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "result_class": "GT-oracle original-pixel-mask component OCR diagnostic; not Table 5",
                "manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
                "source": "official full-resolution Total-Text pixel GT AND qualified Don't Care validity",
                "localization": "cv2.RETR_LIST/CHAIN_APPROX_SIMPLE deduped by 8-connected label; no area filter, top10, or padding",
                "crop": "original RGB inside each tight component bounding rectangle",
                "association": "DIAGNOSTIC_ONLY: unique care GT polygon owner; multi-owner/overlap components unassigned",
                "reconstruction": "concatenate independent PARSeq outputs in x-center order, y-center for GT orientation v",
                "accuracy_denominator": "all 2204 care GT words, including words with no assigned components"}
    _atomic_json(args.output_dir / "protocol.json", protocol)
    protocol_sha = file_sha256(args.output_dir / "protocol.json")
    summary = {"images": 0, "gt_words": 0, "raw_contours": 0, "assigned_contours": 0,
               "ambiguous_or_unassigned_contours": 0, "words_with_no_assigned_contours": 0,
               "words_with_one_assigned_contour": 0, "words_with_multiple_assigned_contours": 0}
    for count, sample in enumerate(manifest.samples, 1):
        rgb, binary = _source(args, sample)
        annotation_path, care = _annotation(args, sample, (rgb.shape[1], rgb.shape[0]))
        owner = _owner_map((rgb.shape[1], rgb.shape[0]), care)
        components = _components(binary, owner)
        rows = []
        for index, word in enumerate(care, 1):
            subset = [item for item in components if item["word_id"] == index]
            ordered = _word_order(subset, word.orientation)
            rows.append({"word_id": index, "text": word.transcription,
                         "orientation": word.orientation,
                         "contour_indices": [subset[i]["contour_index"] for i in ordered]})
        shard = {"schema_version": 1, "protocol_sha256": protocol_sha,
                 "sample_id": sample.sample_id, "annotation_sha256": file_sha256(annotation_path),
                 "masked_source_array_sha256": _array_sha(binary), "components": components,
                 "words": rows}
        _atomic_json(args.output_dir / "samples" / f"{sample.sample_id.replace(':', '__')}.json", shard)
        summary["images"] += 1
        summary["gt_words"] += len(rows)
        summary["raw_contours"] += len(components)
        assigned = sum(item["word_id"] is not None for item in components)
        summary["assigned_contours"] += assigned
        summary["ambiguous_or_unassigned_contours"] += len(components) - assigned
        summary["words_with_no_assigned_contours"] += sum(len(row["contour_indices"]) == 0 for row in rows)
        summary["words_with_one_assigned_contour"] += sum(len(row["contour_indices"]) == 1 for row in rows)
        summary["words_with_multiple_assigned_contours"] += sum(len(row["contour_indices"]) > 1 for row in rows)
        if count % 10 == 0 or count == 300:
            print(f"[original GT contours prepare] {count}/300", flush=True)
    if summary["images"] != 300 or summary["gt_words"] != 2204:
        raise ValueError("TEST-300/2204 denominator mismatch")
    report = {"schema_version": 1, "protocol": protocol, "summary": summary}
    _atomic_json(args.output_dir / "prepared.json", report)
    print(json.dumps(summary, indent=2), flush=True)
    print("PREPARED", args.output_dir / "prepared.json", file_sha256(args.output_dir / "prepared.json"), flush=True)
    return 0


def recognize(args) -> int:
    import torch
    manifest = _inputs(args)
    prepared = json.loads((args.output_dir / "prepared.json").read_text(encoding="utf-8"))
    if prepared["protocol"] != json.loads((args.output_dir / "protocol.json").read_text(encoding="utf-8")):
        raise ValueError("Prepared protocol mismatch")
    if file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint SHA256 mismatch")
    if file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq source archive SHA256 mismatch")
    finished_path = args.output_dir / "results.json"
    if finished_path.exists():
        finished = json.loads(finished_path.read_text(encoding="utf-8"))
        if (finished.get("prepared_sha256") != file_sha256(args.output_dir / "prepared.json")
            or finished.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
            or finished.get("summary", {}).get("gt_words") != 2204):
            raise ValueError("Existing result provenance differs")
        for sample in manifest.samples:
            if not (args.output_dir / "ocr" / f"{sample.sample_id.replace(':', '__')}.json").is_file():
                raise ValueError("Completed result has missing OCR shards")
        print(json.dumps(finished["summary"], indent=2), flush=True)
        print("[reuse] JSON", finished_path, "SHA256", file_sha256(finished_path), flush=True)
        return 0
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("Run PARSeq on PAPER CUDA:0; CPU diagnostic preparation is separate")
    hardware = {"device": args.device, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "batch_size_max": 32}
    recognizer = None
    totals = {"gt_words": 0, "available_reconstructions": 0, "exact_case_sensitive": 0,
              "exact_casefold": 0, "exact_casefold_without_punctuation_or_symbols": 0,
              "ocr_components": 0, "component_outputs_one_character": 0,
              "component_outputs_zero_characters": 0, "component_outputs_multiple_characters": 0}
    word_rows = []
    try:
        for count, sample in enumerate(manifest.samples, 1):
            key = sample.sample_id.replace(":", "__")
            shard = json.loads((args.output_dir / "samples" / f"{key}.json").read_text(encoding="utf-8"))
            if shard["protocol_sha256"] != file_sha256(args.output_dir / "protocol.json"):
                raise ValueError("Sample protocol mismatch")
            rgb, binary = _source(args, sample)
            annotation_path, care = _annotation(args, sample, (rgb.shape[1], rgb.shape[0]))
            owner = _owner_map((rgb.shape[1], rgb.shape[0]), care)
            if (shard["sample_id"] != sample.sample_id or shard["annotation_sha256"] != file_sha256(annotation_path)
                or shard["masked_source_array_sha256"] != _array_sha(binary)
                or shard["components"] != _components(binary, owner)):
                raise ValueError(f"Prepared contours changed: {sample.sample_id}")
            ocr_path = args.output_dir / "ocr" / f"{key}.json"
            if ocr_path.exists():
                ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
                if (ocr["sample_id"] != sample.sample_id or ocr["protocol_sha256"] != shard["protocol_sha256"]
                    or len(ocr["predictions"]) != len(shard["components"])):
                    raise ValueError(f"OCR cache mismatch: {ocr_path}")
                for item, component in zip(ocr["predictions"], shard["components"]):
                    x1, y1, x2, y2 = component["bbox"]
                    if (item["contour_index"] != component["contour_index"]
                        or item["crop_sha256"] != _array_sha(rgb[y1:y2, x1:x2])):
                        raise ValueError(f"OCR crop cache mismatch: {ocr_path}")
            else:
                if shard["components"] and recognizer is None:
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
                        device=args.device, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=args.parseq_source_archive.resolve()))
                    hardware["parseq_model_dtype"] = str(next(recognizer.model.parameters()).dtype)
                predictions = []
                for start in range(0, len(shard["components"]), 32):
                    batch = shard["components"][start:start + 32]
                    crops = [np.ascontiguousarray(rgb[y1:y2, x1:x2]) for item in batch
                             for x1, y1, x2, y2 in [item["bbox"]]]
                    outputs = recognizer.recognize_candidate_batch(crops)
                    if len(outputs) != len(batch):
                        raise ValueError("PARSeq returned fewer outputs than input components")
                    predictions.extend({"contour_index": item["contour_index"], "bbox": item["bbox"],
                                        "crop_sha256": _array_sha(crop), "text": output.raw_text}
                                       for item, crop, output in zip(batch, crops, outputs))
                ocr = {"schema_version": 1, "protocol_sha256": shard["protocol_sha256"],
                       "sample_id": sample.sample_id, "predictions": predictions}
                _atomic_json(ocr_path, ocr)
            texts = {row["contour_index"]: row["text"] for row in ocr["predictions"]}
            totals["ocr_components"] += len(texts)
            totals["component_outputs_one_character"] += sum(len(text) == 1 for text in texts.values())
            totals["component_outputs_zero_characters"] += sum(len(text) == 0 for text in texts.values())
            totals["component_outputs_multiple_characters"] += sum(len(text) > 1 for text in texts.values())
            for word in shard["words"]:
                indices = word["contour_indices"]
                reconstruction = "".join(texts[index] for index in indices)
                truth = word["text"]
                totals["gt_words"] += 1
                totals["available_reconstructions"] += bool(indices)
                totals["exact_case_sensitive"] += bool(indices) and reconstruction == truth
                totals["exact_casefold"] += bool(indices) and reconstruction.casefold() == truth.casefold()
                cleaned_prediction, cleaned_truth = _normalized(reconstruction, "punctuation"), _normalized(truth, "punctuation")
                totals["exact_casefold_without_punctuation_or_symbols"] += bool(indices and cleaned_prediction and cleaned_truth) and cleaned_prediction == cleaned_truth
                word_rows.append({"sample_id": sample.sample_id, "word_id": word["word_id"],
                                  "gt": truth, "reconstruction": reconstruction, "contour_indices": indices,
                                  "component_texts": [texts[index] for index in indices],
                                  "exact_case_sensitive": bool(indices) and reconstruction == truth,
                                  "exact_casefold": bool(indices) and reconstruction.casefold() == truth.casefold()})
            if count % 10 == 0 or count == 300:
                print(f"[original GT component PARSeq] {count}/300, components={totals['ocr_components']}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    if totals["gt_words"] != 2204 or totals["ocr_components"] != prepared["summary"]["raw_contours"]:
        raise ValueError("Finished denominators do not match preparation")
    hardware["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    hardware["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "result_class": "TEST GT-oracle per-component OCR; not Table 5",
              "prepared_sha256": file_sha256(args.output_dir / "prepared.json"),
              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
              "hardware": hardware, "summary": totals, "words": word_rows}
    _atomic_json(args.output_dir / "results.json", report)
    print(json.dumps(totals, indent=2), flush=True)
    print("JSON", args.output_dir / "results.json", "SHA256", file_sha256(args.output_dir / "results.json"), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "recognize"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path)
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "recognize" and not all((args.parseq_source, args.parseq_source_archive, args.parseq_checkpoint)):
        parser.error("recognize requires PARSeq source, archive, and checkpoint")
    return prepare(args) if args.action == "prepare" else recognize(args)


if __name__ == "__main__":
    raise SystemExit(main())
