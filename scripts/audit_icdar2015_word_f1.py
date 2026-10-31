"""Independent SciPy/CSV replay of the frozen approximate ICDAR2015 word F1."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import yaml


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def independent_tp(predictions, words, threshold, text_mode):
    """Independent maximum matching; do not import the reported metric code."""
    edges = []
    def normalized(text):
        return "".join(c for c in text.casefold() if c.isalnum())
    for i, pred in enumerate(predictions):
        a = pred["bbox"]
        for j, word in enumerate(words):
            b = word["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
            if not union or intersection / union < threshold:
                continue
            if text_mode == "casefold_exact" and (not pred["text"] or pred["text"].casefold() != word["text"].casefold()):
                continue
            if text_mode == "alphanumeric_casefold" and (not normalized(pred["text"]) or normalized(pred["text"]) != normalized(word["text"])):
                continue
            edges.append((i, j))
    if not predictions or not words or not edges:
        return 0
    graph = csr_matrix((np.ones(len(edges)), ([i for i, _ in edges], [j for _, j in edges])),
                       shape=(len(predictions), len(words)))
    return int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type="column") >= 0))


def audit(results, data_root):
    report = json.loads(results.read_text())
    protocol = report["protocol"]
    config = protocol["evaluation_config"]
    manifest = yaml.safe_load(Path(protocol["paths"]["test_manifest"]).read_text())
    by_id = {sample["id"]: sample for sample in manifest["samples"]}
    expected_ids = set(by_id)
    if set(row["sample_id"] for row in report["per_image"]) != expected_ids or len(report["per_image"]) != config["expected_images"]:
        raise ValueError("Audit image population differs")
    totals = {name: 0 for name in ("geometry", "casefold_exact", "alphanumeric_casefold")}
    n_pred = n_gt = ignored = crop_count = 0
    for row in report["per_image"]:
        sample = by_id[row["sample_id"]]
        rgb_path = data_root / sample["image"]
        if digest(rgb_path) != sample["image_sha256"]:
            raise ValueError("Audit RGB checksum mismatch")
        with Image.open(rgb_path) as image:
            rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        key = row["sample_id"].replace(":", "_")
        shard = json.loads((results.parent / "samples" / (key + ".json")).read_text())
        mask_path = results.parent / shard["mask_path"]
        if digest(mask_path) != shard["mask_sha256"]:
            raise ValueError("Audit native mask hash differs")
        with np.load(mask_path) as arrays:
            probability, prediction = arrays["probability"], arrays["prediction"]
        side = max(width, height)
        left, top = (side-width)//2, (side-height)//2
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(255, (left, top, left+width, top+height))
        valid_canvas = np.asarray(canvas.resize((256, 256), Image.Resampling.NEAREST)) > 0
        if not np.array_equal(prediction, (probability >= config["probability_threshold"]) & valid_canvas):
            raise ValueError("Audit threshold/canvas reconstruction differs")
        labels, number = label(prediction, structure=np.ones((3, 3)))
        components = []
        for component, slices in enumerate(find_objects(labels), 1):
            yy, xx = slices
            components.append((int(np.count_nonzero(labels[slices] == component)),
                               (xx.start, yy.start, xx.stop, yy.stop)))
        saved_components = [(p["component_area"], tuple(p["bbox_native"])) for p in row["predictions"]]
        if number != len(saved_components) or sorted(components) != sorted(saved_components):
            raise ValueError("Audit independent 8-connected component geometry differs")
        annotation = rgb_path.parent.parent / "annotations" / (row["sample_id"].split(":", 1)[1] + ".txt")
        if digest(annotation) != row["annotation_sha256"]:
            raise ValueError("Audit annotation checksum mismatch")
        words = []
        for source in csv.reader(annotation.read_text(encoding="utf-8-sig").splitlines(), skipinitialspace=True):
            if not source or all(not value.strip() for value in source):
                continue
            points = [int(value) for value in source[:8]]
            text = ",".join(source[8:]).strip()
            if text == "###":
                ignored += 1
                continue
            xs, ys = points[::2], points[1::2]
            words.append({"text": text, "bbox": [min(xs), min(ys), min(width, max(xs)+1), min(height, max(ys)+1)]})
        if [{"text": w["text"], "bbox": w["bbox"]} for w in row["ground_truth"]] != words:
            raise ValueError("Audit GT geometry/transcription differs")
        for pred in row["predictions"]:
            a, b, c, d = pred["bbox_native"]
            inverted = [max(0, a*side//256-left), max(0, b*side//256-top),
                        min(width, (c*side+255)//256-left), min(height, (d*side+255)//256-top)]
            if pred["bbox"] != inverted:
                raise ValueError("Audit inverse letterbox coordinates differ")
            x1, y1, x2, y2 = pred["bbox"]
            crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
            expected_hash = sha256(np.asarray(crop.shape, dtype="<i8").tobytes()+crop.tobytes()).hexdigest()
            if expected_hash != pred["crop_rgb_sha256"]:
                raise ValueError("Audit RGB crop hash differs")
            ocr_path = results.parent / "ocr" / (expected_hash + ".json")
            if digest(ocr_path) != pred["ocr_file_sha256"] or json.loads(ocr_path.read_text())["raw_text"] != pred["text"]:
                raise ValueError("Audit OCR bytes/transcription differ")
            crop_count += 1
        for mode in totals:
            tp = independent_tp(row["predictions"], words, config["matching_iou_threshold"], mode)
            if tp != row["counts"][mode + "_tp"]:
                raise ValueError(f"Independent SciPy matching differs: {row['sample_id']} / {mode}")
            totals[mode] += tp
        n_pred += len(row["predictions"])
        n_gt += len(words)
    if (n_pred != report["aggregate"]["predictions"] or n_gt != config["expected_care_words"]
            or ignored != report["ignored_gt"]):
        raise ValueError("Audit aggregate populations differ")
    metrics = {}
    for mode, tp in totals.items():
        expected = {"tp": tp, "fp": n_pred-tp, "fn": n_gt-tp,
                    "precision": tp/n_pred if n_pred else None, "recall": tp/n_gt if n_gt else None,
                    "f1": 2*tp/(n_pred+n_gt) if n_pred+n_gt else None}
        if expected != report["aggregate"][mode]:
            raise ValueError("Audit aggregate rates differ")
        metrics[mode] = expected
    return {"status": "PASS", "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "source_results_sha256": digest(results), "auditor_sha256": digest(Path(__file__)),
        "images": len(report["per_image"]), "gt_words": n_gt, "predictions": n_pred,
        "crop_hashes_verified": crop_count, "ignored_gt": ignored, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing audit; use a new output path")
    result = audit(args.results, args.data_root)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
