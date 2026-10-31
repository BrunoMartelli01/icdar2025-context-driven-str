"""Synthetic artifact boundary test for the independent scientific auditor."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from scripts.audit_icdar2015_word_f1 import audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_auditor_replays_masks_crops_csv_and_rates_then_rejects_corruption(tmp_path):
    data = tmp_path / "data"
    image_path = data / "datasets/icdar2015/test/images/1.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10), (10, 20, 30)).save(image_path)
    annotations = image_path.parent.parent / "annotations/1.txt"
    annotations.parent.mkdir()
    annotations.write_text("0,0,9,0,9,9,0,9,A!\n10,0,19,0,19,9,10,9,###\n")
    manifest_path = data / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"samples": [{"id": "icdar2015:1",
        "image": str(image_path.relative_to(data)), "image_sha256": sha256(image_path.read_bytes()).hexdigest()}]}))
    output = tmp_path / "result"
    output.mkdir()
    prediction = np.zeros((256, 256), dtype=bool)
    prediction[64:192, :128] = True
    probability = np.where(prediction, 0.9, 0.1).astype(np.float32)
    mask_path = output / "mask.npz"
    np.savez_compressed(mask_path, probability=probability, prediction=prediction)
    write_json(output / "samples/icdar2015_1.json", {
        "mask_path": "mask.npz", "mask_sha256": sha256(mask_path.read_bytes()).hexdigest()})
    with Image.open(image_path) as source:
        crop = np.array(source.convert("RGB"))[:, :10]
    crop_hash = sha256(np.asarray(crop.shape, dtype="<i8").tobytes()+np.ascontiguousarray(crop).tobytes()).hexdigest()
    ocr_path = output / "ocr" / (crop_hash + ".json")
    write_json(ocr_path, {"raw_text": "a"})
    exact = {"tp": 0, "fp": 1, "fn": 1, "precision": 0., "recall": 0., "f1": 0.}
    correct = {"tp": 1, "fp": 0, "fn": 0, "precision": 1., "recall": 1., "f1": 1.}
    report = {"protocol": {"evaluation_config": {"expected_images": 1, "expected_care_words": 1,
        "probability_threshold": 0.5, "matching_iou_threshold": 0.5}, "paths": {"test_manifest": str(manifest_path)}},
        "ignored_gt": 1, "aggregate": {"predictions": 1, "geometry": correct,
            "casefold_exact": exact, "alphanumeric_casefold": correct}, "per_image": [{
                "sample_id": "icdar2015:1", "annotation_sha256": sha256(annotations.read_bytes()).hexdigest(),
                "ground_truth": [{"text": "A!", "bbox": [0, 0, 10, 10]}],
                "predictions": [{"text": "a", "bbox": [0, 0, 10, 10],
                    "bbox_native": [0, 64, 128, 192], "component_area": 128*128,
                    "crop_rgb_sha256": crop_hash, "ocr_file_sha256": sha256(ocr_path.read_bytes()).hexdigest()}],
                "counts": {"geometry_tp": 1, "casefold_exact_tp": 0, "alphanumeric_casefold_tp": 1}}]}
    path = output / "results.json"
    write_json(path, report)
    result = audit(path, data)
    assert result["status"] == "PASS" and result["metrics"]["alphanumeric_casefold"]["f1"] == 1
    report["aggregate"]["alphanumeric_casefold"] = exact
    write_json(path, report)
    with pytest.raises(ValueError, match="aggregate rates"):
        audit(path, data)
