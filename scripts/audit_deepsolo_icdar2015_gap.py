"""Independent lexical/count, calibration optimum and new checkpoint/export audit."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return "".join(character for character in text.casefold() if character.isalnum())


def independent_counts(predictions, gt):
    remaining = Counter(normalized(g["text"]) for g in gt if normalized(g["text"]))
    tp = n = ignored = 0
    for row in predictions:
        if row["ignored"]:
            ignored += 1
            continue
        n += 1
        word = normalized(row["text"])
        if row.get("invalid_envelope") is None and word and remaining[word] > 0:
            tp += 1
            remaining[word] -= 1
    return dict(tp=tp, fp=n-tp, fn=len(gt)-tp, detected_words=n, gt_words=len(gt),
                all_outputs=len(predictions), ignored_outputs=ignored)


def total(rows):
    counts = Counter()
    for row in rows:
        counts.update(row)
    return dict(counts, f1=2*counts["tp"]/(counts["detected_words"]+counts["gt_words"]))


def square_mask(image, resolution):
    width, height = image.size
    side = max(width, height)
    square = Image.new("L", (side, side), 0)
    square.paste(image.convert("L"), ((side-width)//2, (side-height)//2))
    return np.asarray(square.resize((resolution, resolution), Image.Resampling.NEAREST)) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    c = read(args.config)
    root = Path(c["study_root"])
    frozen = read(root / "frozen_calibration_policies.json")
    assert frozen["config_sha256"] == sha(args.config)
    assert frozen["observations_sha256"] == sha(root / "calibration_observations.json")
    assert frozen["grid_sha256"] == sha(root / "calibration_grid.json")
    observations = read(root / "calibration_observations.json")
    grids = read(root / "calibration_grid.json")
    checks = Counter()
    selected = {}
    for variant, grid in grids.items():
        candidates = []
        for row in grid:
            policy = row["policy"]
            counts = []
            for predictions, gt in observations[variant].values():
                accepted = [p for p in predictions if p["score"] >= policy["score_threshold"]
                            and p["recognition_confidence"] >= policy["recognition_threshold"]]
                counts.append(independent_counts(accepted, gt))
            metrics = total(counts)
            for key, value in metrics.items():
                assert value == row["metrics"][key], (variant, key)
            key = (Fraction(2*metrics["tp"], metrics["detected_words"]+metrics["gt_words"]),
                   metrics["tp"], -policy["score_threshold"], -policy["recognition_threshold"])
            candidates.append((key, policy))
            checks["independent_calibration_grid_points"] += 1
        best = max(candidates, key=lambda row: row[0])
        assert best[1] == frozen["selected"][variant]["policy"]
        selected[variant] = best[0][:2]
    assert max(selected, key=lambda name: (*selected[name], -list(selected).index(name))) == frozen["selected_model"]
    result = read(root / "results.json")
    protocol = read(root / "predeclared_protocol.json")
    samples = {s["sample_id"]: s for s in protocol["population"]["test"]}
    cache = {}
    for name, report in result["datasets"]["icdar2015"].items():
        path = root / (name + "_per_image.json")
        assert sha(path) == report["per_image_sha256"]
        rows = read(path)
        assert set(rows) == set(samples)
        replay = []
        for sid, row in rows.items():
            sample = samples[sid]
            if sid not in cache:
                with Path(sample["annotation_path"]).open(encoding="utf-8-sig") as stream:
                    texts = [",".join(parts[8:]) for parts in csv.reader(stream) if parts and ",".join(parts[8:]) != "###"]
                validity_path = Path(c["data_root"]) / "datasets/icdar2015/test/validity" / (sid.split(":")[-1] + ".png")
                with Image.open(validity_path) as im:
                    valid = square_mask(im, c["ignore_resolution"])
                cache[sid] = texts, valid
            texts, valid = cache[sid]
            assert texts == [g["text"] for g in row["ground_truth"]]
            for p in row["selected_predictions"]:
                if p["bbox"] is None:
                    ignored = False
                else:
                    support = np.zeros((sample["image_wh"][1], sample["image_wh"][0]), np.uint8)
                    vertices = np.rint(np.clip(np.asarray(p["boundary_original"]), -2**29, 2**29)).astype(np.int32)
                    cv2.fillPoly(support, [vertices], 255)
                    footprint = square_mask(Image.fromarray(support), c["ignore_resolution"])
                    ignored = bool(footprint.any() and not np.any(footprint & valid))
                assert ignored == p["ignored"]
                checks["independent_polygon_ignore_decisions"] += 1
            count = independent_counts(row["selected_predictions"], row["ground_truth"])
            assert count == row["counts"]
            replay.append(count)
            checks["independent_test_image_count_replays"] += 1
        metrics = total(replay)
        for key, value in metrics.items():
            assert value == report["metrics"][key], (name, key)
        assert report["metrics"]["deepsolo_images"] == report["metrics"]["total_images"] == len(samples)
        checks["complete_test_metric_replays"] += 1
    for variant in ("tt_ic15resize1440", "ic15_official1440"):
        for split in ("validation", "test"):
            directory = root / variant / split
            exported = read(directory / "protocol.json")
            load = exported["load_validation"]
            assert load["strict_compatibility_check"] and not any(load[k] for k in ("missing_keys", "unexpected_keys", "incorrect_shapes"))
            assert exported["resize"] == [1440, 4000]
            expected = protocol["population"][split]
            for sample in expected:
                path = directory / (sample["sample_id"].replace(":", "__") + ".json")
                record = read(path)
                content = {k: v for k, v in record.items() if k != "payload_sha256"}
                assert hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() == record["payload_sha256"]
                assert record["contract"]["image_sha256"] == sample["image_sha256"]
                assert record["contract"]["plan_sha256"] == sha(c["plans"][variant + "_" + split])
                assert record["contract"]["profile_sha256"] == sha(c["profiles"][variant])
                width, height = sample["image_wh"]
                scale = min(1440/min(width, height), 4000/max(width, height))
                shape = [3, ((int(height*scale+0.5)+31)//32)*32, ((int(width*scale+0.5)+31)//32)*32]
                assert record["input_chw"] == shape
                assert record["precision"] == "float32" and record["batch_size"] == 1
                checks["authenticated_new_inference_shards"] += 1
    output = dict(status="PASS", mode="approximate", claim_label="approximate reproduction",
                  checks=dict(checks), results_sha256=sha(root / "results.json"),
                  frozen_choices_sha256=sha(root / "frozen_calibration_policies.json"),
                  audit_code_sha256=sha(__file__),
                  scope="Independent normalized lexical matching, CSV care words, centered polygon ignore, complete grids, frozen choices, shape/precision/strict-load/export bindings. No independent neural inference or official-score equivalence.")
    with (root / "independent_audit.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
