"""Independent fixed-population Hungarian audit; no production scorer imports."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_adaspot_phase0 import count, digest
from scripts.diagnose_adaspot_phase0 import write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash


def audit(root):
    result = json.loads((root / "results.json").read_text())
    protocol = json.loads((root / "protocol_final.json").read_text())
    assert result["status"] == "SUCCESS" and protocol["mode"] == "approximate"
    assert not protocol["test_used_for_selection"]
    for path, expected in protocol["source_hashes"].items():
        assert digest(Path(path)) == expected, path
    source = Path(protocol["source_run"])
    old = json.loads((source / "test_inputs.json").read_text())["full_selected"]
    old_readings = json.loads((source / "test_readings.json").read_text())
    frozen = json.loads((root / "frozen_inputs.json").read_text())
    readings = json.loads((root / "rgb_proxy_readings.json").read_text())
    receipts = json.loads((root / "feature_receipts.json").read_text())
    assert len(receipts) == 300 and [r["sample_id"] for r in receipts] == [s["sample_id"] for s in old]
    counters, audited_proposals = Counter(), 0
    for receipt, sample in zip(receipts, old):
        assert digest(Path(receipt["feature_path"])) == receipt["feature_sha256"]
        assert digest(Path(receipt["image_path"])) == receipt["image_sha256"]
        ids = [p["instance_id"] for p in sample["predictions"]]
        assert receipt["instance_ids"] == ids and receipt["probability_max_abs_error"] <= 1e-6
        with np.load(receipt["feature_path"], allow_pickle=False) as values:
            assert values["instance_ids"].tolist() == ids
            for key, shape in (("light", (8, 32)), ("deep", (16, 64))):
                assert values[key].shape == (len(ids), 64, *shape)
                assert values[key].dtype == np.float32 and np.isfinite(values[key]).all()
        assert [r["instance_id"] for r in receipt["geometries"]] == ids
        for geometry in receipt["geometries"]:
            counters[geometry["fallback_reason"] or "cubic_centerline"] += 1
        audited_proposals += len(ids)
    assert dict(counters) == result["geometry_counts"] and audited_proposals == 2624
    assert result["source_state_before"] == result["source_state_after"]
    assert result["gradient"]["nonzero_feature_gradients"] > 0
    assert result["gradient"]["gradient_device"].startswith("cuda")
    assert result["gradient"]["sampling_backward_device"] == "cpu"
    cases = 0
    for name, arm in result["arms"].items():
        assert len(arm["ocr"]["per_image"]) == 300 and len(frozen[name]) == 300
        totals = Counter()
        subset_totals = Counter()
        subset = result["hash_disjoint_test_subset"]
        excluded = set(subset["excluded_exact_duplicate_test_ids"])
        assert excluded == {"total_text:578", "total_text:656"} and subset["images"] == 298
        reduced = []
        for sample, original, raw in zip(arm["ocr"]["per_image"], old, frozen[name]):
            assert sample["sample_id"] == original["sample_id"] == raw["sample_id"]
            assert sample["ground_truth"] == original["ground_truth"] == raw["ground_truth"]
            assert sample["ignored_instance_ids"] == original["ignored_instance_ids"]
            predictions = sample["predictions"]
            assert len(predictions) == len(original["predictions"]) == len(raw["predictions"])
            for prediction, prior, unscored in zip(predictions, original["predictions"], raw["predictions"]):
                for key in ("bbox", "bbox_native", "bbox_original", "instance_id", "foreground_area"):
                    assert prediction[key] == prior[key] == unscored[key]
                h = prediction["crop_rgb_sha256"]
                assert h == unscored["crop_rgb_sha256"]
                assert prediction["text"] == readings[h]["raw_text"]
                if name == "original_axis":
                    assert h == prior["crop_rgb_sha256"]
                    assert readings[h] == old_readings[h]
            counts = {"gt_words": len(sample["ground_truth"]), "proposals": len(predictions),
                "geometry_iou_ge_50": count(predictions, sample["ground_truth"], False),
                "exact_casefold_iou_ge_50": count(predictions, sample["ground_truth"], True)}
            assert all(sample["counts"][key] == value for key, value in counts.items())
            totals.update(counts)
            cases += 1
            if sample["sample_id"] not in excluded:
                subset_totals.update(counts)
                reduced.append(sample)
        for counts, scored in ((totals, arm["ocr"]), (subset_totals, subset["arms"][name])):
            assert all(scored["aggregate"][key] == value for key, value in counts.items())
            a = scored["aggregate"]
            assert a["precision"] == a["exact_casefold_iou_ge_50"] / a["proposals"]
            assert a["recall"] == a["exact_casefold_iou_ge_50"] / a["gt_words"]
            assert a["hmean"] == 2 * a["exact_casefold_iou_ge_50"] / (a["proposals"] + a["gt_words"])
        assert subset["arms"][name]["per_image"] == reduced
        assert totals["gt_words"] == 2204 and totals["proposals"] == 2624 and totals["geometry_iou_ge_50"] == 1612
        assert subset_totals["gt_words"] == 2187 and subset_totals["proposals"] == 2607
    for h in readings:
        with Image.open(root / "crops" / (h + ".png")) as image:
            assert crop_array_hash(np.asarray(image.convert("RGB"))) == h
    def verify_changes(left, right, changes):
        recovered, damaged = [], []
        for a, b in zip(left["per_image"], right["per_image"]):
            x, y = (set(r["matched_gt_indices"]["exact_casefold_iou_ge_50"]) for r in (a, b))
            recovered.extend([a["sample_id"], i] for i in sorted(y - x))
            damaged.extend([a["sample_id"], i] for i in sorted(x - y))
        assert recovered == changes["recovered_ids"] and damaged == changes["damaged_ids"]
        assert len(recovered) == changes["recovered"] and len(damaged) == changes["damaged"]
        assert changes["net"] == len(recovered) - len(damaged)
    for arm in result["arms"].values():
        verify_changes(result["arms"]["original_axis"]["ocr"], arm["ocr"], arm["changes_vs_original"])
    verify_changes(result["arms"]["bilinear_axis"]["ocr"], result["arms"]["mask_align"]["ocr"], result["changes_mask_vs_bilinear_axis"])
    assert result["arms"]["original_axis"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"] == 1311
    return {"status": "PASS", "result_sha256": digest(root / "results.json"),
        "source_hashes_checked": len(protocol["source_hashes"]), "primary_image_arm_cases": cases,
        "secondary_image_arm_cases": 298 * len(result["arms"]), "actual_feature_instances": audited_proposals,
        "unique_rgb_crops_checked": len(readings), "original_baseline_replayed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    value = audit(args.output_root.resolve())
    write_json(args.output_root / "independent_audit.json", value)
    print(json.dumps(value, indent=2))
