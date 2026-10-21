"""Independent held-out selection, native/RGB matching and artifact audit."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from localize.adaspot_gap_preserve import decode_gap
from scripts.audit_adaspot_phase0 import count, digest
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash


def boxes(raster, ids):
    result = []
    for identity in ids:
        yy, xx = np.where(raster == identity)
        if len(xx): result.append({"bbox": [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]})
    return result


def check_native(labels, maps, metrics):
    ids = [i for i in np.unique(labels[labels > 0]) if np.any((labels == i) & maps["validity"])]
    pred, gt = boxes(labels, ids), boxes(maps["owners"], maps["expected_owner_ids"])
    assert count(pred, gt, False) == metrics["matched_gt_iou_ge_50"]
    assert len(pred) == metrics["raw_proposals"]
    assert len(maps["expected_owner_ids"]) == metrics["gt_words"]


def audit(root):
    if not __debug__: raise RuntimeError("Assertions required; do not run Python -O")
    torch.set_num_threads(4)
    result = json.loads((root/"results.json").read_text())
    assert result["status"] == "SUCCESS" and result["result_class"] == "approximate reproduction"
    protocol, arms = result["protocol"], result["arms"]
    c = protocol["resolved_config"]
    assert protocol["mode"] == "approximate" and not protocol["selection_uses_evaluation"]
    fit, calibration, evaluation = [set(protocol[k]) for k in ("fit_ids", "calibration_ids", "evaluation_ids")]
    assert tuple(map(len, (fit, calibration, evaluation))) == (1030, 100, 125)
    assert not (fit & calibration or fit & evaluation or calibration & evaluation)
    for path, h in protocol["source_hashes"].items(): assert digest(Path(path)) == h, path
    source = Path(protocol["source_run"])
    pp = json.loads((source/"control/run_contract.json").read_text())["resolved_config"]["postprocessing"]
    normalized = Path(protocol.get("normalized_run", root))
    calibration_rows = json.loads((normalized/"calibration_manifest.json").read_text())
    assert {r["sample_id"] for r in calibration_rows} == calibration
    calibration_maps = {}
    for row in calibration_rows:
        assert digest(Path(row["maps_path"])) == row["maps_sha256"]
        with np.load(row["maps_path"], allow_pickle=False) as saved:
            calibration_maps[row["sample_id"]] = {k: saved[k] for k in saved.files}
    normalized_cases = 0
    if "normalized_run" not in protocol:
        manifest = json.loads((Path(protocol["context_run"])/"training_manifest.json").read_text())
        indices, offset = [], 0
        for row in manifest["per_image"]:
            n = sum(row["stratum_counts"])
            if row["sample_id"] in fit: indices.append(torch.arange(offset, offset+n))
            else: assert row["sample_id"] in calibration
            offset += n
        indices = torch.cat(indices)
        assert torch.equal(indices, torch.load(root/"fit_pixel_indices.pt", weights_only=True))
        assert digest(Path(manifest["cache_path"])) == manifest["cache_sha256"]
        cache = torch.load(manifest["cache_path"], map_location="cpu", weights_only=True)
        pixels = cache["patches"][indices]
        var, mean = torch.var_mean(pixels, dim=0, unbiased=False)
        expected = torch.load(root/"normalization.pt", weights_only=True)
        torch.testing.assert_close(mean, expected["mean"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(var.sqrt().clamp_min(c["normalization_std_floor"]), expected["scale"], rtol=1e-5, atol=1e-5)
        normalized_cases = len(pixels)
        del cache, pixels
    readings = json.loads((root/"ocr_readings.json").read_text())
    for h in readings:
        with Image.open(root/"crops"/(h+".png")) as image: rgb = np.asarray(image.convert("RGB"))
        assert crop_array_hash(rgb) == h
    native_cases = ocr_cases = calibration_cases = 0
    exact_by_arm, selected_epochs, orders = {}, {}, []
    for name, arm in arms.items():
        if name != "baseline":
            assert arm["completed_epochs"] == c["epochs"]
            assert [r["epoch"] for r in arm["history"]] == list(range(1, c["epochs"]+1))
            assert all(r["cuda_steps"] > 0 for r in arm["history"])
            orders.append([r["sample_order_sha256"] for r in arm["history"]])
            candidates = arm["candidates"]
            assert len(candidates) == 1+len(c["checkpoint_epochs"])*len(c["thresholds"])*len(c["decoders"])
            assert {r["epoch"] for r in candidates} == {0, *c["checkpoint_epochs"]}
            for candidate in candidates:
                assert {r["sample_id"] for r in candidate["per_image"]} == calibration
                totals = Counter()
                for row in candidate["per_image"]: totals.update(row["counts"])
                assert all(candidate["aggregate"][k] == v for k, v in totals.items())
                assert candidate["aggregate"]["word_f1"] == 2*totals["matched_gt_iou_ge_50"]/(totals["gt_words"]+totals["raw_proposals"])
            chosen = max(candidates, key=lambda r: (r["aggregate"]["word_f1"], -r["aggregate"]["gt_words_touched_by_fused_components"],
                -r["epoch"], r["threshold"] if r["threshold"] is not None else 1., r["decoder"] == "cut"))
            assert chosen == arm["selection"]
            assert digest(root/name/"best.pt") == arm["checkpoint_sha256"]
            checkpoint = torch.load(root/name/"best.pt", map_location="cpu", weights_only=True)
            assert checkpoint["selection"] == chosen
            if name == "decoder":
                assert all(k.startswith(("decoder.", "gap.")) for k in checkpoint["branch"])
                smoke = json.loads((root/name/"first_cuda_step.json").read_text())
                assert smoke["decoder_gradient_l1"] > 0 and smoke["head_weight_update_linf"] > 0 and smoke["teacher_max_error"] == 0
                assert all(r["samples"] == len(fit) and r["cuda_steps"]+r["skipped_all_invalid_batches"] == (len(fit)+c["image_batch_size"]-1)//c["image_batch_size"] for r in arm["history"])
            else:
                smoke = json.loads((root/name/"first_cuda_step.json").read_text())
                assert smoke["output_weight_update_linf"] > 0 and smoke["point_native_error"] <= 1e-4
            assert smoke["status"] == "PASS" and smoke["device"].startswith("cuda")
            selected_epochs[name] = chosen["epoch"]
            for row in chosen["per_image"]:
                sid = row["sample_id"]; maps = calibration_maps[sid]
                if chosen["threshold"] is None: gap = None
                else:
                    with np.load(root/name/"calibration_predictions"/str(chosen["epoch"])/(sid.replace(":", "__")+".npz")) as saved: gap = saved["gap_probability"]
                labels = decode_gap(maps["probabilities"], gap, maps["canvas"], pp, chosen["threshold"], chosen["decoder"])
                check_native(labels, maps, row["counts"]); calibration_cases += 1
        else:
            chosen = {"threshold": None, "decoder": "cut"}
        assert {r["sample_id"] for r in arm["native"]["per_image"]} == evaluation
        native_total = Counter()
        for row in arm["native"]["per_image"]:
            sid = row["sample_id"]
            with np.load(root/name/"evaluation_maps"/(sid.replace(":", "__")+".npz")) as saved: labels, gap = saved["labels"], saved["gap_probability"]
            with np.load(source/"control/validation_exports"/sid.replace(":", "__")/"maps.npz") as saved: maps = {k: saved[k] for k in saved.files}
            expected = decode_gap(maps["probabilities"], gap, maps["canvas"], pp, chosen["threshold"], chosen["decoder"])
            assert np.array_equal(labels, expected)
            if chosen["decoder"] == "preserve": assert np.array_equal(labels > 0, maps["canvas"] & (maps["probabilities"][0] >= pp["text_threshold"]))
            check_native(labels, maps, row["counts"]); native_cases += 1; native_total.update(row["counts"])
        assert all(arm["native"]["aggregate"][k] == v for k, v in native_total.items())
        totals = Counter()
        assert {r["sample_id"] for r in arm["ocr"]["per_image"]} == evaluation
        for row in arm["ocr"]["per_image"]:
            pred, gt = row["predictions"], row["ground_truth"]
            values = {"gt_words": len(gt), "proposals": len(pred), "exact_casefold_iou_ge_50": count(pred, gt, True), "geometry_iou_ge_50": count(pred, gt, False)}
            assert all(row["counts"][k] == v for k, v in values.items())
            assert all(p["text"] == readings[p["crop_rgb_sha256"]]["raw_text"] for p in pred)
            totals.update(values); ocr_cases += 1
        assert all(arm["ocr"]["aggregate"][k] == v for k, v in totals.items())
        assert totals["gt_words"] == 983
        assert arm["ocr"]["aggregate"]["hmean"] == 2*totals["exact_casefold_iou_ge_50"]/(totals["gt_words"]+totals["proposals"])
        exact_by_arm[name] = totals["exact_casefold_iou_ge_50"]
        assert arm["changes_vs_baseline"]["net"] == totals["exact_casefold_iou_ge_50"]-524
    assert exact_by_arm["baseline"] == 524
    if "normalized_run" not in protocol: assert orders[0] == orders[1]
    report = {"status": "PASS", "method": "independent SciPy Hungarian native/RGB matching; complete calibration population/selection, fit-only normalization, source hashes and artifacts",
        "native_image_cases": native_cases, "ocr_image_cases": ocr_cases, "selected_calibration_cases": calibration_cases,
        "normalized_fit_pixel_cases": normalized_cases, "selected_epochs": selected_epochs, "exact_by_arm": exact_by_arm,
        "result_sha256": digest(root/"results.json"), "auditor_sha256": digest(Path(__file__))}
    text = json.dumps(report, indent=2, sort_keys=True)+"\n"
    path = root/"independent_audit.json"
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-root", type=Path, required=True)
    audit(p.parse_args().run_root)
