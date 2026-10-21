"""Independent population, checkpoint, Hungarian metric and routing audit."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

from scripts.audit_adaspot_phase0 import count, digest
from scripts.diagnose_adaspot_phase0 import write_json
from segmentation.data import load_segmentation_manifest


def independent_correctness(sample):
    predictions, gt = sample["predictions"], sample["ground_truth"]
    labels = np.zeros(len(predictions), np.int32)
    if not predictions or not gt:
        return labels.tolist()
    a, b = np.asarray([p["bbox"] for p in predictions]), np.asarray([g["bbox"] for g in gt])
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None] - inter
    edges = np.divide(inter, union, out=np.zeros_like(inter, float), where=union > 0) >= .5
    edges &= np.asarray([[bool(p["light"]["text"]) and p["light"]["text"].casefold() == g["text"].casefold() for g in gt] for p in predictions])
    left, right = linear_sum_assignment(-edges.astype(int))
    labels[left[edges[left, right]]] = 1
    return labels.tolist()


def independent_routes(samples, choice):
    rows, routes = [], []
    for sample in samples:
        predictions, decisions = [], []
        for p in sample["predictions"]:
            probability = float(expit(p["gate_logit"] / choice["temperature"]))
            if probability >= choice["keep_threshold"]:
                route, reading = "keep", p["light"]
            elif probability < choice["reject_threshold"]:
                route, reading = "reject", p["light"]
            else:
                route, reading = "escalate", p["deep"]
            emitted = route != "reject" and bool(reading["text"]) and (route != "escalate" or reading["minimum_probability"] >= choice["deep_threshold"])
            if emitted:
                predictions.append({**p, "text": reading["text"]})
            decisions.append({"instance_id": p["instance_id"], "route": route, "emitted": emitted,
                "text": reading["text"] if emitted else "", "gate_probability": probability})
        rows.append({**sample, "predictions": predictions})
        routes.append(decisions)
    return rows, routes


def totals(rows):
    total = Counter()
    for sample in rows:
        total.update({"gt_words": len(sample["ground_truth"]), "proposals": len(sample["predictions"]),
            "geometry_iou_ge_50": count(sample["predictions"], sample["ground_truth"], False),
            "exact_casefold_iou_ge_50": count(sample["predictions"], sample["ground_truth"], True)})
    total["hmean"] = 2 * total["exact_casefold_iou_ge_50"] / (total["proposals"] + total["gt_words"]) if total["proposals"] + total["gt_words"] else 0.
    return dict(total)


def audit(root):
    result = json.loads((root / "results.json").read_text())
    p = json.loads((root / "protocol_final.json").read_text())
    c = p["resolved_config"]
    assert result["status"] == "SUCCESS" and p["mode"] == "approximate" and not p["test_used_for_selection"]
    for path, expected in p["source_hashes"].items():
        assert digest(Path(path)) == expected, path
    fit, temperature, threshold = [p[key] for key in ("fit_ids", "temperature_ids", "threshold_ids")]
    assert not (set(fit) & set(temperature) or set(fit) & set(threshold) or set(temperature) & set(threshold))
    train = load_segmentation_manifest(Path(p["train_manifest"]), require_checksums=True)
    test = load_segmentation_manifest(Path(p["test_manifest"]), require_checksums=True)
    if p["stage"] == "full":
        assert (len(fit), len(temperature), len(threshold)) == (1155, 50, 50)
        assert set(fit + temperature + threshold) == {s.sample_id for s in train.samples}
    completed = json.loads((root / "training_complete.json").read_text())
    assert digest(Path(completed["checkpoint"])) == completed["checkpoint_sha256"]
    assert completed["completed_epochs"] == p["selected_epoch"]
    assert completed["global_cuda_steps"] == completed["completed_epochs"] * len(fit)
    all_seen = []
    for epoch, path in enumerate(completed["history_paths"], 1):
        row = json.loads(Path(path).read_text())
        assert row["epoch"] == epoch and len(row["sample_ids"]) == len(fit) and set(row["sample_ids"]) == set(fit)
        assert len(set(row["sample_ids"])) == len(fit)
        assert [r["sample_id"] for r in row["roi_audit"]] == row["sample_ids"]
        for r in row["roi_audit"]:
            seed = int(sha256(f"{c['seed']}:{epoch}:{r['sample_id']}".encode()).hexdigest()[:16], 16)
            expected = sorted(np.random.default_rng(seed).choice(r["proposal_count"], min(r["proposal_count"], c["max_training_rois"]), replace=False).tolist())
            assert expected == r["selected_indices"]
            assert r["ocr_supervised"] <= r["matching"]["matched_rois"] <= r["rois"] <= c["max_training_rois"]
        all_seen.extend(row["sample_ids"])
    update = json.loads((root / "first_joint_update.json").read_text())
    assert update["all_parameters_cuda"] and update["sampling_backward_device"] == "cpu"
    assert all(np.isfinite(v) and v > 0 for v in update["gradient_l1"].values())
    assert all(np.isfinite(v) and v > 0 for v in update["parameter_max_abs_change"].values())
    initialization = json.loads((root / "initialization.json").read_text())
    assert initialization["source_strict_load"] and initialization["shared_encoder_trainable"]
    for name in ("light", "deep"):
        assert initialization[name]["strict_load"] and len(initialization[name]["copied_keys"]) == 26
    assert any(key.startswith("decoder.layers.3.") for key in initialization["deep"]["new_keys"])
    caches = {stage: json.loads((root / (stage + "_cache.json")).read_text()) for stage in ("temperature", "threshold", "test")}
    assert [r["sample_id"] for r in caches["temperature"]] == temperature
    assert [r["sample_id"] for r in caches["threshold"]] == threshold
    if p["stage"] == "full":
        assert [r["sample_id"] for r in caches["test"]] == [s.sample_id for s in test.samples]
        assert sum(len(r["ground_truth"]) for r in caches["test"]) == 2204
    for rows in caches.values():
        for row in rows:
            assert digest(Path(row["map_path"])) == row["map_sha256"]
            assert independent_correctness(row) == row["light_correctness_labels"]
            with np.load(row["map_path"], allow_pickle=False) as maps:
                assert maps["probabilities"].shape == (2, 256, 256) and maps["gap"].shape == (256, 256)
                assert np.isfinite(maps["probabilities"]).all() and np.isfinite(maps["gap"]).all()
                assert all(np.any(maps["labels"] == v["instance_id"]) for v in row["predictions"])
    calibrated = json.loads((root / "calibration.json").read_text())
    fit_temp = calibrated["temperature_fit"]
    z = np.asarray([v["gate_logit"] for s in caches["temperature"] for v in s["predictions"]])
    y = np.asarray([v for s in caches["temperature"] for v in s["light_correctness_labels"]])
    def nll(temperature_value):
        a = z / temperature_value
        return float(np.mean(np.logaddexp(0, a) - y * a))
    assert abs(nll(1.) - fit_temp["nll_before"]) < 1e-10
    assert abs(nll(fit_temp["temperature"]) - fit_temp["nll_after"]) < 1e-10
    assert fit_temp["nll_after"] <= fit_temp["nll_before"]
    for candidate in calibrated["candidates"]:
        rows, routes = independent_routes(caches["threshold"], candidate["choice"])
        measured = totals(rows)
        assert measured["exact_casefold_iou_ge_50"] == candidate["exact_words"] and measured["hmean"] == candidate["hmean"]
        assert sum(d["route"] == "escalate" for r in routes for d in r) == candidate["deep_calls"]
    best = max(calibrated["candidates"], key=lambda row: (row["hmean"], -row["deep_calls"]))
    assert best == calibrated["selection"] and result["calibration_choice"] == best["choice"]
    cases = 0
    for name, scored in result["arms"].items():
        if name == "adaptive":
            expected, routes = independent_routes(caches["test"], best["choice"])
            assert routes == result["routes"]
            assert sum(d["route"] == "escalate" for r in routes for d in r) == result["deep_calls"]
        else:
            expected = [{**s, "predictions": [{**v, "text": v[name]["text"]} for v in s["predictions"]]} for s in caches["test"]]
        for a, b in zip(scored["per_image"], expected):
            assert a["predictions"] == b["predictions"] and a["ground_truth"] == b["ground_truth"]
            for key, value in totals([b]).items():
                if key != "hmean":
                    assert a["counts"][key] == value
            cases += 1
        for key, value in totals(expected).items():
            assert scored["aggregate"][key] == value
        reduced = [s for s in expected if s["sample_id"] not in result["hash_disjoint_test_subset"]["excluded_ids"]]
        for key, value in totals(reduced).items():
            assert result["hash_disjoint_test_subset"]["arms"][name]["aggregate"][key] == value
    selective = json.loads((root / "selective_execution.json").read_text())
    for policy in selective["policies"]:
        assert sum(policy["deep_projection_batches"]) == policy["deep_calls"]
        assert policy["raw_strings_replayed"]
    return {"status": "PASS", "result_sha256": digest(root / "results.json"), "result_class": p["result_class"],
        "stage": p["stage"], "training_image_epoch_cases": len(all_seen), "test_image_arm_cases": cases,
        "temperature_calibration_images": len(temperature), "threshold_calibration_images": len(threshold),
        "threshold_candidates_checked": len(calibrated["candidates"]), "source_hashes_checked": len(p["source_hashes"]),
        "all_joint_branches_updated_cuda": True, "selective_deep_execution_verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.output_root.resolve())
    write_json(args.output_root / "independent_audit.json", receipt)
    print(json.dumps(receipt, indent=2))
