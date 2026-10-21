"""Independent Hungarian count audit of frozen AdaSpot Phase-0 outputs (CPU)."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def count(predictions, gt, exact):
    if not predictions or not gt: return 0
    a = np.asarray([p["bbox"] for p in predictions], dtype=float)
    b = np.asarray([p["bbox"] for p in gt], dtype=float)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
        - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=-1)
    union = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=-1)[:, None] + np.maximum(
        0, b[:, 2:] - b[:, :2]).prod(axis=-1)[None, :] - intersection
    edges = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0) >= .5
    if exact:
        edges &= np.asarray([[bool(p["text"]) and p["text"].casefold() == g["text"].casefold()
            for g in gt] for p in predictions], dtype=bool)
    left, right = linear_sum_assignment(-edges.astype(int))
    return int(edges[left, right].sum())


def audit(root):
    result = json.loads((root / "results.json").read_text())
    assert result["status"] == "SUCCESS"
    protocol = result["protocol"]
    for path, h in protocol["source_hashes"].items(): assert digest(Path(path)) == h, path
    frozen = json.loads((root / "frozen_inputs.json").read_text())
    audited_images, audited_proposals = 0, 0
    for arm, values in result["arms"].items():
        frozen_arm, mode = arm.rsplit("/", 1)
        expected = {s["sample_id"]: s for s in frozen[frozen_arm]}
        assert len(values["per_image"]) == protocol["resolved_config"]["expected_images"]
        totals = {"gt_words": 0, "proposals": 0, "geometry_iou_ge_50": 0, "exact_casefold_iou_ge_50": 0}
        readings = json.loads((root / (mode + "_readings.json")).read_text())
        for sample in values["per_image"]:
            original = expected.pop(sample["sample_id"])
            assert original["ground_truth"] == sample["ground_truth"]
            predictions, gt = sample["predictions"], sample["ground_truth"]
            assert len(predictions) == len(original["predictions"])
            for p, raw in zip(predictions, original["predictions"]):
                assert p["bbox"] == raw["bbox"] and p["crop_rgb_sha256"] == raw["crop_rgb_sha256"]
                assert p["text"] == readings[p["crop_rgb_sha256"]]["raw_text"]
            counts = {"gt_words": len(gt), "proposals": len(predictions),
                "geometry_iou_ge_50": count(predictions, gt, False),
                "exact_casefold_iou_ge_50": count(predictions, gt, True)}
            for k, v in counts.items():
                assert sample["counts"][k] == v, (arm, sample["sample_id"], k)
                totals[k] += v
            audited_images += 1; audited_proposals += len(predictions)
        assert not expected
        for k, v in totals.items(): assert values["aggregate"][k] == v, (arm, k)
        assert totals["gt_words"] == protocol["resolved_config"]["expected_words"]
        assert values["aggregate"]["hmean"] == 2 * totals["exact_casefold_iou_ge_50"] / (totals["gt_words"] + totals["proposals"])
    executions = list(result["execution"].values())
    assert all([b["digests"] for b in executions[0]["batch_times"]] ==
        [b["digests"] for b in e["batch_times"]] for e in executions[1:])
    weak = result["arms"]["pred/axis/nar1"]["per_image"]
    strong = result["arms"]["pred/axis/ar1"]["per_image"]
    flat = [p for sample in weak for p in sample["predictions"]]
    ranking = sorted(range(len(flat)), key=lambda i: (
        flat[i]["minimum_token_probability"] if flat[i]["text"] and flat[i]["minimum_token_probability"] is not None else -1., i))
    def mixed_count(selected):
        total, offset = 0, 0
        for a, b in zip(weak, strong):
            p = [q if offset + i in selected else p for i, (p, q) in enumerate(zip(a["predictions"], b["predictions"]))]
            total += count(p, a["ground_truth"], True); offset += len(p)
        return total
    config = protocol["resolved_config"]
    for row in result["routing_proxy"]["rows"]:
        k = int(np.floor(row["requested_fraction"] * len(flat)))
        assert row["escalated"] == k
        assert row["selected_global_indices"] == sorted(ranking[:k])
        assert row["confidence"]["exact_casefold_iou_ge_50"] == mixed_count(set(ranking[:k]))
        expected_random = []
        for repeat in range(config["random_repeats"]):
            rng = np.random.default_rng(config["seed"] + repeat)
            expected_random.append(mixed_count(set(rng.permutation(len(flat))[:k])))
        assert row["random_exact_counts"] == expected_random
    audit = {"status": "PASS", "method": "independent vectorized bbox IoU + SciPy Hungarian maximum-count matching",
        "per_image_arm_cases": audited_images, "proposal_cases": audited_proposals,
        "routing_random_cases": len(result["routing_proxy"]["rows"]) * config["random_repeats"],
        "result_sha256": digest(root / "results.json"), "frozen_inputs_sha256": digest(root / "frozen_inputs.json"),
        "auditor_sha256": digest(Path(__file__)), "source_hashes_verified": len(protocol["source_hashes"])}
    path = root / "independent_audit.json"
    text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    audit(p.parse_args().run_root)
