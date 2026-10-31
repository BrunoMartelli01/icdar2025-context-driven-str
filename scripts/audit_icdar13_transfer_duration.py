"""Independent CSV word-count replay of a completed duration comparison."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def norm(text):
    return "".join(c for c in text.casefold() if c.isalnum())


def f1(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None


def audit(root):
    plan = json.loads((root / "resolved_configuration.json").read_text())
    result = json.loads((root / "results.json").read_text())
    frozen = json.loads((root / "selection_frozen.json").read_text())
    if result["selection_sha256"] != sha(root / "selection_frozen.json") or frozen["test_used_for_selection"]:
        raise ValueError("Selection sealing failure")
    records = {r["sample_id"]: r for r in plan["inherited"]["records"]}
    ids = plan["inherited"]["partitions"]["test"]
    if len(ids) != 233 or len(set(ids)) != 233: raise ValueError("Unexpected test IDs")
    checks = 0; samples_by_arm = {}; curves = {}
    for arm, data in frozen["choices"].items():
        for key, maximum in (("best30", 30), ("best60", 60)):
            eligible = [x for x in data["validation"] if x["epoch"] <= maximum]
            best = max(eligible, key=lambda x: (x["counts"]["f1"], -x["epoch"]))
            if best != data[key] or sha(best["checkpoint"]) != best["checkpoint_sha256"]:
                raise ValueError("Validation selection differs")
        epochs = json.loads((root / arm / "epochs.json").read_text())
        if [x["epoch"] for x in epochs] != list(range(1, 61)):
            raise ValueError("Incomplete epoch history")
        if any(x["ic_images"] != 183 or x["tt_images"] != (183 if arm.endswith("mixed") else 0) or x["steps"] != 23 for x in epochs):
            raise ValueError("Incorrect per-epoch sample count")
        samples_by_arm[arm] = {}
        for epoch, expected in result["test"][arm].items():
            counts = []
            for sid in ids:
                record = records[sid]
                if sha(record["annotation_path"]) != record["annotation_sha256"]: raise ValueError("Annotation changed")
                with Path(record["annotation_path"]).open(encoding="utf-8-sig", newline="") as f:
                    gt = [r[4] for r in csv.reader(f, skipinitialspace=True) if r]
                sample = json.loads((root / arm / "test" / f"epoch_{int(epoch):03d}" / "samples" / (sid.replace(":", "__") + ".json")).read_text())
                if sample["image_sha256"] != record["image_sha256"] or sha(sample["maps_path"]) != sample["maps_sha256"]:
                    raise ValueError("Image or probability-map identity mismatch")
                raw = sample["raw_rows"]; emitted = [r for r in raw if r["sequence_confidence_native"] >= frozen["policy"]["minimum_transcript_probability"] and norm(r["raw_text"])]
                if len(emitted) != len(sample["rows"]) or any(a["raw_text"] != b["raw_text"] or a["bbox"] != b["bbox"] for a, b in zip(emitted, sample["rows"])):
                    raise ValueError("Emission replay mismatch")
                gt_words = Counter(norm(s) for s in gt if norm(s)); pred_words = Counter(norm(r["raw_text"]) for r in emitted)
                tp = sum((gt_words & pred_words).values())
                c = dict(tp=tp, fp=len(emitted)-tp, fn=len(gt)-tp)
                if any(sample["counts"][k] != v for k, v in c.items()): raise ValueError("Independent word count mismatch")
                counts.append(c); checks += 1
            totals = {k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn")}
            if any(totals[k] != expected[k] for k in totals) or abs(f1(counts) - expected["f1"]) > 1e-14:
                raise ValueError("Aggregate mismatch")
            samples_by_arm[arm][epoch] = counts
        curves[arm] = [{"epoch": r["epoch"], "validation_f1": r["counts"]["f1"]} for r in data["validation"]]
    rng = random.Random(20261029); comparisons = {}
    for arm, data in samples_by_arm.items():
        values = []
        for _ in range(2000):
            draw = [rng.randrange(len(ids)) for _ in ids]
            values.append(100 * (f1([data["60"][i] for i in draw]) - f1([data["30"][i] for i in draw])))
        values.sort()
        comparisons[arm] = dict(test_f1_difference_points=100 * (f1(data["60"]) - f1(data["30"])),
            conditional_95_interval_points=[values[49], values[1949]], resamples=2000)
    receipt = dict(status="PASS", results_sha256=sha(root / "results.json"), test_sample_replays=checks,
        validation_curve=curves, fixed_60_minus_30=comparisons,
        bootstrap_scope="Conditional paired image bootstrap on fixed checkpoints; excludes selection, fitting and historical-exposure uncertainty.")
    out = root / "independent_audit.json"
    if out.exists(): raise ValueError("Refuse to overwrite audit")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run_root), indent=2))
