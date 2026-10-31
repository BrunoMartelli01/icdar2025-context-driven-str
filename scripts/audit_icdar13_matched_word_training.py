"""Independently replay ICDAR13 TEST lexical counts from CSV labels and emissions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(value):
    return "".join(c for c in value.casefold() if c.isalnum())


def audit_historical_emissions(population_plan, expert_root, output):
    """Score unchanged cached expert rows with and without symbol-only emission."""
    population_plan, expert_root, output = map(Path, (population_plan, expert_root, output))
    plan = read(population_plan)
    if plan["mode"] != "approximate":
        raise ValueError("Historical comparison requires approximate inputs")
    records = plan["records"]["icdar13"]["test"]
    totals = {name: dict(tp=0, fp=0, fn=0) for name in ("original", "common_emission")}
    changed, sources = [], {}
    for r in records:
        path = expert_root / (r["sample_id"].replace(":", "__") + ".json")
        sample = read(path)
        if sample["image_sha256"] != r["image_sha256"] or digest(r["annotation_path"]) != r["annotation_sha256"]:
            raise ValueError("Historical RGB/annotation identity changed")
        sources[str(path)] = digest(path)
        with Path(r["annotation_path"]).open(encoding="utf-8-sig", newline="") as f:
            labels = list(csv.reader(f, skipinitialspace=True))
        if any(len(row) != 5 for row in labels):
            raise ValueError("Expected official five-column Task1 labels")
        truth = Counter(normalize(row[4]) for row in labels)
        rows = sample["rows"]
        keep = [row for row in rows if normalize(row["raw_text"])]
        for name, values in (("original", rows), ("common_emission", keep)):
            observed = Counter(normalize(row["raw_text"]) for row in values if normalize(row["raw_text"]))
            tp = sum((truth & observed).values())
            for k, v in dict(tp=tp, fp=len(values)-tp, fn=len(labels)-tp).items():
                totals[name][k] += v
        if len(keep) != len(rows):
            changed.append(dict(sample_id=r["sample_id"], rejected=[v["raw_text"] for v in rows if not normalize(v["raw_text"])]))
    def f1(counts):
        return 2*counts["tp"]/(2*counts["tp"]+counts["fp"]+counts["fn"])
    value = dict(mode="approximate", claim_label="approximate reproduction", source_root=str(expert_root),
        test_images=len(records), total_images=len(records), deepsolo_images=0, deepsolo_percentage=0.,
        counts=totals["common_emission"], f1=f1(totals["common_emission"]), original_counts=totals["original"],
        original_f1=f1(totals["original"]), changed_images=changed, files_sha256=sources,
        population_plan_sha256=digest(population_plan), auditor_sha256=digest(__file__))
    with output.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")
    return value


def replay_test(root, plan, name):
    expected = next(r for r in read(root / "results.json")["results"] if r["candidate"] == name and r["split"] == "test")
    lookup = {r["sample_id"]: r for r in plan["test_records"]}
    total = dict(tp=0, fp=0, fn=0, detected_words=0, gt_words=0, all_outputs=0, ignored_outputs=0)
    samples = []
    for sid in plan["partitions"]["test"]:
        record = lookup[sid]
        if digest(record["annotation_path"]) != record["annotation_sha256"]:
            raise ValueError("Changed official Task1 TEST annotation")
        with Path(record["annotation_path"]).open(encoding="utf-8-sig", newline="") as f:
            labels = list(csv.reader(f, skipinitialspace=True))
        if any(len(row) != 5 for row in labels):
            raise ValueError("Official Task1 annotation must have five columns")
        truth = Counter(normalize(row[4]) for row in labels)
        path = root / "evaluation" / name / "test/samples" / (sid.replace(":", "__") + ".json")
        sample = read(path)
        rows = sample["rows"]
        if any(not normalize(r["raw_text"]) for r in rows):
            raise ValueError("Declared no-word rejection was not applied")
        # IC13 Task1 TEST has full validity; all emitted words enter its denominator.
        observed = Counter(normalize(row["raw_text"]) for row in rows)
        tp = sum((truth & observed).values())
        counts = dict(tp=tp, fp=len(rows)-tp, fn=len(labels)-tp, detected_words=len(rows),
            gt_words=len(labels), all_outputs=len(rows), ignored_outputs=0)
        if counts != sample["counts"]:
            raise ValueError("Independent per-image count mismatch: " + sid)
        for k in total:
            total[k] += counts[k]
        samples.append(dict(sample_id=sid, counts=counts, source_sha256=digest(path)))
    f1 = 2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"])
    if any(total[k] != expected["counts"][k] for k in total) or f1 != expected["counts"]["f1"]:
        raise ValueError("Independent TEST aggregate mismatch")
    return dict(candidate=name, counts=total, f1=f1, images=len(samples), samples=samples,
        deepsolo_images=0, total_images=len(samples), deepsolo_percentage=0.)


def audit(root, mode):
    if mode != "approximate":
        raise ValueError("Explicit approximate mode required")
    plan, result = read(root / "resolved_configuration.json"), read(root / "results.json")
    if plan["mode"] != mode or result["mode"] != mode or result["test_used_for_selection"]:
        raise ValueError("Invalid mode/selection claim")
    for file, checksum in plan["code_sha256"].items():
        if digest(root / "code_snapshot" / file) != checksum:
            raise ValueError("Frozen scientific source changed: " + file)
    selection = read(root / "selection_frozen.json")
    independent_winner = sorted(enumerate(selection["all_candidates"]), key=lambda x: (-x[1]["counts"]["f1"], x[0]))[0][1]
    if independent_winner != selection["winner"] or independent_winner != result["winner"]:
        raise ValueError("Independent stable selection replay failed")
    if any(r["split"] != "selection" for r in selection["all_candidates"]):
        raise ValueError("Non-selection candidate entered selection")
    if digest(result["winner"]["checkpoint"]) != result["winner"]["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    epochs = []
    for branch in plan["config"]["branches"]:
        history = read(root / branch / "epochs.json")
        expected_epochs = 1 if plan["smoke"] else plan["config"]["training"]["epochs"]
        if [r["epoch"] for r in history] != list(range(1, expected_epochs+1)) or any(r["samples"] != len(plan["partitions"]["fit"]) for r in history):
            raise ValueError("Incomplete declared training")
        eligible = [(0, read(root / "source/selection/metrics.json")["counts"]["f1"])] + [(r["epoch"], r["selection"]["f1"]) for r in history if "selection" in r]
        chosen = max(eligible, key=lambda r: r[1])
        training = read(root / branch / "training.json")
        if training["status"] != "SUCCESS" or digest(root / branch / "best.pt") != training["checkpoint_sha256"]:
            raise ValueError("Completed branch checkpoint changed")
        if training["best_epoch"] != chosen[0] or training["best_validation"]["f1"] != chosen[1]:
            raise ValueError("Independent checkpoint selection replay failed")
        epochs.append(history)
    if [r["sample_order_sha256"] for r in epochs[0]] != [r["sample_order_sha256"] for r in epochs[1]]:
        raise ValueError("Loss controls had different training sample orders")
    tests = [replay_test(root, plan, name) for name in ("source_matched_fixed", "selected")]
    audit_value = dict(status="PASS", mode=mode, claim_label="approximate reproduction", result_class=plan["result_class"],
        immutable_sources=len(plan["code_sha256"]), complete_training_epochs=sum(map(len, epochs)),
        independent_checkpoint_and_policy_selection="PASS", independent_official_CSV_word_counts="PASS",
        test_results=tests, results_sha256=digest(root / "results.json"),
        plan_sha256=digest(root / "resolved_configuration.json"), auditor_sha256=digest(__file__))
    destination = root / "independent_lexical_audit.json"
    if destination.exists() and read(destination) != audit_value:
        raise ValueError("Preserve the previous audit; do not overwrite changed evidence")
    if not destination.exists():
        destination.write_text(json.dumps(audit_value, indent=2, allow_nan=False) + "\n")
    return audit_value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.run_root, args.mode)
    print(json.dumps({k: v for k, v in value.items() if k != "test_results"}))


if __name__ == "__main__":
    main()
