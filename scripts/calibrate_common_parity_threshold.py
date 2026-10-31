"""Calibrate common-v1 score cuts on unused CAL, then evaluate frozen cuts on TEST."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
import traceback

from recognition.common_parity_threshold import accepted_ids, calibrate, evaluate


REPO = Path(__file__).resolve().parents[1]
CODE = ("recognition/common_parity_threshold.py", "scripts/calibrate_common_parity_threshold.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_inputs(config):
    source = Path(config["source_root"])
    for name, digest in config["pinned_inputs"].items():
        if sha(source / name) != digest:
            raise ValueError("Changed source input: " + name)
    qualification = read(source / "deployment_qualification.json")
    if qualification["status"] != "PASS" or sha(source / qualification["audit_path"]) != qualification["audit_sha256"]:
        raise ValueError("Unqualified source pipeline")
    if read(source / qualification["audit_path"])["status"] != "PASS":
        raise ValueError("Source independent audit failed")
    for name, digest in qualification["artifact_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError("Changed qualified source artifact: " + name)
    plan = read(source / "resolved_configuration.json")
    # Original v1 predates the named policy field. Its saved disjoint roles and
    # selection_frozen IDs below establish the boundary directly.
    if (plan["mode"] != "approximate" or plan["smoke"]
            or plan["config"].get("validation_policy") not in (None, "separate")):
        raise ValueError("A full source run with separate unused CAL is required")
    cases = read(source / "cases.json")
    for name in config["cases"]:
        case = cases[name]
        parts = case["partitions"]
        ids = [s for values in parts.values() for s in values]
        if len(set(ids)) != len(ids) or not parts["calibration"]:
            raise ValueError("Overlapping image roles or absent calibration")
        groups = [{case["tie_keys"][s] for s in values} for values in parts.values()]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise ValueError("Overlapping RGB roles")
        frozen = read(source / "models" / name / "selection_frozen.json")
        if (frozen["test_used_for_selection"] or frozen["calibration_ids"] != parts["calibration"]
                or frozen["selection_ids"] != parts["selection"] or frozen["fit_ids"] != parts["fit"]):
            raise ValueError("Source selection/calibration boundary changed")
    return source, cases


def prepare(config_path, mode):
    if mode != "approximate" or config_path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("Explicit APPROXIMATE configuration namespace required")
    config = read(config_path)
    if (config["mode"] != mode or config["validation_role"] != "calibration"
            or config["claim_label"] != "approximate reproduction"):
        raise ValueError("Invalid scientific configuration")
    verify_inputs(config)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    sources = {name: sha(REPO / name) for name in CODE}
    for name in CODE:
        dest = root / "code_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
    write(root / "resolved_configuration.json", dict(
        mode=mode, claim_label=config["claim_label"], result_class="approximate reproduction",
        config=config, config_sha256=sha(config_path), code_sha256=sources,
        runtime=dict(host=socket.getfqdn(), python=sys.version, platform=platform.platform(),
                     device="cpu", actual_new_neural_inference=0, actual_new_deepsolo_inference=0),
        historical_test_exposure=True, test_used_for_selection=False,
        test_bytes_hashed_but_labels_not_interpreted=True))
    return root


def load_plan(root):
    plan = read(root / "resolved_configuration.json")
    if plan["mode"] != "approximate":
        raise ValueError("APPROXIMATE only")
    for name, digest in plan["code_sha256"].items():
        if sha(REPO / name) != digest:
            raise ValueError("Executed source changed: " + name)
    source, cases = verify_inputs(plan["config"])
    return plan, source, cases


def calibration_stage(root):
    plan, source, cases = load_plan(root)
    frozen = {}
    folder = root / "calibration"
    folder.mkdir(exist_ok=False)
    # This stage never parses a TEST label or TEST branch file.
    for name in plan["config"]["cases"]:
        model = source / "models" / name
        ids = cases[name]["partitions"]["calibration"]
        values = read(model / "scores.json")
        scores = {s: values["scores"][s] for s in ids}
        guard = read(model / "selection_frozen.json")["winner"]["guard"]
        eligible = [s for s in ids if not guard or values["eligible"][s]]
        branches = read(model / "train_branches.json")
        result = calibrate(scores, {s: branches[s] for s in ids}, eligible)
        write(folder / (name + ".json"), result)
        frozen[name] = dict(threshold=result["winner"]["threshold"], guard=guard,
                            calibration_ids=ids, calibration_result_sha256=sha(folder / (name + ".json")))
        point = result["winner"]
        print(json.dumps(dict(stage="calibrated", case=name, threshold=point["threshold"],
                              f1=point["combined"]["f1"], deepsolo_images=point["deepsolo_images"],
                              total_images=len(ids), deepsolo_percentage=point["deepsolo_percentage"])), flush=True)
    write(root / "thresholds_frozen.json", dict(cases=frozen, test_used_for_selection=False,
          test_labels_opened=False, calibration_role="calibration",
          resolved_configuration_sha256=sha(root / "resolved_configuration.json")))


def recount(rows, truth, ignored):
    if len(rows) != len(ignored):
        raise ValueError("Prediction/ignore cardinality mismatch")
    def norm(text):
        return "".join(c for c in text.casefold() if c.isalnum())
    remaining = Counter(norm(g["text"]) for g in truth if norm(g["text"]))
    tp = n = 0
    for row, skip in zip(rows, ignored):
        if skip:
            continue
        n += 1
        token = norm(row["raw_text"])
        if token and remaining[token]:
            remaining[token] -= 1
            tp += 1
    return dict(tp=tp, fp=n-tp, fn=len(truth)-tp, detected_words=n, gt_words=len(truth),
                all_outputs=len(rows), ignored_outputs=sum(ignored))


def rebuilt_branches(source, name, ids, split):
    labels = read(source / "data" / (name + "__" + split + "_labels.json"))
    outputs = read(source / "models" / name / "observable_outputs.json")["corrected_rows"]
    result = {}
    for s in ids:
        gt = labels[s]
        rows = outputs[s]
        ct = recount(rows, gt["ground_truth"], [gt["ignored"][r["source_row"]] for r in rows])
        if ct["gt_words"] != gt["teacher"]["gt_words"]:
            raise ValueError("Teacher care denominator changed")
        result[s] = dict(light=ct, deepsolo=gt["teacher"])
    return result


def paired_interval(branches, accepted, seed, draws):
    import numpy as np
    names = sorted(branches)
    accepted = set(accepted)
    mixed = np.array([[branches[s]["light" if s in accepted else "deepsolo"][k]
                       for k in ("tp", "fp", "fn")] for s in names])
    teacher = np.array([[branches[s]["deepsolo"][k] for k in ("tp", "fp", "fn")] for s in names])
    ix = np.random.default_rng(seed).integers(len(names), size=(draws, len(names)))
    def f1(values):
        v = values[ix].sum(1)
        den = 2*v[:, 0] + v[:, 1] + v[:, 2]
        return np.divide(200*v[:, 0], den, out=np.zeros(len(v)), where=den != 0)
    return dict(delta_f1_points_95_interval=np.quantile(f1(mixed)-f1(teacher), [.025, .975]).tolist(),
                seed=seed, draws=draws, numpy=np.__version__,
                scope="Conditional paired-image bootstrap; excludes fitting/selection variance and prior TEST exposure")


def evaluation_stage(root):
    plan, source, cases = load_plan(root)
    frozen = read(root / "thresholds_frozen.json")
    if (frozen["test_used_for_selection"] or frozen["test_labels_opened"]
            or frozen["resolved_configuration_sha256"] != sha(root / "resolved_configuration.json")):
        raise ValueError("Invalid frozen calibration boundary")
    folder = root / "test"
    folder.mkdir(exist_ok=False)
    results = {}
    # All four thresholds already exist on disk before any TEST labels are parsed.
    for name in plan["config"]["cases"]:
        cut = frozen["cases"][name]
        if sha(root / "calibration" / (name + ".json")) != cut["calibration_result_sha256"]:
            raise ValueError("Changed frozen calibration")
        ids = cases[name]["partitions"]["test"]
        values = read(source / "models" / name / "scores.json")
        scores = {s: values["scores"][s] for s in ids}
        eligible = [s for s in ids if not cut["guard"] or values["eligible"][s]]
        accepted = accepted_ids(scores, eligible, cut["threshold"])
        write(folder / (name + "__decisions_before_labels.json"), dict(accepted_ids=accepted,
              threshold=cut["threshold"], thresholds_frozen_sha256=sha(root / "thresholds_frozen.json")))
        branches = rebuilt_branches(source, name, ids, "test")
        old = read(source / "test" / (name + ".json"))
        if branches != old["branches"] or scores != old["scores"] or set(eligible) != set(old["eligible"]):
            raise ValueError("Qualified TEST replay differs")
        primary = evaluate(branches, accepted)
        ds = evaluate(branches, [])
        light = evaluate(branches, ids)
        from recognition.common_parity_threshold import at_least
        result = dict(threshold=cut["threshold"], primary=primary, all_deepsolo=ds, all_light=light,
                      test_parity=at_least(primary["combined"], ds["combined"]),
                      delta_f1_points=100*(primary["combined"]["f1"]-ds["combined"]["f1"]),
                      bootstrap=paired_interval(branches, accepted, **plan["config"]["bootstrap"]))
        # A separately labelled post-evaluation diagnostic, never an operational choice.
        diagnostic = calibrate(scores, branches, eligible)
        diagnostic.update(result_class="TEST-label diagnostic only", deployable=False,
                          scope="Best TEST cut of the fixed score ordering; not a general image-selection oracle")
        write(folder / (name + "__test_cut_diagnostic.json"), diagnostic)
        write(folder / (name + "__per_image.json"), {
            s: dict(score=scores[s], eligible=s in eligible, selected_expert="light" if s in accepted else "deepsolo",
                    **branches[s]) for s in ids})
        result["test_cut_diagnostic"] = {k: diagnostic[k] for k in ("winner", "deployable", "scope")}
        results[name] = result
        print(json.dumps(dict(stage="test_evaluated", case=name, f1=primary["combined"]["f1"],
                              delta_f1_points=result["delta_f1_points"], test_parity=result["test_parity"],
                              deepsolo_images=primary["deepsolo_images"], total_images=len(ids),
                              deepsolo_percentage=primary["deepsolo_percentage"])), flush=True)
    write(root / "results.json", dict(mode="approximate", claim_label="approximate reproduction",
          status="COMPLETE", results=results, thresholds_frozen_sha256=sha(root / "thresholds_frozen.json"),
          test_used_for_selection=False, historical_test_exposure=True,
          actual_new_neural_inference=0, actual_new_deepsolo_inference=0))


def audit_stage(root):
    plan, source, cases = load_plan(root)
    results = read(root / "results.json")
    frozen = read(root / "thresholds_frozen.json")
    if sha(root / "thresholds_frozen.json") != results["thresholds_frozen_sha256"]:
        raise ValueError("Frozen cuts changed after TEST")
    checked = trials_checked = 0
    for name in plan["config"]["cases"]:
        cut = frozen["cases"][name]
        values = read(source / "models" / name / "scores.json")
        for split, label_split, saved in (("calibration", "train", read(root / "calibration" / (name + ".json"))),
                                          ("test", "test", read(root / "test" / (name + "__test_cut_diagnostic.json")))):
            ids = cases[name]["partitions"][split]
            branches = rebuilt_branches(source, name, ids, label_split)
            checked += len(ids)
            eligible = [s for s in ids if not cut["guard"] or values["eligible"][s]]
            scores = {s: values["scores"][s] for s in ids}
            # Separate implementation: ordered tie groups, Fraction counts, direct sums.
            groups = [[s for s in eligible if scores[s] == v]
                      for v in sorted({scores[s] for s in eligible}, reverse=True)]
            selections = [[]]
            for group in groups:
                selections.append(selections[-1] + group)
            def direct(accepted):
                aa = set(accepted)
                rows = [branches[s]["light" if s in aa else "deepsolo"] for s in ids]
                total = {k: sum(v[k] for v in rows) for k in ("tp", "fp", "fn", "detected_words", "gt_words", "all_outputs", "ignored_outputs")}
                den = total["detected_words"] + total["gt_words"]
                return total, Fraction(2*total["tp"], den) if den else Fraction(0)
            base = direct([])[1]
            if len(selections) != len(saved["trials"]):
                raise ValueError("Missing scalar cut")
            for accepted, trial in zip(selections, saved["trials"]):
                total, f1 = direct(accepted)
                threshold = trial["threshold"]
                expected = sorted(s for s in eligible if threshold["kind"] != "accept_none" and scores[s] >= threshold["value"])
                if sorted(accepted) != expected or sorted(accepted) != trial["accepted_ids"]:
                    raise ValueError("Tie/boundary replay differs")
                if (any(trial["combined"][k] != v for k, v in total.items())
                        or float(f1) != trial["combined"]["f1"] or (f1 >= base) != trial["parity"]):
                    raise ValueError("Independent scalar-cut recount differs")
                if trial["deepsolo_images"] != len(ids)-len(accepted) or trial["total_images"] != len(ids):
                    raise ValueError("Routing denominator differs")
                trials_checked += 1
            optimum = max((v for v in saved["trials"] if v["parity"]), key=lambda v: v["light_images"])
            if optimum != saved["winner"]:
                raise ValueError("Acceptance optimum differs")
            if split == "calibration":
                if optimum["threshold"] != cut["threshold"]:
                    raise ValueError("Calibration-selected threshold differs")
                original = read(source / "models" / name / "train_branches.json")
                if branches != {s: original[s] for s in ids}:
                    raise ValueError("CAL independently recounted branches differ")
            else:
                expected = sorted(s for s in eligible if cut["threshold"]["kind"] != "accept_none" and scores[s] >= cut["threshold"]["value"])
                total, f1 = direct(expected)
                primary = results["results"][name]["primary"]
                decisions = read(root / "test" / (name + "__decisions_before_labels.json"))
                if (expected != primary["accepted_ids"] or expected != decisions["accepted_ids"]
                        or decisions["thresholds_frozen_sha256"] != results["thresholds_frozen_sha256"]
                        or any(primary["combined"][k] != v for k, v in total.items())
                        or float(f1) != primary["combined"]["f1"]):
                    raise ValueError("Frozen TEST decision/recount differs")
    write(root / "independent_audit.json", dict(status="PASS", per_image_recounts=checked,
          scalar_cut_replays=trials_checked, thresholds_frozen_sha256=sha(root / "thresholds_frozen.json"),
          results_sha256=sha(root / "results.json"), auditor_sha256=sha(__file__),
          scope="Independent count, tie-group, exact-F1 optimum and frozen TEST decision replay; upstream fitting qualification inherited"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stage", required=True, choices=["calibrate", "evaluate", "audit"])
    args = parser.parse_args()
    if bool(args.config) == bool(args.run_root):
        parser.error("Provide exactly one explicit configuration or existing run root")
    if args.config and args.stage != "calibrate":
        parser.error("New configurations must calibrate before evaluating")
    root = prepare(args.config, args.mode) if args.config else args.run_root
    try:
        {"calibrate": calibration_stage, "evaluate": evaluation_stage, "audit": audit_stage}[args.stage](root)
    except BaseException as exc:
        write(root / ("failure_" + str(time.time_ns()) + ".json"), dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    print(json.dumps(dict(status="SUCCESS", stage=args.stage, root=str(root))), flush=True)


if __name__ == "__main__":
    main()
