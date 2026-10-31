"""Re-evaluate frozen ICDAR gates with qualified Generic/Weak DeepSolo outputs."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from recognition.clip_router_evaluation import deepsolo_predictions
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_word_f1 import apply_policy, recognition_confidence
from recognition.icdar_budget_router import accept, lexical_counts, rank
from recognition.router_fallback_replay import ordered_frontier, replace_fallback, replay
from scripts.icdar_budget_data import annotation, key, read, sha


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def rational(p):
    return Fraction(2*p["tp"], p["detected_words"]+p["gt_words"])


def independent_counts(predictions, truth):
    def norm(s):
        return "".join(c for c in s.casefold() if c.isalnum())
    care = [p for p in predictions if not p["ignored"]]
    left = Counter(norm(p["text"]) for p in care if norm(p["text"]))
    right = Counter(norm(p["text"]) for p in truth if norm(p["text"]))
    tp = sum((left & right).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp,
                detected_words=len(care), gt_words=len(truth),
                all_outputs=len(predictions), ignored_outputs=len(predictions)-len(care))


def collect_fallback(config, ds, records, root):
    spec = config["new_fallback"][ds]
    protocol = read(Path(spec["raw_folder"])/"protocol.json")
    if protocol["source_identity"]["checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("Unexpected checkpoint")
    expected = read(spec["per_image"])
    if set(expected) != {r["sample_id"] for r in records}:
        raise ValueError("Replacement TEST population differs")
    lexicon = DeepSoloLexicon.from_files(**{
        "vocabulary_path": spec["lexicon_paths"]["vocabulary"],
        "pairs_path": spec["lexicon_paths"]["pairs"], "distance_exclusive": 1.5})
    neural = dict(spec["neural"], resolution="native", decoder="greedy", same_text_nms_iou=None)
    counts, raw_pins, changed_ignore, validity_counts = {}, {}, [], []
    for index, record in enumerate(records, 1):
        sid = record["sample_id"]
        if sha(record["image_path"]) != record["image_sha256"]:
            raise ValueError("Changed RGB")
        gt, validity = annotation(record)
        path = Path(spec["raw_folder"])/(key(sid)+".json")
        raw = read(path)
        payload = {k: v for k, v in raw.items() if k != "payload_sha256"}
        if (raw["payload_sha256"] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                or raw["contract"]["sample_id"] != sid
                or raw["contract"]["image_sha256"] != record["image_sha256"]
                or not raw["response"]["success"]):
            raise ValueError("Unauthenticated raw fallback: "+sid)
        raw_pins[str(path)] = sha(path)
        # Lexical correction receives no annotations. Project only admitted
        # neural detections, after source-compatible confidence filtering.
        source = {d["source_index"]: d for d in raw["response"]["detections"]}
        rows = [dict(instance_id=i, text=d["text"], score=d["score"],
                     recognition_confidence=recognition_confidence(d["raw"]["recognition_class_probabilities"]))
                for i, d in source.items()]
        rows = apply_policy(rows, neural)
        rows, decisions = lexicon.correct(rows, unmatched_policy=spec["unmatched_policy"])
        if [(p["instance_id"], p["text"]) for p in rows] != [
                (p["instance_id"], p["text"]) for p in expected[sid]["predictions"]]:
            raise ValueError("GT-blind lexical deployment differs from saved expert")
        response = dict(raw["response"], detections=[dict(source[p["instance_id"]], text=p["text"]) for p in rows])
        pp = deepsolo_predictions(response, record["image_wh"], record["orientation"], validity,
                                  spec["neural"]["score_threshold"])
        # Empty clipped envelopes remain denominator entries but never match.
        for p in pp:
            if p["invalid_envelope"] is not None:
                p["text"] = ""
        old = {p["instance_id"]: p for p in expected[sid]["predictions"]}
        changed = [dict(instance_id=p["instance_id"], previous=old[p["instance_id"]]["ignored"], current=p["ignored"])
                   for p in pp if p["ignored"] != old[p["instance_id"]]["ignored"]]
        if changed:
            changed_ignore.append(dict(sample_id=sid, changes=changed))
        c = lexical_counts(pp, gt)
        if c != independent_counts(pp, gt) or c["gt_words"] != expected[sid]["counts"]["gt_words"]:
            raise ValueError("Metric or care denominator mismatch")
        if Counter(g["text"] for g in gt) != Counter(g["text"] for g in expected[sid]["ground_truth"]):
            raise ValueError("Care transcriptions differ from original comparison")
        counts[sid] = c
        save(root/"fallback_evaluation"/ds/(key(sid)+".json"), dict(
            sample_id=sid, counts=c, predictions=pp, ground_truth=gt, decisions=decisions,
            annotation_path=record["annotation_path"], annotation_sha256=record["annotation_sha256"],
            raw_path=str(path), raw_sha256=raw_pins[str(path)], checkpoint_sha256=spec["checkpoint_sha256"]))
        save(root/"prediction_only"/ds/(key(sid)+".json"), dict(
            mode="approximate", claim_label="approximate reproduction", sample_id=sid,
            lexicon_kind=spec["lexicon_kind"], checkpoint_sha256=spec["checkpoint_sha256"],
            raw_path=str(path), raw_sha256=raw_pins[str(path)], ground_truth_used=False,
            detections=[{k:v for k,v in d.items() if k != "raw"} for d in response["detections"]]))
        if index % 100 == 0 or index == len(records):
            print(ds, "fallback replay", index, "/", len(records), flush=True)
    if len(counts) != spec["expected_images"] or sum(c["gt_words"] for c in counts.values()) != spec["expected_gt_words"]:
        raise ValueError("Incomplete official TEST")
    return counts, dict(raw_sha256=raw_pins, changed_ignore=changed_ignore)


def intervals(branches, accepted, draws, seed):
    ids = sorted(branches)
    matrix = np.array([[branches[s]["light" if s in accepted else "deepsolo"][k]
                       for k in ("tp", "detected_words", "gt_words")] for s in ids])
    base = np.array([[branches[s]["deepsolo"][k] for k in ("tp", "detected_words", "gt_words")] for s in ids])
    weights = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)
    a, b = weights @ matrix, weights @ base
    af = 2*a[:, 0]/(a[:, 1]+a[:, 2])
    bf = 2*b[:, 0]/(b[:, 1]+b[:, 2])
    return dict(f1_interval95_percent=(100*np.quantile(af, [.025,.975])).tolist(),
                gain_over_new_deepsolo_interval95_pp=(100*np.quantile(af-bf, [.025,.975])).tolist(),
                draws=draws, seed=seed, scope="Conditional fixed-rule image bootstrap; no fitting or threshold-selection uncertainty")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    config = read(args.config)
    if config["mode"] != "approximate" or config["claim_label"] != "approximate reproduction":
        raise ValueError("Explicit approximate mode required")
    for path, digest in config["pinned_inputs"].items():
        if sha(path) != digest:
            raise ValueError("Pinned input changed: "+path)
    root, oldroot = Path(config["study_root"]), Path(config["trained_router_root"])
    plan, frozen, original = [read(oldroot/name) for name in
                              ("resolved_configuration.json", "frozen_choices.json", "results.json")]
    if frozen["test_used_for_selection"] or not frozen["no_refit"]:
        raise ValueError("Source router selection is not frozen")
    results, evidence, table = {}, {}, []
    for ds, spec in config["new_fallback"].items():
        newcounts, evidence[ds] = collect_fallback(config, ds, plan["records"][ds]["test"], root)
        results[ds] = {}
        for arm in ("unet", "yolo"):
            data = read(oldroot/"test_scores"/(ds+"__"+arm+".json"))
            scores, keys = data["scores"], data["tie_keys"]
            branches = replace_fallback(data["branches"], newcounts)
            choice = frozen["choices"][ds][arm]
            policies = {}
            def add(name, accepted, source_point=None, policy=None):
                before = replay(data["branches"], accepted)
                after = replay(branches, accepted)
                if source_point is not None:
                    if (source_point["accepted_ids"] != sorted(accepted) or
                            source_point["combined"]["tp"] != before["tp"] or
                            source_point["combined"]["f1"] != before["f1"]):
                        raise ValueError("Original frozen decisions/metrics failed replay")
                after.update(previous_f1=before["f1"], gain_over_previous_pp=100*(after["f1"]-before["f1"]), policy=policy)
                if name.startswith(("validation_", "batch_")):
                    after["uncertainty"] = intervals(branches, set(accepted), config["bootstrap_draws"], config["seed"])
                policies[name] = after
                table.append(dict(dataset=ds, light_expert=arm, policy=name, threshold=None if policy is None else policy.get("threshold"),
                    previous_f1_percent=100*before["f1"], f1_percent=100*after["f1"], gain_pp=after["gain_over_previous_pp"],
                    tp=after["tp"], fp=after["fp"], fn=after["fn"], gt_words=after["gt_words"],
                    deepsolo_images=after["deepsolo_images"], total_images=after["total_images"], deepsolo_percent=after["deepsolo_percent"]))
            orig = original["datasets"][ds][arm]
            add("all_light", list(branches), orig["all_light"])
            add("all_deepsolo", [], orig["all_deepsolo"])
            for i, policy in enumerate(choice["policies"]):
                add("validation_"+str(policy["target_acceptance"]), accept(scores, policy, keys), orig["thresholds"][i], policy)
            for target, source_point in orig["predeclared_batch_quotas"].items():
                add("batch_"+target, rank(scores, list(branches), float(target), keys), source_point)
            curve = ordered_frontier(branches, scores, keys)
            baseline = curve[0]
            best = max(curve, key=lambda x: (rational(x), -x["deepsolo_images"]))
            no_loss = min((x for x in curve if rational(x) >= rational(baseline)),
                          key=lambda x: (x["deepsolo_images"], -rational(x)))
            save(root/"frontiers"/(ds+"__"+arm+".json"), dict(
                scope="TEST-descriptive fixed-rank frontier; boundary tie keys may split equal scores; no deployment threshold selected", points=curve))
            save(root/"per_image"/(ds+"__"+arm+".json"), dict(
                branches=branches, previous_branches=data["branches"], scores=scores, tie_keys=keys,
                source_model_sha256=choice["choice"]["model_sha256"]))
            results[ds][arm] = dict(expert=choice["expert"], primary_acceptance=orig["primary_acceptance"],
                source_model_sha256=choice["choice"]["model_sha256"], policies=policies,
                test_descriptive_best=best, test_descriptive_minimum_deepsolo_no_loss=no_loss,
                dense_budget_points={str(p):curve[min(round(len(branches)*(1-p/100)),len(curve)-1)] for p in range(10,101,10)})
            primary=policies["batch_"+str(orig["primary_acceptance"])];print(ds, arm, "primary F1", primary["f1"], "DeepSolo", primary["deepsolo_images"], "/",primary["total_images"],flush=True)
    save(root/"input_evidence.json", evidence)
    save(root/"results.json", dict(status="PASS", mode="approximate", claim_label="approximate reproduction",
        config_sha256=sha(args.config), new_training=False, actual_new_neural_inferences=0,
        scope="Logical cached image routing; original trained weights, expert choices, eligibility, cuts and quotas unchanged",
        datasets=results))
    with (root/"thresholds.csv").open("x", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    save(root/"execution.json", dict(status="PASS", results_sha256=sha(root/"results.json"),
        config_sha256=sha(args.config), runner_sha256=sha(__file__), core_sha256=sha(Path(__file__).parents[1]/"recognition/router_fallback_replay.py")))


if __name__ == "__main__":
    main()
