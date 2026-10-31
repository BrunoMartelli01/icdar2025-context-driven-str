"""Independently recount frozen routes and replay every deployed TEST gate."""

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np

from recognition.icdar_budget_deployment import predict
from scripts.icdar_budget_data import annotation, key, predictions, read, sha


FIELDS = ("tp", "fp", "fn", "detected_words", "gt_words", "all_outputs", "ignored_outputs")


def recount(pp, gt):
    def clean(text):
        return "".join(c for c in text.casefold() if c.isalnum())
    words = [clean(p["text"]) for p in pp if not p["ignored"]]
    truth = [clean(g["text"]) for g in gt]
    left, right = Counter(s for s in words if s), Counter(s for s in truth if s)
    tp = sum(min(n, right[s]) for s, n in left.items())
    return dict(tp=tp, fp=len(words)-tp, fn=len(truth)-tp, detected_words=len(words),
                gt_words=len(truth), all_outputs=len(pp), ignored_outputs=len(pp)-len(words))


def assert_route(branches, accepted, result):
    accepted = set(accepted)
    for k in FIELDS:
        assert sum(b["light" if sid in accepted else "deepsolo"][k] for sid,b in branches.items()) == result[k]
    assert result["f1"] == 2*result["tp"]/(result["detected_words"]+result["gt_words"])
    assert result["deepsolo_images"] == len(branches)-len(accepted)
    assert result["total_images"] == len(branches)
    assert result["deepsolo_percent"] == 100*(len(branches)-len(accepted))/len(branches)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    c = read(args.config)
    root, old = Path(c["study_root"]), Path(c["trained_router_root"])
    plan, frozen, results = read(old/"resolved_configuration.json"), read(old/"frozen_choices.json"), read(root/"results.json")
    for p,h in c["pinned_inputs"].items():
        assert sha(p) == h
    checks = dict(gate_replays=0, fallback_word_recounts=0, light_word_recounts=0,
                  frozen_policies=0, frontier_points=0, prediction_only_exports=0)
    source_pins, maxerror = {}, 0.
    for ds, spec in c["new_fallback"].items():
        records = {r["sample_id"]:r for r in plan["records"][ds]["test"]}
        fallback = {}
        truth, validity = {}, {}
        for sid, r in records.items():
            truth[sid], validity[sid] = annotation(r)
            evaluated = read(root/"fallback_evaluation"/ds/(key(sid)+".json"))
            assert evaluated["ground_truth"] == truth[sid]
            fallback[sid] = recount(evaluated["predictions"], truth[sid])
            assert fallback[sid] == evaluated["counts"]
            assert sha(evaluated["raw_path"]) == evaluated["raw_sha256"]
            checks["fallback_word_recounts"] += 1
            exported = read(root/"prediction_only"/ds/(key(sid)+".json"))
            assert exported["ground_truth_used"] is False
            assert not any(k in exported for k in ("ground_truth", "counts", "ignored"))
            assert exported["raw_sha256"] == evaluated["raw_sha256"]
            assert [(d["source_index"],d["text"]) for d in exported["detections"]] == [
                (d["instance_id"],d["text"]) for d in evaluated["predictions"] if d["invalid_envelope"] is None]
            checks["prediction_only_exports"] += 1
        for arm in ("unet", "yolo"):
            data = read(root/"per_image"/(ds+"__"+arm+".json"))
            previous = read(old/"test_scores"/(ds+"__"+arm+".json"))
            choice = frozen["choices"][ds][arm]
            modelpath = old/choice["choice"]["model_path"]
            assert sha(modelpath) == choice["choice"]["model_sha256"] == data["source_model_sha256"]
            bundle = joblib.load(modelpath)
            assert data["scores"] == previous["scores"] and data["tie_keys"] == previous["tie_keys"]
            assert data["previous_branches"] == previous["branches"]
            branches, scores, keys = data["branches"], data["scores"], data["tie_keys"]
            policies = results["datasets"][ds][arm]["policies"]
            for sid, r in records.items():
                path = old/"experts"/choice["expert"]/(key(sid)+".json")
                expert = read(path)
                # Original lightweight exports are plain sealed JSON, unlike
                # the separately authenticated DeepSolo raw-shard envelope.
                assert expert["model"] == choice["expert"]
                assert expert["ground_truth_used"] is False
                assert expert["image_sha256"] == r["image_sha256"]
                source_pins[str(path)] = sha(path)
                counts = recount(predictions(expert["rows"],r,validity[sid]),truth[sid])
                assert counts == branches[sid]["light"] == previous["branches"][sid]["light"]
                assert branches[sid]["deepsolo"] == fallback[sid]
                checks["light_word_recounts"] += 1
                maps = None
                if "maps" in bundle["groups"]:
                    assert sha(expert["maps_path"]) == expert["maps_sha256"]
                    source_pins[expert["maps_path"]] = expert["maps_sha256"]
                    with np.load(expert["maps_path"],allow_pickle=False) as z:
                        maps = z["probabilities"]
                single = predict(bundle,plan["config"]["features"],expert["rows"],r["image_wh"],
                    tie_key=r["rgb_sha256"],policies=choice["policies"],probabilities=maps,frame=expert.get("frame"))
                assert (single["score"] is None) == (scores[sid] is None)
                if single["score"] is not None:
                    error=abs(single["score"]-scores[sid]);maxerror=max(maxerror,error);assert error<1e-9
                for p in choice["policies"]:
                    q=str(p["target_acceptance"])
                    assert single["accept_light"][q] == (sid in policies["validation_"+q]["accepted_ids"])
                checks["gate_replays"] += 1
            order=sorted((s for s in scores if scores[s] is not None),key=lambda s:(-scores[s],keys[s],s))
            for name, result in policies.items():
                if name == "all_light":
                    accepted=list(records)
                elif name == "all_deepsolo":
                    accepted=[]
                elif name.startswith("batch_"):
                    q=float(name[len("batch_"):]);accepted=order[:round(len(records)*q)]
                else:
                    q=float(name[len("validation_"):]);p=next(p for p in choice["policies"] if p["target_acceptance"]==q)
                    t=p["threshold"];accepted=[s for s,v in scores.items() if v is not None and t is not None and
                        (v>t or v==t and keys[s]<=p["tie_key"])]
                    assert result["policy"] == p
                assert sorted(accepted) == result["accepted_ids"]
                assert_route(branches,accepted,result)
                checks["frozen_policies"] += 1
            curve=read(root/"frontiers"/(ds+"__"+arm+".json"))["points"]
            assert len(curve) == len(order)+1
            for n,p in enumerate(curve):
                assert_route(branches,order[:n],p)
                checks["frontier_points"] += 1
            best=max(curve,key=lambda p:(2*p["tp"]/(p["detected_words"]+p["gt_words"]),-p["deepsolo_images"]))
            assert best == results["datasets"][ds][arm]["test_descriptive_best"]
            print(ds,arm,"independent audit PASS",flush=True)
    report=dict(status="PASS",checks=checks,maximum_single_gate_score_error=maxerror,
        source_inputs_sha256=source_pins,results_sha256=sha(root/"results.json"),
        config_sha256=sha(args.config),code_sha256=sha(__file__),
        scope="Complete GT word recounts, source light outputs/maps, unchanged deployed scores/decisions, original policies and every descriptive frontier prefix. Lexicon source equivalence inherited from prior independent audit and exact raw replay.")
    with (root/"independent_audit.json").open("x") as f:
        json.dump(report,f,indent=2);f.write("\n")
    print(checks,flush=True)


if __name__ == "__main__":
    main()
