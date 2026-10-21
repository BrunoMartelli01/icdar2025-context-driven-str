"""Apply one GT-free, image-budget threshold to frozen InstructBLIP TRAIN scores."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from diagnostics.caption_routing_train import checked
from diagnostics.routing_threshold_train import sha, distribution
from scoring.budgeted_paired import budget_threshold, route


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routing-report", type=Path, required=True)
    p.add_argument("--routing-sha256", required=True)
    p.add_argument("--gt-report", type=Path, required=True)
    p.add_argument("--gt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-fallback-fraction", type=float, required=True)
    a = p.parse_args()
    outputs = [a.output_dir / name for name in ("results.json", "frozen-config.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite frozen results/configuration")
    if sha(a.routing_report) != a.routing_sha256:
        raise RuntimeError("routing report hash mismatch")
    source = checked(a.routing_report)
    scope = {"split": "TRAIN", "images": 229, "gt_words": 849}
    if source["scope"] != scope:
        raise RuntimeError("TRAIN229 required")
    rows, provenance = [], {}
    folder = a.routing_report.parent / "routes" / "instructblip" / "whole_caption"
    for path in sorted(folder.glob("[0-9]*.json")):
        sample = checked(path, source["contract_sha256"])
        old = sample["image_routes"]["selected_candidate_paired"]
        if any(sample[k]["window"] != sample["T2"] for k in ("T1_score", "T3_score")):
            raise RuntimeError("whole T2 required")
        arguments = {k: old[k] for k in ("T1", "T2", "T3", "S1", "S3", "L1", "L3")}
        control = route(tau=.8, **arguments)
        if (abs(control.C-old["C"]) > 1e-12 or control.selected_output != old["selected_output"]
                or arguments["T2"] != sample["T2"]):
            raise RuntimeError("paired control mismatch")
        rows.append({"sample_id": sample["sample_id"], "arguments": arguments,
                     "C": control.C, "nonempty": bool(control.selected_output.strip())})
        provenance[str(path)] = sha(path)
    if len(rows) != 229 or {r["sample_id"] for r in rows} != {str(i) for i in range(100, 329)}:
        raise RuntimeError("incorrect TRAIN population")
    # Selection ends here: no GT file has been loaded and no correctness is visible.
    tau = budget_threshold([r["C"] for r in rows], [r["nonempty"] for r in rows],
                           a.max_fallback_fraction)
    config = {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
              "classification": "PRACTICAL_APPROXIMATION", "caption_profile": "instructblip",
              "caption_comparison": "whole_caption", "strategy": "selected_candidate_paired",
              "alpha": .6, "beta": .4, "tau": tau, "empty_output_guard": True,
              "tie_policy": "T1 wins semantic ties; equal C values never split",
              "max_train_fallback_image_fraction": a.max_fallback_fraction,
              "selection": "largest observed C cutoff satisfying image budget; no GT used",
              "source_report_sha256": a.routing_sha256,
              "source_contract_sha256": source["contract_sha256"],
              "strict_defaults_modified": False}
    # GT is used only for reporting after the configuration has been fixed.
    if sha(a.gt_report) != a.gt_sha256:
        raise RuntimeError("GT report hash mismatch")
    gt_source = json.loads(a.gt_report.read_text())
    if gt_source["protocol"]["scope"] != scope:
        raise RuntimeError("GT split mismatch")
    gt = defaultdict(list)
    for word in gt_source["per_word"]:
        gt[word["sample_id"]].append(word["text"])
    if set(gt) != {r["sample_id"] for r in rows} or sum(map(len, gt.values())) != 849:
        raise RuntimeError("GT population mismatch")
    # Explicit word denominator: no hidden GT token expansion.
    if any(len(w.split()) != 1 for words in gt.values() for w in words):
        raise RuntimeError("multi-token GT requires a separately specified matching protocol")
    accepted_correct = accepted_wrong = fallback_words = fallback_count = 0
    for row in rows:
        decision = route(tau=tau, **row.pop("arguments"))
        words = gt[row["sample_id"]]
        expected = Counter(w.casefold() for w in words)
        predicted = Counter(decision.selected_output.casefold().split())
        correct = sum((expected & predicted).values())
        row.update(decision.to_dict(), gt_words=len(words), posthoc_correct_words=correct,
                   posthoc_wrong_tokens=sum(predicted.values())-correct)
        if decision.needs_fallback:
            fallback_count += 1
            fallback_words += len(words)
        else:
            accepted_correct += correct
            accepted_wrong += row["posthoc_wrong_tokens"]
    metrics = {"accepted_images": 229-fallback_count, "fallback_images": fallback_count,
               "fallback_image_fraction": fallback_count/229,
               "accepted_correct_words": accepted_correct, "accepted_wrong_tokens": accepted_wrong,
               "accepted_image_gt_words": 849-fallback_words, "fallback_image_gt_words": fallback_words,
               "perfect_fallback_ceiling_correct_words": accepted_correct+fallback_words,
               "perfect_fallback_ceiling_rate": (accepted_correct+fallback_words)/849,
               "target_correct_words": math.ceil(.8*849), "final_accuracy": None,
               "table5_accuracy": None, "fallback_executed": False}
    report = {"configuration": config, "scope": scope, "metrics": metrics,
              "score_distribution": distribution([r["C"] for r in rows]),
              "posthoc_matching": "image-local casefold whitespace-token multiset; missed words wrong; no spatial constraint",
              "limitations": ["TRAIN budget only, not a TEST guarantee", "No DeepSolo outputs evaluated",
                             "Optimistic ceiling assumes perfect DeepSolo on every rejected image"],
              "gt_report_sha256": a.gt_sha256, "route_file_sha256": provenance, "per_image": rows}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for path, value in zip(outputs, (report, config)):
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(path, "SHA256", sha(path), flush=True)
    print(json.dumps({"tau": tau, **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
