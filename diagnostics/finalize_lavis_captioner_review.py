"""Summarize frozen caption outputs and explicit five-image visual observations.

No model imports, inference, OCR calls, fitting or accuracy-based selection.
"""
import argparse
import json
import statistics
from pathlib import Path

from diagnostics.lavis_captioner_candidates import SAMPLES, atomic, sha, text_metrics, verify_record, write_json


NOTES = {
    "blip-base-coco": {
        "status": "INSUFFICIENT_T2_DETAIL",
        "samples": {
            "1": "Recovers the safety message; minimal scene context.",
            "25": "Generic liquid bottle; omits visible Laboratoires Garnier.",
            "26": "Plausible blue door/sign; omits all visible text.",
            "27": "Chocolate hot instead of brand Chocomel; can is not established by the cropped package view.",
            "28": "Recovers book title; no animal or layout context.",
        },
    },
    "blip2-vitl": {
        "status": "INSUFFICIENT_T2_GROUNDING_AND_DETAIL",
        "samples": {
            "1": "Recovers safety message; wall location is not established by the reflective surface.",
            "25": "Garnierr is misspelled; table is not visible in the crop.",
            "26": "Hallucinates sign text 1 2 instead of visible 2.17; other text omitted.",
            "27": "Chocolate hot is not the brand text; spoon is not visible.",
            "28": "Recovers title; very little scene context.",
        },
    },
    "florence2-base-ft": {
        "status": "LIGHTWEIGHT_COMPARATOR_WITH_GROUNDING_ERRORS",
        "samples": {
            "1": "Recovers safety text and window/building context; reflected scene is described imprecisely as behind window.",
            "25": "Omits Garnier in caption; narrow orange lower edge is described as red stripe; OCR separately recovers brand.",
            "26": "Reads 2.17 but calls its dark digits white; omits descriptive sign text recovered by separate OCR.",
            "27": "Chocomes instead of Chocomel; plastic lid on top is unsupported by the visible printed package.",
            "28": "Recovers title, colors and several animals; fish is not visibly supported (right-side animal appears dolphin-like).",
        },
    },
    "florence2-large-ft": {
        "status": "PREFERRED_VISUAL_T2_CANDIDATE",
        "samples": {
            "1": "Safety text and reflected outdoor context are well grounded.",
            "25": "Recovers Garnier and typography; calls label black although lettering is black; orange circle exceeds visible evidence.",
            "26": "Correct 2.17/dark digits/logo context; main descriptive sign text absent from caption, present only in OCR.",
            "27": "Correct Chocomel hot and package/cup illustration; secondary Dutch text omitted.",
            "28": "Correct title and richer animal context; penguin identification uncertain; word word reflects literal word naming, not a generation loop.",
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/diagnostics/lavis-captioner-candidates-20260914"))
    args = parser.parse_args()
    root = args.root
    comparison = json.loads((root / "comparison.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    for image in protocol["images"].values():
        if sha(image["path"]) != image["sha256"]:
            raise RuntimeError("Frozen diagnostic image changed")
    inputs = {"comparison.json": sha(root / "comparison.json"), "protocol.json": sha(root / "protocol.json")}
    records = {}
    for name in NOTES:
        old = comparison["candidates"][name]
        path = Path(protocol["reference"]["path"]) if name == "florence2-large-ft" else root / (name + ".json")
        if sha(path) != old["json_sha256"]:
            raise RuntimeError(f"Compared candidate artifact changed: {name}")
        record = json.loads(path.read_text())
        historical = name == "florence2-large-ft"
        if not historical:
            verify_record(record, protocol, name, record["device"])
            if record["status"] != "COMPLETE":
                raise RuntimeError(f"Incomplete candidate: {name}")
        samples = {str(int(sid)): value for sid, value in record["samples"].items()}
        if set(samples) != set(map(str, SAMPLES)):
            raise RuntimeError("Wrong diagnostic population")
        inputs[str(path)] = sha(path)
        times = [value["caption_latency_seconds" if historical else "latency_seconds"] for value in samples.values()]
        checkpoint = record.get("checkpoint", {})
        weights = checkpoint.get("checkpoint_bytes")
        if checkpoint.get("separate_frozen_visual_checkpoint"):
            weights += checkpoint["separate_frozen_visual_checkpoint"]["checkpoint_bytes"]
        records[name] = {
            "parameters": old.get("parameters", old.get("parameters_model_card")),
            "parameter_evidence": "historical model-card estimate" if historical else "measured model.parameters",
            "checkpoint_bytes": weights,
            "latency_mean_seconds": statistics.mean(times), "latency_median_seconds": statistics.median(times),
            "latency_scope": "historical generation-only" if historical else "preprocess + generation + decode, excludes model load",
            "word_count_mean": statistics.mean(text_metrics(s["caption"])["word_count"] for s in samples.values()),
            "word_count_range": [min(len(s["caption"].split()) for s in samples.values()), max(len(s["caption"].split()) for s in samples.values())],
            "repeated_trigram_fraction_mean": statistics.mean(text_metrics(s["caption"])["repeated_trigram_fraction"] for s in samples.values()),
            "prompt_echo_count": None if historical else sum(s["prompt_echo"] for s in samples.values()),
            "visual_status": NOTES[name]["status"],
            "samples": {sid: {"caption": sample["caption"], "separate_ocr_not_used_as_T2": sample.get("ocr"),
                               "visual_observation": NOTES[name]["samples"][sid]} for sid, sample in samples.items()},
        }
    result = {
        "schema_version": 1, "mode": "APPROXIMATE", "classification": "PRACTICAL_APPROXIMATION",
        "scope": "five frozen TEST images, visual diagnostic only", "sample_ids": list(SAMPLES),
        "status": "VISUAL_REVIEW_COMPLETE", "inputs_sha256": inputs, "review_script_sha256": sha(__file__),
        "review_method": "Manual inspection of all five original RGB images and all twenty saved captions; observations are qualitative, not accuracy measurements.",
        "candidates": records, "recommended_T2": "florence2-large-ft",
        "lightweight_comparator": "florence2-base-ft",
        "train_recommendation": "GO for a frozen TRAIN diagnostic with Florence-large, subject to pinning its historical model revision and recording full runtime provenance. Florence-base is an optional lightweight comparator. Full TRAIN has not been executed.",
        "limitations": ["No evidence of improved MPNet/routing/recognition performance yet.",
                       "Historical large runtime precision/revision/image hashes are incomplete; latencies are not a controlled speed comparison.",
                       "Mean timings include first-image startup effects; median also reported, no new timing runs.",
                       "No candidate resolves the paper BLIP-2 ViT-B checkpoint ambiguity.",
                       "Separate Florence OCR was inspected but not incorporated into T2 captions or the T2 recommendation."],
    }
    target = root / "final-visual-review.json"
    write_json(target, result)
    lines = ["# Completed five-image captioner visual review", "", "PRACTICAL_APPROXIMATION. No new inference or full TRAIN.", "",
             "| Candidate | Parameters | Mean s | Median s | Mean words | Assessment |", "|---|---:|---:|---:|---:|---|"]
    for name, row in records.items():
        lines.append(f"| {name} | {row['parameters']} | {row['latency_mean_seconds']:.3f} | {row['latency_median_seconds']:.3f} | {row['word_count_mean']:.1f} | {row['visual_status']} |")
    for sid in map(str, SAMPLES):
        lines += ["", f"## Sample {sid}", ""]
        for name, row in records.items():
            value = row["samples"][sid]
            lines += [f"**{name}:** {value['caption']}", "", value["visual_observation"], ""]
    lines += ["## Recommendation", "", result["train_recommendation"], ""] + result["limitations"]
    atomic(root / "final-visual-review.md", "\n".join(lines) + "\n")
    paths = [target, root / "final-visual-review.md", Path(__file__).resolve()]
    atomic(root / "FINAL_REVIEW_SHA256SUMS", "".join(f"{sha(p)}  {p.resolve()}\n" for p in paths))
    # Original raw results and comparison must stay byte-identical.
    for name, expected in inputs.items():
        path = root / name if name in ("comparison.json", "protocol.json") else Path(name)
        if sha(path) != expected:
            raise RuntimeError("Input changed while finalizing review")
    print(f"JSON {target} SHA256 {sha(target)}")


if __name__ == "__main__":
    main()
