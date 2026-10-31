"""Calibration-frozen generic-lexicon guards on unchanged ICDAR word F1."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_lexicon_guard import correct_guarded
from recognition.deepsolo_word_f1 import apply_policy
from recognition.unet_corrected_token_interfaces import lexical_counts, summarize
from scripts.evaluate_deepsolo_icdar_lexicon import read, sha, save, policy_neural


def evaluate(scenes, neural, lexicon, guard):
    records = {}
    changed = 0
    for sid, scene in scenes.items():
        original = apply_policy(scene["predictions"], policy_neural(neural))
        rows, decisions = correct_guarded(original, lexicon, **guard)
        changed += sum(d["original_text"].casefold() != d["corrected_text"].casefold() for d in decisions)
        counts = lexical_counts([dict(raw_text=p["text"] if p.get("invalid_envelope") is None else "")
                                 for p in rows], scene["ground_truth"], [p["ignored"] for p in rows])
        records[sid] = dict(counts=counts, predictions=rows, decisions=decisions,
                            ground_truth=scene["ground_truth"])
    m = summarize(r["counts"] for r in records.values())
    m.update(deepsolo_images=len(scenes), total_images=len(scenes), deepsolo_percent=100.,
             changed_predictions=changed, inference_scope="Cached standalone DeepSolo plus guarded Generic dictionary")
    return m, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    c = read(args.config)
    root = Path(c["study_root"])
    protocol = read(root / "guard_predeclared.json")
    base = read(c["protocol_path"])
    lexical = Path(c["lexicon_root"])
    model = DeepSoloLexicon.from_files(lexical / "GenericVocabulary_new.txt",
        lexical / "GenericVocabulary_pair_list.txt", distance_exclusive=c["source_match_distance_exclusive"])
    if args.stage == "calibrate":
        grids, chosen = {}, {}
        for ds, candidates in base["confidence_candidates"].items():
            scenes = read(root / (ds + "_validation_observations.json"))
            grid = []
            for name, neural in candidates.items():
                for cap in protocol["maximum_recognition_confidences"]:
                    for unique in protocol["require_unique_nearest"]:
                        guard = dict(min_length=protocol["min_correction_length"],
                                     preserve_digits=protocol["preserve_digit_strings"],
                                     max_confidence=cap, require_unique=unique)
                        metrics, _ = evaluate(scenes, neural, model, guard)
                        grid.append(dict(confidence_name=name, neural=neural, guard=guard, metrics=metrics))
            def objective(v):
                m = v["metrics"]
                return (Fraction(2*m["tp"], m["detected_words"]+m["gt_words"]), m["tp"],
                        -m["changed_predictions"], -v["guard"]["max_confidence"],
                        v["guard"]["require_unique"], -list(candidates).index(v["confidence_name"]))
            chosen[ds] = max(grid, key=objective)
            grids[ds] = grid
            print(ds, "GUARDED_CAL", chosen[ds], flush=True)
        save(root / "guard_calibration_grid.json", grids)
        save(root / "guard_frozen_choices.json", dict(test_used=False, choices=chosen,
            config_sha256=sha(args.config), protocol_sha256=sha(root / "guard_predeclared.json"),
            code_sha256=sha(__file__), model_code_sha256=sha(Path(__file__).parents[1]/"recognition/deepsolo_lexicon_guard.py")))
        return
    frozen = read(root / "guard_frozen_choices.json")
    assert frozen["test_used"] is False and frozen["protocol_sha256"] == sha(root / "guard_predeclared.json")
    reports = {}
    for ds, choice in frozen["choices"].items():
        scenes = read(root / (ds + "_test_observations.json"))
        metrics, records = evaluate(scenes, choice["neural"], model, choice["guard"])
        path = root / (ds + "__guarded_generic_per_image.json")
        save(path, records)
        reports[ds] = dict(metrics=metrics, neural=choice["neural"], guard=choice["guard"],
                           per_image_sha256=sha(path), target90_achieved=metrics["f1"] >= .90 if ds == "icdar13" else None)
        print(ds, "GUARDED_TEST", metrics, flush=True)
    save(root / "guard_results.json", dict(status="PASS", mode="approximate",
        claim_label="approximate reproduction", datasets=reports,
        frozen_sha256=sha(root / "guard_frozen_choices.json"), code_sha256=sha(__file__),
        scope="Exploratory guard follow-up; calibration-only choices. Same DeepSolo outputs and full denominators; no other OCR, GT-derived dictionary or new GPU inference."))


if __name__ == "__main__":
    main()
