"""Correct IC13-specific released lexicons; fixed DeepSolo, full233-image word F1."""

import argparse
from fractions import Fraction
from pathlib import Path

from recognition.deepsolo_lexicon import DeepSoloLexicon
from scripts.evaluate_deepsolo_icdar_lexicon import read, sha, save, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    c = read(args.config)
    root, out = Path(c["study_root"]), Path(c["output_root"])
    if c["mode"] != "approximate" or c["claim_label"] != "approximate reproduction":
        raise ValueError("Explicit approximate mode required")
    if sha(c["ic13_lexicon_manifest"]) != c["ic13_lexicon_manifest_sha256"]:
        raise ValueError("Changed IC13 resource manifest")
    lexical = Path(c["ic13_lexicon_root"])
    for name, digest in read(c["ic13_lexicon_manifest"]).items():
        if sha(lexical.parents[1] / name) != digest:
            raise ValueError("Changed released IC13 vocabulary")
    generic = DeepSoloLexicon.from_files(lexical / "GenericVocabulary_new.txt",
        lexical / "GenericVocabulary_pair_list.txt", distance_exclusive=c["source_match_distance_exclusive"])
    candidates = read(c["protocol_path"])["confidence_candidates"]["icdar13"]
    if args.stage == "calibrate":
        scenes = read(root / "icdar13_validation_observations.json")
        grid = []
        for name, neural in candidates.items():
            for unmatched in ("source_reject", "keep_unmatched"):
                metrics, _ = evaluate(scenes, neural, generic, unmatched)
                grid.append(dict(confidence_name=name, neural=neural,
                                 unmatched_policy=unmatched, metrics=metrics))
        def key(v):
            m = v["metrics"]
            return (Fraction(2*m["tp"], m["detected_words"]+m["gt_words"]), m["tp"],
                    v["unmatched_policy"] == "keep_unmatched", -list(candidates).index(v["confidence_name"]))
        choice = max(grid, key=key)
        save(out / "calibration_grid.json", grid)
        save(out / "frozen_choices.json", dict(test_used=False, choice=choice,
            config_sha256=sha(args.config), validation_observations_sha256=sha(root / "icdar13_validation_observations.json"),
            lexicon_manifest_sha256=c["ic13_lexicon_manifest_sha256"], code_sha256=sha(__file__)))
        print("IC13_DATASET_CAL", choice, flush=True)
        return
    frozen = read(out / "frozen_choices.json")
    if frozen["config_sha256"] != sha(args.config) or frozen["test_used"] is not False:
        raise ValueError("Changed calibration choice")
    scenes = read(root / "icdar13_test_observations.json")
    weak = DeepSoloLexicon.from_files(lexical / "ch2_test_vocabulary_new.txt",
        lexical / "ch2_test_vocabulary_pair_list.txt", distance_exclusive=c["source_match_distance_exclusive"])
    strong = {sid: DeepSoloLexicon.from_files(
        lexical / "new_strong_lexicon" / ("new_voc_img_" + sid.split(":")[-1] + ".txt"),
        lexical / "new_strong_lexicon" / ("pair_voc_img_" + sid.split(":")[-1] + ".txt"),
        distance_exclusive=c["source_match_distance_exclusive"]) for sid in scenes}
    reports = {}
    for name, neural in candidates.items():
        for label, lexicon, unmatched in [("None", None, None), ("Generic_source_reject", generic, "source_reject"),
            ("Generic_keep_unmatched", generic, "keep_unmatched"), ("Weak_source_reject", weak, "source_reject"),
            ("Strong_source_reject", strong, "source_reject")]:
            metrics, records = evaluate(scenes, neural, lexicon, unmatched)
            if metrics["total_images"] != 233 or metrics["gt_words"] != 1095:
                raise ValueError("Changed full IC13 care population")
            key = name + "__" + label
            path = out / (key + "_per_image.json")
            save(path, records)
            reports[key] = dict(metrics=metrics, neural=neural, setting=label,
                                 unmatched_policy=unmatched, per_image_sha256=sha(path))
            print(key, metrics, flush=True)
    choice = frozen["choice"]
    primary_key = choice["confidence_name"] + "__Generic_" + choice["unmatched_policy"]
    save(out / "results.json", dict(status="PASS", mode="approximate", claim_label="approximate reproduction",
        primary_key=primary_key, primary=reports[primary_key], reports=reports,
        target90_achieved=reports[primary_key]["metrics"]["f1"] >= .90,
        frozen_sha256=sha(out / "frozen_choices.json"), code_sha256=sha(__file__),
        scope="Released IC13-specific vocabularies. Same full1095-word metric and frozen DeepSolo. Weak/Strong contain benchmark-specific vocabulary information; no other OCR model."))


if __name__ == "__main__":
    main()
