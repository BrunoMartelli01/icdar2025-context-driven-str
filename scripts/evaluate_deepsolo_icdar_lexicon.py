"""Calibrate and evaluate fixed DeepSolo outputs with explicit official vocabularies."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_word_f1 import apply_policy, recognition_confidence
from recognition.clip_router_evaluation import deepsolo_predictions, native_mask
from recognition.unet_corrected_token_interfaces import lexical_counts, summarize


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validated_json(path, pins):
    pins[str(path)] = sha(path)
    return read(path)


def prepare(c, root):
    pins = {}
    old = Path(c["ic13_source_study"])
    plan = validated_json(old / "resolved_configuration.json", pins)
    val = validated_json(old / "supervised_validation_plan.json", pins)
    populations = dict(validation=val["population"]["test"]["icdar13"],
                       test=plan["population"]["test"]["icdar13"])
    from scripts.run_essential_icdar_transfer import truth_and_validity
    from scripts.diagnose_parseq_candidate_ranking_train import _task22_word_texts
    from localize.practical_remap_protocol import train_task22_word_boxes
    from segmentation.data import icdar13_task22_rgb_to_target_and_validity
    for split, samples in populations.items():
        scenes = {}
        for s in samples:
            sid = s["sample_id"]
            if sha(s["image_path"]) != s["image_sha256"]:
                raise ValueError("Changed source RGB: " + sid)
            if split == "test":
                gt, valid, _ = truth_and_validity(s, "icdar13", None)
                pins[s["annotation_path"]] = s["annotation_sha256"]
            else:
                base = Path(c["icdar13_root"]) / "raw/train_segmentation_gt"
                identity = sid.split(":")[-1]
                annotation, mask = base / (identity + "_GT.txt"), base / (identity + "_GT.bmp")
                pins[str(annotation)], pins[str(mask)] = sha(annotation), sha(mask)
                words = _task22_word_texts(annotation)
                boxes = train_task22_word_boxes(mask, annotation) if words else []
                if len(words) != len(boxes):
                    raise ValueError("Canonical Task2.2 word/box mismatch")
                gt = [dict(text=t, bbox=list(b.bbox)) for t, b in zip(words, boxes)]
                with Image.open(mask) as image:
                    _, validity = icdar13_task22_rgb_to_target_and_validity(
                        image, annotation.read_text(encoding="utf-8-sig"))
                    valid = native_mask(validity, c["ignore_resolution"])
            path = old / ("fallback" if split == "test" else "fallback_validation") / (sid.replace(":", "__") + ".json")
            raw = validated_json(path, pins)
            content = {k: v for k, v in raw.items() if k != "payload_sha256"}
            if (raw["payload_sha256"] != hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
                    or raw["contract"]["sample_id"] != sid or raw["contract"]["image_sha256"] != s["image_sha256"]
                    or not raw["response"]["success"]):
                raise ValueError("Changed DeepSolo payload")
            rows = deepsolo_predictions(raw["response"], s["image_wh"], s["orientation"],
                                       valid, c["raw_inference_floor"])
            by = {d["source_index"]: d for d in raw["response"]["detections"]}
            for row in rows:
                row["recognition_confidence"] = recognition_confidence(
                    by[row["instance_id"]]["raw"]["recognition_class_probabilities"])
            scenes[sid] = dict(predictions=rows, ground_truth=gt)
        save(root / ("icdar13_" + split + "_observations.json"), scenes)
    ic15 = Path(c["ic15_source_study"])
    observations = validated_json(ic15 / "calibration_observations.json", pins)["ic15_official1440"]
    save(root / "icdar2015_validation_observations.json",
         {sid: dict(predictions=v[0], ground_truth=v[1]) for sid, v in observations.items()})
    # Fixed source filter .30/no-character-prune contains every stricter candidate.
    source = validated_json(ic15 / "ic15_official1440__upstream_ic15_filter_per_image.json", pins)
    save(root / "icdar2015_test_observations.json",
         {sid: dict(predictions=v["selected_predictions"], ground_truth=v["ground_truth"]) for sid, v in source.items()})
    save(root / "observation_inputs.json", pins)


def policy_neural(values):
    return dict(values, resolution="native", decoder="greedy", same_text_nms_iou=None)


def evaluate(scenes, neural, lexicon, unmatched):
    per_image = {}
    for sid, scene in scenes.items():
        rows = apply_policy(scene["predictions"], policy_neural(neural))
        trace = []
        if lexicon is not None:
            selected = lexicon[sid] if isinstance(lexicon, dict) else lexicon
            rows, trace = selected.correct(rows, unmatched_policy=unmatched)
        counts = lexical_counts([dict(raw_text=p["text"] if p.get("invalid_envelope") is None else "")
                                 for p in rows], scene["ground_truth"], [p["ignored"] for p in rows])
        per_image[sid] = dict(counts=counts, predictions=rows, decisions=trace,
                             ground_truth=scene["ground_truth"])
    metrics = summarize(v["counts"] for v in per_image.values())
    metrics.update(deepsolo_images=len(scenes), total_images=len(scenes), deepsolo_percent=100.,
                   inference_scope="Cached standalone DeepSolo plus CPU lexicon; no router or new neural inference")
    return metrics, per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "calibrate", "evaluate"), required=True)
    args = parser.parse_args()
    c = read(args.config)
    if c["mode"] != "approximate" or c["claim_label"] != "approximate reproduction":
        raise ValueError("Explicit approximate mode required")
    root = Path(c["study_root"])
    for path, digest in [(c["protocol_path"], c["protocol_sha256"]),
                         (c["lexicon_manifest"], c["lexicon_manifest_sha256"]),
                         (c["pinned_source"], c["pinned_source_sha256"])]:
        if sha(path) != digest:
            raise ValueError("Changed pinned lexicon input: " + str(path))
    manifest = read(c["lexicon_manifest"])
    for name, digest in manifest.items():
        if sha(Path(c["lexicon_root"]).parent / name) != digest:
            raise ValueError("Changed vocabulary/pair list: " + name)
    if args.stage == "prepare":
        prepare(c, root)
        return
    protocol = read(c["protocol_path"])
    lexical = Path(c["lexicon_root"])
    generic = DeepSoloLexicon.from_files(lexical / "GenericVocabulary_new.txt",
        lexical / "GenericVocabulary_pair_list.txt", distance_exclusive=c["source_match_distance_exclusive"])
    if args.stage == "calibrate":
        grids, choices, inputs = {}, {}, {}
        for ds, candidates in protocol["confidence_candidates"].items():
            path = root / (ds + "_validation_observations.json")
            scenes = read(path)
            inputs[str(path)] = sha(path)
            grid = []
            for name, neural in candidates.items():
                for unmatched in ("source_reject", "keep_unmatched"):
                    metrics, _ = evaluate(scenes, neural, generic, unmatched)
                    grid.append(dict(confidence_name=name, neural=neural,
                                     unmatched_policy=unmatched, metrics=metrics))
            def key(v):
                m = v["metrics"]
                return (Fraction(2*m["tp"], m["detected_words"]+m["gt_words"]),
                        m["tp"], v["unmatched_policy"] == "keep_unmatched",
                        -list(candidates).index(v["confidence_name"]))
            choices[ds] = max(grid, key=key)
            grids[ds] = grid
            print(ds, "CAL", choices[ds], flush=True)
        save(root / "calibration_grid.json", grids)
        save(root / "frozen_choices.json", dict(status="FROZEN", test_used=False,
            config_sha256=sha(args.config), observation_sha256=inputs,
            choices=choices, grid_sha256=sha(root / "calibration_grid.json"),
            code_sha256=sha(__file__), model_code_sha256=sha(Path(__file__).parents[1] / "recognition/deepsolo_lexicon.py")))
        return
    frozen = read(root / "frozen_choices.json")
    if frozen["config_sha256"] != sha(args.config) or frozen["test_used"] is not False:
        raise ValueError("Changed frozen calibration configuration")
    results = {}
    for ds, candidates in protocol["confidence_candidates"].items():
        scenes = read(root / (ds + "_test_observations.json"))
        reports = {}
        for name, neural in candidates.items():
            arms = [("None", None, None), ("Generic_source_reject", generic, "source_reject"),
                    ("Generic_keep_unmatched", generic, "keep_unmatched")]
            if ds == "icdar2015":
                weak = DeepSoloLexicon.from_files(lexical / "ch4_test_vocabulary_new.txt",
                    lexical / "ch4_test_vocabulary_pair_list.txt", distance_exclusive=c["source_match_distance_exclusive"])
                strong = {sid: DeepSoloLexicon.from_files(
                    lexical / "new_strong_lexicon" / ("new_voc_img_" + sid.split(":")[-1] + ".txt"),
                    lexical / "new_strong_lexicon" / ("pair_voc_img_" + sid.split(":")[-1] + ".txt"),
                    distance_exclusive=c["source_match_distance_exclusive"]) for sid in scenes}
                arms += [("Weak_source_reject", weak, "source_reject"),
                         ("Strong_source_reject", strong, "source_reject")]
            for setting, lexicon, unmatched in arms:
                metrics, records = evaluate(scenes, neural, lexicon, unmatched)
                if (name == next(iter(candidates)) and setting == "None" and
                        (metrics["total_images"] != c["expected_test"][ds]["images"] or
                         metrics["gt_words"] != c["expected_test"][ds]["care_words"] or
                         metrics["f1"] != c["expected_test"][ds]["baseline_f1"])):
                    raise ValueError("Historical unchanged-metric baseline replay failed")
                key = name + "__" + setting
                path = root / (ds + "__" + key + "_per_image.json")
                save(path, records)
                reports[key] = dict(metrics=metrics, neural=neural, unmatched_policy=unmatched,
                                    per_image_sha256=sha(path), setting=setting)
                print(ds, key, metrics["f1"], metrics["tp"], flush=True)
        selected = frozen["choices"][ds]
        primary_key = selected["confidence_name"] + "__Generic_" + selected["unmatched_policy"]
        results[ds] = dict(primary_key=primary_key, primary=reports[primary_key], reports=reports,
                           target90_achieved=reports[primary_key]["metrics"]["f1"] >= 0.90 if ds == "icdar13" else None)
    save(root / "results.json", dict(status="PASS", mode="approximate", claim_label="approximate reproduction",
         datasets=results, frozen_choices_sha256=sha(root / "frozen_choices.json"),
         code_sha256=sha(__file__), zero_new_gpu_inferences=True,
         metric=c["word_metric"], lexicon_inputs_sha256=c["lexicon_manifest_sha256"],
         limits="Weak/Strong use dataset/image-specific benchmark vocabularies; Generic is the standalone operational comparison. IC13 uses shared IC15 general English lexicon as approximate transfer."))


if __name__ == "__main__":
    main()
