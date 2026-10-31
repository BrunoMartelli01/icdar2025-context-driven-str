"""Replay promoted prediction-only dictionary experts on every cached TEST image."""

import argparse
from pathlib import Path

from recognition.deepsolo_lexicon_deployment import DeepSoloVocabularyPostprocessor
from scripts.evaluate_deepsolo_icdar_lexicon import read, sha, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    args = parser.parse_args()
    c = read(args.config)
    root = Path(c["study_root"])
    mapping = read(root/"deployment_profiles.json")
    checks, pins, exports = {}, {}, {}
    for ds in ("icdar2015", "icdar13"):
        if ds == "icdar2015":
            folder = Path(c["ic15_source_study"])/"ic15_official1440/test"
            reports = read(root/"results.json")["datasets"][ds]["reports"]
            results_folder = root
            scenes = read(root/"icdar2015_test_observations.json")
        else:
            folder = Path(c["ic13_source_study"])/"fallback"
            results_folder = root/"ic13_dataset_v1"
            reports = read(results_folder/"results.json")["reports"]
            scenes = read(root/"icdar13_test_observations.json")
        checkpoint = read(folder/"protocol.json")["source_identity"]["checkpoint_sha256"]
        models, expected = {}, {}
        for preset in ("calibrated", "source"):
            path = Path(mapping[ds+"_"+preset])
            pins[str(path)] = sha(path)
            profile = read(path)
            assert profile["checkpoint_sha256"] == checkpoint
            models[preset] = DeepSoloVocabularyPostprocessor(profile)
            name = (ds+"__" if ds == "icdar2015" else "")+profile["report_key"]+"_per_image.json"
            expected[preset] = read(results_folder/name)
            assert sha(results_folder/name) == reports[profile["report_key"]]["per_image_sha256"]
        replays = 0
        for sid in scenes:
            path = folder/(sid.replace(":","__")+".json")
            raw = read(path)
            response = dict(raw["response"], artifact=dict(checkpoint_sha256=checkpoint))
            assert response["success"] and raw["contract"]["sample_id"] == sid
            for preset, model in models.items():
                predicted = model.apply(response)
                rows = expected[preset][sid]["predictions"]
                assert [(p["source_index"],p["text"]) for p in predicted["detections"]] == [(p["instance_id"],p["text"]) for p in rows]
                for d in predicted["detections"]:
                    source = next(p for p in raw["response"]["detections"] if p["source_index"]==d["source_index"])
                    assert d["geometry"] == source["geometry"] and d["raw"] == source["raw"]
                output = args.prediction_output/ds/preset/(sid.replace(":","__")+".json")
                output.parent.mkdir(parents=True,exist_ok=True)
                # Complete raw CTC probabilities remain in the authenticated
                # original shard; exported expert outputs contain no GT fields.
                payload = dict(schema_version=1, mode="approximate", claim_label="approximate reproduction",
                    sample_id=sid, expert="DeepSolo plus dataset Generic vocabulary", preset=preset,
                    checkpoint_sha256=checkpoint, profile_sha256=pins[mapping[ds+"_"+preset]],
                    raw_source_path=str(path), raw_source_sha256=sha(path),
                    detections=[{k:v for k,v in d.items() if k!="raw"} for d in predicted["detections"]],
                    recognized_strings=predicted["recognized_strings"],
                    lexicon_decisions=predicted["lexicon_decisions"], ground_truth_used=False)
                save(output,payload)
                exports[str(output)]=sha(output)
                replays += 1
        checks[ds] = dict(images=len(scenes), profiles=2, prediction_replays=replays)
    save(args.audit_output,dict(status="PASS",mode="approximate",claim_label="approximate reproduction",
        checks=checks,profile_sha256=pins,code_sha256=sha(__file__),
        prediction_outputs_sha256=exports,
        scope="1466 prediction-only expert/image replays; authenticated checkpoint identities, exact admitted IDs/text and unchanged original geometry/raw CTC. No GT input in the deployed corrector."))
    print(checks,flush=True)


if __name__ == "__main__":
    main()
