"""Frozen-source/schema tests; fixture reports do not represent model performance."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.evaluate_totaltext_bert_correction import (
    digest, evaluate, infer, load_captions, load_config, matched_gt,
    payload_digest, prepare, prepared_inputs, read_json, seal_json, shard_path,
    source_inputs, verify_shard,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/approximate/totaltext_v4_bert_correction_v1.json"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def prepared(tmp_path):
    cfg = read_json(CONFIG)
    root = tmp_path / "source"
    sample_id = "total_text:14"
    crop_sha = "b" * 64
    protocol = {"dataset": "total_text", "split": cfg["source"]["split"],
                "parseq_checkpoint_sha256": cfg["source"]["parseq_checkpoint_sha256"],
                "parseq_runtime_archive_sha256": cfg["source"]["parseq_source_archive_sha256"]}
    write(root / "protocol.json", protocol)
    mask = root / "word_f1_control/masks/total_text__14.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"fixture-mask-checksum")
    gt = [{"bbox": [0, 0, 20, 10], "text": "parking"},
          {"bbox": [100, 100, 120, 110], "text": "missing"}]
    proposals = [{"rank": 1, "bbox": [0, 0, 20, 10], "crop_rgb_sha256": crop_sha}]
    sample = {"sample_id": sample_id, "image_sha256": "a" * 64,
              "protocol_sha256": sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              "checkpoint_sha256": cfg["source"]["segmentation_checkpoint_sha256"],
              "ground_truth": gt, "proposals": proposals, "native_mask_file_sha256": digest(mask)}
    write(root / "word_f1_control/samples/total_text__14.json", sample)
    ocr = {"crop_rgb_sha256": crop_sha, "raw_text": "parkin", "minimum_token_probability": .5,
           "parseq_checkpoint_sha256": cfg["source"]["parseq_checkpoint_sha256"],
           "parseq_source_archive_sha256": cfg["source"]["parseq_source_archive_sha256"]}
    write(root / "ocr" / (crop_sha + ".json"), ocr)
    report = {"images": 1, "protocol": protocol, "variants": {"word_f1_control": {
        "checkpoint_sha256": cfg["source"]["segmentation_checkpoint_sha256"],
        "aggregate": {"gt_words": 2, "proposals": 1, "geometry_iou_ge_25": 1,
                      "geometry_iou_ge_50": 1, "geometry_iou_ge_75": 1,
                      "exact_casefold_iou_ge_50": 0},
        "per_image": [{"sample_id": sample_id, "ground_truth": gt,
                       "predictions": [{"bbox": [0, 0, 20, 10], "crop_rgb_sha256": crop_sha,
                                        "text": "parkin"}]}]}}}
    write(root / "results.json", report)
    cfg["source"].update(results_sha256=digest(root / "results.json"),
             protocol_sha256=digest(root / "protocol.json"), expected_images=1,
             expected_care_words=2, expected_proposals=1)
    config = tmp_path / "configs/approximate/fixture.json"
    write(config, cfg)
    args = argparse.Namespace(config=config, source_dir=root, output_dir=tmp_path / "out", captions=None, reuse_ocr=[])
    prepare(args)
    return args


def test_config_isolation_and_immutable_revision(tmp_path):
    cfg = read_json(CONFIG)
    assert load_config(CONFIG)["reproduction_mode"] == "approximate"
    path = tmp_path / "configs/strict/fixture.json"
    write(path, cfg)
    with pytest.raises(ValueError, match="configs/approximate"):
        load_config(path)
    path = tmp_path / "configs/approximate/fixture.json"
    cfg["bert"]["revision"] = "main"
    write(path, cfg)
    with pytest.raises(ValueError, match="immutable"):
        load_config(path)


def test_preparation_is_idempotent_gt_is_separate_and_denominator_preserved(prepared):
    out = prepared.output_dir
    before = {p.name: digest(p) for p in out.glob("*.json")}
    prepare(prepared)
    assert before == {p.name: digest(p) for p in out.glob("*.json")}
    inputs = read_json(out / "inputs.json")
    assert "ground_truth" not in json.dumps(inputs)
    assert "parking" not in json.dumps(inputs)
    baseline = read_json(out / "baseline.json")["metrics"]
    assert baseline["geometry"]["gt_instances"] == 2
    assert baseline["recognition"]["exact_casefold_spatial_one_to_one"] == 0


def test_changed_source_text_and_mask_are_rejected(prepared):
    cfg = load_config(prepared.config)
    path = next((prepared.source_dir / "ocr").glob("*.json"))
    original = read_json(path)
    changed = {**original, "raw_text": "parking"}
    write(path, changed)
    with pytest.raises(ValueError, match="checkpoint/text"):
        source_inputs(prepared.source_dir, cfg)
    write(path, original)
    mask = next((prepared.source_dir / "word_f1_control/masks").iterdir())
    mask.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mask checksum"):
        source_inputs(prepared.source_dir, cfg)


def test_missing_ocr_and_conflicting_sealed_outputs_fail(prepared):
    cfg = load_config(prepared.config)
    path = next((prepared.source_dir / "ocr").glob("*.json"))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        source_inputs(prepared.source_dir, cfg)
    with pytest.raises(ValueError, match="Conflicting sealed"):
        seal_json(prepared.output_dir / "inputs.json", {"different": True})


def test_explicit_historical_ocr_cache_is_verified_and_conflicts_fail(prepared, tmp_path):
    cfg = load_config(prepared.config)
    path = next((prepared.source_dir / "ocr").glob("*.json"))
    reused = tmp_path / "previous-ocr" / path.name
    write(reused, read_json(path))
    path.unlink()
    samples, _, _ = source_inputs(prepared.source_dir, cfg, [reused.parent])
    assert samples[0]["blocks"][0]["text"] == "parkin"
    other = {**read_json(reused), "minimum_token_probability": .99}
    write(path, other)
    with pytest.raises(ValueError, match="Conflicting PARSeq"):
        source_inputs(prepared.source_dir, cfg, [reused.parent])


def test_changed_inputs_and_code_hash_fail_closed(prepared):
    out = prepared.output_dir
    inputs = read_json(out / "inputs.json")
    inputs["samples"][0]["blocks"][0]["text"] = "parking"
    write(out / "inputs.json", inputs)
    with pytest.raises(ValueError, match="checksum"):
        prepared_inputs(out)
    p = read_json(out / "protocol.json")
    p["implementation_sha256"] = {"wrong": "0" * 64}
    write(out / "protocol.json", p)
    prepared_record = read_json(out / "prepared.json")
    prepared_record["protocol_sha256"] = digest(out / "protocol.json")
    write(out / "prepared.json", prepared_record)
    with pytest.raises(ValueError, match="Implementation changed"):
        prepared_inputs(out)


def test_wrong_dataset_caption_binding_is_rejected(prepared, tmp_path):
    sample = read_json(prepared.output_dir / "inputs.json")["samples"][0]
    path = tmp_path / "captions.json"
    obj = {"schema_version": 1, "model_id": "caption-model", "revision": "1" * 40,
           "generation": {"num_beams": 1}, "model_files_sha256": {"weights": "2" * 64},
           "samples": [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                        "text": "A parking sign."}]}
    write(path, obj)
    assert load_captions(path, [sample]) == {sample["sample_id"]: "A parking sign."}
    obj["samples"][0]["image_sha256"] = "f" * 64
    write(path, obj)
    with pytest.raises(ValueError, match="identity/checksum"):
        load_captions(path, [sample])
    obj["samples"][0]["sample_id"] = "icdar13:14"
    write(path, obj)
    with pytest.raises(ValueError, match="identity/checksum"):
        load_captions(path, [sample])


def test_infer_missing_weights_is_actionable_and_does_not_read_gt(prepared, tmp_path):
    (prepared.output_dir / "evaluation_labels.json").unlink()
    (prepared.output_dir / "baseline.json").unlink()
    args = argparse.Namespace(output_dir=prepared.output_dir, hf_cache=tmp_path / "missing-cache", device="cpu")
    with pytest.raises(FileNotFoundError, match="Run fetch-model"):
        infer(args)


def test_matching_is_one_to_one_even_for_duplicate_predictions():
    gt = [{"bbox": [0, 0, 10, 10], "text": "cat"}]
    predictions = [{"bbox": [0, 0, 10, 10], "text": "CAT"}] * 2
    assert matched_gt(predictions, gt) == {0}


def test_evaluation_reconciles_recovery_and_audits_corrupt_shards(prepared):
    out = prepared.output_dir
    protocol, record, inputs = prepared_inputs(out)
    seal_json(out / "runtime.json", {"inputs_sha256": record["inputs_sha256"], "fixture": True})
    runtime_sha = digest(out / "runtime.json")
    sample = inputs["samples"][0]
    block = sample["blocks"][0]
    candidates = [{"text": "parkin", "source": "PARSeq_KEEP", "edit_distance": 0},
                  {"text": "parking", "source": "BERT_VOCABULARY", "edit_distance": 1}]
    scores = [{"mean_log_probability": -10}, {"mean_log_probability": -1}]
    from recognition.bert_correction import CorrectionPolicy, choose_candidate
    decision = choose_candidate(candidates, scores, CorrectionPolicy(**protocol["resolved_config"]["policy"]))
    for condition in protocol["conditions"]:
        trace = {"rank": block["rank"], "bbox": block["bbox"], "crop_rgb_sha256": block["crop_rgb_sha256"],
                 "original_text": block["text"], "text": "parking", "candidates": candidates,
                 "scores": scores, "decision": decision,
                 "parseq_minimum_token_probability": block["minimum_token_probability"]}
        row = {"sample_id": sample["sample_id"], "condition": condition,
               "image_sha256": sample["image_sha256"], "inputs_sha256": record["inputs_sha256"],
               "runtime_sha256": runtime_sha, "trace": [trace], "correction_seconds": 0.,
               "peak_allocated_bytes": None, "peak_reserved_bytes": None}
        row["payload_sha256"] = payload_digest(row)
        seal_json(shard_path(out, condition, sample["sample_id"]), row)
    evaluate(argparse.Namespace(output_dir=out))
    report = read_json(out / "results.json")
    for result in report["conditions"].values():
        assert result["recovered"] == 1 and result["damaged"] == 0 and result["net"] == 1
        assert result["metrics"]["recognition"]["exact_casefold_rate_all_gt"] == .5
    path = shard_path(out, "word_only", sample["sample_id"])
    row = read_json(path)
    row["trace"][0]["text"] = "injected"
    write(path, row)
    with pytest.raises(ValueError, match="shard checksum"):
        verify_shard(path, sample, "word_only", {**record, "policy": protocol["resolved_config"]["policy"]}, runtime_sha)


def test_genuine_tiny_bert_offline_infer_evaluate_and_resume(prepared, tmp_path):
    """Random test-only checkpoint, with the real BERT pretraining/MLM heads."""
    import torch
    from transformers import BertConfig, BertForPreTraining, BertTokenizerFast
    from scripts.evaluate_totaltext_bert_correction import snapshot_path

    cfg = read_json(prepared.config)
    cfg["bert"].update(model_id="fixture/tiny-bert", revision="f" * 40, max_input_tokens=32)
    write(prepared.config, cfg)
    prepared.output_dir = tmp_path / "tiny-bert-output"
    prepare(prepared)
    cache = tmp_path / "test-only-model-cache"
    snapshot = snapshot_path(cfg, cache)
    snapshot.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "parkin", "parking", "the", "is"]
    (snapshot / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(snapshot / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(snapshot)
    torch.manual_seed(123)
    model = BertForPreTraining(BertConfig(vocab_size=len(vocab), hidden_size=16,
                  num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                  max_position_embeddings=64))
    model.save_pretrained(snapshot, safe_serialization=True)
    args = argparse.Namespace(output_dir=prepared.output_dir, hf_cache=cache, device="cpu")
    infer(args)
    evaluate(args)
    runtime = read_json(prepared.output_dir / "runtime.json")
    assert runtime["model_id"] == "fixture/tiny-bert"
    assert not runtime["checkpoint_loading"]["missing_keys"]
    assert runtime["checkpoint_loading"]["unexpected_keys"]
    assert set(runtime["checkpoint_loading"]["unexpected_keys"]) <= set(cfg["bert"]["allowed_unused_checkpoint_keys"])
    assert runtime["device"] == "cpu" and runtime["precision"] == "float32"
    report = read_json(prepared.output_dir / "results.json")
    assert report["test_loaded"] is False and report["training_performed"] is False
    assert report["conditions"]["word_only"]["metrics"]["geometry"] == report["baseline"]["geometry"]
    files = list((prepared.output_dir / "inference").rglob("*.json"))
    before = {str(p): digest(p) for p in files}
    infer(args)
    assert before == {str(p): digest(p) for p in files}
