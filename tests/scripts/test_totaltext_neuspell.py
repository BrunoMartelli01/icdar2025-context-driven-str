"""Frozen-source and model boundary tests; synthetic weights imply no accuracy claim."""

import argparse
import json
import pickle
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

import scripts.evaluate_totaltext_neuspell as runner
from recognition.neuspell_correction import convert_vocabulary, git_blob_digest


CONFIG = Path(__file__).resolve().parents[2] / "configs/approximate/totaltext_letterbox_neuspell_correction_v1.json"
CONFIG_V2 = CONFIG.with_name("totaltext_letterbox_neuspell_correction_v2.json")


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def experiment(tmp_path):
    cfg = runner.read_json(CONFIG)
    cfg["model"].update(model_id="fixture/tiny-spelling", revision="f" * 40,
                         label_count=6, saved_top_k=6, max_input_tokens=32)
    cache, assets = tmp_path / "cache", tmp_path / "assets"
    snapshot = runner.snapshot_path(cfg, cache)
    snapshot.mkdir(parents=True)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "park", "##in", "parking", "the", "sign"]
    (snapshot / "vocab.txt").write_text("\n".join(tokens) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(snapshot / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(snapshot)
    model = BertForTokenClassification(BertConfig(vocab_size=len(tokens), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=64, num_labels=6))
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[2] = 8
    model.config.save_pretrained(snapshot)
    torch.save(model.state_dict(), snapshot / "pytorch_model.bin")
    labels = ["<<PAD>>", "<<UNK>>", "parking", "PARKING", "hotel", "parkin"]
    raw = pickle.dumps({"idx2token": dict(enumerate(labels)),
        "token2idx": {v: i for i, v in enumerate(labels)}, "pad_token": labels[0], "unk_token": labels[1]})
    runner.atomic_bytes(assets / "vocab.pkl", raw)
    runner.atomic_bytes(assets / "author_snippet.py", b"# test-only source fixture\n")
    cfg["upstream_neuspell"].update(vocabulary_git_blob=git_blob_digest(raw),
        snippet_git_blob=git_blob_digest((assets / "author_snippet.py").read_bytes()))
    vocabulary = convert_vocabulary(assets / "vocab.pkl", expected_blob=git_blob_digest(raw), expected_labels=6)
    runner.seal_json(assets / "vocabulary.json", vocabulary)
    manifest = {"model_id": cfg["model"]["model_id"], "model_revision": cfg["model"]["revision"],
        "upstream_neuspell": cfg["upstream_neuspell"],
        "authoritative_weight_sha256": runner.digest(snapshot / "pytorch_model.bin"),
        "model_files_sha256": {n: runner.digest(snapshot / n) for n in runner.MODEL_FILES},
        "auxiliary_files_sha256": {n: runner.digest(assets / n) for n in runner.AUXILIARY_FILES},
        "test_only_synthetic_assets": True}
    runner.seal_json(assets / "assets.json", manifest)
    root = tmp_path / "source"
    source_protocol = {"dataset": "total_text", "split": cfg["source"]["split"],
        "parseq_checkpoint_sha256": cfg["source"]["parseq_checkpoint_sha256"],
        "parseq_runtime_archive_sha256": cfg["source"]["parseq_source_archive_sha256"]}
    write(root / "protocol.json", source_protocol)
    mask = root / "word_f1_control/masks/total_text__14.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"fixture-mask")
    gt = [{"bbox": [0, 0, 20, 10], "text": "parking"},
          {"bbox": [100, 100, 120, 110], "text": "missing"}]
    crop = "b" * 64
    write(root / "word_f1_control/samples/total_text__14.json", {
        "sample_id": "total_text:14", "image_sha256": "a" * 64,
        "protocol_sha256": sha256(json.dumps(source_protocol, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": cfg["source"]["segmentation_checkpoint_sha256"], "ground_truth": gt,
        "proposals": [{"rank": 1, "bbox": [0, 0, 20, 10], "crop_rgb_sha256": crop}],
        "native_mask_file_sha256": runner.digest(mask)})
    write(root / "ocr" / (crop + ".json"), {"crop_rgb_sha256": crop, "raw_text": "parkin",
        "minimum_token_probability": .99,
        "parseq_checkpoint_sha256": cfg["source"]["parseq_checkpoint_sha256"],
        "parseq_source_archive_sha256": cfg["source"]["parseq_source_archive_sha256"]})
    write(root / "results.json", {"images": 1, "protocol": source_protocol, "variants": {"word_f1_control": {
        "checkpoint_sha256": cfg["source"]["segmentation_checkpoint_sha256"],
        "aggregate": {"gt_words": 2, "proposals": 1, "geometry_iou_ge_25": 1,
            "geometry_iou_ge_50": 1, "geometry_iou_ge_75": 1, "exact_casefold_iou_ge_50": 0},
        "per_image": [{"sample_id": "total_text:14", "ground_truth": gt,
            "predictions": [{"bbox": [0, 0, 20, 10], "crop_rgb_sha256": crop, "text": "parkin"}]}]}}})
    cfg["source"].update(expected_images=1, expected_care_words=2, expected_proposals=1,
        results_sha256=runner.digest(root / "results.json"), protocol_sha256=runner.digest(root / "protocol.json"))
    cfg_path = tmp_path / "configs/approximate/fixture.json"
    write(cfg_path, cfg)
    args = argparse.Namespace(config=cfg_path, source_dir=root, output_dir=tmp_path / "out",
        reuse_ocr=[], captions=None, hf_cache=cache, asset_dir=assets, device="cpu")
    runner.prepare(args)
    return args


def test_configuration_namespace_and_mutable_revision_are_rejected(tmp_path):
    cfg = runner.read_json(CONFIG)
    assert runner.config(CONFIG)["model"]["label_count"] == 100002
    assert runner.config(CONFIG)["model"]["tokenizer_do_lower_case"] is False
    v2 = runner.config(CONFIG_V2)
    assert v2["model"]["tokenizer_do_lower_case"] is True
    assert v2["supersedes_config_sha256"] == runner.digest(CONFIG)
    path = tmp_path / "configs/strict/fixture.json"
    write(path, cfg)
    with pytest.raises(ValueError, match="configs/approximate"):
        runner.config(path)
    path = tmp_path / "configs/approximate/fixture.json"
    cfg["model"]["revision"] = "main"
    write(path, cfg)
    with pytest.raises(ValueError, match="pinned"):
        runner.config(path)


def test_real_bert_word_head_offline_end_to_end_and_resume_without_gt(experiment):
    out = experiment.output_dir
    before = {p.name: runner.digest(p) for p in out.glob("*.json")}
    runner.prepare(experiment)
    assert before == {p.name: runner.digest(p) for p in out.glob("*.json")}
    gt = (out / "evaluation_labels.json").read_bytes()
    baseline = (out / "baseline.json").read_bytes()
    (out / "evaluation_labels.json").unlink()
    (out / "baseline.json").unlink()
    runner.infer(experiment)  # GT files do not exist during neural inference.
    runner.atomic_bytes(out / "evaluation_labels.json", gt)
    runner.atomic_bytes(out / "baseline.json", baseline)
    runner.evaluate(experiment)
    result = runner.read_json(out / "results.json")
    assert not result["test_loaded"] and not result["training_performed"]
    for arm in result["conditions"].values():
        assert arm["recovered"] == 1 and arm["damaged"] == 0 and arm["net"] == 1
        assert arm["metrics"]["recognition"]["exact_casefold_rate_all_gt"] == .5
        assert arm["metrics"]["geometry"] == result["baseline"]["geometry"]
    files = list((out / "inference").rglob("*.json"))
    before = {str(p): runner.digest(p) for p in files}
    runner.infer(experiment)
    assert before == {str(p): runner.digest(p) for p in files}
    runtime = runner.read_json(out / "runtime.json")
    assert runtime["loading_audit"]["load_state_dict_strict"]
    assert not runtime["loading_audit"]["missing_keys"]
    assert "ground_truth" not in json.dumps(runner.read_json(out / "inputs.json"))


def test_missing_assets_is_actionable_without_gt(experiment):
    (experiment.output_dir / "evaluation_labels.json").unlink()
    (experiment.output_dir / "baseline.json").unlink()
    experiment.asset_dir = experiment.asset_dir / "missing"
    with pytest.raises(FileNotFoundError, match="fetch-model"):
        runner.infer(experiment)


@pytest.mark.parametrize("expected_lower_case", [True, False])
def test_legacy_tokenizer_omitted_flag_resolves_default_and_mismatch_stays_fatal(experiment, expected_lower_case):
    from transformers import AutoTokenizer
    cfg = runner.read_json(experiment.config)
    snapshot = runner.snapshot_path(cfg, experiment.hf_cache)
    path = snapshot / "tokenizer_config.json"
    write(path, {"special_tokens_map_file": None, "full_tokenizer_file": None})
    (snapshot / "tokenizer.json").unlink()  # Same files as the pinned legacy HF export.
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    assert tokenizer.do_lower_case is True
    assert tokenizer.encode("PARKING") == tokenizer.encode("parking")
    manifest_path = experiment.asset_dir / "assets.json"
    manifest = runner.read_json(manifest_path)
    manifest["model_files_sha256"][path.name] = runner.digest(path)
    write(manifest_path, manifest)
    cfg["model"]["tokenizer_do_lower_case"] = expected_lower_case
    write(experiment.config, cfg)
    experiment.output_dir = experiment.output_dir.parent / "legacy-output"
    runner.prepare(experiment)
    if not expected_lower_case:
        with pytest.raises(ValueError, match="expected do_lower_case=False, loaded True"):
            runner.infer(experiment)
        assert not (experiment.output_dir / "runtime.json").exists()
        return
    runner.infer(experiment)
    runner.evaluate(experiment)
    runtime = runner.read_json(experiment.output_dir / "runtime.json")
    assert runtime["tokenizer_do_lower_case"] is True
    assert len(runner.read_json(experiment.output_dir / "results.json")["conditions"]) == 4


@pytest.mark.parametrize("mutation", ["weight", "label_order", "incomplete_manifest", "vocab_pickle"])
def test_assets_fail_closed_for_corruption_and_wrong_label_order(experiment, mutation):
    cfg = runner.config(experiment.config)
    snapshot = runner.snapshot_path(cfg, experiment.hf_cache)
    manifest = runner.read_json(experiment.asset_dir / "assets.json")
    if mutation == "weight":
        (snapshot / "pytorch_model.bin").write_bytes(b"corrupt")
    elif mutation == "label_order":
        path = experiment.asset_dir / "vocabulary.json"
        obj = runner.read_json(path)
        obj["labels"][2], obj["labels"][3] = obj["labels"][3], obj["labels"][2]
        write(path, obj)
        manifest["auxiliary_files_sha256"]["vocabulary.json"] = runner.digest(path)
    elif mutation == "incomplete_manifest":
        del manifest["model_files_sha256"]["vocab.txt"]
    else:
        (experiment.asset_dir / "vocab.pkl").write_bytes(b"corrupt")
    write(experiment.asset_dir / "assets.json", manifest)
    with pytest.raises(ValueError):
        runner.verify_assets(cfg, experiment.hf_cache, experiment.asset_dir)


def test_corrupted_crop_selection_shard_and_missing_sample_fail(experiment):
    runner.infer(experiment)
    path = next((experiment.output_dir / "inference/word_only").glob("*.json"))
    original = runner.read_json(path)
    changed = json.loads(json.dumps(original))
    changed["trace"][0]["selections"]["conservative"]["text"] = "injected"
    changed["payload_sha256"] = runner.payload_digest({k: v for k, v in changed.items() if k != "payload_sha256"})
    write(path, changed)
    with pytest.raises(ValueError, match="crop/selection"):
        runner.evaluate(experiment)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        runner.evaluate(experiment)


def test_caption_condition_requires_complete_sha_bound_totaltext_images(experiment):
    sample = runner.read_json(experiment.output_dir / "inputs.json")["samples"][0]
    path = experiment.output_dir.parent / "captions.json"
    obj = {"schema_version": 1, "model_id": "fixture/caption", "revision": "1" * 40,
        "generation": {"num_beams": 1}, "model_files_sha256": {"weights": "2" * 64},
        "samples": [{"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                     "text": "the parking sign"}]}
    write(path, obj)
    experiment.captions = path
    experiment.output_dir = experiment.output_dir.parent / "caption-out"
    runner.prepare(experiment)
    runner.infer(experiment)
    runner.evaluate(experiment)
    results = runner.read_json(experiment.output_dir / "results.json")
    assert len(results["conditions"]) == 6
    trace = runner.read_json(next((experiment.output_dir / "inference/local_ocr_caption").glob("*.json")))["trace"][0]
    assert trace["context"]["caption"] == "the parking sign"
    obj["samples"][0]["image_sha256"] = "0" * 64
    write(path, obj)
    experiment.output_dir = experiment.output_dir.parent / "bad-caption-out"
    with pytest.raises(ValueError, match="identity/checksum"):
        runner.prepare(experiment)


def test_fetch_pins_model_and_authenticates_lfs_and_git_blobs(experiment, monkeypatch):
    import huggingface_hub
    import requests
    cfg = runner.config(experiment.config)
    snapshot = runner.snapshot_path(cfg, experiment.hf_cache)
    sha = runner.digest(snapshot / "pytorch_model.bin")
    source = {"vocab.pkl": (experiment.asset_dir / "vocab.pkl").read_bytes(),
              "author_snippet.py": (experiment.asset_dir / "author_snippet.py").read_bytes()}
    calls = []
    class API:
        def model_info(self, model_id, **kwargs):
            assert model_id == cfg["model"]["model_id"]
            assert kwargs == {"revision": cfg["model"]["revision"], "files_metadata": True}
            return SimpleNamespace(sha=cfg["model"]["revision"], siblings=[
                SimpleNamespace(rfilename="pytorch_model.bin", lfs=SimpleNamespace(sha256=sha))])
    def download(model_id, **kwargs):
        assert kwargs["revision"] == cfg["model"]["revision"]
        assert set(kwargs["allow_patterns"]) == set(runner.MODEL_FILES)
        return str(snapshot)
    def get(url, **kwargs):
        assert f"/{cfg['upstream_neuspell']['revision']}/" in url
        assert kwargs["timeout"] == cfg["fetch"]["timeout_seconds"]
        calls.append(url)
        name = "vocab.pkl" if url.endswith("vocab.pkl") else "author_snippet.py"
        return SimpleNamespace(content=source[name], raise_for_status=lambda: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(requests, "get", get)
    experiment.asset_dir = experiment.asset_dir.parent / "new-assets"
    runner.fetch_model(experiment)
    assert len(calls) == 2
    runner.verify_assets(cfg, experiment.hf_cache, experiment.asset_dir)
    runner.fetch_model(experiment)
    assert len(calls) == 2  # Offline reuse after the first authenticated fetch.
