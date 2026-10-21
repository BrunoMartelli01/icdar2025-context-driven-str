import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
from PIL import Image, ImageFont
import pytest

from recognition import clip_innovative_verifier as advanced
from recognition.clip_word_verifier import selections
from scripts import run_clip_innovative_verification as runner
from scripts.run_clip_verification_suite import ensure_snapshot
from scripts.crop_clip_runtime import array_sha


REPO = Path(__file__).resolve().parents[2]


def configs():
    return runner.load_config(REPO/"configs/approximate/clip_innovative_verification_train_v2.json")


def scored():
    image = np.zeros((12, 20, 3), np.uint8)
    image[4:8, 4:16] = 255
    crop = image[4:8, 4:16]
    pool = {"anchor_rank": 1, "baseline_text": "WRONG", "baseline_parseq": -.4,
        "reference": {"candidate_id": "base", "bbox": [4, 4, 16, 8],
                      "crop_shape": list(crop.shape), "crop_array_sha256": array_sha(crop)},
        "words": {"WRONG": {"parseq": -.4}, "RIGHT": {"parseq": -.1}}}
    scores = {}
    for word, value in (("WRONG", .5), ("RIGHT", .6)):
        scores[word] = {f"{view}_{prompt}": value for view in ("crop", "surround", "full", "masked_surround")
            for prompt in ("raw", "written", "ensemble", "context", "context_residual", "shuffled_context")}
    for prompt in ("raw", "written", "ensemble", "context_residual"):
        scores["WRONG"]["masked_surround_"+prompt] = .5
        scores["RIGHT"]["masked_surround_"+prompt] = .2
    return image, {"pools": [pool], "scores": {"1": scores}}


def test_letter_removal_difference_changes_ranking_without_changing_old_scores_or_boxes():
    _, original = scored()
    snapshot = copy.deepcopy(original)
    result = advanced.counterfactual_scores(original)
    assert original == snapshot
    assert result["pools"] == original["pools"]
    assert result["scores"]["1"]["WRONG"]["letter_effect_written"] == 0
    assert result["scores"]["1"]["RIGHT"]["letter_effect_written"] == pytest.approx(.4)
    cfg, old = configs()
    chosen = advanced.innovative_selections(result, old, cfg)
    assert chosen["letter_effect_gate"]["1"] == "RIGHT"
    assert "glyph_verified_gate" not in chosen
    assert len(chosen) == 33
    assert all(chosen[name] == choice for name, choice in selections(original, old).items())


def test_font_checksums_fail_closed(tmp_path):
    path = tmp_path/"font.ttf"
    path.write_bytes(b"font identity fixture")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert advanced.font_identity({"font_sha256": {"font": expected}}, {"font": path})["font"]["sha256"] == expected
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="font checksum"):
        advanced.font_identity({"font_sha256": {"font": expected}}, {"font": path})


def test_renderer_preserves_descenders_and_rejects_invisible_or_unsupported_strings():
    font = ImageFont.load_default(size=24)
    rendered = advanced.render_word("gjpq", font, padding=6, foreground=[0]*3, background=[255]*3)
    values = np.asarray(rendered)
    assert rendered.mode == "RGB"
    assert values.min() == 0 and values.max() == 255
    assert np.all(values[:6] == 255) and np.all(values[-6:] == 255)
    assert np.all(values[:, :6] == 255) and np.all(values[:, -6:] == 255)
    for text in ("", " ", "\n", "é"):
        with pytest.raises(ValueError):
            advanced.render_word(text, font, padding=6, foreground=[0]*3, background=[255]*3)


def test_stretch_preserves_entire_image_at_declared_resolution():
    image = Image.new("RGB", (80, 12), "white")
    image.putpixel((0, 0), (0, 0, 0))
    result = advanced.warp_square(image, 224)
    assert result.size == (224, 224)
    assert np.asarray(result).min() < 10


def test_glyph_probes_and_reverse_controls_never_enter_candidate_pool(monkeypatch):
    cfg, old = configs()
    cfg["font_sha256"] = {"font": "identity"}
    font = ImageFont.load_default(size=24)
    monkeypatch.setattr(advanced.ImageFont, "truetype", lambda *args, **kwargs: font)
    counts = []
    def embeddings(_model, _processor, *, images, texts, **kwargs):
        values = images if images is not None else texts
        counts.append((images is not None, len(values)))
        return np.array([[.6, .8]]*len(values), np.float32)
    monkeypatch.setattr(advanced, "_embeddings", embeddings)
    model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(image_size=224)))
    scorer = advanced.GlyphScorer(model, None, config=cfg, base_config=old,
                                font_paths={"font": Path("fixture")}, device="cpu")
    image, original = scored()
    result = scorer.score(original, image)
    assert result["pools"] == original["pools"]
    assert set(result["scores"]["1"]) == {"WRONG", "RIGHT"}
    assert result["reverse_word_controls"] == {"RIGHT": "THGIR", "WRONG": "GNORW"}
    assert "THGIR" in result["render_metadata"]
    assert result["scores"]["1"]["RIGHT"]["glyph_order_effect"] == 0
    chosen = advanced.innovative_selections(result, old, cfg)
    assert len(chosen) == 40
    assert all(word in {"RIGHT", "WRONG"} for choice in chosen.values() for word in choice.values())
    before = len(counts)
    scorer.score(original, image)
    # Rerunning same words encodes only two fixed input views, reuses all templates.
    assert counts[before:] == [(True, 2)]


def test_empty_words_are_kept_in_all_innovative_denominators():
    cfg, old = configs()
    empty = {"pools": [{"anchor_rank": 1, "baseline_text": "", "words": {"": {"parseq": None}}}],
             "scores": {"1": {}}, "render_metadata": {}}
    chosen = advanced.innovative_selections(empty, old, cfg)
    assert len(chosen) == 40
    assert all(choice == {"1": ""} for choice in chosen.values())


def test_innovation_config_cannot_inherit_from_strict_or_outside_namespace(tmp_path):
    path = tmp_path/"config.json"
    path.write_text(json.dumps(configs()[0]))
    with pytest.raises(ValueError, match="configs/approximate"):
        runner.load_config(path)


def test_downloader_verifies_pinned_commit_and_remote_weight_hash(tmp_path, monkeypatch):
    commit = "a"*40
    weight = b"authentic fixture bytes"
    digest = hashlib.sha256(weight).hexdigest()
    spec = {"model_id": "official/fixture", "revision": commit,
            "weight_file": "pytorch_model.bin", "weight_sha256": None}
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda url: SimpleNamespace(
        etag=digest, commit_hash=commit, size=len(weight)))
    def download(repo_id, filename, *, revision, local_dir):
        assert repo_id == spec["model_id"] and revision == commit
        path = Path(local_dir)/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(weight if filename == spec["weight_file"] else b"{}")
        return str(path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    receipt = ensure_snapshot(spec, tmp_path/commit, offline=False)
    assert receipt["upstream_verified"] and receipt["weight_sha256"] == digest
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda url: SimpleNamespace(
        etag=digest, commit_hash="b"*40, size=len(weight)))
    with pytest.raises(ValueError, match="full commit"):
        ensure_snapshot(spec, tmp_path/commit, offline=False)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda url: SimpleNamespace(
        etag=digest, commit_hash=commit, size=len(weight)))
    weight = b"corrupted download bytes"
    with pytest.raises(ValueError, match="upstream SHA256"):
        ensure_snapshot(spec, tmp_path/commit, offline=False)


def test_offline_missing_version_never_substitutes_existing_model(tmp_path):
    cfg, old = configs()
    spec = old["models"]["vit_b16"]
    with pytest.raises(FileNotFoundError, match="pinned.*snapshot missing"):
        ensure_snapshot(spec, tmp_path/spec["revision"], offline=True)


def test_innovative_runner_propagates_oom_and_records_failure(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(runner, "local_paths", lambda path: {"output_root": tmp_path, "fonts": {}})
    def failed(*args):
        raise torch.cuda.OutOfMemoryError("simulated; no actual GPU inference")
    monkeypatch.setattr(runner, "checked_base", failed)
    args = SimpleNamespace(config=REPO/"configs/approximate/clip_innovative_verification_train_v2.json",
        paths=tmp_path/"profile.json", model="vit_b32", device="cpu", base_run=tmp_path,
        counterfactual_only=False)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated"):
        runner.run(args)
    report = json.loads((tmp_path/"vit_b32/all-methods/failure.json").read_text())
    assert report["out_of_memory"] is True and report["status"] == "FAILED"
    assert report["cuda_peak"]["peak_reserved_bytes"] is None
