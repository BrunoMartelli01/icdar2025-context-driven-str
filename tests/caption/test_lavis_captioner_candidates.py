import copy
import json
import sys
from types import SimpleNamespace
from argparse import Namespace
from pathlib import Path

import pytest

from diagnostics.lavis_captioner_candidates import (
    CANDIDATES, SAMPLES, archive_empty_preflight, audit_visual_backbone, digest, download, prepare, sha, text_metrics, verify_record,
)


def fixture_args(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for sid in SAMPLES:
        (images / f"img_{sid}.jpg").write_bytes(f"frozen RGB fixture {sid}".encode())
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"model_id": "microsoft/Florence-2-large-ft",
        "samples": {str(s): {"caption": "caption"} for s in SAMPLES}}))
    return Namespace(data_root=images, reference=ref, output_dir=tmp_path / "outputs")


def test_frozen_five_samples_and_idempotent_protocol(tmp_path):
    args = fixture_args(tmp_path)
    a = prepare(args)
    assert a == prepare(args)
    assert a["sample_ids"] == [1, 25, 26, 27, 28]
    assert a["classification"] == "PRACTICAL_APPROXIMATION"
    assert (args.output_dir / "runner.py").is_file()
    (args.data_root / "img_1.jpg").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen protocol changed"):
        prepare(args)


def test_reference_wrong_sample_rejected(tmp_path):
    args = fixture_args(tmp_path)
    ref = json.loads(args.reference.read_text())
    ref["samples"].pop("28")
    args.reference.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="Wrong reference samples"):
        prepare(args)


def test_repeated_phrases_and_echo():
    assert text_metrics("a sign says a sign says a sign says")["repeated_trigram_fraction"] > 0.5
    assert text_metrics("Describe this image. A bottle.", "Describe this image.")["prompt_echo"]
    assert text_metrics("A white bottle with black letters.")["repeated_trigram_fraction"] == 0


def test_lavis_is_not_silently_substituted():
    assert CANDIDATES["blip-base-coco"]["model"] == "blip_caption"
    assert CANDIDATES["blip-base-coco"]["is_blip2"] is False
    assert CANDIDATES["blip2-vitl"]["model_type"] == "pretrain_vitL"
    assert len(CANDIDATES["florence2-base-ft"]["revision"]) == 40
    assert CANDIDATES["florence2-base-ft"]["tasks"] == ["<MORE_DETAILED_CAPTION>", "<OCR>"]


def test_corrupt_saved_caption_and_device_rejected():
    protocol = {"images": {"1": {"sha256": "rgb-sha"}}}
    row = {"caption": "A bottle", "image_sha256": "rgb-sha"}
    row["result_sha256"] = digest(row)
    record = {"protocol_sha256": digest(protocol), "candidate": "blip-base-coco",
              "device": "cpu", "samples": {"1": row}}
    verify_record(record, protocol, "blip-base-coco", "cpu")
    with pytest.raises(RuntimeError, match="Incompatible"):
        verify_record(record, protocol, "blip-base-coco", "cuda:0")
    changed = copy.deepcopy(record)
    changed["samples"]["1"]["caption"] = "tampered"
    with pytest.raises(RuntimeError, match="Corrupt"):
        verify_record(changed, protocol, "blip-base-coco", "cpu")


def test_checkpoint_cache_corruption_fails(tmp_path):
    original = tmp_path / "source.pth"
    original.write_bytes(b"checkpoint fixture")
    target = tmp_path / "cache.pth"
    download(original.as_uri(), target, expected=sha(original))
    assert download(original.as_uri(), target, expected=sha(original)) == target
    target.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download(original.as_uri(), target, expected=sha(original))


def test_unreceipted_checkpoint_not_trusted(tmp_path):
    target = tmp_path / "weight.pth"
    target.write_bytes(b"unknown checkpoint")
    with pytest.raises(RuntimeError, match="Unverified existing"):
        download("unused-url", target)


def test_runtime_change_archives_failed_preflight(tmp_path):
    args = fixture_args(tmp_path)
    old = prepare(args)
    result = args.output_dir / "blip-base-coco.json"
    result.write_text(json.dumps({"samples": {}, "status": "FAILED"}))
    new = dict(old, host_helper_sha256="new-runtime-source")
    archive_empty_preflight(args.output_dir, old, new)
    history = args.output_dir / "history" / digest(old)
    assert json.loads((history / "protocol.json").read_text()) == old
    assert (history / result.name).is_file()
    assert not result.exists()


def test_runtime_change_cannot_discard_valid_caption(tmp_path):
    args = fixture_args(tmp_path)
    old = prepare(args)
    result = args.output_dir / "blip-base-coco.json"
    result.write_text(json.dumps({"samples": {"1": {"caption": "valid saved output"}}, "status": "PARTIAL"}))
    with pytest.raises(RuntimeError, match="Saved model outputs exist"):
        archive_empty_preflight(args.output_dir, old, dict(old, host_helper_sha256="new"))
    assert result.is_file()
    assert (args.output_dir / "protocol.json").is_file()


def test_explicit_runtime_repair_preserves_frozen_protocol_and_results(tmp_path):
    args = fixture_args(tmp_path)
    old = prepare(args)
    old["runner_sha256"] = "previous-exact-source"
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(old))
    result = args.output_dir / "blip-base-coco.json"
    result.write_bytes(b"saved successful caption bytes")
    before = (protocol_path.read_bytes(), result.read_bytes())
    args.resume_runtime_repair = True
    assert prepare(args) == old
    assert before == (protocol_path.read_bytes(), result.read_bytes())
    assert list((args.output_dir / "runtime-revisions").glob("*/runtime.json"))
    (args.data_root / "img_1.jpg").write_bytes(b"different scientific input")
    with pytest.raises(RuntimeError, match="Frozen scientific protocol changed"):
        prepare(args)


@pytest.mark.parametrize("fault", [None, "rounded", "tensor", "missing", "unexpected", "unfrozen", "receipt"])
def test_separate_visual_checkpoint_is_verified_not_merely_ignored(tmp_path, monkeypatch, fault):
    import torch
    encoder = torch.nn.Linear(2, 2)
    reference = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    encoder.requires_grad_(fault == "unfrozen")
    if fault == "tensor":
        reference["weight"] += 1
    if fault == "missing":
        reference.pop("bias")
    if fault == "unexpected":
        reference["unknown_visual_layer"] = torch.zeros(1)
    if fault == "rounded":
        with torch.no_grad():
            for value in encoder.parameters():
                value.copy_(value.half().float())
    checkpoint = tmp_path / "clip_vit_L.pth"
    torch.save(reference, checkpoint)
    if fault == "receipt":
        checkpoint.with_suffix(".pth.sha256").write_text("wrong SHA")
    fake_clip = SimpleNamespace(__file__=__file__, interpolate_pos_embed=lambda *args: None)
    monkeypatch.setitem(sys.modules, "lavis.models", SimpleNamespace(clip_vit=fake_clip))
    monkeypatch.setitem(sys.modules, "timm.models", SimpleNamespace(hub=SimpleNamespace(get_cache_dir=lambda: str(tmp_path))))
    cfg = SimpleNamespace(model={"vit_model": "clip_L", "freeze_vit": True,
                                "vit_precision": "fp16" if fault == "rounded" else "fp32"})
    if fault not in (None, "rounded"):
        with pytest.raises(RuntimeError):
            audit_visual_backbone(SimpleNamespace(visual_encoder=encoder), cfg, torch)
    else:
        evidence = audit_visual_backbone(SimpleNamespace(visual_encoder=encoder), cfg, torch)
        assert evidence["verified_tensor_count"] == 2
        assert evidence["checkpoint_sha256"] == sha(checkpoint)
