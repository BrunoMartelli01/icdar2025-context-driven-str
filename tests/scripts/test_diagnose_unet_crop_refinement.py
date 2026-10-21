"""Runner evidence, preprocessing, isolation and failure-contract tests."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

import scripts.diagnose_unet_crop_refinement as runner
from segmentation.data import file_sha256


PROFILE = Path(__file__).resolve().parents[2] / "configs/approximate/unet_crop_refinement_v1.json"


class FixtureModel(torch.nn.Module):
    """Synthetic model solely for testing IO, strict loading and error propagation."""
    def __init__(self, **kwargs):
        super().__init__()
        logits = torch.full((1, 1, 256, 256), -10.0)
        logits[:, :, 100:150, 80:160] = 10.0
        self.logits = torch.nn.Parameter(logits)

    @property
    def architecture_identity(self):
        return "practical-totaltext-v4-highres-agunet-v1"

    def forward(self, tensor):
        return self.logits.expand(tensor.shape[0], -1, -1, -1)


@pytest.fixture
def fixture_args(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PracticalTotalTextHighResAGUNet", FixtureModel)
    namespace = tmp_path / "approximate"
    namespace.mkdir()
    profile = namespace / "refinement.json"
    profile.write_bytes(PROFILE.read_bytes())
    training = namespace / "training.json"
    training.write_text(json.dumps({"reproduction_mode": "approximate", "training": {"skip_floor": 0.25},
                                   "variants": {"letterbox": {"parameters": {"canvas_fill_rgb": [124, 116, 104]}}}}))
    contract = {"architecture": "practical-totaltext-v4-highres-agunet-v1", "skip_floor": 0.25,
                "config_sha256": file_sha256(training), "input_shape": [3, 512, 512],
                "logits_shape": [1, 256, 256], "variant": "letterbox",
                "intervention": {"image_intervention": "letterbox",
                                 "parameters": {"canvas_fill_rgb": [124, 116, 104]}}}
    checkpoint = tmp_path / "model.pt"
    torch.save({"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                "architecture": FixtureModel().architecture_identity, "model_state_dict": FixtureModel().state_dict(),
                "run_contract": contract}, checkpoint)
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({"status": "SUCCESS", "run_contract": contract,
                                   "checkpoint": {"sha256": file_sha256(checkpoint), "epoch": 1}}))
    image = tmp_path / "image.png"
    Image.fromarray(np.full((41, 80, 3), 127, np.uint8)).save(image)
    return argparse.Namespace(config=profile, training_config=training,
                              training_run_manifest=manifest, checkpoint=checkpoint,
                              images=[image], device="cpu", output_dir=tmp_path / "results")


@pytest.fixture(autouse=True)
def restore_deterministic_settings():
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = benchmark


@pytest.mark.parametrize("geometry", ["stretch", "letterbox"])
@pytest.mark.parametrize("image_hw", [(41, 80), (80, 41), (1, 35), (64, 64)])
def test_projection_matches_independent_nearest_resize_and_odd_canvas(geometry, image_hw):
    config = json.loads(PROFILE.read_text())
    spec = deepcopy(config["inference"])
    spec["resize_geometry"] = geometry
    segmenter = runner.RGBSegmenter(FixtureModel(), spec, torch.device("cpu"))
    rgb = np.zeros((*image_hw, 3), np.uint8)
    result = segmenter(rgb)
    native = result["mask_native"]
    height, width = image_hw
    if geometry == "letterbox":
        side = max(image_hw)
        projected = cv2.resize(native, (side, side), interpolation=cv2.INTER_NEAREST)
        y, x = (side - height) // 2, (side - width) // 2
        expected = projected[y:y + height, x:x + width]
    else:
        expected = cv2.resize(native, (width, height), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(result["mask"], expected)
    assert result["metadata"]["input_shape"] == [1, 3, 512, 512]
    assert result["mask"].shape == image_hw


def test_paired_runner_saves_all_hashes_and_resolved_configuration(fixture_args):
    summary = runner.run(fixture_args)
    assert summary["status"] == "SUCCESS"
    assert summary["images_completed"] == summary["images_requested"] == 1
    assert summary["samples"][0]["second_pass_calls"] == 1
    assert summary["memory"]["peak_allocated_bytes"] is None
    protocol = json.loads((fixture_args.output_dir / "resolved-config.json").read_text())
    assert protocol["ground_truth_use"] == "none"
    assert protocol["runtime_scientific_choices"][0]["classification"] == "HARDWARE_DEVIATION"
    assert protocol["config_sha256"] == file_sha256(fixture_args.config)
    directory = fixture_args.output_dir / "image-0001"
    report = json.loads((directory / "report.json").read_text())
    for name, artifact in report["artifacts"].items():
        assert file_sha256(directory / name) == artifact["sha256"]
    assert (directory / "seed-01-seed_support_mask.png").is_file()
    assert (directory / "seed-01-probability_native.npy").is_file()
    with pytest.raises(FileExistsError):
        runner.run(fixture_args)


@pytest.mark.parametrize("field,value", [("reproduction_mode", "strict"), ("claim_label", "strict reproduction")])
def test_strict_fails_before_model_load_or_output(fixture_args, field, value, monkeypatch):
    config = json.loads(fixture_args.config.read_text())
    config[field] = value
    fixture_args.config.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "load_checkpoint", lambda *a, **k: pytest.fail("Must not load"))
    with pytest.raises(ValueError, match="STRICT"):
        runner.run(fixture_args)
    assert not fixture_args.output_dir.exists()


@pytest.mark.parametrize("changes", [{"skip_floor": 0.0}, {"resize_geometry": "stretch"},
                                    {"canvas_fill_rgb": [0, 0, 0]}])
def test_training_preprocessing_mismatch_fails_early(fixture_args, changes):
    config = json.loads(fixture_args.config.read_text())
    config["inference"].update(changes)
    fixture_args.config.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        runner.run(fixture_args)
    assert not fixture_args.output_dir.exists()


def test_checkpoint_hash_mismatch_fails_before_output(fixture_args):
    with fixture_args.checkpoint.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        runner.run(fixture_args)
    assert not fixture_args.output_dir.exists()


def test_incompatible_weights_fail_strictly_and_keep_failure_record(fixture_args):
    payload = torch.load(fixture_args.checkpoint, weights_only=False)
    payload["model_state_dict"] = {"unexpected": torch.zeros(1)}
    torch.save(payload, fixture_args.checkpoint)
    manifest = json.loads(fixture_args.training_run_manifest.read_text())
    manifest["checkpoint"]["sha256"] = file_sha256(fixture_args.checkpoint)
    fixture_args.training_run_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Strict checkpoint"):
        runner.run(fixture_args)
    failure = json.loads((fixture_args.output_dir / "failure.json").read_text())
    assert failure["images_completed"] == 0
    assert failure["images_requested"] == 1
    assert not (fixture_args.output_dir / "summary.json").exists()


def test_oom_during_second_pass_is_logged_and_propagated(fixture_args, monkeypatch):
    calls = []

    def forward(self, tensor):
        calls.append(1)
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("synthetic OOM for failure-contract test")
        return self.logits

    monkeypatch.setattr(FixtureModel, "forward", forward)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        runner.run(fixture_args)
    failure = json.loads((fixture_args.output_dir / "failure.json").read_text())
    assert failure["out_of_memory"] is True
    assert failure["stage"] == "crop_resegmentation"
    assert failure["active_image"] == str(fixture_args.images[0].resolve())
    assert failure["images_requested"] == 1 and failure["images_completed"] == 0


def test_nonfinite_model_output_is_not_thresholded_to_dummy_mask():
    spec = json.loads(PROFILE.read_text())["inference"]
    model = FixtureModel()
    with torch.no_grad():
        model.logits[0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="finite logits"):
        runner.RGBSegmenter(model, spec, torch.device("cpu"))(np.zeros((40, 80, 3), np.uint8))


def test_same_preprocessing_as_qualified_letterbox_path():
    from types import SimpleNamespace
    from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
    spec = json.loads(PROFILE.read_text())["inference"]
    captured = []

    class CaptureModel(FixtureModel):
        def forward(self, tensor):
            captured.append(tensor.clone())
            assert not torch.is_grad_enabled()
            assert not self.training
            return self.logits

    rgb = np.random.default_rng(17).integers(0, 256, size=(37, 81, 3), dtype=np.uint8)
    segmenter = runner.RGBSegmenter(CaptureModel(), spec, torch.device("cpu"))
    segmenter(rgb)
    transform = SimpleNamespace(mode="letterbox", parameters={"canvas_fill_rgb": spec["canvas_fill_rgb"]})
    torch.testing.assert_close(captured[0], _tensor(rgb, transform), rtol=0, atol=0)


def test_native_threshold_is_inclusive():
    spec = json.loads(PROFILE.read_text())["inference"]
    model = FixtureModel()
    with torch.no_grad():
        model.logits.zero_()
    output = runner.RGBSegmenter(model, spec, torch.device("cpu"))(np.zeros((20, 30, 3), np.uint8))
    assert (output["mask_native"] == 1).all()


def test_author_run_uses_exact_saved_config_provenance(tmp_path):
    spec = deepcopy(json.loads(PROFILE.read_text())["inference"])
    spec.update(architecture="author-clarified-ag-unet-2048x1024-v1", input_size=[256, 256],
                native_mask_size=[128, 128], skip_floor=0.0, resize_geometry="stretch")
    training = tmp_path / "resolved_config.yaml"
    training.write_text(json.dumps({"reproduction_mode": "approximate", "architecture": {"identity": spec["architecture"]},
                                   "transforms": {"normalization": "imagenet", "resize_geometry": "stretch",
                                                  "image_size": [256, 256], "mask_size": [128, 128]}}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"provenance-only fixture; no model loading")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "SUCCESS", "architecture": {"identity": spec["architecture"]},
                                   "resolved_config": {"sha256": file_sha256(training)},
                                   "checkpoint": {"sha256": file_sha256(checkpoint)}}))
    assert runner.verify_training_provenance(spec, training, manifest, checkpoint)["run_contract"] is None
    training.write_text(training.read_text() + "\n")
    with pytest.raises(ValueError, match="exact saved"):
        runner.verify_training_provenance(spec, training, manifest, checkpoint)


def test_recorded_model_source_drift_is_rejected(fixture_args):
    manifest = json.loads(fixture_args.training_run_manifest.read_text())
    manifest["run_contract"]["base_model_source_sha256"] = "0" * 64
    fixture_args.training_run_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Model source"):
        runner.run(fixture_args)
    assert not fixture_args.output_dir.exists()


def test_config_namespace_cannot_fall_back_to_demo_or_strict(tmp_path):
    namespace = tmp_path / "demo"
    namespace.mkdir()
    path = namespace / "config.json"
    path.write_bytes(PROFILE.read_bytes())
    with pytest.raises(ValueError, match="STRICT forbidden"):
        runner.load_config(path)
