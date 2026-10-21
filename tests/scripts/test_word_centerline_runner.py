"""Resume, population, crop-export and scientific-mode boundaries."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import run_word_centerline as runner
from scripts.evaluate_word_centerline_ocr import qualify_run
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.training import seed_everything
from segmentation.word_centerline import centerline_targets


def config():
    cfg, _ = runner.read_config(runner.DEFAULT_CONFIG, "approximate")
    return cfg


class TinyFrozen(torch.nn.Module):
    architecture_identity = "centerline-runner-fixture"

    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Conv2d(3, 2, 1)
        self.frozen.requires_grad_(False)
        self.centerline_private = torch.nn.Conv2d(3, 2, 1)

    def forward_evidence(self, image):
        with torch.no_grad():
            words = self.frozen(image)
        private = self.centerline_private(image)
        return {"words": words, "line": private[:, :1], "radius_raw": private[:, 1:]}


@pytest.fixture
def tiny(tmp_path):
    cfg = config()
    cfg.update(epochs=2, batch_size=1, num_workers=0, validation_count=1, expected_validation_words=3)
    cfg["validation_ids_sha256"] = __import__("hashlib").sha256(b"total_text:fixture\n").hexdigest()
    owner = np.zeros((32, 32), np.int32)
    owner[8:14, 2:13] = 1
    owner[17:24, 17:29] = 2
    valid = np.ones_like(owner, bool)
    valid[8, 2] = False
    target = owner > 0
    clipped = owner.copy()
    clipped[~valid] = 0
    targets = centerline_targets(clipped, valid, cfg["targets"])
    sid = "total_text:fixture"
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (96, 32), (60, 90, 120)).save(image_path)
    annotation = tmp_path / "datasets/total_text/train/annotations/fixture.txt"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("qualified fixture annotation")
    batch = {k: torch.from_numpy(targets[k])[None, None] for k in ("line", "radius", "radius_weight", "line_skeleton")}
    batch.update(image=torch.linspace(-1., 1., 3 * 32 * 32).reshape(1, 3, 32, 32),
        mask=torch.from_numpy(target.astype(np.float32))[None, None],
        validity=torch.from_numpy(valid.astype(np.float32))[None, None],
        kernel=torch.from_numpy(target.astype(np.float32))[None, None],
        owners_native=torch.from_numpy(clipped)[None], sample_id=[sid], image_path=[str(image_path)])
    canvas = np.zeros_like(valid)
    canvas[10:22] = True  # original RGB letterbox geometry
    references = {sid: (owner, (1, 2, 3), canvas)}  # third word vanished at native resolution
    return cfg, batch, references, image_path, annotation


def test_configuration_requires_approximate_namespace_and_evidence(tmp_path, monkeypatch):
    cfg = config()
    with pytest.raises(ValueError, match="APPROXIMATE"):
        runner.read_config(runner.DEFAULT_CONFIG, "strict")
    path = tmp_path / "outside.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="namespace"):
        runner.read_config(path, "approximate")
    real_sha = runner.sha256_file
    monkeypatch.setattr(runner, "sha256_file", lambda path: "0" * 64 if str(path).endswith(cfg["source_config"]) else real_sha(path))
    with pytest.raises(ValueError, match="preprocessing"):
        runner.read_config(runner.DEFAULT_CONFIG, "approximate")


@pytest.mark.parametrize("change", ["loss", "targets", "filtered", "radius", "mode"])
def test_invalid_scientific_settings_fail_before_model_loading(change, monkeypatch):
    cfg = config()
    if change == "loss": cfg["loss"]["radius_floor"] = float("nan")
    elif change == "targets": cfg["targets"]["ribbon_fraction"] = 1.
    elif change == "filtered": cfg["postprocessing"]["min_line_area"] = 2
    elif change == "radius": cfg["postprocessing"]["radius_floor"] = 1.
    else: cfg["reproduction_mode"] = "strict"
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k:
        json.dumps(cfg) if path.resolve() == runner.DEFAULT_CONFIG else original(path, *a, **k))
    with pytest.raises(ValueError):
        runner.read_config(runner.DEFAULT_CONFIG, "approximate")


def test_geometry_uses_full_predictions_and_retains_absent_gt():
    labels = np.zeros((8, 12), np.int32)
    owners = np.zeros_like(labels)
    labels[1:7, 1:7] = 1
    owners[3:5, 3:5] = 1
    labels[0, 11] = 2
    valid = np.ones_like(labels, bool)
    valid[0, 11] = False
    metrics, ignored, _, _ = runner.measured_geometry(labels, owners, (1, 2), owners > 0, valid)
    assert ignored == [2]
    assert metrics["gt_words"] == 2 and metrics["gt_words_lost_in_native_target"] == 1
    assert metrics["matched_gt_iou_ge_50"] == 0  # no GT clipping of the oversized box
    assert metrics["raw_proposals"] == 1


def test_validation_targets_are_clipped_before_owner_pairing(tiny, tmp_path):
    cfg, batch, references, _, _ = tiny
    single = {k: value[0] if torch.is_tensor(value) else value[0] for k, value in batch.items()}
    result = runner.preflight([single], [0], references, cfg, tmp_path / "oracle")
    assert result["status"] == "PASS" and result["native_words"] == 2
    assert result["oracle_native_geometry"]["gt_words"] == 3
    assert result["result_class"] == "smoke-test result"
    assert "not learned performance" in result["limitations"]


def test_geometry_exports_are_compatible_with_real_ocr_boundary(tiny, tmp_path, monkeypatch):
    cfg, batch, references, image_path, annotation = tiny
    cfg.update(seed=202503)
    model = TinyFrozen()
    output = tmp_path / "run"
    sample = SimpleNamespace(image_sha256=sha256_file(image_path))
    gt = [{"bbox": [1, 1, 3, 3], "text": text} for text in ("one", "two", "three")]
    monkeypatch.setattr(runner, "_train_gt", lambda *args: (gt, sha256_file(annotation)))
    parts = runner.evaluate(model, [batch], references, cfg, "cpu", output=output,
        samples={batch["sample_id"][0]: sample}, root=tmp_path)
    source = {"scripts/run_word_centerline.py": sha256_file(runner.REPO / "scripts/run_word_centerline.py")}
    contract = {"reproduction_mode": "approximate", "result_class": "approximate reproduction",
                "resolved_config": cfg, "source_sha256": source, "validation_ids_sha256": cfg["validation_ids_sha256"]}
    (output / "best.pt").write_bytes(b"fixture checkpoint checked for content, not loaded as a model")
    digest = sha256_file(output / "best.pt")
    report = {"status": "SUCCESS", "training_complete": True, "completed_epochs": 2,
        "run_contract": contract, "result_class": "approximate reproduction", "checkpoint": {"path": "best.pt", "sha256": digest}}
    geometry = {"status": "SUCCESS", "reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "protocol": {"run_contract": contract, "checkpoint_sha256": digest, "source_seed": cfg["seed"],
            "expected_images": 1, "expected_care_words": 3}, "branches": {"centerline_radius": parts}}
    runner.write_json(output / "run_manifest.json", report)
    runner.write_json(output / "geometry_results.json", geometry)
    assert qualify_run(output, "approximate") == (report, geometry)
    row = parts["baseline"]["per_image"][0]
    for proposal in row["proposals"]:
        x1, y1, x2, y2 = proposal["bbox"]
        assert proposal["crop_rgb_sha256"] == runner.crop_array_hash(np.asarray(Image.open(image_path))[y1:y2, x1:x2])
    # Existing exact maps can be reused; changed predictions fail closed.
    assert runner.evaluate(model, [batch], references, cfg, "cpu", output=output,
        samples={batch["sample_id"][0]: sample}, root=tmp_path) == parts
    maps = output / "samples/total_text__fixture/maps.npz"
    maps.write_bytes(b"damaged export")
    with pytest.raises(ValueError, match="checksum"):
        qualify_run(output, "approximate")


def test_population_cannot_be_shortened_or_reordered(tiny):
    cfg, batch, references, _, _ = tiny
    extra = {**references, "missing": next(iter(references.values()))}
    with pytest.raises(ValueError, match="population"):
        runner.evaluate(TinyFrozen(), [batch], extra, cfg, "cpu")
    with pytest.raises(ValueError, match="denominator"):
        runner.evaluate(TinyFrozen(), [batch], references, {**cfg, "expected_validation_words": 2}, "cpu")


def train_fixture(path, model, cfg, batch, references, *, stop=None):
    path.mkdir()
    optimizer = torch.optim.Adam(model.centerline_private.parameters(), lr=cfg["learning_rate"], weight_decay=.01)
    class Batches:
        def __iter__(self):
            # Consume RNG as a real augmentation would; resume must restore it.
            yield {**batch, "image": batch["image"] + torch.rand_like(batch["image"]) * .1}
    setter = SimpleNamespace(set_epoch=lambda e: None)
    result = runner.train_head(path, model, optimizer, Batches(), [batch], references, setter, setter,
        cfg, {"fixture": "same"}, torch.device("cpu"), stop)
    return result, optimizer


def test_interrupted_training_restores_optimizer_rng_and_selects_epoch0(tiny, tmp_path):
    cfg, batch, references, _, _ = tiny
    seed_everything(123, deterministic_algorithms=True)
    initial = TinyFrozen().state_dict()
    full = TinyFrozen()
    full.load_state_dict(initial, strict=True)
    seed_everything(456, deterministic_algorithms=True)
    complete, _ = train_fixture(tmp_path / "full", full, cfg, batch, references)
    split = TinyFrozen()
    split.load_state_dict(initial, strict=True)
    seed_everything(456, deterministic_algorithms=True)
    partial, optimizer = train_fixture(tmp_path / "split", split, cfg, batch, references, stop=1)
    assert partial["training_complete"] is False and partial["completed_epochs"] == 1
    # Simulate interruption after checkpoint commit and before epoch log commit.
    (tmp_path / "split/epochs.jsonl").write_text("")
    restarted = TinyFrozen()
    restarted.load_state_dict(initial, strict=True)
    optimizer = torch.optim.Adam(restarted.centerline_private.parameters(), lr=cfg["learning_rate"], weight_decay=.01)
    setter = SimpleNamespace(set_epoch=lambda e: None)
    class Batches:
        def __iter__(self):
            yield {**batch, "image": batch["image"] + torch.rand_like(batch["image"]) * .1}
    resumed = runner.train_head(tmp_path / "split", restarted, optimizer, Batches(), [batch], references,
        setter, setter, cfg, {"fixture": "same"}, torch.device("cpu"))
    assert resumed["completed_epochs"] == 2 and resumed["training_complete"]
    for name, value in full.state_dict().items():
        assert torch.equal(value, restarted.state_dict()[name]), name
    assert resumed["checkpoint"]["epoch"] == complete["checkpoint"]["epoch"]
    assert resumed["validation_word_geometry"] == complete["validation_word_geometry"]
    assert json.loads((tmp_path / "split/epochs.jsonl").read_text().splitlines()[0])["recovered_from_checkpoint"]


def test_completion_receipt_cannot_hide_missing_epochs(tiny, tmp_path):
    cfg, batch, references, _, _ = tiny
    model = TinyFrozen()
    receipt, _ = train_fixture(tmp_path / "partial", model, cfg, batch, references, stop=1)
    receipt["run_contract"] = {"fixture": "same"}
    runner._json(tmp_path / "partial/run_manifest.json", receipt)
    with pytest.raises(ValueError, match="completed training"):
        runner.qualify_complete(tmp_path / "partial", cfg, {"fixture": "same"})
    receipt.update(status="SUCCESS", training_complete=True, completed_epochs=2)
    runner._json(tmp_path / "partial/run_manifest.json", receipt)
    with pytest.raises(ValueError, match="history"):
        runner.qualify_complete(tmp_path / "partial", cfg, {"fixture": "same"})


def test_map_and_source_hashes_reject_scientific_mutation(tmp_path):
    path = tmp_path / "maps.npz"
    runner.save_maps(path, {"labels": np.zeros((8, 8), np.int32)})
    with pytest.raises(ValueError, match="maps differ"):
        runner.save_maps(path, {"labels": np.ones((8, 8), np.int32)})
    model = TinyFrozen()
    before = runner.source_state_hash(model.state_dict())
    with torch.no_grad():
        model.centerline_private.bias.add_(1.)
    assert runner.source_state_hash(model.state_dict()) == before
    with torch.no_grad():
        model.frozen.bias.add_(1.)
    assert runner.source_state_hash(model.state_dict()) != before


def test_qualified_parent_code_is_checked_before_transfer(tmp_path, monkeypatch):
    original = tmp_path / "parent.py"
    original.write_text("qualified original model source")
    directory = tmp_path / "source-run"
    directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({"run_contract": {
        "source_sha256": {"parent.py": sha256_file(original)}}}))
    monkeypatch.setattr(runner, "REPO", tmp_path)
    state = {"fixture": torch.ones(1)}
    monkeypatch.setattr(runner, "load_source_state", lambda *args: state)
    assert runner.qualified_source_state(directory, {}) is state
    original.write_text("changed decoder behavior with compatible tensor shapes")
    with pytest.raises(ValueError, match="source-control code changed"):
        runner.qualified_source_state(directory, {})
