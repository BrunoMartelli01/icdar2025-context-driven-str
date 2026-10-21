"""CPU scientific boundary tests for the additive word-head experiment."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from localize.word_multitask import center_peaks, word_instances, parseq_word_proposals
from models.word_multitask import VARIANTS, build_word_model
from recognition.scene_text_datasets import SceneTextInstance
from scripts.run_word_multitask import DEFAULT_CONFIG, read_config, training_config, train_branch
from segmentation.word_multitask import (
    gaussian_word_centers, targets_from_polygons, WordMultitaskDataset,
    masked_bce_dice, masked_gaussian_mse, multitask_loss, train_multitask_epoch,
)
from segmentation.word_multitask_evaluation import aggregate_word_metrics, evaluate_multitask
from segmentation.totaltext_word_kernel import make_word_kernels, instance_geometry
from segmentation.training import save_checkpoint, load_checkpoint
from scripts.train_totaltext_v4_image_frequency_ablation import _transform


@pytest.fixture
def cfg():
    return read_config(DEFAULT_CONFIG, "approximate")[0]


def instance(points, *, ignore=False, granularity="word"):
    return SceneTextInstance(tuple(points), "###" if ignore else "word", ignore, granularity)


def loss_batch():
    text = torch.zeros(2, 1, 12, 16)
    text[:, :, 2:10, 2:14] = 1
    kernel = torch.zeros_like(text)
    kernel[:, :, 4:8, 5:11] = 1
    center = kernel.clone() * 0.3
    center[:, :, 6, 8] = 1
    valid = torch.ones_like(text)
    valid[:, :, 0] = 0
    return {"mask": text, "kernel": kernel, "center": center, "validity": valid,
            "separation": torch.zeros_like(text), "image": torch.ones(2, 3, 12, 16),
            "sample_id": ["word:1", "word:2"]}


def test_polygon_targets_tilted_irregular_small_and_ignore(cfg):
    items = [instance([(2, 2), (18, 5), (16, 14), (1, 11)]),
             instance([(28, 3), (44, 3), (44, 8), (34, 8), (34, 18), (28, 18)]),
             instance([(4, 25), (5, 25), (5, 26), (4, 26)]),
             instance([(45, 20), (60, 20), (60, 30), (45, 30)], ignore=True)]
    result = targets_from_polygons((64, 40), items, **cfg["targets"])
    assert {r["owner_id"] for r in result["centers"]} == {1, 2, 3}
    assert np.all(result["kernel"] <= result["mask"])
    assert not result["center"][result["validity"] == 0].any()
    assert not result["center"][result["owners_native"] == 0].any()
    for row in result["centers"]:
        x, y = row["center_xy_native"]
        assert result["center"][y, x] == 1
        assert result["owners_native"][y, x] == row["owner_id"]
    assert abs(result["centers"][0]["covariance_xy_native"][0][1]) > 0.1


def test_curved_word_centroid_projects_inside_kernel_and_singleton_survives(cfg):
    owners = np.zeros((40, 60), np.int32)
    cv2.ellipse(owners, (20, 20), (14, 10), 0, 20, 160, 1, thickness=5)
    owners[0, 59] = 2
    kernels, _ = make_word_kernels(owners)
    before = owners.copy()
    center, rows = gaussian_word_centers(owners, kernels, **{k: cfg["targets"][k] for k in ("sigma_scale", "min_sigma")})
    assert center[0, 59] == 1
    assert np.array_equal(owners, before)
    for row in rows:
        x, y = row["center_xy_native"]
        assert kernels[y, x] == row["owner_id"]
    assert np.array_equal(center, gaussian_word_centers(owners, kernels, sigma_scale=0.5, min_sigma=1)[0])


def test_empty_and_ignored_images_and_line_annotations(cfg):
    for items in ([], [instance([(0, 0), (7, 0), (7, 7), (0, 7)], ignore=True)]):
        result = targets_from_polygons((8, 8), items, **cfg["targets"])
        assert not result["mask"].any() and not result["kernel"].any() and not result["center"].any()
    with pytest.raises(ValueError, match="one polygon per word"):
        targets_from_polygons((8, 8), [instance([(1, 1), (5, 1), (5, 5)], granularity="line")], **cfg["targets"])
    with pytest.raises(ValueError, match="no kernel"):
        gaussian_word_centers(np.ones((4, 4), np.int32), np.zeros((4, 4), np.int32), sigma_scale=0.5, min_sigma=1)


def test_dataset_replays_paired_rotation_flip_letterbox_owners(tmp_path, cfg):
    _, source = read_config(DEFAULT_CONFIG, "approximate")
    source["training"].update(horizontal_flip_probability=1.0, rotation_probability=1.0,
                              rotation_degrees=[12.0, 12.0])
    transform = _transform(source, "letterbox", train=True)
    transform.set_epoch(7)
    owner = np.zeros((20, 40), np.int32)
    owner[3:17, 3:14] = 1
    owner[3:17, 26:37] = 2
    Image.fromarray(owner).save(tmp_path / "owners.tiff")
    rgb = Image.new("RGB", (40, 20), (120, 120, 120))
    target = Image.fromarray((owner > 0).astype(np.uint8) * 255)
    valid = Image.new("L", (40, 20), 255)
    image, mask, validity, separation = transform.transform_with_validity_and_separation(
        rgb, target, valid, Image.new("L", (40, 20), 0), Image.fromarray(owner), 0)
    class Base:
        data_root = tmp_path
        manifest = SimpleNamespace(samples=[SimpleNamespace(word_owner_annotation_path=Path("owners.tiff"))])
        def __len__(self): return 1
        def __getitem__(self, index):
            return {"image": image, "mask": mask, "validity": validity, "separation": separation, "sample_id": "word:1"}
    base = Base()
    base.transform = transform
    dataset = WordMultitaskDataset(base, **cfg["targets"])
    a, b = dataset[0], dataset[0]
    assert torch.equal(a["mask"], mask) and torch.equal(a["center"], b["center"])
    assert not a["center"][a["validity"] == 0].any()
    assert torch.all(a["kernel"] <= mask)
    assert set(torch.unique(a["owners_native"]).tolist()) == {0, 1, 2}


@pytest.mark.parametrize("variant", VARIANTS)
def test_weighted_losses_gradients_and_ignore(variant, cfg):
    batch = loss_batch()
    logits = torch.zeros(2, VARIANTS[variant], 12, 16, requires_grad=True)
    loss, terms = multitask_loss(logits, batch, variant=variant, config=cfg["loss"])
    assert loss == sum(cfg["loss"]["lambda_" + k] * v for k, v in terms.items())
    loss.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[:, :, 0].any()
    assert torch.all(logits.grad.flatten(2).abs().sum(2) > 0)
    if variant == "text_kernel_center":
        assert logits.grad[0, 2, 6, 8] < 0
        assert logits.grad[0, 2, 1, 0] > 0


def test_bce_dice_reference_and_all_invalid_zero_gradient(cfg):
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[1., 0.], [0., 0.]]]])
    value = masked_bce_dice(logits, target, torch.ones_like(target), bce_weight=1, dice_weight=1, smooth=1)
    assert value.item() == pytest.approx(np.log(2) + 0.5)
    ignored = torch.zeros_like(target)
    zero = masked_bce_dice(logits, target, ignored, bce_weight=1, dice_weight=1, smooth=1)
    zero = zero + masked_gaussian_mse(logits, target, ignored, balanced=True)
    zero.backward()
    assert zero.item() == 0 and not logits.grad.any()
    good = masked_bce_dice(torch.full_like(target, -20), torch.zeros_like(target), torch.ones_like(target),
                           bce_weight=1, dice_weight=1, smooth=1)
    assert good.item() < 1e-6


def test_gaussian_loss_uses_soft_values_and_balancing():
    logits = torch.tensor([[[[0., 0., 0.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 0.25, 0.]]]])
    valid = torch.ones_like(target)
    unbalanced = masked_gaussian_mse(logits, target, valid, balanced=False)
    balanced = masked_gaussian_mse(logits, target, valid, balanced=True)
    assert unbalanced.item() == pytest.approx((0.25 + 0.0625 + 0.25) / 3)
    assert balanced.item() == pytest.approx(((0.25 + 0.0625) / 2 + 0.25) / 2)


@pytest.mark.parametrize("method", ["kernel_bfs", "watershed"])
def test_center_splits_fused_kernel_and_preserves_foreground(method, cfg):
    probabilities = np.zeros((3, 24, 40), np.float32)
    probabilities[0, 3:21, 2:38] = 0.9
    probabilities[1, 8:16, 5:35] = 0.9  # one merged kernel
    probabilities[2, 12, 10] = probabilities[2, 12, 29] = 1
    post = {**cfg["postprocessing"], "method": method}
    canvas = np.ones((24, 40), bool)
    two, _, fg = word_instances(probabilities[:2], canvas, post)
    three, audit, _ = word_instances(probabilities, canvas, post)
    assert len(np.unique(two[two > 0])) == 1
    assert len(np.unique(three[three > 0])) == 2
    assert audit["kernel_components_split_by_centers"] == 1
    assert np.array_equal(three > 0, fg)
    assert np.array_equal(three, word_instances(probabilities, canvas, post)[0])


def test_center_rescues_unseeded_word_and_unseeded_fallback(cfg):
    p = np.zeros((3, 16, 24), np.float32)
    p[0, 1:7, 1:8] = p[0, 10:15, 16:23] = 0.9
    p[1, 2:6, 2:7] = 0.9
    p[2, 12, 19] = 1
    labels, audit, fg = word_instances(p, np.ones((16, 24), bool), cfg["postprocessing"])
    assert audit["center_rescue_markers"] == 1 and audit["instances"] == 2
    p[2] = 0
    labels, audit, _ = word_instances(p, np.ones((16, 24), bool), cfg["postprocessing"])
    assert audit["unseeded_fallback_components"] == 1 and np.array_equal(labels > 0, fg)
    p[:] = 0
    assert not word_instances(p, np.ones((16, 24), bool), cfg["postprocessing"])[0].any()


def test_plateaus_threshold_boundary_and_canvas(cfg):
    center = np.zeros((10, 20), np.float32)
    center[2:4, 3:5] = 0.5
    center[7, 17] = 1
    fg = np.ones_like(center, bool)
    assert center_peaks(center, fg, threshold=0.5, min_distance=2) == [(0.5, 2, 3), (1., 7, 17)]
    p = np.ones((3, 10, 20), np.float32)
    canvas = fg.copy(); canvas[:, :4] = False
    labels, _, _ = word_instances(p, canvas, cfg["postprocessing"])
    assert not labels[:, :4].any()
    with pytest.raises(ValueError, match="probabilities"):
        word_instances(p * np.nan, canvas, cfg["postprocessing"])


def test_parseq_boxes_preserve_touching_ids_area_order_and_inverse_letterbox():
    labels = np.zeros((16, 16), np.int32)
    labels[5:10, 1:8] = 1
    labels[5:10, 8:15] = 2
    rows = parseq_word_proposals(labels, (32, 16), padding=0, min_area=0, max_instances=None)
    assert len(rows) == 2 and rows[0]["bbox"] == [2, 2, 16, 12]
    rgb = np.zeros((16, 32, 3), np.uint8)
    for row in rows:
        x1, y1, x2, y2 = row["bbox"]
        assert rgb[y1:y2, x1:x2].shape == (10, 14, 3)
    labels[10, 2] = 1
    assert parseq_word_proposals(labels, (32, 16), padding=1, min_area=36, max_instances=1)[0]["instance_id"] == 1


def test_word_metrics_merge_split_and_lost_denominator():
    owners = np.zeros((10, 20), np.int32)
    owners[2:8, 1:10] = 1; owners[2:8, 10:19] = 2
    merged = instance_geometry((owners > 0).astype(np.int32), owners, (1, 2, 3))
    rates = aggregate_word_metrics(merged)
    assert rates["merge_rate"] == pytest.approx(2 / 3) and rates["split_rate"] == 0
    split = owners.copy(); split[5:8, 1:10] = 4
    rates = aggregate_word_metrics(instance_geometry(split, owners, (1, 2)))
    assert rates["merge_rate"] == 0 and rates["split_rate"] == 0.5


def test_config_namespace_and_strict_rejection(cfg):
    with pytest.raises(ValueError, match="approximate"):
        read_config(DEFAULT_CONFIG, "strict")
    with pytest.raises(ValueError, match="configs/approximate"):
        read_config(Path("/tmp/word_fake.json"), "approximate")
    tc = training_config(cfg)
    assert tc.loss == "masked_BCE_Dice_and_gaussian_MSE" and tc.batch_size == 8


def test_shared_initialization_shapes_and_parameter_budget():
    common, heads = None, None
    for variant, channels in VARIANTS.items():
        torch.manual_seed(37)
        model = build_word_model(variant, skip_floor=0.25, encoder_pretrained=False).eval()
        if common is not None:
            for name, value in model.state_dict().items():
                if not name.startswith("head."):
                    assert torch.equal(value, common[name])
            assert torch.equal(model.head.weight[:len(heads)], heads)
        assert sum(p.numel() for p in model.parameters()) == 58155081 + 65 * (channels - 1)
        with torch.inference_mode():
            assert model(torch.zeros(1, 3, 32, 64)).shape == (1, channels, 16, 32)
        common = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("head.")}
        heads = model.head.weight.detach().clone()
        del model


def test_cpu_training_smoke_and_strict_checkpoint_roundtrip(tmp_path, cfg):
    class Tiny(torch.nn.Conv2d):
        architecture_identity = "synthetic-word-multitask-test"
    model = Tiny(3, 3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.weight.detach().clone()
    result = train_multitask_epoch(model, [loss_batch()], optimizer, torch.device("cpu"),
        variant="text_kernel_center", loss_config=cfg["loss"])
    assert result["steps"] == 1 and torch.all((model.weight - before).flatten(1).abs().sum(1) > 0)
    tc = training_config(cfg)
    save_checkpoint(tmp_path / "checkpoint.pt", model=model, optimizer=optimizer, epoch=1, global_step=1,
        config=tc, resolved_config=json.dumps(cfg), architecture=model.architecture_identity, run_contract={"variant": "text_kernel_center"})
    copied = Tiny(3, 3, 1)
    payload = load_checkpoint(tmp_path / "checkpoint.pt", model=copied, expected_architecture=model.architecture_identity,
        expected_run_contract={"variant": "text_kernel_center"}, expected_training_config=tc)
    assert torch.equal(model.weight, copied.weight) and payload["training_config"]["loss"] == tc.loss
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(tmp_path / "checkpoint.pt", model=copied, expected_architecture=model.architecture_identity,
                        expected_run_contract={"variant": "text_kernel"})


def test_validation_ignore_geometry_postprocessors_and_visuals(tmp_path, cfg):
    owners = np.zeros((16, 16), np.int32)
    owners[4:12, 2:7] = 1; owners[4:12, 9:14] = 2
    batch = {"image": torch.zeros(1, 3, 32, 32), "sample_id": ["word:1"],
             "mask": torch.from_numpy(owners > 0)[None, None].float(),
             "validity": torch.ones(1, 1, 16, 16), "kernel": torch.zeros(1, 1, 16, 16),
             "center": torch.zeros(1, 1, 16, 16)}
    # Ignored false positive cannot affect precision. Inference still predicts it.
    batch["validity"][:, :, :2] = 0
    output = torch.full((1, 3, 16, 16), -20.)
    output[0, 0, owners > 0] = 20
    output[0, 0, 0, 0] = 20
    output[0, 1, 6:10, 4:6] = output[0, 1, 6:10, 10:12] = 20
    class Frozen(torch.nn.Module):
        def forward(self, image): return output
    rgb_path = tmp_path / "rgb.png"; Image.new("RGB", (32, 32)).save(rgb_path)
    result = evaluate_multitask(Frozen(), [batch], {"word:1": (owners, (1, 2), np.ones((16, 16), bool))},
        torch.device("cpu"), cfg, export_dir=tmp_path / "exports", image_paths={"word:1": rgb_path}, compare_postprocessing=True)
    assert result["aggregate"]["proposal_f1_iou_50"] == 1
    assert result["aggregate"]["foreground_f1_macro"] == 1
    assert result["per_image"][0]["ignored_prediction_ids"]
    assert set(result["postprocessing_ablation"]) == {"connected_components", "kernel_bfs", "watershed"}
    assert (tmp_path / "exports/word__1/targets.png").exists()
    assert (tmp_path / "exports/word__1/predictions.png").exists()
    with np.load(tmp_path / "exports/word__1/maps.npz") as maps:
        assert maps["probabilities"].shape == (3, 16, 16) and maps["labels"].dtype == np.int32


def test_train_resume_matches_uninterrupted_checkpoint_and_recovers_epoch_log(tmp_path, cfg, monkeypatch):
    import scripts.run_word_multitask as runner
    cfg = deepcopy(cfg)
    cfg.update(epochs=2, expected_validation_words=4)
    batch = loss_batch()
    owners = batch["mask"][0, 0].numpy().astype(np.int32)
    owners[:, 8:] *= 2
    references = {sid: (owners, (1, 2), np.ones(owners.shape, bool)) for sid in batch["sample_id"]}
    class Tiny(torch.nn.Conv2d):
        architecture_identity = "synthetic-multitask-resume-test"
    class EpochState:
        def set_epoch(self, epoch): self.epoch = epoch
    def execute(path):
        path.mkdir(exist_ok=True)
        torch.manual_seed(17)
        model = Tiny(3, 3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        result = train_branch(path, model, optimizer, [batch], [batch], references, EpochState(), EpochState(),
            cfg, {"variant": "text_kernel_center"}, torch.device("cpu"))
        return result
    full = execute(tmp_path / "full")
    append = runner._append_epoch
    def interrupt_after_checkpoint(path, row):
        raise RuntimeError("simulated interruption after committed checkpoint, before epoch log")
    monkeypatch.setattr(runner, "_append_epoch", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute(tmp_path / "resume")
    monkeypatch.setattr(runner, "_append_epoch", append)
    resumed = execute(tmp_path / "resume")
    assert resumed["validation_word_geometry"] == full["validation_word_geometry"]
    a = torch.load(tmp_path / "full/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    for key, value in a["model_state_dict"].items():
        assert torch.equal(value, b["model_state_dict"][key])
    assert torch.equal(a["rng_state"]["torch_cpu"], b["rng_state"]["torch_cpu"])
    records = [json.loads(row) for row in (tmp_path / "resume/epochs.jsonl").read_text().splitlines()]
    assert records[0]["recovered_from_checkpoint"]


def test_comparison_reuses_history_and_rejects_unmatched_contracts(tmp_path, cfg):
    from scripts.compare_word_multitask import compare
    from scripts.prune_segmentation_checkpoints import sha256_file
    metrics = {"matched_gt_iou_ge_50": 1, "raw_proposals": 2, "gt_words": 2,
               "gt_words_touched_by_fused_components": 0, "gt_words_split_across_predicted_components": 0,
               "predicted_components_touching_multiple_gt_words": 0}
    metrics = aggregate_word_metrics(metrics)
    for variant in VARIANTS:
        directory = tmp_path / variant
        directory.mkdir()
        (directory / "best.pt").write_bytes(b"fixture-only")
        (directory / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS",
            "run_contract": {"variant": variant, "config": cfg}, "validation_word_geometry": metrics,
            "checkpoint": {"path": "best.pt", "sha256": sha256_file(directory / "best.pt")}, "parameters": 3}))
    historical = tmp_path / "old.json"
    historical.write_text(json.dumps({"status": "SUCCESS", "validation_word_geometry": metrics}))
    result = compare(tmp_path, [historical])
    assert result["historical_references"][0]["comparison_role"] == "historical_reference_only"
    path = tmp_path / "text_kernel_center/run_manifest.json"
    value = json.loads(path.read_text()); value["run_contract"]["config"]["epochs"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="controlled comparison"):
        compare(tmp_path, [])


def test_cli_records_and_propagates_oom_without_alternate_model(tmp_path, monkeypatch):
    import scripts.run_word_multitask as runner
    monkeypatch.setattr("sys.argv", ["run_word_multitask", "--mode", "approximate", "--action", "smoke",
        "--device", "cpu", "--variant", "text_kernel_center", "--target-root", str(tmp_path),
        "--output-root", str(tmp_path)])
    def fail(args):
        args._owns_output = True
        raise torch.cuda.OutOfMemoryError("simulated allocation failure")
    monkeypatch.setattr(runner, "run", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated allocation failure"):
        runner.main()
    report = json.loads((tmp_path / "failure.json").read_text())
    assert report["status"] == "FAILED" and report["oom"] is True
    assert report["variant"] == "text_kernel_center" and report["result_class"] == "smoke-test result"
