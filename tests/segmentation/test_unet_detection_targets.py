from copy import deepcopy
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest
from scipy.io import savemat
import torch

from segmentation.synthtext_words import decode_word_boxes, prepare_synthtext, SynthTextWordDataset, synthtext_word_targets
from segmentation.unet_detection_methods import progressive_targets, detection_loss, train_detection_epoch
from scripts.run_unet_detection_methods import DEFAULT_CONFIG, read_config
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_image_frequency_ablation import _transform


@pytest.fixture
def cfg():
    return read_config(DEFAULT_CONFIG, "approximate")[0]


def owners():
    value = np.zeros((64, 96), np.int32)
    value[8:40, 4:32] = 1; value[20:23, 32:45] = 1; value[8:40, 45:65] = 1
    value[49:60, 70:90] = 2; value[1, 92] = 3
    return value


def test_nested_connected_kernels_preserve_tiny_and_irregular_words():
    owner = owners(); maps = progressive_targets(owner, [.5, .3, .1])
    assert maps.shape == (3, *owner.shape)
    assert np.all(maps[:-1] <= maps[1:]) and np.all(maps <= (owner > 0))
    for index in [1, 2, 3]:
        for kernel in maps:
            assert (kernel[owner == index] > 0).any()
            count, _ = cv2.connectedComponents(((kernel > 0) & (owner == index)).astype(np.uint8), connectivity=8)
            assert count == 2
    assert maps[:, 1, 92].tolist() == [1., 1., 1.]


@pytest.mark.parametrize("fractions", [[.1, .3, .5], [.5, .5, .1], [1, .3, .1], [.5, .3]])
def test_invalid_kernel_scales_fail(fractions):
    with pytest.raises(ValueError): progressive_targets(owners(), fractions)


def batch_from_owners():
    owner = owners(); kernels = torch.from_numpy(progressive_targets(owner, [.5, .3, .1]))[None]
    return {"mask": torch.from_numpy((owner > 0).astype(np.float32))[None, None],
            "kernel": kernels[:, :1], "kernels": kernels,
            "validity": torch.ones(1, 1, *owner.shape), "separation": torch.zeros(1, 1, *owner.shape)}


def test_loss_masks_ignore_and_propagates_all_head_gradients(cfg):
    batch = batch_from_owners(); batch["validity"][..., :4, :] = 0
    logits = torch.randn(1, 4, 64, 96, requires_grad=True)
    loss, terms, supervised = detection_loss(logits, batch, cfg)
    assert supervised and set(terms) == {"text", "kernel", "larger_kernels", "nesting"}
    loss.backward()
    assert not logits.grad[..., :4, :].any()
    assert all(logits.grad[:, i, 4:].abs().sum() > 0 for i in range(4))
    altered = logits.detach().clone(); altered[..., :4, :] = 100
    assert torch.allclose(loss.detach(), detection_loss(altered, batch, cfg)[0])


def test_all_ignored_batch_skips_optimizer_and_weight_decay(cfg):
    batch = batch_from_owners(); batch["validity"].zero_()
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.logit = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, x): return self.logit.expand(1, 4, 64, 96)
    model = Model(); optimizer = torch.optim.Adam(model.parameters(), lr=.1, weight_decay=.5)
    batch.update(image=torch.zeros(1, 3, 128, 192), sample_id=["fixture"],
                 native_words_lost=torch.tensor([1]), outside_image_words=torch.tensor([2]),
                 restored_contact_words=torch.tensor([3]))
    result = train_detection_epoch(model, [batch], optimizer, torch.device("cpu"), cfg)
    assert result["steps"] == 0 and result["skipped_unsupervised_batches"] == 1
    assert model.logit == 1. and not optimizer.state
    assert (result["native_words_lost"], result["outside_image_words"], result["restored_contact_words"]) == (1, 2, 3)


def synth_fixture(path, count=3):
    names, boxes, texts = [np.empty((1, count), object) for _ in range(3)]
    for i in range(count):
        name = f"image{i}.png"; Image.new("RGB", (80, 50), (40 + i, 70, 90)).save(path / name)
        names[0, i] = name
        boxes[0, i] = np.array([[10, 60, 60, 10], [12, 12, 35, 35]], float)
        texts[0, i] = "word"
    savemat(path / "gt.mat", {"imnames": names, "wordBB": boxes, "txt": texts})


def test_synthtext_manifest_is_deterministic_sha_bound_and_excludes_target_rgb(tmp_path):
    synth_fixture(tmp_path)
    excluded = {sha256_file(tmp_path / "image0.png")}
    args = dict(max_images=2, seed=202503, coordinate_origin=0, excluded_image_hashes=excluded)
    first = prepare_synthtext(tmp_path, **args); second = prepare_synthtext(tmp_path, **args)
    assert first == second and len(first["samples"]) == 2
    assert not excluded.intersection(r["image_sha256"] for r in first["samples"])
    assert first["gt_mat_sha256"] == sha256_file(tmp_path / "gt.mat")
    assert first["samples"][0]["words"][0]["polygon_xy"][0] == [10, 12]


def test_synthtext_real_mat_to_paired_training_batch(tmp_path):
    synth_fixture(tmp_path)
    c, source = read_config(DEFAULT_CONFIG, "approximate")
    manifest = prepare_synthtext(tmp_path, max_images=2, seed=17, coordinate_origin=0, excluded_image_hashes=set())
    dataset = SynthTextWordDataset(manifest, _transform(source, "letterbox", train=False), c["kernel_fractions"])
    batch = dataset[0]
    assert batch["image"].shape == (3, 512, 512) and batch["mask"].shape == (1, 256, 256)
    assert batch["native_words_lost"] == 0 and torch.all(batch["kernels"] <= batch["mask"])
    assert torch.all(batch["mask"] <= batch["validity"]) and batch["kernel"].any()
    Image.new("RGB", (80, 50), "red").save(manifest["samples"][0]["image_path"])
    with pytest.raises(ValueError, match="changed"): dataset[0]


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.full((2, 4), np.nan)])
def test_bad_synthtext_boxes_fail(bad):
    with pytest.raises(ValueError): decode_word_boxes(bad, "word", 0)


def test_synthtext_text_count_coordinates_and_missing_data(tmp_path):
    box = np.array([[1, 8, 8, 1], [2, 2, 9, 9]])
    assert decode_word_boxes(box, "word", 1)[0]["polygon_xy"][0] == [0, 1]
    with pytest.raises(ValueError, match="count"): decode_word_boxes(box, "two words", 0)
    with pytest.raises(FileNotFoundError, match="SynthText"):
        prepare_synthtext(tmp_path, max_images=1, seed=1, coordinate_origin=0, excluded_image_hashes=set())


def contact_words():
    return [{"polygon_xy": [[8, 10], [40, 10], [40, 35], [8, 35]], "transcription": "large"},
            {"polygon_xy": [[12, 9], [32, 9], [32, 9], [12, 9]], "transcription": "--"}]


def test_thin_word_restored_only_inside_original_polygon_and_not_in_separator():
    words = contact_words()
    with pytest.raises(ValueError, match=r"synthtext:fixture.*lost words \[2\]"):
        synthtext_word_targets((50, 40), words, policy="legacy_fail_closed", sample_id="synthtext:fixture")
    mask, valid, sep, owners, audit = synthtext_word_targets(
        (50, 40), words, policy="preserve_visible_words_v1", sample_id="synthtext:fixture")
    assert audit["restored_contact_word_ids"] == [2]
    assert audit["region_audit"]["empty_word_count"] == 0
    owner = np.asarray(owners)
    assert set(np.unique(owner)) == {0, 1, 2}
    for index, word in enumerate(words, 1):
        polygon = Image.new("1", (50, 40))
        ImageDraw.Draw(polygon).polygon([tuple(p) for p in word["polygon_xy"]], fill=1)
        assert np.all((owner == index) <= np.asarray(polygon))
    assert np.all(np.asarray(mask)[np.asarray(sep) > 0] == 0)
    assert np.all(np.asarray(valid) == 255)
    # The optional policy cannot leak into existing Total-Text/default behavior.
    from recognition.scene_text_datasets import SceneTextInstance
    from segmentation.totaltext_word_regions import make_word_regions
    instances = tuple(SceneTextInstance(tuple(tuple(p) for p in w["polygon_xy"]), w["transcription"], False, "word") for w in words)
    default = make_word_regions((50, 40), instances)
    explicit = make_word_regions((50, 40), instances, preserve_contact_words=False)
    assert default[-1] == explicit[-1] and default[-1].empty_word_count == 1
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(default[:-1], explicit[:-1]))


def test_off_image_words_are_audited_without_clamping_or_dropping_image():
    visible = {"polygon_xy": [[-3, 4], [7, 4], [7, 12], [-3, 12]], "transcription": "partial"}
    outside = {"polygon_xy": [[30, 4], [39, 4], [39, 12], [30, 12]], "transcription": "outside"}
    words = [outside, visible]
    mask, valid, sep, owners, audit = synthtext_word_targets(
        (30, 20), words, policy="preserve_visible_words_v1", sample_id="fixture")
    assert audit["outside_image_word_ids"] == [1] and audit["visible_words"] == 1
    assert audit["annotation_words"] == 2 and words[0]["polygon_xy"][0] == [30, 4]
    assert np.asarray(mask)[:, 29].sum() == 0 and np.asarray(owners)[7, 0] == 1
    assert np.all(np.asarray(valid) == 255)
    with pytest.raises(ValueError, match="no visible word"):
        synthtext_word_targets((30, 20), [outside], policy="preserve_visible_words_v1", sample_id="fixture")


def test_unrepresentable_duplicate_word_annotations_still_fail_with_sample_context():
    word = {"polygon_xy": [[8, 4], [19, 4], [19, 12], [8, 12]], "transcription": "same"}
    with pytest.raises(ValueError, match=r"duplicate-fixture.*lost words \[2\]"):
        synthtext_word_targets((30, 20), [word, deepcopy(word)],
                              policy="preserve_visible_words_v1", sample_id="duplicate-fixture")
    with pytest.raises(ValueError, match="Unknown SynthText"):
        synthtext_word_targets((30, 20), [word], policy="silent-skip", sample_id="fixture")


def test_audited_manifest_preserves_cohort_and_validates_worker_targets(tmp_path):
    synth_fixture(tmp_path, count=3)
    names, boxes, texts = (np.empty((1, 3), object) for _ in range(3))
    words = contact_words() + [{"polygon_xy": [[81, 4], [95, 4], [95, 15], [81, 15]], "transcription": "outside"}]
    for i in range(3):
        names[0, i] = f"image{i}.png"
        boxes[0, i] = np.stack([np.array(w["polygon_xy"]).T for w in words], axis=2)
        texts[0, i] = "large -- outside"
    savemat(tmp_path / "gt.mat", {"imnames": names, "wordBB": boxes, "txt": texts})
    options = dict(max_images=3, seed=17, coordinate_origin=0, excluded_image_hashes=set())
    legacy = prepare_synthtext(tmp_path, **options)
    manifest = prepare_synthtext(tmp_path, target_policy="preserve_visible_words_v1", **options)
    assert [(r["sample_id"], r["image_sha256"], r["words"]) for r in manifest["samples"]] == [
        (r["sample_id"], r["image_sha256"], r["words"]) for r in legacy["samples"]]
    assert manifest["target_summary"] == {"annotation_words": 9, "visible_words": 6,
                                           "outside_image_words": 3, "restored_contact_words": 3}
    manifest = json.loads(json.dumps(manifest))
    c, source = read_config(DEFAULT_CONFIG, "approximate")
    dataset = SynthTextWordDataset(manifest, _transform(source, "letterbox", train=False),
        c["kernel_fractions"], target_policy="preserve_visible_words_v1")
    batches = list(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0))
    assert sum(len(b["sample_id"]) for b in batches) == 3
    assert sum(int(b["outside_image_words"].sum()) for b in batches) == 3
    assert all(torch.isfinite(b["kernels"]).all() and torch.all(b["kernels"] <= b["mask"]) for b in batches)
    manifest["samples"][0]["target_audit"]["restored_contact_word_ids"] = []
    with pytest.raises(ValueError, match="target audit changed"): dataset[0]
    with pytest.raises(ValueError, match="manifest target policy differs"):
        SynthTextWordDataset(manifest, dataset.transform, c["kernel_fractions"])


def test_audited_synthtext_multiworker_transfer(tmp_path):
    """Requires host POSIX FD/socket permissions; never skip failed transfers."""
    synth_fixture(tmp_path, count=2)
    c, source = read_config(DEFAULT_CONFIG, "approximate")
    manifest = prepare_synthtext(tmp_path, max_images=2, seed=17, coordinate_origin=0,
        excluded_image_hashes=set(), target_policy="preserve_visible_words_v1")
    dataset = SynthTextWordDataset(manifest, _transform(source, "letterbox", train=False),
        c["kernel_fractions"], target_policy="preserve_visible_words_v1")
    # Spawn avoids inheriting OpenMP state from preceding loss tests.
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2,
                                        multiprocessing_context="spawn", timeout=30)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 512, 512)
    assert batch["native_words_lost"].sum() == 0
    assert torch.isfinite(batch["kernels"]).all()


def test_invalid_targets_fail_in_preparation_before_gpu_training(tmp_path):
    synth_fixture(tmp_path, count=1)
    word = np.array([[10, 30, 30, 10], [12, 12, 35, 35]])
    cells = np.empty((1, 1), object); cells[0, 0] = np.stack([word, word], axis=2)
    savemat(tmp_path / "gt.mat", {"imnames": np.array([["image0.png"]], object),
                                  "wordBB": cells, "txt": np.array([["one two"]], object)})
    with pytest.raises(ValueError, match=r"synthtext:0.*lost words"):
        prepare_synthtext(tmp_path, max_images=1, seed=1, coordinate_origin=0,
                          excluded_image_hashes=set(), target_policy="preserve_visible_words_v1")
