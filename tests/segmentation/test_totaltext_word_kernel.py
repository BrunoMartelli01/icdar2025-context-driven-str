import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
from models.unet_adv import PracticalTotalTextHighResAGUNet
from segmentation.totaltext_word_kernel import (
    expand_word_kernels, instance_geometry, instance_proposals,
    make_word_kernels, native_owner_map,
)
from scripts.train_totaltext_word_kernel_100ep import (
    _config, evaluate, native_instances, train_epoch, training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config, _transform
from scripts.evaluate_totaltext_word_kernel_parseq_val import _save_labels


def test_word_nuclei_are_connected_contained_and_do_not_modify_full_mask():
    owners = np.zeros((32, 32), np.int32)
    owners[0:12, 0:12] = 1  # image border
    owners[4:15, 15:30] = 2
    owners[25, 25] = 3       # punctuation survives
    original = owners.copy()
    nuclei, rows = make_word_kernels(owners)
    assert np.array_equal(owners, original)
    assert {r['owner_id'] for r in rows} == {1, 2, 3}
    for owner in (1, 2, 3):
        core = nuclei == owner
        assert core.any() and np.all(owners[core] == owner)
        assert cv2.connectedComponents(core.astype(np.uint8), connectivity=8)[0] == 2
    assert np.count_nonzero(nuclei == 1) < np.count_nonzero(owners == 1)


def test_curved_region_and_disconnected_core_keep_exactly_one_seed():
    owners = np.zeros((40, 40), np.int32)
    cv2.ellipse(owners, (20, 20), (14, 10), 0, 20, 160, 1, thickness=6)
    nuclei, rows = make_word_kernels(owners)
    assert len(rows) == 1 and np.any(nuclei)
    owners[:] = 0
    owners[3:12, 3:12] = owners[20:29, 20:29] = 1
    nuclei, rows = make_word_kernels(owners)
    assert rows[0]['core_fragments_before_selection'] == 2
    assert cv2.connectedComponents((nuclei > 0).astype(np.uint8), connectivity=8)[0] == 2


def test_roi_distance_target_equals_full_canvas_reference():
    owners = np.zeros((40, 60), np.int32)
    owners[0:15, 0:30] = 1
    cv2.ellipse(owners, (42, 27), (12, 8), 0, 20, 160, 2, thickness=6)
    nuclei, _ = make_word_kernels(owners)
    reference = np.zeros_like(owners)
    for owner in (1, 2):
        word = owners == owner
        distance = cv2.distanceTransform(np.pad(word.astype(np.uint8), 1),
            cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        squared_distance = np.rint(distance.astype(np.float64) ** 2).astype(np.int32)
        core = word & (squared_distance * 4 >= squared_distance.max())
        _, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
            core.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
        reference[labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] = owner
    assert np.array_equal(nuclei, reference)


def test_fused_foreground_is_partitioned_not_rebinarized():
    foreground = np.zeros((16, 30), bool)
    foreground[3:13, 2:28] = True
    nuclei = np.zeros_like(foreground)
    nuclei[6:10, 5:8] = nuclei[6:10, 22:25] = True
    labels, audit = expand_word_kernels(foreground, nuclei)
    assert audit['instances'] == 2 and audit['unseeded_fallback_components'] == 0
    assert np.array_equal(labels > 0, foreground)
    assert len(instance_proposals(labels)) == 2
    assert cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)[0] == 2
    assert np.array_equal(labels, expand_word_kernels(foreground, nuclei)[0])


def test_unseeded_foreground_is_kept_and_outside_seed_discarded():
    fg = np.zeros((12, 12), bool)
    fg[1:4, 1:4] = fg[7:10, 7:10] = True
    kernels = np.zeros_like(fg)
    kernels[2, 2] = kernels[0, 11] = True
    labels, audit = expand_word_kernels(fg, kernels)
    assert audit['seed_components'] == 1
    assert audit['unseeded_fallback_components'] == 1
    assert audit['kernel_pixels_outside_foreground'] == 1
    assert np.array_equal(labels > 0, fg)
    empty, empty_audit = expand_word_kernels(np.zeros_like(fg), kernels)
    assert not empty.any() and empty_audit['instances'] == 0


def test_geometry_counts_touching_instance_ids_separately():
    labels = np.zeros((10, 20), np.int32)
    labels[2:8, 1:10] = 1
    labels[2:8, 10:19] = 2
    metrics = instance_geometry(labels, labels, (1, 2))
    assert metrics['matched_gt_iou_ge_50'] == 2
    assert metrics['gt_words_touched_by_fused_components'] == 0
    merged = (labels > 0).astype(np.int32)
    assert instance_geometry(merged, labels, (1, 2))['gt_words_touched_by_fused_components'] == 2


def test_owner_mapping_replays_letterbox_and_validity_without_interpolated_ids():
    owner = np.zeros((20, 40), np.int32)
    owner[4:16, 4:15] = 3
    owner[4:16, 25:36] = 9
    transform = _transform(source_config(), 'letterbox', train=False)
    target = np.ones((1, 256, 256), np.float32)
    valid = target.copy()
    valid[:, :, 180:] = 0
    mapped = native_owner_map(Image.fromarray(owner), transform, 0, target, valid)
    assert set(np.unique(mapped)) <= {0, 3, 9}
    assert mapped[:60].sum() == 0 and mapped[196:].sum() == 0
    assert not mapped[:, 180:].any()


def test_protocol_is_train_only_frozen_and_model_keeps_initial_full_channel():
    config = _config()
    assert config['epochs'] == 100 and config['foreground_threshold'] == 0.65
    assert config['split'] == 'frozen_TRAIN_1130_validation_125'
    torch.manual_seed(5)
    baseline = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False)
    baseline_head = baseline.head.weight.detach().clone()
    del baseline
    torch.manual_seed(5)
    model = PracticalTotalTextWordKernelAGUNet(encoder_pretrained=False).eval()
    assert torch.equal(baseline_head, model.head.weight[:1])
    assert model.architecture_identity != PracticalTotalTextHighResAGUNet.PRACTICAL_IDENTITY
    assert sum(p.numel() for p in model.parameters()) == 58155146
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 32, 32))
    assert output.shape == (1, 2, 16, 16)


def test_training_uses_both_targets_and_dontcare_has_zero_gradient():
    model = torch.nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, 20:60, 20:60] = 1
    kernel = torch.zeros_like(mask)
    kernel[:, :, 30:50, 30:50] = 1
    batch = {'image': torch.ones(1, 3, 256, 256), 'mask': mask, 'kernel': kernel,
             'validity': torch.ones_like(mask), 'separation': torch.zeros_like(mask),
             'sample_id': ['TRAIN:1']}
    before = model.weight.detach().clone()
    result = train_epoch(model, [batch], optimizer, torch.device('cpu'), _config())
    assert result['steps'] == 1 and result['kernel_loss'] > 0
    assert torch.all((model.weight - before).reshape(2, -1).abs().sum(1) > 0)
    from scripts.train_totaltext_letterbox_word_f1_separator_head import balanced_separator_bce
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    valid = torch.tensor([[[[1., 1.], [0., 1.]]]])
    balanced_separator_bce(logits, torch.zeros_like(logits), valid).backward()
    assert logits.grad[0, 0, 1, 0] == 0


def test_prediction_threshold_and_canvas_and_lossless_instance_cache(tmp_path):
    logits = torch.full((2, 256, 256), -10.)
    logits[0, 20:30, 20:50] = 2
    logits[1, 23:27, 22:25] = 2
    logits[1, 23:27, 45:48] = 2
    canvas = np.ones((256, 256), bool)
    canvas[:20] = False
    labels, audit, _ = native_instances(logits, canvas, _config())
    assert audit['instances'] == 2 and not labels[:20].any()
    path = tmp_path / 'labels.png'
    recorded = _save_labels(path, labels)
    assert recorded == _save_labels(path, labels)
    with Image.open(path) as opened:
        assert np.array_equal(labels, np.asarray(opened))
    with pytest.raises(ValueError, match='differs'):
        _save_labels(path, labels + 1)


def test_invalid_shapes_and_kernel_parameters_fail_closed():
    with pytest.raises(ValueError):
        make_word_kernels(np.ones((3, 3), np.float32))
    with pytest.raises(ValueError):
        make_word_kernels(np.zeros((3, 3), np.int32), distance_fraction=1)
    with pytest.raises(ValueError):
        expand_word_kernels(np.ones((3, 3), np.uint8), np.ones((3, 3), bool))


def test_validation_preserves_instance_partition_through_matching():
    logits = torch.full((1, 2, 256, 256), -10.)
    logits[:, 0, 5:15, 2:23] = 10.  # fused complete prediction
    logits[:, 1, 8:12, 5:8] = logits[:, 1, 8:12, 18:21] = 10.
    owners = np.zeros((256, 256), np.int32)
    owners[5:15, 2:12] = 1
    owners[5:15, 13:23] = 2
    class Fixed(torch.nn.Module):
        def forward(self, images):
            return logits
    batch = {'image': torch.zeros(1, 3, 512, 512),
             'mask': torch.from_numpy((owners > 0).astype(np.float32))[None, None],
             'validity': torch.ones(1, 1, 256, 256), 'sample_id': ['TRAIN:synthetic']}
    references = {'TRAIN:synthetic': (owners, (1, 2), np.ones((256, 256), bool))}
    metrics = evaluate(Fixed(), [batch], references, torch.device('cpu'), _config())['aggregate']
    assert metrics['matched_gt_iou_ge_50'] == 2
    assert metrics['raw_proposals'] == 2 and metrics['proposal_f1_iou_50'] == 1
    assert metrics['gt_words_touched_by_fused_components'] == 0


def test_checkpoint_roundtrip_restores_strict_model_optimizer_and_rng(tmp_path):
    from segmentation.training import save_checkpoint, load_checkpoint
    class Tiny(torch.nn.Conv2d):
        architecture_identity = 'synthetic-word-kernel-smoke'
    model = Tiny(3, 2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    tc = training_config(source_config())
    contract = {'split': 'TRAIN_only', 'test': 'resume smoke'}
    path = tmp_path / 'last.pt'
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, global_step=1,
                    config=tc, resolved_config='{}', architecture=model.architecture_identity,
                    run_contract=contract, train_metrics={'loss': 1.}, validation_metrics={'score': 2.})
    expected = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.zero_()
    payload = load_checkpoint(path, model=model, optimizer=optimizer,
        expected_architecture=model.architecture_identity, expected_run_contract=contract,
        expected_training_config=tc, restore_rng=True)
    assert payload['epoch'] == 1 and torch.equal(model.weight, expected)
    with pytest.raises(ValueError):
        load_checkpoint(path, model=model, expected_architecture=model.architecture_identity,
                        expected_run_contract={'split': 'TEST'})
