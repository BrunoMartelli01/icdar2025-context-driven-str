import numpy as np
from PIL import Image
import pytest
import torch

from localize.word_instance_extensions import affinity_component, mask_component, remap_local_box
from models.word_instance_extensions import LocalWordModel
from segmentation.word_instance_data import inference_packets, training_targets
from segmentation.word_instance_extensions import (graph_edges, maximin_loss, maximin_weights,
    prompt_supervision, mask_objective)


def test_prompt_targets_select_one_word_and_keep_neighbor_as_negative():
    owners = np.array([[1, 1, 0, 2, 2], [1, 1, 0, 2, 2]])
    valid = np.ones_like(owners, bool)
    prompt = np.zeros_like(valid); prompt[0, 0] = True
    target, kept, audit = prompt_supervision(owners, valid, prompt, [0, 0, 5, 2],
                                            minimum_majority=.5, minimum_visible=.95)
    assert audit["owner_id"] == 1 and audit["unsupervised_reason"] is None
    assert kept.all() and target.sum() == 4 and not target[:, 3:].any()


@pytest.mark.parametrize("case", ["ignore", "ambiguous", "truncated", "background"])
def test_invalid_prompts_are_unsupervised_not_negative_training_labels(case):
    owners = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    valid = np.ones_like(owners, bool); prompt = np.zeros_like(valid); prompt[0, 1] = True
    window = [0, 0, 4, 2]
    if case == "ignore": valid[prompt] = False
    if case == "ambiguous": prompt[0, 2] = True
    if case == "truncated": window = [0, 0, 1, 2]
    if case == "background": owners[prompt] = 0
    _, kept, audit = prompt_supervision(owners, valid, prompt, window,
                                       minimum_majority=.5, minimum_visible=.95)
    assert not kept.any() and audit["unsupervised_reason"]


def test_maximin_gradients_raise_internal_links_and_lower_merge_bridge():
    z = torch.zeros(4, 1, 4, requires_grad=True)
    with torch.no_grad(): z[0, 0, :3] = torch.tensor([4., 3., 4.])
    owners = np.array([[1, 1, 2, 2]])
    loss = maximin_loss(z, owners, np.ones_like(owners, bool), anchors_per_word=16)
    loss.backward()
    assert z.grad[0, 0, 1] > 0  # high-confidence edge joining two different words
    assert z.grad[0, 0, 0] < 0 and z.grad[0, 0, 2] < 0


def test_unconstrained_maximin_can_penalize_internal_bottleneck_of_a_merged_path():
    z = torch.zeros(4, 1, 4, requires_grad=True)
    with torch.no_grad(): z[0, 0, :3] = torch.tensor([3., 4., 3.])
    loss = maximin_loss(z, np.array([[1, 1, 2, 2]]), np.ones((1, 4), bool), anchors_per_word=16)
    loss.backward()
    # This is an explicit limitation of the unconstrained maximin objective.
    assert z.grad[0, 0, 0] > 0


def test_maximin_uses_best_path_instead_of_misclassified_direct_edge():
    # The direct top edge is weak; the two-edge diagonal/vertical alternative is strong.
    z = torch.full((4, 2, 2), -5.)
    z[0, 0, 0] = -2
    z[1, 0, 0] = 2
    z[2, 0, 0] = 3
    z[1, 0, 1] = 1
    z[0, 1, 0] = 2
    owners = np.array([[1, 1], [0, 0]])
    positions, pos, neg = maximin_weights(z, owners, np.ones((2, 2), bool), anchors_per_word=16)
    assert pos.sum() == pytest.approx(.25) and neg.sum() == 0
    assert torch.min(z.flatten()[positions[pos > 0]]) > 0
    assert 0 not in positions[pos > 0]


def test_maximin_weights_equal_bruteforce_widest_path_pair_loss():
    # Distinct logits avoid ambiguity in selected bottleneck edges.
    owners = np.array([[1, 1, 0], [0, 2, 2]])
    valid = np.ones_like(owners, bool)
    z = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)/7-1
    u, v, locations = graph_edges(owners.shape, valid)
    widest = np.full((6, 6), -np.inf); np.fill_diagonal(widest, np.inf)
    for a, b, p in zip(u, v, locations): widest[a, b] = widest[b, a] = float(z.flatten()[p])
    for k in range(6): widest = np.maximum(widest, np.minimum(widest[:, k, None], widest[None, k, :]))
    positive, negative = [], []
    for a in np.flatnonzero(owners):
        for b in np.flatnonzero(owners):
            if a >= b: continue
            (positive if owners.flat[a] == owners.flat[b] else negative).append(float(np.logaddexp(0, (-1 if owners.flat[a] == owners.flat[b] else 1)*widest[a, b])))
    expected = (np.mean(positive)+np.mean(negative))/2
    actual = maximin_loss(z, owners, valid, anchors_per_word=16)
    assert float(actual) == pytest.approx(expected, abs=1e-6)


def test_ignored_graph_barrier_carries_no_pairs_or_gradient():
    owners = np.array([[1, 0, 2]])
    valid = np.array([[True, False, True]])
    z = torch.randn(4, 1, 3, requires_grad=True)
    loss = maximin_loss(z, owners, valid, anchors_per_word=16)
    loss.backward()
    assert float(loss) == 0 and torch.count_nonzero(z.grad) == 0


def test_mask_and_affinity_choose_prompt_component_without_losing_isolated_pixels():
    p = np.zeros((3, 5)); p[1, 0] = p[1, 4] = 1
    prompt = p.astype(bool); prompt[1, 4] = False
    result = mask_component(p, prompt, np.ones_like(prompt), threshold=.5)
    assert result.sum() == 1 and result[1, 0]
    graph = affinity_component(np.zeros((4, 3, 5)), p.astype(bool), prompt, threshold=.5)
    np.testing.assert_array_equal(result, graph)


def test_local_coordinates_invert_odd_letterbox_and_outward_round():
    # 9x5 -> top padding2, a full native9 window and local18 pixels.
    box = remap_local_box([2, 6, 10, 12], [0, 0, 9, 9], (9, 5), local_size=18, native_size=9)
    assert box == [1, 1, 5, 4]


def test_gt_cannot_change_packet_geometry_and_neighbors_remain_mask_negatives():
    owners = np.zeros((8, 8), int); owners[2:5, 1:3] = 1; owners[2:5, 4:6] = 2
    p = np.stack((owners > 0, owners > 0)).astype(np.float32)
    config = {"local_size": 16, "canvas_fill_rgb": [0, 0, 0], "context_fraction": 1.,
              "minimum_context_pixels": 1, "kernel_threshold": .5, "owner_majority": .5,
              "minimum_visible_fraction": .95}
    rgb = Image.new("RGB", (8, 8))
    packets = inference_packets(rgb, p, owners, np.ones((8, 8), bool), config)
    before = packets[0]["inputs"].clone()
    gt = [{"text": "one"}, {"text": "two"}]
    targets = training_targets(packets[0], owners, np.ones((8, 8), bool), gt, config)
    training_targets(packets[0], np.zeros_like(owners), np.ones((8, 8), bool), gt, config)
    torch.testing.assert_close(before, packets[0]["inputs"], rtol=0, atol=0)
    assert targets["text"] == "one" and targets["mask_valid"].all()


@pytest.mark.parametrize("kind,outputs", [("mask", 1), ("affinity", 4), ("geometry", 4)])
def test_private_models_have_trainable_gradients_and_strict_checkpoint_roundtrip(kind, outputs):
    torch.manual_seed(7)
    model = LocalWordModel(kind, 4)
    x = torch.randn(1, 5, 8, 8)
    box = torch.tensor([[1., 1., 7., 7.]])
    out = model(x, box)
    assert out.shape == ((1, 4) if kind == "geometry" else (1, outputs, 8, 8))
    if kind == "mask": loss = mask_objective(out, torch.ones_like(out), torch.ones_like(out, dtype=torch.bool))
    else: loss = (out-1).square().mean()
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    other = LocalWordModel(kind, 4); other.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(out, other(x, box), rtol=0, atol=0)
