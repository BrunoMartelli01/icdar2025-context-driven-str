import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from models.adaspot_gap_context import (GapContextHead, bridge_strata,
    sampled_patches, stratum_weights)
import scripts.run_adaspot_gap_context as runner


@pytest.mark.parametrize("kernel", [1, 3])
def test_patch_and_native_values_and_gradients_match_at_edges(kernel):
    torch.manual_seed(42)
    head = GapContextHead(kernel, initial_bias=-2., channels=2)
    with torch.no_grad(): head.linear.weight.normal_()
    features = torch.randn(2, 4, 5)
    positions = torch.tensor([0, 4, 5, 12, 19])
    patches = sampled_patches(features, positions)
    full = F.unfold(features[None], 3, padding=1)[0].T
    assert torch.equal(patches, full[positions])
    point = head(patches)
    native = head.native_logits(features).flatten()[positions]
    assert torch.allclose(point, native, atol=1e-6)
    point.sum().backward(); gradient = head.linear.weight.grad.clone()
    head.zero_grad(); native.sum().backward()
    assert torch.allclose(gradient, head.linear.weight.grad, atol=1e-6)
    assert features.grad is None


def test_pointwise_ignores_neighbours_and_fresh_heads_have_identical_outputs():
    a, b = [GapContextHead(k, initial_bias=-2.) for k in (1, 3)]
    patches = torch.randn(3, 576)
    assert sum(p.numel() for p in a.parameters()) == 65
    assert sum(p.numel() for p in b.parameters()) == 577
    assert torch.equal(a(patches), b(patches))
    with torch.no_grad(): a.linear.weight.fill_(1.)
    modified = patches.clone(); modified[:, 0::9] += 20
    assert torch.equal(a(patches), a(modified))
    with pytest.raises(RuntimeError): b.load_state_dict(a.state_dict(), strict=True)


def test_sampling_retains_hard_and_easy_positives_and_never_uses_ignore():
    target = np.zeros((6, 6), bool); target[2, 1:5] = True
    valid = np.ones_like(target); valid[0] = False
    text = np.zeros_like(target); text[:, :3] = True
    groups = bridge_strata(target, valid, text, np.random.default_rng(42), 3)
    assert set(groups[0]) == set(np.flatnonzero(target & text))
    assert set(groups[1]) == set(np.flatnonzero(target & ~text))
    all_indices = np.concatenate(groups)
    assert len(all_indices) == len(set(all_indices))
    assert valid.ravel()[all_indices].all()
    assert text.ravel()[groups[2]].all() and not text.ravel()[groups[3]].any()
    again = bridge_strata(target, valid, text, np.random.default_rng(42), 3)
    assert all(np.array_equal(a, b) for a,b in zip(groups,again))


def test_negative_only_image_is_included_and_bad_targets_fail():
    target = np.zeros((4, 4), bool); valid = np.ones_like(target)
    text = target.copy(); text[0, 0] = True
    groups = bridge_strata(target, valid, text, np.random.default_rng(1), 2)
    assert len(groups[0]) == len(groups[1]) == 0
    assert len(groups[2]) == 1 and len(groups[3]) == 2
    target[0, 0] = True; valid[0, 0] = False
    with pytest.raises(ValueError): bridge_strata(target, valid, text, np.random.default_rng(1), 2)


def test_weights_focus_on_rare_hard_positives_and_keep_equal_stratum_mass():
    classes = torch.tensor([0] + [1]*10 + [2]*20 + [3]*30)
    weights, counts = stratum_weights(classes)
    assert weights[0] == 10*weights[1]
    assert torch.allclose(weights*counts, torch.full((4,), len(classes)/4))
    with pytest.raises(ValueError): stratum_weights(torch.tensor([0, 1, 2]))


def test_patch_bounds_and_shapes_fail_explicitly():
    with pytest.raises(ValueError): sampled_patches(torch.ones(2, 3, 4), [12])
    with pytest.raises(ValueError): GapContextHead(3, initial_bias=-2.)(torch.ones(1, 64))
    with pytest.raises(ValueError): GapContextHead(5, initial_bias=-2.)


def test_configuration_isolation_and_missing_stratum_do_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    config = json.loads((Path(__file__).resolve().parents[2]/"configs/approximate/adaspot_gap_context_v1.json").read_text())
    strict = tmp_path/"configs/strict/test.json"; strict.parent.mkdir(parents=True); strict.write_text(json.dumps(config))
    with pytest.raises(ValueError): runner.read_config(strict, "approximate")
    approximate = tmp_path/"configs/approximate/test.json"; approximate.parent.mkdir(parents=True); approximate.write_text(json.dumps(config))
    assert runner.read_config(approximate, "approximate")["epochs"] == 20
    with pytest.raises(ValueError): runner.read_config(approximate, "strict")


def test_checkpoint_selection_keeps_earliest_and_penalizes_fusions():
    def report(f1, fused): return {"aggregate":{"word_f1":f1,"gt_words_touched_by_fused_components":fused}}
    assert runner.selection_key(report(.6, 2), 0) > runner.selection_key(report(.6, 2), 20)
    assert runner.selection_key(report(.6, 1), 20) > runner.selection_key(report(.6, 2), 0)
    assert runner.selection_key(report(.7, 9), 20) > runner.selection_key(report(.6, 1), 0)


def execution_args(tmp_path, monkeypatch, device):
    repository=tmp_path/"repo";monkeypatch.setattr(runner,"REPO",repository)
    config=repository/"configs/approximate/test.json";config.parent.mkdir(parents=True)
    config.write_bytes((Path(__file__).resolve().parents[2]/"configs/approximate/adaspot_gap_context_v1.json").read_bytes())
    return Namespace(mode="approximate",config=config,source_run=tmp_path/"source",
        previous_probe=tmp_path/"previous",phase0_run=tmp_path/"phase0",word_root=tmp_path/"words",
        output_root=tmp_path/"output",device=device)


def test_existing_output_fails_before_mutation(tmp_path,monkeypatch):
    args=execution_args(tmp_path,monkeypatch,"cuda:0")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    args.output_root.mkdir();sentinel=args.output_root/"results.json";sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):runner.run(args)
    assert not getattr(args,"_output_created",False)
    assert sentinel.read_text()=="preserve" and not (args.output_root/"failure.json").exists()


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_no_cpu_fallback_or_output_when_gpu_unavailable(tmp_path,monkeypatch,device):
    args=execution_args(tmp_path,monkeypatch,device)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="GPU required"):runner.run(args)
    assert not args.output_root.exists()
