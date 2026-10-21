import numpy as np
import pytest
import torch

from models.adaspot_mask_align import MaskAlignOptions, _chain, mask_align_geometry, sample_mask_aligned
from scripts.diagnose_adaspot_mask_align import REPO, validate_config


CONFIG = REPO / "configs/approximate/adaspot_mask_align_v1.json"


def test_isolated_mode_and_predeclared_feature_interfaces():
    c = validate_config(CONFIG, "approximate")
    assert c["selected_source_arm"] == "full_selected"
    assert (c["expected_images"], c["expected_words"], c["expected_proposals"]) == (300, 2204, 2624)
    assert c["light_grid"] == [8, 32] and c["deep_grid"] == [16, 64]
    with pytest.raises(ValueError):
        validate_config(CONFIG, "strict")


def test_namespace_cannot_be_inferred_from_contents(tmp_path):
    foreign = tmp_path / CONFIG.name
    foreign.write_text(CONFIG.read_text())
    with pytest.raises(ValueError):
        validate_config(foreign, "approximate")


def test_digital_staircase_is_one_chain_without_diagonal_shortcuts():
    mask = np.zeros((8, 8), bool)
    pixels = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]
    for p in pixels:
        mask[p] = True
    path, reason = _chain(mask)
    assert reason is None and path.tolist() == [list(p) for p in pixels]


def test_disconnected_skeleton_is_rejected_even_with_two_endpoints():
    mask = np.zeros((8, 8), bool)
    mask[1, 1:5] = True
    mask[6, 6] = True
    _, reason = _chain(mask)
    assert reason == "disconnected_skeleton"


@pytest.mark.parametrize("values", [(0., .25, .5), (.5, 1.1, .5), (.5, .25, -1.), (float('nan'), .25, .5)])
def test_geometry_guards_are_explicit(values):
    with pytest.raises(ValueError):
        MaskAlignOptions(*values)


def test_feature_sampling_preserves_empty_proposal_population():
    features = torch.ones(1, 64, 16, 16)
    grids = torch.empty(0, 8, 32, 2)
    actual = sample_mask_aligned(features, grids, deterministic_cpu_backward=False)
    assert actual.shape == (0, 64, 8, 32)


def test_sampling_rejects_nonfinite_grid_and_mixed_precision():
    features = torch.ones(1, 64, 16, 16)
    grids = torch.ones(1, 8, 32, 2)
    grids[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        sample_mask_aligned(features, grids, deterministic_cpu_backward=False)
    with pytest.raises(ValueError):
        sample_mask_aligned(features.half(), torch.zeros_like(grids), deterministic_cpu_backward=False)


def test_border_instance_is_not_discarded():
    mask = np.zeros((20, 40), bool)
    mask[:8, :35] = True
    geometry = mask_align_geometry(mask, view="mask", options=MaskAlignOptions(.5, .25, .5))
    assert np.isfinite(geometry.grid((8, 32))).all()
    assert geometry.grid((8, 32))[4, 0, 0] < 1
