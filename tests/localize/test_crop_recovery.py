import json
from pathlib import Path

import numpy as np
import pytest

from localize.crop_recovery import (CandidatePool, add_border_variants, add_tile_candidates,
    geodesic_partition, mask_candidates, render_crop, tile_windows, vertical_seam)


CONFIG = Path(__file__).resolve().parents[2] / "configs/approximate/totaltext_crop_recovery_suite_train_v1.json"


def config():
    return json.loads(CONFIG.read_text())


def bridge():
    p = np.zeros((32, 32), np.float32)
    p[8:18, 3:12] = .95
    p[8:18, 20:29] = .95
    p[12:14, 12:20] = .55
    return p


def test_high_threshold_seeds_reconstruct_both_words_and_keep_parent():
    p = bridge()
    image = np.zeros((32, 32, 3), np.uint8)
    pool = mask_candidates(p, image, config())
    base = [r for r in pool.rows.values() if "baseline" in r["methods"]]
    assert len(base) == 1 and base[0]["bbox"] == [3, 8, 29, 18]
    split = [r for r in pool.rows.values() if "threshold_tree" in r["methods"]]
    assert any(r["bbox"] == [3, 8, 12, 18] for r in split)
    assert any(r["bbox"] == [20, 8, 29, 18] for r in split)
    reconstructed = [r for r in split if any("seed_high" in o for o in r["origins"])]
    assert len(reconstructed) >= 2


def test_flood_assigns_full_support_and_never_crosses_background():
    p = bridge()
    support = p >= .5
    seeds = np.zeros(p.shape, np.int32)
    seeds[10, 6] = 1
    seeds[10, 25] = 2
    labels = geodesic_partition(p, support, seeds)
    assert np.all(labels[support] > 0)
    assert np.all(labels[~support] == 0)
    assert labels[10, 6] == 1 and labels[10, 25] == 2
    assert np.all(labels[8:18, 3:12] == 1)
    assert np.all(labels[8:18, 20:29] == 2)
    with pytest.raises(ValueError, match="outside"):
        geodesic_partition(p, np.zeros_like(support), seeds)


def test_erosion_finds_bridge_without_returning_eroded_cores():
    p = bridge()
    pool = mask_candidates(p, np.zeros((32, 32, 3), np.uint8), config())
    rows = [r for r in pool.rows.values() if "erosion_reconstruction" in r["methods"]]
    assert len(rows) >= 2
    assert any(r["bbox"][1:4:2] == [8, 18] for r in rows)


def test_seam_follows_low_cost_curved_corridor():
    cost = np.ones((7, 9), np.float32)
    expected = np.array([2, 2, 3, 4, 4, 3, 2])
    cost[np.arange(7), expected] = 0
    assert np.array_equal(vertical_seam(cost), expected)


def test_rgb_seams_generate_both_sides_and_preserve_parent():
    p = bridge()
    pool = mask_candidates(p, np.zeros((32, 32, 3), np.uint8), config())
    rows = [r for r in pool.rows.values() if "image_seams" in r["methods"]]
    assert any(r["bbox"][0] == 3 and r["bbox"][2] < 20 for r in rows)
    assert any(r["bbox"][0] > 12 and r["bbox"][2] == 29 for r in rows)
    assert any("baseline" in r["methods"] for r in pool.rows.values())


def test_tile_coverage_and_letterbox_inverse_include_offset():
    windows = tile_windows((103, 61), 2, .25)
    coverage = np.zeros((61, 103), int)
    for x1, y1, x2, y2 in windows:
        coverage[y1:y2, x1:x2] += 1
    assert np.all(coverage > 0) and np.any(coverage > 1)
    pool = CandidatePool((100, 60))
    p = np.zeros((32, 32), np.float32)
    # Local window is 40x20, letterboxed with 10px top/bottom.
    p[8:24, :] = 1
    add_tile_candidates(pool, p, [20, 10, 60, 30], [.5], "test")
    assert list(pool.rows.values())[0]["bbox"] == [20, 10, 60, 30]
    with pytest.raises(ValueError):
        tile_windows((10, 10), 2, 1.)


def test_pool_deduplicates_recipes_but_retains_method_membership():
    pool = CandidatePool((20, 20))
    pool.add([2, 2, 10, 8], "baseline", {})
    pool.add([2, 2, 10, 8], "threshold_tree", {"threshold": .7})
    assert len(pool.rows) == 1
    assert set(next(iter(pool.rows.values()))["methods"]) == {"baseline", "threshold_tree"}
    pool.add([2, 2, 10, 8], "crop_variants", {}, neutral_fraction=.1)
    assert len(pool.rows) == 2


def test_border_variants_shrink_one_side_without_expanding_other_side():
    pool = CandidatePool((100, 100))
    pool.add([20, 20, 80, 40], "baseline", {})
    add_border_variants(pool, config()["variants"])
    boxes = [r["bbox"] for r in pool.rows.values()]
    assert [22, 20, 80, 40] in boxes
    assert [20, 20, 78, 40] in boxes
    assert [18, 20, 80, 40] in boxes


def test_oriented_and_neutral_rendering_are_deterministic():
    image = np.full((20, 30, 3), 100, np.uint8)
    image[5:15, 4:24] = 200
    row = {"bbox": [4, 5, 24, 15], "quad": [[4., 5.], [24., 5.], [24., 15.], [4., 15.]],
           "neutral_fraction": .1}
    crop = render_crop(image, row)
    assert crop.shape == (12, 22, 3)
    assert np.array_equal(crop, render_crop(image, row))
    assert crop[0, 0, 0] >= 100


def test_native_square_is_inverted_into_nonsquare_original():
    p = np.zeros((32, 32), np.float32)
    p[8:24, 4:20] = .95
    pool = mask_candidates(p, np.zeros((20, 40, 3), np.uint8), config())
    assert any(r["bbox"] == [5, 0, 25, 20] for r in pool.rows.values() if "baseline" in r["methods"])
