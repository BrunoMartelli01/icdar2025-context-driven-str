import numpy as np

from localize.crop_recovery import adjacent_groups, adjacent_groups_with_members
from localize.structural_word_groups import (TraceCollector, TraceEvent, _merge,
                                             admit_groups)


def _row(cid, box, origins=None):
    origins = origins or [{"method": "threshold_tree", "threshold": .5}]
    return {"candidate_id": cid, "bbox": box, "origins": origins,
            "methods": sorted({origin["method"] for origin in origins})}


def _group(cid, box, *, parent=False, shared=()):
    return {"group_id": cid, "core_candidate_ids": [cid],
            "candidate_ids": sorted({cid, *shared}), "reference_candidate_id": cid,
            "reference_bbox": box, "parent_only": parent}


def _event(cid, pixels, *, view="whole", kind="component", support=None,
           seeds=(), border=False):
    return TraceEvent(cid, "threshold_tree" if view == "whole" else "tiled_segmentation",
                      kind, view, frozenset(pixels), .5, support, seeds, border)


def test_instrumented_seed_unions_preserve_original_region_order_and_pixels():
    regions = [np.zeros((5, 9), bool) for _ in range(3)]
    for index, region in enumerate(regions):
        region[2, 1 + index * 3] = True
    original = adjacent_groups(regions, 3)
    traced = adjacent_groups_with_members(regions, 3)
    assert len(original) == len(traced)
    assert all(np.array_equal(a, b) for a, (b, _) in zip(original, traced))
    assert [len(ids) for _, ids in traced] == [1, 1, 1, 2, 2, 3]


def test_lineage_merges_nested_support_but_not_a_disjoint_neighbor():
    rows = {row["candidate_id"]: row for row in (
        _row("a", [0, 0, 12, 10]), _row("b", [1, 0, 13, 10]),
        _row("c", [13, 0, 25, 10]))}
    groups = [_group(cid, row["bbox"]) for cid, row in rows.items()]
    events = {"a": [_event("a", range(100))],
              "b": [_event("b", range(5, 100))],
              "c": [_event("c", range(100, 200))]}
    result = _merge(groups, rows, events, 1, {cid: cid for cid in rows})
    assert len(result) == 2
    assert any(set(group["core_candidate_ids"]) == {"a", "b"} for group in result)


def test_tile_border_partial_is_linked_without_merging_disjoint_words():
    rows = {row["candidate_id"]: row for row in (
        _row("whole", [0, 0, 20, 10]),
        _row("tile", [0, 0, 8, 10], [{"method": "tiled_segmentation", "tile_id": "t", "threshold": .5}]),
        _row("other", [21, 0, 31, 10]))}
    groups = [_group(cid, row["bbox"]) for cid, row in rows.items()]
    events = {"whole": [_event("whole", range(100))],
              "tile": [_event("tile", range(25), view="t", border=True)],
              "other": [_event("other", range(100, 150))]}
    result = _merge(groups, rows, events, 2, {cid: cid for cid in rows})
    assert len(result) == 2
    assert any(set(group["core_candidate_ids"]) == {"whole", "tile"} for group in result)


def test_multiword_parent_is_shared_then_its_redundant_group_is_removed():
    rows = {row["candidate_id"]: row for row in (
        _row("left", [0, 0, 10, 10]), _row("right", [10, 0, 20, 10]),
        _row("parent", [0, 0, 20, 10]))}
    groups = [_group("left", rows["left"]["bbox"], shared=("parent",)),
              _group("right", rows["right"]["bbox"], shared=("parent",)),
              _group("parent", rows["parent"]["bbox"], parent=True)]
    events = {"left": [_event("left", range(30))],
              "right": [_event("right", range(50, 80))],
              "parent": [_event("parent", range(80, 100))]}
    result = _merge(groups, rows, events, 3, {cid: cid for cid in rows})
    assert len(result) == 2
    assert all("parent" in group["candidate_ids"] for group in result)


def test_seed_pieces_join_only_with_a_compact_gap_and_composite_ocr_agreement():
    support = (.35, .65, 1)
    first = {y * 256 + x for y in range(10, 14) for x in range(10, 14)}
    second = {y * 256 + x for y in range(10, 14) for x in range(14, 18)}
    rows = {row["candidate_id"]: row for row in (
        _row("a", [10, 10, 14, 14]), _row("b", [14, 10, 18, 14]),
        _row("ab", [10, 10, 18, 14]))}
    groups = [_group("a", rows["a"]["bbox"]), _group("b", rows["b"]["bbox"])]
    events = {"a": [_event("a", first, kind="seed_piece", support=support, seeds=(1,))],
              "b": [_event("b", second, kind="seed_piece", support=support, seeds=(2,))],
              "ab": [_event("ab", first | second, kind="seed_union", support=support, seeds=(1, 2))]}
    assert len(_merge(groups, rows, events, 3, {"a": "A", "b": "B", "ab": "AB"})) == 1
    assert len(_merge(groups, rows, events, 3, {"a": "A", "b": "B", "ab": "A B"})) == 2


def test_admission_keeps_consensus_or_image_relative_isolated_word():
    rows = {row["candidate_id"]: row for row in (
        _row("multi", [0, 0, 10, 10], [{"method": "threshold_tree", "threshold": .5},
                                           {"method": "tiled_segmentation", "tile_id": "t", "threshold": .5}]),
        _row("isolated", [20, 0, 30, 10]), _row("empty", [40, 0, 50, 10]))}
    groups = [{"group_id": cid, "candidate_ids": [cid]} for cid in rows]
    kept = admit_groups(groups, rows,
                        {"multi": "word", "isolated": "other", "empty": ""},
                        {"multi": .2, "isolated": .8, "empty": .1})
    assert {group["group_id"] for group in kept} == {"multi", "isolated"}


def test_trace_footprints_use_shared_image_coordinates():
    whole = TraceCollector((256, 256), "image")
    mask = np.zeros((256, 256), bool)
    mask[20:30, 40:50] = True
    whole("a", mask, (0, 0), (0, 0, 256, 256),
          {"method": "threshold_tree", "kind": "component", "threshold": .5, "native_size": 256})
    tile = TraceCollector((256, 256), "image")
    tile_mask = np.zeros((256, 256), bool)
    tile_mask[40:60, 80:100] = True
    tile("b", tile_mask, (0, 0), (0, 0, 128, 128),
         {"method": "tiled_segmentation", "kind": "component", "view": "t", "threshold": .5,
          "native_size": 256})
    assert whole.events["image:a"][0].pixels == tile.events["image:b"][0].pixels


def test_tile_letterbox_pixels_outside_window_do_not_enter_footprint():
    collector = TraceCollector((256, 256), "image")
    mask = np.zeros((256, 256), bool)
    mask[0, 40] = True  # In the letterbox: projects inside the image, outside the tile.
    mask[80, 40] = True  # Inside the tile.
    collector("crop", mask, (0, 0), (50, 100, 178, 164),
              {"method": "tiled_segmentation", "kind": "component", "view": "tile",
               "threshold": .5, "native_size": 256})
    event = collector.events["image:crop"][0]
    assert len(event.pixels) == 1
    assert next(iter(event.pixels)) // 256 >= 100
