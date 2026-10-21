import numpy as np

from recognition.scene_text_datasets import SceneTextInstance
from scripts.diagnose_totaltext_original_gt_components_parseq import (
    _components, _normalized, _owner_map, _word_order,
)


def _word(points, text):
    return SceneTextInstance(tuple(points), text, False, "word")


def test_full_resolution_contours_keep_small_components_without_top10():
    mask = np.zeros((20, 40), dtype=np.uint8)
    for x in range(1, 24, 2):
        mask[4, x] = 1
    owner = _owner_map((40, 20), [_word(((0, 0), (25, 0), (25, 10), (0, 10)), "WORD")])
    components = _components(mask, owner)
    assert len(components) == 12
    assert all(item["foreground_pixels"] == 1 for item in components)
    assert all(item["word_id"] == 1 for item in components)
    assert len(_word_order(components, "h")) == 12
    assert [components[i]["bbox"][0] for i in _word_order(components, "h")] == list(range(1, 24, 2))


def test_cross_word_component_is_unassigned():
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[5, 3:17] = 1
    words = [_word(((0, 0), (8, 0), (8, 11), (0, 11)), "A"),
             _word(((11, 0), (19, 0), (19, 11), (11, 11)), "B")]
    components = _components(mask, _owner_map((20, 12), words))
    assert len(components) == 1
    assert components[0]["word_id"] is None
    assert components[0]["touched_word_ids"] == [1, 2]


def test_component_nested_inside_hole_is_not_lost():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:18, 2:18] = 1
    mask[5:15, 5:15] = 0
    mask[9:11, 9:11] = 1
    owner = _owner_map((20, 20), [_word(((0, 0), (19, 0), (19, 19), (0, 19)), "TEST")])
    components = _components(mask, owner)
    assert len(components) == 2
    assert sorted(item["foreground_pixels"] for item in components) == [4, 156]


def test_vertical_order_and_punctuation_normalization():
    components = [{"bbox": [5, 8, 7, 10]}, {"bbox": [3, 2, 5, 4]}]
    assert _word_order(components, "v") == [1, 0]
    assert _normalized("Fisherman's", "punctuation") == _normalized("FISHERMANS", "punctuation")
    assert _normalized("TV-GAME", "casefold") != _normalized("TVGAME", "casefold")
