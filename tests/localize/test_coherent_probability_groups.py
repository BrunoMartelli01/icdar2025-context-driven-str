import numpy as np
import pytest

from localize.coherent_probability_groups import CoherenceParams, CorridorParams, coherent_group, group_components
from localize.grouped_word_blocks import GroupedWordBlockParams, RawComponent, group_word_blocks


V1 = GroupedWordBlockParams(.35, .7, .5, 2.5, 6)
COH = CoherenceParams(1.5, 3)
COR = CorridorParams(.2, .2)


def mask(boxes):
    value = np.zeros((128, 128), dtype=np.uint8)
    for x, y, w, h in boxes: value[y:y+h, x:x+w] = 1
    return value


def run(binary, mode="GROUP_COHERENCE", support=None):
    return group_components(binary, V1, mode=mode, coherence=COH, corridor=COR, support=support)


def test_global_band_blocks_drifting_chain_not_long_horizontal_word():
    drifting = mask([(10, 10, 5, 10), (17, 12, 5, 10), (24, 14, 5, 10)])
    assert len(group_word_blocks(drifting, drifting, V1).retained) == 1
    assert len(run(drifting)["boxes"]) == 2
    straight = mask([(10, 10, 5, 10), (17, 10, 5, 10), (24, 10, 5, 10), (31, 10, 5, 10)])
    assert len(run(straight)["boxes"]) == 1


def test_spacing_gate_and_short_group_fallback():
    binary = mask([(10, 10, 5, 10), (16, 10, 5, 10), (22, 10, 5, 10), (30, 10, 5, 10)])
    result = run(binary)
    assert len(result["boxes"]) == 2
    assert any("global_spacing_outlier" in d.get("reasons", []) for d in result["decisions"])
    assert len(run(mask([(10, 10, 5, 10), (18, 10, 5, 10)]))["boxes"]) == 1


def test_small_fragment_does_not_define_global_core():
    members = [RawComponent(1, 50, (10, 10, 15, 20), (12, 15)),
               RawComponent(2, 50, (17, 10, 22, 20), (19, 15)),
               RawComponent(3, 2, (20, 8, 21, 10), (20, 9))]
    decision = coherent_group(members, V1, COH)
    assert decision["core_component_ids"] == [1, 2]
    assert decision["accepted"]


def test_corridor_uses_real_support_not_binary_or_gt():
    binary = mask([(10, 10, 5, 10), (17, 10, 5, 10)])
    support = binary.astype(np.float32)
    assert len(run(binary, "PROBABILITY_CORRIDOR", support)["boxes"]) == 2
    support[10:20, 15:17] = .3
    assert len(run(binary, "PROBABILITY_CORRIDOR", support)["boxes"]) == 1
    support[10:20, 16] = 0  # background barrier despite adequate mean elsewhere
    assert len(run(binary, "PROBABILITY_CORRIDOR", support)["boxes"]) == 2


@pytest.mark.parametrize("mode", ["GROUP_COHERENCE", "PROBABILITY_CORRIDOR"])
def test_empty_singleton_top10_border_determinism_no_mutation(mode):
    for boxes, expected in [([], 0), ([(0, 0, 5, 10)], 1),
                            ([(30*x, 40*y, 5, 10) for x in range(4) for y in range(3)], 10)]:
        binary = mask(boxes); before = binary.copy(); support = binary.astype(np.float32)
        a = run(binary, mode, support); b = run(binary, mode, support)
        assert a == b and len(a["boxes"]) == expected
        assert np.array_equal(binary, before)
        assert all(0 <= x1 < x2 <= 128 and 0 <= y1 < y2 <= 128 for x1,y1,x2,y2 in a["boxes"])


@pytest.mark.parametrize("bad", [None, np.full((128,128), .6), np.full((128,128), np.nan), np.zeros((64,64))])
def test_probability_missing_mismatched_or_invalid_fails(bad):
    with pytest.raises(ValueError): run(mask([(10,10,5,10)]), "PROBABILITY_CORRIDOR", bad)


def test_invalid_config_and_mask_fail():
    with pytest.raises(ValueError): CoherenceParams(.5, 3)
    with pytest.raises(ValueError): CorridorParams(-.1, .2)
    with pytest.raises(ValueError): run(np.full((128,128), 255, dtype=np.uint8))
