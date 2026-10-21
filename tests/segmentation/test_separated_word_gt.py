from pathlib import Path

import numpy as np
import pytest

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.separated_word_gt import (
    audit_word_owners, parse_icdar13_character_groups, separated_word_regions, widen_word_gaps,
)
from PIL import Image


def word(points, text="word", ignore=False, granularity="word"):
    return SceneTextInstance(tuple(points), text, ignore, granularity)


def test_touching_and_overlapping_words_survive_as_separate_components():
    instances = (word([(1, 1), (14, 1), (14, 8), (1, 8)], "a"),
                 word([(12, 1), (24, 1), (24, 8), (12, 8)], "b"),
                 word([(24, 8), (30, 8), (30, 15), (24, 15)], "c"))
    mask, valid, separator, owners, audit = separated_word_regions((32, 18), instances)
    owner = np.asarray(owners)
    assert set(np.unique(owner)) == {0, 1, 2, 3}
    assert audit["interword_contact_pairs"] == 0
    assert audit["components_per_word"] == {"1": 1, "2": 1, "3": 1}
    assert np.array_equal(np.asarray(mask) > 0, owner > 0)
    assert np.all(np.asarray(mask)[np.asarray(separator) > 0] == 0)
    assert np.all(np.asarray(valid) == 255)
    repeated = separated_word_regions((32, 18), instances)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(repeated[:4],
                [mask, valid, separator, owners]))


def test_all_ignore_image_is_preserved_as_empty_target():
    result = separated_word_regions((12, 12), (word([(2, 2), (9, 2), (9, 9), (2, 9)], "###", True),))
    mask, valid, _, owners, audit = result
    assert not np.asarray(mask).any()
    assert not np.asarray(owners).any()
    assert np.asarray(valid)[4, 4] == 0
    assert audit["words"] == 0 and audit["ignored_instances"] == 1


def test_text_line_cannot_be_silently_treated_as_word():
    with pytest.raises(ValueError, match="text-line"):
        separated_word_regions((10, 10), (word([(1, 1), (8, 1), (8, 8), (1, 8)], "two words", granularity="text_line"),))


def test_lost_word_is_a_failure_instead_of_silent_omission():
    same = [(1, 1), (8, 1), (8, 8), (1, 8)]
    with pytest.raises(ValueError, match="loses care words"):
        separated_word_regions((10, 10), (word(same, "first"), word(same, "second")))


def test_explicit_care_priority_preserves_words_inside_coarse_ignore_boxes():
    care = word([(3, 3), (8, 3), (8, 8), (3, 8)], "valid")
    ignore = word([(1, 1), (10, 1), (10, 10), (1, 10)], "###", True)
    with pytest.raises(ValueError, match="loses care words"):
        separated_word_regions((12, 12), (care, ignore))
    mask, valid, _, _, audit = separated_word_regions(
        (12, 12), (care, ignore), ignore_policy="care_region_precedence")
    assert np.asarray(mask)[4, 4] == 255 and np.asarray(valid)[4, 4] == 255
    assert np.asarray(valid)[2, 2] == 0
    assert audit["care_ignore_overlap_pixels"] == 36
    assert audit["missing_word_ids"] == []


def test_icdar13_character_groups_and_mixed_ignore_rows(tmp_path: Path):
    source = tmp_path / "100_GT.txt"
    source.write_text('1 2 3 3 4 1 1 4 8 "A"\n'
                      '4 5 6 8 4 5 1 10 8 "B"\n'
                      '#7 8 9 15 4 12 1 18 8 " "\n\n'
                      '10 11 12 3 14 1 11 4 18 "C"\n', encoding="utf-8")
    instances = parse_icdar13_character_groups(source)
    assert [x.transcription for x in instances if not x.ignore] == ["AB", "C"]
    assert sum(x.ignore for x in instances) == 1
    assert instances[1].bbox_xyxy == (1, 1, 10, 8)


def test_audit_detects_diagonal_contact_and_fragmentation():
    owner = np.zeros((8, 8), dtype=np.uint16)
    owner[1, 1] = 1
    owner[2, 2] = 2
    owner[6, 6] = 1
    audit = audit_word_owners(owner, 3)
    assert audit["missing_word_ids"] == [3]
    assert audit["fragmented_word_ids"] == [1]
    assert audit["interword_contact_pairs"] == 1


def test_gap_widening_retreats_near_word_boundaries_and_keeps_text_pixels():
    owner = np.zeros((30, 60), dtype=np.uint16)
    owner[5:25, 5:28] = 1
    owner[5:25, 30:55] = 2
    protected = np.zeros_like(owner, dtype=np.uint8)
    protected[10:20, 10:20] = 255
    protected[10:20, 37:47] = 255
    before = Image.fromarray(owner, mode="I;16")
    mask, after, audit = widen_word_gaps(before, clearance=6, reference_wh=(60, 30),
        max_area_loss=0.4, protected_pixels=Image.fromarray(protected))
    out = np.asarray(after)
    assert audit["changed_words"] == 2 and audit["removed_pixels"] > 0
    assert np.array_equal(out[protected>0], owner[protected>0])
    assert np.array_equal(np.asarray(before), owner)
    assert np.all(out[owner==0] == 0)
    assert np.all(out[:, 5:20] == owner[:, 5:20])
    assert audit["missing_word_ids"] == [] and audit["fragmented_word_ids"] == []
    assert np.max(np.where(out==1)[1]) < 27
    assert np.min(np.where(out==2)[1]) > 30


def test_gap_widening_restores_small_words_instead_of_erasing_them():
    owner = np.zeros((15, 15), dtype=np.uint16)
    owner[2:10, 2:5] = 1
    owner[2:10, 7:12] = 2
    _, after, audit = widen_word_gaps(Image.fromarray(owner, mode="I;16"), clearance=10,
        reference_wh=(15,15), max_area_loss=0.2)
    assert audit["restored_words"] == 2
    assert np.array_equal(np.asarray(after), owner)


def test_degenerate_ignore_line_is_valid_but_degenerate_care_box_is_not(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text('#1 2 3 5 5 2 5 8 5 " "\n\n1 2 3 3 3 1 1 5 5 "A"\n')
    assert len(parse_icdar13_character_groups(path)[0].polygon_xy) == 2
    path.write_text('1 2 3 5 5 2 5 8 5 "A"\n')
    with pytest.raises(ValueError, match="Invalid character"):
        parse_icdar13_character_groups(path)
