import numpy as np
import pytest

from diagnostics.train_crop_completeness import (
    padded_box, counts_in_box, crop_metrics, completeness_class,
)


def test_padding_and_no_crop():
    assert padded_box([1, 2, 8, 9], (10, 10), 6) == [0, 0, 10, 10]
    assert padded_box(None, (10, 10), 6) is None
    assert counts_in_box(np.zeros((2, 2)), None) == {}


def test_contamination_separates_other_word_and_dont_care():
    metric = crop_metrics({1: 10, 2: 0, 3: 5, 4: 9}, [1, 2],
                          {1: 10, 2: 10, 3: 10}, [[1, 2], [3]], {4})
    assert metric['own_foreground_retention'] == .5
    assert metric['missing_entire_glyphs'] == 1
    assert metric['other_care_pixels'] == 5
    assert metric['dont_care_pixels'] == 9
    assert metric['other_words'][0]['word_foreground_fraction'] == .5
    assert completeness_class(True, metric) == 'INCOMPLETE_OTHER_CARE_PRESENT'
    assert completeness_class(False, metric) == 'NO_CROP'


def test_counts_use_exclusive_bounds_and_true_prediction_support():
    packed = np.array([[1, 1, 2], [3, 3, 4]])
    pred = np.array([[0, 1, 1], [0, 0, 1]])
    assert counts_in_box(packed, [0, 0, 2, 2]) == {1: 2, 3: 2}
    assert counts_in_box(packed, [0, 0, 2, 2], pred) == {1: 1}
    with pytest.raises(ValueError):
        counts_in_box(packed, [-1, 0, 2, 2])


def test_complete_crop_can_still_contain_dont_care():
    metric = crop_metrics({1: 5, 4: 5}, [1], {1: 5}, [[1]], {4})
    assert metric['all_own_glyph_pixels_inside']
    assert metric['other_care_fraction_of_care_pixels'] == 0
    assert metric['dont_care_pixels'] == 5
    assert completeness_class(True, metric) == 'COMPLETE_NO_OTHER_CARE'
