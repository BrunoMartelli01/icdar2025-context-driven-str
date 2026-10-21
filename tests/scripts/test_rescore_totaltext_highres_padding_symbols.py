from __future__ import annotations

from scripts.rescore_totaltext_highres_padding_symbols import (
    lexical_multiset_matches, match_word_ids, normalize_word,
)


def test_unicode_casefold_removes_punctuation_symbols_but_not_accents_digits_spaces() -> None:
    assert normalize_word("NURI'S!") == "nuris"
    assert normalize_word("€25–CAFÉ") == "25café"
    assert normalize_word("New York") == "new york"
    assert normalize_word("#?!") == ""


def test_normalized_spatial_matching_is_one_to_one_and_requires_nonempty_text() -> None:
    gt = [{"bbox": [0, 0, 20, 20], "text": "NURIS"},
          {"bbox": [30, 0, 50, 20], "text": "!!!"}]
    predictions = [{"bbox": [0, 0, 20, 20], "text": "NURI'S"},
                   {"bbox": [0, 0, 20, 20], "text": "NURIS!"},
                   {"bbox": [30, 0, 50, 20], "text": "??"}]
    assert match_word_ids(predictions, gt, normalize=True) == {0}
    assert match_word_ids(predictions, gt, normalize=False) == set()
    assert lexical_multiset_matches(predictions, gt) == 1
