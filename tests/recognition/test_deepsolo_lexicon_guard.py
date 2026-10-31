"""Confidence and open-vocabulary protection without access to evaluation labels."""

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_lexicon_guard import correct_guarded


def test_confident_numbers_and_ambiguous_words_are_protected():
    words = ["CAT", "CUT", "CARING", "2015"]
    lex = DeepSoloLexicon(words, {w:w for w in words}, distance_exclusive=1.5)
    rows = [dict(instance_id=i, text=t, recognition_confidence=q) for i,(t,q) in enumerate(
        [("cot", .90), ("carling", .99), ("2010", .8), ("ct", .8)])]
    out, trace = correct_guarded(rows, lex, min_length=3, preserve_digits=True,
                               max_confidence=.95, require_unique=True)
    assert [p["text"] for p in out] == [p["text"] for p in rows]
    assert [t["reason"] for t in trace] == ["protected_ambiguous_neighbor", "protected_high_confidence",
                                           "protected_number", "protected_short_word"]


def test_uncertain_unique_word_is_corrected_at_inclusive_boundary():
    lex = DeepSoloLexicon(["CARING"], {"CARING":"CARING"}, distance_exclusive=1.5)
    rows = [dict(instance_id=0, text="carling", recognition_confidence=.95)]
    out, _ = correct_guarded(rows, lex, min_length=3, preserve_digits=True,
                            max_confidence=.95, require_unique=True)
    assert out[0]["text"] == "CARING"
