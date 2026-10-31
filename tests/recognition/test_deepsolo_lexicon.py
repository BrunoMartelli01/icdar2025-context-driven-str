"""Source-order, edit boundary and denominator safeguards for DeepSolo lexicons."""

import pytest

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.unet_corrected_token_interfaces import lexical_counts


def lexicon(words):
    return DeepSoloLexicon(words, {w.upper(): w for w in words}, distance_exclusive=1.5)


def test_distance_boundary_and_first_minimal_tie():
    model = lexicon(["CAT", "CUT", "DOG"])
    assert model.nearest("cot")["text"] == "CAT"
    assert model.nearest("cut")["text"] == "CUT"
    assert model.nearest("cast")["distance"] == 1
    assert model.nearest("c") is None


def test_unmatched_policies_preserve_predictions_without_using_targets():
    model = lexicon(["CAT"])
    predictions = [dict(instance_id=1, text="cot", ignored=False),
                   dict(instance_id=2, text="1996", ignored=False)]
    kept, trace = model.correct(predictions, unmatched_policy="keep_unmatched")
    assert [p["text"] for p in kept] == ["CAT", "1996"]
    rejected, _ = model.correct(predictions, unmatched_policy="source_reject")
    assert [p["text"] for p in rejected] == ["CAT"]
    assert predictions[0]["text"] == "cot"
    assert trace[1]["match"] is None


def test_pair_canonical_punctuation_and_repeated_instances(tmp_path):
    vocab = tmp_path / "v.txt"
    pairs = tmp_path / "pairs.txt"
    vocab.write_text("CANT\n")
    pairs.write_text("CANT CAN'T\n")
    model = DeepSoloLexicon.from_files(vocab, pairs, distance_exclusive=1.5)
    rows, _ = model.correct([dict(instance_id=i, text="cant", ignored=False) for i in range(2)],
                            unmatched_policy="source_reject")
    counts = lexical_counts([dict(raw_text=p["text"]) for p in rows], [dict(text="cant")], [False, False])
    assert counts["tp"] == 1 and counts["fp"] == 1
    assert rows[0]["text"] == "CAN'T"


def test_incomplete_pairs_and_unknown_policy_fail():
    with pytest.raises(ValueError):
        DeepSoloLexicon(["MISSING"], {}, distance_exclusive=1.5)
    with pytest.raises(ValueError):
        lexicon(["CAT"]).correct([], unmatched_policy="unknown")
