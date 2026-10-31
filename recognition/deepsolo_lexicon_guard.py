"""Observable confidence/ambiguity guards for general-vocabulary DeepSolo correction."""

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


def correct_guarded(predictions, lexicon, *, min_length, preserve_digits,
                    max_confidence, require_unique):
    """Keep open-vocabulary OCR; only change eligible uncertain unambiguous words."""
    corrected, trace = [], []
    for p in predictions:
        match = lexicon.nearest(p["text"])
        reason = "no_neighbor"
        change = False
        if match is not None:
            if match["distance"] == 0:
                change, reason = True, "exact_canonical_pair"
            elif len(p["text"]) < min_length:
                reason = "protected_short_word"
            elif preserve_digits and p["text"].isdigit():
                reason = "protected_number"
            elif p["recognition_confidence"] > max_confidence:
                reason = "protected_high_confidence"
            else:
                options = process.extract(p["text"].upper(), lexicon.words,
                    scorer=Levenshtein.distance, processor=None, score_cutoff=1, limit=2)
                if require_unique and len(options) > 1:
                    reason = "protected_ambiguous_neighbor"
                else:
                    change, reason = True, "accepted_edit_one"
        text = match["text"] if change else p["text"]
        corrected.append(dict(p, text=text))
        trace.append(dict(instance_id=p["instance_id"], original_text=p["text"],
                          corrected_text=text, match=match, reason=reason))
    return corrected, trace
