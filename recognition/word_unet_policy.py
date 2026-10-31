"""GT-free emission policy for the matched two-map word U-Net extension."""

from recognition.lexical_budget_router import correct_rows
from recognition.unet_token_routing import selected_rows
from segmentation.icdar13_matched_training import emit_words


def apply_word_policy(raw, settings, lexicon):
    """Keep native sequence filtering and balanced confidence before correction.

    A lexical edit does not increase OCR confidence or change a crop identity.
    Neither transcripts nor validity masks from annotations are accepted here.
    """
    lexical = settings['lexical']
    if lexical not in ('none', 'keep', 'reject'):
        raise ValueError('Explicit lexical policy required')
    selected = emit_words(selected_rows(raw, settings),
                          settings['minimum_transcript_probability'])
    return correct_rows(selected, lexicon, dict(enabled=lexical != 'none',
        confidence_cap=1.01, protect_numeric=False,
        unmatched_policy='source_reject' if lexical == 'reject' else 'keep_unmatched'))


def normalized_changes(trace):
    normalize = lambda s: ''.join(c for c in s.casefold() if c.isalnum())
    return sum(not t['accepted'] or (t['match'] is not None and
        normalize(t['original_text']) != normalize(t['match']['text'])) for t in trace)


def select_word_policy(points):
    """Stable validation F1, fewer normalized lexical edits, then declared order."""
    if not points or any(p['split'] != 'selection' for p in points):
        raise ValueError('Policy choice may read only TRAIN selection results')
    return max(enumerate(points), key=lambda item: (
        item[1]['counts']['f1'], -item[1]['normalized_changes'], -item[0]))[1]
