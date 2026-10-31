"""Explicit comparison-only text policies; original OCR/GT strings stay intact."""

CASEFOLD = 'casefold'
ALPHANUMERIC = 'alphanumeric_casefold'


def normalize_word_text(text, policy=CASEFOLD):
    if policy not in (CASEFOLD, ALPHANUMERIC):
        raise ValueError('Unknown word comparison normalization: '+str(policy))
    folded=text.casefold()
    return ''.join(c for c in folded if c.isalnum()) if policy==ALPHANUMERIC else folded


def comparison_label(policy=CASEFOLD):
    normalize_word_text('',policy)
    if policy==ALPHANUMERIC:
        return 'Unicode casefold then Unicode alphanumeric characters only; punctuation, symbols and whitespace removed; empty normalized strings cannot match'
    return 'exact Unicode casefold; punctuation retained; empty strings cannot match'
