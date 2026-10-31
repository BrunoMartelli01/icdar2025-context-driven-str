"""Shared optional benchmark-lexicon preparation for uniform OCR/risk models."""

import math
from pathlib import Path
import json

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows


def lexicon(profile, split):
    if profile is None:
        return None
    spec = profile[split]
    words = Path(spec['vocabulary']).read_text().splitlines()
    if spec['pair_format'] == 'json':
        pairs = json.loads(Path(spec['pairs']).read_text())
        return DeepSoloLexicon(words, pairs, distance_exclusive=profile['distance_exclusive'])
    if spec['pair_format'] != 'source_text':
        raise ValueError('Explicit vocabulary pair format required')
    return DeepSoloLexicon.from_files(spec['vocabulary'], spec['pairs'],
                                      distance_exclusive=profile['distance_exclusive'])


def prepare_words(raw, dictionary, policy):
    """Never uses labels; keep raw source indices through edits and rejection."""
    cut = policy['minimum_sequence_probability']
    if not math.isfinite(cut) or not 0 <= cut <= 1:
        raise ValueError('Finite native sequence cutoff in [0,1] required')
    if dictionary is None:
        if policy['enabled']:
            raise ValueError('An enabled lexical policy requires its explicit vocabulary')
        corrected = [dict(r, source_row=i) for i, r in enumerate(raw)]
        trace = [dict(source_row=i, original_text=r['raw_text'], active=False,
            accepted=True, match=None, potential_match=None, changed=False,
            confidence=r['confidence']) for i, r in enumerate(raw)]
    else:
        corrected, trace = correct_rows(raw, dictionary, policy)
    rows = [r for r in corrected if r['sequence_confidence_native'] >= cut and
            any(c.isalnum() for c in r['raw_text'].casefold())]
    accepted = {int(r['source_row']) for r in rows}
    trace = [dict(t, accepted=i in accepted, changed=i in accepted and t['changed'])
             for i, t in enumerate(trace)]
    return rows, trace
