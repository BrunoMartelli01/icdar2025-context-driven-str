"""GT-free replacement of fallback transcripts from optional PARSeq rereads."""

import math

from recognition.common_budget_pipeline import normalize
from recognition.group_word_selection import reading_confidence


def padded_box(box, image_wh, padding):
    if padding < 0 or not isinstance(padding, int):
        raise ValueError('Explicit nonnegative integer crop padding required')
    w, h = image_wh
    a, b, c, d = box
    if not all(math.isfinite(v) for v in box) or not (0 <= a < c <= w and 0 <= b < d <= h):
        raise ValueError('Valid original-frame exclusive fallback box required')
    return [max(0, math.floor(a)-padding), max(0, math.floor(b)-padding),
            min(w, math.ceil(c)+padding), min(h, math.ceil(d)+padding)]


def corrected(text, lexicon):
    match = lexicon.nearest(text) if lexicon is not None else None
    return (match['text'] if match is not None else text), match


def choose_transcript(detection, rereadings, lexicon, policy):
    """Change text only; preserve every fallback instance and its geometry."""
    base, _ = corrected(detection['text'], lexicon)
    if not policy['enabled']:
        return base, dict(changed=False, reason='disabled')
    reading = rereadings.get(str(policy['padding']))
    if reading is None:
        return base, dict(changed=False, reason='invalid_envelope_no_crop')
    raw = reading['raw_text']
    candidate, match = corrected(raw, lexicon)
    confidence = (reading['sequence_confidence_native'] if policy['confidence_kind'] == 'sequence'
                  else reading_confidence(reading)['confidence_balanced'])
    if policy['confidence_kind'] not in ('sequence', 'balanced'):
        raise ValueError('Explicit qualified reread confidence required')
    accept = bool(normalize(raw)) and confidence >= policy['threshold']
    if policy['require_lexical_match'] and match is None:
        accept = False
    return (candidate if accept else base), dict(changed=accept and normalize(candidate) != normalize(base),
        accepted_reread=accept, confidence=confidence, original=base,
        reread_text=candidate, lexical_match=match)


def reread_rows(rows, rereadings_by_instance, lexicon, policy):
    """Apply one explicit reread rule after fusion, preserving every output box."""
    result=[]
    for row in rows:
        r=dict(row)
        if policy['enabled'] and not r.get('fixed_light_complement') and r['bbox'] is not None:
            key=str(r['instance_id'])
            if key not in rereadings_by_instance or str(policy['padding']) not in rereadings_by_instance[key]:
                raise ValueError('Missing actual PARSeq reread for a retained DeepSolo word')
            text,trace=choose_transcript(r,rereadings_by_instance[key],lexicon,policy)
            r.update(raw_text=text,text=text,fallback_reread_trace=trace)
        result.append(r)
    return result
