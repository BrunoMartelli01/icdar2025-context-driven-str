"""Prediction-only admission of adapted detector/PARSeq word hypotheses."""

import math
from localize.independent_word_slots import overlap_iou
from recognition.common_budget_pipeline import normalize
from recognition.lexical_budget_router import correct_rows


def emit(rows, lexicon, policy):
    raw = [dict(r, original_source_row=i) for i, r in enumerate(rows)
        if r['sequence_confidence_native'] >= policy['sequence_threshold'] and
        (not policy['require_alphanumeric'] or normalize(r['raw_text']))]
    lexical = policy['lexical']
    if lexical not in ('none', 'keep', 'reject'):
        raise ValueError('Explicit lexical admission required')
    if lexical == 'none':
        edited = [dict(r, text=r['raw_text'], original_raw_text=r['raw_text']) for r in raw]
        trace = [dict(changed=False, accepted=True) for _ in raw]
    else:
        edited, trace = correct_rows(raw, lexicon, dict(enabled=True,
            confidence_cap=1.01, protect_numeric=False,
            unmatched_policy='source_reject' if lexical == 'reject' else 'keep_unmatched'))
    result = []
    score = lambda r: math.sqrt(r['detector_score']*r['confidence'])
    for row in sorted(edited, key=lambda r: (-score(r), r['candidate_id'])):
        if score(row) < policy['joint_threshold']:
            continue
        if any(normalize(row['raw_text']) == normalize(other['raw_text']) and
            overlap_iou(row['bbox'], other['bbox']) > policy['same_text_nms_iou'] for other in result):
            continue
        result.append(dict(row, source_row=row['original_source_row']))
    return result, sum(t['changed'] or not t['accepted'] for t in trace)
