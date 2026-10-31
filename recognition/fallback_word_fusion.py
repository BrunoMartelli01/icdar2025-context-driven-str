"""GT-free, one-to-one transcript replacement on already routed fallback words."""

from rapidfuzz.distance import Levenshtein

from localize.independent_word_slots import overlap_iou
from recognition.common_budget_pipeline import normalize


def replace_fallback_transcripts(light, fallback, policy):
    result = [dict(r) for r in fallback]
    if not policy['enabled']:
        return result
    threshold, minimum_iou, max_distance = (policy['minimum_word_probability'],
        policy['minimum_iou'], policy['maximum_edit_distance'])
    if not 0 <= threshold <= 1 or not 0 < minimum_iou <= 1 or (max_distance is not None and max_distance < 0):
        raise ValueError('Explicit valid post-fallback transcript policy required')
    pairs = []
    for i, heavy in enumerate(result):
        if heavy['bbox'] is None:
            continue
        for j, word in enumerate(light):
            probability = word['word_probability']
            if not 0 <= probability <= 1:
                raise ValueError('Invalid learned word correctness probability')
            if probability < threshold:
                continue
            iou = overlap_iou(heavy['bbox'], word['bbox'])
            if iou >= minimum_iou:
                pairs.append((-iou, -probability, i, j))
    used_heavy = set(); used_light = set()
    for negative_iou, negative_probability, i, j in sorted(pairs):
        if i in used_heavy or j in used_light:
            continue
        used_heavy.add(i); used_light.add(j)
        old = normalize(result[i]['raw_text']); new = normalize(light[j]['raw_text'])
        if not new or new == old or (max_distance is not None and Levenshtein.distance(old, new) > max_distance):
            continue
        result[i].update(raw_text=light[j]['raw_text'], text=light[j]['raw_text'],
            original_deepsolo_text=result[i]['raw_text'], replaced_by_light=True,
            replacement_word_probability=-negative_probability, replacement_iou=-negative_iou,
            replacement_source_row=int(light[j]['source_row']))
    return result
