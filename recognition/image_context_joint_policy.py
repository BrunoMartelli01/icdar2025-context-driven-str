"""GT-blind composition of frozen correction and false-positive policies."""
from recognition.image_context_correction import apply_choices,filter_readings


def compose(blocks,operations,utilities,correction_threshold,exact,presence,
            filter_threshold,presence_threshold,filter_policy,order):
    if order not in ('filter_first','correct_first'):raise ValueError('Unknown composition order')
    _,removed=filter_readings(blocks,exact,filter_threshold,filter_policy,presence,presence_threshold)
    removed=set(removed)
    eligible=[i for i,o in enumerate(operations) if order=='correct_first' or not removed.intersection(o['parents'])]
    ops=[operations[i] for i in eligible];values=[utilities[i] for i in eligible]
    _,chosen=apply_choices(blocks,ops,values,correction_threshold)
    used={p for i in chosen for p in ops[i]['parents']}
    if order=='correct_first':removed-=used
    replacements={min(ops[i]['parents']):ops[i]['text'] for i in chosen}
    result=[replacements[i] if i in replacements else b['text'] for i,b in enumerate(blocks)
            if i not in removed and (i in replacements or i not in used)]
    return result,dict(removed=sorted(removed),chosen=[eligible[i] for i in chosen])
