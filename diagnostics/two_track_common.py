"""Persistent provenance and fixed-spatial evaluation shared by isolated diagnostics."""
from collections import defaultdict
from pathlib import Path
from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.lavis_captioner_candidates import digest, sha
from diagnostics.contextual_ocr_correction import verify, word_scores, lexical_counts


def source(root, cfg):
    from diagnostics.ocr_target_bound_battery_train import load
    verify(root/'results.json', cfg['source_battery_sha256'])
    _, sp, data, _, _ = load(root)
    return sp, data


def summarize(data, outputs):
    labels = defaultdict(list)
    for w in data['labels']:
        labels[w['sample_id']].append(w)
    rows = []; lex = []
    for sample in data['samples']:
        changed = {b['id']: outputs.get(b['candidate_id'], b['text']) for b in sample['blocks']}
        rows.extend(word_scores(sample, labels[sample['sample_id']], changed))
        lex.append(lexical_counts(list(changed.values()), [w['gt_text'] for w in labels[sample['sample_id']]]))
    return {'gt_words': len(rows), 'exact': sum(w['exact'] for w in rows),
            'recovered': sum(w['corrected_error'] for w in rows), 'damaged': sum(w['damaged_correct'] for w in rows),
            'fuzzy_ge_0_8': sum(w['fuzzy'] >= .8 for w in rows),
            'changed_crops': sum(outputs.get(b['candidate_id'], b['text']) != b['text'] for s in data['samples'] for b in s['blocks']),
            'lexical_matches': sum(x['matches'] for x in lex)}, rows


def freeze(root, cfg, inputs, files):
    protocol = {'config': cfg, 'input_sha256': {str(p): sha(p) for p in inputs},
                'code_sha256': {str(p): sha(p) for p in files},
                'test_loaded': False, 'training': False, 'table5_accuracy': None}
    contract = digest(protocol)
    seal_write(root/'protocol.json', {**protocol, 'contract_sha256': contract})
    return contract


def cache(root, name, contract):
    path = root/'cache'/(name+'.json')
    return checked(path, contract) if path.exists() else None
