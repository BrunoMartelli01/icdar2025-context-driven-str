"""CPU-only offline edit gates; preserve every frozen experiment and model."""
import argparse
from collections import Counter
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.contextual_ocr_conservative_crop_train import (
    load_pilot, make_prompt, request_identity, check_request, interpret_text, output_path,
)
from diagnostics.contextual_ocr_correction import verify, word_scores, lexical_counts
from diagnostics.lavis_captioner_candidates import sha, digest


def apply_gate(original, proposed, max_distance=None):
    """No labels, confidences, context, or geometric annotations allowed here."""
    distance = Levenshtein.distance(original.casefold(), proposed.casefold())
    raw_distance = Levenshtein.distance(original, proposed)
    if proposed == original:
        reason = 'unchanged'
    elif not proposed:
        reason = 'rejected_empty'
    elif any(c.isspace() for c in proposed):
        reason = 'rejected_whitespace'
    elif max_distance is not None and distance > max_distance:
        reason = 'rejected_distance'
    else:
        reason = 'accepted_change'
    text = proposed if reason == 'accepted_change' else original
    return {'text': text, 'reason': reason, 'edit_distance_casefold': distance,
            'edit_distance_raw': raw_distance, 'changed': text != original}


def aggregate(rows, lexical, crops):
    exact = sum(r['exact'] for r in rows)
    matches = sum(r['matches'] for r in lexical)
    tokens = sum(r['predicted'] for r in lexical)
    words = len(rows)
    precision = matches/tokens if tokens else 0.
    recall = matches/words if words else 0.
    return {'gt_words': words, 'exact': exact, 'exact_rate': exact/words if words else 0.,
            'recovered': sum(r['corrected_error'] for r in rows),
            'damaged': sum(r['damaged_correct'] for r in rows),
            'fuzzy_ge_0_8': sum(r['fuzzy'] >= .8 for r in rows),
            'image_lexical_exact_matches': matches, 'image_lexical_recall': recall,
            'image_lexical_precision': precision,
            'image_lexical_f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
            'predicted_tokens': tokens, 'changed_crops': sum(r['changed'] for r in crops),
            'gate_reasons': dict(Counter(r['reason'] for r in crops))}


def decision(s):
    return 'GO_TRAIN_DIAGNOSTIC_ONLY' if (
        s['recovered'] > s['damaged'] and
        (s['exact'] > 502 or s['image_lexical_exact_matches'] > 503) and
        s['predicted_tokens'] <= 1203) else 'NO-GO'


def run(a):
    cfg = json.loads(a.config.read_text())
    if (cfg['mode'] != 'APPROXIMATE' or cfg['split'] != 'TRAIN'
            or cfg['max_edit_distances'] != [1, 2] or cfg['new_inference']
            or cfg['training'] or cfg['test_loaded']):
        raise ValueError('frozen CPU TRAIN gate diagnostic required')
    verify(a.source_root/'results.json', cfg['source_report_sha256'])
    protocol, samples, contract = load_pilot(a.source_root)
    source = checked(a.source_root/'results.json', contract)
    verify(a.audit, protocol['audit_sha256'])
    audit = json.loads(a.audit.read_text())
    if (len(samples), len(audit['per_word']), sum(len(s['blocks']) for s in samples)) != (229, 849, 1203):
        raise RuntimeError('frozen population mismatch')
    by_gt = {s['sample_id']: [r for r in audit['per_word'] if r['sample_id'] == s['sample_id']] for s in samples}
    generations, hashes = {}, {}
    for sample in samples:
        for block in sample['blocks']:
            path = output_path(a.source_root, cfg['model'], cfg['variant'], sample['sample_id'], block['id'])
            relative = str(path.relative_to(a.source_root))
            verify(path, source['inference_shards_sha256'][relative])
            message = make_prompt(sample, block, cfg['variant'], protocol['config'])
            g = check_request(path, contract, request_identity(sample, block, cfg['variant'], message))
            verify(a.source_root/'models'/(cfg['model']+'.json'), g['model_metadata_sha256'])
            parsed = interpret_text(g['raw_output'], block['text'], g['token_limit_abstention'])
            if any(g[k] != value for k, value in parsed.items()):
                raise RuntimeError('source parser mismatch')
            generations[(sample['sample_id'], block['id'])] = g
            hashes[relative] = sha(path)
    variants = [('V1', None), ('QWEN_CAPTION_UNFILTERED', None), ('SINGLE_WORD', None)]
    variants += [('EDIT_LE_'+str(n), n) for n in cfg['max_edit_distances']]
    summaries, details = {}, {}
    for name, limit in variants:
        rows, lexical, crops = [], [], []
        for sample in samples:
            current = {}
            for block in sample['blocks']:
                g = generations[(sample['sample_id'], block['id'])]
                original, proposed = block['text'], g['text']
                if name in ('V1', 'QWEN_CAPTION_UNFILTERED'):
                    text = original if name == 'V1' else proposed
                    result = {'text': text, 'reason': 'control', 'changed': text != original}
                else:
                    result = apply_gate(original, proposed, limit)
                current[block['id']] = result['text']
                crops.append({'sample_id': sample['sample_id'], 'candidate_id': block['candidate_id'],
                              'original': original, 'proposed': proposed, **result})
            rows.extend(word_scores(sample, by_gt[sample['sample_id']], current))
            lexical.append({'sample_id': sample['sample_id'], **lexical_counts(
                [current[b['id']] for b in sample['blocks']], [r['gt_text'] for r in by_gt[sample['sample_id']]])})
        summary = aggregate(rows, lexical, crops)
        if summary['exact'] != 502+summary['recovered']-summary['damaged']:
            raise RuntimeError('recovery/damage arithmetic mismatch')
        summaries[name] = summary
        details[name] = {'per_word': rows, 'per_image_lexical': lexical, 'per_crop': crops}
    for name, key in [('V1', 'baseline'), ('QWEN_CAPTION_UNFILTERED', cfg['model']+'/'+cfg['variant'])]:
        expected = source['summary'][key]
        got = summaries[name]
        for dst, src in [('exact', 'exact'), ('recovered', 'recovered'), ('damaged', 'damaged'),
                         ('image_lexical_exact_matches', 'lexical_matches'), ('predicted_tokens', 'lexical_predicted_tokens'),
                         ('changed_crops', 'changed_targets')]:
            if got[dst] != expected[src]:
                raise RuntimeError(f'baseline reproduction failed: {name}/{dst}')
    single = summaries['SINGLE_WORD']
    if (single['image_lexical_exact_matches'], single['changed_crops'], single['predicted_tokens'],
            single['gate_reasons'].get('rejected_whitespace', 0)) != (493, 269, 1203, 303):
        raise RuntimeError('single-word control differs from reported frozen diagnostic')
    for name in ('EDIT_LE_1', 'EDIT_LE_2'):
        summaries[name]['decision'] = decision(summaries[name])
    dependencies = [__file__, str(a.config), 'diagnostics/contextual_ocr_conservative_crop_train.py',
                    'diagnostics/contextual_ocr_correction.py', 'diagnostics/caption_routing_train.py',
                    'diagnostics/lavis_captioner_candidates.py']
    import importlib.metadata
    experiment = {'config': cfg, 'source_report_sha256': sha(a.source_root/'results.json'),
                  'source_contract': contract, 'audit_sha256': sha(a.audit),
                  'code_sha256': {p: sha(p) for p in dependencies},
                  'rapidfuzz_version': importlib.metadata.version('rapidfuzz'),
                  'scope': {'split': 'TRAIN', 'images': 229, 'gt_words': 849, 'crops': 1203},
                  'GT_used_in_gate': False, 'new_inference': False, 'cpu_only': True}
    result = {'classification': 'PRACTICAL_APPROXIMATION', 'protocol': experiment,
              'contract_sha256': digest(experiment), 'summary': summaries, 'details': details,
              'inference_shards_sha256': hashes, 'table5_accuracy': None,
              'pipeline_promoted': False, 'assignment': 'frozen audit best-IoU GT-to-crop; no fragment grouping'}
    seal_write(a.output_dir/'results.json', result)
    print(json.dumps(summaries, indent=2), flush=True)
    print('JSON', a.output_dir/'results.json', 'SHA256', sha(a.output_dir/'results.json'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
