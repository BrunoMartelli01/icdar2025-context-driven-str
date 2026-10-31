"""Independently replay a frozen IC13 TEST route, words, crops and counts."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from recognition.adapted_word_policy import emit
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.fusion_deployment_factory import load_fusion_budget_pipeline
from recognition.selected_deepsolo_evidence import inspect_fresh_export
from scripts.audit_lightweight_word_pipeline import recount, total
from scripts.crop_clip_runtime import array_sha
from scripts.evaluate_fixed_expert_budget import check_inputs, expected_frontend_checkpoint
from scripts.icdar_budget_data import annotation, predictions, read, sha
from scripts.train_lightweight_word_router import teacher
from scripts.run_yolo_f1_quality_study import seal


def audit(root):
    plan = read(root/'resolved_configuration.json'); c = plan['config']; check_inputs(c)
    gate = Path(c['router_root']); pipeline = load_fusion_budget_pipeline(gate)
    gate_plan = pipeline.plan; checkpoint = expected_frontend_checkpoint(c, gate_plan, gate)
    decision = read(root/'decisions_before_fallback.json'); stored = read(root/'observations.json')
    output = read(root/'per_image.json'); aggregate = read(root/'results.json')
    if (decision['ground_truth_used'] or decision['fallback_outputs_loaded'] or
            decision['observations_sha256'] != sha(root/'observations.json') or
            decision['router_frozen_sha256'] != sha(gate/'selection_frozen.json') or
            aggregate['test_used_for_selection']):
        raise ValueError('Changed pre-fallback TEST boundary')
    if c.get('fresh_deepsolo'):
        fresh = inspect_fresh_export(c, decision)
        if (aggregate['fresh_deepsolo_evidence'] != fresh or
                aggregate['actual_new_deepsolo_neural_inferences'] != len(decision['fallback_ids'])):
            raise ValueError('Fresh selected DeepSolo accounting differs')
    original = read(gate_plan['config']['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['test']}; ids = sorted(records)
    index = read(Path(c['expert_export'])/'index.json');
    if set(ids) != set(output) or set(ids) != set(stored) or set(ids) != set(index) or len(ids) != c['total_images']:
        raise ValueError('Incomplete TEST population')
    profile = read(c['fallback_profile']); lex = DeepSoloLexicon.from_files(
        profile['lexicon_paths']['vocabulary'], profile['lexicon_paths']['pairs'],
        distance_exclusive=profile['distance_exclusive'])
    override = gate_plan['config'].get('emission_override')
    policy = read(override['policy'])['winner']['policy'] if override else None
    fallback_config = dict(fallback_profile=c['fallback_profile'], train_protocol=c['test_protocol'], raw_train_root=c['raw_test_root'])
    reread_index = {}; reread = None
    needs_reread = pipeline.choice['fallback_profile'].startswith('reread_')
    if needs_reread:
        reread = Path(c['reread_export']); reread_index = read(reread/'index.json')
        frozen = gate_plan['config']['fallback_profiles'][pipeline.choice['fallback_profile']]['reread_profile']['policy']
        fallback_config['reread_profile'] = dict(index=str(reread/'index.json'), policy=frozen)
    selected = set(decision['fallback_ids']); accepted = set(decision['accepted_ids'])
    if ((needs_reread and set(reread_index) != selected) or selected & accepted or
            selected | accepted != set(ids) or len(selected) != c['fallback_images']):
        raise ValueError('TEST exact budget or reread subset differs')
    deployment = {}; source_crops = reread_crops = 0
    for sid in ids:
        record = records[sid]; entry = index[sid]
        if sha(entry['path']) != entry['sha256'] or sha(record['image_path']) != record['image_sha256']:
            raise ValueError('Changed source expert image/output')
        value = read(entry['path'])
        if (value['ground_truth_used'] or value['image_sha256'] != record['image_sha256'] or
                value['checkpoint_sha256'] != checkpoint):
            raise ValueError('Unqualified source expert metadata')
        with Image.open(record['image_path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        for row in value['raw_rows']:
            a, b, d, e = row['bbox']
            if array_sha(np.ascontiguousarray(rgb[b:e, a:d])) != row['crop_rgb_sha256']:
                raise ValueError('Changed expert PARSeq crop')
            source_crops += 1
        words = emit(value['raw_rows'], lex, policy)[0] if override else value['rows']
        o = pipeline.observe(value['raw_rows'], words, record['image_wh'])
        np.testing.assert_allclose(o['score'], stored[sid]['score'], rtol=1e-12, atol=1e-12)
        if o['rows'] != stored[sid]['rows']:
            raise ValueError('Annotation-free TEST expert deployment differs')
        deployment[sid] = o; gt, valid = annotation(record)
        if gt != output[sid]['ground_truth']:
            raise ValueError('Changed TEST annotation')
        if sid in accepted:
            rows = o['rows']; flags = [p['ignored'] for p in predictions(rows, record, valid)]
        else:
            if needs_reread:
                entry = reread_index[sid]
                if sha(entry['path']) != entry['sha256']:
                    raise ValueError('Changed selected fallback reread')
                sample = read(entry['path'])
                if sample['ground_truth_used'] or sample['image_sha256'] != record['image_sha256']:
                    raise ValueError('Selected fallback reread binding differs')
                for r in sample['rows']:
                    for crop in r['rereadings'].values():
                        a, b, d, e = crop['crop_bbox']
                        if array_sha(np.ascontiguousarray(rgb[b:e, a:d])) != crop['crop_rgb_sha256']:
                            raise ValueError('Changed selected fallback PARSeq crop')
                        reread_crops += 1
            fallback = teacher(fallback_config, None, record, lex)
            for row in fallback['rows']:
                if row.get('invalid_envelope') is not None:
                    row.update(raw_text='', text='')
            rows = pipeline.combine_after_fallback(o, fallback['rows'])
            flags = [predictions([r], record, valid)[0]['ignored'] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
        if rows != output[sid]['rows'] or flags != output[sid]['ignored'] or recount(rows, gt, flags) != output[sid]['counts']:
            raise ValueError('Independent TEST per-image output or lexical recount differs')
    keys = {s: records[s]['rgb_sha256'] for s in ids}
    eligible = [s for s in ids if deployment[s]['rows']] if pipeline.winner['guard'] else ids
    chosen = sorted(eligible, key=lambda s: (-deployment[s]['score'], keys[s], s))[:len(ids)-len(selected)]
    if sorted(chosen) != decision['accepted_ids']:
        raise ValueError('Independent TEST exact-budget route differs')
    summary = total(output[s]['counts'] for s in ids)
    if (any(aggregate['counts'][k] != v for k, v in summary.items()) or
            aggregate['counts']['gt_words'] != c['care_words'] or
            source_crops != read(Path(c['expert_export'])/'completion.json')['actual_parseq_crop_inferences'] or
            (needs_reread and reread_crops != read(reread/'completion.json')['actual_parseq_crop_inferences']) or
            (not needs_reread and reread_crops != 0)):
        raise ValueError('Complete TEST counts or actual inference totals differ')
    for path, digest in read(root/'input_sha256.json').items():
        if sha(path) != digest:
            raise ValueError('Changed pinned TEST input')
    report = dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        images_replayed=len(ids), source_crops_checked=source_crops, fallback_crops_checked=reread_crops,
        selected_fallback_replays=len(selected), counts=aggregate['counts'],
        deepsolo_images=len(selected), total_images=len(ids), deepsolo_percentage=100*len(selected)/len(ids),
        test_used_for_selection=False, historical_test_exposure=True,
        result_sha256=sha(root/'results.json'), decision_sha256=sha(root/'decisions_before_fallback.json'),
        refit_contract_sha256=sha(c['frontend_refit_contract']) if c.get('frontend_refit_contract') else None,
        auditor_sha256=sha(__file__),
        scope=('Independent crop, GT-free routing and lexical multiset recount; '
               + ('fresh selected DeepSolo' if c.get('fresh_deepsolo') else 'cached DeepSolo')
               + ', no online latency measurement'))
    seal(root/'independent_test_audit_v1.json', report)
    print(__import__('json').dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root)
