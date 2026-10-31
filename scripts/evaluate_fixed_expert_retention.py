"""Route without TEST labels, then recount qualified TT/IC15 fallback outputs."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import counts, micro
from recognition.fusion_deployment_factory import load_fusion_budget_pipeline
from scripts.audit_lightweight_word_pipeline import recount
from scripts.icdar_budget_data import read, sha
from scripts.prepare_fixed_expert_retention import observe
from scripts.run_yolo_f1_quality_study import seal


def check_config(root, config_path):
    plan = read(root/'resolved_configuration.json')
    if plan['mode'] != 'approximate' or plan['smoke'] or plan['source_kind'] != 'qualified_common_outputs':
        raise ValueError('Full APPROXIMATE common-source result required')
    if sha(config_path) != plan['config_sha256']:
        raise ValueError('Changed explicit retention configuration')
    return plan, read(config_path)


def source_paths(root, plan):
    """Resolve qualified common observations through a frozen gate parent, if any."""
    common = Path(plan['source_root']); seen = set(); observations = root/'test_observations.json'
    while not (common/'cases.json').is_file():
        if common in seen or len(seen) >= 4:
            raise ValueError('Invalid common-source gate ancestry')
        if not observations.is_file() and (common/'test_observations.json').is_file():
            observations = common/'test_observations.json'
        seen.add(common); parent = read(common/'resolved_configuration.json')
        common = Path(parent['source_root'])
    if not observations.is_file():
        raise ValueError('Frozen gate needs qualified TEST observations')
    return common, observations


def route(root, config_path):
    plan, c = check_config(root, config_path); pipeline = load_fusion_budget_pipeline(root)
    source, obs_path = source_paths(root, plan); name = plan['source_case']; cases = read(source/'cases.json'); case = cases[name]
    obs = read(obs_path); ids = plan['partitions']['test']
    if set(obs) != set(ids) or len(ids) != c['budgets'][case['dataset']]['images']:
        raise ValueError('Complete expected TEST population required')
    old = read(source/'data'/(name+'__observations.json')); maps = read(source/'data'/(name+'__maps.json'))
    output = read(source/'models'/name/'observable_outputs.json'); deployed = {}
    for sid in ids:
        x = old[sid]
        expected = observe(x['rows'], output['corrected_rows'][sid], output['traces'][sid], x['image_wh'], maps[sid]['maps'])
        if obs[sid] != expected:
            raise ValueError('TEST observable expert output differs from the qualified source')
        o = obs[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
    keys = {s: case['tie_keys'][s] for s in ids}; quota = c['budgets'][case['dataset']]['fallback_images']
    decision = pipeline.route(deployed, keys, fallback_images=quota)
    eligible = [s for s in ids if deployed[s]['rows']] if pipeline.winner['guard'] else ids
    independently_ranked = sorted(eligible, key=lambda s: (-deployed[s]['score'], keys[s], s))[:len(ids)-quota]
    if decision['accepted_ids'] != sorted(independently_ranked):
        raise ValueError('Independent TEST quota rank differs')
    output_dir = root/'test'; output_dir.mkdir(exist_ok=False)
    seal(output_dir/'observations.json', deployed)
    seal(output_dir/'decisions_before_fallback.json', dict(decision,
        observations_sha256=sha(output_dir/'observations.json'),
        test_observations_sha256=sha(obs_path),
        qualification_sha256=sha(root/'deployment_qualification.json'),
        ground_truth_used=False, fallback_outputs_loaded=False))
    print('FROZEN RETENTION TEST ROUTE', name, decision['deepsolo_images'],
          decision['total_images'], decision['deepsolo_percentage'], flush=True)


def evaluate(root, config_path):
    plan, c = check_config(root, config_path); pipeline = load_fusion_budget_pipeline(root)
    source, obs_path = source_paths(root, plan); name = plan['source_case']; case = read(source/'cases.json')[name]
    out = root/'test'; decision = read(out/'decisions_before_fallback.json'); observed = read(out/'observations.json')
    if (decision['ground_truth_used'] or decision['fallback_outputs_loaded'] or
            decision['observations_sha256'] != sha(out/'observations.json') or
            decision['test_observations_sha256'] != sha(obs_path) or
            decision['qualification_sha256'] != sha(root/'deployment_qualification.json')):
        raise ValueError('Changed pre-fallback TEST decisions')
    ids = plan['partitions']['test']; selected = set(decision['fallback_ids']); accepted = set(decision['accepted_ids'])
    if selected & accepted or selected | accepted != set(ids) or len(selected) != c['budgets'][case['dataset']]['fallback_images']:
        raise ValueError('Retention TEST budget differs from frozen decisions')
    labels = read(source/'data'/(name+'__test_labels.json'))
    if case['dataset'] == 'totaltext':
        native_path = Path(c['tt_native_root'])/'test/branches.json'; old_pins = read(source/'source_input_sha256.json')
        if old_pins.get(str(native_path)) != sha(native_path):
            raise ValueError('Changed qualified Total-Text TEST fallback cache')
        native = read(native_path)['records']; archive = None
        pins = {str(native_path): sha(native_path)}
    else:
        native = None; archive_path = Path(c['ic15_teacher_root'])/'counts.json'
        audit_path = Path(c['ic15_teacher_root'])/'independent_audit.json'; qualified = read(audit_path)
        if qualified['status'] != 'PASS' or qualified['counts_sha256'] != sha(archive_path):
            raise ValueError('Changed qualified ICDAR2015 TEST teacher archive')
        archive = read(archive_path)['samples']; pins = {str(archive_path): sha(archive_path), str(audit_path): sha(audit_path)}
    results = {}; light_control = []
    for sid in ids:
        o = observed[sid]; truth = labels[sid]['ground_truth']; ignored = labels[sid]['ignored']
        light_flags = [ignored[int(r['source_row'])] for r in o['rows']]
        light = counts(o['rows'], truth, light_flags); light_control.append(light)
        if sid in accepted:
            rows = o['rows']; flags = light_flags; fallback_hash = None
        else:
            if native is not None:
                base = native[sid]['deepsolo']; fallback_hash = pins[str(native_path)]
            else:
                entry = archive[sid]; file = Path(entry['evidence_path'])
                if sha(file) != entry['evidence_sha256']:
                    raise ValueError('Changed individual ICDAR2015 fallback evidence')
                value = read(file); base = value['predictions']; fallback_hash = entry['evidence_sha256']
                if value['ground_truth'] != truth:
                    raise ValueError('Fallback and current care annotations differ')
                pins[str(file)] = fallback_hash
            base = [dict(r, raw_text='' if r.get('invalid_envelope') is not None else r['text'], word_source='deepsolo') for r in base]
            if recount(base, truth, [r['ignored'] for r in base]) != labels[sid]['teacher']:
                raise ValueError('Independent native fallback recount differs')
            rows = pipeline.combine_after_fallback(o, base)
            flags = [ignored[int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
        measured = counts(rows, truth, flags)
        if measured != recount(rows, truth, flags):
            raise ValueError('Independent combined TEST recount differs')
        results[sid] = dict(action='deepsolo' if sid in selected else 'light',
            rows=rows, ignored=flags, counts=measured, ground_truth=truth,
            light_counts=light, score=o['score'], fallback_sha256=fallback_hash)
    aggregate = micro(v['counts'] for v in results.values())
    expected_care = read(source/'resolved_configuration.json')['config']['budgets'][case['dataset']]['care_words']
    if aggregate['images'] != len(ids) or aggregate['gt_words'] != expected_care:
        raise ValueError('Retention full TEST denominator differs')
    baseline = read(source/'results.json')['results'][name]['primary']
    score = baseline['combined']['f1']
    if baseline['deepsolo_images'] != len(selected) or baseline['total_images'] != len(ids):
        raise ValueError('Reference comparison uses a different budget')
    seal(out/'per_image.json', results); seal(out/'input_sha256.json', pins)
    result = dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction', case=name,
        counts=aggregate, deepsolo_images=len(selected), total_images=len(ids),
        deepsolo_percentage=100*len(selected)/len(ids), baseline_f1=score,
        delta_f1_points=100*(aggregate['f1']-score), minimum_f1=c['minimum_f1'][name],
        target_met=(aggregate['f1'] > c['minimum_f1'][name] if case['dataset'] == 'totaltext'
                    else aggregate['f1'] >= c['minimum_f1'][name]-1e-12),
        all_light=dict(counts=micro(light_control), deepsolo_images=0, total_images=len(ids), deepsolo_percentage=0.),
        selected_fallback_evidence_images=len(selected), actual_new_deepsolo_neural_inferences=0,
        actual_new_neural_inferences=0, test_used_for_selection=False, historical_test_exposure=True,
        additional_map_models=pipeline.choice['additional_map_models'],
        scope='Logical routing over qualified cached real predictions; no online latency measurement')
    seal(out/'results.json', result)
    print(__import__('json').dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--stage', choices=['route', 'evaluate'], required=True)
    a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        (route if a.stage == 'route' else evaluate)(a.run_root, a.config)
