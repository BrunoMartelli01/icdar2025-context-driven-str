"""Recompute all fixed-budget retention TEST decisions and lexical totals."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from recognition.fusion_deployment_factory import load_fusion_budget_pipeline
from scripts.audit_lightweight_word_pipeline import recount, total
from scripts.evaluate_fixed_expert_retention import source_paths
from scripts.icdar_budget_data import read, sha
from scripts.prepare_fixed_expert_retention import observe
from scripts.run_yolo_f1_quality_study import seal


def audit_cases(configured, requested):
    names = list(configured if requested is None else requested)
    if not names or len(set(names)) != len(names) or not set(names) <= set(configured):
        raise ValueError('Audit cases must be a nonempty subset of the declared cases')
    return names


def audit(root, config_path, requested_cases=None, report_name='independent_test_audit_v1.json'):
    if Path(report_name).name != report_name:
        raise ValueError('Audit report name must be a plain filename')
    c = read(config_path); source = Path(c['source_root']); cases = read(source/'cases.json')
    original = read(source/'results.json')['results']; aggregate = {}; counters = dict(
        test_scores_replayed=0, test_image_recounts=0, selected_fallback_replays=0)
    for name in audit_cases(c['cases'], requested_cases):
        folder = root/name; plan = read(folder/'resolved_configuration.json')
        if plan['config_sha256'] != sha(config_path) or plan['test_labels_opened'] or plan['mode'] != 'approximate':
            raise ValueError('Changed retention configuration or training boundary')
        pipeline = load_fusion_budget_pipeline(folder); test = folder/'test'; saved = read(test/'results.json')
        decision = read(test/'decisions_before_fallback.json'); outputs = read(test/'per_image.json')
        _, obs_path = source_paths(folder, plan)
        stored_obs = read(test/'observations.json'); observations = read(obs_path)
        if (decision['ground_truth_used'] or decision['fallback_outputs_loaded'] or
                decision['observations_sha256'] != sha(test/'observations.json') or
                decision['test_observations_sha256'] != sha(obs_path)):
            raise ValueError('Changed pre-fallback retained decision')
        case = cases[name]; ids = plan['partitions']['test']; keys = case['tie_keys']
        if set(ids) != set(outputs) or set(ids) != set(observations):
            raise ValueError('Missing or extra TEST image')
        source_obs = read(source/'data'/(name+'__observations.json')); maps = read(source/'data'/(name+'__maps.json'))
        expert = read(source/'models'/name/'observable_outputs.json'); labels = read(source/'data'/(name+'__test_labels.json'))
        recalculated = {}; selected = set(decision['fallback_ids'])
        if case['dataset'] == 'totaltext':
            fallback_path = Path(c['tt_native_root'])/'test/branches.json'
            if sha(fallback_path) != read(source/'source_input_sha256.json')[str(fallback_path)]:
                raise ValueError('Changed native fallback file')
            native = read(fallback_path)['records']; teacher = None
        else:
            native = None; teacher_path = Path(c['ic15_teacher_root'])/'counts.json'
            evidence = read(Path(c['ic15_teacher_root'])/'independent_audit.json')
            if evidence['status'] != 'PASS' or evidence['counts_sha256'] != sha(teacher_path):
                raise ValueError('Changed native fallback audit')
            teacher = read(teacher_path)['samples']
        for sid in ids:
            o = observations[sid]; s = source_obs[sid]
            if o != observe(s['rows'], expert['corrected_rows'][sid], expert['traces'][sid], s['image_wh'], maps[sid]['maps']):
                raise ValueError('Changed source TEST observation')
            deployed = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
            np.testing.assert_allclose(deployed['score'], stored_obs[sid]['score'], rtol=1e-12, atol=1e-12)
            if deployed['rows'] != stored_obs[sid]['rows']:
                raise ValueError('Changed annotation-free deployment words')
            recalculated[sid] = deployed; counters['test_scores_replayed'] += 1
            result = outputs[sid]; truth = labels[sid]['ground_truth']; ignored = labels[sid]['ignored']
            if result['ground_truth'] != truth or result['action'] != ('deepsolo' if sid in selected else 'light'):
                raise ValueError('Changed TEST ground truth or action')
            if sid in selected:
                if native is not None:
                    base = native[sid]['deepsolo']
                else:
                    entry = teacher[sid]; path = Path(entry['evidence_path'])
                    if sha(path) != entry['evidence_sha256']:
                        raise ValueError('Changed selected fallback evidence')
                    base = read(path)['predictions']
                base = [dict(r, raw_text='' if r.get('invalid_envelope') is not None else r['text'], word_source='deepsolo') for r in base]
                rows = pipeline.combine_after_fallback(deployed, base)
                flags = [ignored[int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                counters['selected_fallback_replays'] += 1
            else:
                rows = deployed['rows']; flags = [ignored[int(r['source_row'])] for r in rows]
            if rows != result['rows'] or flags != result['ignored'] or recount(rows, truth, flags) != result['counts']:
                raise ValueError('Independent retained TEST per-image replay differs')
            counters['test_image_recounts'] += 1
        quota = c['budgets'][case['dataset']]['fallback_images']; keep = len(ids)-quota
        eligible = [s for s in ids if recalculated[s]['rows']] if pipeline.winner['guard'] else ids
        light = sorted(eligible, key=lambda s: (-recalculated[s]['score'], keys[s], s))[:keep]
        if sorted(light) != decision['accepted_ids'] or len(selected) != quota:
            raise ValueError('Independent retained TEST quota ranking differs')
        value = total(outputs[s]['counts'] for s in ids)
        if any(saved['counts'][k] != v for k, v in value.items()) or saved['counts']['images'] != len(ids):
            raise ValueError('Independent retained TEST aggregate differs')
        baseline = original[name]['primary']
        if baseline['deepsolo_images'] != quota or saved['baseline_f1'] != baseline['combined']['f1']:
            raise ValueError('Retention baseline comparison differs')
        aggregate[name] = dict(f1=saved['counts']['f1'], baseline_f1=saved['baseline_f1'],
            delta_f1_points=saved['delta_f1_points'], deepsolo_images=quota, total_images=len(ids),
            deepsolo_percentage=100*quota/len(ids), additional_map_models=saved['additional_map_models'],
            results_sha256=sha(test/'results.json'), qualification_sha256=sha(folder/'deployment_qualification.json'))
    path = root/report_name
    seal(path, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        counters=counters, results=aggregate, historical_test_exposure=True,
        scope='Independent count/deployment replay of frozen logical cached routes; no online latency claim',
        auditor_sha256=sha(__file__)))
    print(__import__('json').dumps(dict(status='PASS', counters=counters, results=aggregate)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--retention-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--cases', nargs='+')
    p.add_argument('--report-name', default='independent_test_audit_v1.json'); a = p.parse_args()
    with threadpool_limits(limits=4):
        audit(a.retention_root, a.config, a.cases, a.report_name)
