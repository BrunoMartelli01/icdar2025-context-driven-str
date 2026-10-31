"""Recompute the cost-aware one-standard-error rule and GT-free deployment."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from recognition.fixed_expert_deployment import FixedExpertPipeline
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def audit(root):
    plan = read(root/'resolved_configuration.json'); cfg = plan['one_se_config']; parent = Path(plan['source_root'])
    pp = read(parent/'resolved_configuration.json'); view = pp['crossfit_config']['source_view']
    if plan['mode'] != 'approximate' or plan['test_labels_opened'] or plan['test_used_for_selection']:
        raise ValueError('TRAIN/validation-only approximate selection required')
    qualification = parent/cfg['qualification_name']; q = read(qualification)
    if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256'] or sha(qualification) != plan['parent_qualification_sha256']:
        raise ValueError('Changed independent parent qualification')
    for path, digest in q['artifact_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed audited parent trial or model')
    case = read(parent/'data'/view/'case.json'); ids = case['fit_ids']+case['validation_ids']; source = Path(pp['source_root'])
    candidates = []
    for detail in read(parent/'targets.json'):
        for t in read(parent/'models'/view/detail['target_name']/'trials.json'):
            if (t['result']['combined']['f1'] >= pp['crossfit_config']['minimum_validation_f1'] and
                    (t['models'] or t['heuristic'] in ('expected_errors', 'weakest_word'))):
                maps = int(case['map_available'] and any('maps' in g for g in t['groups']))*case['map_extra_models']
                candidates.append(dict(detail, winner=t, additional_map_models=maps))
    maximum = max(candidates, key=lambda d: d['winner']['selection_score'])
    branches = read(source/'models'/view/maximum['target_name']/'train_branches.json')
    keep = set(maximum['winner']['crossfit']['accepted_ids']+maximum['winner']['result']['accepted_ids'])
    values = np.array([[branches[s]['light' if s in keep else 'deepsolo'][k]
        for k in ('tp', 'fp', 'fn')] for s in ids], dtype=np.int64)
    indices = np.random.default_rng(cfg['seed']).integers(len(ids), size=(cfg['bootstrap_draws'], len(ids)))
    draw = values[indices].sum(axis=1); scores = 2*draw[:, 0]/(2*draw[:, 0]+draw[:, 1]+draw[:, 2])
    standard_error = float(np.std(scores, ddof=1)); floor = maximum['winner']['selection_score']-standard_error
    within = [x for x in candidates if x['winner']['selection_score'] >= floor]
    def independent_key(x):
        t = x['winner']
        return (x['additional_map_models'], x['extra_reader_crops'], t['scalars'], -t['selection_score'],
            len(t['models']), t['models'], t['reference'] or 0, t['heuristic'] or '', t['guard'], x['target_name'])
    expected = min(within, key=independent_key); frozen = read(root/'selection_frozen.json')
    metadata = frozen['uncertainty']
    if (frozen['winner'] != expected or frozen['test_used_for_selection'] or
            metadata['candidate_count'] != len(candidates) or metadata['eligible_within_one_se'] != len(within)):
        raise ValueError('Independent one-standard-error winner differs')
    np.testing.assert_allclose([metadata['bootstrap_standard_error'], metadata['one_se_floor']],
                               [standard_error, floor], rtol=1e-12, atol=1e-12)
    for p, digest in frozen['artifact_sha256'].items():
        if sha(root/p) != digest:
            raise ValueError('Changed frozen one-SE artifact')
    pipeline = FixedExpertPipeline(root, verify_qualification=False)
    observations = read(source/'data'/view/'observations.json'); val = case['validation_ids']; deployed = {}
    for sid in val:
        o = observations[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
    quota = round(len(val)*case['fallback_images']/case['total_images'])
    decision = pipeline.route(deployed, {s: case['keys'][s] for s in val}, fallback_images=quota)
    if decision['accepted_ids'] != expected['winner']['result']['accepted_ids']:
        raise ValueError('One-SE annotation-free validation route differs')
    pins = {str(qualification): sha(qualification)}
    for rel in frozen['artifact_sha256']:
        p = root/rel; pins[str(p)] = sha(p)
    if plan['config'].get('emission_override') is not None:
        p = Path(plan['config']['emission_override']['policy']); pins[str(p)] = sha(p)
    for name in ('fixed_expert_deployment.py', 'fixed_expert_budget.py', 'common_budget_pipeline.py'):
        p = Path(__file__).resolve().parents[1]/'recognition'/name; pins[str(p)] = sha(p)
    report = root/'independent_audit_v1.json'
    seal(report, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        candidates=len(candidates), within_one_se=len(within), bootstrap_draws=cfg['bootstrap_draws'],
        standard_error=standard_error, validation_deployment_replays=len(val),
        selected=expected['target_name'], selected_heuristic=expected['winner']['heuristic'],
        test_used_for_selection=False, auditor_sha256=sha(__file__),
        scope='Conditional bootstrap, non-nested source stacking and prior candidate selection remain limitations.'))
    seal(root/'deployment_qualification.json', dict(status='PASS', audit_path=str(report),
        audit_sha256=sha(report), artifact_sha256=pins))
    print(__import__('json').dumps(dict(status='PASS', candidates=len(candidates),
        within_one_se=len(within), selected=expected['target_name'],
        heuristic=expected['winner']['heuristic'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True); a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root)
