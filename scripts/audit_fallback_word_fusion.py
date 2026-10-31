"""Independently recount and replay every frozen fallback fusion policy."""

import argparse
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from threadpoolctl import threadpool_limits

from recognition.risk_fusion_deployment import RiskFusionPipeline
from scripts.audit_fixed_expert_budget import norm, overlap
from scripts.audit_lightweight_word_pipeline import recount, total
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def independent_fusion(light, fallback, policy, complement):
    result = [dict(r) for r in fallback]
    if policy['enabled']:
        pairs = []
        for i, heavy in enumerate(result):
            if heavy['bbox'] is None:
                continue
            for j, word in enumerate(light):
                iou = overlap(heavy['bbox'], word['bbox'])
                if word['word_probability'] >= policy['minimum_word_probability'] and iou >= policy['minimum_iou']:
                    pairs.append((-iou, -word['word_probability'], i, j))
        used_heavy = set(); used_light = set()
        for _, _, i, j in sorted(pairs):
            if i in used_heavy or j in used_light:
                continue
            used_heavy.add(i); used_light.add(j)
            old, new = norm(result[i]['raw_text']), norm(light[j]['raw_text'])
            distance = policy['maximum_edit_distance']
            if new and old != new and (distance is None or Levenshtein.distance(old, new) <= distance):
                result[i]['raw_text'] = light[j]['raw_text']; result[i]['audit_replaced'] = True
    n = sum(bool(r.get('audit_replaced')) for r in result)
    if complement['confidence'] is not None:
        for word in light:
            if word['word_probability'] >= complement['confidence'] and not any(
                    r['bbox'] is not None and overlap(word['bbox'], r['bbox']) >= complement['maximum_overlap'] for r in result):
                result.append(dict(word, fixed_light_complement=True))
    return result, n


def audit(root):
    plan = read(root/'resolved_configuration.json'); cfg = plan['fusion_selection_config']
    if plan['mode'] != 'approximate' or not plan['risk_fusion'] or plan['test_labels_opened'] or plan['test_used_for_selection']:
        raise ValueError('TRAIN-only approximate fusion audit required')
    for path, digest in read(root/'source_input_sha256.json').items():
        if sha(path) != digest:
            raise ValueError('Changed fusion input')
    parent = Path(plan['source_root']); q = read(parent/'deployment_qualification.json')
    if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256'] or sha(parent/'deployment_qualification.json') != plan['parent_qualification_sha256']:
        raise ValueError('Changed parent risk-gate qualification')
    for path, digest in q['artifact_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed parent risk-gate artifact')
    parent_plan = read(parent/'resolved_configuration.json'); source = Path(parent_plan['source_root'])
    choice = read(parent/'selection_frozen.json')['winner']; gate = choice['winner']; view = choice['view']
    case = read(root/'data'/view/'case.json'); fit, val = case['fit_ids'], case['validation_ids']; ids = fit+val
    obs = read(source/'data'/view/'observations.json'); labels = read(source/'data'/view/'labels.json')
    probabilities = read(source/'models'/view/'word_probabilities.json'); fallback = read(source/'data/fallbacks.json')[choice['fallback_profile']]
    light = {}; scores = {}
    for sid in ids:
        p = probabilities[sid]
        light[sid] = [dict(r, word_probability=float(p[int(r['source_row'])])) for r in obs[sid]['rows']
                      if p[int(r['source_row'])] >= gate['admission_threshold']]
        indices = {int(r['source_row']) for r in light[sid]}
        if gate['heuristic'] == 'expected_errors':
            scores[sid] = -sum((1-v) if i in indices else v for i, v in enumerate(p))
        elif gate['heuristic'] == 'weakest_word':
            scores[sid] = min((p[i] for i in indices), default=0.)
        else:
            raise ValueError('Unknown frozen trained risk aggregation')
    quotas = {'fit': round(len(fit)*case['fallback_images']/case['total_images']),
              'validation': round(len(val)*case['fallback_images']/case['total_images'])}
    kept = {}
    for name, population in [('fit', fit), ('validation', val)]:
        eligible = [s for s in population if light[s]] if gate['guard'] else population
        kept[name] = sorted(eligible, key=lambda s: (-scores[s], case['keys'][s], s))[:len(population)-quotas[name]]
    if sorted(kept['fit']) != gate['crossfit']['accepted_ids'] or sorted(kept['validation']) != gate['result']['accepted_ids']:
        raise ValueError('Independent original route differs')
    accepted = set(kept['fit']+kept['validation']); trials = read(root/'trials.json'); recounts = 0
    for trial in trials:
        branches = read(root/'training_branches'/(str(trial['index'])+'.json')); changes = 0
        for sid in ids:
            gt = labels[sid]['ground_truth']; ignored = labels[sid]['raw_ignored']; words = light[sid]
            fused, changed = independent_fusion(words, fallback[sid]['rows'], trial['policy'], choice['complement'])
            if sid not in accepted:
                changes += changed
            for branch, rows in [('light', words), ('deepsolo', fused)]:
                flags = [ignored[int(r['source_row'])] if branch == 'light' or r.get('fixed_light_complement') else r['ignored'] for r in rows]
                if recount(rows, gt, flags) != branches[sid][branch]:
                    raise ValueError('Independent fusion branch recount differs')
                recounts += 1
        for part, population, result in [('fit', fit, trial['crossfit']), ('validation', val, trial['validation'])]:
            aggregate = total(branches[s]['light' if s in kept[part] else 'deepsolo'] for s in population)
            if aggregate != result['combined'] or sorted(kept[part]) != result['accepted_ids']:
                raise ValueError('Independent fusion route totals differ')
        pooled = total(branches[s]['light' if s in accepted else 'deepsolo'] for s in ids)
        if pooled != trial['pooled_counts'] or changes != trial['changed_routed_words']:
            raise ValueError('Independent fusion pooled totals or change count differ')
    baseline = trials[0]['validation']['combined']['f1']
    eligible = [t for t in trials if t['validation']['combined']['f1'] >=
                max(cfg['minimum_validation_f1'], baseline-cfg['validation_tolerance'])]
    best = max(t['selection_score'] for t in eligible)
    expected = min((t for t in eligible if t['selection_score'] >= best-cfg['selection_tolerance']),
        key=lambda t: (t['changed_routed_words'], -t['selection_score'], t['index']))
    frozen = read(root/'selection_frozen.json')
    if (frozen['winner'] != choice or frozen['fusion_selection'] != expected or
            frozen['fusion_policy'] != expected['policy'] or frozen['test_used_for_selection']):
        raise ValueError('Frozen fusion selection differs')
    pipeline = RiskFusionPipeline(root, verify_qualification=False); deployed = {}
    for sid in val:
        o = obs[sid]; deployed[sid] = pipeline.observe(o['raw'], o['rows'], o['image_wh'], trace=o['trace'], map_summary=o['map_summary'])
        np.testing.assert_allclose(deployed[sid]['probabilities'], probabilities[sid], rtol=1e-12, atol=1e-12)
    decision = pipeline.route(deployed, {s: case['keys'][s] for s in val}, fallback_images=quotas['validation'])
    if decision['accepted_ids'] != sorted(kept['validation']):
        raise ValueError('Annotation-free fusion parent route differs')
    fallback_replays = 0
    for sid in set(val)-set(kept['validation']):
        actual = pipeline.combine_after_fallback(deployed[sid], fallback[sid]['rows'])
        independent, _ = independent_fusion(light[sid], fallback[sid]['rows'], expected['policy'], choice['complement'])
        if [(r['raw_text'], r['bbox']) for r in actual] != [(r['raw_text'], r['bbox']) for r in independent]:
            raise ValueError('Annotation-free post-fallback fusion differs')
        fallback_replays += 1
    pins = {str(parent/'deployment_qualification.json'): sha(parent/'deployment_qualification.json'),
            str(root/'selection_frozen.json'): sha(root/'selection_frozen.json'),
            str(root/'source_input_sha256.json'): sha(root/'source_input_sha256.json')}
    for rel in frozen['artifact_sha256']:
        p = root/rel; pins[str(p)] = sha(p)
    for name in ('fallback_word_fusion.py', 'risk_fusion_deployment.py', 'fusion_deployment_factory.py'):
        p = Path(__file__).resolve().parents[1]/'recognition'/name; pins[str(p)] = sha(p)
    report = root/'independent_audit_v1.json'
    seal(report, dict(status='PASS', mode='approximate', claim_label='approximate reproduction',
        branch_recounts=recounts, policy_replays=len(trials), validation_deployment_replays=len(val),
        validation_fallback_replays=fallback_replays, selection=expected,
        test_used_for_selection=False, auditor_sha256=sha(__file__),
        scope='Independent frozen gate, geometry, text, care and budget recount; expert/word stacking remains conditional.'))
    seal(root/'deployment_qualification.json', dict(status='PASS', audit_path=str(report),
        audit_sha256=sha(report), artifact_sha256=pins))
    print(__import__('json').dumps(dict(status='PASS', branch_recounts=recounts,
        policy_replays=len(trials), validation_deployment_replays=len(val), validation_fallback_replays=fallback_replays)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True); a = p.parse_args()
    with threadpool_limits(limits=read(a.run_root/'resolved_configuration.json')['config']['cpu_threads']):
        audit(a.run_root)
