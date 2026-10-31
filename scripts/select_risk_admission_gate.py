"""Select GT-free word-risk admission and fixed-budget routing on TRAIN/validation."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import counts, micro, choose_light, route
from recognition.fixed_expert_budget import fixed_words_with_risk, fixed_heuristic, conditional_complement
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def selection_key(value, prefer_fewer_filtered_words):
    t = value['winner']
    quality = (t['removed_words'], -t['selection_score']) if prefer_fewer_filtered_words else (
        -t['selection_score'], t['removed_words'])
    return (value['additional_map_models'], value['extra_reader_crops'], *quality,
            t['admission_threshold'], t['heuristic'], t['guard'], value['target_name'])


def run(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate configuration required')
    c = read(config_path)
    if c['mode'] != 'approximate' or c['bootstrap_draws'] < 2:
        raise ValueError('Explicit approximate risk-selection configuration required')
    for path, digest in c['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed pinned risk-selection source')
    source = Path(c['source_root']); q = read(source/'deployment_qualification.json')
    if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
        raise ValueError('Independently qualified source gate required')
    for path, digest in q['artifact_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed qualified source gate artifact')
    plan = read(source/'resolved_configuration.json'); view = c['source_view']
    case = read(source/'data'/view/'case.json'); fit = case['fit_ids']; val = case['validation_ids']; ids = fit+val
    if c['require_expert_excluded'] and not case['expert_image_excluded']:
        raise ValueError('Image-excluded FIT expert predictions required')
    obs = read(source/'data'/view/'observations.json'); labels = read(source/'data'/view/'labels.json')
    probability = read(source/'models'/view/'word_probabilities.json'); fallback = read(source/'data/fallbacks.json')
    profiles = [x for x in read(source/'models'/view/'completed.json')['results'] if x['primary_eligible']]
    if not profiles or any(not 0 <= t <= 1 for t in c['admission_thresholds']):
        raise ValueError('Valid admission grid and eligible fallback profiles required')
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    settings = dict(plan['config']); settings['output_root'] = str(root); settings['required_views'] = [view]
    settings['scientific_choices'] = c['scientific_choices']
    files = {}
    for package in settings['snapshot_packages']:
        for path in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
            files[str(path.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    resolved = dict(plan, config=settings, source_root=str(source), source_sha256=files,
        risk_admission=True, risk_selection_config=c, risk_selection_config_sha256=sha(config_path),
        source_configuration_sha256=sha(source/'resolved_configuration.json'),
        source_qualification_sha256=sha(source/'deployment_qualification.json'),
        test_labels_opened=False, test_used_for_selection=False)
    seal(root/'resolved_configuration.json', resolved)
    data = root/'data'/view; data.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'data'/view/'case.json', data/'case.json')
    folder = root/'models'/view; folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'models'/view/'word_full.joblib', folder/'word_full.joblib')
    quotas = {name: round(len(values)*case['fallback_images']/case['total_images'])
              for name, values in [('fit', fit), ('validation', val)]}
    trials = []; all_branches = {}
    for detail in profiles:
        profile = detail['fallback_profile']; complement = detail['complement']
        for ti, threshold in enumerate(c['admission_thresholds']):
            light = {}; branches = {}; removed = 0
            for sid in ids:
                all_words = fixed_words_with_risk(obs[sid]['rows'], probability[sid])
                words = [r for r in all_words if r['word_probability'] >= threshold]
                removed += len(all_words)-len(words); light[sid] = words
                gt = labels[sid]['ground_truth']; flags = labels[sid]['raw_ignored']
                light_count = counts(words, gt, [flags[int(r['source_row'])] for r in words])
                fused = conditional_complement(words, fallback[profile][sid]['rows'], complement)
                ignored = [flags[int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in fused]
                branches[sid] = dict(light=light_count, deepsolo=counts(fused, gt, ignored))
            key = detail['target_name']+'__'+str(ti); all_branches[key] = branches
            for heuristic in c['heuristics']:
                scores = {s: fixed_heuristic(obs[s]['raw'], light[s], probability[s], heuristic) for s in ids}
                for guard in c['guards']:
                    kept = {}; selected = set()
                    for part, population in [('fit', fit), ('validation', val)]:
                        eligible = [s for s in population if light[s]] if guard else population
                        if len(eligible) < len(population)-quotas[part]:
                            break
                        accepted = choose_light({s: scores[s] for s in population},
                            {s: case['keys'][s] for s in population}, quotas[part], eligible)
                        kept[part] = route({s: branches[s] for s in population}, accepted); selected.update(accepted)
                    if len(kept) != 2:
                        continue
                    pooled = micro(branches[s]['light' if s in selected else 'deepsolo'] for s in ids)
                    t = dict(models=[], groups=[], scalars=0, reference=None, heuristic=heuristic, guard=guard,
                        admission_threshold=threshold, result=kept['validation'], crossfit=kept['fit'],
                        pooled_counts=pooled, selection_score=pooled['f1'], removed_words=removed,
                        training_branch_key=key, total_selection_images=len(ids),
                        deepsolo_selection_images=quotas['fit']+quotas['validation'],
                        deepsolo_selection_percentage=100*(quotas['fit']+quotas['validation'])/len(ids))
                    trials.append(dict(detail, winner=t, additional_map_models=0))
    eligible = [v for v in trials if v['winner']['result']['combined']['f1'] >= c['minimum_validation_f1']]
    if not eligible:
        raise ValueError('No trained risk-admission gate satisfies the validation floor')
    best = max(eligible, key=lambda v: v['winner']['selection_score']); t = best['winner']
    kept = set(t['crossfit']['accepted_ids']+t['result']['accepted_ids']); branches = all_branches[t['training_branch_key']]
    values = np.array([[branches[s]['light' if s in kept else 'deepsolo'][k] for k in ('tp', 'fp', 'fn')] for s in ids], np.int64)
    indices = np.random.default_rng(c['seed']).integers(len(ids), size=(c['bootstrap_draws'], len(ids)))
    boot = values[indices].sum(axis=1); den = 2*boot[:, 0]+boot[:, 1]+boot[:, 2]
    if np.any(den == 0):
        raise ValueError('Empty bootstrap lexical denominator')
    se = float(np.std(2*boot[:, 0]/den, ddof=1)); lower = best['winner']['selection_score']-se
    within = [v for v in eligible if v['winner']['selection_score'] >= lower]
    chosen = min(within, key=lambda v: selection_key(v, c.get('prefer_fewer_filtered_words', False)))
    for name, branches in all_branches.items():
        seal(root/'training_branches'/(name+'.json'), branches)
    seal(root/'trials.json', trials)
    target = folder/chosen['target_name']; target.mkdir(parents=True, exist_ok=False)
    seal(target/'winner.json', chosen['winner'])
    source_pins = {}
    for rel in ['resolved_configuration.json', 'deployment_qualification.json', 'data/fallbacks.json',
                'data/'+view+'/case.json', 'data/'+view+'/observations.json', 'data/'+view+'/labels.json',
                'models/'+view+'/word_probabilities.json', 'models/'+view+'/word_folds.json',
                'models/'+view+'/word_full.joblib', 'models/'+view+'/completed.json']:
        path = source/rel; source_pins[str(path)] = sha(path)
    if settings.get('emission_override') is not None:
        path = Path(settings['emission_override']['policy']); source_pins[str(path)] = sha(path)
    seal(root/'source_input_sha256.json', source_pins)
    frozen_files = [root/'resolved_configuration.json', data/'case.json', folder/'word_full.joblib', target/'winner.json']
    seal(root/'selection_frozen.json', dict(status='FROZEN', winner=chosen,
        uncertainty=dict(best_selection_score=best['winner']['selection_score'],
            bootstrap_standard_error=se, one_se_floor=lower, eligible_candidates=len(eligible),
            candidates_within_one_se=len(within), bootstrap_draws=c['bootstrap_draws'], seed=c['seed'],
            scope='Conditional image bootstrap after finite candidate search; not nested generalization uncertainty'),
        artifact_sha256={str(p.relative_to(root)): sha(p) for p in frozen_files}, test_used_for_selection=False,
        validation_target_met=True))
    print(__import__('json').dumps(dict(status='FROZEN', target=chosen['target_name'],
        heuristic=chosen['winner']['heuristic'], admission_threshold=chosen['winner']['admission_threshold'],
        pooled_f1=chosen['winner']['selection_score'], validation_f1=chosen['winner']['result']['combined']['f1'],
        deepsolo_images=chosen['winner']['deepsolo_selection_images'], total_images=len(ids),
        deepsolo_percentage=chosen['winner']['deepsolo_selection_percentage'], standard_error=se)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True); a = p.parse_args()
    with threadpool_limits(limits=read(a.config)['cpu_threads']):
        run(a.config)
