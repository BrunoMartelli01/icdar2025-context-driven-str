"""Select a no-extra-network fallback transcript fusion on TRAIN/validation."""

import argparse
import itertools
from pathlib import Path
import shutil

from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import counts, micro, choose_light, route
from recognition.fallback_word_fusion import replace_fallback_transcripts
from recognition.fixed_expert_budget import fixed_words_with_risk, fixed_heuristic, conditional_complement
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def run(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate fusion configuration required')
    c = read(path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed fusion selection source')
    parent = Path(c['source_root']); parent_plan = read(parent/'resolved_configuration.json')
    q = read(parent/'deployment_qualification.json')
    if (not parent_plan.get('risk_admission') or q['status'] != 'PASS' or
            sha(q['audit_path']) != q['audit_sha256']):
        raise ValueError('An independently qualified trained risk gate is required')
    for file, digest in q['artifact_sha256'].items():
        if sha(file) != digest:
            raise ValueError('Changed qualified parent gate artifact')
    choice = read(parent/'selection_frozen.json')['winner']; gate = choice['winner']; view = choice['view']
    source = Path(parent_plan['source_root']); case = read(parent/'data'/view/'case.json')
    fit, val = case['fit_ids'], case['validation_ids']; ids = fit+val
    obs = read(source/'data'/view/'observations.json'); labels = read(source/'data'/view/'labels.json')
    probabilities = read(source/'models'/view/'word_probabilities.json')
    fallback = read(source/'data/fallbacks.json')[choice['fallback_profile']]
    light = {}; scores = {}
    for sid in ids:
        words = fixed_words_with_risk(obs[sid]['rows'], probabilities[sid])
        light[sid] = [r for r in words if r['word_probability'] >= gate['admission_threshold']]
        scores[sid] = fixed_heuristic(obs[sid]['raw'], light[sid], probabilities[sid], gate['heuristic'])
    quotas = {name: round(len(population)*case['fallback_images']/case['total_images'])
              for name, population in [('fit', fit), ('validation', val)]}
    kept = {}
    for name, population in [('fit', fit), ('validation', val)]:
        eligible = [s for s in population if light[s]] if gate['guard'] else population
        kept[name] = choose_light({s: scores[s] for s in population}, {s: case['keys'][s] for s in population},
                                  quotas[name], eligible)
    if (sorted(kept['fit']) != gate['crossfit']['accepted_ids'] or
            sorted(kept['validation']) != gate['result']['accepted_ids']):
        raise ValueError('Frozen parent TRAIN/validation route changed')
    accepted = set(kept['fit']+kept['validation'])
    policies = [dict(enabled=False, minimum_word_probability=None, minimum_iou=None, maximum_edit_distance=None)]
    policies += [dict(enabled=True, minimum_word_probability=p, minimum_iou=iou, maximum_edit_distance=distance)
                 for p, iou, distance in itertools.product(c['word_probabilities'], c['minimum_ious'], c['maximum_edit_distances'])]
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    settings = dict(parent_plan['config']); settings['output_root'] = str(root); settings['scientific_choices'] = c['scientific_choices']
    files = {}
    for package in settings['snapshot_packages']:
        for file in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/file.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, dest)
            files[str(file.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(parent_plan, config=settings, source_root=str(parent),
        source_sha256=files, risk_fusion=True, fusion_selection_config=c, fusion_selection_config_sha256=sha(path),
        parent_configuration_sha256=sha(parent/'resolved_configuration.json'),
        parent_qualification_sha256=sha(parent/'deployment_qualification.json'),
        test_labels_opened=False, test_used_for_selection=False))
    folder = root/'data'/view; folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent/'data'/view/'case.json', folder/'case.json')
    trials = []
    for i, policy in enumerate(policies):
        branches = {}; changed_on_routed = 0
        for sid in ids:
            gt = labels[sid]['ground_truth']; ignored = labels[sid]['raw_ignored']; words = light[sid]
            light_count = counts(words, gt, [ignored[int(r['source_row'])] for r in words])
            replaced = replace_fallback_transcripts(words, fallback[sid]['rows'], policy)
            if sid not in accepted:
                changed_on_routed += sum(bool(r.get('replaced_by_light')) for r in replaced)
            fused = conditional_complement(words, replaced, choice['complement'])
            flags = [ignored[int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in fused]
            branches[sid] = dict(light=light_count, deepsolo=counts(fused, gt, flags))
        fit_result = route({s: branches[s] for s in fit}, kept['fit'])
        val_result = route({s: branches[s] for s in val}, kept['validation'])
        pooled = micro(branches[s]['light' if s in accepted else 'deepsolo'] for s in ids)
        result = dict(index=i, policy=policy, crossfit=fit_result, validation=val_result,
            pooled_counts=pooled, selection_score=pooled['f1'], changed_routed_words=changed_on_routed,
            deepsolo_images=quotas['fit']+quotas['validation'], total_images=len(ids),
            deepsolo_percentage=100*(quotas['fit']+quotas['validation'])/len(ids))
        trials.append(result); seal(root/'training_branches'/(str(i)+'.json'), branches)
    baseline = trials[0]['validation']['combined']['f1']
    eligible = [t for t in trials if t['validation']['combined']['f1'] >=
                max(c['minimum_validation_f1'], baseline-c['validation_tolerance'])]
    if not eligible:
        raise ValueError('No fusion policy meets the declared validation non-regression guard')
    best = max(t['selection_score'] for t in eligible)
    winner = min((t for t in eligible if t['selection_score'] >= best-c['selection_tolerance']),
        key=lambda t: (t['changed_routed_words'], -t['selection_score'], t['index']))
    seal(root/'trials.json', trials); seal(root/'fusion_policy.json', winner)
    pins = {}
    for file in [parent/'resolved_configuration.json', parent/'selection_frozen.json', parent/'deployment_qualification.json',
                 source/'data'/view/'case.json', source/'data'/view/'observations.json', source/'data'/view/'labels.json',
                 source/'data/fallbacks.json', source/'models'/view/'word_probabilities.json']:
        pins[str(file)] = sha(file)
    seal(root/'source_input_sha256.json', pins)
    frozen_files = [root/'resolved_configuration.json', root/'fusion_policy.json', folder/'case.json']
    seal(root/'selection_frozen.json', dict(status='FROZEN', winner=choice, fusion_policy=winner['policy'],
        fusion_selection=winner, artifact_sha256={str(p.relative_to(root)): sha(p) for p in frozen_files},
        test_used_for_selection=False, validation_target_met=True))
    print(__import__('json').dumps(dict(status='FROZEN', policy=winner['policy'],
        pooled_f1=winner['selection_score'], validation_f1=winner['validation']['combined']['f1'],
        replaced_words=winner['changed_routed_words'], deepsolo_images=winner['deepsolo_images'],
        total_images=winner['total_images'], deepsolo_percentage=winner['deepsolo_percentage'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True); a = p.parse_args()
    with threadpool_limits(limits=read(a.config)['cpu_threads']):
        run(a.config)
