"""Select an existing emission grid using image-excluded TRAIN predictions."""

import argparse
import itertools
from pathlib import Path
import shutil

from recognition.adapted_word_policy import emit
from recognition.common_budget_pipeline import counts, micro
from recognition.fixed_expert_budget import fixed_trace
from scripts.icdar_budget_data import read, sha
from scripts.prepare_fixed_expert_budget import vocabulary
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def run(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(config_path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for path, digest in c['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed recalibration source: '+path)
    source = Path(c['source_root']); original = read(source/'resolved_configuration.json')
    case = read(source/'data/oof/case.json')
    if not case['expert_image_excluded'] or case['fit_ids'] != original['partitions']['fit']:
        raise ValueError('Genuine complete image-excluded FIT source required')
    lex = vocabulary(original['config']['fallback_profiles'][c['lexicon_profile']]['train_lexicon'])
    obs = read(source/'data/oof/observations.json'); labels = read(source/'data/oof/labels.json')
    trials = []
    for seq, joint, lexical in itertools.product(c['sequence_thresholds'], c['joint_thresholds'], c['lexical_policies']):
        policy = dict(sequence_threshold=seq, joint_threshold=joint, lexical=lexical,
            require_alphanumeric=True, same_text_nms_iou=c['same_text_nms_iou'])
        values = []; edits = 0
        for sid in case['fit_ids']:
            rows, changed = emit(obs[sid]['raw'], lex, policy); edits += changed
            values.append(counts(rows, labels[sid]['ground_truth'],
                [labels[sid]['raw_ignored'][int(r['source_row'])] for r in rows]))
        trials.append(dict(policy=policy, counts=micro(values), lexical_edits=edits,
            deepsolo_images=0, total_images=len(values), deepsolo_percentage=0.))
    best = max(t['counts']['f1'] for t in trials)
    winner = min((t for t in trials if t['counts']['f1'] >= best-c['selection_tolerance']),
        key=lambda t: (t['lexical_edits'], -t['counts']['f1'], t['policy']['sequence_threshold'],
                       t['policy']['joint_threshold'], t['policy']['lexical']))
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    seal(root/'emission_frozen.json', dict(status='FROZEN', winner=winner, trials=trials,
        selection_scope='All183 image-excluded FIT observations; no TEST or46-image validation labels',
        test_used_for_selection=False, source_case_sha256=sha(source/'data/oof/case.json'),
        raw_detector_expert_sha256=sha(original['config']['expert_frozen'])))
    settings = dict(original['config']); settings['output_root'] = str(root)
    settings['emission_override'] = dict(policy=str(root/'emission_frozen.json'),
        train_lexicon=settings['fallback_profiles'][c['lexicon_profile']]['train_lexicon'])
    settings['pinned_inputs'] = dict(settings['pinned_inputs'], **{
        str(root/'emission_frozen.json'): sha(root/'emission_frozen.json')})
    settings['scientific_choices']['ASSUMPTION'] += (
        ' Follow-up: freeze the same predeclared emission grid on actual image-excluded183 FIT '
        'predictions instead of small46-image validation; same padding0/checkpoint. '
        'Benchmark TRAIN Generic supplies declared word side information. '
        'Policy selection across folds is conditional, not a nested unbiased estimate.')
    code = {}
    for package in settings['snapshot_packages']:
        for path in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
            code[str(path.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    plan = dict(original, config=settings, source_sha256=code,
        recalibration_config_sha256=sha(config_path), parent_run=str(source),
        parent_configuration_sha256=sha(source/'resolved_configuration.json'))
    seal(root/'resolved_configuration.json', plan)
    for name in ('fallbacks.json', 'fallback_input_sha256.json'):
        destination = root/'data'/name; destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source/'data'/name, destination)
    for view in settings['required_views']:
        old = source/'data'/view; new = root/'data'/view
        observations = read(old/'observations.json'); truth = read(old/'labels.json')
        for sid, o in observations.items():
            rows, _ = emit(o['raw'], lex, winner['policy'])
            o.update(rows=rows, trace=fixed_trace(o['raw'], rows))
            truth[sid]['light'] = counts(rows, truth[sid]['ground_truth'],
                [truth[sid]['raw_ignored'][int(r['source_row'])] for r in rows])
        seal(new/'observations.json', observations); seal(new/'labels.json', truth)
        for name in ('case.json', 'input_sha256.json'):
            shutil.copyfile(old/name, new/name)
    print(__import__('json').dumps(dict(status='PREPARED', winner=winner,
        test_used_for_selection=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.config)
