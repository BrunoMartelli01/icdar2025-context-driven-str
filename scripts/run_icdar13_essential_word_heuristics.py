"""Freeze a small validation-selected gate before scoring CAL and complete TEST."""

import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.essential_word_heuristics import risk_scores, select_deepsolo
from recognition.f1_quality_study import aggregate
from scripts.icdar_budget_data import read, sha, key
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]
FIELDS = ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs')


def prepare(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate profile required')
    c = read(path)
    assert c['mode'] == 'approximate' and c['claim_label'] == 'approximate reproduction'
    for name, digest in c['pinned_inputs'].items():
        assert sha(name) == digest, name
    assert read(Path(c['expert_root'])/'independent_audit.json')['status'] == 'PASS'
    assert read(Path(c['train_teacher_root']).parent/'independent_audit.json')['status'] == 'PASS'
    expert_plan = read(Path(c['expert_root'])/'resolved_configuration.json')
    training = read(Path(expert_plan['config']['training_root'])/'resolved_configuration.json')
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    code = {}
    for folder in ('models', 'recognition', 'segmentation', 'scripts', 'localize',
                   'caption', 'scoring', 'pipeline', 'research'):
        for file in (REPO/folder).rglob('*.py'):
            relative = file.relative_to(REPO)
            dest = root/'code_snapshot'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
            code[str(relative)] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction', config=c, config_sha256=sha(path),
        partitions=training['partitions'], records={r['sample_id']: r
            for r in training['train_records']+training['test_records']},
        expert_configuration=expert_plan['config'], code_sha256=code,
        historical_test_exposure=True, test_used_for_selection=False,
        selection_scope='Nine fixed score functions; choose using heldout selection only, '
                        'without regressor fitting or seen-FIT training outcomes'))
    print('PREPARED', root, flush=True)


def observe(plan, root, role):
    c = plan['config']
    expert = Path(c['expert_root'])
    if role == 'test':
        lexicon = DeepSoloLexicon.from_files(plan['expert_configuration']['test_vocabulary'],
            plan['expert_configuration']['test_pairs'], distance_exclusive=c['distance_exclusive'])
    else:
        words = (expert/'fit_generic_vocabulary.txt').read_text().splitlines()
        lexicon = DeepSoloLexicon(words, read(expert/'fit_generic_pairs.json'),
                                 distance_exclusive=c['distance_exclusive'])
    observations = {}
    for sid in plan['partitions'][role]:
        sample_path = expert/'selected_expert'/role/'samples'/(key(sid)+'.json')
        maps_path = expert/'selected_expert'/role/'maps'/(key(sid)+'.npz')
        data = read(sample_path)
        assert data['maps_sha256'] == sha(maps_path)
        assert not data['ground_truth_used_for_prediction']
        with np.load(maps_path, allow_pickle=False) as maps:
            # The score boundary accepts observations only, never counts/GT.
            scores = risk_scores(data['rows'], data['raw_rows'], maps['probabilities'],
                plan['records'][sid]['image_wh'], lexicon, c['feature_settings'])
        assert set(scores) == set(c['heuristics'])
        value = dict(scores=scores, image_sha256=data['image_sha256'],
            light_sample_sha256=sha(sample_path), maps_sha256=sha(maps_path),
            ground_truth_used_for_prediction=False)
        seal(root/'observations'/role/(key(sid)+'.json'), value)
        observations[sid] = value
    return observations


def branches(plan, root, role):
    c = plan['config']
    teacher = read(c['fixed_test_counts']) if role == 'test' else read(
        Path(c['train_teacher_root'])/'counts.json')
    result = {}
    for sid in plan['partitions'][role]:
        source = Path(c['expert_root'])/'selected_expert'/role/'samples'/(key(sid)+'.json')
        data = read(source)
        reference = teacher['samples'][sid]
        assert data['image_sha256'] == reference['image_sha256']
        assert sha(reference['evidence_path']) == reference['evidence_sha256']
        assert data['counts']['gt_words'] == reference['counts']['gt_words']
        result[sid] = dict(light=data['counts'], deepsolo=reference['counts'])
    seal(root/'branches'/(role+'.json'), result)
    return result


def summarize(paired, sent):
    sent = set(sent)
    assert sent <= set(paired)
    return dict(counts=aggregate([branch['deepsolo' if sid in sent else 'light']
        for sid, branch in paired.items()]), deepsolo_ids=sorted(sent),
        deepsolo_images=len(sent), total_images=len(paired),
        deepsolo_percentage=100*len(sent)/len(paired),
        scope='Logical whole-image decisions on cached fixed expert outputs')


def select(plan, root):
    c = plan['config']
    observations = observe(plan, root, 'selection')
    paired = branches(plan, root, 'selection')
    keys = {sid: data['image_sha256'] for sid, data in observations.items()}
    points = []
    for heuristic in c['heuristics']:
        sent = select_deepsolo({sid: d['scores'][heuristic] for sid, d in observations.items()},
                              keys, c['validation_deepsolo_images'])
        points.append(dict(heuristic=heuristic, **summarize(paired, sent)))
    best = max(enumerate(points), key=lambda item: (item[1]['counts']['f1'], -item[0]))[1]
    seal(root/'selection.json', dict(points=points, chosen=best,
        selection_ids=plan['partitions']['selection'], test_used_for_selection=False,
        config_sha256=sha(root/'resolved_configuration.json')))
    print('FROZEN', best, flush=True)


def evaluate(plan, root):
    c = plan['config']
    frozen = read(root/'selection.json')
    assert not frozen['test_used_for_selection']
    assert frozen['config_sha256'] == sha(root/'resolved_configuration.json')
    results = {}
    for role in ('calibration', 'test'):
        observations = observe(plan, root, role)
        paired = branches(plan, root, role)
        keys = {sid: data['image_sha256'] for sid, data in observations.items()}
        budget = c['test_deepsolo_images'] if role == 'test' else c['validation_deepsolo_images']
        sent = select_deepsolo({sid: d['scores'][frozen['chosen']['heuristic']]
            for sid, d in observations.items()}, keys, budget)
        results[role] = dict(primary=summarize(paired, sent),
            light_only=summarize(paired, []), deepsolo_only=summarize(paired, list(paired)))
        if role == 'test':
            assert len(paired) == c['expected_test_images']
            assert results[role]['primary']['counts']['gt_words'] == c['expected_test_care_words']
        print('ROUTER', role, results[role]['primary'], flush=True)
    seal(root/'results.json', dict(status='COMPLETE', mode='approximate',
        claim_label='approximate reproduction', results=results,
        selected_heuristic=frozen['chosen']['heuristic'], selection_sha256=sha(root/'selection.json'),
        actual_new_neural_inferences=0, added_model_parameters=0,
        target_achieved=results['test']['primary']['counts']['f1'] >= c['target_f1']))


def audit(plan, root, output):
    c = plan['config']
    frozen = read(root/'selection.json')
    checks = Counter()
    for role in ('selection', 'calibration', 'test'):
        # Re-execute the observation-only score functions and compare immutable records.
        observations = observe(plan, root, role)
        paired = branches(plan, root, role)
        keys = {sid: d['image_sha256'] for sid, d in observations.items()}
        points = frozen['points'] if role == 'selection' else [
            read(root/'results.json')['results'][role]['primary']]
        for index, point in enumerate(points):
            heuristic = c['heuristics'][index] if role == 'selection' else frozen['chosen']['heuristic']
            budget = c['test_deepsolo_images'] if role == 'test' else c['validation_deepsolo_images']
            sent = select_deepsolo({sid: d['scores'][heuristic] for sid, d in observations.items()}, keys, budget)
            assert point['deepsolo_ids'] == sorted(sent)
            counts = Counter()
            for sid, branch in paired.items():
                counts.update(branch['deepsolo' if sid in sent else 'light'])
            for field in FIELDS:
                assert point['counts'][field] == counts[field]
            assert point['counts']['f1'] == 2*counts['tp']/(counts['detected_words']+counts['gt_words'])
            assert point['deepsolo_images'] == budget and point['total_images'] == len(paired)
            assert point['deepsolo_percentage'] == 100*budget/len(paired)
            checks['routing_points'] += 1
        checks['observation_image_replays'] += len(observations)
    chosen = max(enumerate(frozen['points']), key=lambda item: (item[1]['counts']['f1'], -item[0]))[1]
    assert chosen == frozen['chosen']
    assert read(root/'results.json')['selection_sha256'] == sha(root/'selection.json')
    seal(output, dict(status='PASS', checks=dict(checks),
        results_sha256=sha(root/'results.json'), actual_new_neural_inferences=0,
        scope='GT-free feature/decision replay and independent Counter aggregate sums; '
              'underlying per-image expert and teacher count audits linked in pinned inputs'))
    print('PASS', dict(checks), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage', choices=['prepare', 'select', 'evaluate', 'audit'], required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.config)
        return
    root = args.run_root
    plan = read(root/'resolved_configuration.json')
    for name, digest in plan['config']['pinned_inputs'].items():
        assert sha(name) == digest, name
    for name, digest in plan['code_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest, name
    if args.stage == 'audit':
        if args.output is None:
            raise ValueError('Explicit audit destination required')
        audit(plan, root, args.output)
    else:
        {'select': select, 'evaluate': evaluate}[args.stage](plan, root)


if __name__ == '__main__':
    main()
