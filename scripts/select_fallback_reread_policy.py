"""Choose one low-cost fallback transcript replacement on TRAIN validation."""

import argparse
import itertools
from pathlib import Path

from recognition.common_budget_pipeline import counts, micro
from recognition.fallback_reread import choose_transcript
from scripts.audit_lightweight_word_pipeline import recount
from scripts.icdar_budget_data import read, sha
from scripts.run_lightweight_word_selection import dictionary
from scripts.train_lightweight_word_router import teacher
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
            raise ValueError('Changed rereading policy input: '+path)
    source = Path(c['admission_root']); plan = read(source/'resolved_configuration.json')
    data = read(source/'train_data.json'); fitting = plan['partitions']['fit']
    available = read(Path(c['rereading_root'])/'index.json')
    validation = [s for s in plan['partitions']['validation'] if s in available]
    if not c['smoke'] and len(validation) != c['expected_validation_images']:
        raise ValueError('Complete declared validation required')
    if not validation:
        raise ValueError('No validation images in rereading evidence')
    lexicon = dictionary(plan, data, fitting)
    baseline = {}; readings = {}; truth = {}; pins = {}
    for sid in validation:
        value = teacher(c, plan, plan['records'][sid], lexicon)
        entry = available[sid]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed rereading')
        raw = read(entry['path'])
        if raw['fallback_sha256'] != value['source_sha256'] or raw['ground_truth_used']:
            raise ValueError('Changed or GT-dependent rereading binding')
        baseline[sid] = value
        readings[sid] = {r['source_index']: r for r in raw['rows']}
        truth[sid] = data[sid]['ground_truth']; pins[entry['path']] = entry['sha256']
    policies = [dict(enabled=False)] + [dict(enabled=True, padding=padding,
        confidence_kind=kind, threshold=threshold, require_lexical_match=required)
        for padding, kind, threshold, required in itertools.product(
            c['paddings'], c['confidence_kinds'], c['thresholds'], c['require_lexical_match'])]
    trials = []
    for policy in policies:
        values = []; changes = 0
        for sid in validation:
            rows = []
            for base in baseline[sid]['rows']:
                raw = readings[sid][base['instance_id']]
                text, trace = choose_transcript(dict(text=raw['ctc_text']),
                    raw['rereadings'], lexicon, policy)
                changes += trace['changed']; rows.append(dict(base, raw_text=text, text=text))
            ignored = [row['ignored'] for row in rows]
            measured = counts(rows, truth[sid], ignored)
            if measured != recount(rows, truth[sid], ignored):
                raise ValueError('Independent reread policy counts differ')
            values.append(measured)
        trials.append(dict(policy=policy, counts=micro(values), changed_words=changes,
            deepsolo_images=len(validation), total_images=len(validation), deepsolo_percentage=100.))
    winner = max(enumerate(trials), key=lambda pair: (pair[1]['counts']['f1'],
        not pair[1]['policy']['enabled'], -pair[1]['changed_words'], -pair[0]))[1]
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    seal(root/'resolved_configuration.json', dict(config=c, config_sha256=sha(config_path),
        producer_sha256=sha(__file__), selection_ids=validation, mode='approximate',
        claim_label='approximate reproduction', test_used_for_selection=False))
    gain = winner['counts']['f1']-trials[0]['counts']['f1']
    seal(root/'frozen.json', dict(status='FROZEN', winner=winner, baseline=trials[0],
        validation_gain_f1_points=100*gain, promotion_eligible=winner['policy']['enabled'] and gain >= c['minimum_validation_gain'],
        test_used_for_selection=False, train_only=True))
    seal(root/'trials.json', trials); seal(root/'rereading_input_sha256.json', pins)
    print(__import__('json').dumps(dict(status='FROZEN', baseline_f1=trials[0]['counts']['f1'],
        chosen_f1=winner['counts']['f1'], policy=winner['policy'], gain_points=100*gain,
        deepsolo_images=len(validation), total_images=len(validation), deepsolo_percentage=100.)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.config)
