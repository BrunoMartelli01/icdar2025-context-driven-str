"""TRAIN-validation-only shared emission/lexicon control for the trained U-Nets."""

import argparse
import itertools
from pathlib import Path

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import predictions
from segmentation.uniform_unet_fusion import annotation, word_counts, aggregate_counts, file_hash
from recognition.uniform_lexical_risk import lexicon, prepare_words


def select(config_path, dataset):
    c = read(config_path)
    if c['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate lexical study profile required')
    for path, digest in c['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed qualified lexical control input: '+path)
    source = Path(c['training_root'])/dataset
    plan = read(source/'resolved_configuration.json')
    index = read(source/'export_train/index.json')
    if read(source/'independent_training_audit.json')['status'] != 'PASS':
        raise ValueError('Audited shared U-Net required')
    out = Path(c['output_root'])/dataset
    out.mkdir(parents=True, exist_ok=False)
    dictionary = lexicon(c['lexicons'][dataset], 'train')
    val = plan['partitions']['selection']
    lookup = {r['sample_id']: r for r in plan['train_records']}
    observations, truth, flags = {}, {}, {}
    for sid in val:
        entry = index[sid]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Changed qualified U-Net reading')
        observations[sid] = read(entry['path'])
        if observations[sid]['ground_truth_used']:
            raise ValueError('Expert generation consumed annotations')
        truth[sid], valid = annotation(lookup[sid])
        flags[sid] = [p['ignored'] for p in predictions(observations[sid]['raw_rows'], lookup[sid], valid)]
    grid = []
    for cut in c['sequence_thresholds']:
        grid.append(dict(enabled=False, confidence_cap=0., protect_numeric=True,
                         unmatched_policy='keep_unmatched', minimum_sequence_probability=cut))
        if dictionary is not None:
            grid += [dict(enabled=True, confidence_cap=cap, protect_numeric=True,
                unmatched_policy=unmatched, minimum_sequence_probability=cut)
                for cap, unmatched in itertools.product(c['confidence_caps'], c['unmatched_policies'])]
    trials = []
    for policy in grid:
        counts, edits = [], 0
        for sid in val:
            rows, trace = prepare_words(observations[sid]['raw_rows'], dictionary, policy)
            counts.append(word_counts(rows, truth[sid], [flags[sid][int(r['source_row'])] for r in rows]))
            edits += sum(t['changed'] for t in trace)
        trials.append(dict(policy=policy, counts=aggregate_counts(counts), edited_words=edits,
                           deepsolo_images=0, total_images=len(val), deepsolo_percentage=0.))
    winner = max(enumerate(trials), key=lambda p: (p[1]['counts']['f1'], -p[0]))[1]
    write(out/'lexical_trials.json', trials)
    write(out/'lexical_selection_frozen.json', dict(status='FROZEN', mode='approximate', winner=winner,
        dataset=dataset, selection_ids=val, test_used_for_selection=False,
        historical_test_exposure=True, config_sha256=file_hash(config_path), source_sha256=file_hash(__file__),
        source_training_plan_sha256=file_hash(source/'resolved_configuration.json'),
        source_export_index_sha256=file_hash(source/'export_train/index.json')))
    print('FROZEN LEXICAL CONTROL', dataset, winner, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13','icdar2015','totaltext'], required=True)
    a = p.parse_args(); select(a.config, a.dataset)
