"""Prepare TRAIN-only fixed-emission cases and authenticated fallback targets."""

import argparse
from pathlib import Path
import shutil

from recognition.common_budget_pipeline import counts
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.fixed_expert_budget import fixed_trace
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.train_lightweight_word_router import teacher
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def prepare(config_path, run_name, smoke):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate configuration required')
    c = read(config_path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for p, digest in c['pinned_inputs'].items():
        if sha(p) != digest:
            raise ValueError('Changed fixed-expert input: '+p)
    root = Path(c['output_root'])/run_name; root.mkdir(parents=True, exist_ok=False)
    source = {}
    for package in c['snapshot_packages']:
        for p in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/p.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
            source[str(p.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    original = read(c['original_plan']); parts = original['partitions']['icdar13']
    fit = parts['fit']; val = sorted(parts['selection']+parts['calibration'])
    if smoke:
        fit = fit[:c['smoke_fit_images']]; val = val[:c['smoke_validation_images']]
    seal(root/'resolved_configuration.json', dict(config=c, config_sha256=sha(config_path),
        mode='approximate', claim_label='approximate reproduction', smoke=smoke,
        result_class='smoke-test result' if smoke else 'approximate reproduction',
        source_sha256=source, partitions=dict(fit=fit, validation=val, test=parts['test']),
        test_labels_opened=False, historical_test_exposure=True))
    return root


def vocabulary(profile):
    if profile['format'] == 'pairs_only_json':
        pairs = read(profile['pairs'])
        words = sorted(pairs)
    elif profile['format'] == 'vocabulary_and_json_pairs':
        words = Path(profile['vocabulary']).read_text().splitlines()
        pairs = read(profile['pairs'])
    else:
        raise ValueError('Explicit supported TRAIN vocabulary representation required')
    return DeepSoloLexicon(words, pairs, distance_exclusive=profile['distance_exclusive'])


def fallback_data(plan, root):
    c = plan['config']; original = read(c['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['train']}
    ids = plan['partitions']['fit']+plan['partitions']['validation']
    all_profiles = {}; pins = {}
    for name, profile in c['fallback_profiles'].items():
        lex = vocabulary(profile['train_lexicon'])
        entries = {}
        for sid in ids:
            value = teacher(profile, None, records[sid], lex)
            for row in value['rows']:
                if row.get('invalid_envelope') is not None:
                    row.update(raw_text='', text='')
            truth, _ = annotation(records[sid])
            value['counts'] = counts(value['rows'], truth, [r['ignored'] for r in value['rows']])
            entries[sid] = value; pins[value['source_path']] = value['source_sha256']
        all_profiles[name] = entries
    seal(root/'data/fallbacks.json', all_profiles); seal(root/'data/fallback_input_sha256.json', pins)


def prepare_view(plan, root, view):
    c = plan['config']; original = read(c['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['train']}
    fit = plan['partitions']['fit']; val = plan['partitions']['validation']
    seen = Path(c['seen_export']); seen_index = read(seen/'index.json')
    if read(seen/'completion.json')['status'] != 'COMPLETE':
        raise ValueError('Complete full-fitting expert export required')
    sources = {s: seen_index[s] for s in fit+val}; pins = {str(seen/'index.json'): sha(seen/'index.json')}
    expected_weights = {s: read(c['expert_frozen'])['winner']['checkpoint_sha256'] for s in fit+val}
    if view == 'oof':
        folds = read(c['folds_manifest'])['folds']; covered = []
        for f in folds:
            directory = Path(c['fold_root'])/('fold_'+str(f['fold']))/'heldout'
            complete = read(directory/'completion.json')
            if complete['status'] != 'COMPLETE' or not complete['image_excluded'] or complete['test_used']:
                raise ValueError('Complete image-excluded fold export required')
            index = read(directory/'index.json')
            exported = read(directory/'resolved_configuration.json')
            if set(index) != set(f['heldout_ids']):
                raise ValueError('Fold export population differs')
            pins[str(directory/'index.json')] = sha(directory/'index.json')
            pins[str(directory/'completion.json')] = sha(directory/'completion.json')
            pins[str(directory/'resolved_configuration.json')] = sha(directory/'resolved_configuration.json')
            for sid in f['heldout_ids']:
                if sid in fit:
                    sources[sid] = index[sid]; covered.append(sid)
                    expected_weights[sid] = exported['checkpoint_sha256']
        if sorted(covered) != sorted(fit):
            raise ValueError('Incomplete or repeated OOF fitting population')
    elif view != 'seen':
        raise ValueError('Unknown explicit fixed-expert source view')
    observations = {}; labels = {}; keys = {}
    for sid in fit+val:
        record = records[sid]; entry = sources[sid]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed real expert sample')
        sample = read(entry['path'])
        if sample['ground_truth_used'] or sample['image_sha256'] != record['image_sha256']:
            raise ValueError('Invalid expert/RGB binding')
        if (sample['checkpoint_sha256'] != expected_weights[sid] or
                sample['frozen_expert_sha256'] != sha(c['expert_frozen'])):
            raise ValueError('Changed fixed-expert checkpoint or emission policy')
        if view == 'oof' and sid in fit and not sample['image_excluded']:
            raise ValueError('FIT image was not excluded from expert fitting')
        raw, rows = sample['raw_rows'], sample['rows']; trace = fixed_trace(raw, rows)
        gt, valid = annotation(record)
        ignored = [p['ignored'] for p in predictions(raw, record, valid)]
        observations[sid] = dict(raw=raw, rows=rows, trace=trace,
            image_wh=record['image_wh'], map_summary=None)
        labels[sid] = dict(ground_truth=gt, raw_ignored=ignored,
            light=counts(rows, gt, [ignored[int(r['source_row'])] for r in rows]))
        keys[sid] = record['rgb_sha256']; pins[entry['path']] = entry['sha256']
    folder = root/'data'/view
    seal(folder/'observations.json', observations); seal(folder/'labels.json', labels)
    seal(folder/'input_sha256.json', pins)
    seal(folder/'case.json', dict(name=view, fit_ids=fit, validation_ids=val,
        keys=keys, expert_image_excluded=view == 'oof', map_available=False, map_extra_models=0,
        fallback_images=c['fallback_images'], total_images=c['total_images'],
        fallback_profiles=list(c['fallback_profiles']), ground_truth_feature_use=False))
    print('PREPARED FIXED EXPERT', view, len(fit), len(val), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path)
    p.add_argument('--run-root', type=Path); p.add_argument('--run-name', default='full')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--views', nargs='+')
    p.add_argument('--stage', choices=['prepare', 'data', 'all'], default='all')
    args = p.parse_args(); root = args.run_root or prepare(args.config, args.run_name, args.smoke)
    plan = read(root/'resolved_configuration.json')
    if plan['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    if args.stage != 'prepare':
        if not (root/'data/fallbacks.json').exists():
            fallback_data(plan, root)
        for view in args.views or plan['config']['required_views']:
            prepare_view(plan, root, view)


if __name__ == '__main__':
    main()
