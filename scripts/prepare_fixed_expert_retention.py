"""Adapt authenticated TT/IC15 experts to the same fixed-emission gate trainer."""

import argparse
import copy
from pathlib import Path
import shutil

from recognition.common_budget_pipeline import counts
from recognition.fixed_expert_budget import fixed_trace
from scripts.icdar_budget_data import read, sha, key
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def observe(raw, emitted, trace, image_wh, maps):
    # Some authenticated TT U-Net observations preserve only the previously
    # qualified balanced confidence. Mark that proxy explicitly; do not invent
    # token probabilities or claim an unavailable native sequence score.
    def confidence_schema(row):
        row = dict(row)
        if 'sequence_confidence_native' not in row:
            row.update(sequence_confidence_native=row['confidence'],
                sequence_confidence_origin='explicit proxy from existing balanced confidence')
        return row
    raw = [confidence_schema(r) for r in raw]; emitted = [confidence_schema(r) for r in emitted]
    return dict(raw=raw, rows=emitted, trace=fixed_trace(raw, emitted, trace),
                image_wh=image_wh, map_summary=maps)


def prepare(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed common retention input: '+file)
    source = Path(c['source_root']); q = read(source/'deployment_qualification.json')
    if q['status'] != 'PASS' or sha(source/q['audit_path']) != q['audit_sha256']:
        raise ValueError('Independently qualified original common source required')
    for file, digest in q['artifact_sha256'].items():
        if sha(source/file) != digest:
            raise ValueError('Changed qualified original common artifact')
    cases = read(source/'cases.json'); template = read(c['template_gate_plan'])['config']
    for name in c['cases']:
        case = cases[name]; parts = case['partitions']; dataset = case['dataset']
        fit = parts['fit']; val = sorted(parts['selection']+parts['calibration']); ids = fit+val
        obs = read(source/'data'/(name+'__observations.json'))
        labels = read(source/'data'/(name+'__train_labels.json'))
        maps = read(source/'data'/(name+'__maps.json'))
        emitted = read(source/'models'/name/'observable_outputs.json')
        if dataset == 'totaltext':
            native = read(Path(c['tt_native_root'])/'train/branches.json')['records']
        else:
            native = None
        observations = {}; targets = {}; fallback = {}; pins = {}
        for sid in ids:
            a = obs[sid]; labels_for_image = labels[sid]
            observations[sid] = observe(a['rows'], emitted['corrected_rows'][sid],
                emitted['traces'][sid], a['image_wh'], maps[sid]['maps'])
            targets[sid] = dict(ground_truth=labels_for_image['ground_truth'],
                raw_ignored=labels_for_image['ignored'], light=counts(emitted['corrected_rows'][sid],
                    labels_for_image['ground_truth'], [labels_for_image['ignored'][int(r['source_row'])]
                        for r in emitted['corrected_rows'][sid]]))
            if native is not None:
                rows = native[sid]['deepsolo']
            else:
                file = Path(c['ic15_teacher_root'])/'per_image'/(key(sid)+'.json')
                value = read(file); pins[str(file)] = sha(file); rows = value['predictions']
                if value['raw_sha256'] != sha(value['raw_path']):
                    raise ValueError('Changed IC15 source teacher envelope')
                pins[value['raw_path']] = value['raw_sha256']
            rows = [dict(r, raw_text='' if r.get('invalid_envelope') is not None else r['text'],
                word_source='deepsolo') for r in rows]
            measured = counts(rows, labels_for_image['ground_truth'], [r['ignored'] for r in rows])
            if measured != labels_for_image['teacher']:
                raise ValueError('Preserved fallback/count protocol differs: '+sid)
            fallback[sid] = dict(rows=rows, counts=measured)
        root = Path(c['output_root'])/name; root.mkdir(parents=True, exist_ok=False)
        settings = copy.deepcopy(template)
        for field in ('expert_frozen', 'seen_export', 'folds_manifest', 'fold_root', 'original_plan'):
            settings.pop(field, None)
        settings.update(required_views=['seen'], output_root=str(root),
            fallback_images=c['budgets'][dataset]['fallback_images'],
            total_images=c['budgets'][dataset]['images'], minimum_f1=c['minimum_f1'][name],
            fallback_profiles=dict(native_benchmark=dict(primary_eligible=True, extra_reader_crops=0)),
            pinned_inputs=dict(c['pinned_inputs']))
        settings['scientific_choices'] = c['scientific_choices']
        code = {}
        for package in settings['snapshot_packages']:
            for file in (REPO/package).rglob('*.py'):
                destination = root/'code_snapshot'/file.relative_to(REPO)
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, destination)
                code[str(file.relative_to(REPO))] = sha(destination)
        shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
        seal(root/'resolved_configuration.json', dict(config=settings, mode='approximate',
            claim_label='approximate reproduction', result_class='approximate reproduction', smoke=False,
            config_sha256=sha(path), source_sha256=code, source_kind='qualified_common_outputs',
            source_root=str(source), source_case=name, partitions=dict(fit=fit, validation=val, test=parts['test']),
            test_labels_opened=False, historical_test_exposure=True))
        seal(root/'data/seen/observations.json', observations); seal(root/'data/seen/labels.json', targets)
        seal(root/'data/seen/input_sha256.json', pins)
        seal(root/'data/seen/case.json', dict(name='seen', fit_ids=fit, validation_ids=val,
            keys={s: case['tie_keys'][s] for s in ids}, expert_image_excluded=False,
            map_available=True, map_extra_models=int(case['expert'] == 'yolo'),
            fallback_images=settings['fallback_images'], total_images=settings['total_images'],
            fallback_profiles=['native_benchmark'], ground_truth_feature_use=False))
        seal(root/'data/fallbacks.json', dict(native_benchmark=fallback)); seal(root/'data/fallback_input_sha256.json', pins)
        test = {s: observe(obs[s]['rows'], emitted['corrected_rows'][s], emitted['traces'][s],
            obs[s]['image_wh'], maps[s]['maps']) for s in parts['test']}
        seal(root/'test_observations.json', test)
        print('PREPARED RETENTION', name, len(fit), len(val), len(test), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); prepare(a.config)
