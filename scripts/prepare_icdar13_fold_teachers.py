"""Prepare and independently audit TRAIN-only Generic teachers for future OOF gates."""

import argparse
from collections import Counter
from pathlib import Path
import shutil

from recognition.clip_router_evaluation import deepsolo_predictions
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.fold_safe_teacher import canonical_vocabulary, checked_response, correct_detections
from recognition.icdar13_hypothesis_tests import crossfit_folds
from recognition.icdar_budget_router import lexical_counts
from recognition.f1_quality_study import aggregate
from scripts.icdar_budget_data import read, sha, key, annotation
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def prepare(path):
    if path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(path)
    assert c['mode'] == 'approximate' and c['claim_label'] == 'approximate reproduction'
    for name, digest in c['pinned_inputs'].items():
        assert sha(name) == digest, name
    training = read(c['training_plan'])
    profile, protocol = read(c['fallback_profile']), read(c['raw_protocol'])
    assert profile['checkpoint_sha256'] == protocol['source_identity']['checkpoint_sha256']
    assert profile['native_resize'] == protocol['resize']
    assert protocol['inference_floor'] <= profile['neural']['score_threshold']
    assert c['unmatched_policy'] == profile['unmatched_policy']
    records = {r['sample_id']: r for r in training['train_records']
               if r['sample_id'].startswith('icdar13:train:')}
    fit = training['partitions']['fit']
    folds = crossfit_folds([records[sid] for sid in fit], c['folds'], c['fold_seed'])
    scopes = [dict(name='full_fit', vocabulary_label_ids=fit,
                   prediction_ids=sorted(records))]
    for index, heldout in enumerate(folds):
        scopes.append(dict(name='fold_'+str(index),
            vocabulary_label_ids=sorted(set(fit)-set(heldout)), prediction_ids=heldout))
    test_rgb = {r['rgb_sha256'] for r in training['test_records']}
    assert not test_rgb & {r['rgb_sha256'] for r in records.values()}
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    source = {}
    for folder in ('recognition', 'scripts', 'segmentation', 'models', 'localize',
                   'deepsolo', 'caption', 'scoring', 'pipeline', 'research'):
        for file in (REPO/folder).rglob('*.py'):
            relative = file.relative_to(REPO)
            dest = root/'code_snapshot'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
            source[str(relative)] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction', config=c, config_sha256=sha(path),
        partitions=training['partitions'], records=records, folds=folds, scopes=scopes,
        source_sha256=source, test_used_for_selection=False,
        scope='TRAIN supervision preparation only; no light decoder or router trained',
        pretraining_exposure='The fixed DeepSolo teacher and transferred encoder already used IC13 TRAIN upstream'))
    print('PREPARED', root, 'scopes', len(scopes), flush=True)


def dictionary(plan, scope):
    c, records = plan['config'], plan['records']
    labels = {}
    for sid in scope['vocabulary_label_ids']:
        truth, _ = annotation(records[sid])
        labels[sid] = [word['text'] for word in truth]
    canonical = canonical_vocabulary(Path(c['common_vocabulary']).read_text().splitlines(),
                                      scope['vocabulary_label_ids'], labels)
    return canonical, DeepSoloLexicon(sorted(canonical), canonical,
                                     distance_exclusive=c['distance_exclusive'])


def independent(predictions, truth):
    normal = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
    care = [p for p in predictions if not p['ignored']]
    observed = Counter(normal(p['text']) for p in care
                       if normal(p['text']) and p['invalid_envelope'] is None)
    target = Counter(normal(g['text']) for g in truth if normal(g['text']))
    tp = sum((observed & target).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp, detected_words=len(care),
                gt_words=len(truth), all_outputs=len(predictions),
                ignored_outputs=len(predictions)-len(care))


def build(plan, root):
    c = plan['config']
    profile, protocol = read(c['fallback_profile']), read(c['raw_protocol'])
    summaries = {}
    for scope in plan['scopes']:
        directory = root/scope['name']
        canonical, lexicon = dictionary(plan, scope)
        directory.mkdir(parents=True, exist_ok=True)
        words = directory/'vocabulary.txt'
        expected = '\n'.join(sorted(canonical))+'\n'
        if words.exists():
            assert words.read_text() == expected
        else:
            words.write_text(expected)
        seal(directory/'pairs.json', canonical)
        samples = {}
        for sid in scope['prediction_ids']:
            record = plan['records'][sid]
            assert sha(record['image_path']) == record['image_sha256']
            raw_path = Path(c['raw_train'])/(key(sid)+'.json')
            raw = checked_response(read(raw_path), protocol['binding'], sid, record['image_sha256'])
            detections, decisions = correct_detections(raw, lexicon, profile['neural'], c['unmatched_policy'])
            # Labels are read after complete prediction, solely to form supervised targets.
            truth, validity = annotation(record)
            pp = deepsolo_predictions(dict(detections=detections), record['image_wh'],
                record['orientation'], validity, profile['neural']['score_threshold'])
            for p in pp:
                if p['invalid_envelope'] is not None:
                    p['text'] = ''
            counts = lexical_counts(pp, truth)
            assert independent(pp, truth) == counts
            dest = directory/'samples'/(key(sid)+'.json')
            seal(dest, dict(sample_id=sid, predictions=pp, ground_truth=truth,
                counts=counts, decisions=decisions, raw_path=str(raw_path), raw_sha256=sha(raw_path),
                image_sha256=record['image_sha256'], vocabulary_sha256=sha(words),
                pairs_sha256=sha(directory/'pairs.json'),
                checkpoint_sha256=profile['checkpoint_sha256'],
                ground_truth_used_for_prediction=False))
            samples[sid] = dict(counts=counts, evidence_path=str(dest), evidence_sha256=sha(dest),
                                image_sha256=record['image_sha256'])
        total = aggregate([row['counts'] for row in samples.values()])
        summaries[scope['name']] = dict(samples=samples, counts=total,
            vocabulary_label_ids=scope['vocabulary_label_ids'], deepsolo_images=len(samples),
            total_images=len(samples), deepsolo_percentage=100.,
            inference_scope='Cached standalone teacher supervision, not a router or new neural calls')
        seal(directory/'counts.json', summaries[scope['name']])
        print('TEACHER', scope['name'], len(samples), 'images', flush=True)
    seal(root/'results.json', dict(status='COMPLETE', mode='approximate',
        claim_label='approximate reproduction', scopes=summaries,
        actual_new_deepsolo_recognizer_calls=0, test_predictions_used=0,
        test_lexicon_files_read=[], plan_sha256=sha(root/'resolved_configuration.json')))


def audit(plan, root, output):
    result = read(root/'results.json')
    assert result['plan_sha256'] == sha(root/'resolved_configuration.json')
    c, records = plan['config'], plan['records']
    profile, protocol = read(c['fallback_profile']), read(c['raw_protocol'])
    counters = Counter()
    assert sorted(sid for fold in plan['folds'] for sid in fold) == sorted(plan['partitions']['fit'])
    groups = [{records[sid]['rgb_sha256'] for sid in fold} for fold in plan['folds']]
    assert not any(a & b for i, a in enumerate(groups) for b in groups[i+1:])
    for scope in plan['scopes']:
        name = scope['name']
        allowed, held = set(scope['vocabulary_label_ids']), set(scope['prediction_ids'])
        if name != 'full_fit':
            assert allowed == set(plan['partitions']['fit'])-held and not allowed & held
        assert allowed <= set(plan['partitions']['fit'])
        assert not allowed & set(plan['partitions']['selection']+plan['partitions']['calibration'])
        directory = root/name
        # Independent dictionary reconstruction from exactly the declared source IDs.
        canonical = {w.strip().upper(): w.strip()
            for w in Path(c['common_vocabulary']).read_text().splitlines()}
        for sid in scope['vocabulary_label_ids']:
            truth, _ = annotation(records[sid])
            for word in truth:
                canonical[word['text'].strip().upper()] = word['text'].strip()
        assert read(directory/'pairs.json') == canonical
        assert (directory/'vocabulary.txt').read_text().splitlines() == sorted(canonical)
        lexicon = DeepSoloLexicon(sorted(canonical), canonical, distance_exclusive=c['distance_exclusive'])
        saved = result['scopes'][name]
        total = Counter()
        assert set(saved['samples']) == held
        for sid in scope['prediction_ids']:
            record = records[sid]
            ref = saved['samples'][sid]
            assert sha(ref['evidence_path']) == ref['evidence_sha256']
            data = read(ref['evidence_path'])
            assert sha(data['raw_path']) == data['raw_sha256']
            assert data['vocabulary_sha256'] == sha(directory/'vocabulary.txt')
            assert data['pairs_sha256'] == sha(directory/'pairs.json')
            assert not data['ground_truth_used_for_prediction']
            raw = checked_response(read(data['raw_path']), protocol['binding'], sid, record['image_sha256'])
            detections, decisions = correct_detections(raw, lexicon, profile['neural'], c['unmatched_policy'])
            truth, valid = annotation(record)
            replay = deepsolo_predictions(dict(detections=detections), record['image_wh'],
                record['orientation'], valid, profile['neural']['score_threshold'])
            for p in replay:
                if p['invalid_envelope'] is not None:
                    p['text'] = ''
            assert replay == data['predictions'] and decisions == data['decisions']
            assert truth == data['ground_truth']
            value = independent(replay, truth)
            assert value == data['counts'] == ref['counts']
            total.update(value)
            counters['authenticated_prediction_replays'] += 1
        for field in ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs'):
            assert total[field] == saved['counts'][field]
        assert saved['counts']['f1'] == 2*total['tp']/(total['detected_words']+total['gt_words'])
        assert saved['deepsolo_images'] == saved['total_images'] == len(held)
        assert saved['deepsolo_percentage'] == 100.
        counters['scopes'] += 1
    seal(output, dict(status='PASS', checks=dict(counters), results_sha256=sha(root/'results.json'),
        actual_new_deepsolo_recognizer_calls=0, test_predictions_used=0, test_lexicon_files_read=[],
        code_sha256=sha(__file__), scope='Independent FIT/fold dictionary construction, '
        'authenticated raw binding, complete pixel-envelope/correction replay and Counter counts'))
    print('PASS', dict(counters), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'build', 'audit'], required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.config)
        return
    root = args.run_root
    plan = read(root/'resolved_configuration.json')
    for name, digest in plan['config']['pinned_inputs'].items():
        assert sha(name) == digest, name
    for name, digest in plan['source_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest, name
    if args.stage == 'build':
        build(plan, root)
    else:
        if args.output is None:
            raise ValueError('Explicit new audit output required')
        audit(plan, root, args.output)


if __name__ == '__main__':
    main()
