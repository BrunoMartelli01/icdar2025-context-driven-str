"""Recount TRAIN-only Generic DeepSolo preparation from authenticated raw envelopes."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.deepsolo_word_f1 import recognition_confidence
from recognition.clip_router_evaluation import deepsolo_predictions
from scripts.icdar_budget_data import annotation, key, read, sha
from scripts.run_yolo_f1_quality_study import seal


def counts(predicted, truth):
    normalize = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
    care = [p for p in predicted if not p['ignored']]
    a = Counter(normalize(p['text']) for p in care if normalize(p['text']))
    b = Counter(normalize(g['text']) for g in truth if normalize(g['text']))
    tp = sum((a & b).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp,
        detected_words=len(care), gt_words=len(truth),
        all_outputs=len(predicted), ignored_outputs=len(predicted)-len(care))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['study_root'])
    plan = read(Path(config['original_router_root']) / 'resolved_configuration.json')
    samples = read(root / 'teacher/icdar13/counts.json')
    reference = read(config['teacher_reference'])
    provenance = read(root / 'train_generic_provenance.json')
    assert provenance['test_transcriptions_read'] == 0
    prepared_config = root / 'resolved_configuration.json'
    assert provenance['config_sha256'] == sha(prepared_config)
    original_config = read(prepared_config)
    assert all(config[name] == value for name, value in original_config.items()
               if name != 'pinned_inputs')
    assert sha(config['train_generic_base']) == provenance['common_base_sha256']
    base = {word.strip().upper(): word.strip() for word in
        Path(config['train_generic_base']).read_text().splitlines()}
    train_words = {}
    for record in plan['records']['icdar13']['train']:
        gt, _ = annotation(record)
        for word in gt:
            train_words.setdefault(word['text'].strip().upper(), word['text'].strip())
    base.update(train_words)
    assert len(base) == provenance['words']
    assert len(train_words) == provenance['train_word_types']
    assert Path(root / 'train_generic_vocabulary.txt').read_text().splitlines() == sorted(base)
    assert read(root / 'train_generic_pairs.json') == base
    assert samples['train_vocabulary_sha256'] == sha(root / 'train_generic_vocabulary.txt')
    assert samples['train_pairs_sha256'] == sha(root / 'train_generic_pairs.json')
    dictionary = DeepSoloLexicon(sorted(base), base, distance_exclusive=1.5)
    train_ids = {r['sample_id'] for r in plan['records']['icdar13']['train']}
    test_ids = {r['sample_id'] for r in plan['records']['icdar13']['test']}
    assert set(samples['samples']) == train_ids | test_ids
    for sid in test_ids:
        assert samples['samples'][sid] == reference['samples'][sid]
    for record in plan['records']['icdar13']['train']:
        sid = record['sample_id']
        path = Path(config['teacher_raw_train']) / (key(sid) + '.json')
        raw = read(path)
        body = {name: value for name, value in raw.items() if name != 'payload_sha256'}
        assert raw['payload_sha256'] == hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()
        assert raw['response']['success'] and not raw['ground_truth_used']
        assert raw['contract']['image_sha256'] == record['image_sha256']
        assert raw['contract']['sample_id'] == sid
        assert samples['input_sha256'][str(path)] == sha(path)
        neural = config['teacher_neural']
        rows = [dict(instance_id=d['source_index'], text=d['text'], score=d['score'])
            for d in raw['response']['detections']
            if d['score'] >= neural['score_threshold'] and
            recognition_confidence(d['raw']['recognition_class_probabilities'])
            >= neural['recognition_threshold']]
        rows.sort(key=lambda p: (-p['score'], p['instance_id']))
        selected, trace = dictionary.correct(rows,
            unmatched_policy=config['train_teacher_unmatched_policy'])
        by_index = {d['source_index']: d for d in raw['response']['detections']}
        detections = [dict(by_index[p['instance_id']], text=p['text']) for p in selected]
        truth, validity = annotation(record)
        predicted = deepsolo_predictions(dict(detections=detections),
            record['image_wh'], record['orientation'], validity, neural['score_threshold'])
        for p in predicted:
            if p['invalid_envelope'] is not None:
                p['text'] = ''
        expected = counts(predicted, truth)
        entry = samples['samples'][sid]
        assert entry['counts'] == expected
        assert entry['image_sha256'] == record['image_sha256']
        assert entry['evidence_sha256'] == sha(entry['evidence_path'])
        evidence = read(entry['evidence_path'])
        assert evidence['predictions'] == predicted and evidence['ground_truth'] == truth
        assert evidence['counts'] == expected and evidence['decisions'] == trace
    seal(root / 'teacher_independent_audit.json', dict(status='PASS',
        raw_envelopes=len(train_ids), unchanged_test_entries=len(test_ids),
        train_transcription_sources=len(train_ids), test_transcriptions_used_for_train=0,
        teacher_sha256=sha(root / 'teacher/icdar13/counts.json'),
        code_sha256=sha(__file__)))
    print('PASS', len(train_ids), 'TRAIN envelopes,', len(test_ids),
          'unchanged TEST entries', flush=True)


if __name__ == '__main__':
    main()
