"""CPU-only teacher boundaries; no validation/test dictionary is accepted."""

import hashlib
import json

from recognition.deepsolo_word_f1 import recognition_confidence


def canonical_vocabulary(base_words, fitting_ids, transcripts):
    if set(transcripts) != set(fitting_ids) or len(set(fitting_ids)) != len(fitting_ids):
        raise ValueError('Transcript keys must equal the declared fitting population')
    canonical = {w.strip().upper(): w.strip() for w in base_words}
    for sid in fitting_ids:
        for word in transcripts[sid]:
            canonical[word.strip().upper()] = word.strip()
    if not canonical or '' in canonical:
        raise ValueError('Nonempty complete vocabulary entries required')
    return canonical


def checked_response(envelope, binding, sample_id, image_sha256):
    body = {k: v for k, v in envelope.items() if k != 'payload_sha256'}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if envelope['payload_sha256'] != expected:
        raise ValueError('Changed raw teacher payload')
    contract = envelope['contract']
    if any(contract[k] != v for k, v in binding.items()):
        raise ValueError('Changed raw teacher protocol binding')
    if contract['sample_id'] != sample_id or contract['image_sha256'] != image_sha256:
        raise ValueError('Raw teacher image identity mismatch')
    if not envelope['response']['success']:
        raise RuntimeError('Raw teacher inference failed')
    return envelope['response']


def correct_detections(response, lexicon, neural, unmatched_policy):
    raw, rows = {}, []
    for detection in response['detections']:
        index = detection['source_index']
        if index in raw:
            raise ValueError('Duplicate teacher source index')
        raw[index] = detection
        if (detection['score'] >= neural['score_threshold'] and
                recognition_confidence(detection['raw']['recognition_class_probabilities']) >=
                neural['recognition_threshold']):
            rows.append(dict(instance_id=index, text=detection['text'], score=detection['score']))
    rows.sort(key=lambda row: (-row['score'], row['instance_id']))
    corrected, decisions = lexicon.correct(rows, unmatched_policy=unmatched_policy)
    return [dict(raw[row['instance_id']], text=row['text']) for row in corrected], decisions
