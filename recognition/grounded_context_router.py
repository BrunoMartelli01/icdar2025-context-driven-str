"""GT-blind region inventory, local evidence and frozen visual router features."""
from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein


def box_area(box):
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])


def intersection(a, b):
    return box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def normalized_text(text):
    # Preserve punctuation, digits and real repetitions; normalize case/whitespace only.
    return ' '.join(text.casefold().split())


def similarity(a, b):
    a, b = normalized_text(a), normalized_text(b)
    return 1-Levenshtein.distance(a, b)/max(len(a), len(b), 1)


def generation_incomplete(output, maximum_new_tokens):
    """A forced EOS at the generation limit does not establish natural completion."""
    if maximum_new_tokens < 1 or output['generated_tokens'] < 0:
        raise ValueError('Explicit generation lengths required')
    return not output['ended_with_eos'] or output['generated_tokens'] >= maximum_new_tokens


def validate_rows(rows, image_wh):
    w, h = image_wh
    if min(w, h) <= 0 or len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid image or duplicate candidate identity')
    for row in rows:
        b = row['bbox']
        if (len(b) != 4 or any(type(v) is not int for v in b)
                or not 0 <= b[0] < b[2] <= w or not 0 <= b[1] < b[3] <= h
                or not isinstance(row['raw_text'], str)):
            raise ValueError('Original exclusive integer boxes and raw strings required')


def inventory_features(ocr, objects, rows, image_wh, settings):
    """Independent OCR quadrilaterals are fallible predictions, never annotations.

    Native-pixel polygon coverage allows many YOLO words inside one OCR line.
    The lexical comparison uses that line's spatially associated YOLO strings.
    """
    validate_rows(rows, image_wh)
    w, h = image_wh
    covered = np.zeros((h, w), dtype=bool)
    for row in rows:
        x1, y1, x2, y2 = row['bbox']; covered[y1:y2, x1:x2] = True
    if len(ocr['quad_boxes']) != len(ocr['labels']) or len(objects['bboxes']) != len(objects['labels']):
        raise ValueError('Independent region/label counts differ')
    evidence = []; invalid = 0
    for quad, text in zip(ocr['quad_boxes'], ocr['labels']):
        if len(quad) != 8 or not np.isfinite(quad).all() or not isinstance(text, str):
            raise ValueError('Malformed independent OCR output')
        mask = Image.new('1', (w, h)); ImageDraw.Draw(mask).polygon(list(zip(quad[::2], quad[1::2])), fill=1)
        region = np.asarray(mask, dtype=bool); area = int(region.sum())
        if area == 0:
            invalid += 1
            evidence.append({'text': text, 'quad': quad, 'valid': False, 'reason': 'no_in_image_pixels'})
            continue
        support = []
        for row in rows:
            x1, y1, x2, y2 = row['bbox']
            fraction = float(region[y1:y2, x1:x2].sum()/box_area(row['bbox']))
            if fraction >= settings['word_in_region_fraction']: support.append(row)
        support.sort(key=lambda r: (r['bbox'][0], r['bbox'][1], r['candidate_id']))
        reading = ' '.join(r['raw_text'] for r in support)
        coverage = float((region & covered).sum()/area)
        evidence.append({'text': text, 'quad': quad, 'valid': True, 'pixels': area,
            'coverage': coverage, 'word_count': len(text.split()), 'reading': reading,
            'similarity': similarity(text, reading), 'candidate_ids': [r['candidate_id'] for r in support]})
    valid = [r for r in evidence if r['valid']]
    n = len(valid); total_words = sum(r['word_count'] for r in valid)
    absent = [r for r in valid if r['coverage'] < settings['region_covered_fraction']]
    associated = {cid for r in valid for cid in r['candidate_ids']}
    object_evidence = []
    for box, label in zip(objects['bboxes'], objects['labels']):
        if len(box) != 4 or not np.isfinite(box).all() or not isinstance(label, str):
            raise ValueError('Malformed independent object output')
        if not any(re.search(r'\b'+re.escape(word)+r'\b', label.casefold()) for word in settings['text_object_terms']):
            continue
        b = [max(0, math.floor(box[0])), max(0, math.floor(box[1])), min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))]
        present = [r['candidate_id'] for r in rows if intersection(b, r['bbox'])/box_area(r['bbox']) >= settings['word_in_region_fraction']]
        object_evidence.append({'label': label, 'bbox': box, 'candidate_ids': present,
            'valid': box_area(b) > 0})
    values = {
        'inventory_log_regions': math.log1p(n), 'inventory_log_words': math.log1p(total_words),
        'inventory_log_missing_regions': math.log1p(len(absent)),
        'inventory_uncovered_fraction': sum(1-r['coverage'] for r in valid)/max(n, 1),
        'inventory_missing_word_proxy': math.log1p(sum(r['word_count']*(1-r['coverage']) for r in valid)),
        'inventory_lexical_disagreement': sum(1-r['similarity'] for r in valid)/max(n, 1),
        'inventory_lexical_minimum': min((r['similarity'] for r in valid), default=0.),
        'inventory_unsupported_yolo_fraction': sum(r['candidate_id'] not in associated for r in rows)/max(len(rows), 1),
        'inventory_empty': float(n == 0), 'inventory_invalid_regions': float(invalid),
        'inventory_log_text_objects': math.log1p(len(object_evidence)),
        'inventory_log_empty_text_objects': math.log1p(sum(r['valid'] and not r['candidate_ids'] for r in object_evidence)),
        'inventory_word_count_delta': (len(rows)-total_words)/max(len(rows), total_words, 1),
    }
    return values, {'ocr_regions': evidence, 'text_objects': object_evidence, 'ground_truth_used': False}


def line_members(rows, index, settings):
    """Bounded local context, not an asserted true reading order for curved text."""
    target = rows[index]['bbox']; th = target[3]-target[1]
    ranked = []
    for i, row in enumerate(rows):
        box = row['bbox']; bh = box[3]-box[1]
        vertical = max(0, min(target[3], box[3])-max(target[1], box[1]))/min(th, bh)
        gap = max(0, max(target[0], box[0])-min(target[2], box[2]))/max(th, bh)
        if i == index or (vertical >= settings['line_vertical_overlap'] and gap <= settings['line_max_gap_heights']
                          and min(th, bh)/max(th, bh) >= settings['line_min_height_ratio']):
            ranked.append((gap, abs(i-index), i))
    keep = [i for _, _, i in sorted(ranked)[:settings['maximum_line_words']]]
    return sorted(keep, key=lambda i: (rows[i]['bbox'][0], rows[i]['bbox'][1], rows[i]['candidate_id']))


def verification_jobs(rows, word_evidence, inventory, image_wh, settings):
    validate_rows(rows, image_wh)
    lookup = {r['candidate_id']: r for r in word_evidence}
    if set(lookup) != {r['candidate_id'] for r in rows}: raise ValueError('Word evidence identity differs')
    def risk(i):
        cid = rows[i]['candidate_id']
        support = [r['similarity'] for r in inventory['ocr_regions'] if r['valid'] and cid in r['candidate_ids']]
        agreement = max(support, default=0.)
        return (lookup[cid]['visual']*agreement, lookup[cid]['visual'], cid)
    selected = sorted(range(len(rows)), key=risk)[:settings['maximum_verified_words']]
    jobs = []
    w, h = image_wh
    for i in selected:
        row = rows[i]
        if not row['raw_text'].strip(): continue
        members = line_members(rows, i, settings); boxes = [rows[j]['bbox'] for j in members]
        pad = settings['line_padding_heights']*(row['bbox'][3]-row['bbox'][1])
        outer = [max(0, math.floor(min(b[0] for b in boxes)-pad)), max(0, math.floor(min(b[1] for b in boxes)-pad)),
                 min(w, math.ceil(max(b[2] for b in boxes)+pad)), min(h, math.ceil(max(b[3] for b in boxes)+pad))]
        text = row['raw_text']
        candidates = [text]
        # A visual foil, not an output candidate or GT-derived spelling.
        foil = text[::-1] if text[::-1].casefold() != text.casefold() else text[:-1]+('X' if text[-1].casefold() != 'x' else 'Z')
        candidates.append(foil)
        jobs.append({'candidate_id': row['candidate_id'], 'target_box': row['bbox'], 'context_box': outer,
            'line_candidate_ids': [rows[j]['candidate_id'] for j in members], 'alternatives': candidates,
            'line_transcript': ' '.join(rows[j]['raw_text'] for j in members)})
    return jobs


def local_views(rgb, job, mask_value):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3: raise ValueError('Original RGB required')
    validate_rows([{'candidate_id': 'probe', 'bbox': job['context_box'], 'raw_text': ''},
                   {'candidate_id': 'target', 'bbox': job['target_box'], 'raw_text': ''}], list(rgb.shape[:2][::-1]))
    x1,y1,x2,y2 = job['context_box']; a,b,c,d = job['target_box']
    if not x1 <= a < c <= x2 or not y1 <= b < d <= y2: raise ValueError('Target must be inside local context')
    original = rgb[y1:y2,x1:x2].copy(); erased = original.copy()
    erased[b-y1:d-y1,a-x1:c-x1] = mask_value
    return {'local': original, 'erased': erased, 'blank': np.full_like(original, mask_value)}


def verification_features(probes, total_words):
    fields = ('support', 'margin', 'visual_drop', 'visual_margin', 'blank_corrected', 'erased_support')
    values = {'verification_fraction': len(probes)/max(total_words, 1), 'verification_missing': float(not probes)}
    for name in fields:
        x = []
        for p in probes:
            scores = p['answers']; a = scores['local'][0]['yes_probability']; b = scores['local'][1]['yes_probability']
            e = scores['erased'][0]['yes_probability']; f = scores['erased'][1]['yes_probability']
            v = {'support': a, 'margin': a-b, 'visual_drop': a-e, 'visual_margin': (a-b)-(e-f),
                 'blank_corrected': a-scores['blank'][0]['yes_probability'], 'erased_support': e}[name]
            x.append(v)
        values['verification_'+name+'_mean'] = float(np.mean(x)) if x else 0.
        values['verification_'+name+'_minimum'] = min(x, default=0.)
    return values


def feature_sets(base, schemas, inventory, verification, embedding):
    vector = dict(base)
    vector.update(inventory); vector.update(verification)
    if len(embedding) == 0 or not np.isfinite(embedding).all(): raise ValueError('Finite frozen visual embedding required')
    visual = {'vision_'+str(i): float(x) for i, x in enumerate(embedding)}
    vector.update(visual)
    core = schemas['core']
    names = {'core': list(core), 'inventory': core+list(inventory),
        'verified': core+list(inventory)+list(verification), 'visual': core+list(visual),
        'combined': core+list(inventory)+list(verification)+list(visual)}
    return vector, names
