"""GT-free prompt features and TRAIN-only calibration for experimental routers."""
from __future__ import annotations

import math
from recognition.clip_crop_admission import percentile


def mutations(text):
    """Deterministic spelling controls, never drawn from an annotation lexicon."""
    if not text.strip():
        return []
    i = len(text) // 2
    replacement = 'x' if text[i].casefold() != 'x' else 'z'
    candidates = [text[:i] + replacement + text[i+1:], text[::-1]]
    if len(text) > 1:
        candidates.append(text[:i] + text[i+1:])
    return sorted({v for v in candidates if v.strip() and v != text})


def prompt_chunks(template, text, tokenizer, limit):
    """Preserve every character, without silent truncation; repeat prefix per chunk.

    Prefer whitespace boundaries. Extremely long words are split by characters.
    Chunking is an experimental aggregation assumption, not a CLIP guarantee.
    """
    if template.count('{text}') != 1 or limit < 1:
        raise ValueError('Invalid prompt/context limit')
    size = lambda value: len(tokenizer(template.format(text=value),
        add_special_tokens=True, truncation=False)['input_ids'])
    if size('') >= limit:
        raise ValueError('Prompt prefix exhausts context')
    if not text:
        return [template.format(text='')]
    parts, remaining = [], text
    while remaining:
        if size(remaining) <= limit:
            parts.append(remaining)
            break
        # Scan prefixes instead of assuming monotonic BPE length.
        valid = [i for i in range(1, len(remaining)+1) if size(remaining[:i]) <= limit]
        if not valid:
            raise ValueError('A character cannot fit in model context')
        end = max(valid)
        boundary = remaining.rfind(' ', 0, end+1)
        if boundary > 0:
            end = boundary+1
        parts.append(remaining[:end])
        remaining = remaining[end:]
    if ''.join(parts) != text:
        raise AssertionError('Chunking lost transcript characters')
    prompts = [template.format(text=p) for p in parts]
    if any(len(tokenizer(p, add_special_tokens=True, truncation=False)['input_ids']) > limit for p in prompts):
        raise ValueError('Prompt chunk exceeds context')
    return prompts


def transcript(rows):
    return ' '.join(r['raw_text'] for r in sorted(rows,
        key=lambda r: (r['bbox'][1], r['bbox'][0], r['instance_id'])))


def fit_reference(features, fit_ids):
    selected = [features[sid] for sid in fit_ids]
    return {'word': {name: sorted(w[name] for s in selected for w in s['word_features'])
                    for name in selected[0]['word_feature_names']},
            'scene': {name: sorted(s['scene_features'][name] for s in selected if s['eligible'])
                     for name in selected[0]['scene_features']},
            'native': sorted(v for s in selected for v in s['native'])}


def image_scores(s, ref):
    names = ['native_only', *('crop_'+k for k in s['word_feature_names']),
             *('scene_'+k for k in s['scene_features']), 'crop_scene_en', 'margin_scene_en']
    if not s['eligible']:
        return dict.fromkeys(names)
    native = [percentile(ref['native'], v) for v in s['native']]
    out = {'native_only': min(native)}
    for k in s['word_feature_names']:
        out['crop_'+k] = min((n*percentile(ref['word'][k], w[k]))**.5
                            for n, w in zip(native, s['word_features']))
    for k, v in s['scene_features'].items():
        out['scene_'+k] = (min(native)*percentile(ref['scene'][k], v))**.5
    scene = percentile(ref['scene']['all_en'], s['scene_features']['all_en'])
    # Retain the weakest word; an easy word never rescues an unreliable word.
    out['crop_scene_en'] = (out['crop_baseline']**2 * scene)**(1/3)
    out['margin_scene_en'] = (out['crop_margin']**2 * scene)**(1/3)
    if any(not math.isfinite(v) for v in out.values()):
        raise ValueError('Nonfinite image score')
    return out


def coverage_threshold(scores, ids, fraction):
    """Inclusive score threshold nearest the target count, without GT labels."""
    if not 0 < fraction <= 1 or not ids or len(set(ids)) != len(ids):
        raise ValueError('Invalid coverage population')
    values = [scores[sid] for sid in ids if scores[sid] is not None]
    if not values:
        return None
    candidates = sorted(set(values))
    target = len(ids)*fraction
    return min(candidates, key=lambda t: (abs(sum(v >= t for v in values)-target), -t))


def budget_accept(scores, ids, count):
    """Descriptive exact budget; GT-free deterministic ID tie breaking."""
    if not 0 <= count <= len(ids):
        raise ValueError('Invalid image budget')
    valid = sorted((sid for sid in ids if scores[sid] is not None), key=lambda sid: (-scores[sid], sid))
    return set(valid[:count])
