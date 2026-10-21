"""CPU TRAIN-only post-hoc crop completeness/contamination; no new inference."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from diagnostics.directional_neighbors_train import annotation_word_colors
from diagnostics.gt_mask_localization_oracle import checked_native, save_json, sha, verify_file
from diagnostics.train_v1_crop_glyph_audit import color_code, color_counts, packed_rgb
from pipeline.frozen_segmentation import backproject_binary_mask
from segmentation.data import load_segmentation_manifest


def padded_box(box, shape, pad):
    if box is None:
        return None
    h, w = shape
    x1, y1, x2, y2 = box
    return [max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)]


def counts_in_box(packed, box, pred=None):
    if box is None:
        return {}
    x1, y1, x2, y2 = box
    h, w = packed.shape
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('invalid xyxy exclusive crop')
    region = packed[y1:y2, x1:x2]
    if pred is not None:
        region = region[np.asarray(pred[y1:y2, x1:x2], dtype=bool)]
    return color_counts(region)


def crop_metrics(counts, own_codes, totals, word_codes, dc_codes):
    fractions = [counts.get(c, 0)/totals[c] for c in own_codes]
    own = sum(counts.get(c, 0) for c in own_codes)
    other = []
    for i, codes in enumerate(word_codes):
        if set(codes) == set(own_codes):
            continue
        n = sum(counts.get(c, 0) for c in codes)
        if n:
            other.append({'gt_index': i, 'pixels': n,
                          'word_foreground_fraction': n/sum(totals[c] for c in codes)})
    foreign = sum(r['pixels'] for r in other)
    dc = sum(counts.get(c, 0) for c in dc_codes)
    return {
        'own_foreground_retention': own/sum(totals[c] for c in own_codes),
        'min_glyph_retention': min(fractions),
        'missing_entire_glyphs': sum(f == 0 for f in fractions),
        'all_own_glyph_pixels_inside': all(f == 1 for f in fractions),
        'other_care_pixels': foreign,
        'other_care_fraction_of_care_pixels': foreign/(own+foreign) if own+foreign else None,
        'other_words': other,
        'dont_care_pixels': dc,
    }


def completeness_class(available, metric):
    if not available:
        return 'NO_CROP'
    complete = 'COMPLETE' if metric['all_own_glyph_pixels_inside'] else 'INCOMPLETE'
    other = 'OTHER_CARE_PRESENT' if metric['other_care_pixels'] else 'NO_OTHER_CARE'
    return complete+'_'+other


def summarize(rows):
    wrong = [r for r in rows if not r['exact']]
    incomplete = [r for r in wrong if not r['actual']['all_own_glyph_pixels_inside']]
    return {
        'words': len(rows), 'exact': sum(r['exact'] for r in rows), 'errors': len(wrong),
        'error_completeness_classes': dict(Counter(r['completeness_class'] for r in wrong)),
        'correct_completeness_classes': dict(Counter(r['completeness_class'] for r in rows if r['exact'])),
        'errors_with_other_care_pixels': sum(r['actual']['other_care_pixels'] > 0 for r in wrong),
        'errors_with_dont_care_pixels': sum(r['actual']['dont_care_pixels'] > 0 for r in wrong),
        'incomplete_or_absent_crop_errors': len(incomplete),
        'incomplete_errors_all_glyphs_have_some_prediction_support': sum(r['glyphs_without_predicted_support'] == 0 for r in incomplete),
        'incomplete_errors_any_glyph_without_prediction_support': sum(r['glyphs_without_predicted_support'] > 0 for r in incomplete),
        'incomplete_errors_no_glyph_with_prediction_support': sum(r['glyphs_without_predicted_support'] == r['glyph_count'] for r in incomplete),
        'incomplete_errors_omitted_gt_pixels_have_prediction_support': sum(r['supported_own_pixels_outside_actual_crop'] > 0 for r in incomplete),
        'oracle_incomplete_errors_all_glyphs_retained': sum(r['oracle']['all_own_glyph_pixels_inside'] for r in incomplete),
        'oracle_incomplete_errors_all_glyphs_retained_no_other_care': sum(r['oracle']['all_own_glyph_pixels_inside'] and not r['oracle']['other_care_pixels'] for r in incomplete),
        'oracle_incomplete_errors_all_glyphs_retained_no_other_care_or_dc': sum(r['oracle']['all_own_glyph_pixels_inside'] and not r['oracle']['other_care_pixels'] and not r['oracle']['dont_care_pixels'] for r in incomplete),
        'oracle_incomplete_errors_improved_foreground_retention': sum(r['oracle']['own_foreground_retention'] > r['actual']['own_foreground_retention'] for r in incomplete),
        'oracle_errors_foreground_retention_worsened': sum(r['oracle']['own_foreground_retention'] < r['actual']['own_foreground_retention'] for r in wrong),
        'oracle_errors_other_care_pixels_increased': sum(r['oracle']['other_care_pixels'] > r['actual']['other_care_pixels'] for r in wrong),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'audit', 'glyph-audit', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cfg = json.loads(args.config.read_text())
    scope = {'split': 'TRAIN', 'images': 229, 'gt_words': 849}
    if cfg['scope'] != scope or cfg['reproduction_mode'] != 'approximate':
        raise ValueError('approximate TRAIN only')
    verify_file(args.audit, cfg['audit_sha256'])
    verify_file(args.glyph_audit, cfg['glyph_audit_sha256'])
    audit = json.loads(args.audit.read_text())
    glyph_audit = json.loads(args.glyph_audit.read_text())
    if audit['protocol']['scope'] != scope or glyph_audit['scope'] != scope:
        raise ValueError('input scope mismatch')
    oracle_path = Path(audit['protocol']['oracle_path'])
    verify_file(oracle_path, audit['protocol']['oracle_sha256'])
    oracle = json.loads(oracle_path.read_text())
    for path, expected in oracle['protocol']['code_sha256'].items():
        verify_file(path, expected)
    paths = oracle['protocol']['source_paths']
    source_path = Path(paths['source_report'])
    verify_file(source_path, audit['protocol']['source_report_sha256'])
    source = json.loads(source_path.read_text())
    verify_file(paths['train_manifest'], oracle['protocol']['train_manifest_sha256'])
    manifest = load_segmentation_manifest(paths['train_manifest'], require_checksums=False)
    if manifest.split_role != 'train' or len(manifest.samples) != 229:
        raise ValueError('TRAIN229 required')
    pad = oracle['protocol']['parameters']['v1']['pad']
    per_image = defaultdict(list)
    for row in audit['per_word']:
        per_image[row['sample_id']].append(row)
    glyph_refs = {(r['sample_id'], r['gt_index']): r for r in glyph_audit['per_word']}
    root = Path(paths['data_root'])
    rows = []
    for index, sample in enumerate(manifest.samples, 1):
        if index == 1 or index % 25 == 0 or index == 229:
            print(f'[crop completeness CPU] {index}/229', flush=True)
        for rel in (sample.mask_path, sample.validity_annotation_path):
            verify_file(root/rel, source['protocol']['dataset_files'][str(rel)]['sha256'])
        native, meta = checked_native(source_path.parent/'masks'/f'{sample.sample_id}.json',
            source['contract_sha256'], sample.sample_id,
            source['protocol']['dataset_files'][str(sample.image_path)]['sha256'],
            audit['protocol']['checkpoint_sha256'])
        with Image.open(root/sample.mask_path) as opened:
            packed = packed_rgb(opened.convert('RGB'))
        pred = backproject_binary_mask(native, packed.shape)
        if hashlib.sha256(pred.tobytes()).hexdigest() != meta['segmentation_metadata']['output_mask_sha256']:
            raise RuntimeError('frozen backprojection changed')
        annotation = (root/sample.validity_annotation_path).read_text(encoding='utf-8-sig')
        word_codes = [[color_code(c) for c in colors] for colors in annotation_word_colors(annotation)]
        dc_codes = {color_code(list(map(int, line.lstrip()[1:].split()[:3])))
                    for line in annotation.splitlines() if line.lstrip().startswith('#')}
        flattened = [c for codes in word_codes for c in codes]
        if len(set(flattened)) != len(flattened) or set(flattened) & dc_codes:
            raise ValueError('ambiguous glyph color ownership')
        totals = color_counts(packed)
        supported = color_counts(packed[np.asarray(pred, dtype=bool)])
        if any(totals.get(c, 0) == 0 for c in flattened):
            raise ValueError('annotated glyph color missing')
        words = per_image[sample.sample_id]
        if len(words) != len(word_codes):
            raise ValueError('GT word count mismatch')
        for row, own_codes in zip(words, word_codes):
            actual_box = row['crop_bbox'] if row['available'] else None
            union_box = padded_box(row['oracle_component_union_bbox'], packed.shape, pad)
            actual = crop_metrics(counts_in_box(packed, actual_box), own_codes, totals, word_codes, dc_codes)
            union = crop_metrics(counts_in_box(packed, union_box), own_codes, totals, word_codes, dc_codes)
            prior = glyph_refs[(sample.sample_id, row['gt_index'])]
            if actual['min_glyph_retention'] != prior['minimum_glyph_fraction'] or actual['own_foreground_retention'] != prior['care_foreground_fraction_in_crop']:
                raise RuntimeError('glyph audit mismatch')
            supported_inside = counts_in_box(packed, actual_box, pred)
            support = [supported.get(c, 0)/totals[c] for c in own_codes]
            rows.append({
                'sample_id': sample.sample_id, 'gt_index': row['gt_index'],
                'gt_text': row['gt_text'], 'parseq_text': row['parseq_text'], 'exact': row['exact'],
                'crop_failure': row['crop_failure'], 'geometry_category': row['geometry_category'],
                'actual_bbox': actual_box, 'oracle_union_padded_bbox': union_box,
                'actual': actual, 'oracle': union,
                'completeness_class': completeness_class(row['available'], actual),
                'glyph_count': len(own_codes), 'glyph_prediction_support_fractions': support,
                'glyphs_without_predicted_support': sum(v == 0 for v in support),
                'supported_own_pixels_outside_actual_crop': sum(supported.get(c, 0)-supported_inside.get(c, 0) for c in own_codes),
            })
    if len(rows) != 849 or sum(r['exact'] for r in rows) != 502 or len(per_image) != 229:
        raise RuntimeError('frozen population changed')
    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir/'per-word.csv'
    with csv_path.open('w', newline='') as stream:
        fields = ['sample_id', 'gt_index', 'gt_text', 'parseq_text', 'exact', 'completeness_class',
                  'glyphs_without_predicted_support', 'supported_own_pixels_outside_actual_crop',
                  'actual_retention', 'oracle_retention', 'actual_other_pixels', 'oracle_other_pixels']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'actual_retention': row['actual']['own_foreground_retention'],
                'oracle_retention': row['oracle']['own_foreground_retention'],
                'actual_other_pixels': row['actual']['other_care_pixels'],
                'oracle_other_pixels': row['oracle']['other_care_pixels']})
    report = {
        'protocol': {**cfg, 'config_sha256': sha(args.config), 'audit_path': str(args.audit.resolve()),
            'glyph_audit_path': str(args.glyph_audit.resolve()), 'code_sha256': sha(__file__),
            'padding': pad, 'source_report_sha256': sha(source_path), 'device': 'cpu',
            'counterfactual': 'GT-associated raw-component union plus frozen padding; DIAGNOSTIC_ONLY; no OCR'},
        'summary': summarize(rows), 'per_word': rows, 'csv_sha256': sha(csv_path),
    }
    save_json(args.output_dir/'results.json', report)
    print(json.dumps(report['summary'], indent=2))
    print('JSON', args.output_dir/'results.json', 'SHA256', sha(args.output_dir/'results.json'))


if __name__ == '__main__':
    main()
