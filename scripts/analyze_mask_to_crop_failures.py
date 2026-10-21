"""CPU-only failure diagnosis of a TRAIN-selected U-Net's frozen TEST crops."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from localize.native_repository_contours import remap_native_box
from scripts.analyze_native128_failures import augment, box_matrices, exact_edges


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def alphanumeric(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def at_most_one_edit(a, b):
    a, b = a.casefold(), b.casefold()
    if abs(len(a) - len(b)) > 1: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b): a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]: i += 1
    return a[i:] == b[i + 1:]


def diagnose(shard, words, ocr, config):
    """All proposals/readings preexist; GT is used only for these diagnostics."""
    pad, threshold = str(config['padding_native']), config['iou_threshold']
    proposals = shard['variants'][pad]['proposals']
    iou, coverage, pred_coverage = box_matrices(proposals, words)
    exact = augment(exact_edges(proposals, words, ocr, threshold=threshold))
    geometry = augment(iou >= threshold, exact)
    ids, localized = set(exact.values()), set(geometry.values())
    contours = [p['contour_index'] for p in proposals]
    if len(set(contours)) != len(contours): raise ValueError('Duplicate baseline contour identity')
    positions = {cid: i for i, cid in enumerate(contours)}
    alternatives = np.zeros((len(proposals), len(words)), bool)
    for variant in shard['variants'].values():
        other = variant['proposals']
        edges = exact_edges(other, words, ocr, threshold=threshold)
        for j, p in enumerate(other):
            if p['contour_index'] in positions:
                base = proposals[positions[p['contour_index']]]
                if p['bbox'] != base['bbox']: raise ValueError('Contour detection box changed across crop views')
                alternatives[positions[p['contour_index']]] |= edges[j]
    oracle = augment(alternatives, exact)
    padded = augment(exact_edges(proposals, words, ocr, 'padded_bbox_original', threshold))
    texts = np.asarray([[ocr[p['crop_array_sha256']]['raw_text'].casefold() == w['text'].casefold()
                         for w in words] for p in proposals], bool).reshape(len(proposals), len(words))
    relaxed = augment(texts & (iou > 0), exact)
    raw = [{'bbox': remap_native_box(c['bbox_native'], shard['image_hw'])}
           for c in shard['mask_measurements']['components_8_connected']]
    raw_iou, _, _ = box_matrices(raw, words)
    components = shard['mask_measurements']['components_8_connected']
    if len(components) != shard['variants'][pad]['candidates_before_cap']:
        raise ValueError('Raw CC boxes do not establish the uncapped external-contour population')
    raw_index = {c['label']: i for i, c in enumerate(components)}
    for p in proposals:
        c = components[raw_index[p['component_label']]]
        if c['bbox_native'] != p['bbox_native']:
            raise ValueError('Retained contour/CC box identity differs')
    raw_initial = {raw_index[proposals[i]['component_label']]: g for i, g in geometry.items()}
    raw_geometry = augment(raw_iou >= threshold, raw_initial)
    merge = coverage >= config['merge_bbox_coverage']
    merge_rows = merge.sum(axis=1) >= 2
    rows = []
    for j, word in enumerate(words):
        best = int(np.argmax(iou[:, j])) if len(proposals) else None
        p = proposals[best] if best is not None else None
        text = ocr[p['crop_array_sha256']]['raw_text'] if p else None
        short = min((word['bbox'][2]-word['bbox'][0])*config['native_shape'][1]/shard['image_hw'][1],
                    (word['bbox'][3]-word['bbox'][1])*config['native_shape'][0]/shard['image_hw'][0])
        category = 'correct' if j in ids else 'localized_ocr_or_crop_error' if j in localized else 'not_localized'
        rows.append({'sample_id': shard['sample_id'], 'gt_index': j, 'gt_text': word['text'], 'gt_bbox': word['bbox'],
            'category': category, 'native_bbox_short_side': short,
            'size_bin': 'le4' if short <= 4 else '4to8' if short <= 8 else 'gt8',
            'best_iou': float(iou[best, j]) if best is not None else 0.,
            'best_proposal': p, 'best_ocr_text': text,
            'any_retained_bbox_overlap': bool(np.any(iou[:, j] > 0)),
            'any_retained_iou50': bool(np.any(iou[:, j] >= threshold)),
            'raw_component_bbox_iou50': bool(np.any(raw_iou[:, j] >= threshold)),
            'localized_uncapped_geometry': j in raw_geometry.values(),
            'merge_bbox_proxy': bool(np.any(merge_rows & merge[:, j])),
            'fragment_bbox_proxy': bool(j not in localized and np.sum(
                (coverage[:, j] >= config['fragment_min_gt_coverage']) &
                (pred_coverage[:, j] >= config['merge_bbox_coverage'])) >= 2),
            'best_text_one_edit': bool(text is not None and text.casefold() != word['text'].casefold() and at_most_one_edit(text, word['text'])),
            'best_text_alnum_only': bool(text is not None and text.casefold() != word['text'].casefold() and alphanumeric(text) == alphanumeric(word['text'])),
            'correct_padded_box_metric': j in padded.values(),
            'correct_positive_overlap_text': j in relaxed.values(),
            'correct_padding_oracle': j in oracle.values()})
    return rows, {'gt_words': len(words), 'proposals': len(proposals), 'exact': len(exact),
                  'localized': len(geometry), 'padding_oracle': len(oracle), 'padded_box_metric': len(padded),
                  'positive_overlap_exact': len(relaxed), 'uncapped_geometry':len(raw_geometry),
                  'candidates_before_cap': shard['variants'][pad]['candidates_before_cap']}


def run(args):
    config_path = args.config.resolve()
    if args.mode != 'approximate' or REPO / 'configs/approximate' not in config_path.parents:
        raise ValueError('Explicit APPROXIMATE configuration required')
    cfg = read(config_path)
    if cfg['reproduction_mode'] != args.mode or cfg['experiment_kind'] != 'mask_to_crop_failure_research_v1':
        raise ValueError('Unsupported analysis configuration')
    root = args.run_root.resolve(); source = root / 'ocr' / ('area' + str(cfg['source_area_filter']))
    training = root / 'training/total-text-guarded-v3-word-selection-500ep-v1'
    selected = read(training / 'checkpoint_selection.json')[cfg['source_selection']]
    epoch = selected['epoch']; checkpoint = training / 'checkpoints' / selected['checkpoint']
    if digest(checkpoint) != selected['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
    receipt = read(training / 'run_manifest.json')
    if receipt['status'] != 'SUCCESS' or not receipt['training_complete'] or receipt['test_loaded']:
        raise ValueError('Complete TRAIN-only selected model required')
    protocol_path = source / 'protocol.json'; protocol = read(protocol_path); protocol_sha = digest(protocol_path)
    results = read(source / 'results.json'); audit = read(source / 'audit.json')
    if results['status'] != 'SUCCESS' or audit['status'] != 'PASS' or results['protocol_sha256'] != protocol_sha:
        raise ValueError('Qualified frozen evaluation required')
    if (protocol['plan']['max_proposals_per_image'] != cfg['source_cap']
            or protocol['plan']['min_unpadded_bbox_area_native'] != cfg['source_area_filter']
            or len(protocol['sample_ids']) != cfg['expected_images']
            or results['checkpoints'][str(epoch)]['checkpoint_sha256'] != selected['checkpoint_sha256']):
        raise ValueError('Frozen inference/selection protocol differs')
    source_files = [protocol_path, source/'results.json', source/'audit.json', source/'ground_truth.json',
                    source/'ocr.jsonl', training/'checkpoint_selection.json', training/'run_manifest.json']
    source_hashes = {str(p): digest(p) for p in source_files}
    ocr = {}
    for line in (source / 'ocr.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['protocol_sha256'] != protocol_sha: raise ValueError('OCR cache protocol differs')
        key = row['crop_array_sha256']
        if key in ocr and ocr[key]['raw_text'] != row['raw_text']: raise ValueError('Conflicting frozen OCR')
        ocr[key] = row
    gt = read(source/'ground_truth.json')['ground_truth']
    words, totals, per_image, shard_hashes = [], Counter(), [], {}
    for sid in protocol['sample_ids']:
        path = source / f'epoch-{epoch:04d}' / 'proposals' / (sid.replace(':','__')+'.json')
        shard = read(path); shard_hashes[str(path)] = digest(path)
        if shard['sample_id'] != sid or shard['protocol_sha256'] != protocol_sha or shard['epoch'] != epoch:
            raise ValueError('Frozen proposal identity differs')
        rows, summary = diagnose(shard, gt[sid], ocr, cfg)
        words.extend(rows); totals.update(summary); per_image.append({'sample_id':sid, **summary})
    baseline = next(r for r in read(root/'ocr_summary.json')['results'] if r['epoch']==epoch
                    and r['bbox_min_area']==cfg['source_area_filter'] and r['padding_native']==cfg['padding_native']
                    and r['box_metric']=='unpadded_detection_box')
    if (len(per_image) != cfg['expected_images'] or totals['gt_words'] != cfg['expected_care_words']
            or totals['exact'] != baseline['correct_words'] or totals['localized'] != baseline['localized_words']
            or totals['proposals'] != baseline['proposals']): raise AssertionError('Baseline replay differs')
    categories = dict(Counter(r['category'] for r in words))
    if sum(categories.values()) != cfg['expected_care_words']: raise AssertionError('Nonexclusive failure partition')
    size = {}
    for group in ['le4','4to8','gt8']:
        subset = [r for r in words if r['size_bin']==group]
        size[group] = {'words':len(subset), 'categories':dict(Counter(r['category'] for r in subset))}
    failed = [r for r in words if r['category']=='not_localized']
    localized_failed = [r for r in words if r['category']=='localized_ocr_or_crop_error']
    summary = {'status':'SUCCESS', 'reproduction_mode':'approximate',
        'result_class':'approximate reproduction; frozen-output exploratory TEST diagnosis',
        'selected_epoch':epoch, 'selected_by':cfg['source_selection'], 'checkpoint':str(checkpoint),
        'checkpoint_sha256':selected['checkpoint_sha256'], 'totals':dict(totals), 'categories':categories, 'size_bins':size,
        'not_localized_diagnostics':{'no_retained_bbox_overlap':sum(not r['any_retained_bbox_overlap'] for r in failed),
            'raw_component_bbox_can_match_but_retained_cannot':sum(r['raw_component_bbox_iou50'] and not r['any_retained_iou50'] for r in failed),
            'additional_word_matches_after_removing_cap_geometry_only':totals['uncapped_geometry']-totals['localized'],
            'merge_bbox_proxy':sum(r['merge_bbox_proxy'] for r in failed),
            'fragment_bbox_proxy':sum(r['fragment_bbox_proxy'] for r in failed)},
        'localized_failure_diagnostics':{'one_edit_best_iou_text':sum(r['best_text_one_edit'] for r in localized_failed),
            'alphanumeric_only_best_iou_text':sum(r['best_text_alnum_only'] for r in localized_failed)},
        'caveats':['bbox proxies overlap and are not pixel topology', 'below-cap V3 OCR is unavailable; uncapped OCR is not inferred',
                   'padding oracle and relaxed box matching use GT only diagnostically, never inference or newly achieved performance']}
    output = args.output.resolve()
    if output.exists(): raise ValueError('Preserve existing analyses; choose a new output directory')
    output.mkdir(parents=True)
    provenance = {'reproduction_mode':'approximate','result_class':'approximate reproduction; frozen-output exploratory TEST diagnosis',
        'configuration':cfg, 'configuration_path':str(config_path),'configuration_sha256':digest(config_path),
        'source_hashes':source_hashes,'proposal_hashes':shard_hashes,'script_sha256':digest(Path(__file__)),
        'paper_sha256':digest(REPO/'papers/2503.15639v2.pdf'), 'environment':{'python':platform.python_version(),'numpy':np.__version__},
        'neural_inference':False,'training':False,'gpu_allocation':False,'source_outputs_modified':False}
    for filename, data in [('summary.json',summary),('words.json',words),('per_image.json',per_image),('provenance.json',provenance)]:
        (output/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for path, expected in source_hashes.items():
        if digest(path) != expected: raise ValueError('Source changed during analysis: '+path)
    (output/'audit.json').write_text(json.dumps({'status':'PASS','source_hashes_unchanged':True,
        'baseline_replayed_exactly':True,'all_matching_cardinalities_independently_checked_with_scipy':True,
        'images':len(per_image),'care_words':len(words)},indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
