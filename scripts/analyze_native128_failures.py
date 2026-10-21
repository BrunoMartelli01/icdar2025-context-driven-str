"""CPU-only APPROXIMATE frozen OCR failure and native TRAIN target analysis."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.ndimage import find_objects

from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.tighter_word_gt import native_training_owners

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')) + '\n')


def box_matrices(proposals, words, box_key='bbox'):
    a = np.asarray([p[box_key] for p in proposals], float).reshape(-1, 4)
    b = np.asarray([w['bbox'] for w in words], float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(hi-lo, 0), axis=-1)
    aa = np.prod(np.maximum(a[:, 2:]-a[:, :2], 0), axis=-1)
    bb = np.prod(np.maximum(b[:, 2:]-b[:, :2], 0), axis=-1)
    union = aa[:, None] + bb[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    coverage_gt = np.divide(inter, bb[None, :], out=np.zeros_like(inter), where=bb[None, :] > 0)
    coverage_pred = np.divide(inter, aa[:, None], out=np.zeros_like(inter), where=aa[:, None] > 0)
    return iou, coverage_gt, coverage_pred


def augment(edges, initial=None):
    """Maximum matching preserving all initially matched GT vertices.

    Initial mappings are prediction -> GT. Augmenting paths can remap existing
    pairs but cannot unmatch an initially matched GT; this enables additive
    diagnostics without inconsistent matching ties between stages.
    """
    edges = np.asarray(edges, bool)
    if edges.ndim != 2:
        raise ValueError('Matching requires a 2D edge matrix')
    by_gt = {}
    for pred, gt in (initial or {}).items():
        if not (0 <= pred < edges.shape[0] and 0 <= gt < edges.shape[1]) or not edges[pred, gt] or gt in by_gt:
            raise ValueError('Invalid initial matching')
        by_gt[gt] = pred

    def visit(pred, seen):
        for gt in np.flatnonzero(edges[pred]):
            gt = int(gt)
            if gt in seen:
                continue
            seen.add(gt)
            previous = by_gt.get(gt)
            if previous is None or visit(previous, seen):
                by_gt[gt] = pred
                return True
        return False

    matched_preds = set(by_gt.values())
    for pred in range(edges.shape[0]):
        if pred not in matched_preds:
            visit(pred, set())
            matched_preds = set(by_gt.values())
    result = {pred: gt for gt, pred in by_gt.items()}
    independent = int(np.sum(maximum_bipartite_matching(csr_matrix(edges), perm_type='column') >= 0))
    if len(result) != independent:
        raise AssertionError('Custom matching differs from independent SciPy cardinality')
    return result


def exact_edges(proposals, words, ocr, box_key='bbox', threshold=.5):
    iou, _, _ = box_matrices(proposals, words, box_key)
    text = np.asarray([[ocr[p['crop_array_sha256']]['raw_text'].casefold() == w['text'].casefold()
                        for w in words] for p in proposals], bool).reshape(len(proposals), len(words))
    return (iou >= threshold) & text


def classify_image(shard, words, ocr, threshold=.5, cap=10, padding=1,
                   coverage=.5, fragment_min=.05):
    proposals = shard['variants'][str(padding)]['proposals']
    n, g = len(proposals), len(words)
    iou, gt_cov, pred_cov = box_matrices(proposals, words)
    edges = exact_edges(proposals, words, ocr, threshold=threshold)
    e10 = augment(edges[:cap]); eall = augment(edges, e10)
    geometry = augment(iou >= threshold, eall)
    all_correct, limited_correct, localized = set(eall.values()), set(e10.values()), set(geometry.values())
    if not limited_correct <= all_correct <= localized:
        raise AssertionError('Diagnostic stage sets must be nested')
    padded_exact = augment(exact_edges(proposals, words, ocr, 'padded_bbox_original', threshold))
    # A component can supply only ONE word even when several padding readings
    # exist. Collapsing alternatives by contour prevents duplicate crop oracles.
    contour_ids = sorted(p['contour_index'] for p in proposals)
    index = {v: i for i, v in enumerate(contour_ids)}
    alternative_edges = np.zeros((n, g), bool)
    for variant in shard['variants'].values():
        values = variant['proposals']
        current = exact_edges(values, words, ocr, threshold=threshold)
        for i, p in enumerate(values):
            alternative_edges[index[p['contour_index']]] |= current[i]
    initial = {index[proposals[p]['contour_index']]: gt for p, gt in eall.items()}
    alternative = augment(alternative_edges, initial)
    merged_proxy = gt_cov >= coverage
    merged_rows = np.sum(merged_proxy, axis=1) >= 2
    records = []
    for j, word in enumerate(words):
        category = ('correct_top10' if j in limited_correct else
                    'correct_only_beyond_cap' if j in all_correct else
                    'localized_but_ocr_or_crop_error' if j in localized else 'not_localized_iou50')
        best = int(np.argmax(iou[:, j])) if n else None
        fragments = np.flatnonzero((gt_cov[:, j] >= fragment_min) & (pred_cov[:, j] >= coverage))
        # Bounding-box proxies are deliberately NOT called pixel-mask topology.
        fragment_proxy = j not in localized and len(fragments) >= 2
        merge_proxy = bool(np.any(merged_rows & merged_proxy[:, j]))
        native_w = (word['bbox'][2]-word['bbox'][0])*128/shard['image_hw'][1]
        native_h = (word['bbox'][3]-word['bbox'][1])*128/shard['image_hw'][0]
        records.append({'sample_id':shard['sample_id'], 'gt_index':j, 'gt_text':word['text'],
            'gt_bbox':word['bbox'], 'category':category, 'native_bbox_short_side':min(native_w,native_h),
            'best_iou':float(iou[best,j]) if n else 0.,
            'best_proposal':proposals[best] if n else None,
            'best_ocr_text':ocr[proposals[best]['crop_array_sha256']]['raw_text'] if n else None,
            'has_any_iou50_candidate':bool(np.any(iou[:,j] >= threshold)),
            'fragment_bbox_proxy':bool(fragment_proxy), 'merge_bbox_proxy':merge_proxy,
            'correct_in_padding_oracle':j in alternative.values(),
            'correct_with_padded_box_metric':j in padded_exact.values()})
    return {'gt':g, 'predictions':n, 'correct_top10':len(e10), 'correct_unlimited':len(eall),
        'geometry_unlimited':len(geometry), 'correct_padded_box_metric':len(padded_exact),
        'correct_per_contour_padding_oracle':len(alternative),
        'proposals_no_care_bbox_overlap':int(np.sum(np.max(gt_cov,axis=1)==0)) if g else n,
        'merge_bbox_proxy_proposals':int(merged_rows.sum()),
        'categories':dict(Counter(x['category'] for x in records)), 'words':records}


def owner_stats(native, expected_ids):
    output = {}
    windows = find_objects(native, max_label=max(expected_ids, default=0))
    for owner in expected_ids:
        window = windows[owner-1]
        if window is None:
            output[owner] = {'components':0,'pixels':0,'bbox':None,'min_bbox_side':0}
            continue
        roi = (native[window] == owner).astype(np.uint8)
        ys, xs = window
        output[owner] = {'components':int(cv2.connectedComponents(roi,connectivity=8)[0]-1),
            'pixels':int(roi.sum()), 'bbox':[xs.start,ys.start,xs.stop,ys.stop],
            'min_bbox_side':min(xs.stop-xs.start,ys.stop-ys.start)}
    return output


def summarize_owner_rows(rows):
    return {'words':len(rows), 'missing_words':sum(r['components']==0 for r in rows),
        'fragmented_words':sum(r['components']>1 for r in rows),
        'excess_components':sum(max(0,r['components']-1) for r in rows),
        'min_bbox_side_le2_surviving':sum(0<r['min_bbox_side']<=2 for r in rows),
        'foreground_pixels':sum(r['pixels'] for r in rows)}


def run(config_path, output):
    started = time.monotonic(); c = json.loads(config_path.read_text())
    if config_path.resolve().parent != REPO/'configs/approximate' or c['reproduction_mode']!='approximate':
        raise ValueError('Isolated APPROXIMATE configuration required')
    if output.exists():
        raise FileExistsError('Use a fresh result directory')
    cv2.setNumThreads(c['cpu_threads']); source=Path(c['source_directory']); review=Path(c['review_directory'])
    publications={}
    for name,directory in [('source',source),('review',review)]:
        if file_sha256(directory/'publication_manifest.json') != c[name+'_publication_manifest_sha256']:
            raise ValueError('Frozen publication manifest changed')
        publications[name]={x['path']:x['sha256'] for x in json.loads((directory/'publication_manifest.json').read_text())['files']}

    def verified(directory, prefix, relative):
        path=directory/relative
        if file_sha256(path)!=publications[prefix][relative]:
            raise ValueError('Frozen source changed: '+str(path))
        return json.loads(path.read_text())

    old=verified(source,'source','full/results.json'); old_protocol=verified(source,'source','full/protocol.json')
    gt=verified(source,'source','full/ground_truth.json')['ground_truth']
    if len(gt)!=c['expected_images'] or sum(map(len,gt.values()))!=c['expected_care_words']:
        raise ValueError('Complete fixed TEST population required')
    ocr_path=source/'full/ocr.jsonl'
    if file_sha256(ocr_path)!=publications['source']['full/ocr.jsonl']:
        raise ValueError('Frozen OCR changed')
    ocr={r['crop_array_sha256']:r for line in ocr_path.read_text().splitlines() if (r:=json.loads(line))}
    protocol={'claim_label':c['claim_label'],'reproduction_mode':'approximate','resolved_config':c,
        'config_sha256':file_sha256(config_path),'device':'cpu','gpu_inference':False,
        'source_protocol_sha256':old['protocol_sha256'],
        'code_sha256':{str(p):file_sha256(p) for p in [Path(__file__).resolve(),REPO/'segmentation/tighter_word_gt.py']},
        'method':'Seed exact top10, augment unlimited exact, then geometry; independently verify all matching cardinalities with SciPy',
        'causal_claim':False,'official_table5_claim':False}
    write(output/'protocol.json',protocol)
    results={};word_rows=[]
    for model in c['models']:
        totals=Counter();category=Counter();records=[]
        for sid in sorted(gt):
            rel=f"full/{model}/proposals/{sid.replace(':','__')}.json"
            shard=verified(source,'source',rel)
            value=classify_image(shard,gt[sid],ocr,c['iou_threshold'],c['cap'],c['padding'],c['proxy_coverage'],c['fragment_min_word_fraction'])
            totals.update({k:v for k,v in value.items() if isinstance(v,int)})
            category.update(value['categories']);records.extend(value['words'])
        for cap,field in [('10','correct_top10'),('all','correct_unlimited')]:
            assert totals[field]==old['results'][model]['0'][cap]['1']['exact_end_to_end']['tp']
        assert totals['geometry_unlimited']==old['results'][model]['0']['all']['1']['geometry']['iou_ge_0_50_one_to_one']
        bins={}
        for label,low,hi in [('le2',0,2),('2to4',2,4),('4to8',4,8),('gt8',8,float('inf'))]:
            subset=[r for r in records if low<r['native_bbox_short_side']<=hi]
            bins[label]={'words':len(subset),'categories':dict(Counter(r['category'] for r in subset))}
        results[model]={'totals':dict(totals),'categories':dict(category),'native_size_bins':bins,
            'fragment_bbox_proxy_words':sum(r['fragment_bbox_proxy'] for r in records),
            'merge_bbox_proxy_words':sum(r['merge_bbox_proxy'] for r in records)}
        word_rows.extend(dict(model=model,**row) for row in records)
        print(json.dumps({'model':model,**results[model]}),flush=True)
    write(output/'test_words.json',word_rows)
    write(output/'test_summary.json',results)
    review_config=json.loads((review/'source/configs/approximate/native128_gt_review_v1.json').read_text())
    statistics=verified(review,'review','visuals/gt_statistics.json')
    expected_topology=verified(review,'review','target_topology.json')
    families={v['id']:v for v in review_config['gt_versions'] if v['id'] in ['separated','compact']}
    train={};all_owner_rows=[];train_source=[]
    for family,v in families.items():
        manifest_path=Path(v['manifest'])
        if file_sha256(manifest_path)!=v['manifest_sha256']:
            raise ValueError('TRAIN manifest changed')
        manifest=load_segmentation_manifest(manifest_path,require_checksums=True)
        if manifest.split_role!='train' or len(manifest.samples)!=1255:
            raise ValueError('Expected official TRAIN1255; no TEST target tuning')
        frozen={x['sample_id']:x for x in statistics['versions'][family]['per_image']}
        for i,s in enumerate(manifest.samples,1):
            arrays=[]
            for relative,digest,mode in [(s.word_owner_annotation_path,s.word_owner_annotation_sha256,'I'),
                    (s.validity_annotation_path,s.validity_annotation_sha256,'L'),
                    (s.separation_annotation_path,s.separation_annotation_sha256,'L')]:
                path=Path(v['data_root'])/relative
                if file_sha256(path)!=digest:
                    raise ValueError('TRAIN source changed')
                with Image.open(path) as im:
                    arrays.append(np.asarray(im.convert(mode)).copy())
                train_source.append({'family':family,'sample_id':s.sample_id,'path':str(path),'sha256':digest})
            ids=[int(x) for x in np.unique(arrays[0]) if x>0]
            sizes=c['native_target_resolutions'] if family=='compact' else [128]
            for size in sizes:
                native=native_training_owners(*arrays,(size,size))
                digest=sha256((native>0).astype(np.uint8).tobytes()).hexdigest()
                if size==128 and digest!=frozen[s.sample_id]['native_target_array_sha256']:
                    raise ValueError('Actual native TRAIN transform hash replay failed')
                values=owner_stats(native,ids)
                binary_components=int(cv2.connectedComponents((native>0).astype(np.uint8),connectivity=8)[0]-1)
                if sum(x['components'] for x in values.values())!=binary_components:
                    raise ValueError('Distinct words touch after native conversion')
                all_owner_rows.extend(dict(family=family,size=size,sample_id=s.sample_id,owner=owner,**row) for owner,row in values.items())
            if i%250==0 or i==len(manifest.samples):
                print(f'[TRAIN] {family} {i}/{len(manifest.samples)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    for family,size in [('separated',128),('compact',128),('compact',256)]:
        label=f'{family}_{size}';subset=[r for r in all_owner_rows if r['family']==family and r['size']==size]
        train[label]=summarize_owner_rows(subset)
        if size==128:
            expected=expected_topology['versions'][family]
            assert all(train[label][k]==expected[k] for k in ['missing_words','fragmented_words','excess_components'])
    separated={(r['sample_id'],r['owner']):r for r in all_owner_rows if r['family']=='separated'}
    compact={(r['sample_id'],r['owner']):r for r in all_owner_rows if r['family']=='compact' and r['size']==128}
    assert set(separated)==set(compact)
    train['fallback_opportunities']={
        'compact_fragmented_and_separated_connected':sum(b['components']>1 and separated[k]['components']==1 for k,b in compact.items()),
        'compact_missing_and_separated_connected':sum(b['components']==0 and separated[k]['components']==1 for k,b in compact.items()),
        'both_fragmented':sum(b['components']>1 and separated[k]['components']>1 for k,b in compact.items())}
    write(output/'train_words.json',all_owner_rows);write(output/'train_sources.json',train_source)
    write(output/'train_summary.json',train)
    report={'status':'SUCCESS','claim_label':c['claim_label'],'result_class':'approximate reproduction; full TEST diagnosis plus full unaugmented TRAIN audit',
        'images':len(gt),'care_words':sum(map(len,gt.values())),'test':results,'train':train,
        'independent_matching_cardinality_checks':'PASS (SciPy every match)', 'native128_array_hash_replays':2510,
        'native256_is_target_counterfactual_not_new_model':True,'gpu_inference':False,'saved_crops_or_weights':0,
        'elapsed_seconds':time.monotonic()-started,'protocol_sha256':file_sha256(output/'protocol.json')}
    write(output/'results.json',report)
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>c['output_budget_bytes']:
        raise RuntimeError('Numeric diagnostic budget exceeded')
    print(json.dumps({k:v for k,v in report.items() if k not in ['test']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        run(args.config,args.output)
    except Exception as error:
        import traceback
        write(args.output/'failure.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc()})
        raise
