"""Separate fixed-group shape diagnostic. prepare is CPU; infer is manual PAPER only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from diagnostics.two_track_common import source, summarize, freeze, cache
from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.lavis_captioner_candidates import digest, sha, write_json
from diagnostics.contextual_ocr_correction import verify
from diagnostics.fixed_group_curve_geometry import fit, render


def pixels_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare(a):
    from diagnostics.gt_mask_localization_oracle import checked_native
    from segmentation.data import load_segmentation_manifest
    from pipeline.frozen_segmentation import backproject_binary_mask
    from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
    cfg = json.loads(a.config.read_text())
    if (cfg['mode'], cfg['split']) != ('APPROXIMATE', 'TRAIN'):
        raise ValueError('APPROXIMATE TRAIN only')
    _, data = source(a.source, cfg)
    verify(a.oracle, cfg['oracle_sha256']); oracle = json.loads(a.oracle.read_text())
    op = oracle['protocol']; paths = op['source_paths']; report_path = Path(paths['source_report'])
    verify(report_path, op['source_report_sha256'])
    verify(paths['train_manifest'], op['train_manifest_sha256'])
    report = json.loads(report_path.read_text())
    manifest = load_segmentation_manifest(paths['train_manifest'], require_checksums=False)
    if manifest.split_role != 'train' or len(manifest.samples) != 229:
        raise ValueError('TRAIN229 required')
    contract = freeze(a.output_dir, cfg, [a.config, a.oracle, report_path, a.source/'results.json'],
        [__file__, 'diagnostics/fixed_group_curve_geometry.py', 'diagnostics/two_track_common.py',
         'localize/grouped_word_blocks.py', 'pipeline/frozen_segmentation.py', 'recognition/parseq.py'])
    params = GroupedWordBlockParams(**op['parameters']['v1'])
    samples = {s['sample_id']: s for s in data['samples']}; rows = []; reasons = Counter()
    cv2.setNumThreads(1)
    for n, sample in enumerate(manifest.samples, 1):
        s = samples[sample.sample_id]; image_path = a.data_root/sample.image_path
        verify(image_path, s['image_sha256'])
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert('RGB'))
        mask_path = report_path.parent/'masks'/f'{sample.sample_id}.json'
        native, meta = checked_native(mask_path, report['contract_sha256'], sample.sample_id,
            s['image_sha256'], op['checkpoint_sha256'])
        projected = backproject_binary_mask(native, rgb.shape[:2])
        if pixels_sha(projected) != meta['segmentation_metadata']['output_mask_sha256']:
            raise RuntimeError('Frozen projected mask mismatch')
        groups = group_word_blocks(projected, projected, params)
        # Source OCR ordering need not be grouping rank ordering; identity is exact bbox.
        frozen = {tuple(b['bbox']): b for b in s['blocks']}
        if len(frozen) != len(s['blocks']) or set(frozen) != {g.padded_bbox for g in groups.retained}:
            raise RuntimeError('Frozen V1 group identity mismatch')
        _, labels = cv2.connectedComponents(projected.astype(np.uint8), connectivity=params.connectivity)
        for group in groups.retained:
            b = frozen[group.padded_bbox]; x1,y1,x2,y2 = group.bbox
            own = np.isin(labels[y1:y2,x1:x2], group.raw_component_ids)
            row = {'candidate_id': b['candidate_id'], 'sample_id': s['sample_id'], 'bbox': b['bbox'],
                   'v1_text': b['text'], 'raw_component_ids': list(group.raw_component_ids),
                   'image_path': str(image_path.resolve()), 'image_sha256': s['image_sha256'],
                   'native_cache_sha256': sha(mask_path), 'variants': {}}
            for kind in cfg['variants'][1:]:
                spec, reason = fit(own, (x1,y1), rgb.shape, kind, cfg)
                reasons[kind+':'+reason] += 1
                row['variants'][kind] = {'spec': spec, 'reason': reason,
                    'pixel_sha256': pixels_sha(render(rgb, spec)) if spec else None}
            rows.append(row)
        print(f'[curve CPU prepare] {n}/229 sample={sample.sample_id}', flush=True)
    baseline, _ = summarize(data, {})
    if baseline['exact'] != 502 or baseline['lexical_matches'] != 503 or len(rows) != 1203:
        raise RuntimeError('V1 population mismatch')
    seal_write(a.output_dir/'inputs.json', {'contract_sha256': contract, 'rows': rows,
        'baseline': baseline, 'reasons': dict(reasons), 'source_root': str(a.source.resolve())})
    print(json.dumps({'baseline': baseline, 'geometry': dict(reasons)}, indent=2), flush=True)


def infer(a):
    from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_ARCHIVE_SOURCE
    p = checked(a.output_dir/'protocol.json'); c = p['contract_sha256']
    for f,h in {**p['input_sha256'], **p['code_sha256']}.items(): verify(f,h)
    inputs = checked(a.output_dir/'inputs.json', c)
    _, data = source(Path(inputs['source_root']), p['config'])
    jobs = [(r,k,v) for r in inputs['rows'] for k,v in r['variants'].items() if v['spec']]
    results = {}; recognizer = None; current = None; rgb = None
    import torch
    from diagnostics.contextual_ocr_correction import runtime_info
    device = torch.device(a.device)
    if device.type != 'cuda': raise ValueError('Manual PAPER CUDA inference only')
    torch.manual_seed(p['config']['seed']); torch.backends.cudnn.benchmark = False
    rev = p['config']['parseq_runtime_revision']
    base = a.artifact_root/'parseq'
    try:
        for n,(row,kind,v) in enumerate(jobs, 1):
            name = digest({'candidate_id': row['candidate_id'], 'kind': kind, 'pixels': v['pixel_sha256']})
            saved = cache(a.output_dir, name, c)
            if saved is None:
                if current != row['image_path']:
                    verify(row['image_path'], row['image_sha256'])
                    with Image.open(row['image_path']) as im: rgb = np.asarray(im.convert('RGB'))
                    current = row['image_path']
                crop = render(rgb, v['spec'])
                if pixels_sha(crop) != v['pixel_sha256']: raise RuntimeError('Rectified RGB changed')
                if recognizer is None:
                    torch.cuda.set_device(device); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=base/f'parseq-{rev}', checkpoint=base/'parseq-bb5792a6.pt',
                        device=str(device), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=base/f'parseq-{rev}.tar.gz'))
                torch.cuda.synchronize(device); start = time.perf_counter()
                result = recognizer.recognize(crop, input_role='T3_localized_crop').to_dict()
                torch.cuda.synchronize(device)
                seal_write(a.output_dir/'cache'/(name+'.json'), {'contract_sha256': c, 'name': name,
                    'result': result, 'seconds': time.perf_counter()-start,
                    'runtime': runtime_info(torch,device), 'batch_size': 1, 'dtype': 'float32',
                    'shape': list(crop.shape), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
                saved = cache(a.output_dir, name, c)
            if saved['name'] != name: raise RuntimeError('Wrong crop cache')
            results[(row['candidate_id'],kind)] = saved['result']['raw_text']
            print(f'[curve PARSeq {n}/{len(jobs)}] {row["candidate_id"]} {kind} V1={row["v1_text"]!r} OUT={saved["result"]["raw_text"]!r}', flush=True)
        summaries = {}; per_word = {}
        for kind in p['config']['variants']:
            mapping = {r['candidate_id']: results.get((r['candidate_id'], kind), r['v1_text']) for r in inputs['rows']}
            summaries[kind], per_word[kind] = summarize(data, mapping)
        seal_write(a.output_dir/'results.json', {'contract_sha256': c, 'classification': 'PRACTICAL_APPROXIMATION',
            'scope': {'split':'TRAIN','images':229,'gt_words':849}, 'summary': summaries,
            'geometry': inputs['reasons'], 'per_word': per_word, 'table5_accuracy': None})
        print(json.dumps(summaries, indent=2)); print('JSON',a.output_dir/'results.json','SHA256',sha(a.output_dir/'results.json'))
    except Exception as exc:
        write_json(a.output_dir/'last-error.json', {'type':type(exc).__name__,'message':str(exc),
            'oom':isinstance(exc,torch.cuda.OutOfMemoryError)})
        raise
    finally:
        if recognizer is not None: recognizer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare','infer'])
    for name in ('config','source','oracle','data-root','output-dir','artifact-root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    if a.output_dir is None: p.error('--output-dir required')
    needed = ('config','source','oracle','data_root') if a.stage=='prepare' else ('artifact_root',)
    for name in needed:
        if getattr(a,name) is None: p.error('--'+name.replace('_','-')+' required')
    (prepare if a.stage=='prepare' else infer)(a)


if __name__ == '__main__': main()
