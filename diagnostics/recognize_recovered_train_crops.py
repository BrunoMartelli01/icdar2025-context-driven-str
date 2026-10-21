"""PARSeq-only diagnostic on frozen80 TRAIN geometric recoveries and GT_BOX controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from diagnostics.gt_mask_localization_oracle import save_json, sha, verify_file
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, ParseqApproxConfig,
    ParseqApproxRecognizer, verify_parseq_runtime_source,
)
from segmentation.data import load_segmentation_manifest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def selected_words(rows):
    return [r for r in rows if not r['exact']
            and not r['actual']['all_own_glyph_pixels_inside']
            and r['oracle']['all_own_glyph_pixels_inside']
            and not r['oracle']['other_care_pixels']
            and not r['oracle']['dont_care_pixels']]


def scores(text, gt):
    fuzzy = ratio(text.casefold(), gt.casefold())/100 if text is not None else 0.
    return {'text': text, 'exact_casefold': text is not None and text.casefold() == gt.casefold(),
            'rapidfuzz_casefold': fuzzy, 'fuzzy_ge_0_8': fuzzy >= .8}


def read_cached(path, contract, request):
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    payload = saved['payload']
    if saved['sha256'] != digest(payload) or payload['contract'] != contract or payload['request'] != request:
        raise RuntimeError(f'cache provenance mismatch: {path}')
    return payload['result']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'data-root', 'parseq-source', 'parseq-source-archive', 'parseq-checkpoint', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-report-sha256', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    verify_file(a.report, a.expected_report_sha256)
    report = json.loads(a.report.read_text())
    if report['protocol']['scope'] != {'split': 'TRAIN', 'images': 229, 'gt_words': 849}:
        raise ValueError('TRAIN229/849 only')
    audit_path = Path(report['protocol']['audit_path'])
    verify_file(audit_path, report['protocol']['audit_sha256'])
    audit = json.loads(audit_path.read_text())
    oracle_path = Path(audit['protocol']['oracle_path'])
    verify_file(oracle_path, audit['protocol']['oracle_sha256'])
    oracle = json.loads(oracle_path.read_text())
    paths = oracle['protocol']['source_paths']
    verify_file(paths['source_report'], audit['protocol']['source_report_sha256'])
    source = json.loads(Path(paths['source_report']).read_text())
    verify_file(paths['train_manifest'], oracle['protocol']['train_manifest_sha256'])
    manifest = load_segmentation_manifest(paths['train_manifest'], require_checksums=False)
    if manifest.split_role != 'train' or len(manifest.samples) != 229:
        raise ValueError('TRAIN229 only')
    samples = {s.sample_id: s for s in manifest.samples}
    old = {(r['sample_id'], r['gt_index']): r for r in audit['per_word']}
    selected = selected_words(report['per_word'])
    if len(selected) != 80 or len({(r['sample_id'], r['gt_index']) for r in selected}) != 80:
        raise ValueError('frozen80 recoveries changed')
    verify_file(a.parseq_checkpoint, PARSEQ_CHECKPOINT_SHA256)
    identity = verify_parseq_runtime_source(a.parseq_source, source_type=PARSEQ_ARCHIVE_SOURCE,
                                           source_archive=a.parseq_source_archive)
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    protocol = {
        'classification': 'APPROXIMATE_DIAGNOSTIC', 'reproduction_mode': 'approximate',
        'source_scope': report['protocol']['scope'], 'evaluated_gt_words': 80,
        'selection': 'Frozen GT-dependent geometrically recovered errors ONLY; not general accuracy',
        'report_sha256': sha(a.report), 'code_sha256': sha(__file__),
        'parseq_adapter_sha256': sha('recognition/parseq.py'),
        'checkpoint_sha256': sha(a.parseq_checkpoint), 'runtime_source': identity,
        'device': a.device, 'dtype': 'float32', 'batch_size': 1, 'seed': 0,
        'torch': torch.__version__, 'numpy': np.__version__,
        'cuda_runtime': torch.version.cuda, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'table5_accuracy': None, 'segmentation_run': False, 'test_loaded': False,
        'scientific_choices': [
            {'classification': 'ASSUMPTION', 'choice': 'GT-dependent crop counterfactual and GT_BOX controls on selected80 failures only; V1 OCR reused, no tuning.'},
            {'classification': 'HARDWARE_DEVIATION', 'choice': 'Standalone PARSeq batch1; other models not loaded.'}],
    }
    contract = digest(protocol)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = a.output_dir/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('existing run protocol differs')
    save_json(protocol_path, protocol)
    cache = a.output_dir/'ocr-cache'
    cache.mkdir(exist_ok=True)
    recognizer = None
    rows = []
    try:
        for number, word in enumerate(selected, 1):
            sid, wi = word['sample_id'], word['gt_index']
            frozen = old[(sid, wi)]
            sample = samples[sid]
            image_path = a.data_root/sample.image_path
            image_sha = verify_file(image_path, source['protocol']['dataset_files'][str(sample.image_path)]['sha256'])
            with Image.open(image_path) as opened:
                rgb = np.asarray(opened.convert('RGB'))
            outputs = {}
            for kind, box in [('RECOVERED', word['oracle_union_padded_bbox']), ('GT_BOX', frozen['gt_bbox'])]:
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
                    raise ValueError('invalid crop bbox')
                request = {'sample_id': sid, 'image_sha256': image_sha, 'bbox': box}
                path = cache/(digest(request)+'.json')
                result = read_cached(path, contract, request)
                if result is None:
                    if recognizer is None:
                        if a.device.startswith('cuda'):
                            torch.cuda.set_device(torch.device(a.device))
                            torch.cuda.init()
                            torch.cuda.reset_peak_memory_stats(torch.device(a.device))
                        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                            runtime_source=a.parseq_source, checkpoint=a.parseq_checkpoint,
                            device=a.device, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                            runtime_source_archive=a.parseq_source_archive))
                    start = time.perf_counter()
                    result = recognizer.recognize(rgb[y1:y2, x1:x2], input_role='T3_localized_crop').to_dict()
                    result['seconds'] = time.perf_counter()-start
                    payload = {'contract': contract, 'request': request, 'result': result}
                    save_json(path, {'sha256': digest(payload), 'payload': payload})
                outputs[kind] = scores(result['raw_text'], word['gt_text'])
            rows.append({'sample_id': sid, 'gt_index': wi, 'gt_text': word['gt_text'],
                         'V1': scores(word['parseq_text'], word['gt_text']), **outputs})
            print(f'[recovered TRAIN PARSeq] {number}/80 sample={sid} word={wi}', flush=True)
        summary = {kind: {'words': len(rows),
                    'exact_casefold': sum(r[kind]['exact_casefold'] for r in rows),
                    'fuzzy_ge_0_8': sum(r[kind]['fuzzy_ge_0_8'] for r in rows)}
                   for kind in ('V1', 'RECOVERED', 'GT_BOX')}
        memory = None
        if recognizer is not None and a.device.startswith('cuda'):
            memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(a.device)}
        output = a.output_dir/'results.json'
        if output.exists():
            previous = json.loads(output.read_text())
            if previous['protocol'] != protocol or previous['summary'] != summary or previous['per_word'] != rows:
                raise RuntimeError('existing completed report differs; preserved without overwriting')
        else:
            save_json(output, {'protocol': protocol, 'summary': summary, 'per_word': rows,
                               'memory_this_execution': memory, 'all_ocr_cached': recognizer is None})
        print(json.dumps(summary, indent=2), flush=True)
        print('JSON', output, 'SHA256', sha(output), flush=True)
    except Exception as exc:
        save_json(a.output_dir/'last-error.json', {'type': type(exc).__name__, 'message': str(exc),
                                                 'completed_words_this_execution': len(rows)})
        raise
    finally:
        if recognizer is not None:
            recognizer.release()


if __name__ == '__main__':
    main()
