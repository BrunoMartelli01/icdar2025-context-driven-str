"""Frozen-map contour/seeded/refined crop comparison with real, cached PARSeq."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from localize.word_guided_contours import METHODS, word_contour_crops
from models.word_multitask import VARIANTS
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.compare_word_multitask import compare
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct, _geometry_match, _train_gt
from scripts.evaluate_totaltext_word_kernel_parseq_val import _ocr_text, _save_labels
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.training import seed_everything
from segmentation.word_multitask_evaluation import aggregate_word_metrics, render_word_maps

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'configs/approximate/word_contour_refinement_v1.json'


def write_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing artifact differs; choose a NEW output directory: ' + str(path))
    else:
        _atomic_json(path, value)


def load_config(path, mode):
    path = path.resolve()
    if mode != 'approximate' or (REPO/'configs/approximate').resolve() not in path.parents:
        raise ValueError('Explicit APPROXIMATE config required; no namespace fallback')
    cfg = json.loads(path.read_text())
    if (cfg['reproduction_mode'] != 'approximate' or cfg['result_class'] != 'approximate reproduction'
            or cfg['variants'] != list(VARIANTS) or cfg['methods'] != list(METHODS)
            or cfg['recognition']['decode_ar'] is not True or cfg['recognition']['refine_iters'] != 1
            or cfg['recognition']['casefold'] is not True
            or cfg['recognition']['symbol_normalized_secondary_metric'] is not True
            or type(cfg['recognition']['batch_size']) is not int or cfg['recognition']['batch_size'] < 1):
        raise ValueError('Invalid frozen crop/PARSeq comparison config')
    return cfg


def score_predictions(predictions, gt, *, text_available):
    counts = {'gt_words': len(gt), 'proposals': len(predictions),
              'geometry_iou_ge_50': len(_geometry_match(predictions, gt, .5))}
    if text_available:
        counts['exact_casefold_iou_ge_50'] = len(_correct(predictions, gt, normalize=False))
        counts['exact_no_symbols_iou_ge_50'] = len(_correct(predictions, gt, normalize=True))
    return counts


def ratios(counts):
    result = dict(counts)
    words, proposals = counts['gt_words'], counts['proposals']
    for key, prefix in [('geometry_iou_ge_50', 'geometry'), ('exact_casefold_iou_ge_50', 'exact_casefold'),
                        ('exact_no_symbols_iou_ge_50', 'exact_no_symbols')]:
        if key in counts:
            matches = counts[key]
            result[prefix+'_precision'] = matches/proposals if proposals else 0.
            result[prefix+'_recall_all_gt'] = matches/words if words else 0.
            result[prefix+'_f1'] = 2*matches/(words+proposals) if words+proposals else 0.
    if 'exact_casefold_iou_ge_50' in counts:
        matched = counts['geometry_iou_ge_50']
        result['exact_casefold_to_geometry_match_ratio'] = counts['exact_casefold_iou_ge_50']/matched if matched else 0.
        result['conditional_ratio_limitation'] = 'Maximum geometry and text-constrained matchings can differ; not a fixed-pair recognizer accuracy.'
    return result


def load_sources(run_root, cfg):
    compare(run_root, [])  # Success, checksum and complete matched-contract checks.
    sources = {}
    common = None
    for variant in VARIANTS:
        folder = run_root/variant
        report = json.loads((folder/'run_manifest.json').read_text())
        validation = json.loads((folder/'validation.json').read_text())
        if (validation['run_contract'] != report['run_contract']
                or validation['checkpoint_sha256'] != report['checkpoint']['sha256']
                or validation['aggregate'] != report['validation_word_geometry']):
            raise ValueError('Saved validation provenance differs: ' + variant)
        contract = report['run_contract']
        if contract['reproduction_mode'] != 'approximate' or report['result_class'] != 'approximate reproduction':
            raise ValueError('Only approximate multi-task sources are supported')
        for name, digest in contract['source_sha256'].items():
            if sha256_file(REPO/name) != digest:
                raise ValueError('Frozen scientific source changed: ' + name)
        ids = [row['sample_id'] for row in validation['per_image']]
        if common is None:
            common = ids
        if ids != common or len(ids) != len(set(ids)) or len(ids) != contract['config']['validation_count']:
            raise ValueError('Incomplete or inconsistent frozen validation sample set')
        if validation['aggregate']['gt_words'] != cfg['metric_protocol']['expected_care_words']:
            raise ValueError('Expected care-word denominator differs')
        for row in validation['per_image']:
            directory = folder/'validation_exports'/row['sample_id'].replace(':','__')
            for name, digest in row['exports'].items():
                if sha256_file(directory/name) != digest:
                    raise ValueError('Saved export checksum differs: ' + str(directory/name))
        sources[variant] = {'folder': folder, 'report': report, 'validation': validation}
    return sources


def geometry_protocol(args, cfg, sources):
    modules = ['scripts/evaluate_word_multitask_crops.py', 'localize/word_guided_contours.py',
               'localize/repository_contours.py', 'localize/word_multitask.py', 'localize/crop_recovery.py',
               'segmentation/totaltext_word_kernel.py', 'segmentation/word_multitask_evaluation.py',
               'segmentation/totaltext_image_frequency_augment.py', 'recognition/parseq.py',
               'recognition/scene_text_v1_evaluation.py', 'scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py',
               'scripts/evaluate_totaltext_highres_contour_padding.py', 'scripts/rescore_totaltext_highres_padding_symbols.py']
    return {'schema_version': 1, 'reproduction_mode': 'approximate', 'result_class': 'approximate reproduction',
            'config': cfg, 'config_sha256': sha256_file(args.config.resolve()),
            'source_seed': sources['text']['report']['run_contract']['config']['seed'],
            'run_root': str(args.run_root.resolve()), 'source_sha256': {name: sha256_file(REPO/name) for name in modules},
            'source_reports': {name: {'run_manifest_sha256': sha256_file(value['folder']/'run_manifest.json'),
                'validation_sha256': sha256_file(value['folder']/'validation.json')} for name,value in sources.items()},
            'opencv': __import__('cv2').__version__, 'numpy': np.__version__}


def verify_geometry_exports(result, output):
    checked = {}
    for branch in result['branches'].values():
        for method in branch.values():
            for row in method['per_image']:
                for filename, expected in [(row['image_path'],row['image_sha256']),
                                           (row['annotation_path'],row['annotation_sha256'])]:
                    if filename not in checked: checked[filename] = sha256_file(Path(filename))
                    if checked[filename] != expected: raise ValueError('Original image/annotation changed: '+filename)
                for filename, digest in row['exports'].items():
                    if sha256_file(output/filename) != digest:
                        raise ValueError('Completed crop export checksum differs: ' + filename)


def build_geometry(args, cfg, sources, protocol):
    output = args.output_root.resolve()
    target_root = Path(sources['text']['report']['run_contract']['runtime_inputs']['target_root'])
    source_cfg = sources['text']['report']['run_contract']['config']
    manifest_path = target_root/source_cfg['train_manifest']
    if sha256_file(manifest_path) != source_cfg['train_manifest_sha256']:
        raise ValueError('Qualified target manifest changed')
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != 'total_text' or manifest.split_role != 'train':
        raise ValueError('Only frozen TRAIN validation is authorized by this protocol')
    _, indices, fingerprint = _split_indices(manifest, source_cfg['seed'], source_cfg['validation_count'])
    sample_ids = [manifest.samples[index].sample_id for index in indices]
    if fingerprint != source_cfg['validation_ids_sha256']:
        raise ValueError('Frozen TRAIN holdout fingerprint differs')
    cached = output/'geometry_results.json'
    if cached.exists():
        result = json.loads(cached.read_text())
        if result['protocol'] != protocol or result['status'] != 'SUCCESS':
            raise ValueError('Existing geometry result contract differs')
        verify_geometry_exports(result, output)
        print('[reuse geometry] '+str(cached), flush=True)
        return result
    by_id = {sample.sample_id:sample for sample in manifest.samples}
    branches = {}
    for variant, source in sources.items():
        saved_rows = source['validation']['per_image']
        if [row['sample_id'] for row in saved_rows] != sample_ids:
            raise ValueError('Saved map order differs from frozen manifest split')
        methods = ['findcontours'] if variant == 'text' else cfg['methods']
        branch = {method: {'per_image': [], 'native_totals': Counter(), 'original_totals': Counter()} for method in methods}
        post = source['report']['run_contract']['config']['postprocessing']
        if post['padding'] != 0 or post['min_area'] != 0 or post['max_instances'] is not None:
            raise ValueError('Matched v1 crop comparison requires existing pad0/no-filter/no-cap settings')
        for position,saved in enumerate(saved_rows,1):
            sample_id = saved['sample_id']; key = sample_id.replace(':','__')
            directory = source['folder']/'validation_exports'/key
            exported = json.loads((directory/'proposals.json').read_text())
            image_path = Path(exported['image_path'])
            if sha256_file(image_path) != exported['image_sha256'] or exported['image_sha256'] != by_id[sample_id].image_sha256:
                raise ValueError('Original RGB checksum differs')
            with Image.open(image_path) as image:
                rgb = np.ascontiguousarray(np.asarray(image.convert('RGB')))
            with np.load(directory/'maps.npz',allow_pickle=False) as maps:
                probabilities, canvas = maps['probabilities'], maps['canvas']
                if probabilities.shape != (VARIANTS[variant],256,256):
                    raise ValueError('Saved map shape/head count differs')
                for method, part in branch.items():
                    proposals, labels, audit = word_contour_crops(probabilities, canvas, (rgb.shape[1],rgb.shape[0]),
                        post, cfg['refinement'], method=method, cached_labels=maps['labels'] if method != 'findcontours' else None)
                    # Freeze GT-blind crop recipes before reading/scoring care annotations.
                    for proposal in proposals:
                        x1,y1,x2,y2 = proposal['bbox']
                        proposal['crop_rgb_sha256'] = crop_array_hash(rgb[y1:y2,x1:x2])
                    base = Path(variant)/method/'samples'/key
                    label_path = output/base/'instances.png'
                    _save_labels(label_path,labels)
                    recipe = {'sample_id':sample_id,'result_class':'approximate reproduction',
                              'image_path':str(image_path),'image_sha256':exported['image_sha256'],
                              'source_maps_sha256':saved['exports']['maps.npz'],'proposals':proposals,'audit':audit}
                    write_json(output/base/'proposals.json',recipe)
                    visual = output/base/'predictions.png'
                    if not visual.exists(): render_word_maps(Image.fromarray(rgb),probabilities,proposals,visual)
                    ignored = [int(label) for label in np.unique(labels[labels>0])
                               if not np.any((labels == label) & maps['validity'])]
                    care_labels = labels.copy();care_labels[np.isin(care_labels,ignored)]=0
                    native = instance_geometry(care_labels,maps['owners'],maps['expected_owner_ids'])
                    if method == 'seeded' and any(native[k] != saved[k] for k in native):
                        raise RuntimeError('Cached seeded geometry changed')
                    gt, annotation_sha = _train_gt(by_id[sample_id],target_root,(rgb.shape[1],rgb.shape[0]))
                    care = [row for row in proposals if row['instance_id'] not in ignored]
                    original = score_predictions(care,gt,text_available=False)
                    files = {str(base/name):sha256_file(output/base/name) for name in ['instances.png','proposals.json','predictions.png']}
                    part['per_image'].append({**recipe,'ignored_prediction_ids':ignored,'ground_truth':gt,
                        'annotation_path':str(target_root/'datasets/total_text/train/annotations'/
                                              (sample_id.split(':',1)[1]+'.txt')),
                        'annotation_sha256':annotation_sha,'native_geometry':native,'original_geometry':original,'exports':files})
                    part['native_totals'].update(native);part['original_totals'].update(original)
            if position % 25 == 0: print(f'[crops {variant}] {position}/{len(saved_rows)}',flush=True)
        for method, part in branch.items():
            if part['original_totals']['gt_words'] != cfg['metric_protocol']['expected_care_words']:
                raise RuntimeError('Care-word GT denominator changed')
            part['native'] = aggregate_word_metrics(dict(part.pop('native_totals')))
            part['original'] = ratios(dict(part.pop('original_totals')))
        branches[variant] = branch
    result = {'status':'SUCCESS','result_class':'approximate reproduction','reproduction_mode':'approximate',
              'protocol':protocol,'branches':branches,'segmentation_inference':'none; frozen maps reused'}
    write_json(cached,result)
    return result


def validate_ocr(record, crop_sha):
    _ocr_text(record,crop_sha)
    if (record.get('decode_ar',True) is not True or record.get('refine_iters',1) != 1
            or record.get('predicted_length') != len(record['raw_text'])):
        raise ValueError('OCR decoding cache differs from pinned runtime defaults')
    return record['raw_text']


def recognize(args, cfg, geometry):
    if any(value is None for value in [args.parseq_source,args.parseq_source_archive,args.parseq_checkpoint]):
        raise ValueError('Provide the three qualified local PARSeq paths; no download')
    if sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256 or sha256_file(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError('Pinned PARSeq artifacts changed')
    output = args.output_root.resolve()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly select --device cpu or run on the GPU host')
    seed_everything(geometry['protocol']['source_seed'], deterministic_algorithms=True)
    origins = []
    for directory in args.reuse_ocr:
        provider = directory.resolve().parent/'protocol.json'
        if not directory.is_dir() or not provider.is_file():
            raise ValueError('Explicit OCR reuse requires a provider protocol.json: '+str(directory))
        meta = json.loads(provider.read_text())
        if meta.get('parseq_checkpoint_sha256') != PARSEQ_CHECKPOINT_SHA256 or meta.get('parseq_runtime_archive_sha256') != PARSEQ_RUNTIME_ARCHIVE_SHA256:
            raise ValueError('OCR provider uses different pinned artifacts')
        origins.append({'directory':str(directory.resolve()),'protocol_sha256':sha256_file(provider),
                        'device':meta.get('device'),'batch_size':meta.get('parseq_batch_size'),
                        'legacy_decoding_evidence':'Pinned official runtime defaults AR=True/refine1, verified by existing adapter; ASSUMPTION for legacy missing fields'})
    protocol = {'schema_version':1,'result_class':'approximate reproduction','reproduction_mode':'approximate',
        'geometry_results_sha256':sha256_file(output/'geometry_results.json'),'recognition':cfg['recognition'],
        'parseq_checkpoint_sha256':PARSEQ_CHECKPOINT_SHA256,'parseq_runtime_archive_sha256':PARSEQ_RUNTIME_ARCHIVE_SHA256,
        'paths':{name:str(getattr(args,name).resolve()) for name in ['parseq_source','parseq_source_archive','parseq_checkpoint']},
        'device':args.device,'cpu_threads':args.cpu_threads,'reuse_ocr':origins,
        'environment':experiment_environment(device),'cache_qualification':'first16_reused_RGB_hashes_lexicographic_exact_text_replay'}
    write_json(output/'recognition_protocol.json',protocol)
    completed = output/'recognition_results.json'
    if completed.exists():
        result = json.loads(completed.read_text())
        if result['protocol'] != protocol or result['status'] != 'SUCCESS': raise ValueError('Recognition result provenance changed')
        for key,digest in result['ocr_records_sha256'].items():
            if sha256_file(output/'ocr'/f'{key}.json') != digest: raise ValueError('Completed OCR cache changed')
        print('[reuse recognition] '+str(completed),flush=True)
        return result
    crops = {}
    for branch in geometry['branches'].values():
        for part in branch.values():
            for row in part['per_image']:
                for proposal in row['proposals']:
                    crops.setdefault(proposal['crop_rgb_sha256'],(row['image_path'],proposal['bbox']))
    records, pending, reused = {}, [], []
    for key in sorted(crops):
        own = output/'ocr'/f'{key}.json'
        if own.exists():
            records[key]=json.loads(own.read_text());validate_ocr(records[key],key)
            if 'reuse' in records[key]: reused.append(key)
            continue
        candidates=[]
        for origin in origins:
            path=Path(origin['directory'])/f'{key}.json'
            if path.exists():
                value=json.loads(path.read_text());validate_ocr(value,key)
                candidates.append((value,path,origin))
        if candidates:
            if len({value['raw_text'] for value,_,_ in candidates}) != 1:
                raise ValueError('Conflicting identical-crop cached transcriptions: '+key)
            value,path,origin=candidates[0]
            value={**value,'reuse':{'path':str(path),'record_sha256':sha256_file(path),'provider':origin}}
            write_json(own,value);records[key]=value;reused.append(key)
        else: pending.append(key)
    def batch_images(keys):
        images={};batch=[]
        for key in keys:
            filename,box=crops[key]
            if filename not in images:
                with Image.open(filename) as image: images[filename]=np.asarray(image.convert('RGB'))
            x1,y1,x2,y2=box
            crop=np.ascontiguousarray(images[filename][y1:y2,x1:x2])
            if crop_array_hash(crop)!=key: raise ValueError('Frozen RGB crop bytes changed')
            batch.append(crop)
        return batch
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    memory={'model_set':'PARSeq only; segmentation maps cached','device':str(device),'dtype':'float32',
            'batch_size':cfg['recognition']['batch_size'],'input_shape':[cfg['recognition']['batch_size'],3,32,128],
            'gpu_peak_allocated_bytes':None,'gpu_peak_reserved_bytes':None,'oom':False}
    recognizer=None;start=time.perf_counter();qualified=[]
    try:
        if pending or reused:
            recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
                checkpoint=args.parseq_checkpoint,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=args.parseq_source_archive,device=args.device,
                decode_ar=True,refine_iters=1))
        for first in range(0,min(16,len(reused)),cfg['recognition']['batch_size']):
            keys=reused[first:min(first+cfg['recognition']['batch_size'],16)]
            values=recognizer.recognize_candidate_batch(batch_images(keys))
            if len(values)!=len(keys): raise RuntimeError('PARSeq cache qualification output count differs')
            for key,value in zip(keys,values):
                if value.raw_text!=records[key]['raw_text']:
                    raise RuntimeError('Cross-device OCR cache text mismatch; use a NEW output with no --reuse-ocr: '+key)
                qualified.append(key)
        for first in range(0,len(pending),cfg['recognition']['batch_size']):
            keys=pending[first:first+cfg['recognition']['batch_size']]
            values=recognizer.recognize_candidate_batch(batch_images(keys))
            if len(values)!=len(keys): raise RuntimeError('PARSeq batch output count differs')
            for key,value in zip(keys,values):
                record={'schema_version':1,'crop_rgb_sha256':key,'parseq_checkpoint_sha256':PARSEQ_CHECKPOINT_SHA256,
                        'parseq_source_archive_sha256':PARSEQ_RUNTIME_ARCHIVE_SHA256,'decode_ar':True,'refine_iters':1,
                        'execution':{'device':str(device),'dtype':'float32','batch_size':cfg['recognition']['batch_size']},
                        **value.to_dict()}
                validate_ocr(record,key);write_json(output/'ocr'/f'{key}.json',record);records[key]=record
            if first//cfg['recognition']['batch_size'] % 25 == 0:
                print(f'[PARSeq] {first+len(keys)}/{len(pending)} new crops; {len(reused)} reused',flush=True)
    finally:
        if device.type == 'cuda':
            memory.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if recognizer is not None: recognizer.release()
    branches={}
    for variant,branch in geometry['branches'].items():
        branches[variant]={}
        for method,part in branch.items():
            totals=Counter();rows=[]
            for row in part['per_image']:
                ignored=set(row['ignored_prediction_ids'])
                all_predictions=[{**proposal,'text':records[proposal['crop_rgb_sha256']]['raw_text']} for proposal in row['proposals']]
                care=[value for value in all_predictions if value['instance_id'] not in ignored]
                counts=score_predictions(care,row['ground_truth'],text_available=True);totals.update(counts)
                rows.append({'sample_id':row['sample_id'],'ground_truth':row['ground_truth'],
                    'predictions_all':all_predictions,'ignored_prediction_ids':sorted(ignored),'counts':counts,
                    'correct_casefold_gt_indices':sorted(_correct(care,row['ground_truth'],normalize=False))})
            branches[variant][method]={'aggregate':ratios(dict(totals)),'native':part['native'],'per_image':rows}
    result={'status':'SUCCESS','result_class':'approximate reproduction','reproduction_mode':'approximate',
            'protocol':protocol,'branches':branches,'memory':memory,'elapsed_seconds':time.perf_counter()-start,
            'unique_rgb_crops':len(crops),'reused_ocr_records':len(reused),'new_ocr_records':len(pending),
            'cache_replay_qualification':{'checked':len(qualified),'matched':len(qualified),'crop_sha256':qualified},
            'ocr_records_sha256':{key:sha256_file(output/'ocr'/f'{key}.json') for key in sorted(records)}}
    write_json(completed,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--action',choices=['preflight','geometry','recognize','all'],required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--cpu-threads',type=int,default=2)
    for name in ['parseq-source','parseq-source-archive','parseq-checkpoint']:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--reuse-ocr',type=Path,action='append',default=[])
    args=parser.parse_args()
    cfg=load_config(args.config,args.mode)
    output=args.output_root.resolve();run_root=args.run_root.resolve()
    if args.cpu_threads<1: raise ValueError('CPU thread count must be positive')
    if output==run_root or run_root in output.parents or output in run_root.parents or output==REPO:
        raise ValueError('Use an isolated new output directory outside original experiment results')
    torch.set_num_threads(args.cpu_threads)
    sources=load_sources(run_root,cfg)
    protocol=geometry_protocol(args,cfg,sources)
    if args.action=='preflight':
        print(json.dumps({'status':'PREFLIGHT_PASS','variants':list(sources),'images':125,'expected_care_words':983}));return 0
    target_root=Path(sources['text']['report']['run_contract']['runtime_inputs']['target_root']).resolve()
    if output==target_root or target_root in output.parents or output in target_root.parents:
        raise ValueError('Output must not overlap the qualified dataset')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'runner.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another crop evaluation holds this output lock') from error
        write_json(output/'geometry_protocol.json',protocol)
        try:
            geometry=build_geometry(args,cfg,sources,protocol)
            result=recognize(args,cfg,geometry) if args.action in ('recognize','all') else geometry
        except Exception as error:
            _atomic_json(output/'failure.json',{'status':'FAILED','action':args.action,'result_class':'approximate reproduction',
                'type':type(error).__name__,'message':str(error),'oom':isinstance(error,torch.cuda.OutOfMemoryError),
                'device':args.device,'dtype':'float32','model_set':'PARSeq only; frozen segmentation maps',
                'batch_size':cfg['recognition']['batch_size'],'input_shape':[cfg['recognition']['batch_size'],3,32,128],
                'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None})
            raise
    for variant,branch in result['branches'].items():
        for method,part in branch.items():
            metrics=part.get('aggregate',part.get('original'))
            print(variant,method,json.dumps(metrics),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
