"""APPROXIMATE native Long-CLIP/LLM2CLIP routers on the frozen OCR population."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from recognition.clip_router_context_chunks import prompt_chunks
from recognition.clip_router_prompt_ablation import budget_accept, coverage_threshold, fit_reference, image_scores
from recognition.clip_specialized_runtime import EncoderBridge, file_sha
from scripts import run_clip_router_prompt_ablation as frozen
from scripts.crop_clip_runtime import sha, write_json
from scripts.run_clip_high_recall_admission import payload_digest, seal

REPO = frozen.REPO
CONFIG = REPO/'configs/approximate/clip_specialized_routers_v1.json'
PATHS = REPO/'configs/approximate/clip_specialized_routers_local_paths_v1.json'
CODE = (*frozen.CODE,'recognition/clip_router_context_chunks.py',
        'recognition/clip_specialized_runtime.py','scripts/clip_specialized_worker.py',
        'scripts/run_clip_specialized_routers.py')


def identity_for(name, cfg, paths):
    spec = copy.deepcopy(cfg['models'][name]); locations = paths['model_paths'][name]
    identity = {**spec,'seed':cfg['execution']['seed'],'cpu_threads':cfg['execution']['cpu_threads'],
                'batch_size':cfg['execution']['batch_size'],'file_sha256':{}}
    for component, info in spec['artifacts'].items():
        directory = Path(locations[component])
        if directory.name != info['revision']:
            raise ValueError('Artifact directory must name its full pinned revision')
        for file, digest in info['weight_sha256'].items():
            path = directory/file
            if file_sha(path) != digest:
                raise ValueError('Native checkpoint SHA256 mismatch: '+str(path))
            identity['file_sha256'][str(path)] = digest
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ('.json','.py','.txt','.model'):
                identity['file_sha256'][str(path)] = file_sha(path)
        identity[component+'_directory'] = str(directory)
    if spec['backend'] == 'longclip_native':
        identity['source_root'] = locations['source_root']
        identity['checkpoint'] = str(Path(identity['weights_directory'])/spec['weight_file'])
    else:
        identity['text_revision'] = spec['artifacts']['text']['revision']
        identity['mask_source'] = locations['mask_source']
    for component, files in spec['source_sha256'].items():
        base = Path(locations[component])
        for relative, digest in files.items():
            path = base/relative
            if file_sha(path) != digest:
                raise ValueError('Native source SHA256 mismatch: '+str(path))
            identity['file_sha256'][str(path)] = digest
    return identity


def run_model(args, name, cfg, paths, observed, branches, input_hashes, word_branches):
    identity = identity_for(name,cfg,paths)
    plan = {'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'result_class':'smoke-test result' if args.limit else 'approximate reproduction',
        'table_5_claim':False,'config':cfg,'config_sha256':sha(args.config),'model':identity,
        'source_sha256':input_hashes,'branch_metrics_sha256':cfg['branch_metrics_sha256'],
        'word_branch_sha256':cfg['word_branch_sha256'],'code_sha256':{p:sha(REPO/p) for p in CODE},
        'device':args.device,'limit':args.limit,'worker_python':paths['model_paths'][name]['python'],
        'parent_software':{'python':__import__('sys').version,'torch':torch.__version__,'numpy':np.__version__},
        'hardware':{'gpu':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None},
        'feature_processing_precision':'float32','deterministic_algorithms':True,'tf32':False,
        'selection_metric':'legacy_bbox_word_precision_on_TRAIN45',
        'primary_report_metric':'image_word_multiset_casefold_exact; punctuation retained; no bbox IoU'}
    import hashlib
    root = paths['output_root']/name/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    seal(root/'resolved_configuration.json',plan); digest = sha(root/'resolved_configuration.json')
    for relative, expected in plan['code_sha256'].items():
        content = (REPO/relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Experiment source changed during plan resolution')
        snapshot = root/'code_snapshot'/relative
        if snapshot.exists() and sha(snapshot) != expected:
            raise ValueError('Experiment source snapshot changed')
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():
            snapshot.write_bytes(content)
    fs, bridge, ids, worker_memory = {}, None, {}, None
    try:
        for split in ('train','test'):
            samples = observed[split][:args.limit] if args.limit else observed[split]
            ids[split] = [s['sample_id'] for s in samples]
            for index,sample in enumerate(samples,1):
                path = root/split/(sample['sample_id'].replace(':','__')+'.json')
                if path.exists():
                    feature = json.loads(path.read_text())
                    if feature['plan_sha256'] != digest or feature['payload_sha256'] != payload_digest(feature):
                        raise ValueError('Native feature cache changed')
                else:
                    if bridge is None:
                        bridge = EncoderBridge(identity,args.device,root/'bridge',plan['worker_python'])
                        write_json(root/'worker_qualification.json',bridge.ready)
                    start = time.monotonic()
                    old_chunks = frozen.prompt_chunks
                    try:
                        frozen.prompt_chunks = prompt_chunks
                        feature = frozen.features(bridge,cfg,sample)
                    finally:
                        frozen.prompt_chunks = old_chunks
                    feature.update(plan_sha256=digest,runtime={'seconds':time.monotonic()-start,
                        'device':args.device,'precision':identity['precision'],'batch_size':identity['batch_size'],
                        'image_shape':[3,identity['image_size'],identity['image_size']],
                        'models':[a['model_id'] for a in identity['artifacts'].values()],
                        'memory':bridge.memory,'out_of_memory':False})
                    feature['payload_sha256'] = payload_digest(feature); seal(path,feature)
                fs[sample['sample_id']] = feature
                if index % 10 == 0 or index == len(samples):
                    print(f'[{name} {split}] {index}/{len(samples)}',flush=True)
        partition = frozen.partition(ids['train'],cfg['calibration_images'],cfg['execution']['seed']) if not args.limit else {
            'calibration':ids['train'],'validation':ids['train']}
        reference = fit_reference(fs,partition['calibration'])
        scores = {sid:image_scores(f,reference) for sid,f in fs.items()}
        selections = {}
        for family in next(iter(scores.values())):
            values = {sid:s[family] for sid,s in scores.items()}
            cut = coverage_threshold(values,partition['calibration'],cfg['coverage'])
            rows = [r for r in branches['train'] if r['sample_id'] in partition['validation']]
            selections[family] = {'threshold':cut,'validation':frozen.evaluate(rows,values,cut,
                partition['validation'],sum(r['parseq']['gt_words'] for r in rows))}
        winner = max(selections,key=lambda f:(selections[f]['validation']['precision_retained_parseq'] or -1,
            selections[f]['validation']['words_retained_parseq'],f))
        selection = {'winner':winner,'families':selections,'reference':reference,
            'fit_ids':partition['calibration'],'selection_ids':partition['validation'],
            'selection_uses_test_labels':False}
        seal(root/'train_selection.json',selection)
        comparisons, geometric = {}, {}
        for family, selection in selections.items():
            comparisons[family], geometric[family] = {}, {}
            values = {sid:s[family] for sid,s in scores.items()}
            for split in ('train','test'):
                keep = budget_accept(values,ids[split],min(cfg['comparison_budget'][split],len(ids[split])))
                policies = {'train_threshold':(values,selection['threshold']),
                    'same_budget_descriptive':({sid:float(sid in keep) for sid in ids[split]},1.)}
                for target, source in [(comparisons,word_branches),(geometric,branches)]:
                    rows = [r for r in source[split] if r['sample_id'] in ids[split]]
                    gt = sum(r['parseq']['gt_words'] for r in rows)
                    target[family][split] = {p:frozen.evaluate(rows,v,t,ids[split],gt) for p,(v,t) in policies.items()}
        original = {}
        for split in ('train','test'):
            rows = [r for r in word_branches[split] if r['sample_id'] in ids[split]]
            old_scores = {r['sample_id']:r['score'] for r in branches[split]}
            original[split] = frozen.evaluate(rows,old_scores,cfg['original_threshold'],ids[split],
                sum(r['parseq']['gt_words'] for r in rows))
        worker_memory = {
            key: max((f['runtime']['memory'][key] for f in fs.values()
                      if f['runtime']['memory'][key] is not None), default=None)
            for key in ('peak_allocated_bytes','peak_reserved_bytes','peak_cpu_rss_bytes')}
        # A successful result also requires a successful native process exit.
        if bridge is not None:
            closing, bridge = bridge, None
            closing.close()
        result = {'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
            'result_class':plan['result_class'],'model':name,'winner_train':winner,
            'selection_uses_test_labels':False,'table_5_claim':False,'split_ids':ids,
            'plan_sha256':digest,'train_selection_sha256':sha(root/'train_selection.json'),
            'comparisons':comparisons,'geometric_comparisons':geometric,'original_router':original,
            'metric_protocol':plan['primary_report_metric'],'same_budget_test_is_descriptive':True,
            'elapsed_feature_seconds':sum(f['runtime']['seconds'] for f in fs.values()),
            'memory':worker_memory}
        seal(root/'results.json',result)
        write_json(paths['output_root']/name/'latest.json',{'results_path':str(root/'results.json'),
            'results_sha256':sha(root/'results.json')})
        print(json.dumps({'model':name,'results_path':str(root/'results.json'),'winner_train':winner}),flush=True)
        return result
    except Exception as error:
        write_json(root/'failure.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
            'memory':bridge.memory if bridge else worker_memory})
        raise
    finally:
        if bridge is not None:
            bridge.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,default=CONFIG); p.add_argument('--paths',type=Path,default=PATHS)
    p.add_argument('--models',nargs='+',default=['longclip_b','longclip_l','llm2clip_l14_336'])
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--check',action='store_true'); p.add_argument('--limit',type=int)
    p.add_argument('--output-root',type=Path)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    cfg, paths, observed, branches, hashes = frozen.load_protocol(args)
    if cfg['ambiguity_record'] != 'CLIP-SPECIALIZED-ROUTERS-001':
        raise ValueError('Explicit specialized approximate configuration required')
    word_path = Path(paths['word_branches'])
    if sha(word_path) != cfg['word_branch_sha256']:
        raise ValueError('Frozen word-evaluation source changed')
    word = json.loads(word_path.read_text())
    for path,digest in word['input_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Word-evaluation dependency changed: '+path)
    for split in ('train','test'):
        frozen.validate_population(word['splits'][split],[s['sample_id'] for s in observed[split]],cfg['gt_words'][split])
    random.seed(cfg['execution']['seed']); np.random.seed(cfg['execution']['seed'])
    if args.check:
        print(json.dumps({'status':'SOURCE_PREFLIGHT_PASS','cuda_available':torch.cuda.is_available(),
            'split_images':{s:len(v) for s,v in observed.items()},'new_inference_executed':False,
            'selected_models':args.models}),flush=True)
        return
    for name in args.models:
        run_model(args,name,cfg,paths,observed,branches,hashes,word['splits'])


if __name__ == '__main__':
    main()
