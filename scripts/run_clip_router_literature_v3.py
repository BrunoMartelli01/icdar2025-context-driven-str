"""Isolated, pinned CLIP-family tests motivated by scene-text literature.

Reuses the frozen v2 scoring/evaluation without editing its historical code.
New checkpoints are downloaded only with --download; no dependencies installed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import re

from recognition.clip_literature_runtime import LiteratureEncoder
from recognition.clip_router_context_chunks import POLICY, prompt_chunks
from recognition.clip_router_prompt_ablation import image_scores
from recognition.clip_router_threshold_sweep import curves
from scripts import run_clip_router_prompt_ablation as frozen
from scripts.crop_clip_runtime import sha, write_json
from scripts.run_clip_high_recall_admission import payload_digest, seal

REPO=frozen.REPO
CONFIG=REPO/'configs/approximate/clip_router_literature_v3.json'
PATHS=REPO/'configs/approximate/clip_router_literature_local_paths_v3.json'
RECEIPT='pinned_lfs_receipt.json'
EXTRA_CODE=('recognition/clip_literature_runtime.py','scripts/run_clip_router_literature_v3.py',
            'recognition/clip_router_context_chunks.py')
FROZEN_IDENTITY=frozen.model_identity
FROZEN_DOWNLOAD=frozen.download_model


def metadata_receipt(spec, info):
    if info.sha!=spec['revision']:
        raise ValueError('Hub returned a different revision')
    files=[s for s in info.siblings if s.rfilename==spec['weight_file']]
    if len(files)!=1 or files[0].lfs is None:
        raise ValueError('Authoritative LFS checksum unavailable for pinned weight file')
    lfs=files[0].lfs
    digest=lfs['sha256'] if isinstance(lfs,dict) else lfs.sha256
    if not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Invalid published LFS checksum')
    return {'schema_version':1,'model_id':spec['model_id'],'revision':spec['revision'],
        'weight_file':spec['weight_file'],'published_weight_sha256':digest,
        'evidence':'Hugging Face model_info(files_metadata=True) at immutable revision',
        'source_url':f"https://huggingface.co/{spec['model_id']}/tree/{spec['revision']}"}


def published_receipt(spec):
    from huggingface_hub import HfApi
    info=HfApi().model_info(spec['model_id'],revision=spec['revision'],files_metadata=True)
    return metadata_receipt(spec,info)


def download_model(spec, root):
    # Resolve the published checksum before downloading/executing the weights.
    receipt=published_receipt(spec) if spec['weight_sha256'] is None else None
    directory=FROZEN_DOWNLOAD(spec,root)
    digest=spec['weight_sha256'] if receipt is None else receipt['published_weight_sha256']
    if sha(directory/spec['weight_file'])!=digest:
        raise ValueError('Downloaded checkpoint differs from published pinned SHA256')
    if receipt is not None:seal(directory/RECEIPT,receipt)
    return directory


def model_identity(spec, directory):
    resolved=copy.deepcopy(spec)
    if resolved['weight_sha256'] is None:
        path=directory/RECEIPT
        if not path.exists():
            raise ValueError('Pinned LFS receipt missing: run this v3 command with --download')
        receipt=json.loads(path.read_text())
        if any(receipt[k]!=spec[k] for k in ('model_id','revision','weight_file')):
            raise ValueError('Published-checksum receipt belongs to different weights')
        digest=receipt['published_weight_sha256']
        if not re.fullmatch('[0-9a-f]{64}',digest):raise ValueError('Invalid pinned checksum receipt')
        resolved['weight_sha256']=digest
        resolved['checksum_receipt_sha256']=sha(path)
    return FROZEN_IDENTITY(resolved,directory)


def protocol(args):
    cfg,paths,observed,branches,hashes=frozen.load_protocol(args)
    if (cfg.get('schema_version')!=3 or cfg.get('ambiguity_record')!='CLIP-ROUTER-LITERATURE-003'
            or cfg['execution']['batch_size']!=2 or cfg['execution']['precision']!='float32'
            or cfg.get('prompt_chunking')!=POLICY):
        raise ValueError('Explicit isolated v3 protocol required')
    grid=cfg['report_thresholds']
    if (not grid or grid!=sorted(set(grid))
            or any(not math.isfinite(t) or not 0<=t<=1 for t in grid)):
        raise ValueError('Explicit finite percentile-score threshold grid required')
    for name in args.models:
        spec=cfg['models'][name]
        if (not re.fullmatch('[0-9a-f]{40}',spec['revision'])
                or Path(spec['weight_file']).name!=spec['weight_file']
                or spec['backend'] not in ('clip','siglip')
                or spec['text_preprocessing']!=('lowercase_before_tokenizer'
                    if spec['backend']=='siglip' else 'native_tokenizer')
                or spec['weight_sha256'] is not None and not re.fullmatch('[0-9a-f]{64}',spec['weight_sha256'])):
            raise ValueError('Invalid pinned model spec: '+name)
    return cfg,paths,observed,branches,hashes


def threshold_reports(output, outputs, branches):
    """Exploratory score curves for TRAIN-chosen families; no TEST selection."""
    report={'schema_version':3,'reproduction_mode':'approximate',
        'claim_label':'approximate reproduction','table_5_claim':False,
        'threshold_selected_on_test':False,'curve_is_exploratory':True,'models':{}}
    table=[]
    for entry in outputs:
        result_path=Path(entry['results']);root=result_path.parent
        result=json.loads(result_path.read_text())
        plan_path=root/'resolved_configuration.json';selection_path=root/'train_selection.json'
        if (result['result_class']!='approximate reproduction'
                or sha(plan_path)!=result['plan_sha256']
                or sha(selection_path)!=result['train_selection_sha256']):
            raise ValueError('Threshold curves require complete sealed results')
        cfg=json.loads(plan_path.read_text())['config']
        selection=json.loads(selection_path.read_text());family=selection['winner']
        model={'results_sha256':sha(result_path),'family_train':family,
            'threshold_train':selection['families'][family]['threshold'],'splits':{}}
        for split,ids in result['split_ids'].items():
            scores={}
            for sid in ids:
                feature=json.loads((root/split/(sid.replace(':','__')+'.json')).read_text())
                if (feature['payload_sha256']!=payload_digest(feature)
                        or feature['plan_sha256']!=result['plan_sha256'] or feature['sample_id']!=sid):
                    raise ValueError('Threshold feature cache changed')
                scores[sid]=image_scores(feature,selection['reference'])[family]
            replay=[{**r,'score':scores[r['sample_id']]} for r in branches[split]]
            cut=[] if model['threshold_train'] is None else [model['threshold_train']]
            grid=sorted(set(cfg['report_thresholds']+cut))
            model['splits'][split]=curves(replay,grid,ids,cfg['gt_words'][split])
            for p in model['splits'][split]['grid']:
                m=p['combined_model']
                table.append({'model':entry['model'],'family_train':family,'split':split,
                    'threshold':p['threshold'],'parseq_images':p['parseq_images'],
                    'deepsolo_images':p['deepsolo_images'],'parseq_words':p['words_retained_parseq'],
                    'parseq_precision':p['precision_retained_parseq'],
                    'deepsolo_words':p['words_from_deepsolo'],
                    'final_words':m['detected_words'] if m else None,
                    'final_tp':m['tp'] if m else None,'final_fp':m['fp'] if m else None,
                    'final_fn':m['fn'] if m else None,
                    'final_precision':m['accuracy_on_detected_words'] if m else None,
                    'final_accuracy_tn0':m['accuracy_tn0'] if m else None,
                    'final_recall':m['recall'] if m else None,'final_f1':m['word_f1'] if m else None,
                    'missing_deepsolo_images':len(p['missing_deepsolo_sample_ids'])})
        report['models'][entry['model']]=model
    write_json(output/'thresholds.json',report)
    with (output/'thresholds.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]))
        writer.writeheader();writer.writerows(table)
    return report


def compare_original(output, outputs):
    """Report the unchanged old gate alongside the new TRAIN-selected families."""
    # Caller creates the regular summary with the supplied path profile first.
    loaded=[json.loads(Path(o['results']).read_text()) for o in outputs]
    if any(r['original_router']!=loaded[0]['original_router'] for r in loaded):
        raise ValueError('Frozen reference router differs across experiments')
    plan_path=Path(outputs[0]['results']).parent/'resolved_configuration.json'
    reference_cut=json.loads(plan_path.read_text())['config']['original_threshold']
    baseline={'status':'SUCCESS','claim_label':'approximate reproduction',
        'reproduction_mode':'approximate','table_5_claim':False,
        'source':'Frozen B16 whole-image router','original_threshold':reference_cut,
        'splits':loaded[0]['original_router']}
    write_json(output/'original_router.json',baseline)
    for split,p in baseline['splits'].items():
        print(json.dumps({'reference':True,'split':split,'parseq_images':p['parseq_images'],
            'words':p['combined_model']['detected_words'],'precision':p['combined_model']['accuracy_on_detected_words'],
            'accuracy_tn0':p['combined_model']['accuracy_tn0']}),flush=True)
    for r in loaded:
        for split,v in r['comparisons'][r['winner_train']].items():
            p=v['same_budget_descriptive'];m=p['combined_model']
            print(json.dumps({'model':r['model'],'family_train':r['winner_train'],'split':split,
                'parseq_images':p['parseq_images'],'parseq_words':p['words_retained_parseq'],
                'parseq_precision':p['precision_retained_parseq'],'final_words':m['detected_words'] if m else None,
                'final_precision':m['accuracy_on_detected_words'] if m else None,
                'final_accuracy_tn0':m['accuracy_tn0'] if m else None,
                'missing_fallback_images':len(p['missing_deepsolo_sample_ids'])}),flush=True)


def run(args):
    cfg,paths,observed,branches,hashes=protocol(args)
    status=[]
    for name in args.models:
        spec=cfg['models'][name]
        directory=Path(paths['models'].get(name,paths['download_root']/spec['revision'])).resolve()
        weight=directory/spec['weight_file']
        ready=weight.exists() and (spec['weight_sha256'] is not None or (directory/RECEIPT).exists())
        if args.download and weight.exists() and not ready:
            receipt=published_receipt(spec)
            if sha(weight)!=receipt['published_weight_sha256']:
                raise ValueError('Existing weights differ from published pinned SHA256')
            seal(directory/RECEIPT,receipt);ready=True
        if ready:model_identity(spec,directory)
        status.append({'model':name,'ready':ready,'directory':str(directory),
            'published_revision':spec['revision'],'image_size':spec['image_size']})
    if args.check:
        print(json.dumps({'status':'PREFLIGHT_PASS','models':status,
            'split_images':{s:len(v) for s,v in observed.items()},
            'cuda_available':__import__('torch').cuda.is_available(),
            'source_integrity_checked':True,'new_inference_executed':False}),flush=True)
        return status
    # Configure callbacks for this isolated process only. Historical files stay immutable.
    old=(frozen.Encoder,frozen.model_identity,frozen.download_model,frozen.CODE,frozen.prompt_chunks)
    try:
        frozen.Encoder=LiteratureEncoder;frozen.model_identity=model_identity
        frozen.download_model=download_model;frozen.CODE=(*frozen.CODE,*EXTRA_CODE)
        frozen.prompt_chunks=prompt_chunks
        outputs=frozen.run(args)
    finally:
        frozen.Encoder,frozen.model_identity,frozen.download_model,frozen.CODE,frozen.prompt_chunks=old
    if not args.limit:
        from scripts.summarize_clip_router_prompt_ablation import summary
        out=paths['output_root']/'summary'
        profile=json.loads(args.paths.read_text());profile['output_root']=str(paths['output_root'])
        summary_paths=out/'resolved_paths.json';write_json(summary_paths,profile)
        summary(summary_paths,args.models,out);compare_original(out,outputs)
        threshold_reports(out,outputs,branches)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--paths',type=Path,default=PATHS)
    p.add_argument('--models',nargs='+',default=['datacomp_l14','dfn_h14','siglip2_l16_384'])
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--download',action='store_true');p.add_argument('--check',action='store_true')
    p.add_argument('--output-root',type=Path);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    if args.check and args.download:p.error('--check performs no downloads; omit --download')
    run(args)


if __name__=='__main__':main()
