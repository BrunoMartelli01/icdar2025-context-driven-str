"""Approximate GT-crop PARSeq control; never a deployable inference method."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from scripts.run_yolo_f1_quality_study import read,seal,sha
from scripts.token_context_data import rgb
from scripts.crop_clip_runtime import array_sha
from scripts.contextual_lm_verifier_runtime import state_digest
from recognition.deepsolo_word_f1 import normalize_word
from recognition.parseq import ParseqApproxRecognizer,ParseqApproxConfig,PARSEQ_ARCHIVE_SOURCE


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--paths',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=read(args.config);paths=read(args.paths);source=Path(paths['source_root']);original=read(source/'resolved_configuration.json')
    if cfg['reproduction_mode']!='approximate' or args.config.resolve().parent!=Path(__file__).resolve().parents[1]/'configs/approximate':
        raise ValueError('Explicit approximate control profile required')
    if sha(source/'results.json')!=cfg['source_results_sha256']:raise ValueError('Qualified source changed')
    controls=read(args.controls);partitions=controls['partitions'];train=read(source/'train/branches.json')['records']
    counts={str(i):sum(np.searchsorted(cfg['word_count_edges'],len(train[s]['ground_truth']),side='left')==i for s in partitions['validation']) for i in range(len(cfg['word_count_edges'])+1)}
    matched=[]
    for bin_index,n in counts.items():
        candidates=[s for s in partitions['fit'] if int(np.searchsorted(cfg['word_count_edges'],len(train[s]['ground_truth']),side='left'))==int(bin_index)]
        ordered=sorted(candidates,key=lambda s:(hashlib.sha256(f"{cfg['seed']}:{s}".encode()).digest(),s))
        if n>len(ordered):raise ValueError('Insufficient matched fitting controls')
        matched+=ordered[:n]
    groups=dict(fit_matched=sorted(matched),validation=partitions['validation'],test=partitions['test'])
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    asset_paths=original['paths']
    for name in ('parseq_checkpoint','parseq_archive'):
        if sha(asset_paths[name])!=original['config']['assets'][name]:raise ValueError('Qualified PARSeq asset changed')
    model=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(asset_paths['parseq_source']),
        checkpoint=Path(asset_paths['parseq_checkpoint']),runtime_source_archive=Path(asset_paths['parseq_archive']),
        runtime_source_type=PARSEQ_ARCHIVE_SOURCE,device='cpu'))
    model.model.requires_grad_(False).eval();before=state_digest(model.model);result={}
    try:
        for group,ids in groups.items():
            split='test' if group=='test' else 'train';records=read(source/split/'branches.json')['records']
            samples={s['sample_id']:s for s in original['samples'][split]};rows=[]
            for index,sid in enumerate(ids,1):
                image=rgb(samples[sid]);gt=records[sid]['ground_truth']
                for offset in range(0,len(gt),cfg['batch_size']):
                    target=gt[offset:offset+cfg['batch_size']];crops=[];boxes=[]
                    for word in target:
                        a,b,c,d=word['bbox'];box=[max(0,math.floor(a)),max(0,math.floor(b)),min(image.shape[1],math.ceil(c)),min(image.shape[0],math.ceil(d))]
                        a,b,c,d=box
                        if c<=a or d<=b:raise ValueError('Empty care GT crop; never drop its denominator')
                        boxes.append(box);crops.append(np.ascontiguousarray(image[b:d,a:c]))
                    decoded=model.recognize_candidate_batch(crops)
                    if len(decoded)!=len(target):raise ValueError('Incomplete real PARSeq GT crop response')
                    for i,(word,value,box,crop) in enumerate(zip(target,decoded,boxes,crops)):
                        prediction=value.to_dict();a,b=normalize_word(prediction['raw_text']),normalize_word(word['text'])
                        rows.append(dict(sample_id=sid,gt_index=offset+i,gt=word['text'],prediction=prediction,bbox=box,
                            crop_rgb_sha256=array_sha(crop),correct=bool(a and b and a==b)))
                if index%25==0 or index==len(ids):print(f'[GT crop control {group}] {index}/{len(ids)}',flush=True)
            correct=sum(r['correct'] for r in rows)
            result[group]=dict(images=len(ids),gt_words=len(rows),correct_words=correct,
                paired_recognition_accuracy=correct/len(rows),deepsolo_images=0,total_images=len(ids),deepsolo_percentage=0.,outputs=rows)
        after=state_digest(model.model)
        if before!=after:raise ValueError('Frozen PARSeq changed')
        seal(args.output,dict(status='SUCCESS',claim_label='approximate reproduction',reproduction_mode='approximate',
            analysis_class='GT_CROP_DIAGNOSTIC_NOT_DEPLOYABLE',results=result,
            configuration_sha256=sha(args.config),code_sha256=sha(__file__),controls_sha256=sha(args.controls),
            model_state_before=before,model_state_after=after,device='cpu',precision='float32',batch_size=cfg['batch_size'],
            model_input_shape=[cfg['batch_size'],3,32,128],fit_sampling='Match validation image counts in each GT word-count bin; deterministic hash order',
            metric='Paired GT word recognition after Unicode casefold+isalnum; empty normalized strings never match; every care crop retained',
            routing_scope='No routing decisions;0 new DeepSolo calls'))
        print(args.output)
    finally:model.release()


if __name__=='__main__':main()
