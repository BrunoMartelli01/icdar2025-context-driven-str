"""Read-only exposure, difficulty, OCR-filter and calibration controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from scripts.run_yolo_f1_quality_study import read,sha,seal
from scripts.unet_token_router_stages import light_predictions
from scripts.replay_router_improved_deepsolo import metric_counts,independent_counts
from scripts.token_context_data import key
from recognition.router_validation import expert_holdout_ids,independent_partitions
from recognition.f1_quality_study import aggregate


def endpoint(counts,heavy=False):
    value=aggregate(counts);n=value['images']
    return dict(metrics=value,deepsolo_images=n if heavy else 0,total_images=n,
                deepsolo_percentage=100. if heavy else 0.)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',choices=['approximate'],required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cfg=read(args.config);paths=read(args.paths);root=Path(paths['source_root']);plan=read(root/'resolved_configuration.json')
    if cfg['reproduction_mode']!='approximate' or args.config.resolve().parent!=Path(__file__).resolve().parents[1]/'configs/approximate':
        raise ValueError('Explicit approximate analysis profile required')
    for name,digest in cfg['source_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Qualified source changed')
    contract=read(paths['unet_contract']);manifest=Path(paths['expert_manifest'])
    if sha(manifest)!=contract['resolved_config']['word_manifest_sha256']:raise ValueError('Expert manifest changed')
    ordered=re.findall(r'^- id: (.+)$',manifest.read_text(),re.M)
    validation=expert_holdout_ids(ordered,contract['resolved_config']['seed'],contract['resolved_config']['validation_count'],contract['validation_ids_sha256'])
    partition=independent_partitions(plan['samples']['train'],validation,plan['samples']['test'])
    cohorts={};strata={};filter_study={};inputs={str(manifest):sha(manifest)}
    diagnostics=cfg['diagnostics']
    for split,source_split in [('fit','train'),('validation','train'),('test','test')]:
        raw=read(root/source_split/'branches.json');samples={s['sample_id']:s for s in plan['samples'][source_split]}
        ids=partition[split];branches=raw['branches'];records=raw['records']
        cohorts[split]={model:endpoint(branches[model][s]['light'] for s in ids) for model in ('unet','yolo')}
        cohorts[split]['deepsolo']=endpoint((branches['unet'][s]['deepsolo'] for s in ids),True)
        bins={kind:{} for kind in ['gt_word_count','gt_median_relative_height']}
        filters={str(t):[] for t in diagnostics['ocr_sequence_thresholds']}
        original_width=[];word_counts=[];heights=[];proposal_coverage=[];same_primary=0
        for sid in ids:
            sample=samples[sid];gt=records[sid]['ground_truth'];side=max(sample['image_wh']);height=float(np.median([(g['bbox'][3]-g['bbox'][1])/side for g in gt])) if gt else 0.
            original_width.append(side);word_counts.append(len(gt));heights.append(height)
            for kind,value,edges in [('gt_word_count',len(gt),diagnostics['gt_word_count_edges']),
                                     ('gt_median_relative_height',height,diagnostics['gt_relative_height_edges'])]:
                index=int(np.searchsorted(edges,value,side='left'));bins[kind].setdefault(str(index),[]).append(sid)
            path=root/source_split/'unet'/(key(sid)+'.json');inputs[str(path)]=sha(path)
            candidate=read(path)['candidates'];predictions=light_predictions([r for r in candidate],sample,plan['config']['metric']['native_ignore_size'])
            for t in diagnostics['ocr_sequence_thresholds']:
                kept=[p for p,row in zip(predictions,candidate) if row['sequence_confidence_native']>=t]
                counts=metric_counts(kept,gt,'casefold_alnum')
                if counts!=independent_counts(kept,gt,'casefold_alnum'):raise ValueError('Independent OCR-filter count control failed')
                filters[str(t)].append(counts)
                if t==plan['config']['unet']['minimum_transcript_probability']:
                    if counts!=branches['unet'][sid]['light']:raise ValueError('Primary OCR filter did not replay')
                    same_primary+=1
        filter_study[split]={t:endpoint(v) for t,v in filters.items()}
        strata[split]={kind:{b:dict(images=len(members),
                    experts={model:endpoint(branches[model][s]['light'] for s in members) for model in ('unet','yolo')},
                    deepsolo=endpoint((branches['unet'][s]['deepsolo'] for s in members),True))
                    for b,members in groups.items()} for kind,groups in bins.items()}
        cohorts[split]['metadata']=dict(images=len(ids),gt_words=sum(word_counts),
                word_count_quantiles=np.quantile(word_counts,[.1,.5,.9]).tolist(),
                median_relative_gt_height_quantiles=np.quantile(heights,[.1,.5,.9]).tolist(),
                original_maximum_side_quantiles=np.quantile(original_width,[.1,.5,.9]).tolist(),
                primary_filter_replay_images=same_primary)
    frozen=read(root/'frozen_choices.json');data=read(root/'train/branches.json')['branches'];bias={}
    for model,c in frozen['choices'].items():
        epoch=str(c['epoch']);oof={}
        for fold in range(plan['config']['cv_folds']):
            p=root/'training'/(model+'__fold'+str(fold)+'.json');inputs[str(p)]=sha(p)
            for sid,values in read(p)['checkpoints'][epoch]['count_predictions'].items():
                if sid in oof:raise ValueError('Duplicate held-out prediction')
                oof[sid]=values
        if set(oof)!=set(c['final_train']['count_predictions']):raise ValueError('Fixed-image refit/OOF population mismatch')
        reference=aggregate(data[model][s]['deepsolo'] for s in oof)['f1'];bias[model]={}
        for cohort,ids in [('all_original_train',sorted(oof)),('expert_fit',partition['fit']),('expert_holdout',partition['validation'])]:
            actual=np.array([[data[model][s]['light'][k]-data[model][s]['deepsolo'][k] for k in ('tp','detected_words')] for s in ids],float)
            bias[model][cohort]={}
            for predictor,values in [('held_out_router',oof),('full_refitted_router',c['final_train']['count_predictions'])]:
                predicted=np.array([values[s] for s in ids]);error=predicted-actual
                bias[model][cohort][predictor]=dict(images=len(ids),mean_delta_tp_bias=float(error[:,0].mean()),
                    mean_delta_prediction_count_bias=float(error[:,1].mean()),
                    mean_utility_bias=float((2*error[:,0]-reference*error[:,1]).mean()),
                    mean_absolute_count_error=float(np.abs(error).mean()),reference_f1=reference,
                    same_images_and_ground_truth=True,selection_use=False)
    output=dict(reproduction_mode='approximate',claim_label='approximate reproduction',analysis_class='POSTHOC_CAUSE_CONTROLS',
        source_results_sha256=sha(root/'results.json'),configuration_sha256=sha(args.config),code_sha256=sha(__file__),
        inputs_sha256=inputs,partitions=partition,expert_exposure_endpoints=cohorts,difficulty_strata=strata,
        ocr_filter_controls=filter_study,fixed_image_oof_refit_bias=bias,
        routing_scope='Cached endpoint analysis; actual new DeepSolo inference=0',
        caveat='Exposure and controlled prediction comparisons are evidence; one frozen expert and reused validation do not uniquely identify every causal effect')
    seal(args.output,output);print(args.output)


if __name__=='__main__':main()
