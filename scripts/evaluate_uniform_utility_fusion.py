"""Freeze all common-utility TEST routes before metric/fallback access."""

import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read,write
from scripts.icdar_budget_data import key,predictions
from scripts.evaluate_uniform_unet_fusion import fallback_rows
from segmentation.uniform_unet_fusion import annotation,file_hash,word_counts,aggregate_counts
from recognition.uniform_utility_deployment import UniformUtilityPipeline
from recognition.uniform_encoder_context import raw_context


def evaluate(config_path,evaluation_config,dataset):
    started=time.monotonic();c=read(config_path);e=read(evaluation_config)
    source=Path(c['training_root'])/dataset;root=Path(c['output_root'])/dataset
    plan=read(source/'resolved_configuration.json');frozen=read(root/'selection_frozen.json')
    out=root/'evaluation';out.mkdir(exist_ok=False)
    index=read(source/'export_test/index.json');ids=plan['partitions']['test']
    if set(index)!=set(ids) or len(ids)!=e['expected_images'][dataset]:raise ValueError('Complete TEST frontend required')
    scenes,maps={},{}
    for sid in ids:
        entry=index[sid]
        if file_hash(entry['path'])!=entry['sha256']:raise ValueError('Changed TEST frontend')
        scenes[sid]=read(entry['path'])
        if scenes[sid]['ground_truth_used']:raise ValueError('Frontend consumed labels')
        mp=source/'export_test/maps'/(key(sid)+'.npz')
        if file_hash(mp)!=scenes[sid]['maps_sha256']:raise ValueError('Changed predicted TEST map')
        with np.load(mp,allow_pickle=False) as z:maps[sid]=z['probabilities']
    context=raw_context(c,{s:dict(rgb_sha256=scenes[s]['rgb_sha256']) for s in ids})
    pipelines,observed,decisions={},{},{}
    for q in frozen['policies']:
        pipeline=UniformUtilityPipeline(config_path,dataset,int(q));pipelines[q]=pipeline
        observed[q]={s:pipeline.observe(scenes[s]['raw_rows'],scenes[s]['image_wh'],maps[s],
            context[s] if context is not None else None) for s in ids}
        decisions[q]=pipeline.route_batch(observed[q],{s:scenes[s]['rgb_sha256'] for s in ids})
    write(out/'observations.json',observed)
    write(out/'decisions_before_fallback.json',dict(status='FROZEN',decisions=decisions,
        observations_sha256=file_hash(out/'observations.json'),frozen_sha256=file_hash(root/'selection_frozen.json'),
        export_index_sha256=file_hash(source/'export_test/index.json'),ground_truth_used=False,fallback_outputs_loaded=False))
    # Only annotation-free route plans above can decide which fallback is needed.
    for path,digest in e['pinned_inputs'].items():
        if file_hash(path)!=digest:raise ValueError('Changed benchmark fallback source: '+path)
    lookup={r['sample_id']:r for r in plan['test_records']};truth,flags,heavy,hcounts={},{},{},{}
    spec=e['fallback_test_sources'][dataset];archive=read(spec['path']) if spec['kind']!='icdar13_files' else None
    for sid in ids:
        truth[sid],valid=annotation(lookup[sid]);flags[sid]=[p['ignored'] for p in predictions(scenes[sid]['raw_rows'],lookup[sid],valid)]
        heavy[sid],expected,_=fallback_rows(e,dataset,sid,archive)
        hcounts[sid]=word_counts(heavy[sid],truth[sid],[r['ignored'] for r in heavy[sid]])
        if expected is not None and expected!=hcounts[sid]:raise ValueError('Fallback metric/care mismatch')
    baseline=aggregate_counts(hcounts.values())
    if baseline['gt_words']!=e['expected_care_words'][dataset]:raise ValueError('Incomplete TEST care denominator')
    outputs,results={},{}
    for q,pipeline in pipelines.items():
        accepted=set(decisions[q]['accepted_ids']);outputs[q]={}
        for sid in ids:
            o=observed[q][sid]
            if sid in accepted:rows=o['rows'];ignored=[flags[sid][int(r['source_row'])] for r in rows]
            else:
                rows=pipeline.combine_after_fallback(o,heavy[sid]);ignored=[flags[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
            outputs[q][sid]=dict(rows=rows,ignored=ignored,ground_truth=truth[sid],counts=word_counts(rows,truth[sid],ignored),action='light' if sid in accepted else 'deepsolo')
        counts=aggregate_counts(v['counts'] for v in outputs[q].values())
        if counts['images']!=len(ids) or counts['gt_words']!=e['expected_care_words'][dataset]:raise ValueError('Omitted TEST image/care word')
        results[q]=dict(counts=counts,deepsolo_images=int(q),total_images=len(ids),deepsolo_percentage=100*int(q)/len(ids),
            target=plan['target'],target_met=counts['f1']>plan['target'],actual_new_deepsolo_neural_calls=0)
    write(out/'per_image.json',outputs)
    write(out/'results.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',dataset=dataset,
        primary=results[str(plan['budget'])],results=results,primary_quota=plan['budget'],
        validation_recommended_quota=frozen['validation_recommended_fallback_images'],
        all_deepsolo=dict(counts=baseline,deepsolo_images=len(ids),total_images=len(ids),deepsolo_percentage=100.),
        frontend_inference_reused=True,actual_new_neural_calls=0,additional_neural_models=0,
        historical_test_exposure=True,test_used_for_selection=False,seconds_cpu_evaluation=time.monotonic()-started,
        config_sha256=file_hash(config_path),evaluation_config_sha256=file_hash(evaluation_config),source_sha256=file_hash(__file__),
        scope='Authentic cached frontend/fallback replay. Quotas were predeclared; other TEST points cannot select deployment. No online latency claim.'))
    print('UTILITY FULL TEST',dataset,results[str(plan['budget'])],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--evaluation-config',type=Path,required=True);p.add_argument('--dataset',choices=['icdar13','icdar2015','totaltext'],required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):evaluate(a.config,a.evaluation_config,a.dataset)
