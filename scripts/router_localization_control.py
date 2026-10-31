"""Nested one-to-one crop/recognition/filter coverage diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_yolo_f1_quality_study import read,seal,sha
from scripts.unet_token_router_stages import light_predictions
from scripts.token_context_data import key
from recognition.scene_text_v1_evaluation import maximum_matching,box_iou
from recognition.deepsolo_word_f1 import normalize_word


def nested_coverage(rows,gt,iou_threshold,sequence_threshold):
    geometry=[];text=[];selected=[]
    for i,row in enumerate(rows):
        for j,g in enumerate(gt):
            if box_iou(row['bbox'],g['bbox'])<iou_threshold:continue
            geometry.append((i,j));a,b=normalize_word(row['text']),normalize_word(g['text'])
            if a and b and a==b:
                text.append((i,j))
                if row['sequence_confidence_native']>=sequence_threshold:selected.append((i,j))
    match=lambda edges:len(maximum_matching(len(rows),len(gt),edges))
    g,t,s=match(geometry),match(text),match(selected)
    if not 0<=s<=t<=g<=len(gt):raise ValueError('Invalid nested matching cardinalities')
    return dict(gt_words=len(gt),geometry_covered=g,text_correct_before_filter=t,text_correct_after_filter=s,
        no_geometric_coverage=len(gt)-g,covered_but_no_correct_reading=g-t,
        correct_reading_removed_by_filter=t-s,retained_spatial_correct=s)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    for name in ('config','paths','controls','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cfg=read(a.config);paths=read(a.paths);source=Path(paths['source_root']);plan=read(source/'resolved_configuration.json');controls=read(a.controls)
    if cfg['reproduction_mode']!='approximate' or sha(source/'results.json')!=cfg['source_results_sha256']:raise ValueError('Qualified approximate source required')
    output={}
    for cohort,ids in controls['partitions'].items():
        split='test' if cohort=='test' else 'train';raw=read(source/split/'branches.json')['records'];samples={s['sample_id']:s for s in plan['samples'][split]}
        records={};totals={}
        for sid in ids:
            rows=read(source/split/'unet'/(key(sid)+'.json'))['candidates']
            pred=light_predictions(rows,samples[sid],plan['config']['metric']['native_ignore_size'])
            care=[dict(p,sequence_confidence_native=r['sequence_confidence_native']) for p,r in zip(pred,rows) if not p['ignored']]
            value=nested_coverage(care,raw[sid]['ground_truth'],cfg['iou_threshold'],plan['config']['unet']['minimum_transcript_probability'])
            records[sid]=value
            for k,v in value.items():totals[k]=totals.get(k,0)+v
        for k in ('geometry_covered','text_correct_before_filter','text_correct_after_filter'):
            totals[k+'_fraction']=totals[k]/totals['gt_words']
        output[cohort]=dict(images=len(ids),counts=totals,per_image=records,deepsolo_images=0,total_images=len(ids),deepsolo_percentage=0.)
    seal(a.output,dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',
        analysis_class='DIAGNOSTIC_SPATIAL_COVERAGE_NOT_PRIMARY_F1',cohorts=output,iou_threshold=cfg['iou_threshold'],
        sequence_threshold=plan['config']['unet']['minimum_transcript_probability'],config_sha256=sha(a.config),
        code_sha256=sha(__file__),source_results_sha256=sha(source/'results.json'),controls_sha256=sha(a.controls),
        matching='Deterministic maximum one-to-one bbox matching. Nested edge sets: geometric; geometric+correct text; geometric+correct text+sequence filter.',
        caveat='Failed IoU coverage includes merges, misses and shifted envelopes; does not imply that no text pixel is inside a crop. This spatial diagnostic differs from primary lexical F1.',
        actual_new_deepsolo_inference=0))
    print(a.output)


if __name__=='__main__':main()
