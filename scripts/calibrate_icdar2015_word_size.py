"""Cached GT-blind component-size calibration on fixed ICDAR2015 VAL100 only."""

import argparse
import json
from pathlib import Path
import time

from recognition.icdar2015_word_f1 import aggregate_counts, sample_counts
from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO, load_training_population, references
from segmentation.icdar2015_multiview import emitted
from segmentation.icdar2015_target70 import load_profile
from segmentation.icdar2015_word_size import size_admitted
from scripts.train_icdar2015_totaltext_transfer import write_json


def load_json(path):
    return json.loads(path.read_text())


def verify_inputs(a):
    cfg=load_json(a.config)
    if a.config.resolve().parent.name!='approximate' or cfg['reproduction_mode']!='approximate' or cfg['claim_label']!='approximate reproduction':
        raise ValueError('Explicit approximate size-calibration profile required')
    for field,path in [('source_multiview_results_sha256',a.source_results),('source_multiview_profile_sha256',a.source_config)]:
        if file_sha256(path)!=cfg[field]:raise ValueError('Pinned source differs: '+field)
    source=load_json(a.source_results);sc=load_json(a.source_config)
    if (source['status']!='SUCCESS' or source['test_data_used'] or source['run_contract']['config_sha256']!=file_sha256(a.source_config)
            or sc['source_checkpoint_sha256']!=cfg['source_checkpoint_sha256'] or len(source['candidates'])!=cfg['expected_source_policies']):
        raise ValueError('Expected frozen VAL-only c467 multiview source required')
    if (len(cfg['projected_component_area_minimums'])!=4 or len(cfg['original_box_height_minimums'])!=3
            or cfg['projected_component_area_minimums']!=[0,4,16,64] or cfg['original_box_height_minimums']!=[0,4,8]):
        raise ValueError('Unexpected predeclared size grid')
    selection=load_json(Path(source['run_contract']['source_selection']))
    training_config=load_profile(REPO/selection['config']['training_profile'])
    manifest,_,val=load_training_population(training_config,a.data_root)
    refs=references(manifest,a.data_root,val)
    if len(refs)!=100 or sum(len(r['ground_truth']) for r in refs.values())!=430:
        raise ValueError('Fixed VAL100/430care required')
    shapes={v['name']+('_flip' if v['horizontal_flip'] else ''):v['hw'] for v in sc['views']}
    streams={}
    for name,entry in source['streams'].items():
        path=a.source_results.parent/entry['per_image_file']
        if file_sha256(path)!=entry['per_image_sha256']:raise ValueError('Source stream checksum changed: '+name)
        rows=load_json(path)
        if len(rows)!=100 or len({r['sample_id'] for r in rows})!=100:raise ValueError('Stream population changed')
        for row in rows:
            if row['sample_id'] not in refs or row['ground_truth']!=refs[row['sample_id']]['ground_truth']:
                raise ValueError('Canonical annotations and saved stream differ')
            for p in row['predictions']:
                if p['stream_id']!=name or p['view'] not in shapes:raise ValueError('Invalid source prediction identity')
        streams[name]={r['sample_id']:r['predictions'] for r in rows}
    if len(streams)!=16 or len(shapes)!=8:raise ValueError('Expected all16 streams and8 views')
    return cfg,sc,source,refs,shapes,streams


def run(a):
    cfg,sc,source,refs,shapes,streams=verify_inputs(a)
    contract=dict(profile=str(a.config.resolve()),profile_sha256=file_sha256(a.config),source_results=str(a.source_results.resolve()),
        source_results_sha256=file_sha256(a.source_results),source_profile=str(a.source_config.resolve()),
        source_profile_sha256=file_sha256(a.source_config),source_checkpoint_sha256=cfg['source_checkpoint_sha256'],
        data_root=str(a.data_root.resolve()),source_sha256={name:file_sha256(REPO/name) for name in
        ('scripts/calibrate_icdar2015_word_size.py','segmentation/icdar2015_word_size.py','segmentation/icdar2015_multiview.py',
         'recognition/icdar2015_word_f1.py')},model_count=1,neural_inference_performed=False,ocr_inference_performed=False)
    write_json(a.output_dir/'run_contract.json',contract)
    candidate_summaries=[];best=None;original_by_index={c['index']:c for c in source['candidates']};start=time.monotonic()
    if set(original_by_index)!=set(range(216)):raise ValueError('Original candidate indices changed')
    records=a.output_dir/'candidate_per_image.jsonl';records.parent.mkdir(parents=True,exist_ok=True)
    with records.open('w') as sink:
        for area in cfg['projected_component_area_minimums']:
            for height in cfg['original_box_height_minimums']:
                admitted={stream:{key:[p for p in preds if size_admitted(p,refs[key]['image_wh'],shapes[p['view']],
                    minimum_area=area,minimum_height=height)] for key,preds in rows.items()} for stream,rows in streams.items()}
                for old_index in range(216):
                    original=original_by_index[old_index]
                    rows=[];counts=[]
                    for key,ref in refs.items():
                        pool=[p for stream in original['streams'] for p in admitted[stream][key]]
                        keep=emitted(pool,**original['policy'])
                        count=sample_counts(keep,ref['ground_truth'],iou_threshold=sc['matching_iou_threshold'])
                        counts.append(count)
                        rows.append(dict(sample_id=key,counts=count,prediction_ids=[[p['stream_id'],p['component_index']] for p in keep]))
                    aggregate=aggregate_counts(counts)
                    if area==height==0 and aggregate!=original['aggregate']:
                        raise RuntimeError('Zero-size original policy replay changed: '+str(old_index))
                    index=len(candidate_summaries)
                    entry=dict(index=index,source_index=old_index,pool=original['pool'],streams=original['streams'],
                        policy=original['policy'],minimum_projected_area=area,minimum_original_height=height,aggregate=aggregate)
                    sink.write(json.dumps(dict(candidate=entry,per_image=rows),sort_keys=True,separators=(',',':'))+'\n')
                    candidate_summaries.append(entry)
                    key=(aggregate['alphanumeric_casefold']['f1'],aggregate['alphanumeric_casefold']['recall'],
                         -len(entry['streams']),-area,-height,-old_index)
                    if best is None or key>best[0]:best=(key,entry,rows)
                write_json(a.output_dir/'live_status.json',dict(status='CALIBRATING',completed_size_pairs=3*cfg['projected_component_area_minimums'].index(area)+cfg['original_box_height_minimums'].index(height)+1,total_size_pairs=12,candidates=len(candidate_summaries)))
                print(f'area={area} height={height}: {len(candidate_summaries)} candidates, best F1={best[1]["aggregate"]["alphanumeric_casefold"]["f1"]:.6f}',flush=True)
    if len(candidate_summaries)!=2592:raise RuntimeError('Incomplete candidate grid')
    chosen=best[1]
    full=[]
    for row in best[2]:
        key=row['sample_id'];ids={tuple(x) for x in row['prediction_ids']}
        # Reproduce the same ranked order as the policy, then cross-check IDs/counts.
        keep=emitted([p for stream in chosen['streams'] for p in streams[stream][key] if size_admitted(p,refs[key]['image_wh'],
            shapes[p['view']],minimum_area=chosen['minimum_projected_area'],minimum_height=chosen['minimum_original_height'])],**chosen['policy'])
        if {tuple((p['stream_id'],p['component_index'])) for p in keep}!=ids or sample_counts(keep,refs[key]['ground_truth'],iou_threshold=sc['matching_iou_threshold'])!=row['counts']:
            raise RuntimeError('Selected full-output replay differs')
        full.append(dict(sample_id=key,ground_truth=refs[key]['ground_truth'],predictions=keep,counts=row['counts']))
    write_json(a.output_dir/'selected_per_image.json',full)
    result=dict(status='SUCCESS',claim_label='approximate reproduction',result_class='fixed VAL100 cached calibration; not TEST',
        config=cfg,run_contract=contract,candidates=candidate_summaries,selected=chosen,selected_per_image_file='selected_per_image.json',
        selected_per_image_sha256=file_sha256(a.output_dir/'selected_per_image.json'),candidate_per_image_file=records.name,
        candidate_per_image_sha256=file_sha256(records),source_original_selected=source['selected'],
        zero_size_replayed_candidates=216,failed_samples=0,test_data_used=False,neural_inference_performed=False,
        ocr_inference_performed=False,deepsolo_images=0,total_images=100,deepsolo_percentage=0.,elapsed_seconds=time.monotonic()-start)
    write_json(a.output_dir/'results.json',result)
    write_json(a.output_dir/'live_status.json',dict(status='SUCCESS',candidates=len(candidate_summaries)))
    metric=chosen['aggregate']['alphanumeric_casefold']
    print(f'SELECTED F1={metric["f1"]:.6f}, TP={metric["tp"]}, FP={metric["fp"]}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','source-results','source-config','data-root','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise ValueError('Preserve immutable output directory')
    try:run(a)
    except Exception as e:
        write_json(a.output_dir/'failure.json',dict(status='FAILED',error_type=type(e).__name__,error=str(e)))
        raise


if __name__=='__main__':main()
